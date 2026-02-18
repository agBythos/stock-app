"""
validation/ensemble_trainer.py — Ensemble Walk-Forward Trainer
==============================================================

在每個 fold 上訓練多個模型並彙總結果。

支援的模型：
  - RandomForestClassifier  (sklearn)
  - CatBoostClassifier      (catboost，若已安裝)
  - XGBClassifier           (xgboost，若已安裝)
  - LGBMClassifier          (lightgbm，若已安裝)

特徵說明：
  外部傳入 DataFrame 必須包含：
  - features 欄位（由 feature_cols 指定）
  - target 欄位（0/1 分類，已完成 shift 防止前視偏差）

Look-ahead Bias 防護：
  本模組假設目標變數已在外部 shift：
    df["target"] = (df["close"].pct_change(label_horizon)
                    .shift(-label_horizon) > 0).astype(int)
  訓練時只使用 WalkForwardFold.purged_train_idx，測試時使用 test_idx。

作者：Bythos（sub-agent）
建立：2026-02-18
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.walk_forward import WalkForwardConfig
from validation.purged_walk_forward import WalkForwardFold

warnings.filterwarnings("ignore", category=UserWarning)


# ── 可選依賴 ──────────────────────────────────────────────────────────────────

try:
    from catboost import CatBoostClassifier
    _CATBOOST = True
except ImportError:
    _CATBOOST = False

try:
    from xgboost import XGBClassifier
    _XGBOOST = True
except ImportError:
    _XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM = True
except ImportError:
    _LIGHTGBM = False


# ============================================================================
# FoldResult — 單一 fold / 模型結果
# ============================================================================


@dataclass
class FoldResult:
    """單一 fold + 單一模型的回測結果"""

    fold_id: int
    model_name: str

    # 分類指標
    accuracy: float = 0.0
    roc_auc: float = 0.0

    # 預測值
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba: np.ndarray = field(default_factory=lambda: np.array([]))  # P(up)

    # 樣本數
    n_train: int = 0
    n_test: int = 0

    # 錯誤
    error: Optional[str] = None
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "roc_auc": round(self.roc_auc, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "skipped": self.skipped,
            "error": self.error,
        }


@dataclass
class EnsembleFoldResult:
    """單一 fold 的 ensemble 結果"""

    fold_id: int
    model_results: List[FoldResult] = field(default_factory=list)

    # Ensemble 均值
    ensemble_accuracy: float = 0.0
    ensemble_roc_auc: float = 0.0
    ensemble_proba: np.ndarray = field(default_factory=lambda: np.array([]))

    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "ensemble_accuracy": round(self.ensemble_accuracy, 4),
            "ensemble_roc_auc": round(self.ensemble_roc_auc, 4),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "models": [m.to_dict() for m in self.model_results],
        }


# ============================================================================
# EnsembleWalkForwardTrainer
# ============================================================================


class EnsembleWalkForwardTrainer:
    """
    多模型 Ensemble Walk-Forward 訓練器

    在每個（未跳過的）WalkForwardFold 上：
    1. 用 purged_train_idx 訓練所有模型
    2. 在 test_idx 上評估
    3. 回傳各模型及 ensemble 的結果

    使用範例::

        from config.walk_forward import WalkForwardConfig
        from validation.purged_walk_forward import PurgedWalkForwardSplitter
        from validation.ensemble_trainer import EnsembleWalkForwardTrainer

        cfg = WalkForwardConfig()
        splitter = PurgedWalkForwardSplitter(cfg)
        trainer = EnsembleWalkForwardTrainer(cfg)

        folds = splitter.split(df)
        results = trainer.train_all_folds(df, folds, feature_cols, target_col="target")
    """

    # 模型名稱 → 工廠函式
    def __init__(self, config: WalkForwardConfig):
        self.cfg = config

    # ------------------------------------------------------------------
    # 模型工廠
    # ------------------------------------------------------------------

    def _make_models(self) -> Dict[str, Any]:
        """建立模型字典（只包含已安裝的模型）"""
        rs = self.cfg.random_state
        models: Dict[str, Any] = {}

        # RandomForest（永遠可用）
        models["random_forest"] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            random_state=rs,
            n_jobs=-1,
        )

        # CatBoost
        if _CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_seed=rs,
                verbose=False,
                allow_writing_files=False,
            )

        # XGBoost
        if _XGBOOST:
            models["xgboost"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=rs,
                eval_metric="logloss",
                verbosity=0,
            )

        # LightGBM
        if _LIGHTGBM:
            models["lightgbm"] = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=rs,
                verbosity=-1,
            )

        return models

    # ------------------------------------------------------------------
    # 單一 fold 訓練
    # ------------------------------------------------------------------

    def train_fold(
        self,
        df: pd.DataFrame,
        fold: WalkForwardFold,
        feature_cols: List[str],
        target_col: str = "target",
    ) -> EnsembleFoldResult:
        """
        在單一 fold 上訓練並評估

        Args:
            df          : 完整 DataFrame（含 features + target）
            fold        : WalkForwardFold 物件
            feature_cols: 特徵欄位名稱列表
            target_col  : 目標欄位名稱

        Returns:
            EnsembleFoldResult
        """
        if fold.skipped:
            return EnsembleFoldResult(
                fold_id=fold.fold_id,
                skipped=True,
                skip_reason=fold.skip_reason,
            )

        # ── 取出訓練/測試資料 ────────────────────────────────────────
        train_df = df.iloc[fold.purged_train_idx]
        test_df = df.iloc[fold.test_idx]

        # 移除含 NaN 的樣本
        train_df = train_df[feature_cols + [target_col]].dropna()
        test_df = test_df[feature_cols + [target_col]].dropna()

        if len(train_df) < self.cfg.min_train_samples:
            return EnsembleFoldResult(
                fold_id=fold.fold_id,
                skipped=True,
                skip_reason=f"train size after dropna={len(train_df)} < {self.cfg.min_train_samples}",
            )

        if len(test_df) == 0:
            return EnsembleFoldResult(
                fold_id=fold.fold_id,
                skipped=True,
                skip_reason="test set empty after dropna",
            )

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values.astype(int)
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values.astype(int)

        # ── 標準化（每個 fold 獨立 fit）────────────────────────────────
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ── 訓練各模型 ────────────────────────────────────────────────
        models = self._make_models()
        fold_results: List[FoldResult] = []
        all_probas: List[np.ndarray] = []

        for model_name, model in models.items():
            result = self._train_single_model(
                model_name=model_name,
                model=model,
                X_train=X_train_scaled,
                y_train=y_train,
                X_test=X_test_scaled,
                y_test=y_test,
                fold_id=fold.fold_id,
                n_train=len(train_df),
                n_test=len(test_df),
            )
            fold_results.append(result)
            if not result.skipped and len(result.y_proba) > 0:
                all_probas.append(result.y_proba)

        # ── Ensemble 均值 ─────────────────────────────────────────────
        if all_probas:
            ensemble_proba = np.mean(all_probas, axis=0)
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            ensemble_acc = float(accuracy_score(y_test, ensemble_pred))
            try:
                ensemble_auc = float(roc_auc_score(y_test, ensemble_proba))
            except ValueError:
                ensemble_auc = 0.5
        else:
            ensemble_proba = np.array([])
            ensemble_acc = 0.0
            ensemble_auc = 0.5

        return EnsembleFoldResult(
            fold_id=fold.fold_id,
            model_results=fold_results,
            ensemble_accuracy=ensemble_acc,
            ensemble_roc_auc=ensemble_auc,
            ensemble_proba=ensemble_proba,
        )

    def _train_single_model(
        self,
        model_name: str,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fold_id: int,
        n_train: int,
        n_test: int,
    ) -> FoldResult:
        """訓練單一模型並評估"""
        try:
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, y_pred))

            # 取 P(class=1)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = y_pred.astype(float)

            try:
                auc = float(roc_auc_score(y_test, proba))
            except ValueError:
                auc = 0.5

            return FoldResult(
                fold_id=fold_id,
                model_name=model_name,
                accuracy=accuracy,
                roc_auc=auc,
                y_pred=y_pred,
                y_proba=proba,
                n_train=n_train,
                n_test=n_test,
            )

        except Exception as e:
            warnings.warn(f"Fold {fold_id} model={model_name} error: {e}")
            return FoldResult(
                fold_id=fold_id,
                model_name=model_name,
                skipped=True,
                error=str(e),
                n_train=n_train,
                n_test=n_test,
            )

    # ------------------------------------------------------------------
    # 全部 fold 訓練
    # ------------------------------------------------------------------

    def train_all_folds(
        self,
        df: pd.DataFrame,
        folds: List[WalkForwardFold],
        feature_cols: List[str],
        target_col: str = "target",
        verbose: bool = True,
    ) -> List[EnsembleFoldResult]:
        """
        對所有 fold 執行訓練並回傳結果列表

        Args:
            df          : 完整 DataFrame
            folds       : PurgedWalkForwardSplitter.split() 的輸出
            feature_cols: 特徵欄位
            target_col  : 目標欄位
            verbose     : 印出每個 fold 的摘要

        Returns:
            List[EnsembleFoldResult]
        """
        results = []
        n_total = len(folds)
        n_active = sum(1 for f in folds if not f.skipped)

        if verbose:
            print(f"\n{'='*60}")
            print(f"EnsembleWalkForwardTrainer: {n_active}/{n_total} active folds")
            print(f"Models: {list(self._make_models().keys())}")
            print(f"{'='*60}")

        for fold in folds:
            result = self.train_fold(df, fold, feature_cols, target_col)
            results.append(result)

            if verbose and not result.skipped:
                print(
                    f"  Fold {result.fold_id:3d} | "
                    f"acc={result.ensemble_accuracy:.4f} "
                    f"auc={result.ensemble_roc_auc:.4f} "
                    f"({len(result.model_results)} models)"
                )

        return results

    @staticmethod
    def summarize(results: List[EnsembleFoldResult]) -> Dict[str, Any]:
        """
        彙總所有 fold 的統計指標

        Args:
            results: train_all_folds 的輸出

        Returns:
            dict 含平均值、標準差等
        """
        active = [r for r in results if not r.skipped]

        if not active:
            return {
                "n_total_folds": len(results),
                "n_active_folds": 0,
                "note": "No active folds",
            }

        accs = [r.ensemble_accuracy for r in active]
        aucs = [r.ensemble_roc_auc for r in active]

        return {
            "n_total_folds": len(results),
            "n_active_folds": len(active),
            "n_skipped_folds": len(results) - len(active),
            "ensemble_accuracy": {
                "mean": round(float(np.mean(accs)), 4),
                "std": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
                "min": round(float(min(accs)), 4),
                "max": round(float(max(accs)), 4),
            },
            "ensemble_roc_auc": {
                "mean": round(float(np.mean(aucs)), 4),
                "std": round(float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0, 4),
                "min": round(float(min(aucs)), 4),
                "max": round(float(max(aucs)), 4),
            },
        }

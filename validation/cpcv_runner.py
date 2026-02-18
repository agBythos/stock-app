"""
validation/cpcv_runner.py — CPCVRunner + CPCVReport
====================================================

CPCV 驗證框架的頂層 Orchestrator。

整合：
  1. CPCVSplitter      → C(N,k) 個 fold
  2. EnsembleWalkForwardTrainer（複用 Phase 3.5）→ ML 指標
  3. run_backtrader_fold()（複用 Phase 3.5）→ Backtrader 績效
  4. CPCVSplitter.build_paths()  → φ 條 backtest paths
  5. Sharpe 分佈統計（均值、標準差、CI-95、PBO）

CPCV 優於 Walk-Forward 之處：
  - 同樣資料產生 φ 條（而非 1 條）backtest paths
  - 可計算 Sharpe Ratio 分佈（均值、標準差、信賴區間）
  - 大幅降低 overfitting 誤判率

作者：Bythos（sub-agent phase4-cpcv-impl）
建立：2026-02-18
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.cpcv import CPCVConfig
from config.walk_forward import WalkForwardConfig
from validation.backtrader_bridge import BacktraderFoldResult, run_backtrader_fold
from validation.cpcv_splitter import CPCVFold, CPCVPath, CPCVSplitter
from validation.ensemble_trainer import EnsembleFoldResult, EnsembleWalkForwardTrainer

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# CPCVFoldReport — 單一 fold 報告
# ============================================================================


@dataclass
class CPCVFoldReport:
    """單一 CPCV fold 的完整報告（ML + Backtrader）"""

    fold_id: int
    combination: Tuple[int, ...]
    train_groups: List[int]
    test_groups: List[int]
    train_bars: int = 0
    test_bars: int = 0

    # ML 指標
    ensemble_accuracy: float = 0.0
    ensemble_roc_auc: float = 0.0
    model_scores: Dict[str, float] = field(default_factory=dict)

    # Backtrader 績效
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0

    skipped: bool = False
    skip_reason: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "combination": list(self.combination),
            "train_groups": self.train_groups,
            "test_groups": self.test_groups,
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "ml": {
                "ensemble_accuracy": round(self.ensemble_accuracy, 4),
                "ensemble_roc_auc": round(self.ensemble_roc_auc, 4),
                "model_scores": {k: round(v, 4) for k, v in self.model_scores.items()},
            },
            "backtest": {
                "total_return_pct": round(self.total_return_pct, 4),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "max_drawdown_pct": round(self.max_drawdown_pct, 4),
                "win_rate_pct": round(self.win_rate_pct, 4),
                "total_trades": self.total_trades,
            },
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
        }


# ============================================================================
# CPCVPathReport — 單條 backtest path 統計
# ============================================================================


@dataclass
class CPCVPathReport:
    """單條 backtest path 的績效統計"""

    path_id: int
    group_id: int
    fold_ids: List[int]
    n_bars: int = 0

    # Path 級別績效
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0      # |total_return| / max_drawdown（防止 0 分母）

    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path_id": self.path_id,
            "group_id": self.group_id,
            "fold_ids": self.fold_ids,
            "n_bars": self.n_bars,
            "total_return_pct": round(self.total_return_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "error": self.error,
        }


# ============================================================================
# CPCVReport — 完整 CPCV 報告
# ============================================================================


@dataclass
class CPCVReport:
    """完整 CPCV 執行報告"""

    symbol: str
    run_timestamp: str
    config: Dict[str, Any]

    # Fold 級別
    folds: List[CPCVFoldReport] = field(default_factory=list)
    n_total_folds: int = 0
    n_active_folds: int = 0
    n_skipped_folds: int = 0

    # Path 級別（CPCV 獨有）
    paths: List[CPCVPathReport] = field(default_factory=list)
    n_backtest_paths: int = 0

    # 彙總統計（從 paths 計算）
    summary_sharpe_mean: float = 0.0
    summary_sharpe_std: float = 0.0
    summary_sharpe_ci_95: Tuple[float, float] = (0.0, 0.0)
    summary_return_mean: float = 0.0
    summary_max_drawdown_mean: float = 0.0
    summary_pbo: float = 0.0       # P(Sharpe <= 0) = Probability of Backtest Overfitting

    def to_dict(self) -> dict:
        active_folds = [f for f in self.folds if not f.skipped]
        accs = [f.ensemble_accuracy for f in active_folds]
        aucs = [f.ensemble_roc_auc for f in active_folds]

        return {
            "symbol": self.symbol,
            "run_timestamp": self.run_timestamp,
            "config": self.config,
            "n_total_folds": self.n_total_folds,
            "n_active_folds": self.n_active_folds,
            "n_skipped_folds": self.n_skipped_folds,
            "n_backtest_paths": self.n_backtest_paths,
            "folds": [f.to_dict() for f in self.folds],
            "paths": [p.to_dict() for p in self.paths],
            "summary": {
                "n_total_folds": self.n_total_folds,
                "n_active_folds": self.n_active_folds,
                "n_skipped_folds": self.n_skipped_folds,
                "n_backtest_paths": self.n_backtest_paths,
                "sharpe_mean": round(self.summary_sharpe_mean, 4),
                "sharpe_std": round(self.summary_sharpe_std, 4),
                "sharpe_ci_95": [
                    round(self.summary_sharpe_ci_95[0], 4),
                    round(self.summary_sharpe_ci_95[1], 4),
                ],
                "sharpe_pbo": round(self.summary_pbo, 4),
                "return_mean_pct": round(self.summary_return_mean, 4),
                "max_drawdown_mean_pct": round(self.summary_max_drawdown_mean, 4),
                "ml": {
                    "ensemble_accuracy_mean": round(float(np.mean(accs)), 4) if accs else 0.0,
                    "ensemble_roc_auc_mean": round(float(np.mean(aucs)), 4) if aucs else 0.0,
                },
            },
        }


# ============================================================================
# CPCVRunner — 頂層 Orchestrator
# ============================================================================


class CPCVRunner:
    """
    Combinatorially Purged Cross-Validation Orchestrator

    完整執行流程：
      原始資料 → 特徵工程 → 建目標變數（shift）
      → CPCVSplitter.split() → 每 fold 訓練（ensemble）+ Backtrader 回測
      → build_paths() → 每 path 計算績效
      → Sharpe 分佈統計

    使用範例::

        cfg = CPCVConfig()
        runner = CPCVRunner(cfg)
        report = runner.run(df, symbol="2330.TW")
        print(report.to_dict())

    特徵工程（與 PurgedWalkForwardRunner 相同）：
      RSI(14), MACD histogram, MA crossover, Bollinger %B,
      Volume ratio, Momentum (5/10/20d), Volatility(20d)
    """

    FEATURE_COLS = [
        "rsi_14",
        "macd_hist",
        "ma_cross_10_30",
        "ma_cross_20_60",
        "bb_pct_b",
        "vol_ratio",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "volatility_20d",
    ]

    TARGET_COL = "target"

    def __init__(self, cfg: Optional[CPCVConfig] = None):
        self.cfg = cfg or CPCVConfig()
        self.cfg.validate()
        self._splitter = CPCVSplitter(self.cfg)

        # 複用 Phase 3.5 的 EnsembleWalkForwardTrainer（使用相同介面的 WalkForwardConfig）
        wf_cfg = WalkForwardConfig(
            label_horizon=self.cfg.label_horizon,
            embargo_bars=self.cfg.embargo_bars,
            min_train_samples=self.cfg.min_train_samples,
            commission_rate=self.cfg.commission_rate,
            commission_discount=self.cfg.commission_discount,
            sell_tax_rate=self.cfg.sell_tax_rate,
            initial_capital=self.cfg.initial_capital,
            random_state=self.cfg.random_state,
        )
        self._trainer = EnsembleWalkForwardTrainer(wf_cfg)
        self._wf_cfg = wf_cfg

    # ------------------------------------------------------------------
    # 特徵工程（與 Phase 3.5 完全相同）
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標特徵並建立目標變數（含正確 shift）

        防洩漏保護：
            target[t] = (close[t + label_horizon] > close[t]) ? 1 : 0
            shift(-label_horizon) → target[t] 使用未來資料
            但訓練時只使用 purged_train_idx（已排除洩漏樣本）
        """
        out = df.copy()
        out.columns = [c.lower() for c in out.columns]

        close = out["close"]
        volume = out["volume"]

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD histogram
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = macd - signal_line

        # MA crossover
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma30 = close.rolling(30).mean()
        ma60 = close.rolling(60).mean()
        out["ma_cross_10_30"] = (ma10 - ma30) / close
        out["ma_cross_20_60"] = (ma20 - ma60) / close

        # Bollinger %B
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        denom = (bb_upper - bb_lower).replace(0, np.nan)
        out["bb_pct_b"] = (close - bb_lower) / denom

        # Volume ratio
        vol_ma = volume.rolling(20).mean().replace(0, np.nan)
        out["vol_ratio"] = volume / vol_ma

        # Momentum
        out["momentum_5d"] = close.pct_change(5)
        out["momentum_10d"] = close.pct_change(10)
        out["momentum_20d"] = close.pct_change(20)

        # Volatility
        out["volatility_20d"] = close.pct_change().rolling(20).std()

        # Target（正確 shift，防止前視偏差）
        future_return = close.pct_change(self.cfg.label_horizon).shift(-self.cfg.label_horizon)
        out[self.TARGET_COL] = (future_return > 0).astype(float)
        out.loc[out[self.TARGET_COL].isna(), self.TARGET_COL] = np.nan

        return out

    # ------------------------------------------------------------------
    # 主要入口
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        verbose: bool = True,
        run_backtrader: bool = True,
    ) -> CPCVReport:
        """
        執行完整的 CPCV 驗證

        Args:
            df             : 原始 OHLCV DataFrame
            symbol         : 股票代碼（用於報告）
            verbose        : 印出進度
            run_backtrader : 是否執行 Backtrader 回測

        Returns:
            CPCVReport
        """
        timestamp = datetime.now().isoformat()

        if verbose:
            print(f"\n{'='*65}")
            print(f"CPCVRunner: {symbol}")
            print(f"資料: {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
            print(f"Config: N={self.cfg.n_groups}, k={self.cfg.k_test_groups}, "
                  f"C(N,k)={self.cfg.n_combinations}, φ={self.cfg.n_backtest_paths}")
            print(f"Purge: label_horizon={self.cfg.label_horizon}, "
                  f"embargo={self.cfg.embargo_bars}")
            print(f"{'='*65}")

        # ── 1. 特徵工程 ────────────────────────────────────────────
        feat_df = self.build_features(df)

        # ── 2. 生成 C(N,k) 個 fold ────────────────────────────────
        folds = self._splitter.split(feat_df, verbose=False)
        n_active = sum(1 for f in folds if not f.skipped)

        if verbose:
            print(f"\n生成 {len(folds)} folds（{n_active} active）\n")

        # ── 3. 訓練所有 folds（複用 Phase 3.5 trainer）───────────
        # CPCVFold 與 WalkForwardFold 介面相容（均有 purged_train_idx, test_idx, fold_id）
        ml_results = self._trainer.train_all_folds(
            feat_df, folds, self.FEATURE_COLS, self.TARGET_COL, verbose=verbose
        )

        # ── 4. 整合 fold 結果（ML + Backtrader）──────────────────
        fold_reports: List[CPCVFoldReport] = []

        for fold, ml_result in zip(folds, ml_results):
            report = self._build_fold_report(
                fold=fold,
                ml_result=ml_result,
                feat_df=feat_df,
                orig_df=df,
                run_backtrader=run_backtrader,
                verbose=verbose,
            )
            fold_reports.append(report)

        # ── 5. 重建 backtest paths ────────────────────────────────
        cpcv_paths = self._splitter.build_paths(folds)

        # ── 6. 對每條 path 計算績效 ───────────────────────────────
        path_reports: List[CPCVPathReport] = []

        for cpcv_path in cpcv_paths:
            path_report = self._build_path_report(
                cpcv_path=cpcv_path,
                fold_reports=fold_reports,
                feat_df=feat_df,
                orig_df=df,
                run_backtrader=run_backtrader,
            )
            path_reports.append(path_report)

        # ── 7. 計算 Sharpe 分佈統計 ───────────────────────────────
        sharpe_stats = self._compute_path_stats(path_reports)

        n_total = len(fold_reports)
        n_active_count = sum(1 for f in fold_reports if not f.skipped)

        if verbose:
            print(f"\n{'='*65}")
            print(f"CPCV 完成: {symbol}")
            print(f"  Folds          : {n_active_count}/{n_total} active")
            print(f"  Backtest paths : {len(path_reports)}")
            print(f"  Sharpe 分佈    : mean={sharpe_stats['sharpe_mean']:.4f} "
                  f"± {sharpe_stats['sharpe_std']:.4f}")
            print(f"  CI-95          : [{sharpe_stats['sharpe_ci_95'][0]:.4f}, "
                  f"{sharpe_stats['sharpe_ci_95'][1]:.4f}]")
            print(f"  PBO (P≤0)      : {sharpe_stats['pbo']:.2%}")
            print(f"{'='*65}")

        return CPCVReport(
            symbol=symbol,
            run_timestamp=timestamp,
            config=self.cfg.to_dict(),
            folds=fold_reports,
            n_total_folds=n_total,
            n_active_folds=n_active_count,
            n_skipped_folds=n_total - n_active_count,
            paths=path_reports,
            n_backtest_paths=len(path_reports),
            summary_sharpe_mean=sharpe_stats["sharpe_mean"],
            summary_sharpe_std=sharpe_stats["sharpe_std"],
            summary_sharpe_ci_95=tuple(sharpe_stats["sharpe_ci_95"]),
            summary_return_mean=sharpe_stats["return_mean"],
            summary_max_drawdown_mean=sharpe_stats["max_drawdown_mean"],
            summary_pbo=sharpe_stats["pbo"],
        )

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _build_fold_report(
        self,
        fold: CPCVFold,
        ml_result: EnsembleFoldResult,
        feat_df: pd.DataFrame,
        orig_df: pd.DataFrame,
        run_backtrader: bool,
        verbose: bool,
    ) -> CPCVFoldReport:
        """建立單一 fold 的完整報告"""

        if fold.skipped or ml_result.skipped:
            reason = fold.skip_reason or ml_result.skip_reason
            return CPCVFoldReport(
                fold_id=fold.fold_id,
                combination=fold.combination,
                train_groups=fold.train_group_ids,
                test_groups=fold.test_group_ids,
                train_bars=fold.n_train_purged,
                test_bars=fold.n_test,
                skipped=True,
                skip_reason=reason,
            )

        # 收集各模型 accuracy
        model_scores = {}
        for mr in ml_result.model_results:
            if not mr.skipped:
                model_scores[mr.model_name] = mr.accuracy

        report = CPCVFoldReport(
            fold_id=fold.fold_id,
            combination=fold.combination,
            train_groups=fold.train_group_ids,
            test_groups=fold.test_group_ids,
            train_bars=fold.n_train_purged,
            test_bars=fold.n_test,
            ensemble_accuracy=ml_result.ensemble_accuracy,
            ensemble_roc_auc=ml_result.ensemble_roc_auc,
            model_scores=model_scores,
        )

        # Backtrader 回測（可選）
        if run_backtrader and len(ml_result.ensemble_proba) > 0:
            try:
                bt_result = self._run_backtrader_for_fold(
                    fold=fold,
                    proba=ml_result.ensemble_proba,
                    feat_df=feat_df,
                    orig_df=orig_df,
                )
                report.total_return_pct = bt_result.total_return_pct
                report.sharpe_ratio = bt_result.sharpe_ratio
                report.max_drawdown_pct = bt_result.max_drawdown_pct
                report.win_rate_pct = bt_result.win_rate_pct
                report.total_trades = bt_result.total_trades

                if bt_result.skipped:
                    report.error = bt_result.error

            except Exception as e:
                report.error = f"Backtrader error: {e}"
                warnings.warn(f"CPCVFold {fold.fold_id} backtrader error: {e}")

        return report

    def _run_backtrader_for_fold(
        self,
        fold: CPCVFold,
        proba: np.ndarray,
        feat_df: pd.DataFrame,
        orig_df: pd.DataFrame,
    ) -> BacktraderFoldResult:
        """將 ML 機率信號轉為 Backtrader 回測"""
        test_df_feat = feat_df.iloc[fold.test_idx]
        test_dates = test_df_feat.index

        signals: Dict[str, float] = {}
        for date, proba_val in zip(test_dates, proba):
            if hasattr(date, "date"):
                date_str = date.date().isoformat()
            else:
                date_str = str(date)[:10]
            signals[date_str] = float(proba_val)

        price_df = orig_df.copy()
        price_df.columns = [c.lower() for c in price_df.columns]
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)
        if price_df.index.tz is not None:
            price_df.index = price_df.index.tz_localize(None)

        test_price_df = price_df.iloc[fold.test_idx]

        return run_backtrader_fold(
            price_df=test_price_df,
            signals=signals,
            config=self._wf_cfg,
            fold_id=fold.fold_id,
        )

    def _build_path_report(
        self,
        cpcv_path: CPCVPath,
        fold_reports: List[CPCVFoldReport],
        feat_df: pd.DataFrame,
        orig_df: pd.DataFrame,
        run_backtrader: bool,
    ) -> CPCVPathReport:
        """
        對單條 backtest path 計算績效

        path 由多個 fold 的測試片段組成。
        這裡以各 fold 中對應片段的 Sharpe 加權平均作為 path 績效估計。
        """
        fold_id_set = set(cpcv_path.fold_ids)
        contributing = [f for f in fold_reports if f.fold_id in fold_id_set and not f.skipped]

        if not contributing:
            return CPCVPathReport(
                path_id=cpcv_path.path_id,
                group_id=cpcv_path.group_id,
                fold_ids=cpcv_path.fold_ids,
                n_bars=len(cpcv_path.test_idx),
                error="no active contributing folds",
            )

        # Path 績效：對各 fold 指標做加權平均（以 test_bars 為權重）
        total_bars = sum(f.test_bars for f in contributing)

        if total_bars == 0:
            return CPCVPathReport(
                path_id=cpcv_path.path_id,
                group_id=cpcv_path.group_id,
                fold_ids=cpcv_path.fold_ids,
                n_bars=len(cpcv_path.test_idx),
                error="total_bars=0",
            )

        weights = [f.test_bars / total_bars for f in contributing]

        sharpe = sum(f.sharpe_ratio * w for f, w in zip(contributing, weights))
        total_return = sum(f.total_return_pct * w for f, w in zip(contributing, weights))
        max_dd = sum(f.max_drawdown_pct * w for f, w in zip(contributing, weights))

        # Calmar ratio = |total_return| / max_drawdown（防止 0 分母）
        calmar = abs(total_return) / max(abs(max_dd), 1e-6)

        return CPCVPathReport(
            path_id=cpcv_path.path_id,
            group_id=cpcv_path.group_id,
            fold_ids=cpcv_path.fold_ids,
            n_bars=len(cpcv_path.test_idx),
            total_return_pct=float(total_return),
            sharpe_ratio=float(sharpe),
            max_drawdown_pct=float(max_dd),
            calmar_ratio=float(calmar),
        )

    def _compute_path_stats(self, paths: List[CPCVPathReport]) -> dict:
        """
        計算 path 間的 Sharpe 分佈統計

        Returns:
            {
                "sharpe_mean": float,
                "sharpe_std": float,
                "sharpe_ci_95": [lower, upper],
                "pbo": float,   # P(Sharpe <= 0)
                "return_mean": float,
                "max_drawdown_mean": float,
            }
        """
        valid = [p for p in paths if p.error is None]

        if not valid:
            return {
                "sharpe_mean": 0.0,
                "sharpe_std": 0.0,
                "sharpe_ci_95": [0.0, 0.0],
                "pbo": 1.0,
                "return_mean": 0.0,
                "max_drawdown_mean": 0.0,
            }

        sharpes = [p.sharpe_ratio for p in valid]
        returns = [p.total_return_pct for p in valid]
        drawdowns = [p.max_drawdown_pct for p in valid]
        n = len(sharpes)

        mean_s = float(np.mean(sharpes))
        std_s = float(np.std(sharpes, ddof=1)) if n > 1 else 0.0

        # 95% 信賴區間（mean ± 1.96 × std/√n）
        margin = 1.96 * std_s / sqrt(n) if n > 0 else 0.0
        ci_lower = mean_s - margin
        ci_upper = mean_s + margin

        # PBO：P(Sharpe <= 0)
        pbo = float(np.mean([s <= 0 for s in sharpes]))

        return {
            "sharpe_mean": mean_s,
            "sharpe_std": std_s,
            "sharpe_ci_95": [ci_lower, ci_upper],
            "pbo": pbo,
            "return_mean": float(np.mean(returns)),
            "max_drawdown_mean": float(np.mean(drawdowns)),
        }

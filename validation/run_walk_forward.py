"""
validation/run_walk_forward.py — PurgedWalkForwardRunner
========================================================

Walk-Forward 驗證框架的頂層 Orchestrator。

整合：
  1. PurgedWalkForwardSplitter — 生成無洩漏的 fold
  2. EnsembleWalkForwardTrainer — 多模型訓練/評估
  3. run_backtrader_fold() — 轉換信號 → 回測績效

完整流程：
  原始資料 → 特徵工程 → 建目標變數（shift）
  → Purged Walk-Forward Split → 每 fold 訓練 + 預測
  → 信號 → Backtrader → 績效指標
  → 彙總報告

Look-ahead Bias 防護（多層）：
  1. 目標變數強制 shift（本模組建立）：
     target[t] = (close[t+h] - close[t]) / close[t] > 0
     → close[t+h] 屬於未來，但訓練時 t+h 已在 purge zone 外
  2. Purge：移除訓練集末尾 label_horizon bars（含測試期資料的樣本）
  3. Embargo：額外 embargo_bars bars 緩衝
  4. 嚴格驗證：train_idx ∩ test_idx = ∅

作者：Bythos（sub-agent）
建立：2026-02-18
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.walk_forward import WalkForwardConfig
from validation.backtrader_bridge import BacktraderFoldResult, run_backtrader_fold
from validation.ensemble_trainer import EnsembleFoldResult, EnsembleWalkForwardTrainer
from validation.purged_walk_forward import PurgedWalkForwardSplitter, WalkForwardFold

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# FoldReport — 單一 fold 完整報告
# ============================================================================


@dataclass
class FoldReport:
    """單一 fold 的完整報告（ML + Backtrader）"""

    fold_id: int
    train_bars: int = 0
    test_bars: int = 0

    # ML 指標
    ensemble_accuracy: float = 0.0
    ensemble_roc_auc: float = 0.0

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
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "ml": {
                "ensemble_accuracy": round(self.ensemble_accuracy, 4),
                "ensemble_roc_auc": round(self.ensemble_roc_auc, 4),
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


@dataclass
class WalkForwardReport:
    """完整 Walk-Forward 驗證報告"""

    symbol: str
    run_timestamp: str
    config: Dict[str, Any]
    folds: List[FoldReport] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "run_timestamp": self.run_timestamp,
            "config": self.config,
            "folds": [f.to_dict() for f in self.folds],
            "summary": self.summary,
        }


# ============================================================================
# PurgedWalkForwardRunner — 主 Orchestrator
# ============================================================================


class PurgedWalkForwardRunner:
    """
    Purged Walk-Forward 驗證框架 Orchestrator

    使用範例::

        import yfinance as yf
        from config.walk_forward import WalkForwardConfig
        from validation.run_walk_forward import PurgedWalkForwardRunner

        cfg = WalkForwardConfig()
        runner = PurgedWalkForwardRunner(cfg)

        df = yf.download("2330.TW", start="2020-01-01", end="2024-12-31")
        report = runner.run(df, symbol="2330.TW")
        print(report.to_dict())

    特徵工程（內建）：
    - RSI(14), MACD histogram, MA crossover (10/30, 20/60)
    - Bollinger %B, Volume ratio, Momentum (5/10/20d), Volatility(20d)

    目標變數（內建，正確 shift）：
    - target[t] = (close[t + label_horizon] > close[t]) ? 1 : 0
    - 注意：NaN 會在末尾產生（label_horizon 個樣本），dropna 後訓練
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

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.cfg = config or WalkForwardConfig()
        self.cfg.validate()
        self._splitter = PurgedWalkForwardSplitter(self.cfg)
        self._trainer = EnsembleWalkForwardTrainer(self.cfg)

    # ------------------------------------------------------------------
    # 特徵工程
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算技術指標特徵並建立目標變數（含正確 shift）

        Args:
            df: OHLCV DataFrame（欄位大小寫均可）

        Returns:
            原始 df + feature cols + "target" col

        Look-ahead Bias 保護：
            target[t] 使用 close[t + label_horizon]（未來），
            但訓練時只使用 purged_train_idx（已排除洩漏樣本）。
        """
        out = df.copy()
        out.columns = [c.lower() for c in out.columns]

        close = out["close"]
        volume = out["volume"]

        # ── 1. RSI(14) ────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - (100 / (1 + rs))

        # ── 2. MACD histogram ─────────────────────────────────────────
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = macd - signal

        # ── 3. MA crossover ───────────────────────────────────────────
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma30 = close.rolling(30).mean()
        ma60 = close.rolling(60).mean()
        out["ma_cross_10_30"] = (ma10 - ma30) / close
        out["ma_cross_20_60"] = (ma20 - ma60) / close

        # ── 4. Bollinger Band %B ──────────────────────────────────────
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower)

        # ── 5. Volume ratio ───────────────────────────────────────────
        out["vol_ratio"] = volume / volume.rolling(20).mean()

        # ── 6. Momentum ───────────────────────────────────────────────
        out["momentum_5d"] = close.pct_change(5)
        out["momentum_10d"] = close.pct_change(10)
        out["momentum_20d"] = close.pct_change(20)

        # ── 7. Volatility ─────────────────────────────────────────────
        out["volatility_20d"] = close.pct_change().rolling(20).std()

        # ── 8. 目標變數（正確 shift，防止前視偏差）────────────────────
        # target[t] = 1 if close[t + label_horizon] > close[t] else 0
        # shift(-label_horizon) → 把未來值往前移，使 target[t] = future value
        # 注意：末尾 label_horizon 個 bar 的 target 為 NaN（正確行為）
        future_return = close.pct_change(self.cfg.label_horizon).shift(-self.cfg.label_horizon)
        out[self.TARGET_COL] = (future_return > 0).astype(float)
        # NaN 保留（在訓練時由 dropna 處理）
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
    ) -> WalkForwardReport:
        """
        執行完整的 Purged Walk-Forward 驗證

        Args:
            df             : 原始 OHLCV DataFrame
            symbol         : 股票代碼（用於報告）
            verbose        : 印出進度
            run_backtrader : 是否執行 Backtrader 回測（False 則只做 ML 評估）

        Returns:
            WalkForwardReport
        """
        timestamp = datetime.now().isoformat()

        if verbose:
            print(f"\n{'='*65}")
            print(f"PurgedWalkForwardRunner: {symbol}")
            print(f"資料: {len(df)} bars  |  {df.index[0].date()} → {df.index[-1].date()}")
            print(f"Config: train={self.cfg.train_window}b / "
                  f"test={self.cfg.test_window}b / "
                  f"step={self.cfg.step_size}b / "
                  f"purge={self.cfg.label_horizon}b / "
                  f"embargo={self.cfg.embargo_bars}b")
            print(f"成本: buy={self.cfg.effective_buy_rate:.4%} / "
                  f"sell={self.cfg.effective_sell_rate:.4%} / "
                  f"round-trip={self.cfg.round_trip_cost:.4%}")
            print(f"{'='*65}")

        # ── 1. 特徵工程 ────────────────────────────────────────────────
        feat_df = self.build_features(df)

        # ── 2. 生成 folds ─────────────────────────────────────────────
        folds = self._splitter.split(feat_df, verbose=False)
        n_active = sum(1 for f in folds if not f.skipped)

        if verbose:
            print(f"\n生成 {len(folds)} folds（{n_active} active）\n")

        # ── 3. 訓練所有 folds ─────────────────────────────────────────
        ml_results = self._trainer.train_all_folds(
            feat_df, folds, self.FEATURE_COLS, self.TARGET_COL, verbose=verbose
        )

        # ── 4. 整合結果 ────────────────────────────────────────────────
        fold_reports = []
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

        # ── 5. 彙總統計 ────────────────────────────────────────────────
        summary = self._summarize(fold_reports)

        if verbose:
            print(f"\n{'='*65}")
            print(f"Walk-Forward 完成: {symbol}")
            print(f"  Active folds    : {summary['n_active_folds']}/{summary['n_total_folds']}")
            print(f"  Ensemble Acc    : {summary.get('ml', {}).get('ensemble_accuracy', {}).get('mean', 0):.4f}")
            print(f"  Ensemble AUC    : {summary.get('ml', {}).get('ensemble_roc_auc', {}).get('mean', 0):.4f}")
            if run_backtrader and "backtest" in summary:
                bt_s = summary["backtest"]
                print(f"  Avg Return      : {bt_s.get('avg_return_pct', 0):.4f}%")
                print(f"  Avg Sharpe      : {bt_s.get('avg_sharpe', 0):.4f}")
                print(f"  Avg MaxDD       : {bt_s.get('avg_max_drawdown_pct', 0):.4f}%")
            print(f"{'='*65}")

        return WalkForwardReport(
            symbol=symbol,
            run_timestamp=timestamp,
            config=self.cfg.to_dict(),
            folds=fold_reports,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _build_fold_report(
        self,
        fold: WalkForwardFold,
        ml_result: EnsembleFoldResult,
        feat_df: pd.DataFrame,
        orig_df: pd.DataFrame,
        run_backtrader: bool,
        verbose: bool,
    ) -> FoldReport:
        """建立單一 fold 的完整報告"""

        if fold.skipped or ml_result.skipped:
            reason = fold.skip_reason or ml_result.skip_reason
            return FoldReport(
                fold_id=fold.fold_id,
                skipped=True,
                skip_reason=reason,
                train_bars=fold.n_train_purged,
                test_bars=fold.n_test,
            )

        # ML 指標
        report = FoldReport(
            fold_id=fold.fold_id,
            train_bars=fold.n_train_purged,
            test_bars=fold.n_test,
            ensemble_accuracy=ml_result.ensemble_accuracy,
            ensemble_roc_auc=ml_result.ensemble_roc_auc,
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
                warnings.warn(f"Fold {fold.fold_id} backtrader error: {e}")

        return report

    def _run_backtrader_for_fold(
        self,
        fold: WalkForwardFold,
        proba: np.ndarray,
        feat_df: pd.DataFrame,
        orig_df: pd.DataFrame,
    ) -> BacktraderFoldResult:
        """將 ML 機率信號轉為 Backtrader 回測"""

        # 取出測試期的原始價格資料
        test_df_feat = feat_df.iloc[fold.test_idx]
        test_dates = test_df_feat.index

        # 建立信號字典（date_str → P(up)）
        signals: Dict[str, float] = {}
        for i, (date, proba_val) in enumerate(zip(test_dates, proba)):
            if hasattr(date, "date"):
                date_str = date.date().isoformat()
            else:
                date_str = str(date)[:10]
            signals[date_str] = float(proba_val)

        # 取對應的 OHLCV 資料
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
            config=self.cfg,
            fold_id=fold.fold_id,
        )

    @staticmethod
    def _summarize(fold_reports: List[FoldReport]) -> Dict[str, Any]:
        """彙總所有 fold 的統計指標"""
        active = [f for f in fold_reports if not f.skipped]

        if not active:
            return {
                "n_total_folds": len(fold_reports),
                "n_active_folds": 0,
                "note": "No active folds",
            }

        accs = [f.ensemble_accuracy for f in active]
        aucs = [f.ensemble_roc_auc for f in active]
        returns = [f.total_return_pct for f in active]
        sharpes = [f.sharpe_ratio for f in active]
        drawdowns = [f.max_drawdown_pct for f in active]

        return {
            "n_total_folds": len(fold_reports),
            "n_active_folds": len(active),
            "n_skipped_folds": len(fold_reports) - len(active),
            "ml": {
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
            },
            "backtest": {
                "avg_return_pct": round(float(np.mean(returns)), 4),
                "median_return_pct": round(float(np.median(returns)), 4),
                "std_return_pct": round(float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0, 4),
                "best_return_pct": round(float(max(returns)), 4),
                "worst_return_pct": round(float(min(returns)), 4),
                "avg_sharpe": round(float(np.mean(sharpes)), 4),
                "avg_max_drawdown_pct": round(float(np.mean(drawdowns)), 4),
                "positive_return_folds": sum(1 for r in returns if r > 0),
            },
        }

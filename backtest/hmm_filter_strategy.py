"""
hmm_filter_strategy.py - HMM Filter Strategy
=============================================

HMMFilterStrategy: uses MarketHMM to predict market regime (Bull/Sideways/Bear)
and layers it on top of RF signals as a filter:

  - Bull   (state 0) -> allow RF BUY signals
  - Sideways (state 1) -> block new entries; hold existing position
  - Bear   (state 2) -> force close position (regardless of RF signal)

Strategy params::

    forward_days          RF prediction horizon in days (default 5)
    confidence_threshold  RF confidence threshold (default 0.50)
    retrain_period        RF retrain every N bars (default 60)
    hmm_window            HMM training window in bars (default 252)
    hmm_n_states          number of HMM hidden states (default 3)
    hmm_retrain_period    HMM retrain every N bars (default 252)

Regime labels (MarketHMM standardised):
  0 -> Bull
  1 -> Sideways
  2 -> Bear

Author: Bythos (sub-agent)
Created: 2026-02-18
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import backtrader as bt
import numpy as np
import pandas as pd

from hmm.market_hmm import MarketHMM
from backtest.rf_strategy import RandomForestPredictor

warnings.filterwarnings("ignore", category=UserWarning)


# ── Regime constants ──────────────────────────────────────────────────────
REGIME_BULL = 0
REGIME_SIDEWAYS = 1
REGIME_BEAR = 2
REGIME_LABELS = {REGIME_BULL: "Bull", REGIME_SIDEWAYS: "Sideways", REGIME_BEAR: "Bear"}
REGIME_UNKNOWN = -1  # not yet predicted


class HMMFilterStrategy(bt.Strategy):
    """
    HMM Filter Strategy (inherits backtrader.Strategy).

    Combines MarketHMM regime prediction with RandomForestPredictor signals.
    Only enters long positions in Bull regime; forces exit in Bear regime.

    Example usage::

        from backtest.hmm_filter_strategy import HMMFilterStrategy
        from backtest.backtrader_engine import BacktraderEngine

        engine = BacktraderEngine("2330.TW", initial_capital=500_000)
        result = engine.run(
            strategy_class=HMMFilterStrategy,
            data=ohlcv_df,
            strategy_params={
                "hmm_window": 252,
                "hmm_n_states": 3,
                "confidence_threshold": 0.55,
            },
            strategy_name="HMM Filter Strategy",
        )
    """

    params = (
        # RF params
        ("forward_days", 5),
        ("confidence_threshold", 0.50),
        ("retrain_period", 60),
        # HMM params
        ("hmm_window", 252),
        ("hmm_n_states", 3),
        ("hmm_retrain_period", 252),
    )

    def __init__(self):
        self.order: Optional[bt.Order] = None

        # RF predictor
        self.predictor = RandomForestPredictor(
            forward_days=self.p.forward_days,
            confidence_threshold=self.p.confidence_threshold,
        )
        self.days_since_rf_train: int = 0
        self.rf_trained: bool = False

        # HMM model
        self.hmm = MarketHMM(n_states=self.p.hmm_n_states)
        self.days_since_hmm_train: int = 0
        self.hmm_trained: bool = False

        # Current regime (REGIME_UNKNOWN until HMM is fitted)
        self.current_regime: int = REGIME_UNKNOWN

        # Equity curve tracking
        self._equity_dates: List[str] = []
        self._equity_values: List[float] = []

        # Trade statistics for testing / reporting
        # NOTE: must NOT be named self.stats — conflicts with Backtrader built-in
        self.trade_stats: Dict[str, int] = {
            "bull_allowed": 0,
            "sideways_blocked": 0,
            "bear_forced_close": 0,
            "rf_buy_signals": 0,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_history_df(self, max_bars: int = 500) -> pd.DataFrame:
        """
        Convert Backtrader data feed into a pandas OHLCV DataFrame.

        Returns
        -------
        DataFrame with columns: Open, High, Low, Close, Volume
        """
        d = self.datas[0]
        n = min(max_bars, len(d))
        indices = range(-n + 1, 1)

        df = pd.DataFrame(
            {
                "Open": [d.open[i] for i in indices],
                "High": [d.high[i] for i in indices],
                "Low": [d.low[i] for i in indices],
                "Close": [d.close[i] for i in indices],
                "Volume": [d.volume[i] for i in indices],
            },
            index=pd.DatetimeIndex([d.datetime.date(i) for i in indices]),
        )
        return df

    def _get_hmm_df(self, max_bars: Optional[int] = None) -> pd.DataFrame:
        """
        Return lowercase-column DataFrame (close, volume) for HMM input.
        """
        n = max_bars or self.p.hmm_window
        df = self._get_history_df(max_bars=n)
        return df.rename(columns={"Close": "close", "Volume": "volume"})[["close", "volume"]]

    def log(self, msg: str):
        """Print a timestamped log message."""
        dt = self.datas[0].datetime.date(0)
        print(f"[HMM-RF {dt}] {msg}")

    def _regime_label(self) -> str:
        return REGIME_LABELS.get(self.current_regime, "Unknown")

    # ── Backtrader hooks ──────────────────────────────────────────────────

    def notify_order(self, order: bt.Order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    def next(self):
        # Equity curve recording
        dt = self.datas[0].datetime.date(0).isoformat()
        val = self.broker.getvalue()
        self._equity_dates.append(dt)
        self._equity_values.append(val)

        if self.order:
            return

        # ── HMM train / retrain ──────────────────────────────────────────
        self.days_since_hmm_train += 1
        need_hmm_train = (
            not self.hmm_trained
            or self.days_since_hmm_train >= self.p.hmm_retrain_period
        )

        if need_hmm_train:
            hmm_df = self._get_hmm_df(max_bars=self.p.hmm_window)
            if len(hmm_df) >= 50:
                try:
                    self.hmm = MarketHMM(n_states=self.p.hmm_n_states)
                    self.hmm.fit(hmm_df)
                    self.hmm_trained = True
                    self.days_since_hmm_train = 0
                    self.log(f"HMM retrained on {len(hmm_df)} bars")
                except Exception as e:
                    self.log(f"HMM training failed: {e}")

        # ── Predict current regime ────────────────────────────────────────
        if self.hmm_trained:
            try:
                hmm_df = self._get_hmm_df(max_bars=self.p.hmm_window)
                proba_df = self.hmm.predict_proba(hmm_df)
                latest_proba = proba_df.iloc[-1]
                regime_map = {
                    "Bull": REGIME_BULL,
                    "Sideways": REGIME_SIDEWAYS,
                    "Bear": REGIME_BEAR,
                }
                best_col = latest_proba[["Bull", "Sideways", "Bear"]].idxmax()
                self.current_regime = regime_map[best_col]
            except Exception as e:
                self.log(f"HMM predict failed: {e}")
                self.current_regime = REGIME_UNKNOWN

        # ── Bear: force close ─────────────────────────────────────────────
        if self.current_regime == REGIME_BEAR and self.position:
            self.log(
                f"BEAR regime -> force close {self.position.size}"
                f" @ {self.data.close[0]:.2f}"
            )
            self.order = self.sell(size=self.position.size)
            self.trade_stats["bear_forced_close"] += 1
            return

        # ── RF train / retrain ────────────────────────────────────────────
        self.days_since_rf_train += 1
        need_rf_train = (
            not self.rf_trained
            or self.days_since_rf_train >= self.p.retrain_period
        )

        if need_rf_train:
            hist_df = self._get_history_df(max_bars=500)
            ok = self.predictor.train(hist_df)
            if ok:
                self.rf_trained = True
                self.days_since_rf_train = 0
                self.log(f"RF retrained on {len(hist_df)} bars")

        if not self.rf_trained:
            return

        # ── RF prediction ─────────────────────────────────────────────────
        pred_df = self._get_history_df(max_bars=252)
        prediction = self.predictor.predict("STOCK", pred_df)
        signal = prediction.get("signal", "HOLD")
        confidence = prediction.get("confidence", 0)

        # ── HMM-filtered trading decision ──────────────────────────────────
        if not self.position:
            if signal == "BUY":
                self.trade_stats["rf_buy_signals"] += 1

                if self.current_regime == REGIME_BULL:
                    # Bull regime: allow entry
                    size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
                        self.log(
                            f"BUY {size} @ {self.data.close[0]:.2f}"
                            f" (regime=Bull, conf={confidence:.2f}%)"
                        )
                        self.trade_stats["bull_allowed"] += 1
                elif self.current_regime == REGIME_SIDEWAYS:
                    # Sideways: block new entries
                    self.log(
                        f"BUY blocked (regime=Sideways, conf={confidence:.2f}%)"
                    )
                    self.trade_stats["sideways_blocked"] += 1
                else:
                    # Bear or Unknown: do not enter
                    self.log(
                        f"BUY blocked (regime={self._regime_label()},"
                        f" conf={confidence:.2f}%)"
                    )
        else:
            # Holding position: exit on RF SELL (Bear handled above)
            if signal == "SELL":
                self.log(
                    f"SELL {self.position.size} @ {self.data.close[0]:.2f}"
                    f" (RF signal, regime={self._regime_label()},"
                    f" conf={confidence:.2f}%)"
                )
                self.order = self.sell(size=self.position.size)

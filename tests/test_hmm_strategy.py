"""
stock-app/tests/test_hmm_strategy.py

Unit tests for HMMFilterStrategy (Phase 5.2).

Test coverage:
  TC01 - HMMFilterStrategy can be instantiated with default params
  TC02 - HMMFilterStrategy params: hmm_window, hmm_n_states accessible
  TC03 - Bull regime: RF BUY signal stats are consistent
  TC04 - Sideways regime: sideways_blocked counter is a non-negative int
  TC05 - Bear regime: bear_forced_close counter is a non-negative int
  TC06 - REGIME constants: BULL=0, SIDEWAYS=1, BEAR=2, UNKNOWN=-1
  TC07 - Unknown regime: BUY blocked when HMM can't train (too few bars)
  TC08 - Integration: run a minimal backtest end-to-end without errors
  TC09 - REGIME_LABELS covers all three states
  TC10 - HMMFilterStrategy exports from backtest package
  TC11 - Equity curve is populated after backtest
  TC12 - trade_stats dict has expected keys with non-negative int values
  TC13 - hmm_retrain_period param is respected
  TC14 - retrain_period param affects RF retraining cycle
  TC15 - backtest result dict has all expected keys
"""

from __future__ import annotations

import sys
import os

# Ensure stock-app root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import backtrader as bt

from backtest.hmm_filter_strategy import (
    HMMFilterStrategy,
    REGIME_BULL,
    REGIME_SIDEWAYS,
    REGIME_BEAR,
    REGIME_UNKNOWN,
    REGIME_LABELS,
)
from hmm.market_hmm import MarketHMM


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_ohlcv_df(
    n_days: int = 400,
    seed: int = 42,
    regime: str = "mixed",
) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame for testing.

    regime: "bull" / "bear" / "sideways" / "mixed"
    """
    rng = np.random.RandomState(seed)

    n_returns = n_days - 1  # we need exactly this many log-returns

    if regime == "bull":
        log_returns = rng.normal(0.001, 0.008, n_returns)
    elif regime == "bear":
        log_returns = rng.normal(-0.002, 0.015, n_returns)
    elif regime == "sideways":
        log_returns = rng.normal(0.0001, 0.005, n_returns)
    else:  # mixed: three regime blocks summing to exactly n_returns
        # Divide n_returns into 3 parts; last block absorbs any remainder
        third = n_returns // 3
        sizes = [third, third, n_returns - 2 * third]
        params_list = [(0.002, 0.008, 0), (-0.002, 0.015, 1), (0.0, 0.005, 2)]
        parts = [
            np.random.RandomState(seed + es).normal(drft, v, sz)
            for (drft, v, es), sz in zip(params_list, sizes)
        ]
        log_returns = np.concatenate(parts)  # exactly n_returns elements

    assert len(log_returns) == n_returns, (
        f"log_returns length mismatch: {len(log_returns)} != {n_returns}"
    )

    prices = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    volumes = rng.randint(1_000_000, 5_000_000, size=n_days).astype(float)

    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")

    open_ = prices * (1 + rng.uniform(-0.003, 0.003, n_days))
    high_ = prices * (1 + rng.uniform(0.0, 0.010, n_days))
    low_ = prices * (1 - rng.uniform(0.0, 0.010, n_days))

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high_,
            "Low": low_,
            "Close": prices,
            "Volume": volumes,
        },
        index=idx,
    )
    # Ensure High >= Close >= Low
    df["High"] = df[["High", "Close", "Open"]].max(axis=1)
    df["Low"] = df[["Low", "Close", "Open"]].min(axis=1)
    return df


def _run_backtest(
    df: pd.DataFrame,
    strategy_params: dict | None = None,
    initial_capital: float = 100_000,
) -> dict:
    """
    Run a Backtrader cerebro with HMMFilterStrategy and return a result dict.
    """
    params = strategy_params or {}

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)

    data_feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(HMMFilterStrategy, **params)

    strats = cerebro.run()
    strat = strats[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_capital) / initial_capital * 100

    return {
        "strategy": strat,
        "final_value": final_value,
        "initial_capital": initial_capital,
        "total_return": total_return,
        "equity_dates": strat._equity_dates,
        "equity_values": strat._equity_values,
        "stats": strat.trade_stats,
    }


# ── TC01 ──────────────────────────────────────────────────────────────────

def test_tc01_instantiation_default_params():
    """HMMFilterStrategy can be instantiated with default params inside cerebro."""
    df = _make_ohlcv_df(n_days=350)
    result = _run_backtest(df)
    assert result["strategy"] is not None


# ── TC02 ──────────────────────────────────────────────────────────────────

def test_tc02_params_accessible():
    """hmm_window and hmm_n_states params are accessible on the strategy."""
    df = _make_ohlcv_df(n_days=350)
    result = _run_backtest(df, strategy_params={"hmm_window": 200, "hmm_n_states": 3})
    strat = result["strategy"]
    assert strat.p.hmm_window == 200
    assert strat.p.hmm_n_states == 3


# ── TC03 ──────────────────────────────────────────────────────────────────

def test_tc03_bull_regime_stats_consistent():
    """
    bull_allowed <= rf_buy_signals (every allowed buy came from an RF BUY signal).
    sideways_blocked + bull_allowed <= rf_buy_signals (no double counting).
    """
    df = _make_ohlcv_df(n_days=400, regime="bull", seed=42)
    result = _run_backtest(df, strategy_params={"confidence_threshold": 0.45})
    stats = result["stats"]
    assert stats["bull_allowed"] <= stats["rf_buy_signals"]
    assert stats["bull_allowed"] + stats["sideways_blocked"] <= stats["rf_buy_signals"]


# ── TC04 ──────────────────────────────────────────────────────────────────

def test_tc04_sideways_blocking_is_int():
    """sideways_blocked counter is a non-negative int."""
    df = _make_ohlcv_df(n_days=400, regime="sideways", seed=10)
    result = _run_backtest(df, strategy_params={"confidence_threshold": 0.45})
    stats = result["stats"]
    assert isinstance(stats["sideways_blocked"], int)
    assert stats["sideways_blocked"] >= 0


# ── TC05 ──────────────────────────────────────────────────────────────────

def test_tc05_bear_forced_close_is_int():
    """bear_forced_close counter is a non-negative int."""
    df = _make_ohlcv_df(n_days=400, regime="mixed", seed=42)
    result = _run_backtest(df, strategy_params={"confidence_threshold": 0.45})
    stats = result["stats"]
    assert isinstance(stats["bear_forced_close"], int)
    assert stats["bear_forced_close"] >= 0


# ── TC06 ──────────────────────────────────────────────────────────────────

def test_tc06_regime_constants_correct():
    """REGIME_BULL=0, REGIME_SIDEWAYS=1, REGIME_BEAR=2, REGIME_UNKNOWN=-1."""
    assert REGIME_BULL == 0
    assert REGIME_SIDEWAYS == 1
    assert REGIME_BEAR == 2
    assert REGIME_UNKNOWN == -1
    assert REGIME_LABELS[REGIME_BULL] == "Bull"
    assert REGIME_LABELS[REGIME_SIDEWAYS] == "Sideways"
    assert REGIME_LABELS[REGIME_BEAR] == "Bear"


# ── TC07 ──────────────────────────────────────────────────────────────────

def test_tc07_short_data_no_buys():
    """
    With very short data (35 bars), HMM cannot train (needs >= 50 observations
    after feature engineering), so current_regime stays UNKNOWN and no buys occur.
    """
    df = _make_ohlcv_df(n_days=35, regime="bull")
    result = _run_backtest(df, strategy_params={"hmm_window": 252})
    stats = result["stats"]
    assert stats["bull_allowed"] == 0


# ── TC08 ──────────────────────────────────────────────────────────────────

def test_tc08_integration_end_to_end():
    """Run a full backtest with realistic data length without errors."""
    df = _make_ohlcv_df(n_days=500, regime="mixed", seed=7)
    result = _run_backtest(df, strategy_params={"hmm_window": 200, "retrain_period": 50})
    assert "total_return" in result
    assert isinstance(result["total_return"], float)
    assert result["final_value"] > 0


# ── TC09 ──────────────────────────────────────────────────────────────────

def test_tc09_regime_labels_complete():
    """REGIME_LABELS covers all three standardised states."""
    assert set(REGIME_LABELS.keys()) == {REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR}
    assert all(isinstance(v, str) for v in REGIME_LABELS.values())


# ── TC10 ──────────────────────────────────────────────────────────────────

def test_tc10_package_export():
    """HMMFilterStrategy is exported from backtest package."""
    from backtest import HMMFilterStrategy as HFS
    from backtest import REGIME_BULL as RB, REGIME_SIDEWAYS as RS, REGIME_BEAR as RBr
    assert HFS is HMMFilterStrategy
    assert RB == 0
    assert RS == 1
    assert RBr == 2


# ── TC11 ──────────────────────────────────────────────────────────────────

def test_tc11_equity_curve_populated():
    """Equity curve lists are populated during backtest."""
    df = _make_ohlcv_df(n_days=300, regime="bull")
    result = _run_backtest(df)
    assert len(result["equity_dates"]) > 0
    assert len(result["equity_values"]) == len(result["equity_dates"])
    assert all(v > 0 for v in result["equity_values"])


# ── TC12 ──────────────────────────────────────────────────────────────────

def test_tc12_trade_stats_structure():
    """trade_stats dict has expected keys and all values are non-negative ints."""
    df = _make_ohlcv_df(n_days=400, regime="mixed")
    result = _run_backtest(df, strategy_params={"confidence_threshold": 0.45})
    stats = result["stats"]
    expected_keys = {"bull_allowed", "sideways_blocked", "bear_forced_close", "rf_buy_signals"}
    assert expected_keys == set(stats.keys())
    for k, v in stats.items():
        assert isinstance(v, int), f"trade_stats[{k}] should be int, got {type(v)}"
        assert v >= 0, f"trade_stats[{k}] should be non-negative"


# ── TC13 ──────────────────────────────────────────────────────────────────

def test_tc13_hmm_retrain_period_param():
    """hmm_retrain_period param is stored correctly."""
    df = _make_ohlcv_df(n_days=400)
    result = _run_backtest(df, strategy_params={"hmm_retrain_period": 100})
    strat = result["strategy"]
    assert strat.p.hmm_retrain_period == 100


# ── TC14 ──────────────────────────────────────────────────────────────────

def test_tc14_rf_retrain_period_param():
    """retrain_period param is stored correctly."""
    df = _make_ohlcv_df(n_days=400)
    result = _run_backtest(df, strategy_params={"retrain_period": 30})
    strat = result["strategy"]
    assert strat.p.retrain_period == 30


# ── TC15 ──────────────────────────────────────────────────────────────────

def test_tc15_backtest_result_structure():
    """_run_backtest returns dict with all expected keys."""
    df = _make_ohlcv_df(n_days=300)
    result = _run_backtest(df)
    required_keys = {
        "strategy", "final_value", "initial_capital", "total_return",
        "equity_dates", "equity_values", "stats",
    }
    assert required_keys.issubset(set(result.keys()))
    assert result["initial_capital"] == 100_000
    assert isinstance(result["final_value"], float)

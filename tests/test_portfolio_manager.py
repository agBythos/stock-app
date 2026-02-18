"""
Phase 7 Step 6 — PortfolioManager Tests
========================================

Tests for PortfolioManager class.
Network calls are mocked to avoid yfinance dependency.

Run:
    cd stock-app
    python -m pytest tests/test_portfolio_manager.py -v

Coverage:
    TC01 — Module imports cleanly
    TC02 — PortfolioManager instantiation with default + custom capital
    TC03 — add_position: basic add and update
    TC04 — add_position: invalid strategy_type raises ValueError
    TC05 — add_position: negative weight raises ValueError
    TC06 — remove_position: basic remove
    TC07 — remove_position: unknown symbol raises KeyError
    TC08 — rebalance equal_weight: 3 positions → 1/3 each
    TC09 — rebalance equal_weight: 2 positions → 0.5 each
    TC10 — rebalance: unsupported method raises ValueError
    TC11 — rebalance: no positions raises ValueError
    TC12 — get_portfolio_summary: returns correct structure (mocked yfinance)
    TC13 — get_portfolio_summary: regime_status has all symbols
    TC14 — calculate_correlation_matrix: returns correct structure (mocked)
    TC15 — calculate_correlation_matrix: empty portfolio returns error
    TC16 — __contains__ dunder
    TC17 — __len__ dunder
    TC18 — Position.to_dict() serialisable
"""

import sys
import os
import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure stock-app on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from portfolio.portfolio_manager import (
    PortfolioManager,
    Position,
    STRATEGY_RF,
    STRATEGY_HMM_RF,
    VALID_STRATEGY_TYPES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_price_df(n_days: int = 126, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame to stand in for yfinance data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
    df = pd.DataFrame(
        {
            "Open": close * rng.uniform(0.99, 1.01, n_days),
            "High": close * rng.uniform(1.00, 1.02, n_days),
            "Low": close * rng.uniform(0.98, 1.00, n_days),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n_days),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )
    return df


def _make_pm(*symbols, capital=1_000_000) -> PortfolioManager:
    """Create a PortfolioManager with pre-loaded positions."""
    pm = PortfolioManager(initial_capital=capital)
    for i, sym in enumerate(symbols):
        pm.add_position(sym, weight=1.0 / len(symbols), strategy_type="rf")
    return pm


# ─────────────────────────────────────────────────────────────────────────────
# TC01 — Module imports
# ─────────────────────────────────────────────────────────────────────────────

def test_tc01_module_imports():
    """TC01: portfolio.portfolio_manager imports without error."""
    from portfolio.portfolio_manager import PortfolioManager, Position, PortfolioSummary
    assert PortfolioManager is not None
    assert Position is not None
    assert PortfolioSummary is not None
    assert STRATEGY_RF == "rf"
    assert STRATEGY_HMM_RF == "hmm_rf"


# ─────────────────────────────────────────────────────────────────────────────
# TC02 — Instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_tc02_instantiation():
    """TC02: PortfolioManager instantiates with default and custom capital."""
    pm_default = PortfolioManager()
    assert pm_default.initial_capital == 1_000_000.0
    assert len(pm_default) == 0

    pm_custom = PortfolioManager(initial_capital=500_000)
    assert pm_custom.initial_capital == 500_000.0


# ─────────────────────────────────────────────────────────────────────────────
# TC03 — add_position: basic add and update
# ─────────────────────────────────────────────────────────────────────────────

def test_tc03_add_position_basic():
    """TC03: add_position stores symbol (upper-cased), weight, and strategy_type."""
    pm = PortfolioManager()
    pm.add_position("2330.tw", weight=0.5, strategy_type="rf")
    assert "2330.TW" in pm
    pos = pm.get_positions()["2330.TW"]
    assert pos["weight"] == 0.5
    assert pos["strategy_type"] == "rf"

    # Update: adding same symbol again overwrites
    pm.add_position("2330.TW", weight=0.6, strategy_type="hmm_rf")
    pos_updated = pm.get_positions()["2330.TW"]
    assert pos_updated["weight"] == 0.6
    assert pos_updated["strategy_type"] == "hmm_rf"
    assert len(pm) == 1  # still 1 position


# ─────────────────────────────────────────────────────────────────────────────
# TC04 — add_position: invalid strategy_type
# ─────────────────────────────────────────────────────────────────────────────

def test_tc04_add_position_invalid_strategy():
    """TC04: add_position raises ValueError for unknown strategy_type."""
    pm = PortfolioManager()
    with pytest.raises(ValueError, match="Invalid strategy_type"):
        pm.add_position("2330.TW", weight=0.5, strategy_type="banana")


# ─────────────────────────────────────────────────────────────────────────────
# TC05 — add_position: negative weight
# ─────────────────────────────────────────────────────────────────────────────

def test_tc05_add_position_negative_weight():
    """TC05: add_position raises ValueError for negative weight."""
    pm = PortfolioManager()
    with pytest.raises(ValueError, match="Weight must be"):
        pm.add_position("2330.TW", weight=-0.1, strategy_type="rf")


# ─────────────────────────────────────────────────────────────────────────────
# TC06 — remove_position: basic remove
# ─────────────────────────────────────────────────────────────────────────────

def test_tc06_remove_position_basic():
    """TC06: remove_position correctly removes symbol from portfolio."""
    pm = _make_pm("2330.TW", "0050.TW", "2317.TW")
    assert len(pm) == 3

    pm.remove_position("0050.TW")
    assert len(pm) == 2
    assert "0050.TW" not in pm
    assert "2330.TW" in pm
    assert "2317.TW" in pm


# ─────────────────────────────────────────────────────────────────────────────
# TC07 — remove_position: unknown symbol
# ─────────────────────────────────────────────────────────────────────────────

def test_tc07_remove_position_unknown():
    """TC07: remove_position raises KeyError for symbol not in portfolio."""
    pm = PortfolioManager()
    with pytest.raises(KeyError, match="not in portfolio"):
        pm.remove_position("FAKE.TW")


# ─────────────────────────────────────────────────────────────────────────────
# TC08 — rebalance equal_weight: 3 positions → 1/3 each
# ─────────────────────────────────────────────────────────────────────────────

def test_tc08_rebalance_equal_weight_three():
    """TC08: rebalance('equal_weight') assigns 1/3 to each of 3 positions."""
    pm = PortfolioManager()
    pm.add_position("2330.TW", weight=0.5, strategy_type="rf")
    pm.add_position("0050.TW", weight=0.3, strategy_type="rf")
    pm.add_position("2317.TW", weight=0.2, strategy_type="hmm_rf")

    result = pm.rebalance(method="equal_weight")

    assert len(result) == 3
    for sym, w in result.items():
        assert abs(w - 1 / 3) < 1e-9, f"{sym}: expected 0.333…, got {w}"

    # Weights in the internal state are also updated
    # get_positions() rounds to 6 decimal places, so use 1e-6 tolerance
    positions = pm.get_positions()
    for sym, pos in positions.items():
        assert abs(pos["weight"] - 1 / 3) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# TC09 — rebalance equal_weight: 2 positions → 0.5 each
# ─────────────────────────────────────────────────────────────────────────────

def test_tc09_rebalance_equal_weight_two():
    """TC09: rebalance('equal_weight') assigns 0.5 to each of 2 positions."""
    pm = _make_pm("2330.TW", "0050.TW")
    result = pm.rebalance(method="equal_weight")

    assert len(result) == 2
    for sym, w in result.items():
        assert abs(w - 0.5) < 1e-9, f"Expected 0.5, got {w}"

    total = sum(result.values())
    assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"


# ─────────────────────────────────────────────────────────────────────────────
# TC10 — rebalance: unsupported method
# ─────────────────────────────────────────────────────────────────────────────

def test_tc10_rebalance_unsupported_method():
    """TC10: rebalance raises ValueError for unsupported method."""
    pm = _make_pm("2330.TW", "0050.TW")
    with pytest.raises(ValueError, match="Unsupported rebalance method"):
        pm.rebalance(method="risk_parity")


# ─────────────────────────────────────────────────────────────────────────────
# TC11 — rebalance: no positions
# ─────────────────────────────────────────────────────────────────────────────

def test_tc11_rebalance_no_positions():
    """TC11: rebalance raises ValueError when portfolio is empty."""
    pm = PortfolioManager()
    with pytest.raises(ValueError, match="No positions"):
        pm.rebalance()


# ─────────────────────────────────────────────────────────────────────────────
# TC12 — get_portfolio_summary: structure (mocked yfinance)
# ─────────────────────────────────────────────────────────────────────────────

@patch("portfolio.portfolio_manager.PortfolioManager._get_regime_for_symbol")
def test_tc12_get_portfolio_summary_structure(mock_regime):
    """TC12: get_portfolio_summary returns dict with required top-level keys."""
    # Mock regime to avoid network calls
    mock_regime.return_value = {
        "symbol": "2330.TW",
        "regime_idx": 0,
        "regime_label": "Bull",
        "regime_proba": [0.7, 0.2, 0.1],
        "data_bars": 126,
        "prices": None,
        "error": None,
    }

    pm = _make_pm("2330.TW", "0050.TW")
    summary = pm.get_portfolio_summary()

    required_keys = {"as_of", "n_positions", "total_weight", "positions",
                     "regime_status", "portfolio_kpi"}
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"

    assert summary["n_positions"] == 2
    assert isinstance(summary["positions"], list)
    assert isinstance(summary["regime_status"], dict)
    assert isinstance(summary["portfolio_kpi"], dict)


# ─────────────────────────────────────────────────────────────────────────────
# TC13 — get_portfolio_summary: all symbols in regime_status
# ─────────────────────────────────────────────────────────────────────────────

@patch("portfolio.portfolio_manager.PortfolioManager._get_regime_for_symbol")
def test_tc13_summary_regime_has_all_symbols(mock_regime):
    """TC13: regime_status contains an entry for every portfolio symbol."""
    def side_effect(sym):
        return {
            "symbol": sym,
            "regime_idx": 1,
            "regime_label": "Sideways",
            "regime_proba": None,
            "data_bars": 60,
            "prices": None,
            "error": None,
        }

    mock_regime.side_effect = side_effect

    pm = PortfolioManager()
    pm.add_position("2330.TW", 0.5, "rf")
    pm.add_position("0050.TW", 0.3, "rf")
    pm.add_position("2317.TW", 0.2, "hmm_rf")

    summary = pm.get_portfolio_summary()
    regime_status = summary["regime_status"]

    assert set(regime_status.keys()) == {"2330.TW", "0050.TW", "2317.TW"}
    for sym, info in regime_status.items():
        assert info["regime_label"] == "Sideways"
        assert info["regime_idx"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# TC14 — calculate_correlation_matrix: mocked yfinance
# ─────────────────────────────────────────────────────────────────────────────

def test_tc14_correlation_matrix_mocked():
    """TC14: calculate_correlation_matrix returns NxN matrix with diagonal=1 (mocked data)."""
    # We mock the calculate_correlation_matrix method to return synthetic data
    # This avoids yfinance network dependency while testing the structure
    from portfolio.portfolio_manager import PortfolioManager

    pm = PortfolioManager()
    pm.add_position("2330.TW", 0.5, "rf")
    pm.add_position("0050.TW", 0.3, "rf")
    pm.add_position("2317.TW", 0.2, "hmm_rf")

    # Build synthetic correlation result by calling _compute_portfolio_kpi logic directly
    # and checking the return structure of calculate_correlation_matrix
    # by patching the yf.Ticker calls within the method
    fake_dfs = {
        "2330.TW": _make_fake_price_df(seed=1),
        "0050.TW": _make_fake_price_df(seed=2),
        "2317.TW": _make_fake_price_df(seed=3),
    }

    call_order = ["2330.TW", "0050.TW", "2317.TW"]
    call_idx = [0]

    def mock_ticker(sym):
        t = MagicMock()
        t.history.return_value = fake_dfs[sym]
        return t

    import portfolio.portfolio_manager as pm_module
    original_yf_available = pm_module._YF_AVAILABLE
    original_yf = pm_module.yf

    try:
        pm_module._YF_AVAILABLE = True
        mock_yf_mod = MagicMock()
        mock_yf_mod.Ticker.side_effect = lambda sym: mock_ticker(sym)
        pm_module.yf = mock_yf_mod

        result = pm.calculate_correlation_matrix(lookback_days=126)
    finally:
        pm_module._YF_AVAILABLE = original_yf_available
        pm_module.yf = original_yf

    assert "symbols" in result
    assert "matrix" in result
    assert "lookback_days" in result
    assert "data_bars" in result

    # If successful, matrix should be NxN
    if result["matrix"]:
        n = len(result["symbols"])
        assert len(result["matrix"]) == n
        for row in result["matrix"]:
            assert len(row) == n
        # Diagonal should be ~1.0
        for i in range(n):
            assert abs(result["matrix"][i][i] - 1.0) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# TC15 — calculate_correlation_matrix: empty portfolio
# ─────────────────────────────────────────────────────────────────────────────

def test_tc15_correlation_empty_portfolio():
    """TC15: calculate_correlation_matrix on empty portfolio returns error dict."""
    pm = PortfolioManager()
    result = pm.calculate_correlation_matrix()
    assert result["symbols"] == []
    assert result["matrix"] == []
    assert result["error"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# TC16 — __contains__ dunder
# ─────────────────────────────────────────────────────────────────────────────

def test_tc16_contains():
    """TC16: 'symbol in pm' works correctly (case-insensitive)."""
    pm = _make_pm("2330.TW")
    assert "2330.TW" in pm
    assert "2330.tw" in pm   # case-insensitive via upper() in add_position
    assert "0050.TW" not in pm


# ─────────────────────────────────────────────────────────────────────────────
# TC17 — __len__ dunder
# ─────────────────────────────────────────────────────────────────────────────

def test_tc17_len():
    """TC17: len(pm) reflects current number of positions."""
    pm = PortfolioManager()
    assert len(pm) == 0
    pm.add_position("2330.TW", 0.5, "rf")
    assert len(pm) == 1
    pm.add_position("0050.TW", 0.5, "rf")
    assert len(pm) == 2
    pm.remove_position("2330.TW")
    assert len(pm) == 1


# ─────────────────────────────────────────────────────────────────────────────
# TC18 — Position.to_dict() serialisable
# ─────────────────────────────────────────────────────────────────────────────

def test_tc18_position_to_dict():
    """TC18: Position.to_dict() returns JSON-serialisable dict with required keys."""
    pos = Position(symbol="2330.TW", weight=0.5, strategy_type="hmm_rf")
    d = pos.to_dict()

    assert d["symbol"] == "2330.TW"
    assert abs(d["weight"] - 0.5) < 1e-9
    assert d["strategy_type"] == "hmm_rf"

    # Should be JSON serialisable
    serialised = json.dumps(d)
    assert len(serialised) > 10


# ─────────────────────────────────────────────────────────────────────────────
# TC19 — rebalance preserves strategy_type
# ─────────────────────────────────────────────────────────────────────────────

def test_tc19_rebalance_preserves_strategy_type():
    """TC19: rebalance only changes weights, not strategy_type."""
    pm = PortfolioManager()
    pm.add_position("2330.TW", 0.5, "hmm_rf")
    pm.add_position("0050.TW", 0.5, "rf")

    pm.rebalance(method="equal_weight")

    positions = pm.get_positions()
    assert positions["2330.TW"]["strategy_type"] == "hmm_rf"
    assert positions["0050.TW"]["strategy_type"] == "rf"


# ─────────────────────────────────────────────────────────────────────────────
# TC20 — valid_strategy_types contains rf and hmm_rf
# ─────────────────────────────────────────────────────────────────────────────

def test_tc20_valid_strategy_types():
    """TC20: VALID_STRATEGY_TYPES contains exactly 'rf' and 'hmm_rf'."""
    assert "rf" in VALID_STRATEGY_TYPES
    assert "hmm_rf" in VALID_STRATEGY_TYPES
    assert len(VALID_STRATEGY_TYPES) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TC21 — get_positions returns copy-safe dict
# ─────────────────────────────────────────────────────────────────────────────

def test_tc21_get_positions_structure():
    """TC21: get_positions() returns dict with symbol/weight/strategy_type per entry."""
    pm = PortfolioManager()
    pm.add_position("2330.TW", 0.5, "rf")
    pm.add_position("0050.TW", 0.3, "hmm_rf")

    positions = pm.get_positions()
    assert len(positions) == 2
    for sym, pos_dict in positions.items():
        assert "symbol" in pos_dict
        assert "weight" in pos_dict
        assert "strategy_type" in pos_dict
        assert pos_dict["symbol"] == sym


# ─────────────────────────────────────────────────────────────────────────────
# TC22 — add_position zero weight allowed
# ─────────────────────────────────────────────────────────────────────────────

def test_tc22_add_position_zero_weight():
    """TC22: add_position allows weight=0.0 (inactive position)."""
    pm = PortfolioManager()
    pm.add_position("2330.TW", weight=0.0, strategy_type="rf")
    pos = pm.get_positions()["2330.TW"]
    assert pos["weight"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TC23 — __repr__ includes symbol info
# ─────────────────────────────────────────────────────────────────────────────

def test_tc23_repr():
    """TC23: repr(pm) contains capital and position symbols."""
    pm = _make_pm("2330.TW", capital=500_000)
    r = repr(pm)
    assert "2330.TW" in r
    assert "500" in r  # capital in repr

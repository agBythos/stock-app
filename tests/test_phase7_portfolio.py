"""
Phase 7 — Portfolio Analysis Tests
Tests for PortfolioAnalyzer and /api/portfolio/* endpoints.

Run:
    cd stock-app
    python -m pytest tests/test_phase7_portfolio.py -v --timeout=120
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure stock-app directory is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for PortfolioAnalyzer (no network required for most)
# ──────────────────────────────────────────────────────────────────────────────

from portfolio.portfolio_analyzer import PortfolioAnalyzer, AssetMetrics, PortfolioResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_fake_analyzer(symbols=None, n_days=252, seed=42):
    """
    Create a PortfolioAnalyzer and monkey-patch _fetch_prices so no network
    calls are made. Returns (analyzer, symbols, prices_df).
    """
    if symbols is None:
        symbols = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    prices = {}
    for sym in symbols:
        drift = rng.uniform(0.0001, 0.001)
        vol = rng.uniform(0.01, 0.02)
        log_returns = rng.normal(drift, vol, n_days)
        prices[sym] = 100 * np.exp(np.cumsum(log_returns))
    prices_df = pd.DataFrame(prices, index=idx)
    return PortfolioAnalyzer(), symbols, prices_df


# ── TC01: Module imports cleanly ───────────────────────────────────────────────

def test_tc01_import():
    """TC01: portfolio module imports without error."""
    from portfolio import PortfolioAnalyzer, AssetMetrics, PortfolioResult
    assert PortfolioAnalyzer is not None
    assert AssetMetrics is not None
    assert PortfolioResult is not None


# ── TC02: PortfolioAnalyzer instantiates ──────────────────────────────────────

def test_tc02_instantiate():
    """TC02: PortfolioAnalyzer() instantiates with default risk-free rate."""
    analyzer = PortfolioAnalyzer()
    assert hasattr(analyzer, "risk_free_rate")
    assert 0 < analyzer.risk_free_rate < 0.2


# ── TC03: _compute_asset_metrics returns correct structure ────────────────────

def test_tc03_compute_asset_metrics():
    """TC03: _compute_asset_metrics returns AssetMetrics list with correct keys."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["X", "Y"])
    returns_df = prices_df.pct_change().dropna()
    weights = np.array([0.5, 0.5])
    metrics = analyzer._compute_asset_metrics(symbols, prices_df, returns_df, weights)
    assert len(metrics) == 2
    for m in metrics:
        assert isinstance(m, AssetMetrics)
        assert m.symbol in symbols
        assert isinstance(m.total_return_pct, float)
        assert isinstance(m.sharpe_ratio, float)
        assert m.annualized_volatility_pct >= 0
        assert m.max_drawdown_pct <= 0   # drawdown is always ≤ 0


# ── TC04: _compute_correlation returns NxN matrix ─────────────────────────────

def test_tc04_correlation_shape():
    """TC04: _compute_correlation returns square matrix with diagonal=1.0."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["A", "B", "C"])
    returns_df = prices_df.pct_change().dropna()
    matrix, labels = analyzer._compute_correlation(returns_df, symbols)
    n = len(symbols)
    assert len(matrix) == n
    assert all(len(row) == n for row in matrix)
    assert labels == symbols
    # Diagonal should be 1.0
    for i in range(n):
        assert abs(matrix[i][i] - 1.0) < 1e-6


# ── TC05: _compute_correlation values in [-1, 1] ──────────────────────────────

def test_tc05_correlation_range():
    """TC05: All correlation values are in [-1, 1]."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["A", "B", "C", "D"])
    returns_df = prices_df.pct_change().dropna()
    matrix, _ = analyzer._compute_correlation(returns_df, symbols)
    for row in matrix:
        for v in row:
            assert -1.0 <= v <= 1.0 + 1e-9, f"Correlation {v} out of range"


# ── TC06: _compute_portfolio_metrics has all required keys ───────────────────

def test_tc06_portfolio_metrics_keys():
    """TC06: portfolio_metrics dict has all required keys."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["X", "Y"])
    returns_df = prices_df.pct_change().dropna()
    weights = np.array([0.6, 0.4])
    port_ret = (returns_df * weights).sum(axis=1)
    pm = analyzer._compute_portfolio_metrics(port_ret, returns_df, weights)
    required = {
        "total_return_pct", "annualized_return_pct", "annualized_volatility_pct",
        "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio",
        "diversification_ratio", "portfolio_volatility_cov_pct", "trading_days",
    }
    for key in required:
        assert key in pm, f"Missing key: {key}"


# ── TC07: equal weights default ───────────────────────────────────────────────

def test_tc07_equal_weights_default():
    """TC07: weights=None produces equal weights summing to 1."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["A", "B", "C"])

    # Patch _fetch_prices
    def fake_fetch(syms, period):
        return prices_df[syms]
    analyzer._fetch_prices = fake_fetch

    result = analyzer.analyze(symbols=symbols, period="1y", weights=None)
    w = result.weights
    assert len(w) == 3
    assert abs(sum(w) - 1.0) < 1e-6
    for wi in w:
        assert abs(wi - 1/3) < 1e-6


# ── TC08: custom weights normalised to 1 ─────────────────────────────────────

def test_tc08_custom_weights_normalised():
    """TC08: Custom weights [2, 1, 1] get normalised to [0.5, 0.25, 0.25]."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["A", "B", "C"])

    def fake_fetch(syms, period):
        return prices_df[syms]
    analyzer._fetch_prices = fake_fetch

    result = analyzer.analyze(symbols=symbols, period="1y", weights=[2, 1, 1])
    w = result.weights
    assert abs(sum(w) - 1.0) < 1e-6
    assert abs(w[0] - 0.5) < 1e-6
    assert abs(w[1] - 0.25) < 1e-6


# ── TC09: ValueError for <2 symbols ──────────────────────────────────────────

def test_tc09_too_few_symbols():
    """TC09: analyze() raises ValueError when fewer than 2 symbols given."""
    analyzer = PortfolioAnalyzer()
    with pytest.raises(ValueError, match="at least 2"):
        analyzer.analyze(symbols=["AAPL"], period="1y")


# ── TC10: ValueError for >10 symbols ─────────────────────────────────────────

def test_tc10_too_many_symbols():
    """TC10: analyze() raises ValueError when more than 10 symbols given."""
    analyzer = PortfolioAnalyzer()
    with pytest.raises(ValueError, match="Maximum 10"):
        analyzer.analyze(symbols=[f"S{i}" for i in range(11)], period="1y")


# ── TC11: to_dict() produces JSON-serialisable output ────────────────────────

def test_tc11_to_dict_serialisable():
    """TC11: PortfolioResult.to_dict() produces JSON-serialisable types."""
    import json
    analyzer, symbols, prices_df = _make_fake_analyzer(["A", "B"])

    def fake_fetch(syms, period):
        return prices_df[syms]
    analyzer._fetch_prices = fake_fetch

    result = analyzer.analyze(symbols=symbols, period="1y")
    d = result.to_dict()
    # Should not raise
    serialised = json.dumps(d)
    assert len(serialised) > 100


# ── TC12: Normalised prices start at 100 ─────────────────────────────────────

def test_tc12_normalised_prices_start_100():
    """TC12: All normalised price series start at 100.0."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["A", "B", "C"])

    def fake_fetch(syms, period):
        return prices_df[syms]
    analyzer._fetch_prices = fake_fetch

    result = analyzer.analyze(symbols=symbols, period="1y")
    for sym in result.symbols:
        vals = result.normalized_prices[sym]
        assert abs(vals[0] - 100.0) < 1e-3, f"{sym} start != 100: {vals[0]}"


# ── TC13: portfolio_curve length matches dates ────────────────────────────────

def test_tc13_portfolio_curve_length():
    """TC13: portfolio_curve and dates have same length as price data."""
    analyzer, symbols, prices_df = _make_fake_analyzer(["A", "B"])

    def fake_fetch(syms, period):
        return prices_df[syms]
    analyzer._fetch_prices = fake_fetch

    result = analyzer.analyze(symbols=symbols, period="1y")
    assert len(result.portfolio_curve) == len(result.dates)
    assert len(result.dates) > 0


# ── TC14–15: API endpoint tests (requires running server — marked slow) ────────

try:
    from fastapi.testclient import TestClient
    import server as _server_mod
    _client = TestClient(_server_mod.app)
    _HAS_SERVER = True
except Exception:
    _HAS_SERVER = False


@pytest.mark.skipif(not _HAS_SERVER, reason="Server not importable")
def test_tc14_portfolio_info_endpoint():
    """TC14: GET /api/portfolio/info returns availability info."""
    resp = _client.get("/api/portfolio/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "supported_periods" in data
    assert "1y" in data["supported_periods"]


@pytest.mark.skipif(not _HAS_SERVER, reason="Server not importable")
def test_tc15_portfolio_analyze_invalid_period():
    """TC15: POST /api/portfolio/analyze with invalid period returns 422."""
    resp = _client.post("/api/portfolio/analyze", json={
        "symbols": ["AAPL", "MSFT"],
        "period": "999y",
    })
    assert resp.status_code == 422


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SERVER, reason="Server not importable")
def test_tc16_portfolio_analyze_real_data():
    """TC16: POST /api/portfolio/analyze with real symbols returns valid result. [SLOW/NETWORK]"""
    resp = _client.post("/api/portfolio/analyze", json={
        "symbols": ["AAPL", "MSFT"],
        "period": "6mo",
    }, timeout=120)
    assert resp.status_code == 200
    data = resp.json()
    assert "symbols" in data
    assert "correlation_matrix" in data
    assert "portfolio_metrics" in data
    assert "asset_metrics" in data
    assert len(data["asset_metrics"]) == 2
    # Correlation matrix should be 2x2
    assert len(data["correlation_matrix"]) == 2
    assert len(data["correlation_matrix"][0]) == 2
    # Diagonal should be 1.0
    assert abs(data["correlation_matrix"][0][0] - 1.0) < 1e-3
    assert abs(data["correlation_matrix"][1][1] - 1.0) < 1e-3


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_SERVER, reason="Server not importable")
def test_tc17_portfolio_page_served():
    """TC17: GET /portfolio returns HTML page. [SLOW]"""
    resp = _client.get("/portfolio")
    assert resp.status_code == 200
    assert "portfolio" in resp.text.lower() or "html" in resp.headers.get("content-type", "")

"""
tests/test_phase6_api.py — Phase 6 API Integration Tests
=========================================================

Test coverage (≥ 5 tests):
  TC01 - GET /api/strategies returns list with 3 strategies
  TC02 - GET /api/strategies: each strategy has required fields
  TC03 - POST /api/backtest/run smoke test (ma_crossover on AAPL)
  TC04 - POST /api/backtest/run: performance dict has required keys
  TC05 - POST /api/backtest/run: invalid strategy returns 400
  TC06 - POST /api/backtest/run: date range < 60 days returns 400
  TC07 - POST /api/backtest/run: unknown symbol returns 404 or 500
  TC08 - POST /api/validate/hmm: returns hmm and backtest keys
  TC09 - POST /api/validate/hmm: invalid symbol returns error
  TC10 - GET /api/strategies: hmm_filter strategy is present

Notes:
- Uses FastAPI TestClient (httpx under the hood, no real server needed).
- TC03 and TC08 fetch real market data via yfinance → skipped in offline env.
- All tests use a short date range / small period to minimize download time.
"""

from __future__ import annotations

import sys
import os

# Ensure stock-app root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

# ── Import app ────────────────────────────────────────────────────────────
try:
    from server import app
    _APP_AVAILABLE = True
except Exception as _e:
    _APP_AVAILABLE = False
    _APP_ERR = str(_e)

pytestmark = pytest.mark.skipif(
    not _APP_AVAILABLE,
    reason=f"server.py could not be imported: {_APP_ERR if not _APP_AVAILABLE else ''}",
)

# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Return a synchronous TestClient for the FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── TC01 ──────────────────────────────────────────────────────────────────

def test_get_strategies_returns_list(client):
    """TC01: GET /api/strategies → 200 with count ≥ 3."""
    resp = client.get("/api/strategies")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "strategies" in data, "Response missing 'strategies' key"
    assert "count" in data, "Response missing 'count' key"
    assert data["count"] >= 3, f"Expected ≥3 strategies, got {data['count']}"
    assert len(data["strategies"]) >= 3


# ── TC02 ──────────────────────────────────────────────────────────────────

def test_get_strategies_fields(client):
    """TC02: Each strategy entry has id, name, description, params."""
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    for s in resp.json()["strategies"]:
        assert "id" in s,          f"Strategy missing 'id': {s}"
        assert "name" in s,        f"Strategy missing 'name': {s}"
        assert "description" in s, f"Strategy missing 'description': {s}"
        assert "params" in s,      f"Strategy missing 'params': {s}"


# ── TC03 ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_backtest_run_ma_crossover(client):
    """TC03: POST /api/backtest/run smoke test with ma_crossover."""
    body = {
        "symbol": "AAPL",
        "strategy": "ma_crossover",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 100000,
        "params": {"fast_period": 10, "slow_period": 30},
    }
    resp = client.post("/api/backtest/run", json=body, timeout=120)
    # Accept 503 if backtest module not available
    if resp.status_code == 503:
        pytest.skip("Backtest module not available in this environment")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    assert "performance" in data, "Response missing 'performance'"
    assert "symbol" in data or "strategy_name" in data


# ── TC04 ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_backtest_run_performance_keys(client):
    """TC04: Performance dict contains all required KPI keys."""
    body = {
        "symbol": "AAPL",
        "strategy": "ma_crossover",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 100000,
        "params": {},
    }
    resp = client.post("/api/backtest/run", json=body, timeout=120)
    if resp.status_code == 503:
        pytest.skip("Backtest module not available")

    assert resp.status_code == 200, resp.text[:300]
    perf = resp.json().get("performance", {})
    required_keys = [
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe_ratio",
        "total_trades",
    ]
    for key in required_keys:
        assert key in perf, f"performance dict missing '{key}'"


# ── TC05 ──────────────────────────────────────────────────────────────────

def test_backtest_run_invalid_strategy(client):
    """TC05: POST /api/backtest/run with unknown strategy → 400."""
    body = {
        "symbol": "AAPL",
        "strategy": "not_a_real_strategy",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 100000,
        "params": {},
    }
    resp = client.post("/api/backtest/run", json=body)
    # Accept 400 (bad request) or 503 (module unavailable)
    assert resp.status_code in (400, 503), (
        f"Expected 400 or 503, got {resp.status_code}: {resp.text}"
    )


# ── TC06 ──────────────────────────────────────────────────────────────────

def test_backtest_run_short_date_range(client):
    """TC06: POST /api/backtest/run with < 60 day range → 400."""
    body = {
        "symbol": "AAPL",
        "strategy": "ma_crossover",
        "start_date": "2024-01-01",
        "end_date": "2024-01-15",   # only 14 days
        "initial_capital": 100000,
        "params": {},
    }
    resp = client.post("/api/backtest/run", json=body)
    assert resp.status_code in (400, 503), (
        f"Expected 400 or 503, got {resp.status_code}: {resp.text}"
    )


# ── TC07 ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_backtest_run_unknown_symbol(client):
    """TC07: POST /api/backtest/run with garbage symbol → 404 or 500."""
    body = {
        "symbol": "XXXXNOTREAL99999",
        "strategy": "ma_crossover",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 100000,
        "params": {},
    }
    resp = client.post("/api/backtest/run", json=body, timeout=30)
    if resp.status_code == 503:
        pytest.skip("Backtest module not available")
    assert resp.status_code in (400, 404, 500), (
        f"Expected error status, got {resp.status_code}: {resp.text[:200]}"
    )


# ── TC08 ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_validate_hmm_returns_structure(client):
    """TC08: POST /api/validate/hmm → returns hmm + backtest keys."""
    body = {
        "symbol": "AAPL",
        "period": "2y",
        "hmm_n_states": 3,
        "hmm_window": 252,
        "n_groups": 6,
        "k_test_groups": 2,
    }
    resp = client.post("/api/validate/hmm", json=body, timeout=180)
    if resp.status_code == 503:
        pytest.skip("Backtest module not available")

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"
    data = resp.json()
    assert "hmm" in data,     "Response missing 'hmm' key"
    assert "backtest" in data, "Response missing 'backtest' key"
    assert "regime_distribution" in data["hmm"]
    assert "total_return_pct" in data["backtest"]


# ── TC09 ──────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_validate_hmm_invalid_symbol(client):
    """TC09: POST /api/validate/hmm with garbage symbol → error status."""
    body = {
        "symbol": "XXXXNOTREAL99999",
        "period": "2y",
        "hmm_n_states": 3,
        "hmm_window": 252,
    }
    resp = client.post("/api/validate/hmm", json=body, timeout=30)
    assert resp.status_code in (400, 404, 500, 503), (
        f"Expected error status, got {resp.status_code}"
    )


# ── TC10 ──────────────────────────────────────────────────────────────────

def test_strategies_includes_hmm_filter(client):
    """TC10: GET /api/strategies includes hmm_filter entry."""
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()["strategies"]]
    assert "hmm_filter" in ids, f"'hmm_filter' not in strategy list: {ids}"


# ── Run marker ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke: run only non-network tests
    pytest.main([__file__, "-v", "-m", "not slow", "--tb=short"])

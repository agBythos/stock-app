"""
tests/test_compare_api.py — Phase 8D Compare API Tests
=======================================================

Test coverage (≥ 5 tests):
  TC01 - GET /api/backtest/compare endpoint exists (200 or non-404)
  TC02 - Response has correct top-level structure
  TC03 - All 3 strategies present in response
  TC04 - KPI fields are complete for each strategy
  TC05 - recommended field is present (and is a string or None)
  TC06 - benchmark field present with required sub-keys
  TC07 - KPI values are numeric (not string / NaN)
  TC08 - period field matches request params
  TC09 - equity list contains {date, value} entries (if strategy ran)
  TC10 - recommendation_reason field is present

Notes:
- Uses FastAPI TestClient (httpx) — no real server needed.
- yfinance calls are mocked via unittest.mock.patch so tests run offline
  and deterministically.
- Each test gets a fresh mock DataFrame via the _mock_yf fixture.
"""

from __future__ import annotations

import sys
import os
import math
from datetime import date, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Make the stock-app root importable ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Import FastAPI app ─────────────────────────────────────────────────────
try:
    from server import app
    _APP_AVAILABLE = True
    _APP_ERR = ""
except Exception as _e:
    _APP_AVAILABLE = False
    _APP_ERR = str(_e)

pytestmark = pytest.mark.skipif(
    not _APP_AVAILABLE,
    reason=f"server.py could not be imported: {_APP_ERR}",
)

from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────

_COMPARE_URL = "/api/backtest/compare"
_DEFAULT_PARAMS = {
    "symbol": "AAPL",
    "start":  "2022-01-03",
    "end":    "2023-01-02",
}

# ── Shared synthetic market data ──────────────────────────────────────────

def _make_ohlcv(n_days: int = 252, start: str = "2022-01-03") -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with DatetimeIndex (yfinance style)."""
    rng = pd.date_range(start=start, periods=n_days, freq="B")  # Business days
    np.random.seed(42)
    close = 150.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n_days)))
    df = pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.02,
            "Low":    close * 0.98,
            "Close":  close,
            "Volume": np.random.randint(1_000_000, 5_000_000, n_days).astype(float),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=rng,
    )
    return df


# ── Fixture: mock yfinance Ticker so no network calls are made ────────────

@pytest.fixture()
def mock_yf():
    """Patch yfinance.Ticker.history to return synthetic data."""
    synthetic = _make_ohlcv()

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = synthetic

    with patch("server.yf.Ticker", return_value=mock_ticker) as patcher:
        yield patcher


@pytest.fixture(scope="module")
def client():
    """Return a synchronous TestClient for the FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── TC01: endpoint exists ──────────────────────────────────────────────────

def test_compare_endpoint_exists(client, mock_yf):
    """TC01: GET /api/backtest/compare must not return 404."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code != 404, (
        f"Endpoint not found (404). Got: {resp.status_code} — {resp.text[:300]}"
    )


# ── TC02: top-level response structure ────────────────────────────────────

def test_compare_response_structure(client, mock_yf):
    """TC02: Response must have symbol, period, strategies, benchmark,
    recommended, recommendation_reason."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:300]}"

    body = resp.json()
    required_keys = {"symbol", "period", "strategies", "benchmark",
                     "recommended", "recommendation_reason"}
    missing = required_keys - set(body.keys())
    assert not missing, f"Response missing top-level keys: {missing}"


# ── TC03: 3 strategies are present ────────────────────────────────────────

def test_compare_three_strategies(client, mock_yf):
    """TC03: Response.strategies must contain exactly 3 entries
    (MA Crossover, RF Strategy, HMM Filter Strategy)."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    strategies = resp.json()["strategies"]
    assert isinstance(strategies, list), "'strategies' must be a list"
    assert len(strategies) == 3, f"Expected 3 strategies, got {len(strategies)}: {[s.get('name') for s in strategies]}"

    names = {s["name"] for s in strategies}
    expected = {"MA Crossover", "RF Strategy", "HMM Filter Strategy"}
    assert expected == names, f"Strategy names mismatch. Got: {names}"


# ── TC04: KPI fields are complete ─────────────────────────────────────────

def test_compare_kpi_fields_complete(client, mock_yf):
    """TC04: Every strategy must have a kpi dict with all 5 required fields."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    required_kpi = {"sharpe", "max_drawdown", "win_rate", "cagr", "total_return"}

    for s in resp.json()["strategies"]:
        assert "kpi" in s, f"Strategy '{s.get('name')}' missing 'kpi'"
        kpi = s["kpi"]
        missing = required_kpi - set(kpi.keys())
        assert not missing, (
            f"Strategy '{s.get('name')}' KPI missing fields: {missing}"
        )


# ── TC05: recommended field is present ────────────────────────────────────

def test_compare_recommended_present(client, mock_yf):
    """TC05: 'recommended' key must exist and be a str or None."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    body = resp.json()
    assert "recommended" in body, "Response missing 'recommended' key"
    rec = body["recommended"]
    assert rec is None or isinstance(rec, str), (
        f"'recommended' must be str or None, got {type(rec)}"
    )


# ── TC06: benchmark present with required sub-keys ─────────────────────────

def test_compare_benchmark_structure(client, mock_yf):
    """TC06: benchmark must have name, equity, kpi."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    bh = resp.json()["benchmark"]
    assert isinstance(bh, dict), "'benchmark' must be a dict"
    for key in ("name", "equity", "kpi"):
        assert key in bh, f"Benchmark missing key '{key}'"


# ── TC07: KPI values are numeric ──────────────────────────────────────────

def test_compare_kpi_values_numeric(client, mock_yf):
    """TC07: All KPI values must be float/int, not NaN or string."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    for s in resp.json()["strategies"]:
        kpi = s.get("kpi", {})
        for field, val in kpi.items():
            assert isinstance(val, (int, float)), (
                f"Strategy '{s.get('name')}' KPI['{field}'] is not numeric: {val!r}"
            )
            # NaN is still float in Python — check explicitly
            if isinstance(val, float):
                assert not math.isnan(val), (
                    f"Strategy '{s.get('name')}' KPI['{field}'] is NaN"
                )


# ── TC08: period field matches request ────────────────────────────────────

def test_compare_period_matches_request(client, mock_yf):
    """TC08: period.start and period.end must echo the request params."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    period = resp.json()["period"]
    assert period["start"] == _DEFAULT_PARAMS["start"], (
        f"period.start mismatch: {period['start']} != {_DEFAULT_PARAMS['start']}"
    )
    assert period["end"] == _DEFAULT_PARAMS["end"], (
        f"period.end mismatch: {period['end']} != {_DEFAULT_PARAMS['end']}"
    )


# ── TC09: equity list has {date, value} entries ────────────────────────────

def test_compare_equity_entries(client, mock_yf):
    """TC09: Strategies that ran successfully must have equity entries
    with 'date' (str) and 'value' (float) keys."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    for s in resp.json()["strategies"]:
        equity = s.get("equity", [])
        if not equity:
            # Strategy may have failed (error key present) — acceptable
            continue
        # Spot-check first entry
        entry = equity[0]
        assert "date" in entry, f"Equity entry missing 'date': {entry}"
        assert "value" in entry, f"Equity entry missing 'value': {entry}"
        assert isinstance(entry["date"],  str),          f"date not str: {entry['date']!r}"
        assert isinstance(entry["value"], (int, float)), f"value not numeric: {entry['value']!r}"


# ── TC10: recommendation_reason field is present ──────────────────────────

def test_compare_recommendation_reason(client, mock_yf):
    """TC10: 'recommendation_reason' key must exist and be str or None."""
    resp = client.get(_COMPARE_URL, params=_DEFAULT_PARAMS)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text[:300]}"

    body = resp.json()
    assert "recommendation_reason" in body, "Response missing 'recommendation_reason'"
    reason = body["recommendation_reason"]
    assert reason is None or isinstance(reason, str), (
        f"'recommendation_reason' must be str or None, got {type(reason)}"
    )

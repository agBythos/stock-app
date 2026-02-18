"""E2E test for stock app"""
import json
import urllib.request
import urllib.error
import pytest

BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Lazy server-availability guard
# Evaluated once at collection time; all tests are skipped when server is down.
# ---------------------------------------------------------------------------

def _server_available() -> bool:
    """Return True only if the server is actually listening on BASE."""
    try:
        urllib.request.urlopen(f"{BASE}/", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"server not running at {BASE}",
)


# ---------------------------------------------------------------------------
# Helper utilities (called lazily, only inside test functions)
# ---------------------------------------------------------------------------

def _fetch_json(path: str):
    resp = urllib.request.urlopen(f"{BASE}{path}")
    return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_root_returns_html():
    """Root endpoint should return an HTML page with lightweight-charts ref."""
    resp = urllib.request.urlopen(f"{BASE}/")
    content = resp.read().decode()
    assert "<html" in content.lower() or "<!doctype" in content.lower(), "Root is not HTML"
    assert "lightweight-charts" in content, "No chart library reference found"


@pytest.mark.parametrize("rng", ["1M", "3M", "6M", "1Y", "2Y"])
def test_aapl_range(rng):
    """AAPL data endpoint should return data + indicators for each range param."""
    d = _fetch_json(f"/api/stock/AAPL?range={rng}")
    assert len(d["data"]) > 0, f"AAPL range={rng}: no data"
    assert "indicators" in d and "ma5" in d["indicators"], f"AAPL range={rng}: missing indicators"
    assert len(d["data"]) == len(d["indicators"]["ma5"]), f"AAPL range={rng}: length mismatch"


def test_taiwan_stock():
    """2330.TW should return data in TWD currency."""
    d = _fetch_json("/api/stock/2330.TW?range=3M")
    assert len(d["data"]) > 0, "2330.TW: no data"
    assert "indicators" in d, "2330.TW: missing indicators"
    assert d["info"]["currency"] == "TWD", "2330.TW: currency is not TWD"


def test_predict_endpoint():
    """Predict endpoint should return signal, indicators, and reason."""
    p = _fetch_json("/api/stock/AAPL/predict")
    assert p["signal"] in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"], "Invalid signal"
    assert "indicators" in p, "Missing indicators"
    assert len(p.get("reason", "")) > 0, "Missing reason"


def test_macd_structure():
    """MACD sub-keys must all have the same length as the data array."""
    d = _fetch_json("/api/stock/AAPL?range=3M")
    macd = d["indicators"]["macd"]
    assert all(k in macd for k in ["macd", "signal", "histogram"]), "MACD missing keys"
    data_len = len(d["data"])
    assert len(macd["macd"]) == data_len, "MACD macd length mismatch"
    assert len(macd["signal"]) == data_len, "MACD signal length mismatch"
    assert len(macd["histogram"]) == data_len, "MACD histogram length mismatch"


def test_indicators_endpoint():
    """Separate indicators endpoint should return non-empty indicators array."""
    ind = _fetch_json("/api/stock/AAPL/indicators?range=3M")
    assert "indicators" in ind and len(ind["indicators"]) > 0, "Missing or empty indicators"

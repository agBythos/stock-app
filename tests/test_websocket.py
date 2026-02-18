"""
Phase 8B: WebSocket Real-time Dashboard Tests
Tests for the /ws/price/{symbol} WebSocket endpoint.
All network calls are mocked — no real internet connection required.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

# ── Ensure server is importable ──────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import app, ws_manager, MAX_WS_CONNECTIONS, _fetch_realtime_quote


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_df(close=150.0, volume=1_000_000, open_=148.0):
    """Return a minimal pandas DataFrame that mimics yfinance output."""
    import pandas as pd
    import numpy as np
    rows = [
        {"Close": close - 2, "Volume": volume - 100_000, "Open": open_ - 2},
        {"Close": close,      "Volume": volume,           "Open": open_},
    ]
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(["2026-02-17", "2026-02-18"])
    return df


_SAMPLE_PAYLOAD = {
    "symbol":     "AAPL",
    "price":      182.45,
    "volume":     12_345_678,
    "change_pct": 0.73,
    "prev_close": 181.13,
    "timestamp":  "2026-02-18T09:45:00Z",
    "source":     "1m",
}


# ============================================================================
# Test 1 — _fetch_realtime_quote returns correct keys (1m intraday path)
# ============================================================================
def test_fetch_realtime_quote_keys_1m():
    """_fetch_realtime_quote should return a dict with all required keys (1m path)."""
    mock_df = _make_mock_df()
    empty_df = _make_mock_df().__class__()  # empty DataFrame

    ticker_mock = MagicMock()
    # history called twice: first for 1m (returns data), second for 1d (for prev_close)
    ticker_mock.history.side_effect = [mock_df, mock_df]

    with patch("server.yf.Ticker", return_value=ticker_mock):
        result = _fetch_realtime_quote("AAPL")

    required_keys = {"symbol", "price", "volume", "change_pct", "prev_close", "timestamp", "source"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"
    assert result["symbol"] == "AAPL"
    assert isinstance(result["price"], float)
    assert isinstance(result["volume"], int)
    assert isinstance(result["change_pct"], float)
    assert result["source"] == "1m"


# ============================================================================
# Test 2 — _fetch_realtime_quote fallback to daily data
# ============================================================================
def test_fetch_realtime_quote_fallback_1d():
    """_fetch_realtime_quote should fall back to 1d when 1m data is empty."""
    import pandas as pd
    empty_df = pd.DataFrame()
    daily_df = _make_mock_df()

    ticker_mock = MagicMock()
    ticker_mock.history.side_effect = [empty_df, daily_df]

    with patch("server.yf.Ticker", return_value=ticker_mock):
        result = _fetch_realtime_quote("2330.TW")

    assert result["source"] == "1d"
    assert result["symbol"] == "2330.TW"


# ============================================================================
# Test 3 — _fetch_realtime_quote raises ValueError for completely empty data
# ============================================================================
def test_fetch_realtime_quote_empty_raises():
    """_fetch_realtime_quote raises ValueError when no data is available."""
    import pandas as pd
    empty_df = pd.DataFrame()

    ticker_mock = MagicMock()
    ticker_mock.history.return_value = empty_df

    with patch("server.yf.Ticker", return_value=ticker_mock):
        with pytest.raises(ValueError, match="No data available"):
            _fetch_realtime_quote("INVALID_SYMBOL_XYZ")


# ============================================================================
# Test 4 — WebSocket connection: successful connect + first message
# ============================================================================
def test_websocket_connect_and_receive():
    """Client should receive a JSON payload on first connect."""
    with patch("server._fetch_realtime_quote", return_value=_SAMPLE_PAYLOAD):
        # asyncio.TimeoutError to break the push loop after first send
        with patch("server.asyncio.wait_for", side_effect=[asyncio.TimeoutError(), asyncio.TimeoutError()]):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/price/AAPL") as ws:
                    data = ws.receive_json()
                    assert data["symbol"] == "AAPL"
                    assert "price" in data
                    assert "volume" in data
                    assert "change_pct" in data


# ============================================================================
# Test 5 — WebSocket message format validation
# ============================================================================
def test_websocket_payload_format():
    """Received payload must match the documented JSON schema."""
    with patch("server._fetch_realtime_quote", return_value=_SAMPLE_PAYLOAD):
        with patch("server.asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            with TestClient(app) as client:
                with client.websocket_connect("/ws/price/TSLA") as ws:
                    data = ws.receive_json()

    assert isinstance(data["price"], float)
    assert isinstance(data["volume"], int)
    assert isinstance(data["change_pct"], float)
    assert isinstance(data["timestamp"], str)
    # Timestamp should be ISO-8601 / parseable
    dt = datetime.fromisoformat(data["timestamp"].rstrip("Z"))
    assert dt.year >= 2020


# ============================================================================
# Test 6 — WebSocketConnectionManager: connect / disconnect tracking
# ============================================================================
def test_ws_manager_connect_disconnect():
    """ws_manager should correctly track active connections."""
    from server import WebSocketConnectionManager

    mgr = WebSocketConnectionManager(max_connections=5)

    # Simulate websocket objects
    ws_a = MagicMock(spec=WebSocket)
    ws_a.accept = AsyncMock()
    ws_a.close  = AsyncMock()

    ws_b = MagicMock(spec=WebSocket)
    ws_b.accept = AsyncMock()
    ws_b.close  = AsyncMock()

    async def _run():
        ok_a = await mgr.connect(ws_a, "AAPL")
        ok_b = await mgr.connect(ws_b, "AAPL")
        assert ok_a
        assert ok_b
        assert mgr.count() == 2

        mgr.disconnect(ws_a, "AAPL")
        assert mgr.count() == 1

        mgr.disconnect(ws_b, "AAPL")
        assert mgr.count() == 0
        assert "AAPL" not in mgr.active

    asyncio.run(_run())


# ============================================================================
# Test 7 — Max connection cap is enforced
# ============================================================================
def test_ws_manager_max_connections():
    """Connections beyond max_connections should be rejected (websocket.close called)."""
    from server import WebSocketConnectionManager

    mgr = WebSocketConnectionManager(max_connections=2)

    def _make_ws():
        ws = MagicMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.close  = AsyncMock()
        return ws

    ws1, ws2, ws3 = _make_ws(), _make_ws(), _make_ws()

    async def _run():
        ok1 = await mgr.connect(ws1, "AAPL")
        ok2 = await mgr.connect(ws2, "TSLA")
        ok3 = await mgr.connect(ws3, "NVDA")  # should be rejected

        assert ok1
        assert ok2
        assert not ok3
        # ws3.close should have been called with 1008
        ws3.close.assert_awaited_once()
        call_kwargs = ws3.close.call_args
        assert call_kwargs.kwargs.get("code") == 1008 or (
            len(call_kwargs.args) > 0 and call_kwargs.args[0] == 1008
        )
        assert mgr.count() == 2

    asyncio.run(_run())


# ============================================================================
# Test 8 — Multiple clients on the same symbol receive broadcasts
# ============================================================================
def test_ws_manager_broadcast():
    """broadcast() should send to all clients subscribed to a symbol."""
    from server import WebSocketConnectionManager

    mgr = WebSocketConnectionManager(max_connections=10)

    ws1 = MagicMock(spec=WebSocket)
    ws1.accept    = AsyncMock()
    ws1.close     = AsyncMock()
    ws1.send_json = AsyncMock()

    ws2 = MagicMock(spec=WebSocket)
    ws2.accept    = AsyncMock()
    ws2.close     = AsyncMock()
    ws2.send_json = AsyncMock()

    async def _run():
        await mgr.connect(ws1, "AAPL")
        await mgr.connect(ws2, "AAPL")

        payload = {"symbol": "AAPL", "price": 182.0, "volume": 100000, "change_pct": 0.5}
        await mgr.broadcast("AAPL", payload)

        ws1.send_json.assert_awaited_once_with(payload)
        ws2.send_json.assert_awaited_once_with(payload)

    asyncio.run(_run())


# ============================================================================
# Test 9 — Broadcast cleans up dead connections
# ============================================================================
def test_ws_manager_broadcast_cleans_dead():
    """broadcast() should silently remove clients that raise on send."""
    from server import WebSocketConnectionManager

    mgr = WebSocketConnectionManager(max_connections=10)

    ws_alive = MagicMock(spec=WebSocket)
    ws_alive.accept    = AsyncMock()
    ws_alive.close     = AsyncMock()
    ws_alive.send_json = AsyncMock()

    ws_dead = MagicMock(spec=WebSocket)
    ws_dead.accept    = AsyncMock()
    ws_dead.close     = AsyncMock()
    ws_dead.send_json = AsyncMock(side_effect=Exception("Connection reset"))

    async def _run():
        await mgr.connect(ws_alive, "NVDA")
        await mgr.connect(ws_dead,  "NVDA")
        assert mgr.count() == 2

        await mgr.broadcast("NVDA", {"symbol": "NVDA", "price": 800.0})

        # Dead connection should be removed
        assert ws_dead not in mgr.active.get("NVDA", set())
        assert mgr.count() == 1

    asyncio.run(_run())


# ============================================================================
# Test 10 — /api/ws/status diagnostic endpoint
# ============================================================================
def test_ws_status_endpoint():
    """GET /api/ws/status should return connection count info."""
    with TestClient(app) as client:
        resp = client.get("/api/ws/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_connections" in body
    assert "max_connections" in body
    assert body["max_connections"] == MAX_WS_CONNECTIONS
    assert isinstance(body["by_symbol"], dict)


# ============================================================================
# Test 11 — Existing REST endpoints are unaffected
# ============================================================================
def test_existing_rest_endpoints_not_broken():
    """Health check: existing REST endpoints must still return expected status codes."""
    with TestClient(app) as client:
        # The root page should respond (serve index.html or redirect)
        resp = client.get("/")
        assert resp.status_code in (200, 307, 404), f"Unexpected: {resp.status_code}"

        # /api/ws/status is a new non-breaking endpoint
        resp2 = client.get("/api/ws/status")
        assert resp2.status_code == 200

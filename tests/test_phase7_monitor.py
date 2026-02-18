"""
test_phase7_monitor.py — Phase 7 Step 3-4 測試
================================================

測試項目：
  1. Regime 切換偵測（Bull→Bear, Bear→Bull, 無變化）
  2. State.json 讀寫（首次執行、正常讀寫、JSON 損壞重建）
  3. API endpoint /api/monitor/check 回傳格式

全程使用 mock，不真實發送 Discord Webhook，不下載股票數據。

作者：Bythos（sub-agent phase7-step34）
建立：2026-02-18
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 確保 stock-app 根目錄在 sys.path
# ---------------------------------------------------------------------------

STOCK_APP_ROOT = Path(__file__).resolve().parent.parent
if str(STOCK_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCK_APP_ROOT))

from alerts.regime_monitor import (
    _load_state,
    _save_state,
    check_regime_change,
    get_all_regime_status,
)


# ===========================================================================
# 工具函式
# ===========================================================================


def _make_state_file(tmp_path: Path, data: dict) -> Path:
    """在 tmp 目錄建立 state.json，回傳路徑。"""
    p = tmp_path / "state.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ===========================================================================
# 1. _load_state / _save_state
# ===========================================================================


class TestLoadState:
    """state.json 讀取邏輯"""

    def test_no_file_returns_empty_dict(self, tmp_path):
        """不存在的路徑應回傳空 dict"""
        p = tmp_path / "nonexistent.json"
        assert _load_state(p) == {}

    def test_valid_json_returns_data(self, tmp_path):
        """正常 JSON 應正確讀取"""
        data = {"2330.TW": {"regime": "Bull", "since": "2026-02-18", "last_alert": None}}
        p = _make_state_file(tmp_path, data)
        result = _load_state(p)
        assert result["2330.TW"]["regime"] == "Bull"

    def test_corrupted_json_returns_empty_dict(self, tmp_path):
        """JSON 損壞應回傳空 dict，不 crash"""
        p = tmp_path / "state.json"
        p.write_text("{broken json{{{{", encoding="utf-8")
        result = _load_state(p)
        assert result == {}

    def test_non_dict_json_returns_empty_dict(self, tmp_path):
        """非 dict 的 JSON（如 list）應回傳空 dict"""
        p = tmp_path / "state.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        result = _load_state(p)
        assert result == {}


class TestSaveState:
    """state.json 寫入邏輯"""

    def test_save_creates_file(self, tmp_path):
        """save 後檔案應存在"""
        p = tmp_path / "state.json"
        data = {"2330.TW": {"regime": "Bear", "since": "2026-02-18", "last_alert": "2026-02-18"}}
        _save_state(data, p)
        assert p.exists()

    def test_save_roundtrip(self, tmp_path):
        """save 後 load 應取回相同數據"""
        p = tmp_path / "state.json"
        data = {"0050.TW": {"regime": "Sideways", "since": "2026-01-01", "last_alert": None}}
        _save_state(data, p)
        loaded = _load_state(p)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        """save 應自動建立父目錄"""
        p = tmp_path / "nested" / "deep" / "state.json"
        _save_state({"x": {}}, p)
        assert p.exists()


# ===========================================================================
# 2. check_regime_change
# ===========================================================================


class TestCheckRegimeChange:
    """Regime 切換偵測邏輯"""

    def _mock_send(self):
        """回傳 patch target 字串"""
        return "alerts.regime_monitor.send_regime_change_alert"

    # ── 首次執行 ──────────────────────────────────────────────────────────

    def test_first_run_no_alert(self, tmp_path):
        """首次執行（symbol 不在 state）：建立初始狀態，不發 alert"""
        p = tmp_path / "state.json"
        with patch(self._mock_send()) as mock_send:
            result = check_regime_change("2330.TW", "Bull", 850.0, state_path=p)
        assert result is False
        mock_send.assert_not_called()
        # 狀態應被寫入
        state = _load_state(p)
        assert state["2330.TW"]["regime"] == "Bull"

    def test_first_run_initializes_state(self, tmp_path):
        """首次執行後，state.json 應包含正確 since 日期"""
        from datetime import date
        p = tmp_path / "state.json"
        with patch(self._mock_send()):
            check_regime_change("2317.TW", "Bear", 100.0, state_path=p)
        state = _load_state(p)
        assert state["2317.TW"]["since"] == date.today().isoformat()

    # ── 無切換 ────────────────────────────────────────────────────────────

    def test_no_change_no_alert(self, tmp_path):
        """regime 未變化 → 不發 alert，回傳 False"""
        data = {"2330.TW": {"regime": "Bull", "since": "2026-02-01", "last_alert": None}}
        p = _make_state_file(tmp_path, data)
        with patch(self._mock_send()) as mock_send:
            result = check_regime_change("2330.TW", "Bull", 860.0, state_path=p)
        assert result is False
        mock_send.assert_not_called()

    # ── Bull → Bear ───────────────────────────────────────────────────────

    def test_bull_to_bear_triggers_alert(self, tmp_path):
        """Bull → Bear 應發送 alert，回傳 True"""
        data = {"2330.TW": {"regime": "Bull", "since": "2026-02-01", "last_alert": None}}
        p = _make_state_file(tmp_path, data)
        with patch(self._mock_send()) as mock_send:
            mock_send.return_value = True
            result = check_regime_change("2330.TW", "Bear", 820.0, state_path=p)
        assert result is True
        mock_send.assert_called_once_with(
            symbol="2330.TW",
            old_regime="Bull",
            new_regime="Bear",
            price=820.0,
            webhook_url=None,
        )

    def test_bull_to_bear_updates_state(self, tmp_path):
        """Bull → Bear 後，state 應更新為 Bear"""
        data = {"2330.TW": {"regime": "Bull", "since": "2026-02-01", "last_alert": None}}
        p = _make_state_file(tmp_path, data)
        with patch(self._mock_send()):
            check_regime_change("2330.TW", "Bear", 820.0, state_path=p)
        state = _load_state(p)
        assert state["2330.TW"]["regime"] == "Bear"

    # ── Bear → Bull ───────────────────────────────────────────────────────

    def test_bear_to_bull_triggers_alert(self, tmp_path):
        """Bear → Bull 應發送 alert，回傳 True"""
        data = {"0050.TW": {"regime": "Bear", "since": "2026-01-15", "last_alert": "2026-01-15"}}
        p = _make_state_file(tmp_path, data)
        with patch(self._mock_send()) as mock_send:
            mock_send.return_value = True
            result = check_regime_change("0050.TW", "Bull", 180.0, state_path=p)
        assert result is True
        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs["old_regime"] == "Bear"
        assert kwargs["new_regime"] == "Bull"

    def test_bear_to_bull_updates_state(self, tmp_path):
        """Bear → Bull 後，state 應更新為 Bull"""
        data = {"0050.TW": {"regime": "Bear", "since": "2026-01-15", "last_alert": "2026-01-15"}}
        p = _make_state_file(tmp_path, data)
        with patch(self._mock_send()):
            check_regime_change("0050.TW", "Bull", 180.0, state_path=p)
        state = _load_state(p)
        assert state["0050.TW"]["regime"] == "Bull"

    # ── 損壞 state 處理 ───────────────────────────────────────────────────

    def test_corrupted_state_treated_as_first_run(self, tmp_path):
        """損壞的 state.json → 重建，不 crash，不發 alert"""
        p = tmp_path / "state.json"
        p.write_text("{{{not_valid_json", encoding="utf-8")
        with patch(self._mock_send()) as mock_send:
            result = check_regime_change("2330.TW", "Sideways", 830.0, state_path=p)
        assert result is False
        mock_send.assert_not_called()

    # ── 多 symbol 互不干擾 ────────────────────────────────────────────────

    def test_multiple_symbols_independent(self, tmp_path):
        """不同 symbol 的 state 互不影響"""
        data = {
            "2330.TW": {"regime": "Bull", "since": "2026-02-01", "last_alert": None},
            "2317.TW": {"regime": "Bear", "since": "2026-02-01", "last_alert": "2026-02-01"},
        }
        p = _make_state_file(tmp_path, data)
        with patch(self._mock_send()) as mock_send:
            # 2330.TW 無切換
            r1 = check_regime_change("2330.TW", "Bull", 850.0, state_path=p)
            # 2317.TW 切換
            r2 = check_regime_change("2317.TW", "Bull", 110.0, state_path=p)
        assert r1 is False
        assert r2 is True
        # 確認只有一次 alert
        assert mock_send.call_count == 1


# ===========================================================================
# 3. get_all_regime_status
# ===========================================================================


class TestGetAllRegimeStatus:
    """get_all_regime_status 工具函式"""

    def test_returns_empty_for_no_state(self, tmp_path):
        p = tmp_path / "state.json"
        assert get_all_regime_status(state_path=p) == {}

    def test_returns_all_symbols(self, tmp_path):
        data = {
            "2330.TW": {"regime": "Bull", "since": "2026-02-18", "last_alert": None},
            "0050.TW": {"regime": "Bear", "since": "2026-02-18", "last_alert": "2026-02-18"},
        }
        p = _make_state_file(tmp_path, data)
        result = get_all_regime_status(state_path=p)
        assert set(result.keys()) == {"2330.TW", "0050.TW"}


# ===========================================================================
# 4. API Endpoint /api/monitor/check
# ===========================================================================


class TestMonitorCheckEndpoint:
    """
    /api/monitor/check GET endpoint 測試

    使用 TestClient + patch，不真實呼叫 yfinance / HMM / Discord。
    """

    @pytest.fixture
    def client(self):
        """建立 FastAPI TestClient"""
        from fastapi.testclient import TestClient
        from server import app
        return TestClient(app, raise_server_exceptions=False)

    def _make_fake_df(self, n=100):
        """建立假的 yfinance DataFrame"""
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2025-06-01", periods=n, freq="B")
        close = 850 + np.random.randn(n).cumsum()
        volume = np.ones(n) * 1_000_000
        return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)

    def test_endpoint_returns_200(self, client, tmp_path):
        """endpoint 應回傳 200"""
        fake_df = self._make_fake_df()

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_df

        mock_hmm = MagicMock()
        mock_hmm.predict.return_value = [0] * len(fake_df)  # 全 Bull

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("hmm.market_hmm.MarketHMM", return_value=mock_hmm), \
             patch("alerts.regime_monitor.check_regime_change", return_value=False), \
             patch("alerts.regime_monitor.get_all_regime_status", return_value={}):
            resp = client.get("/api/monitor/check")

        assert resp.status_code == 200

    def test_endpoint_response_schema(self, client, tmp_path):
        """回應必須包含 checked / skipped / alerts_sent / regime_status"""
        fake_df = self._make_fake_df()

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_df

        mock_hmm = MagicMock()
        mock_hmm.predict.return_value = [0] * len(fake_df)

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("hmm.market_hmm.MarketHMM", return_value=mock_hmm), \
             patch("alerts.regime_monitor.check_regime_change", return_value=False), \
             patch("alerts.regime_monitor.get_all_regime_status", return_value={}):
            resp = client.get("/api/monitor/check")

        data = resp.json()
        assert "checked" in data
        assert "skipped" in data
        assert "alerts_sent" in data
        assert "regime_status" in data

    def test_endpoint_graceful_skip_on_empty_data(self, client):
        """數據為空時，symbol 應出現在 skipped，不 crash"""
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # 空數據

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("alerts.regime_monitor.get_all_regime_status", return_value={}):
            resp = client.get("/api/monitor/check")

        assert resp.status_code == 200
        data = resp.json()
        # 全部 symbol 應被 skip
        assert data["checked"] == []
        assert len(data["skipped"]) == 3  # 3 個 monitor symbols

    def test_endpoint_alerts_sent_count(self, client):
        """alerts_sent 應正確計數"""
        fake_df = self._make_fake_df()

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_df

        mock_hmm = MagicMock()
        mock_hmm.predict.return_value = [2] * len(fake_df)  # Bear

        # 模擬 2 個 symbol 發生切換
        call_results = [True, True, False]
        call_iter = iter(call_results)

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("hmm.market_hmm.MarketHMM", return_value=mock_hmm), \
             patch("alerts.regime_monitor.check_regime_change", side_effect=call_results), \
             patch("alerts.regime_monitor.get_all_regime_status", return_value={}):
            resp = client.get("/api/monitor/check")

        data = resp.json()
        assert data["alerts_sent"] == 2

    def test_endpoint_regime_status_in_response(self, client):
        """regime_status 應包含 get_all_regime_status 的回傳值"""
        fake_df = self._make_fake_df()
        mock_status = {
            "2330.TW": {"regime": "Bear", "since": "2026-02-18", "last_alert": "2026-02-18"}
        }

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = fake_df

        mock_hmm = MagicMock()
        mock_hmm.predict.return_value = [2] * len(fake_df)

        with patch("yfinance.Ticker", return_value=mock_ticker), \
             patch("hmm.market_hmm.MarketHMM", return_value=mock_hmm), \
             patch("alerts.regime_monitor.check_regime_change", return_value=False), \
             patch("alerts.regime_monitor.get_all_regime_status", return_value=mock_status):
            resp = client.get("/api/monitor/check")

        data = resp.json()
        assert data["regime_status"] == mock_status

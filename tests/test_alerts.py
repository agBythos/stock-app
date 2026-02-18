"""
test_alerts.py — Discord Alert 模組測試
========================================

所有測試均使用 mock，不真實發送 Discord Webhook。
使用 unittest.mock.patch 攔截 requests.post。

涵蓋項目：
- format_alert_message 格式正確性
- send_alert：有/無 webhook_url、HTTP 錯誤、requests 失敗
- send_regime_change_alert：regime 切換訊息格式
- 模型持久化：save / load / load_or_train

作者：Bythos（sub-agent phase7-step12）
建立：2026-02-18
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 確保 stock-app 根目錄在 sys.path
# ---------------------------------------------------------------------------

STOCK_APP_ROOT = Path(__file__).resolve().parent.parent
if str(STOCK_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCK_APP_ROOT))

from alerts.discord_alert import (
    format_alert_message,
    send_alert,
    send_regime_change_alert,
)
from backtest.rf_strategy import RandomForestPredictor


# ===========================================================================
# 工具函式
# ===========================================================================


def _make_mock_response(status_code: int = 204, raise_exc: Exception = None):
    """建立 mock requests.Response"""
    resp = MagicMock()
    resp.status_code = status_code
    if raise_exc:
        resp.raise_for_status.side_effect = raise_exc
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_dummy_ohlcv(n: int = 200) -> pd.DataFrame:
    """建立假 OHLCV 資料（用於 RF 訓練）"""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 10.0)  # 不讓價格為負
    df = pd.DataFrame(
        {
            "Open":   close * (1 + rng.normal(0, 0.005, n)),
            "High":   close * (1 + rng.uniform(0, 0.01, n)),
            "Low":    close * (1 - rng.uniform(0, 0.01, n)),
            "Close":  close,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )
    return df


# ===========================================================================
# format_alert_message 測試
# ===========================================================================


class TestFormatAlertMessage:
    def test_basic_format(self):
        msg = format_alert_message("2330.TW", "BUY", 850.0, "Bull", 72.5)
        assert msg == "🚨 2330.TW BUY @850.00 | HMM:Bull | Conf:72.5%"

    def test_sell_signal(self):
        msg = format_alert_message("AAPL", "SELL", 195.50, "Bear", 65.0)
        assert "SELL" in msg
        assert "Bear" in msg
        assert "@195.50" in msg

    def test_hold_signal(self):
        msg = format_alert_message("2317.TW", "HOLD", 100.0, "Neutral", 52.3)
        assert "HOLD" in msg
        assert "52.3%" in msg

    def test_price_decimal_format(self):
        """價格應格式化為兩位小數"""
        msg = format_alert_message("X", "BUY", 1.0, "Bull", 60.0)
        assert "@1.00" in msg

    def test_confidence_one_decimal(self):
        """信心度應格式化為一位小數"""
        msg = format_alert_message("X", "BUY", 100.0, "Bull", 72.123)
        assert "72.1%" in msg

    def test_contains_emoji(self):
        msg = format_alert_message("X", "BUY", 100.0, "Bull", 60.0)
        assert "🚨" in msg


# ===========================================================================
# send_alert 測試
# ===========================================================================


class TestSendAlert:
    def test_skip_when_no_webhook(self, monkeypatch):
        """未設定 DISCORD_WEBHOOK_URL 時應 skip（回傳 False）"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        result = send_alert("2330.TW", "BUY", 850.0, "Bull", 72.5)
        assert result is False

    def test_use_env_webhook(self, monkeypatch):
        """有環境變數時應發送"""
        monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake")
        mock_resp = _make_mock_response(204)
        with patch("alerts.discord_alert._requests.post", return_value=mock_resp) as mock_post:
            result = send_alert("2330.TW", "BUY", 850.0, "Bull", 72.5)
        assert result is True
        mock_post.assert_called_once()

    def test_use_explicit_webhook_url(self, monkeypatch):
        """明確傳入 webhook_url 時應優先使用"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        mock_resp = _make_mock_response(204)
        with patch("alerts.discord_alert._requests.post", return_value=mock_resp) as mock_post:
            result = send_alert("2330.TW", "BUY", 850.0, "Bull", 72.5,
                                webhook_url="https://discord.com/api/webhooks/explicit")
        assert result is True
        mock_post.assert_called_once()

    def test_message_format_in_payload(self, monkeypatch):
        """發送的 payload content 應符合格式"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        mock_resp = _make_mock_response(204)
        with patch("alerts.discord_alert._requests.post", return_value=mock_resp) as mock_post:
            send_alert("TSMC", "SELL", 700.0, "Bear", 80.0,
                       webhook_url="https://discord.com/api/webhooks/test")
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert "🚨" in payload["content"]
        assert "TSMC" in payload["content"]
        assert "SELL" in payload["content"]

    def test_http_error_returns_false(self, monkeypatch):
        """HTTP 錯誤應回傳 False（不拋例外）"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        from requests.exceptions import HTTPError
        mock_resp = _make_mock_response(400, raise_exc=HTTPError("400 Bad Request"))
        with patch("alerts.discord_alert._requests.post", return_value=mock_resp):
            result = send_alert("X", "BUY", 1.0, "Bull", 50.0,
                                webhook_url="https://discord.com/api/webhooks/test")
        assert result is False

    def test_connection_error_returns_false(self, monkeypatch):
        """連線錯誤應回傳 False（不拋例外）"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        from requests.exceptions import ConnectionError as ReqConnError
        with patch("alerts.discord_alert._requests.post", side_effect=ReqConnError("conn refused")):
            result = send_alert("X", "BUY", 1.0, "Bull", 50.0,
                                webhook_url="https://discord.com/api/webhooks/test")
        assert result is False


# ===========================================================================
# send_regime_change_alert 測試
# ===========================================================================


class TestSendRegimeChangeAlert:
    def test_skip_when_no_webhook(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        result = send_regime_change_alert("2330.TW", "Bull", "Bear", 850.0)
        assert result is False

    def test_sends_regime_change(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        mock_resp = _make_mock_response(204)
        with patch("alerts.discord_alert._requests.post", return_value=mock_resp) as mock_post:
            result = send_regime_change_alert(
                "2330.TW", "Bull", "Bear", 850.0,
                webhook_url="https://discord.com/api/webhooks/test"
            )
        assert result is True
        mock_post.assert_called_once()

    def test_regime_change_message_content(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        mock_resp = _make_mock_response(204)
        with patch("alerts.discord_alert._requests.post", return_value=mock_resp) as mock_post:
            send_regime_change_alert(
                "2330.TW", "Bull", "Bear", 850.0,
                webhook_url="https://discord.com/api/webhooks/test"
            )
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs.kwargs["json"]
        content = payload["content"]
        assert "Bull" in content
        assert "Bear" in content
        assert "2330.TW" in content


# ===========================================================================
# 模型持久化測試（RandomForestPredictor save/load）
# ===========================================================================


class TestModelPersistence:
    def test_save_raises_if_not_trained(self, tmp_path):
        """未訓練的模型 save() 應拋 RuntimeError"""
        predictor = RandomForestPredictor()
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.save("TEST", model_dir=tmp_path)

    def test_save_creates_file(self, tmp_path):
        """訓練後 save() 應在 models/ 建立 pkl 檔案"""
        df = _make_dummy_ohlcv(200)
        predictor = RandomForestPredictor()
        predictor.train(df)
        path = predictor.save("TEST", model_date="2026-02-18", model_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".pkl"
        assert "rf_TEST_2026-02-18" in path.name

    def test_load_restores_model(self, tmp_path):
        """load() 應恢復模型狀態，且 is_trained=True"""
        df = _make_dummy_ohlcv(200)
        predictor = RandomForestPredictor(forward_days=5, confidence_threshold=0.6)
        predictor.train(df)
        predictor.save("TEST", model_date="2026-02-18", model_dir=tmp_path)

        loaded = RandomForestPredictor.load("TEST", model_date="2026-02-18", model_dir=tmp_path)
        assert loaded.is_trained is True
        assert loaded.forward_days == 5
        assert loaded.confidence_threshold == 0.6

    def test_load_missing_raises_file_not_found(self, tmp_path):
        """快取不存在時 load() 應拋 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            RandomForestPredictor.load("NONEXIST", model_date="2000-01-01", model_dir=tmp_path)

    def test_loaded_model_can_predict(self, tmp_path):
        """載入的模型應能正常預測"""
        df = _make_dummy_ohlcv(200)
        predictor = RandomForestPredictor()
        predictor.train(df)
        predictor.save("TEST", model_date="2026-02-18", model_dir=tmp_path)

        loaded = RandomForestPredictor.load("TEST", model_date="2026-02-18", model_dir=tmp_path)
        result = loaded.predict("TEST", df)
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0 <= result["confidence"] <= 100

    def test_model_path_sanitizes_symbol(self, tmp_path):
        """symbol 中的 '.' 應被替換為 '_'"""
        path = RandomForestPredictor._model_path("2330.TW", "2026-02-18", tmp_path)
        assert "2330_TW" in path.name
        assert "." not in path.stem  # stem 不含副檔名

    def test_load_or_train_trains_on_miss(self, tmp_path):
        """快取不存在時 load_or_train 應訓練並儲存"""
        df = _make_dummy_ohlcv(200)
        predictor = RandomForestPredictor.load_or_train(
            "TEST2", df,
            model_date="2026-02-18",
            model_dir=tmp_path,
        )
        assert predictor.is_trained is True
        # 快取應已建立
        path = RandomForestPredictor._model_path("TEST2", "2026-02-18", tmp_path)
        assert path.exists()

    def test_load_or_train_loads_on_hit(self, tmp_path):
        """快取存在時 load_or_train 應直接載入，不重新訓練"""
        df = _make_dummy_ohlcv(200)
        # 先建立快取
        p1 = RandomForestPredictor()
        p1.train(df)
        p1.save("TEST3", model_date="2026-02-18", model_dir=tmp_path)
        mtime_before = RandomForestPredictor._model_path("TEST3", "2026-02-18", tmp_path).stat().st_mtime

        # load_or_train 應不重新訓練（mtime 不變）
        p2 = RandomForestPredictor.load_or_train(
            "TEST3", df,
            model_date="2026-02-18",
            model_dir=tmp_path,
        )
        mtime_after = RandomForestPredictor._model_path("TEST3", "2026-02-18", tmp_path).stat().st_mtime
        assert mtime_before == mtime_after
        assert p2.is_trained is True

"""
test_phase9.py — Phase 9 測試
==============================

Phase 9A: Risk Parity Rebalancing
Phase 9B: Signal Alert (check_and_send_signal_alerts)

所有網路呼叫均 mock，不發送真實 Discord webhook。

Run:
    cd stock-app
    python -m pytest tests/test_phase9.py -v

Coverage:
    TC01 — risk_parity: 3 個標的，權重總和 = 1.0
    TC02 — risk_parity: 低波動率標的得到較高權重
    TC03 — risk_parity: 單一標的 → weight = 1.0
    TC04 — risk_parity: yfinance 不可用時拋 ValueError
    TC05 — risk_parity: 資料不足時拋 ValueError
    TC06 — risk_parity: 切換 equal_weight → risk_parity → 驗證更新
    TC07 — rebalance: 不支援的 method 仍拋 ValueError
    TC08 — signal_alert: predict_signal 回傳正確結構
    TC09 — signal_alert: check_and_send_signal_alerts 無 webhook → 不發送
    TC10 — signal_alert: BUY 信心度超門檻 → 發送 discord
    TC11 — signal_alert: HOLD 信號 → 不發送 discord
    TC12 — signal_alert: 預測失敗 → 列入 skipped
    TC13 — signal_alert: alert_on_signals=[] → 不發送
    TC14 — signal_alert: confidence_threshold 過濾（低信心度不發）
    TC15 — signal_alert: 多標的混合結果
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ── sys.path setup ─────────────────────────────────────────────────────────
STOCK_APP_ROOT = Path(__file__).resolve().parent.parent
if str(STOCK_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCK_APP_ROOT))

from portfolio.portfolio_manager import PortfolioManager, STRATEGY_RF, STRATEGY_HMM_RF
from alerts.signal_alert import (
    check_and_send_signal_alerts,
    predict_signal,
    DEFAULT_SYMBOLS,
)


# ===========================================================================
# 工具函式
# ===========================================================================

def _make_ohlcv(n: int = 252, annual_vol: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """建立具有指定年化波動率的假 OHLCV DataFrame"""
    rng = np.random.default_rng(seed)
    # 產生指定波動率的日報酬
    daily_vol = annual_vol / np.sqrt(252)
    daily_returns = rng.normal(0, daily_vol, n)
    close = 100.0 * np.cumprod(1 + daily_returns)
    close = np.maximum(close, 1.0)

    df = pd.DataFrame(
        {
            "Open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
            "High":   close * (1 + rng.uniform(0.001, 0.01, n)),
            "Low":    close * (1 - rng.uniform(0.001, 0.01, n)),
            "Close":  close,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )
    return df


def _mock_ticker(hist_df: pd.DataFrame):
    """建立 yfinance Ticker mock"""
    ticker = MagicMock()
    ticker.history.return_value = hist_df
    return ticker


# ===========================================================================
# Phase 9A: Risk Parity Rebalancing Tests
# ===========================================================================

class TestRiskParityRebalance:

    def _make_pm_3pos(self) -> PortfolioManager:
        pm = PortfolioManager()
        pm.add_position("2330.TW", 0.5, STRATEGY_HMM_RF)
        pm.add_position("0050.TW", 0.3, STRATEGY_RF)
        pm.add_position("2317.TW", 0.2, STRATEGY_RF)
        return pm

    def test_tc01_risk_parity_weights_sum_to_one(self):
        """TC01: risk_parity 三標的，權重總和 ≈ 1.0"""
        pm = self._make_pm_3pos()

        # 各標的不同波動率
        hist_low   = _make_ohlcv(252, annual_vol=0.10, seed=1)   # 低波動
        hist_mid   = _make_ohlcv(252, annual_vol=0.20, seed=2)   # 中波動
        hist_high  = _make_ohlcv(252, annual_vol=0.35, seed=3)   # 高波動

        def fake_ticker(sym):
            mapping = {
                "2330.TW": hist_low,
                "0050.TW": hist_mid,
                "2317.TW": hist_high,
            }
            return _mock_ticker(mapping[sym.upper()])

        with patch("portfolio.portfolio_manager.yf.Ticker", side_effect=fake_ticker):
            weights = pm.rebalance(method="risk_parity")

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum {total} ≠ 1.0"

    def test_tc02_low_vol_gets_higher_weight(self):
        """TC02: 低波動率標的應獲得更高權重"""
        pm = PortfolioManager()
        pm.add_position("LOW", 0.5, STRATEGY_RF)    # 低波動
        pm.add_position("HIGH", 0.5, STRATEGY_RF)   # 高波動

        hist_low  = _make_ohlcv(252, annual_vol=0.10, seed=10)
        hist_high = _make_ohlcv(252, annual_vol=0.40, seed=20)

        def fake_ticker(sym):
            mapping = {"LOW": hist_low, "HIGH": hist_high}
            return _mock_ticker(mapping[sym.upper()])

        with patch("portfolio.portfolio_manager.yf.Ticker", side_effect=fake_ticker):
            weights = pm.rebalance(method="risk_parity")

        assert weights["LOW"] > weights["HIGH"], (
            f"Low-vol should have higher weight: LOW={weights['LOW']:.4f} HIGH={weights['HIGH']:.4f}"
        )

    def test_tc03_single_position_gets_full_weight(self):
        """TC03: 單一持倉 risk_parity → weight = 1.0"""
        pm = PortfolioManager()
        pm.add_position("SOLO", 0.3, STRATEGY_RF)

        hist = _make_ohlcv(252, annual_vol=0.2, seed=5)
        with patch("portfolio.portfolio_manager.yf.Ticker", return_value=_mock_ticker(hist)):
            weights = pm.rebalance(method="risk_parity")

        assert abs(weights["SOLO"] - 1.0) < 1e-9

    def test_tc04_yfinance_unavailable_raises(self):
        """TC04: yfinance 不可用時應拋 ValueError"""
        pm = PortfolioManager()
        pm.add_position("2330.TW", 0.5, STRATEGY_RF)

        import portfolio.portfolio_manager as pm_mod
        orig = pm_mod._YF_AVAILABLE
        try:
            pm_mod._YF_AVAILABLE = False
            with pytest.raises(ValueError, match="yfinance not available"):
                pm.rebalance(method="risk_parity")
        finally:
            pm_mod._YF_AVAILABLE = orig

    def test_tc05_insufficient_data_raises(self):
        """TC05: 資料不足時應拋 ValueError"""
        pm = PortfolioManager()
        pm.add_position("2330.TW", 0.5, STRATEGY_RF)

        tiny_hist = _make_ohlcv(10)  # < 20 bars
        with patch("portfolio.portfolio_manager.yf.Ticker", return_value=_mock_ticker(tiny_hist)):
            with pytest.raises(ValueError, match="Insufficient data"):
                pm.rebalance(method="risk_parity")

    def test_tc06_switch_from_equal_to_risk_parity(self):
        """TC06: 從 equal_weight 切換到 risk_parity 後權重應改變"""
        pm = PortfolioManager()
        pm.add_position("A", 0.5, STRATEGY_RF)
        pm.add_position("B", 0.5, STRATEGY_RF)

        # After equal_weight, both should be 0.5
        ew = pm.rebalance(method="equal_weight")
        assert abs(ew["A"] - 0.5) < 1e-9
        assert abs(ew["B"] - 0.5) < 1e-9

        # After risk_parity (different vols), weights should differ
        hist_a = _make_ohlcv(252, annual_vol=0.10, seed=100)
        hist_b = _make_ohlcv(252, annual_vol=0.30, seed=200)

        def fake_ticker(sym):
            return _mock_ticker({"A": hist_a, "B": hist_b}[sym.upper()])

        with patch("portfolio.portfolio_manager.yf.Ticker", side_effect=fake_ticker):
            rp = pm.rebalance(method="risk_parity")

        assert rp["A"] != pytest.approx(0.5), "risk_parity should differ from equal_weight"
        assert abs(sum(rp.values()) - 1.0) < 1e-9

    def test_tc07_unsupported_method_still_raises(self):
        """TC07: 不支援的 method 仍應拋 ValueError"""
        pm = PortfolioManager()
        pm.add_position("2330.TW", 0.5, STRATEGY_RF)
        with pytest.raises(ValueError, match="Unsupported rebalance method"):
            pm.rebalance(method="markowitz")


# ===========================================================================
# Phase 9B: Signal Alert Tests
# ===========================================================================

class TestSignalAlert:

    def _mock_pred(self, signal: str = "BUY", confidence: float = 72.5) -> MagicMock:
        """建立 RandomForestPredictor mock"""
        predictor = MagicMock()
        predictor.predict.return_value = {
            "signal":     signal,
            "confidence": confidence,
        }
        return predictor

    def test_tc08_predict_signal_structure(self):
        """TC08: predict_signal 回傳正確結構"""
        hist = _make_ohlcv(200)
        mock_pred = MagicMock()
        mock_pred.predict.return_value = {"signal": "BUY", "confidence": 75.0}

        with patch("alerts.signal_alert.RandomForestPredictor", return_value=mock_pred):
            with patch("alerts.signal_alert._HMM_AVAILABLE", False):
                result = predict_signal("2330.TW", hist)

        assert "symbol" in result
        assert result["symbol"] == "2330.TW"
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 100.0
        assert "price" in result
        assert "regime" in result
        assert "error" in result

    def test_tc09_no_webhook_no_alert(self, monkeypatch):
        """TC09: 無 DISCORD_WEBHOOK_URL 且 alert_on_signals=["BUY","SELL"] → 不發送"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        hist = _make_ohlcv(200)

        with patch("alerts.signal_alert._fetch_ohlcv", return_value=hist):
            with patch("alerts.signal_alert.predict_signal", return_value={
                "symbol": "2330.TW", "signal": "BUY", "confidence": 80.0,
                "price": 850.0, "regime": "Bull", "error": None,
            }):
                result = check_and_send_signal_alerts(
                    symbols=["2330.TW"],
                    confidence_threshold=60.0,
                    webhook_url=None,  # no webhook
                )

        assert result["alerts_sent"] == 0

    def test_tc10_buy_signal_high_confidence_sends_alert(self, monkeypatch):
        """TC10: BUY 信心度超門檻 + webhook_url 設定 → 發送 discord"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        hist = _make_ohlcv(200)

        with patch("alerts.signal_alert._fetch_ohlcv", return_value=hist):
            with patch("alerts.signal_alert.predict_signal", return_value={
                "symbol": "2330.TW", "signal": "BUY", "confidence": 80.0,
                "price": 850.0, "regime": "Bull", "error": None,
            }):
                with patch("alerts.signal_alert.send_alert", return_value=True) as mock_send:
                    result = check_and_send_signal_alerts(
                        symbols=["2330.TW"],
                        confidence_threshold=60.0,
                        webhook_url="https://discord.com/api/webhooks/test",
                    )

        assert result["alerts_sent"] == 1
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert call_kwargs.kwargs["signal"] == "BUY"

    def test_tc11_hold_signal_no_alert(self, monkeypatch):
        """TC11: HOLD 信號不發送 Discord alert"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        hist = _make_ohlcv(200)

        with patch("alerts.signal_alert._fetch_ohlcv", return_value=hist):
            with patch("alerts.signal_alert.predict_signal", return_value={
                "symbol": "2330.TW", "signal": "HOLD", "confidence": 85.0,
                "price": 850.0, "regime": "Sideways", "error": None,
            }):
                with patch("alerts.signal_alert.send_alert", return_value=True) as mock_send:
                    result = check_and_send_signal_alerts(
                        symbols=["2330.TW"],
                        confidence_threshold=60.0,
                        webhook_url="https://discord.com/api/webhooks/test",
                    )

        assert result["alerts_sent"] == 0
        mock_send.assert_not_called()

    def test_tc12_prediction_failure_goes_to_skipped(self, monkeypatch):
        """TC12: 預測失敗（error 非 None）→ 列入 skipped"""
        hist = _make_ohlcv(200)

        with patch("alerts.signal_alert._fetch_ohlcv", return_value=hist):
            with patch("alerts.signal_alert.predict_signal", return_value={
                "symbol": "FAIL.TW", "signal": "HOLD", "confidence": 0.0,
                "price": float("nan"), "regime": "Unknown",
                "error": "Training failed",
            }):
                result = check_and_send_signal_alerts(symbols=["FAIL.TW"])

        assert "FAIL.TW" in result["skipped"]
        assert result["alerts_sent"] == 0

    def test_tc13_alert_on_signals_empty_prevents_discord(self, monkeypatch):
        """TC13: alert_on_signals=[] → 不發送任何 alert"""
        hist = _make_ohlcv(200)

        with patch("alerts.signal_alert._fetch_ohlcv", return_value=hist):
            with patch("alerts.signal_alert.predict_signal", return_value={
                "symbol": "2330.TW", "signal": "BUY", "confidence": 99.0,
                "price": 850.0, "regime": "Bull", "error": None,
            }):
                with patch("alerts.signal_alert.send_alert", return_value=True) as mock_send:
                    result = check_and_send_signal_alerts(
                        symbols=["2330.TW"],
                        webhook_url="https://discord.com/api/webhooks/test",
                        alert_on_signals=[],
                    )

        assert result["alerts_sent"] == 0
        mock_send.assert_not_called()

    def test_tc14_confidence_threshold_filters_low_confidence(self, monkeypatch):
        """TC14: 低信心度（< threshold）不發送 alert"""
        hist = _make_ohlcv(200)

        with patch("alerts.signal_alert._fetch_ohlcv", return_value=hist):
            with patch("alerts.signal_alert.predict_signal", return_value={
                "symbol": "2330.TW", "signal": "BUY", "confidence": 45.0,  # < 60 threshold
                "price": 850.0, "regime": "Bull", "error": None,
            }):
                with patch("alerts.signal_alert.send_alert", return_value=True) as mock_send:
                    result = check_and_send_signal_alerts(
                        symbols=["2330.TW"],
                        confidence_threshold=60.0,
                        webhook_url="https://discord.com/api/webhooks/test",
                    )

        assert result["alerts_sent"] == 0
        mock_send.assert_not_called()

    def test_tc15_multi_symbol_mixed_results(self, monkeypatch):
        """TC15: 多標的混合結果（BUY+HOLD+失敗）"""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        hist = _make_ohlcv(200)

        pred_map = {
            "2330.TW": {"symbol": "2330.TW", "signal": "BUY", "confidence": 75.0,
                        "price": 850.0, "regime": "Bull", "error": None},
            "0050.TW": {"symbol": "0050.TW", "signal": "HOLD", "confidence": 50.0,
                        "price": 100.0, "regime": "Sideways", "error": None},
            "2317.TW": {"symbol": "2317.TW", "signal": "SELL", "confidence": 70.0,
                        "price": 200.0, "regime": "Bear", "error": "Network error"},
        }

        def fake_predict(symbol, _hist):
            return pred_map[symbol]

        with patch("alerts.signal_alert._fetch_ohlcv", return_value=hist):
            with patch("alerts.signal_alert.predict_signal", side_effect=fake_predict):
                with patch("alerts.signal_alert.send_alert", return_value=True) as mock_send:
                    result = check_and_send_signal_alerts(
                        symbols=["2330.TW", "0050.TW", "2317.TW"],
                        confidence_threshold=60.0,
                        webhook_url="https://discord.com/api/webhooks/test",
                    )

        # 2330.TW: BUY conf=75 → alerted
        # 0050.TW: HOLD → not alerted
        # 2317.TW: error → skipped
        assert "2330.TW" in result["checked"]
        assert "0050.TW" in result["checked"]
        assert "2317.TW" in result["skipped"]
        assert result["alerts_sent"] == 1
        assert len(result["results"]) == 3

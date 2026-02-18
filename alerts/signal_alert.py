"""
signal_alert.py — ML 訊號 Discord 推播模組
==========================================

對指定股票清單執行 Random Forest 預測，
當預測信號為 BUY 或 SELL 且信心度超過門檻時，
自動發送 Discord 通知。

用法::

    from alerts.signal_alert import check_and_send_signal_alerts

    result = check_and_send_signal_alerts(
        symbols=["2330.TW", "0050.TW"],
        confidence_threshold=60.0,
    )
    # result: {"checked": [...], "alerts_sent": int, "results": [...]}

環境變數：
    DISCORD_WEBHOOK_URL: Discord Incoming Webhook URL

作者：Bythos（sub-agent phase9-stock-app）
建立：2026-02-18
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 可選依賴
# ---------------------------------------------------------------------------

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore
    _YF_AVAILABLE = False

try:
    from backtest.rf_strategy import RandomForestPredictor
    _RF_AVAILABLE = True
except ImportError as _e:
    logger.warning(f"[SignalAlert] RandomForestPredictor not available: {_e}")
    _RF_AVAILABLE = False

try:
    from alerts.discord_alert import send_alert
    _DISCORD_AVAILABLE = True
except ImportError as _e:
    logger.warning(f"[SignalAlert] discord_alert not available: {_e}")
    _DISCORD_AVAILABLE = False

try:
    from hmm.market_hmm import MarketHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = ["2330.TW", "0050.TW", "2317.TW"]
DEFAULT_CONFIDENCE_THRESHOLD = 60.0   # 只推播信心度 >= 60%
DEFAULT_LOOKBACK_DAYS = 180            # 下載 180 天歷史資料


# ---------------------------------------------------------------------------
# 核心函式
# ---------------------------------------------------------------------------


def _fetch_ohlcv(symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    下載 OHLCV 歷史數據。

    Args:
        symbol:       股票代碼
        lookback_days: 歷史天數

    Returns:
        OHLCV DataFrame（含 Open/High/Low/Close/Volume）

    Raises:
        RuntimeError: yfinance 不可用或下載失敗
    """
    if not _YF_AVAILABLE:
        raise RuntimeError("yfinance not installed")

    end = datetime.now()
    start = end - timedelta(days=int(lookback_days * 1.5))
    ticker = yf.Ticker(symbol)
    hist = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if hist.empty or len(hist) < 50:
        raise RuntimeError(f"Insufficient data for {symbol}: {len(hist)} bars")
    return hist


def _get_current_regime(hist: pd.DataFrame) -> str:
    """
    使用 MarketHMM 偵測當前市場 regime。

    Args:
        hist: OHLCV DataFrame

    Returns:
        regime 字串（"Bull" / "Bear" / "Sideways" / "Unknown"）
    """
    if not _HMM_AVAILABLE:
        return "Unknown"
    try:
        hmm = MarketHMM(n_states=3)
        hmm.fit(hist)
        result = hmm.predict_current_state(hist)
        labels = {0: "Bull", 1: "Sideways", 2: "Bear"}
        return labels.get(result.get("state", -1), "Unknown")
    except Exception as exc:
        logger.debug(f"[SignalAlert] HMM regime failed: {exc}")
        return "Unknown"


def predict_signal(
    symbol: str,
    hist: pd.DataFrame,
) -> Dict[str, Any]:
    """
    對單一股票執行 RF 預測，回傳訊號與信心度。

    Args:
        symbol: 股票代碼
        hist:   OHLCV DataFrame（至少 50 筆）

    Returns:
        {
            "symbol":     str,
            "signal":     "BUY" | "SELL" | "HOLD",
            "confidence": float (0–100),
            "price":      float,
            "regime":     str,
            "error":      str | None,
        }
    """
    result: Dict[str, Any] = {
        "symbol":     symbol,
        "signal":     "HOLD",
        "confidence": 0.0,
        "price":      float("nan"),
        "regime":     "Unknown",
        "error":      None,
    }

    if not _RF_AVAILABLE:
        result["error"] = "RandomForestPredictor not available"
        return result

    try:
        price = float(hist["Close"].iloc[-1])
        result["price"] = round(price, 2)

        # Regime
        result["regime"] = _get_current_regime(hist)

        # RF prediction
        predictor = RandomForestPredictor(forward_days=5)
        predictor.train(hist)
        pred = predictor.predict(symbol, hist)

        result["signal"] = pred.get("signal", "HOLD")
        result["confidence"] = float(pred.get("confidence", 0.0))

    except Exception as exc:
        logger.error(f"[SignalAlert] predict_signal failed for {symbol}: {exc}")
        result["error"] = str(exc)

    return result


def check_and_send_signal_alerts(
    symbols: Optional[List[str]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    webhook_url: Optional[str] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    alert_on_signals: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    對指定股票執行預測，高信心度的 BUY/SELL 訊號透過 Discord 推播。

    Args:
        symbols:              股票清單（None 使用預設 DEFAULT_SYMBOLS）
        confidence_threshold: 最低信心度門檻（百分比，預設 60.0）
        webhook_url:          Discord Webhook URL（None 讀環境變數）
        lookback_days:        歷史數據天數（預設 180）
        alert_on_signals:     推播哪些訊號（預設 ["BUY", "SELL"]）

    Returns:
        {
            "timestamp":    str (ISO),
            "checked":      [symbol, ...],
            "skipped":      [symbol, ...],   # 預測失敗的
            "alerts_sent":  int,
            "results":      [{symbol, signal, confidence, price, regime, alerted, error}, ...]
        }
    """
    _symbols = symbols or DEFAULT_SYMBOLS
    _alert_signals = alert_on_signals if alert_on_signals is not None else ["BUY", "SELL"]

    checked: List[str] = []
    skipped: List[str] = []
    alerts_sent: int = 0
    results: List[Dict[str, Any]] = []

    for sym in _symbols:
        entry: Dict[str, Any] = {"symbol": sym, "alerted": False}
        try:
            hist = _fetch_ohlcv(sym, lookback_days=lookback_days)
            pred = predict_signal(sym, hist)
            entry.update(pred)

            if pred["error"]:
                skipped.append(sym)
            else:
                checked.append(sym)
                # 發送 Discord alert
                if (pred["signal"] in _alert_signals
                        and pred["confidence"] >= confidence_threshold
                        and _DISCORD_AVAILABLE):
                    sent = send_alert(
                        symbol=sym,
                        signal=pred["signal"],
                        price=pred["price"],
                        regime=pred["regime"],
                        confidence=pred["confidence"],
                        webhook_url=webhook_url,
                    )
                    entry["alerted"] = sent
                    if sent:
                        alerts_sent += 1
                        logger.info(
                            f"[SignalAlert] Alert sent: {sym} {pred['signal']} "
                            f"conf={pred['confidence']:.1f}%"
                        )

        except Exception as exc:
            logger.error(f"[SignalAlert] check failed for {sym}: {exc}")
            entry["error"] = str(exc)
            entry.setdefault("signal", "HOLD")
            entry.setdefault("confidence", 0.0)
            entry.setdefault("price", float("nan"))
            entry.setdefault("regime", "Unknown")
            skipped.append(sym)

        results.append(entry)

    from datetime import timezone
    return {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "checked":     checked,
        "skipped":     skipped,
        "alerts_sent": alerts_sent,
        "results":     results,
    }

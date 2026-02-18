"""
discord_alert.py — Discord Webhook 通知模組
============================================

透過 Discord Webhook 發送股票信號 Alert。

若環境變數 ``DISCORD_WEBHOOK_URL`` 未設定，所有呼叫均靜默略過（skip），
不拋出例外，適合 CI 環境及本地開發使用。

Alert 格式：
    🚨 {symbol} {signal} @{price} | HMM:{regime} | Conf:{confidence}%

用法::

    from alerts.discord_alert import send_alert

    send_alert(
        symbol="2330.TW",
        signal="BUY",
        price=850.0,
        regime="Bull",
        confidence=72.5,
    )

環境變數：
    DISCORD_WEBHOOK_URL: Discord Incoming Webhook URL
                         若未設定，自動 skip（不報錯）

作者：Bythos（sub-agent phase7-step12）
建立：2026-02-18
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from unittest.mock import MagicMock

logger = logging.getLogger(__name__)

# 嘗試匯入 requests，若不可用則 graceful degrade
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False
    logger.warning("[AlertModule] 'requests' not installed; Discord alerts disabled.")


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def format_alert_message(
    symbol: str,
    signal: str,
    price: float,
    regime: str,
    confidence: float,
) -> str:
    """
    格式化 Alert 訊息字串（純函式，方便測試）

    Args:
        symbol:     股票代碼，e.g. "2330.TW"
        signal:     交易信號，e.g. "BUY" / "SELL" / "HOLD"
        price:      當前價格
        regime:     HMM 市場狀態，e.g. "Bull" / "Bear" / "Neutral"
        confidence: 信心度（百分比，0–100）

    Returns:
        格式化後的 alert 字串

    Example:
        >>> format_alert_message("2330.TW", "BUY", 850.0, "Bull", 72.5)
        '🚨 2330.TW BUY @850.00 | HMM:Bull | Conf:72.5%'
    """
    return f"🚨 {symbol} {signal} @{price:.2f} | HMM:{regime} | Conf:{confidence:.1f}%"


def send_alert(
    symbol: str,
    signal: str,
    price: float,
    regime: str,
    confidence: float,
    webhook_url: Optional[str] = None,
    timeout: int = 10,
) -> bool:
    """
    發送 Discord Alert

    若 ``DISCORD_WEBHOOK_URL`` 未設定（且未傳入 webhook_url），靜默略過。

    Args:
        symbol:      股票代碼
        signal:      交易信號（BUY / SELL / HOLD）
        price:       當前價格
        regime:      HMM 市場狀態
        confidence:  信心度（百分比）
        webhook_url: Webhook URL；若 None 則讀取環境變數
        timeout:     HTTP 請求逾時秒數（預設 10）

    Returns:
        True  — 成功發送
        False — 略過（無 webhook）或發送失敗

    Raises:
        不拋出例外（所有錯誤均記錄至 logger）
    """
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        logger.debug("[AlertModule] DISCORD_WEBHOOK_URL not set; skipping alert.")
        return False

    if not _REQUESTS_AVAILABLE:  # pragma: no cover
        logger.warning("[AlertModule] requests not available; cannot send alert.")
        return False

    message = format_alert_message(symbol, signal, price, regime, confidence)
    payload = {"content": message}

    try:
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        logger.info(f"[AlertModule] Alert sent: {message}")
        return True
    except Exception as exc:
        logger.error(f"[AlertModule] Failed to send alert: {exc}")
        return False


def send_regime_change_alert(
    symbol: str,
    old_regime: str,
    new_regime: str,
    price: float,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    發送 Regime 切換通知（特化版，Step 3 使用）

    Args:
        symbol:     股票代碼
        old_regime: 切換前的市場狀態
        new_regime: 切換後的市場狀態
        price:      切換時的價格
        webhook_url: Webhook URL；若 None 則讀取環境變數

    Returns:
        True 若成功發送
    """
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        logger.debug("[AlertModule] DISCORD_WEBHOOK_URL not set; skipping regime change alert.")
        return False

    if not _REQUESTS_AVAILABLE:  # pragma: no cover
        logger.warning("[AlertModule] requests not available; cannot send alert.")
        return False

    emoji_map = {
        "Bull": "🐂",
        "Bear": "🐻",
        "Neutral": "😐",
        "Volatile": "⚡",
    }
    old_emoji = emoji_map.get(old_regime, "❓")
    new_emoji = emoji_map.get(new_regime, "❓")

    message = (
        f"📊 **{symbol}** Regime 切換！\n"
        f"{old_emoji} {old_regime} → {new_emoji} {new_regime} @{price:.2f}"
    )
    payload = {"content": message}

    try:
        resp = _requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info(f"[AlertModule] Regime change alert sent: {old_regime} → {new_regime}")
        return True
    except Exception as exc:
        logger.error(f"[AlertModule] Failed to send regime change alert: {exc}")
        return False

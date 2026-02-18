"""
regime_monitor.py — Regime 變化偵測模組
=========================================

比對上次 regime vs 當前 regime，偵測切換事件並發送 Discord Alert。

狀態持久化至 ``stock-app/alerts/state.json``，避免重啟後重複通知。

State 格式::

    {
        "2330.TW": {
            "regime":     "Bull",
            "since":      "2026-02-18",
            "last_alert": "2026-02-18"
        }
    }

用法::

    from alerts.regime_monitor import check_regime_change

    alert_sent = check_regime_change("2330.TW", "Bear", 850.0)

作者：Bythos（sub-agent phase7-step34）
建立：2026-02-18
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from alerts.discord_alert import send_regime_change_alert  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 預設 state.json 路徑（與本模組同目錄）
# ---------------------------------------------------------------------------

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "state.json"


# ---------------------------------------------------------------------------
# State 讀寫
# ---------------------------------------------------------------------------


def _load_state(state_path: Path) -> Dict[str, Any]:
    """
    讀取 state.json。

    - 若檔案不存在 → 回傳空 dict（首次執行）
    - 若 JSON 損壞   → 記錄警告，回傳空 dict（重建）

    Args:
        state_path: state.json 路徑

    Returns:
        Dict，key 為 symbol，value 為 regime 狀態資訊
    """
    if not state_path.exists():
        logger.info("[RegimeMonitor] state.json not found; starting fresh.")
        return {}

    try:
        with state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Root element is not a dict")
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"[RegimeMonitor] state.json corrupted ({exc}); rebuilding.")
        return {}


def _save_state(state: Dict[str, Any], state_path: Path) -> None:
    """
    將 state dict 寫入 state.json。

    Args:
        state:      要儲存的狀態 dict
        state_path: 目標路徑

    Raises:
        不拋出例外；若寫入失敗則記錄 error
    """
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.debug(f"[RegimeMonitor] State saved to {state_path}")
    except OSError as exc:
        logger.error(f"[RegimeMonitor] Failed to save state: {exc}")


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def check_regime_change(
    symbol: str,
    current_regime: str,
    price: float,
    state_path: Optional[Path] = None,
    webhook_url: Optional[str] = None,
) -> bool:
    """
    偵測 regime 是否切換；若切換則發送 Discord Alert 並更新狀態。

    首次執行（無 state.json 或 symbol 不在 state 中）：
        - 建立初始狀態，**不發** alert
        - 回傳 False

    Args:
        symbol:         股票代碼，e.g. "2330.TW"
        current_regime: 當前市場狀態，e.g. "Bull" / "Bear" / "Neutral"
        price:          當前價格
        state_path:     state.json 路徑（None 使用預設路徑）
        webhook_url:    Discord Webhook URL（None 讀環境變數）

    Returns:
        True  — 偵測到切換且 alert 已發送（或嘗試發送）
        False — 無切換（或首次執行）
    """
    _state_path = state_path or _DEFAULT_STATE_PATH
    today_str = date.today().isoformat()

    # 1. 讀取現有狀態
    state = _load_state(_state_path)

    # 2. 首次執行：建立初始狀態，不發 alert
    if symbol not in state:
        logger.info(
            f"[RegimeMonitor] First run for {symbol}; "
            f"initialising regime={current_regime}. No alert sent."
        )
        state[symbol] = {
            "regime":     current_regime,
            "since":      today_str,
            "last_alert": None,
        }
        _save_state(state, _state_path)
        return False

    # 3. 比對 regime
    prev_regime = state[symbol].get("regime", "")

    if prev_regime == current_regime:
        logger.debug(
            f"[RegimeMonitor] {symbol}: regime unchanged ({current_regime}). No alert."
        )
        return False

    # 4. Regime 切換！更新狀態並發送 alert
    logger.info(
        f"[RegimeMonitor] {symbol}: regime changed {prev_regime} → {current_regime} @ {price}"
    )
    state[symbol] = {
        "regime":     current_regime,
        "since":      today_str,
        "last_alert": today_str,
    }
    _save_state(state, _state_path)

    alert_sent = send_regime_change_alert(
        symbol=symbol,
        old_regime=prev_regime,
        new_regime=current_regime,
        price=price,
        webhook_url=webhook_url,
    )

    return True  # regime changed (alert may or may not have been sent depending on webhook)


def get_all_regime_status(
    state_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    讀取並回傳所有 symbol 的目前 regime 狀態。

    Args:
        state_path: state.json 路徑（None 使用預設路徑）

    Returns:
        state dict（可能為空 dict）
    """
    _state_path = state_path or _DEFAULT_STATE_PATH
    return _load_state(_state_path)

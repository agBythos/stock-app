"""
scheduler.py — 排程自動監控整合
================================

每日台股收盤後（15:30 Taiwan time, UTC+8）自動執行 regime 監控，
偵測到 regime 切換時寫入 ``stock-app/alerts/alert-log.json``。

Alert Log 格式::

    [
        {
            "timestamp":  "2026-02-18T07:30:00+00:00",
            "symbol":     "2330.TW",
            "old_regime": "Bull",
            "new_regime": "Bear",
            "confidence": 0.82
        },
        ...
    ]

快速使用::

    # 啟動排程器（在 FastAPI startup 事件中呼叫）
    from scheduler import scheduler
    scheduler.start()

    # 手動執行一次檢查
    result = await scheduler.run_once()

    # 讀取 alert log（最近 50 條）
    alerts = read_alert_log(limit=50)

    # 停止排程器
    scheduler.stop()

作者：Bythos（sub-agent stock-app-phase7-step6）
建立：2026-02-18
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常數 & 路徑
# ---------------------------------------------------------------------------

# 所在目錄（stock-app/）
_SCHEDULER_DIR = Path(__file__).resolve().parent

# alert-log.json 預設路徑
DEFAULT_ALERT_LOG_PATH = _SCHEDULER_DIR / "alerts" / "alert-log.json"

# state.json 預設路徑
DEFAULT_STATE_PATH = _SCHEDULER_DIR / "alerts" / "state.json"

# 監控標的清單
DEFAULT_MONITOR_SYMBOLS: List[str] = ["2330.TW", "2317.TW", "0050.TW"]

# 排程時間（台股時間 UTC+8）
SCHEDULE_HOUR: int = 15    # 15:00 台灣時間
SCHEDULE_MINUTE: int = 30  # :30

# Alert log 最大保留條數
MAX_ALERT_LOG_ENTRIES: int = 200


# ---------------------------------------------------------------------------
# Alert Log 工具函式
# ---------------------------------------------------------------------------


def append_alert_log(
    entry: Dict[str, Any],
    log_path: Path = DEFAULT_ALERT_LOG_PATH,
) -> None:
    """
    將 alert 條目追加到 alert-log.json。

    - 若檔案不存在 → 自動建立
    - 若 JSON 損壞  → 記錄警告並以 [新條目] 重建
    - 超過 MAX_ALERT_LOG_ENTRIES（200）條時，只保留最新的 200 條

    Args:
        entry:    dict，至少包含 timestamp、symbol、old_regime、new_regime
        log_path: alert-log.json 路徑（預設為 DEFAULT_ALERT_LOG_PATH）
    """
    # 確保目錄存在
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 讀取現有記錄
    existing: List[Dict[str, Any]] = []
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                existing = data
            else:
                logger.warning("[AlertLog] alert-log.json root is not a list; rebuilding.")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(f"[AlertLog] alert-log.json corrupted ({exc}); rebuilding.")

    # 追加新條目
    existing.append(entry)

    # 超過上限時截斷（保留最新的）
    if len(existing) > MAX_ALERT_LOG_ENTRIES:
        existing = existing[-MAX_ALERT_LOG_ENTRIES:]

    # 原子性寫入：先寫 tmp，再 rename
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=".alert-log-", suffix=".tmp", dir=str(log_path.parent)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        Path(tmp_name).replace(log_path)
        logger.debug(f"[AlertLog] Appended entry for {entry.get('symbol', '?')}; total={len(existing)}")
    except OSError as exc:
        logger.error(f"[AlertLog] Failed to write alert-log.json: {exc}")


def read_alert_log(
    log_path: Path = DEFAULT_ALERT_LOG_PATH,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    讀取 alert-log.json，回傳最近 ``limit`` 條記錄。

    - 若檔案不存在 → 回傳空 list
    - 若 JSON 損壞  → 記錄警告並回傳空 list

    Args:
        log_path: alert-log.json 路徑（預設 DEFAULT_ALERT_LOG_PATH）
        limit:    最多回傳幾條（預設 50）

    Returns:
        最近 limit 條 alert 記錄（最新的在最後）
    """
    if not log_path.exists():
        return []

    try:
        with log_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("[AlertLog] alert-log.json root is not a list.")
            return []
        return data[-limit:]
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"[AlertLog] alert-log.json read error ({exc}).")
        return []


# ---------------------------------------------------------------------------
# 核心監控邏輯
# ---------------------------------------------------------------------------


async def run_monitor_once(
    symbols: Optional[List[str]] = None,
    alert_log_path: Path = DEFAULT_ALERT_LOG_PATH,
    state_path: Optional[Path] = None,
    webhook_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    執行一次完整的 regime 監控週期（coroutine）。

    對每個 symbol：
      1. 下載近 6 個月 OHLCV 數據
      2. 訓練 MarketHMM（n_states=3）並預測當前 regime
      3. 讀取前次 regime（state.json）
      4. 呼叫 check_regime_change() 偵測切換
      5. 若切換 → append_alert_log()

    Args:
        symbols:       監控標的清單（None 使用 DEFAULT_MONITOR_SYMBOLS）
        alert_log_path: alert-log.json 路徑
        state_path:    state.json 路徑（None 使用預設）
        webhook_url:   Discord Webhook URL（None 讀環境變數）

    Returns:
        {
            "checked":      [...],
            "skipped":      [...],
            "alerts_sent":  int,
            "regime_status": {...},
        }
    """
    _symbols = symbols or DEFAULT_MONITOR_SYMBOLS
    _state_path = state_path or DEFAULT_STATE_PATH

    checked: List[str] = []
    skipped: List[str] = []
    alerts_sent: int = 0

    try:
        import yfinance as yf
        from hmm.market_hmm import MarketHMM
        from alerts.regime_monitor import (
            check_regime_change,
            get_all_regime_status,
            _load_state,
        )
    except ImportError as ie:
        logger.error(f"[Scheduler] Import failed: {ie}")
        return {
            "checked": [],
            "skipped": _symbols,
            "alerts_sent": 0,
            "regime_status": {},
            "error": str(ie),
        }

    for symbol in _symbols:
        try:
            # 1. 下載資料（在線程池中執行以免阻塞 event loop）
            loop = asyncio.get_event_loop()
            ticker_obj = yf.Ticker(symbol)
            raw_df = await loop.run_in_executor(
                None, lambda: ticker_obj.history(period="6mo")
            )

            if raw_df is None or raw_df.empty or len(raw_df) < 60:
                logger.info(f"[Scheduler] {symbol}: insufficient data, skipping.")
                skipped.append(symbol)
                continue

            # 2. 訓練 HMM（在線程池中執行）
            hmm_model = MarketHMM(n_states=3)
            await loop.run_in_executor(None, lambda: hmm_model.fit(raw_df))

            # 3. 預測 regime
            regimes = hmm_model.predict(raw_df)
            latest_regime_idx = int(regimes.iloc[-1])
            regime_label_map = {0: "Bull", 1: "Sideways", 2: "Bear"}
            current_regime = regime_label_map.get(latest_regime_idx, f"State{latest_regime_idx}")

            # 取最新收盤價
            close_col = "Close" if "Close" in raw_df.columns else "close"
            latest_price = float(raw_df[close_col].iloc[-1])

            # 4. 取 HMM 信心度（當前 bar 的 posterior prob）
            confidence = 0.0
            try:
                proba_df = hmm_model.predict_proba(raw_df)
                if current_regime in proba_df.columns:
                    confidence = float(proba_df[current_regime].iloc[-1])
            except Exception:
                pass

            # 5. 讀取切換前的 regime（用於 alert log）
            state_before = _load_state(_state_path)
            old_regime = state_before.get(symbol, {}).get("regime", None)

            # 6. 偵測切換
            changed = check_regime_change(
                symbol=symbol,
                current_regime=current_regime,
                price=latest_price,
                state_path=_state_path,
                webhook_url=webhook_url,
            )

            # 7. 若切換 → 寫入 alert log
            if changed and old_regime is not None:
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "old_regime": old_regime,
                    "new_regime": current_regime,
                    "confidence": round(confidence, 4),
                }
                append_alert_log(entry, alert_log_path)
                alerts_sent += 1
                logger.info(
                    f"[Scheduler] {symbol}: regime change {old_regime} → {current_regime} "
                    f"logged (confidence={confidence:.3f})"
                )
            else:
                logger.debug(f"[Scheduler] {symbol}: regime={current_regime}, no change.")

            checked.append(symbol)

        except Exception as exc:
            import traceback
            traceback.print_exc()
            logger.error(f"[Scheduler] {symbol} error: {exc}")
            skipped.append(symbol)

    regime_status = get_all_regime_status(_state_path)

    return {
        "checked":       checked,
        "skipped":       skipped,
        "alerts_sent":   alerts_sent,
        "regime_status": regime_status,
    }


# ---------------------------------------------------------------------------
# 排程器類別
# ---------------------------------------------------------------------------


def get_next_run_time(
    hour: int = SCHEDULE_HOUR,
    minute: int = SCHEDULE_MINUTE,
    tz_offset: int = 8,  # UTC+8 (Asia/Taipei)
) -> datetime:
    """
    計算下次排程執行時間（UTC+8 的 hour:minute）。

    - 若今天的該時間已過 → 回傳明天的該時間（UTC）
    - 若今天的該時間未到 → 回傳今天的該時間（UTC）

    Args:
        hour:      目標小時（台灣時間，24h）
        minute:    目標分鐘
        tz_offset: 時區偏移（預設 8 = UTC+8）

    Returns:
        datetime (UTC, timezone-aware)
    """
    tz = timezone(timedelta(hours=tz_offset))
    now_tw = datetime.now(tz)

    # 今天的目標時間（台灣時間）
    target_tw = now_tw.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if now_tw >= target_tw:
        # 已過 → 明天
        target_tw = target_tw + timedelta(days=1)

    # 轉為 UTC 回傳
    return target_tw.astimezone(timezone.utc)


class MonitorScheduler:
    """
    非同步排程器：每日 15:30 台灣時間自動執行 regime 監控。

    Lifecycle::

        scheduler = MonitorScheduler()
        scheduler.start()          # 在 FastAPI startup 事件中呼叫
        await scheduler.run_once() # 手動觸發一次
        scheduler.stop()           # 在 FastAPI shutdown 事件中呼叫

    Attributes:
        symbols:        監控標的清單
        alert_log_path: alert-log.json 路徑
        state_path:     state.json 路徑
        schedule_hour:  台灣時間排程小時（預設 15）
        schedule_minute: 台灣時間排程分鐘（預設 30）
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        alert_log_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
        schedule_hour: int = SCHEDULE_HOUR,
        schedule_minute: int = SCHEDULE_MINUTE,
        webhook_url: Optional[str] = None,
    ):
        self.symbols = symbols or DEFAULT_MONITOR_SYMBOLS
        self.alert_log_path = alert_log_path or DEFAULT_ALERT_LOG_PATH
        self.state_path = state_path or DEFAULT_STATE_PATH
        self.schedule_hour = schedule_hour
        self.schedule_minute = schedule_minute
        self.webhook_url = webhook_url

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[Dict[str, Any]] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        啟動背景排程任務。

        應在 FastAPI startup 事件中呼叫。
        若已在執行中，呼叫本方法無效（idempotent）。
        """
        if self._running:
            logger.warning("[Scheduler] Already running; start() call ignored.")
            return

        self._running = True
        try:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._loop())
            logger.info(
                f"[Scheduler] Started. Next run at "
                f"{get_next_run_time(self.schedule_hour, self.schedule_minute):%Y-%m-%dT%H:%M:%S} UTC"
            )
        except RuntimeError:
            # 若沒有 running event loop（e.g. 測試環境）
            logger.warning("[Scheduler] No running event loop; task not scheduled.")
            self._running = False

    def stop(self) -> None:
        """
        停止背景排程任務。

        應在 FastAPI shutdown 事件中呼叫（可選）。
        """
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            logger.info("[Scheduler] Stopped.")
        self._task = None

    async def run_once(self) -> Dict[str, Any]:
        """
        手動執行一次完整的監控週期（coroutine）。

        Returns:
            {checked, skipped, alerts_sent, regime_status}
        """
        logger.info("[Scheduler] run_once() triggered.")
        self._last_run = datetime.now(timezone.utc)
        result = await run_monitor_once(
            symbols=self.symbols,
            alert_log_path=self.alert_log_path,
            state_path=self.state_path,
            webhook_url=self.webhook_url,
        )
        self._last_result = result
        logger.info(
            f"[Scheduler] run_once() done: checked={result['checked']}, "
            f"alerts_sent={result['alerts_sent']}"
        )
        return result

    def get_status(self) -> Dict[str, Any]:
        """
        回傳排程器目前狀態（供 API 查詢）。

        Returns:
            {
                "running":    bool,
                "last_run":   str or None,
                "next_run":   str,
                "last_result": dict or None,
            }
        """
        next_run = get_next_run_time(self.schedule_hour, self.schedule_minute)
        return {
            "running":    self._running,
            "last_run":   self._last_run.isoformat() if self._last_run else None,
            "next_run":   next_run.isoformat(),
            "last_result": self._last_result,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """
        主排程迴圈。

        每次計算到下次 15:30 台灣時間的剩餘秒數，sleep 後執行 run_once()。
        """
        logger.info("[Scheduler] Background loop started.")
        while self._running:
            try:
                next_run = get_next_run_time(self.schedule_hour, self.schedule_minute)
                now_utc = datetime.now(timezone.utc)
                wait_seconds = max(0.0, (next_run - now_utc).total_seconds())

                logger.info(
                    f"[Scheduler] Next run in {wait_seconds:.0f}s "
                    f"({next_run:%Y-%m-%dT%H:%M:%S} UTC)"
                )
                await asyncio.sleep(wait_seconds)

                if not self._running:
                    break

                await self.run_once()

            except asyncio.CancelledError:
                logger.info("[Scheduler] Loop cancelled.")
                break
            except Exception as exc:
                logger.error(f"[Scheduler] Loop error: {exc}; retrying in 60s.")
                await asyncio.sleep(60)

        logger.info("[Scheduler] Background loop exited.")


# ---------------------------------------------------------------------------
# 模組全域排程器實例
# ---------------------------------------------------------------------------

# server.py 匯入此實例並呼叫 scheduler.start()
scheduler = MonitorScheduler()

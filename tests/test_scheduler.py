"""
test_scheduler.py — Phase 7 Step 6 排程器測試
================================================

測試項目：
  1.  append_alert_log — 空檔案新增第一條記錄
  2.  append_alert_log — 多條記錄累積正確
  3.  append_alert_log — 超過 MAX_ALERT_LOG_ENTRIES 截斷至 200 條
  4.  append_alert_log — JSON 損壞時重建（只保留新條目）
  5.  read_alert_log — 檔案不存在回傳空 list
  6.  read_alert_log — 回傳最近 N 條（預設 50）
  7.  read_alert_log — limit 參數生效
  8.  read_alert_log — JSON 損壞時回傳空 list
  9.  get_next_run_time — 結果為 UTC timezone-aware datetime
  10. get_next_run_time — 若今天目標時間已過 → 回傳明天
  11. get_next_run_time — 若今天目標時間未到 → 回傳今天
  12. MonitorScheduler — 初始化屬性正確
  13. MonitorScheduler.get_status — 停止時 running=False
  14. MonitorScheduler.stop — 重複呼叫不報錯（idempotent）
  15. run_monitor_once — import 失敗時 graceful 回傳 error key
  16. run_monitor_once — 資料不足時 skip symbol（mock yfinance）
  17. run_monitor_once — 正常執行，無 regime change → alerts_sent=0
  18. run_monitor_once — 正常執行，有 regime change → alerts_sent=1 且 alert-log 新增
  19. /api/monitor/alert-log endpoint — 回傳正確 JSON 格式
  20. /api/monitor/alert-log endpoint — limit 參數生效
  21. alert log entry 格式驗證（必要欄位）
  22. append_alert_log — 原子性寫入（tmp file 清理）

全程使用 mock，不真實下載股票數據，不發送 Discord Webhook。

作者：Bythos（sub-agent stock-app-phase7-step6）
建立：2026-02-18
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# sys.path 確保 stock-app root 可用
# ---------------------------------------------------------------------------

STOCK_APP_ROOT = Path(__file__).resolve().parent.parent
if str(STOCK_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCK_APP_ROOT))

from scheduler import (
    append_alert_log,
    read_alert_log,
    get_next_run_time,
    run_monitor_once,
    MonitorScheduler,
    MAX_ALERT_LOG_ENTRIES,
    DEFAULT_ALERT_LOG_PATH,
    DEFAULT_STATE_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_log(tmp_path) -> Path:
    """回傳一個臨時 alert-log.json 路徑（不預先建立）"""
    return tmp_path / "alert-log.json"


@pytest.fixture
def tmp_state(tmp_path) -> Path:
    """回傳一個臨時 state.json 路徑（不預先建立）"""
    return tmp_path / "state.json"


def _make_entry(symbol="2330.TW", old="Bull", new="Bear", conf=0.80) -> dict:
    """建立一個合法的 alert entry"""
    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "symbol":     symbol,
        "old_regime": old,
        "new_regime": new,
        "confidence": conf,
    }


# ---------------------------------------------------------------------------
# TC-01: append_alert_log — 空檔案新增第一條記錄
# ---------------------------------------------------------------------------


def test_append_creates_file_with_one_entry(tmp_log):
    """空檔案時 append_alert_log 應建立檔案並寫入一條記錄"""
    entry = _make_entry()
    append_alert_log(entry, log_path=tmp_log)

    assert tmp_log.exists(), "alert-log.json 應被建立"
    data = json.loads(tmp_log.read_text(encoding="utf-8"))
    assert isinstance(data, list), "根元素應為 list"
    assert len(data) == 1
    assert data[0]["symbol"] == "2330.TW"
    assert data[0]["old_regime"] == "Bull"
    assert data[0]["new_regime"] == "Bear"


# ---------------------------------------------------------------------------
# TC-02: append_alert_log — 多條記錄累積正確
# ---------------------------------------------------------------------------


def test_append_accumulates_entries(tmp_log):
    """多次 append 應累積所有記錄"""
    for i in range(5):
        append_alert_log(_make_entry(symbol=f"SYM{i}"), log_path=tmp_log)

    data = json.loads(tmp_log.read_text(encoding="utf-8"))
    assert len(data) == 5
    symbols = [d["symbol"] for d in data]
    assert symbols == ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"]


# ---------------------------------------------------------------------------
# TC-03: append_alert_log — 超過 MAX_ALERT_LOG_ENTRIES 截斷
# ---------------------------------------------------------------------------


def test_append_truncates_when_exceeds_max(tmp_log):
    """超過 MAX_ALERT_LOG_ENTRIES（200）條時，只保留最新的 200 條"""
    # 先寫入 MAX 條
    for i in range(MAX_ALERT_LOG_ENTRIES):
        append_alert_log(_make_entry(symbol=f"SYM{i:04d}"), log_path=tmp_log)

    # 再追加一條 → 應截斷至 MAX 條
    append_alert_log(_make_entry(symbol="NEW_ENTRY"), log_path=tmp_log)

    data = json.loads(tmp_log.read_text(encoding="utf-8"))
    assert len(data) == MAX_ALERT_LOG_ENTRIES
    # 最後一條應為 NEW_ENTRY
    assert data[-1]["symbol"] == "NEW_ENTRY"
    # 最早那條（SYM0000）應被移除
    assert data[0]["symbol"] != "SYM0000"


# ---------------------------------------------------------------------------
# TC-04: append_alert_log — JSON 損壞時重建
# ---------------------------------------------------------------------------


def test_append_rebuilds_on_corrupt_json(tmp_log):
    """alert-log.json 損壞時應清空並只保留新記錄"""
    tmp_log.parent.mkdir(parents=True, exist_ok=True)
    tmp_log.write_text("NOT_VALID_JSON{{{", encoding="utf-8")

    entry = _make_entry(symbol="AFTER_CORRUPT")
    append_alert_log(entry, log_path=tmp_log)

    data = json.loads(tmp_log.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["symbol"] == "AFTER_CORRUPT"


# ---------------------------------------------------------------------------
# TC-05: read_alert_log — 檔案不存在回傳空 list
# ---------------------------------------------------------------------------


def test_read_returns_empty_when_no_file(tmp_log):
    """alert-log.json 不存在時 read_alert_log 應回傳 []"""
    result = read_alert_log(log_path=tmp_log)
    assert result == []


# ---------------------------------------------------------------------------
# TC-06: read_alert_log — 有資料時回傳最近 50 條（預設）
# ---------------------------------------------------------------------------


def test_read_returns_last_50_by_default(tmp_log):
    """有 100 條資料時，read_alert_log() 預設回傳最後 50 條"""
    entries = [_make_entry(symbol=f"SYM{i:04d}") for i in range(100)]
    tmp_log.parent.mkdir(parents=True, exist_ok=True)
    tmp_log.write_text(json.dumps(entries), encoding="utf-8")

    result = read_alert_log(log_path=tmp_log)
    assert len(result) == 50
    # 最後 50 條：SYM0050..SYM0099
    assert result[0]["symbol"] == "SYM0050"
    assert result[-1]["symbol"] == "SYM0099"


# ---------------------------------------------------------------------------
# TC-07: read_alert_log — limit 參數生效
# ---------------------------------------------------------------------------


def test_read_respects_limit(tmp_log):
    """limit=10 時只回傳最後 10 條"""
    entries = [_make_entry(symbol=f"SYM{i:04d}") for i in range(30)]
    tmp_log.parent.mkdir(parents=True, exist_ok=True)
    tmp_log.write_text(json.dumps(entries), encoding="utf-8")

    result = read_alert_log(log_path=tmp_log, limit=10)
    assert len(result) == 10
    assert result[0]["symbol"] == "SYM0020"
    assert result[-1]["symbol"] == "SYM0029"


# ---------------------------------------------------------------------------
# TC-08: read_alert_log — JSON 損壞時回傳空 list
# ---------------------------------------------------------------------------


def test_read_returns_empty_on_corrupt_json(tmp_log):
    """alert-log.json JSON 損壞時 read_alert_log 應回傳 []"""
    tmp_log.parent.mkdir(parents=True, exist_ok=True)
    tmp_log.write_text("CORRUPT{{[", encoding="utf-8")

    result = read_alert_log(log_path=tmp_log)
    assert result == []


# ---------------------------------------------------------------------------
# TC-09: get_next_run_time — 結果為 UTC timezone-aware datetime
# ---------------------------------------------------------------------------


def test_get_next_run_time_is_utc_aware():
    """get_next_run_time 回傳值應為 UTC timezone-aware datetime"""
    result = get_next_run_time(hour=15, minute=30)
    assert isinstance(result, datetime)
    assert result.tzinfo is not None
    assert result.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# TC-10: get_next_run_time — 若今天目標時間已過 → 回傳明天
# ---------------------------------------------------------------------------


def test_get_next_run_time_returns_tomorrow_if_past():
    """若今天 15:30 台灣時間已過，應回傳明天的 15:30"""
    # Mock now_tw 為 15:35（已過 15:30）
    tz_tw = timezone(timedelta(hours=8))
    fake_now = datetime(2026, 2, 18, 15, 35, 0, tzinfo=tz_tw)  # 已過 15:30

    with patch("scheduler.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        result = get_next_run_time(hour=15, minute=30)

    # 應為 2026-02-19 15:30 台灣時間 = 2026-02-19 07:30 UTC
    expected_utc = datetime(2026, 2, 19, 7, 30, 0, tzinfo=timezone.utc)
    # 比較日期和小時（不比秒，因為可能有微秒差）
    assert result.date() == expected_utc.date()
    assert result.hour == expected_utc.hour
    assert result.minute == expected_utc.minute


# ---------------------------------------------------------------------------
# TC-11: get_next_run_time — 若今天目標時間未到 → 回傳今天
# ---------------------------------------------------------------------------


def test_get_next_run_time_returns_today_if_future():
    """若今天 15:30 台灣時間未到，應回傳今天的 15:30"""
    tz_tw = timezone(timedelta(hours=8))
    fake_now = datetime(2026, 2, 18, 10, 0, 0, tzinfo=tz_tw)  # 未到 15:30

    with patch("scheduler.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        result = get_next_run_time(hour=15, minute=30)

    # 應為 2026-02-18 15:30 台灣時間 = 2026-02-18 07:30 UTC
    expected_utc = datetime(2026, 2, 18, 7, 30, 0, tzinfo=timezone.utc)
    assert result.date() == expected_utc.date()
    assert result.hour == expected_utc.hour
    assert result.minute == expected_utc.minute


# ---------------------------------------------------------------------------
# TC-12: MonitorScheduler — 初始化屬性正確
# ---------------------------------------------------------------------------


def test_scheduler_init_default_attrs():
    """MonitorScheduler 初始化時屬性應與常數一致"""
    from scheduler import DEFAULT_MONITOR_SYMBOLS, SCHEDULE_HOUR, SCHEDULE_MINUTE

    s = MonitorScheduler()
    assert s.symbols == DEFAULT_MONITOR_SYMBOLS
    assert s.alert_log_path == DEFAULT_ALERT_LOG_PATH
    assert s.state_path == DEFAULT_STATE_PATH
    assert s.schedule_hour == SCHEDULE_HOUR
    assert s.schedule_minute == SCHEDULE_MINUTE
    assert s._running is False
    assert s._task is None
    assert s._last_run is None


# ---------------------------------------------------------------------------
# TC-13: MonitorScheduler.get_status — 停止時 running=False
# ---------------------------------------------------------------------------


def test_scheduler_get_status_when_stopped():
    """未啟動的排程器 get_status 應回傳 running=False"""
    s = MonitorScheduler()
    status = s.get_status()

    assert status["running"] is False
    assert status["last_run"] is None
    assert "next_run" in status
    assert status["last_result"] is None


# ---------------------------------------------------------------------------
# TC-14: MonitorScheduler.stop — 重複呼叫不報錯
# ---------------------------------------------------------------------------


def test_scheduler_stop_idempotent():
    """多次呼叫 stop() 不應拋出例外"""
    s = MonitorScheduler()
    s.stop()  # 未啟動就 stop
    s.stop()  # 再次 stop


# ---------------------------------------------------------------------------
# TC-15: run_monitor_once — import 失敗時 graceful 回傳 error key
# ---------------------------------------------------------------------------


def test_run_monitor_once_import_failure(tmp_log, tmp_state):
    """yfinance import 失敗時應回傳含 error key 的 dict，不拋例外"""
    with patch.dict("sys.modules", {"yfinance": None}):
        result = asyncio.run(run_monitor_once(
            symbols=["2330.TW"],
            alert_log_path=tmp_log,
            state_path=tmp_state,
        ))

    assert "error" in result
    assert result["alerts_sent"] == 0
    assert result["checked"] == []
    assert "2330.TW" in result["skipped"]


# ---------------------------------------------------------------------------
# TC-16: run_monitor_once — 資料不足時 skip symbol
# ---------------------------------------------------------------------------


def test_run_monitor_once_skips_insufficient_data(tmp_log, tmp_state):
    """下載的資料不足 60 bars 時應 skip 該 symbol"""
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()  # 空 DataFrame

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = asyncio.run(run_monitor_once(
            symbols=["2330.TW"],
            alert_log_path=tmp_log,
            state_path=tmp_state,
        ))

    assert "2330.TW" in result["skipped"]
    assert result["alerts_sent"] == 0
    # alert log 應為空
    assert not tmp_log.exists() or read_alert_log(log_path=tmp_log) == []


# ---------------------------------------------------------------------------
# TC-17: run_monitor_once — 無 regime change → alerts_sent=0
# ---------------------------------------------------------------------------


def test_run_monitor_once_no_change(tmp_log, tmp_state):
    """若 regime 未切換，alerts_sent 應為 0，alert log 不新增"""
    import pandas as pd
    import numpy as np

    # 建立 100 bars 的假資料
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    fake_df = pd.DataFrame(
        {
            "Open":   np.random.uniform(500, 600, 100),
            "High":   np.random.uniform(600, 700, 100),
            "Low":    np.random.uniform(400, 500, 100),
            "Close":  np.random.uniform(500, 600, 100),
            "Volume": np.random.randint(1000000, 5000000, 100),
        },
        index=dates,
    )

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = fake_df

    # Mock MarketHMM: 總是回傳 Bull
    mock_hmm = MagicMock()
    bull_series = pd.Series([0] * 100, index=dates)
    mock_hmm.predict.return_value = bull_series
    proba_df = pd.DataFrame(
        {"Bull": [0.9] * 100, "Sideways": [0.05] * 100, "Bear": [0.05] * 100},
        index=dates,
    )
    mock_hmm.predict_proba.return_value = proba_df

    # 先設定 state.json 已有 Bull（模擬「無切換」）
    tmp_state.parent.mkdir(parents=True, exist_ok=True)
    tmp_state.write_text(
        json.dumps({"2330.TW": {"regime": "Bull", "since": "2026-02-17", "last_alert": None}}),
        encoding="utf-8",
    )

    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch("hmm.market_hmm.MarketHMM", return_value=mock_hmm),
    ):
        result = asyncio.run(run_monitor_once(
            symbols=["2330.TW"],
            alert_log_path=tmp_log,
            state_path=tmp_state,
        ))

    assert "2330.TW" in result["checked"]
    assert result["alerts_sent"] == 0
    # alert log 不應新增（可能不存在）
    assert read_alert_log(log_path=tmp_log) == []


# ---------------------------------------------------------------------------
# TC-18: run_monitor_once — 有 regime change → alerts_sent=1 且 alert-log 新增
# ---------------------------------------------------------------------------


def test_run_monitor_once_regime_change(tmp_log, tmp_state):
    """偵測到 regime 切換時 alerts_sent=1，alert-log.json 應新增一條記錄"""
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    fake_df = pd.DataFrame(
        {
            "Open":   np.random.uniform(500, 600, 100),
            "High":   np.random.uniform(600, 700, 100),
            "Low":    np.random.uniform(400, 500, 100),
            "Close":  np.random.uniform(500, 600, 100),
            "Volume": np.random.randint(1000000, 5000000, 100),
        },
        index=dates,
    )

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = fake_df

    # Mock MarketHMM: 回傳 Bear（最新 state=2）
    mock_hmm = MagicMock()
    bear_series = pd.Series([2] * 100, index=dates)
    mock_hmm.predict.return_value = bear_series
    proba_df = pd.DataFrame(
        {"Bull": [0.05] * 100, "Sideways": [0.05] * 100, "Bear": [0.9] * 100},
        index=dates,
    )
    mock_hmm.predict_proba.return_value = proba_df

    # 先設定 state.json 已有 Bull（模擬「有切換 Bull→Bear」）
    tmp_state.parent.mkdir(parents=True, exist_ok=True)
    tmp_state.write_text(
        json.dumps({"2330.TW": {"regime": "Bull", "since": "2026-02-17", "last_alert": None}}),
        encoding="utf-8",
    )

    # Mock Discord alert（不真實發送）
    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch("hmm.market_hmm.MarketHMM", return_value=mock_hmm),
        patch("alerts.discord_alert.send_regime_change_alert", return_value=False),
    ):
        result = asyncio.run(run_monitor_once(
            symbols=["2330.TW"],
            alert_log_path=tmp_log,
            state_path=tmp_state,
        ))

    assert "2330.TW" in result["checked"]
    assert result["alerts_sent"] == 1

    # alert-log.json 應新增一條記錄
    alerts = read_alert_log(log_path=tmp_log)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["symbol"] == "2330.TW"
    assert alert["old_regime"] == "Bull"
    assert alert["new_regime"] == "Bear"
    assert "timestamp" in alert
    assert "confidence" in alert


# ---------------------------------------------------------------------------
# TC-19: /api/monitor/alert-log endpoint — 回傳正確 JSON 格式
# ---------------------------------------------------------------------------


def test_alert_log_endpoint_returns_correct_format(tmp_path):
    """GET /api/monitor/alert-log 應回傳 {alerts: [...], total: int} 格式"""
    from fastapi.testclient import TestClient
    import server as _server

    # 寫入一條假的 alert log
    log_path = tmp_path / "alert-log.json"
    entries = [_make_entry(symbol="2330.TW")]
    log_path.write_text(json.dumps(entries), encoding="utf-8")

    # Patch _read_alert_log 使用臨時路徑
    original_read = _server._read_alert_log if _server._SCHEDULER_AVAILABLE else None

    def mock_read_alert_log(limit=50):
        return read_alert_log(log_path=log_path, limit=limit)

    with patch.object(_server, "_read_alert_log", mock_read_alert_log):
        with patch.object(_server, "_SCHEDULER_AVAILABLE", True):
            client = TestClient(_server.app)
            response = client.get("/api/monitor/alert-log")

    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "total" in data
    assert isinstance(data["alerts"], list)
    assert data["total"] == len(data["alerts"])


# ---------------------------------------------------------------------------
# TC-20: /api/monitor/alert-log endpoint — limit 參數生效
# ---------------------------------------------------------------------------


def test_alert_log_endpoint_limit_param(tmp_path):
    """GET /api/monitor/alert-log?limit=5 應只回傳 5 條"""
    from fastapi.testclient import TestClient
    import server as _server

    log_path = tmp_path / "alert-log.json"
    entries = [_make_entry(symbol=f"SYM{i:02d}") for i in range(20)]
    log_path.write_text(json.dumps(entries), encoding="utf-8")

    def mock_read_alert_log(limit=50):
        return read_alert_log(log_path=log_path, limit=limit)

    with patch.object(_server, "_read_alert_log", mock_read_alert_log):
        with patch.object(_server, "_SCHEDULER_AVAILABLE", True):
            client = TestClient(_server.app)
            response = client.get("/api/monitor/alert-log?limit=5")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 5
    assert len(data["alerts"]) == 5


# ---------------------------------------------------------------------------
# TC-21: alert log entry 格式驗證（必要欄位）
# ---------------------------------------------------------------------------


def test_alert_log_entry_has_required_fields(tmp_log):
    """append_alert_log 寫入的 entry 應包含所有必要欄位"""
    entry = {
        "timestamp":  "2026-02-18T07:30:00+00:00",
        "symbol":     "2317.TW",
        "old_regime": "Sideways",
        "new_regime": "Bear",
        "confidence": 0.75,
    }
    append_alert_log(entry, log_path=tmp_log)

    result = read_alert_log(log_path=tmp_log)
    assert len(result) == 1
    rec = result[0]

    required_fields = {"timestamp", "symbol", "old_regime", "new_regime", "confidence"}
    for field in required_fields:
        assert field in rec, f"Missing field: {field}"

    assert rec["symbol"] == "2317.TW"
    assert rec["old_regime"] == "Sideways"
    assert rec["new_regime"] == "Bear"
    assert rec["confidence"] == 0.75


# ---------------------------------------------------------------------------
# TC-22: append_alert_log — tmp file 不殘留在目錄
# ---------------------------------------------------------------------------


def test_append_no_tmp_files_left(tmp_log):
    """append_alert_log 寫入後不應留下 .tmp 暫存檔案"""
    entry = _make_entry()
    append_alert_log(entry, log_path=tmp_log)

    # 目錄中不應有 .tmp 檔案
    tmp_files = list(tmp_log.parent.glob("*.tmp"))
    assert len(tmp_files) == 0, f"Found leftover tmp files: {tmp_files}"

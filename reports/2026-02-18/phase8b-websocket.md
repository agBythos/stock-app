# Phase 8B — WebSocket Real-time Dashboard

**日期**: 2026-02-18  
**Session**: stock-app-phase8b-websocket  
**狀態**: ✅ 完成

---

## 目標

在既有 Stock App REST API 基礎上，加入 WebSocket 即時行情推送功能，包含：
- 後端 WebSocket endpoint（`/ws/price/{symbol}`）
- 前端即時行情顯示區塊
- 完整 Mock 測試（11 個測試，全部通過）

---

## 新增 / 修改的檔案

| 檔案 | 動作 | 說明 |
|------|------|------|
| `server.py` | 修改 | 新增 WebSocket import、`WebSocketConnectionManager`、`_fetch_realtime_quote()`、`/ws/price/{symbol}` endpoint、`/api/ws/status` |
| `static/index.html` | 修改 | 新增即時行情 Panel HTML + 完整 WebSocket JS 邏輯（連線/斷線/重連/UI更新） |
| `tests/test_websocket.py` | 新增 | 11 個 WebSocket 測試（全部通過） |

---

## 後端變更（`server.py`）

### 新增 Import
```python
from fastapi import ..., WebSocket, WebSocketDisconnect
import asyncio
```

### `WebSocketConnectionManager`
- `max_connections = 10`（全域上限）
- `active: Dict[str, set]` — 依 symbol 分群管理 WebSocket 物件
- `connect()` — 超過上限時以 code 1008 拒絕連線
- `disconnect()` — 自動清理空的 symbol bucket
- `broadcast()` — 廣播 JSON payload；發送失敗的 dead connections 自動移除

### `_fetch_realtime_quote(symbol: str)`
- 優先取 **1 分鐘 intraday** 資料（`yf.Ticker.history(period="1d", interval="1m")`）
- Fallback：**5 日日 K**（`interval="1d"`）
- 計算：`change_pct = (price - prev_close) / prev_close * 100`
- 回傳欄位：`symbol`, `price`, `volume`, `change_pct`, `prev_close`, `timestamp`, `source`

### WebSocket Endpoint: `GET /ws/price/{symbol}`
- 每 **5 秒**推送一次報價 JSON
- 使用 `asyncio.wait_for(receive_text(), timeout=5.0)` 實現非阻塞等待
- `WebSocketDisconnect` 例外觸發 `ws_manager.disconnect()` 清理
- `_fetch_realtime_quote` 失敗時推送 `{"symbol": ..., "error": ...}` 而非崩潰

### 診斷 Endpoint: `GET /api/ws/status`
```json
{
  "total_connections": 0,
  "max_connections": 10,
  "by_symbol": {}
}
```

---

## 前端變更（`static/index.html`）

### 新增 UI 區塊（即時行情 Panel）
位置：股票搜尋結果卡下方、時間範圍按鈕上方

顯示內容：
- **連線狀態指示燈**：綠（連線中）、黃（重連中）、紅（斷線）
- **即時價格** — 小數 2 位
- **日漲跌幅** — 正值綠色（`positive` class）、負值紅色（`negative` class）
- **成交量** — 千分位格式
- 更新時間戳 / 前收 / 來源

### 新增 JS：WebSocket 邏輯
| 函式 | 說明 |
|------|------|
| `_wsOpen(symbol)` | 建立 WebSocket，設置 onopen/onmessage/onerror/onclose |
| `_wsSetStatus(state)` | 更新指示燈顏色 + 文字 + 按鈕顯示 |
| `_wsUpdateUI(data)` | 更新 DOM 價格/漲跌幅/成交量 |
| `wsConnect()` | 手動連線（按鈕） |
| `wsDisconnect()` | 手動斷線（按鈕） |
| `_wsAttachToSymbol(symbol)` | 在 `loadStock()` 成功後自動呼叫，切換 symbol |

### 自動重連機制
- 斷線後以**指數退避**重連：1s → 2s → 4s → … → 30s（最大）
- code 1008（上限拒絕）不重連，改顯示 Toast 提示

---

## 測試結果

```
tests/test_websocket.py::test_fetch_realtime_quote_keys_1m          PASSED
tests/test_websocket.py::test_fetch_realtime_quote_fallback_1d      PASSED
tests/test_websocket.py::test_fetch_realtime_quote_empty_raises      PASSED
tests/test_websocket.py::test_websocket_connect_and_receive          PASSED
tests/test_websocket.py::test_websocket_payload_format               PASSED
tests/test_websocket.py::test_ws_manager_connect_disconnect          PASSED
tests/test_websocket.py::test_ws_manager_max_connections             PASSED
tests/test_websocket.py::test_ws_manager_broadcast                   PASSED
tests/test_websocket.py::test_ws_manager_broadcast_cleans_dead       PASSED
tests/test_websocket.py::test_ws_status_endpoint                     PASSED
tests/test_websocket.py::test_existing_rest_endpoints_not_broken     PASSED

11 passed, 5 warnings in 7.53s
```

合計測試：**11/11 通過**（全 Mock，無真實網路連線）

既有測試（`test_phase7_monitor.py`）：**23/23** 仍全部通過，REST API 未受影響。

---

## 測試覆蓋項目

| # | 測試 | 場景 |
|---|------|------|
| 1 | `test_fetch_realtime_quote_keys_1m` | 1m intraday 路徑：欄位完整性 |
| 2 | `test_fetch_realtime_quote_fallback_1d` | 1d fallback 路徑 |
| 3 | `test_fetch_realtime_quote_empty_raises` | 無資料時拋出 ValueError |
| 4 | `test_websocket_connect_and_receive` | WebSocket 連線 + 第一筆資料接收 |
| 5 | `test_websocket_payload_format` | JSON schema 驗證（型別 + timestamp 格式） |
| 6 | `test_ws_manager_connect_disconnect` | 連線追蹤 + 斷線清理 |
| 7 | `test_ws_manager_max_connections` | 超過 10 條連線時 code 1008 拒絕 |
| 8 | `test_ws_manager_broadcast` | 同一 symbol 多客戶端廣播 |
| 9 | `test_ws_manager_broadcast_cleans_dead` | Dead connection 自動清理 |
| 10 | `test_ws_status_endpoint` | `/api/ws/status` 格式 |
| 11 | `test_existing_rest_endpoints_not_broken` | 既有 REST endpoint 回歸測試 |

---

## HITL 操作記錄

**Level 0 操作**：無  
**Level 1 操作**：無  
**HITL_BLOCKED**：無  

所有操作範疇：
- 新增/修改 workspace 內的 Python / HTML 檔案（正常開發操作）
- 執行 `python -m pytest`（唯讀測試，無副作用）
- 未執行任何破壞性指令、sudo、gateway 操作或金融交易

---

## 架構注意事項

1. **不影響既有 REST API**：所有新增路由均為 `/ws/` 或 `/api/ws/`，與現有 `/api/stock/`、`/api/predict/` 等無衝突
2. **連線管理**：`ws_manager` 為 module-level singleton，FastAPI app 生命週期內持續存在
3. **yfinance 限制**：市場休市時 1m 資料可能延遲或空，fallback 邏輯確保不崩潰
4. **Windows 相容**：asyncio 使用 `run_in_executor` 將同步的 yfinance 呼叫轉為非阻塞，避免事件迴圈阻塞

# Backtrader Phase 3 Fix Report
**Date**: 2026-02-17  
**Branch**: Phase 3 (P0 Bug Fixes)  
**Files Modified**: `static/index.html`, `server.py`

---

## 修復清單

### Bug #1 — `result.metrics` → `result.data.performance`
- **檔案**: `static/index.html` (line ~1114)
- **問題**: `displayBacktestResults` 讀取 `result.metrics`，但 API 回傳結構為 `result.data.performance`
- **修復**: `const metrics = result.metrics || {}` → `const metrics = result.data.performance || {}`
- **狀態**: ✅ 已修復

---

### Bug #2 — `total_return * 100`（重複乘百倍）
- **檔案**: `static/index.html` (lines ~1114–1141)
- **問題**: 前端對 `total_return`、`max_drawdown`、`win_rate` 再乘以 100，但 API 已回傳 `%` 值
  - 例：API 回傳 `15.23`（代表 15.23%），前端再乘 100 顯示為 `1523%`
- **修復**:
  ```js
  // Before (錯誤)
  const totalReturn = (metrics.total_return || 0) * 100;
  const maxDrawdown = (metrics.max_drawdown || 0) * 100;
  const winRate = (metrics.win_rate || 0) * 100;

  // After (正確)
  const totalReturn = metrics.total_return || 0;
  const maxDrawdown = metrics.max_drawdown || 0;
  const winRate = metrics.win_rate || 0;
  ```
- **狀態**: ✅ 已修復

---

### Bug #3 — `result.equity_curve` → `result.data.equity_curve`
- **檔案**: `static/index.html` (line ~1162)
- **問題**: 呼叫 `displayEquityCurve(result.equity_curve, result.buy_hold_curve)` 路徑錯誤
- **修復**: `displayEquityCurve(result.data.equity_curve)`
- **狀態**: ✅ 已修復

---

### Bug #4 — `result.buy_hold_curve` 欄位不存在
- **檔案**: `static/index.html` (line ~1177)
- **問題**: API 沒有 `buy_hold_curve` 欄位；benchmark 資料內嵌於 `equity_curve[n].benchmark`
- **修復**: 
  - `displayEquityCurve` 改為單參數
  - 從每個點的 `.benchmark` 欄位提取 Buy & Hold 曲線
  - Buy & Hold 報酬改為讀 `result.data.benchmark.total_return`
  ```js
  // 從內嵌欄位提取 benchmark
  const hasBenchmark = equityCurve.some(p => p.benchmark != null);
  if (hasBenchmark) {
      const buyHoldData = equityCurve
          .filter(p => p.benchmark != null)
          .map(point => ({ time: point.date, value: point.benchmark }));
      buyHoldSeries.setData(buyHoldData);
  }
  ```
- **狀態**: ✅ 已修復

---

### Bug #5 — 交易明細欄位全錯
- **檔案**: `static/index.html` (table headers + JS render, lines ~590 and ~1230)
- **問題**: 前端讀取 `trade.date / trade.action / trade.size / trade.price`，但 API 回傳欄位為：
  ```
  entry_date, exit_date, entry_price, exit_price, shares, pnl, pnl_percent, commission
  ```
- **修復**:
  - HTML 表頭：5欄 → 7欄（進場日期 / 出場日期 / 進場價 / 出場價 / 股數 / 損益 / 損益%）
  - JS render：改用正確欄位名稱，`colspan="5"` → `colspan="7"`
  - 新增 `pnl_percent` 顯示欄位
- **狀態**: ✅ 已修復

---

### Bug #6 — Backend `/api/backtest/strategies` 未包含 RF
- **檔案**: `server.py` (`list_strategies()` 函式, line ~2399)
- **問題**: 策略列表硬編碼只有 3 個策略（MA Crossover / RSI Reversal / MACD Signal），缺少 RF
- **修復**: 新增第 4 個策略物件：
  ```python
  {
      "name": "rf",
      "display_name": "Random Forest ML",
      "description": "機器學習策略：使用 Random Forest 分類器...",
      "parameters": {
          "forward_days": { "default": 5, "min": 1, "max": 20 },
          "confidence_threshold": { "default": 0.50, "min": 0.40, "max": 0.80 },
          "retrain_period": { "default": 60, "min": 30, "max": 120 }
      }
  }
  ```
- **狀態**: ✅ 已修復

---

### Bug #7 — 前端下拉選單無 RF 選項
- **檔案**: `static/index.html` (`#backtestStrategy` select, line ~499)
- **問題**: 策略下拉選單沒有 Random Forest ML 選項
- **修復**: 新增 `<option value="rf">Random Forest ML</option>`
- **狀態**: ✅ 已修復

---

## 驗證結果（靜態審查）

| 檢查項目 | 結果 |
|---|---|
| `result.metrics` 已消除 | ✅ grep 無結果 |
| `result.data.performance` 使用 | ✅ line 1114 確認 |
| `* 100` 乘法已移除（total_return/max_drawdown/win_rate） | ✅ grep 無餘留 |
| `result.equity_curve` 已消除 | ✅ grep 無結果 |
| `result.data.equity_curve` 使用 | ✅ line 1162 確認 |
| `result.buy_hold_curve` 已消除 | ✅ grep 無結果 |
| benchmark 從 `point.benchmark` 提取 | ✅ line 1184–1190 確認 |
| `result.trades` 已消除 | ✅ 改為 `result.data.trades` |
| 交易明細欄位修正 (`entry_date`, `exit_date`, `entry_price`, `exit_price`, `shares`, `pnl_percent`) | ✅ lines 1240–1245 確認 |
| 表格 colspan 從 5 改為 7 | ✅ line 1220 確認 |
| Backend RF 策略已加入 strategies 列表 | ✅ server.py line 2399 確認 |
| 前端 RF `<option>` 已加入 | ✅ index.html line 502 確認 |

---

## 影響範圍

- `static/index.html`: 回測結果顯示區段（`displayBacktestResults` / `displayEquityCurve` / `displayTradeHistory`）、HTML 表頭、策略下拉選單
- `server.py`: `list_strategies()` endpoint 新增 RF 策略定義

## 注意事項

- 所有修復為純前端欄位對應修正，不影響後端計算邏輯
- RF 策略的 `BacktestEngine.STRATEGY_MAP` 及 `RFStrategy` class 於 Phase 2 已存在，僅需補上 API 列表與前端選項

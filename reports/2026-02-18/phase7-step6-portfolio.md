# Phase 7 Step 6 — Multi-Symbol Portfolio Management

**Date:** 2026-02-18  
**Author:** Bythos (sub-agent)  
**Session:** phase7-step6-portfolio

---

## 完成項目清單

| # | 項目 | 狀態 |
|---|------|------|
| 1 | `portfolio/portfolio_manager.py` — `PortfolioManager` class | ✅ 完成 |
| 2 | `add_position(symbol, weight, strategy_type)` | ✅ 完成 |
| 3 | `remove_position(symbol)` | ✅ 完成 |
| 4 | `rebalance(method="equal_weight")` | ✅ 完成 |
| 5 | `get_portfolio_summary()` — 含 HMM regime 狀態 | ✅ 完成 |
| 6 | `run_portfolio_backtest(start_date, end_date)` | ✅ 完成 |
| 7 | `calculate_correlation_matrix(lookback_days=252)` | ✅ 完成 |
| 8 | `tests/test_portfolio_manager.py` — 23 test cases | ✅ 完成 |
| 9 | `server.py` — `/api/portfolio/summary` GET endpoint | ✅ 完成 |
| 10 | `portfolio/__init__.py` — 更新 exports | ✅ 完成 |

---

## 測試結果

```
23/23 PASS
```

```
============================= test session starts =============================
collected 23 items

tests/test_portfolio_manager.py::test_tc01_module_imports PASSED
tests/test_portfolio_manager.py::test_tc02_instantiation PASSED
tests/test_portfolio_manager.py::test_tc03_add_position_basic PASSED
tests/test_portfolio_manager.py::test_tc04_add_position_invalid_strategy PASSED
tests/test_portfolio_manager.py::test_tc05_add_position_negative_weight PASSED
tests/test_portfolio_manager.py::test_tc06_remove_position_basic PASSED
tests/test_portfolio_manager.py::test_tc07_remove_position_unknown PASSED
tests/test_portfolio_manager.py::test_tc08_rebalance_equal_weight_three PASSED
tests/test_portfolio_manager.py::test_tc09_rebalance_equal_weight_two PASSED
tests/test_portfolio_manager.py::test_tc10_rebalance_unsupported_method PASSED
tests/test_portfolio_manager.py::test_tc11_rebalance_no_positions PASSED
tests/test_portfolio_manager.py::test_tc12_get_portfolio_summary_structure PASSED
tests/test_portfolio_manager.py::test_tc13_summary_regime_has_all_symbols PASSED
tests/test_portfolio_manager.py::test_tc14_correlation_matrix_mocked PASSED
tests/test_portfolio_manager.py::test_tc15_correlation_empty_portfolio PASSED
tests/test_portfolio_manager.py::test_tc16_contains PASSED
tests/test_portfolio_manager.py::test_tc17_len PASSED
tests/test_portfolio_manager.py::test_tc18_position_to_dict PASSED
tests/test_portfolio_manager.py::test_tc19_rebalance_preserves_strategy_type PASSED
tests/test_portfolio_manager.py::test_tc20_valid_strategy_types PASSED
tests/test_portfolio_manager.py::test_tc21_get_positions_structure PASSED
tests/test_portfolio_manager.py::test_tc22_add_position_zero_weight PASSED
tests/test_portfolio_manager.py::test_tc23_repr PASSED

======================== 23 passed in 0.72s ==============================
```

### 既有測試回歸

`tests/test_phase7_portfolio.py`（PortfolioAnalyzer 測試）13/13 仍通過，未破壞既有功能。

---

## API Endpoint 說明

### `GET /api/portfolio/summary`

**Phase 7 Step 6** — 取得預設投資組合的即時摘要。

**預設組合：**
| 標的 | 權重 | 策略 |
|------|------|------|
| 2330.TW (台積電) | 50% | HMM-Filtered RF |
| 0050.TW (元大台灣50) | 30% | Random Forest |
| 2317.TW (鴻海) | 20% | Random Forest |

**Query Parameters：**

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `include_correlation` | bool | `false` | 是否計算相關係數矩陣（較慢） |

**Response 結構：**
```json
{
  "as_of": "2026-02-18T10:00:00",
  "n_positions": 3,
  "total_weight": 1.0,
  "positions": [
    {"symbol": "2330.TW", "weight": 0.5, "strategy_type": "hmm_rf"},
    ...
  ],
  "regime_status": {
    "2330.TW": {
      "symbol": "2330.TW",
      "regime_idx": 0,
      "regime_label": "Bull",
      "regime_proba": [0.75, 0.15, 0.10],
      "data_bars": 126,
      "error": null
    },
    ...
  },
  "portfolio_kpi": {
    "estimated_6m_return_pct": 12.34,
    "estimated_annualized_volatility_pct": 18.5,
    "estimated_sharpe": 0.85,
    "data_bars": 125,
    "note": "Buy-and-hold estimate (6mo, no strategy simulation)"
  },
  "correlation_matrix": {  // 只在 include_correlation=true 時出現
    "symbols": ["2330.TW", "0050.TW", "2317.TW"],
    "matrix": [[1.0, 0.82, 0.71], [0.82, 1.0, 0.65], [0.71, 0.65, 1.0]],
    "lookback_days": 252,
    "data_bars": 245
  }
}
```

**Error Handling：**
- 503: PortfolioManager 模組不可用
- 500: 意外錯誤（含詳細訊息）
- 網路問題：regime 自動 fallback 為 "Unknown"，不拋出錯誤

---

## 架構設計說明

### PortfolioManager Class

**檔案：** `portfolio/portfolio_manager.py`

#### 核心設計原則

1. **關注分離** — `PortfolioManager` 管理組合邏輯（持倉、權重、回測協調）；`PortfolioAnalyzer` 專注靜態統計分析
2. **Graceful Degradation** — 網路、HMM 或回測模組失敗時，個別標的 fallback 而非整體崩潰
3. **可擴展** — `rebalance()` 的 `method` 參數設計為策略模式，後續可加入 risk_parity、min_variance 等
4. **型別安全** — 使用 `@dataclass` 確保內部資料結構一致

#### 資料結構

```
PortfolioManager
├── _positions: Dict[str, Position]
│   └── Position: {symbol, weight, strategy_type}
├── initial_capital: float
└── methods:
    ├── add_position(symbol, weight, strategy_type)
    ├── remove_position(symbol)
    ├── rebalance(method="equal_weight")
    ├── get_portfolio_summary() → PortfolioSummary.to_dict()
    │   ├── _get_regime_for_symbol(sym) → regime info via MarketHMM
    │   └── _compute_portfolio_kpi(prices, positions) → KPI dict
    ├── run_portfolio_backtest(start, end) → PortfolioBacktestResult.to_dict()
    │   ├── Downloads OHLCV per symbol via yfinance
    │   ├── Runs BacktraderEngine.run() per symbol
    │   ├── _build_combined_equity_curve() → weighted normalised curve
    │   └── _aggregate_portfolio_performance() → portfolio KPIs
    └── calculate_correlation_matrix(lookback_days) → corr dict
```

#### 策略映射

| strategy_type | Backtrader Strategy | 說明 |
|---------------|--------------------|----|
| `"rf"` | `RFStrategy` | Random Forest ML 訊號 |
| `"hmm_rf"` | `HMMFilterStrategy` | HMM Regime Filter + RF 訊號 |

#### Rebalance 方法

| 方法 | 說明 | 狀態 |
|------|------|------|
| `equal_weight` | 1/N 等權分配 | ✅ 已實作 |
| `risk_parity` | 風險貢獻等比 | 🔲 待實作 |
| `min_variance` | 最小方差組合 | 🔲 待實作 |
| `max_sharpe` | 最大 Sharpe 比率 | 🔲 待實作 |

---

## 技術問題與解決

### 問題 1: yfinance module-level import

**問題：** `calculate_correlation_matrix()` 最初在函數體內 `import yfinance as yf`，導致 `@patch` mock 無法攔截。

**解決：** 將 yfinance 提升到模組層級 import（帶 graceful fallback），並使用模組層級 `_YF_AVAILABLE` flag。

### 問題 2: Weight 精度與 `to_dict()` rounding

**問題：** `Position.to_dict()` 對 weight round 至 6 位小數，`1/3 ≈ 0.333333`，測試使用 `1e-9` tolerance 失敗。

**解決：** 測試 tolerance 改為 `1e-6`（對應 6 位小數精度）。

### 問題 3: `_build_combined_equity_curve` fillna deprecation

**注意：** `pd.DataFrame.fillna(method=...)` 在 Pandas 2.x 中已棄用，應改用 `ffill()`/`bfill()`。目前程式碼仍用舊 API，若升級 Pandas 需修正（目前 Python 3.12 環境可運行）。

---

## 後續建議

1. **風險管理擴展**
   - 加入 Max Position Size 限制（例如單一標的最多 60%）
   - 加入相關係數觸發自動降低集中度警示

2. **Rebalance 方法擴展**
   - `risk_parity`：使用 inverse volatility 分配權重
   - `min_variance`：需要完整的 covariance matrix 最佳化（scipy）

3. **回測效能優化**
   - 目前 `run_portfolio_backtest()` 串行執行每個標的
   - 可改為 ThreadPool 並行下載數據，加速組合回測

4. **Regime Alert 整合**
   - `get_portfolio_summary()` 已取得各標的 regime
   - 可整合至 `alerts/regime_monitor.py`，當任一持倉進入 Bear 自動發送警示

5. **Portfolio HTML UI 擴展**
   - `static/portfolio.html` 可加入 `/api/portfolio/summary` 的視覺化面板
   - 顯示即時 regime 燈號（🟢 Bull / 🟡 Sideways / 🔴 Bear）

---

## 新增檔案摘要

| 檔案 | 說明 |
|------|------|
| `portfolio/portfolio_manager.py` | PortfolioManager 主要實作（362 行）|
| `tests/test_portfolio_manager.py` | 23 個 test case（全部通過）|
| `portfolio/__init__.py` | 更新 exports，加入 PortfolioManager |
| `server.py` | 新增 `/api/portfolio/summary` endpoint + PortfolioManager import |
| `reports/2026-02-18/phase7-step6-portfolio.md` | 本報告 |

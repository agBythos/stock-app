# Phase 2 — Backtrader 整合報告

**日期：** 2026-02-18  
**Sub-agent：** phase2-backtrader  
**狀態：** ✅ 完成

---

## 📋 任務摘要

整合 Backtrader 為獨立可重用模組，建立完整的台股回測引擎，支援 Walk-Forward 驗證框架。

---

## ✅ 完成項目

### 1. 環境確認

- **Backtrader 版本：** 1.9.78.123（已安裝，無需另行安裝）
- **位置：** `C:\Users\Darren\AppData\Local\Programs\Python\Python312\Lib\site-packages`

### 2. 新建檔案結構

```
stock-app/
└── backtest/
    ├── __init__.py              ✅ 模組入口
    └── backtrader_engine.py    ✅ 核心引擎（~500 行）
```

### 3. `backtest/backtrader_engine.py` 實作內容

#### 主要類別

| 類別 | 功能 |
|------|------|
| `TaiwanCommission` | 台股交易成本計算 |
| `BacktraderEngine` | 主要回測引擎 |
| `BacktestResult` | 回測結果資料類別 |
| `WalkForwardResult` | Walk-Forward 結果資料類別 |
| `TradeRecord` | 單筆交易記錄 |
| `MACrossoverStrategy` | 均線交叉策略 |
| `RSIReversalStrategy` | RSI 反轉策略 |
| `MACDSignalStrategy` | MACD 信號策略 |

#### 台灣交易成本（重要修正）

```
買入手續費：0.1425% × 0.6折 = 0.0855%
賣出手續費：0.1425% × 0.6折 = 0.0855%
賣出證交稅：0.3%
總賣出成本：0.0855% + 0.3% = 0.3855%
```

**驗證（1000股 @ NT$100）：**
- 買入成本：NT$85.5（費率 0.0855% ✅）
- 賣出成本：NT$385.5（費率 0.3855% ✅）

**已修正的 Bug（server.py 中存在）：**  
`bt.CommInfoBase.COMM_PERC` 會自動將 `commission` param 除以 100。  
server.py 使用 `commission=0.001425`，導致 `self.p.commission = 1.425e-05`，  
實際費率只有 **0.001425%**（應為 0.1425%，差 100 倍）。  
新模組改用自訂 param 名 (`buy_rate`, `sell_rate`, `sell_tax_rate`) 避免此問題。

#### BacktraderEngine 核心介面

```python
# 單次回測
engine = BacktraderEngine(symbol="2330.TW", initial_capital=100_000)
result = engine.run(
    strategy_class=MACrossoverStrategy,
    data=df,                              # pandas DataFrame (OHLCV)
    strategy_params={"fast_period": 10, "slow_period": 30},
)

# Walk-Forward 驗證
wf_result = engine.walk_forward(
    strategy_class=MACrossoverStrategy,
    data=df,
    train_months=6,
    test_months=1,
    strategy_params={"fast_period": 10, "slow_period": 30},
)
```

#### BacktestResult 輸出格式

```json
{
  "symbol": "2330.TW",
  "strategy_name": "MA Crossover",
  "period": { "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" },
  "initial_capital": 100000,
  "performance": {
    "final_value": 107250.0,
    "total_return_pct": 7.25,
    "sharpe_ratio": 1.23,
    "max_drawdown_pct": 3.45,
    "win_rate_pct": 60.0,
    "total_trades": 10,
    "winning_trades": 6,
    "losing_trades": 4,
    "avg_trade_pnl": 500.0,
    "profit_factor": 2.1
  },
  "equity_curve": [{"date": "YYYY-MM-DD", "value": 100000.0}, ...],
  "trades": [...]
}
```

#### WalkForwardResult 摘要格式

```json
{
  "summary": {
    "total_windows": 19,
    "successful_windows": 19,
    "avg_return_pct": 0.11,
    "std_return_pct": 2.14,
    "best_return_pct": 4.52,
    "worst_return_pct": -5.25,
    "avg_sharpe": -0.07,
    "avg_max_drawdown_pct": 2.22,
    "total_trades": 6,
    "avg_trades_per_window": 0.3
  }
}
```

---

## 🧪 測試結果

### 模組驗證（全部通過）

```
[OK] Import 成功
[OK] 買入成本正確：NT$85.50（費率 0.0855%）
[OK] 賣出成本正確：NT$385.50（費率 0.3855%）
[OK] BacktraderEngine 初始化正常
[OK] 內建策略：['ma_crossover', 'rsi_reversal', 'macd_signal']
[OK] Sharpe 計算正常
[OK] 最大回撤計算正常
```

### 單次回測（合成資料 250 天）

```
期間: 2023-01-02 ~ 2023-12-15
初始資金: NT$100,000
最終資產: NT$97,296.95
總報酬: -2.703%（隨機數據，結果符合預期）
交易次數: 5
Equity Curve: 220 個資料點
```

### Walk-Forward（合成資料 500 天，3月訓練/1月測試）

```
總窗口: 19
成功窗口: 19（100% 完成率）
平均報酬: 0.11%
報酬標準差: 2.14%
最佳窗口: +4.52%
最差窗口: -5.25%
```

---

## 🔧 設計決策

### 獨立模組設計
- **不依賴 server.py**：可單獨 import 使用，適合測試和批次運行
- **向後相容**：server.py 的現有 `TaiwanStockCommission` 和 `BacktestEngine` 保持不變

### Walk-Forward 實作細節
- **預熱期（warmup_days=60）：** 測試期前加入 60 天歷史資料，確保技術指標（MA60 等）有足夠計算資料
- **窗口滾動：** 每次向前移動 `test_months` 個月（rolling window）
- **績效邊界：** 統計僅計算測試期資料，訓練期不計入績效

### 績效計算
- **Sharpe Ratio：** 年化（`mean/std * sqrt(252)`），無風險利率 = 0
- **最大回撤：** 基於日淨值序列計算 `(value - peak) / peak`
- **Profit Factor：** 總獲利 / |總虧損|（無限大 = 全勝）

---

## ⚠️ 已知限制

1. **RF 策略未整合至獨立模組**：`RFStrategy` 依賴 `RandomForestPredictor`（在 server.py 中），待 Phase 2 後續整合
2. **Walk-Forward 無參數最佳化**：當前為固定參數，後續可加入 `optuna` 超參數搜尋
3. **資料來源**：需呼叫者自行提供 DataFrame（不含 yfinance 下載邏輯，保持模組純淨）

---

## 🔮 後續建議（Phase 2.2）

1. **整合 RF 策略**：建立 `RFStrategyWrapper` 可傳入自訂 Predictor 物件
2. **Walk-Forward 報告視覺化**：生成 HTML 報告（Plotly）
3. **進階成本模型**：支援整股/零股不同手續費、台積電 ADR 等跨市場
4. **並行回測**：多策略/多股票同時執行（`multiprocessing`）

---

*Report generated by sub-agent phase2-backtrader @ 2026-02-18 01:30 GMT+8*

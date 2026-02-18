# Survivorship Bias 存活者偏差

> 文件版本：v1.0 | 最後更新：2026-02-17

---

## 什麼是 Survivorship Bias（存活者偏差）？

存活者偏差（Survivorship Bias）是一種選擇性偏誤：**只關注通過篩選的樣本，忽略已被淘汰的樣本**，導致對整體表現產生過於樂觀的評估。

### 股市中的例子

| 情境 | 有偏差 | 正確做法 |
|------|--------|----------|
| 回測 2010–2024 年台股 | 只用「現在還上市」的股票 | 包含 2010–2024 年間所有曾存在的股票（包含已下市者） |
| 分析「10倍大牛股」特徵 | 只分析最後真的漲10倍的股票 | 分析所有「當時看起來有潛力」的股票 |
| 基金績效比較 | 只顯示還在運作的基金 | 包含已清盤/合併的基金 |

---

## 為什麼 yfinance 有這個問題？

`yfinance` 使用 Yahoo Finance API 拉取數據。Yahoo Finance **只維護當前上市股票的歷史數據**，不保留已下市（delisted）、退市、被合併或破產股票的歷史。

### 具體問題

```
已下市股票（舉例）：
- 博達科技（DRAM廠，2004 年下市）
- 茂矽電子（半導體，已下市）
- 雅虎台灣（2013 下市）
→ 這些股票在 yfinance 中查不到歷史數據
```

### 影響鏈

```
yfinance 拉不到下市股票
→ 回測池只有「活下來」的好公司
→ 回測排除了大量 -50%、-80%、-100% 的倒霉蛋
→ 回測報酬率系統性虛高
```

---

## 對我們回測結果的影響估計

根據學術研究（Elton et al. 1996、Malkiel 1995）和業界估計：

| 市場 | 存活者偏差估計影響 |
|------|------------------|
| 美股（S&P 500 成分股） | **年化 +1.5% ~ +2%** |
| 台股（散戶比例高、新創公司多） | **年化 +2% ~ +5%** |
| 中小型股策略 | **+3% ~ +8%**（下市比例更高） |

### 我們目前使用的股票

我們主要回測對象（2330.TW、AAPL 等大型藍籌股）受存活者偏差影響**相對較小**，因為這些公司在回測期間基本都未下市。

**高風險場景**（存活者偏差影響大）：
- 測試「低本益比策略」——因為 PE 低的股票常是業績下滑的公司
- 測試「小市值效應」——小公司下市率遠高於大公司
- 回測期間超過 10 年的策略

---

## 台股替代數據源

若需要包含已下市股票的完整歷史數據：

### 1. FinMind API（免費）

- **網址**：https://finmindtrade.com/
- **覆蓋範圍**：台股全市場，包含已下市股票
- **資料內容**：日 K、財報、法人買賣超、籌碼面
- **免費額度**：每天限制 API 請求次數（約 600 次/天）
- **Python 套件**：`pip install FinMind`

```python
from FinMind.data import DataLoader

api = DataLoader()
# 包含已下市股票的歷史日K
df = api.taiwan_stock_daily(
    stock_id='2330',
    start_date='2010-01-01',
    end_date='2024-12-31'
)
```

### 2. TEJ 台灣經濟新報（付費）

- **網址**：https://www.tej.com.tw/
- **覆蓋範圍**：最完整的台股歷史數據（含下市、重新上市、配息調整）
- **資料內容**：日 K、財報、信用評等、產業分類、事件數據
- **定價**：依授權方案，學術版約 NT$30,000/年起
- **適用場景**：學術研究、專業量化基金

### 3. 台灣證券交易所 OpenAPI（免費，有限）

- **網址**：https://openapi.twse.com.tw/
- **限制**：主要提供近期數據，歷史不完整

---

## 後續改善建議

1. **短期**（已完成）：在所有回測 API response 加入 `survivorship_bias_warning` 警告欄位，提醒使用者注意

2. **中期**：整合 FinMind API 作為 yfinance 的補充數據源，特別是台股小型股回測

3. **長期**：考慮接入 TEJ 數據（如有商業需求），實現完整無偏差的台股回測

---

## 參考資料

- Elton, E. J., Gruber, M. J., & Blake, C. R. (1996). *Survivor bias and mutual fund performance*. Review of Financial Studies.
- Malkiel, B. G. (1995). *Returns from investing in equity mutual funds 1971 to 1991*. Journal of Finance.
- Liang, B. (2000). *Hedge funds: The living and the dead*. Journal of Financial and Quantitative Analysis.

# Phase 5 HMM 市場機制識別設計文件

> **日期**：2026-02-18  
> **版本**：v1.0  
> **前提**：Phase 4 CPCV(6,2) 驗證框架已完成（28/28 tests PASS）  
> **技術棧**：FastAPI + Backtrader + scikit-learn + yfinance

---

## 1. 背景與目標

### 1.1 為何需要 HMM？

現有 RF 策略基於靜態特徵工程，假設市場行為的統計分佈在時間上保持穩定（Stationarity）。然而真實市場存在**機制轉換（Regime Switching）**：

| 市場機制 | 特徵 | RF 行為 |
|---------|------|---------|
| 牛市（Bull） | 低波動、持續上漲 | 容易過度樂觀 |
| 熊市（Bear） | 高波動、持續下跌 | 信號失靈，止損不及 |
| 震盪（Sideways/Volatile） | 高頻反轉、假突破 | 頻繁交易、磨損嚴重 |

**Hidden Markov Model（HMM）** 是捕捉隱藏狀態序列的概率圖模型，適合對市場機制建模，因為：
- 市場機制（狀態）不直接可觀測，只能從價格/波動推斷
- 機制具有**持續性**（一旦進入牛市，傾向維持）→ Markov 轉移矩陣能捕捉
- 使用 Baum-Welch 算法（EM）可在無標籤數據上做非監督學習

### 1.2 Phase 5 核心目標

```
Phase 5 = HMM 機制識別層 + RF 策略整合
```

1. **Phase 5.1**：HMM 訓練與狀態序列推斷（offline）
2. **Phase 5.2**：HMM 狀態作為 RF 特徵或條件過濾器（整合）
3. **Phase 5.3**：CPCV 驗證框架支援 HMM + RF 聯合評估

---

## 2. HMM 在股票分析的用途

### 2.1 機制識別模型

設定 **N 個隱藏狀態**（推薦從 N=3 開始）：

```
狀態 0：Low-Volatility Bull  （低波動上漲）
狀態 1：High-Volatility Bear  （高波動下跌）
狀態 2：Sideways/Choppy       （震盪/橫盤）
```

HMM 的三個核心參數：
- **π**（初始分佈）：t=0 時各狀態機率
- **A**（轉移矩陣，N×N）：P(S_t+1 | S_t)
- **B**（發射分佈）：觀測值 O_t 在各狀態的條件分佈（Gaussian Mixture 或 Diagonal Covariance）

### 2.2 推斷算法

| 算法 | 用途 | 說明 |
|------|------|------|
| Forward-Backward | 訓練（Baum-Welch） | 計算 α, β 變量 |
| Viterbi | 最優狀態路徑 | 離線回測標記用 |
| Forward-only | 即時推斷 | 線上交易信號 |

> ⚠️ **防洩漏原則**：HMM 訓練必須只用 Train fold 數據，狀態推斷必須用 forward-only（不使用未來數據）。

---

## 3. 特徵選擇

### 3.1 HMM 觀測序列特徵（推薦組合）

```python
# 核心觀測特徵（每日）
hmm_features = [
    "log_return",          # log(Close_t / Close_t-1)：平穩化
    "volatility_20d",      # 20 日滾動波動率（std of log_return * sqrt(252)）
    "volume_ratio",        # Volume / Volume_20d_MA：成交量異常偵測
]
```

**最小可行觀測向量**：`[log_return, volatility_20d]`（2D Gaussian HMM）

**擴充觀測向量**（Phase 5.2+）：

```python
hmm_features_extended = [
    "log_return",          # 日報酬
    "volatility_5d",       # 短期波動
    "volatility_20d",      # 中期波動
    "volume_ratio",        # 成交量比率
    "high_low_range",      # (High - Low) / Close：日內振幅
    "macd_signal",         # MACD histogram 正負
]
```

### 3.2 特徵工程注意事項

```python
# 1. 平穩性處理
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

# 2. 波動率（年化）
df["volatility_20d"] = df["log_return"].rolling(20).std() * np.sqrt(252)

# 3. 成交量比率
df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

# 4. 移除 NaN（rolling 產生的前 N 行）
obs = df[hmm_features].dropna().values

# 5. 標準化（Gaussian HMM 對 scale 敏感）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
obs_scaled = scaler.fit_transform(obs)  # ← 注意：只 fit Train，transform Test
```

---

## 4. 推薦 Python 庫

### 4.1 庫比較

| 庫 | 優點 | 缺點 | 推薦場景 |
|----|------|------|---------|
| **hmmlearn** | scikit-learn API、穩定、輕量、支援 GaussianHMM/GMMHMM | 不支援 GPU、序列長度有限制 | **Phase 5.1 推薦** |
| **pomegranate** | 功能豐富（HMM+MixtureModel）、支援 GPU | v1.0 API 重大改版、文檔較少 | Phase 5.2 擴充 |
| **ssm（State Space Models）** | Bayesian 方法、不確定性量化 | 較複雜、依賴多 | 研究用途 |
| **自實作（NumPy）** | 完全控制、無依賴 | 開發成本高 | 不推薦 |

### 4.2 Phase 5.1 推薦：hmmlearn

```python
# 安裝
pip install hmmlearn>=0.3.0

# 基本使用
from hmmlearn.hmm import GaussianHMM

model = GaussianHMM(
    n_components=3,          # 隱藏狀態數
    covariance_type="diag",  # 對角協方差（比 full 穩定）
    n_iter=100,              # EM 最大迭代次數
    random_state=42,
)
model.fit(obs_train)         # Baum-Welch 訓練

states = model.predict(obs_test)  # Viterbi 解碼（離線）
# 或 forward-only：
log_prob, posteriors = model.score_samples(obs_test)
current_state = np.argmax(posteriors[-1])  # 最新狀態
```

### 4.3 狀態數選擇（N 的選擇）

```python
# AIC/BIC 模型選擇
from hmmlearn.hmm import GaussianHMM

results = []
for n in range(2, 6):
    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=100)
    model.fit(obs_train)
    score = model.score(obs_train)
    n_params = n**2 + n * obs_dim * 2 - 1  # 轉移矩陣 + 均值 + 方差
    bic = -2 * score + n_params * np.log(len(obs_train))
    results.append({"n": n, "bic": bic, "log_likelihood": score})

# 選 BIC 最小的 n
best_n = min(results, key=lambda x: x["bic"])["n"]
```

**經驗值**：台灣股市（台積電等大型股）通常 N=3 最優。

---

## 5. 與現有 RF 策略的整合方式

### 5.1 整合架構圖

```
Raw OHLCV Data
      │
      ├─── Feature Engineering ──► [10個技術指標特徵]
      │                                     │
      └─── HMM Feature Engineering         │
                │                          │
                ▼                          │
         GaussianHMM.fit()                 │
                │                          │
         HMM 狀態推斷                       │
          [state_0, state_1, state_2]      │
                │                          │
                ▼                          ▼
         ┌──────────────────────────────────┐
         │   整合策略（兩種模式）             │
         └──────────────────────────────────┘
              │                    │
         Mode A: Feature       Mode B: Filter
         HMM 狀態→ one-hot       HMM 狀態→ 開倉條件
         加入 RF 特徵集           
              │                    │
              ▼                    ▼
         RF 預測信號          條件式交易信號
              │                    │
              └──────────┬─────────┘
                         ▼
                   Backtrader 回測
                         │
                   CPCV(6,2) 驗證
```

### 5.2 Mode A：HMM 狀態作為 RF 特徵

```python
# 在 RFStrategy / RandomForestPredictor 中加入 HMM 特徵
class HMMEnhancedPredictor(RandomForestPredictor):
    def __init__(self, hmm_model, n_hmm_states=3, **rf_kwargs):
        super().__init__(**rf_kwargs)
        self.hmm_model = hmm_model
        self.n_hmm_states = n_hmm_states

    def prepare_features(self, df):
        # 原有 10 個技術指標特徵
        features = super().prepare_features(df)
        
        # 計算 HMM 觀測序列
        obs = compute_hmm_observations(df)
        
        # 推斷狀態（forward-only，防洩漏）
        _, posteriors = self.hmm_model.score_samples(obs)
        
        # One-hot 或 posterior probability 作為特徵
        hmm_features = posteriors  # shape: (n_samples, n_hmm_states)
        # 或 one-hot：pd.get_dummies(states)
        
        return np.hstack([features, hmm_features])  # 10 + 3 = 13 維
```

**優點**：讓 RF 自動學習機制條件下的最優決策  
**缺點**：HMM 狀態噪聲可能干擾 RF，需要更多數據才能學好

### 5.3 Mode B：HMM 狀態作為交易過濾器（推薦先做）

```python
# 在 RFStrategy 中加入機制過濾邏輯
class HMMFilteredRFStrategy(RFStrategy):
    """
    只在 HMM 識別為 Bull 機制時做多，Bear 機制時空（或不交易）
    """
    params = (
        ("hmm_model", None),
        ("trade_in_states", [0]),     # 只在狀態 0（Bull）交易
        ("hmm_obs_window", 60),       # HMM 觀測視窗
    )

    def next(self):
        # 取最近 N 天的 HMM 觀測特徵
        obs = self._get_hmm_observations(window=self.p.hmm_obs_window)
        
        # Forward-only 推斷（不用 Viterbi，防未來洩漏）
        _, posteriors = self.p.hmm_model.score_samples(obs)
        current_state = np.argmax(posteriors[-1])
        
        # 機制過濾
        if current_state not in self.p.trade_in_states:
            # 非目標機制：平倉或不開新倉
            if self.position:
                self.close()
            return  # 不執行原有 RF 邏輯
        
        # 執行原有 RF 策略邏輯
        super().next()
```

**優點**：邏輯清晰、易解釋、可獨立測試 HMM 效果  
**缺點**：機制分類錯誤時全面失效

### 5.4 與 CPCV 驗證框架的整合

```python
# 修改 CPCVRunner 支援 HMM 預訓練
class HMMCPCVRunner(CPCVRunner):
    def run_fold(self, df_train, df_test, fold_idx):
        # 1. 在 Train fold 訓練 HMM（防洩漏：只用 train 數據）
        hmm_obs_train = compute_hmm_observations(df_train)
        scaler = StandardScaler().fit(hmm_obs_train)
        hmm_model = GaussianHMM(n_components=3, ...).fit(
            scaler.transform(hmm_obs_train)
        )
        
        # 2. 在 Test fold 推斷狀態（只用 forward-only）
        hmm_obs_test = compute_hmm_observations(df_test)
        
        # 3. 用帶 HMM 的策略進行 backtest
        strategy = HMMFilteredRFStrategy(hmm_model=hmm_model, ...)
        return super().run_fold(df_train, df_test, fold_idx, strategy=strategy)
```

> **關鍵**：HMM 的 `StandardScaler` 的 `fit` 必須只在 train fold，`transform` 才能應用到 test。

---

## 6. 實作優先順序

### Phase 5.1：HMM 核心模組（2-3 天）

**目標**：建立獨立的 `hmm/` 模組，可訓練和推斷

```
stock-app/
├── hmm/
│   ├── __init__.py          # exports: HMMModel, HMMConfig, compute_hmm_obs
│   ├── hmm_model.py         # GaussianHMM 封裝，fit/predict/score_samples
│   ├── hmm_features.py      # compute_hmm_observations()，特徵工程
│   └── hmm_config.py        # HMMConfig dataclass
├── tests/
│   └── test_hmm_core.py     # 單元測試
```

**交付物**：
- [ ] `HMMModel.fit(df_train)` → 訓練 HMM
- [ ] `HMMModel.infer_state(df)` → forward-only 推斷最新狀態
- [ ] `HMMModel.state_sequence(df)` → Viterbi 回測用
- [ ] BIC 自動選擇最優 N（2~5）
- [ ] 基本單元測試（合成數據驗證）

**測試指標**：
```python
# 測試 HMM 能否分離高/低波動機制
assert model.means_[bull_state]["volatility_20d"] < model.means_[bear_state]["volatility_20d"]
```

---

### Phase 5.2：RF 策略整合（2-3 天）

**目標**：實作 Mode B（過濾器），加入 API 端點

```
stock-app/
├── backtest/
│   ├── hmm_rf_strategy.py   # HMMFilteredRFStrategy（extends RFStrategy）
│   └── __init__.py          # 加入 HMMFilteredRFStrategy export
├── server.py                # 加入 /backtest/hmm-rf endpoint
```

**新增 API 端點**：
```
POST /backtest/hmm-rf
{
  "symbol": "2330.TW",
  "start": "2018-01-01",
  "end": "2024-01-01",
  "hmm_states": 3,
  "trade_in_states": [0],    # 只在 Bull 做多
  "hmm_features": ["log_return", "volatility_20d", "volume_ratio"]
}
```

**交付物**：
- [ ] `HMMFilteredRFStrategy` 可在 BacktraderEngine 執行
- [ ] HMM 訓練嵌入 Walk-Forward 流程（每個 walk-forward fold 各自訓練）
- [ ] API 端點回傳機制分佈統計（% bull / % bear / % sideways）

---

### Phase 5.3：CPCV 聯合驗證（2-3 天）

**目標**：用 CPCV(6,2) 框架評估 HMM+RF 聯合策略，生成完整報告

```
stock-app/
├── validation/
│   ├── hmm_cpcv_runner.py   # HMMCPCVRunner（extends CPCVRunner）
│   └── __init__.py          # 加入 HMMCPCVRunner export
├── tests/
│   └── test_hmm_cpcv.py     # 整合測試（目標：新增 ~8 tests）
```

**報告擴充**：
- CPCV 報告加入 HMM 機制統計（各 path 的機制分佈）
- 各機制下的 Sharpe Ratio（Bull Sharpe vs Bear Sharpe）
- 機制轉移矩陣視覺化

**交付物**：
- [ ] `HMMCPCVRunner.run()` → 回傳 `HMMCPCVReport`（extends CPCVReport）
- [ ] 新增 `/validation/hmm-cpcv` API 端點
- [ ] 整合測試全部 PASS

---

## 7. 預估工作量

| Phase | 工作內容 | 預估天數 | 難度 |
|-------|---------|---------|------|
| 5.1 | HMM 核心模組（hmm/） | 2~3 天 | ⭐⭐ |
| 5.1 | 單元測試 | 0.5 天 | ⭐ |
| 5.2 | HMMFilteredRFStrategy | 1~2 天 | ⭐⭐ |
| 5.2 | Walk-Forward HMM 整合 | 1 天 | ⭐⭐⭐ |
| 5.2 | API 端點 | 0.5 天 | ⭐ |
| 5.3 | HMMCPCVRunner | 1~2 天 | ⭐⭐⭐ |
| 5.3 | 整合測試 | 1 天 | ⭐⭐ |
| **合計** | | **7~10 天** | |

> **建議起點**：先完成 Phase 5.1 的 `HMMModel`，並用歷史數據（2330.TW 5 年）做視覺化驗證（繪製機制標記在 K 線圖上），確認 HMM 的機制識別合理後再整合。

---

## 8. 風險與注意事項

### 8.1 防洩漏（Data Leakage）

| 風險點 | 問題 | 解決方案 |
|-------|------|---------|
| HMM 用全量數據訓練 | 未來資訊洩漏給過去 | 每個 fold 獨立訓練 HMM |
| Viterbi 解碼用全序列 | 利用未來狀態 | 推斷時只用 forward-only |
| StandardScaler fit on all | 測試集統計污染訓練集 | Scaler 只 fit train fold |

### 8.2 HMM 訓練穩定性

- Baum-Welch 容易陷入**局部最優**：多次 random init 取最佳 log-likelihood
- 短序列（< 100 天）不穩定：建議最少 252 天（1 年）訓練數據
- 狀態排列不固定：每次訓練狀態 0/1/2 的意義可能互換 → 需要**狀態排序（按波動率升序）**

```python
# 狀態標準化：按均值波動率排序，確保跨 fold 一致性
volatility_means = model.means_[:, volatility_feature_idx]
state_order = np.argsort(volatility_means)  # 低→高
# state_order[0] = Low-vol Bull, state_order[-1] = High-vol Bear
```

### 8.3 計算效能

- `GaussianHMM.fit()` 在 1000 天數據上約 0.1~0.5 秒
- CPCV(6,2) 有 C(6,2)=15 paths × 4 folds = 60 次訓練 → 約 6~30 秒
- 可接受，不需要 GPU 加速

### 8.4 超參數

建議 Phase 5.1 固定以下超參數，後續再做調優：
```python
HMMConfig(
    n_states=3,
    covariance_type="diag",  # 比 "full" 穩定
    n_iter=100,
    n_init=5,                # 5 次隨機初始化，取最佳
    min_train_days=252,
)
```

---

## 9. 依賴套件更新

```
# requirements.txt 新增
hmmlearn>=0.3.0
```

無需更改其他依賴。`hmmlearn` 依賴 `numpy` 和 `scipy`（已存在），與 `sklearn` API 一致。

---

## 10. 下一步行動

1. **立即**：`pip install hmmlearn` 並確認版本（`0.3.x`）
2. **Phase 5.1 開始**：建立 `stock-app/hmm/` 目錄和 `HMMConfig` dataclass
3. **驗收標準**：用 2330.TW 2018-2024 數據，HMM 能合理標記 COVID 崩跌（2020-03）為 Bear 機制
4. **後續擴充**：Phase 5.2 完成後，考慮加入 **Mode A**（HMM 狀態作為 RF 特徵）做 A/B 比較

---

*文件由 Bythos 生成於 2026-02-18，基於 Phase 4 CPCV(6,2) 架構設計。*

# Ensemble Model Comparison Design
## RandomForest vs XGBoost vs LightGBM vs CatBoost for 台股預測

**日期**: 2026-02-17  
**背景**: Stock App 目前使用 RandomForestClassifier（已知 2330.TW 約 75% 勝率）  
**目標**: 評估三種 Gradient Boosting 模型是否能超越現有 RF baseline

---

## 1. 文獻研究摘要

### 1.1 各模型特性對比

| 特性 | RandomForest | XGBoost | LightGBM | CatBoost |
|------|-------------|---------|----------|----------|
| 樹成長方向 | 平行/bagging | 層序(Level-wise) | 葉優先(Leaf-wise) | 對稱樹(Symmetric) |
| 正規化 | 無（靠bagging） | L1+L2（alpha, lambda） | L1+L2 | 對稱結構 |
| Overfitting 風險 | 低 | 中 | 高（需調 num_leaves） | 低（ordered boosting） |
| 金融時序適用性 | 中 | 中 | 高 | **最高** |
| 訓練速度 | 慢 | 中 | **最快** | 中 |
| 類別特徵 | 需手動 | 不支援 | 支援 | **原生支援** |

### 1.2 金融數據研究結論

根據量化研究文獻（Shwartz-Ziv & Armon 2021, arXiv:2106.03253）：

> **XGBoost > Deep Learning** on tabular data（包含金融特徵表格）  
> **CatBoost ≈ LightGBM > XGBoost** on highly structured/ordered tabular data

**關鍵差異（金融時序）**：

1. **CatBoost 的 Ordered Boosting**
   - 避免 target leakage（金融時序中的常見問題）
   - 每個樣本只用它之前的樣本訓練，天然防止時序洩露
   - 對股票預測特別重要，因為訓練數據是時序排列的

2. **LightGBM 的 GOSS（Gradient-based One-Side Sampling）**
   - 保留梯度大的樣本（市場異常點），對捕捉趨勢轉折有優勢
   - 訓練速度快，適合做 walk-forward 滾動訓練

3. **XGBoost 的優勢**
   - 最穩健，過擬合控制最全面（L1+L2+subsample）
   - Kaggle 金融競賽最常見選擇（尤其是結構化特徵）

4. **RandomForest 的局限**
   - Bagging 方法：各樹獨立，無法學習前一棵樹的錯誤
   - 缺乏 early stopping，無法在金融時序中自適應

---

## 2. 超參數調優建議（金融數據專用）

### 2.1 RandomForest（Baseline）

```python
RandomForestClassifier(
    n_estimators=200,       # 增加樹數量
    max_depth=8,            # 限制深度防止過擬合
    min_samples_leaf=20,    # 每葉至少20個樣本
    max_features='sqrt',    # 隨機選 sqrt(n_features) 個特徵
    random_state=42,
    n_jobs=-1
)
```

**金融數據注意事項**：
- `min_samples_leaf=20`：防止在噪聲數據上過擬合
- `max_depth=8`：金融信號通常是淺層的，深樹會學到噪聲

### 2.2 XGBoost

```python
XGBClassifier(
    n_estimators=500,       # 用 early stopping 決定實際數量
    max_depth=5,            # 金融數據不需要太深
    learning_rate=0.05,     # 小學習率 + 多樹
    subsample=0.8,          # 行抽樣防過擬合
    colsample_bytree=0.8,   # 列抽樣防過擬合
    reg_alpha=0.1,          # L1 正規化
    reg_lambda=1.0,         # L2 正規化
    min_child_weight=5,     # 葉節點最小樣本權重
    gamma=0.1,              # 最小增益閾值
    scale_pos_weight=1,     # 類別不平衡時調整
    eval_metric='logloss',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)
```

**關鍵調優邏輯**：
- `learning_rate=0.05` + `n_estimators=500`：小步走多步，比 `lr=0.1` + 250 樹泛化更好
- `subsample=0.8`：隨機抽樣行，類似 RF 的 bagging 效果
- `min_child_weight=5`：等同 RF 的 `min_samples_leaf`，防止葉節點太小

### 2.3 LightGBM

```python
LGBMClassifier(
    n_estimators=500,       # 用 early stopping 決定
    num_leaves=31,          # LightGBM 核心參數！(不是 max_depth)
    max_depth=6,            # 配合 num_leaves 限制
    learning_rate=0.05,
    subsample=0.8,          # 行抽樣
    colsample_bytree=0.8,   # 列抽樣 (LightGBM 稱 feature_fraction)
    reg_alpha=0.1,          # L1
    reg_lambda=1.0,         # L2
    min_child_samples=20,   # 葉節點最小樣本數（金融必調！）
    min_gain_to_split=0.01, # 最小分裂增益
    bagging_freq=5,         # 每5輪做一次 bagging
    verbose=-1,
    random_state=42,
    n_jobs=-1
)
```

**LightGBM 金融防過擬合重點**：
- `num_leaves ≤ 2^max_depth`：確保樹不會太複雜
- `min_child_samples=20`：**最重要**，防止學到稀疏的噪聲模式
- `bagging_freq=5`：啟用 bagging（需配合 `subsample`）

### 2.4 CatBoost

```python
CatBoostClassifier(
    iterations=500,             # 用 early stopping 決定
    depth=6,                    # 對稱樹深度
    learning_rate=0.05,
    l2_leaf_reg=3.0,            # L2 正規化（CatBoost 默認 3，金融可提高）
    border_count=64,            # 數值特徵分桶數
    od_type='Iter',             # Overfitting detector 類型
    od_wait=50,                 # 等待 50 輪無改善才停止
    use_best_model=True,        # 自動選最佳迭代
    eval_metric='Logloss',
    random_seed=42,
    verbose=100
)
```

**CatBoost 金融優勢**：
- `use_best_model=True`：自動避免過擬合，適合金融時序
- `od_wait=50`：等待更多輪次確認過擬合（金融信號波動大）
- Ordered boosting（默認開啟）：自動防止時序 target leakage

---

## 3. Early Stopping 最佳實踐

### 3.1 時序分割（Time Series Split）

```python
# 金融數據必須用時序分割，不能隨機分割！
from sklearn.model_selection import TimeSeriesSplit

# 方案一：簡單 70/30 時序分割
def time_series_split(X, y, test_ratio=0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

# 方案二：Walk-Forward（更嚴謹）
def walk_forward_split(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(X)
```

### 3.2 Early Stopping 使用方式

```python
# XGBoost - 需要設 eval_set
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # 必須提供驗證集
    verbose=100
)
# early_stopping_rounds 已在 constructor 設定

# LightGBM - 使用 callbacks
import lightgbm as lgb
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# CatBoost - eval_set + od_wait
cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=100
)
```

---

## 4. 對比實驗方案

### 4.1 實驗設定

```
股票池：2330.TW（台積電）+ 2317.TW（鴻海）+ 0050.TW（台灣50）
訓練期：2020-01-01 ~ 2023-12-31（4年）
測試期：2024-01-01 ~ 2025-12-31（2年 out-of-sample）
分割比例：80% train / 20% validation（for early stopping）
時序分割：先 train，再 val，再 test（嚴格時序）
```

### 4.2 共同 Feature Set（與 Stock App 一致）

```python
FEATURE_NAMES = [
    'rsi_14',          # RSI(14)
    'macd_hist',       # MACD Histogram
    'ma_cross_10_30',  # MA(10)/MA(30) cross ratio
    'ma_cross_20_60',  # MA(20)/MA(60) cross ratio
    'bb_pct_b',        # Bollinger Band %B
    'vol_ratio',       # Volume ratio vs 20d avg
    'momentum_5d',     # 5日動量
    'momentum_10d',    # 10日動量
    'momentum_20d',    # 20日動量
    'volatility_20d',  # 20日波動率
]
```

### 4.3 評估指標

```
- Accuracy（準確率）：整體預測正確率
- Precision（精確率）：預測買入中實際上漲的比例（重要！）
- Recall（召回率）：實際上漲中被預測到的比例
- F1 Score：Precision 和 Recall 的調和平均
- AUC-ROC：模型辨別能力
- Sharpe Ratio（交易模擬）：基於預測信號的策略夏普比率
```

---

## 5. 完整 Python 代碼實現

### 5.1 通用 Feature Engineering（與現有 Stock App 兼容）

```python
import pandas as pd
import numpy as np

def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    計算技術指標特徵（與現有 Stock App server.py 兼容）
    """
    df = data.copy()
    
    # RSI(14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    df['macd_hist'] = macd - macd.ewm(span=9).mean()
    
    # MA Crossovers
    for short, long in [(10, 30), (20, 60)]:
        df[f'ma_cross_{short}_{long}'] = (
            df['Close'].rolling(short).mean() - df['Close'].rolling(long).mean()
        ) / df['Close']
    
    # Bollinger Bands %B
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_pct_b'] = (df['Close'] - (bb_mid - 2*bb_std)) / (4 * bb_std)
    
    # Volume Ratio
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Momentum
    for days in [5, 10, 20]:
        df[f'momentum_{days}d'] = df['Close'].pct_change(days)
    
    # Volatility
    df['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
    
    feature_names = [
        'rsi_14', 'macd_hist', 'ma_cross_10_30', 'ma_cross_20_60',
        'bb_pct_b', 'vol_ratio', 'momentum_5d', 'momentum_10d',
        'momentum_20d', 'volatility_20d'
    ]
    return df[feature_names]


def calculate_target(data: pd.DataFrame, forward_days: int = 5) -> pd.Series:
    """未來 N 天漲跌（1=漲, 0=跌）"""
    future_return = data['Close'].pct_change(forward_days).shift(-forward_days)
    return (future_return > 0).astype(int)
```

### 5.2 RandomForest（Baseline，已優化）

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def build_rf_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

def train_rf(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    model = build_rf_model()
    model.fit(X_scaled, y_train)
    
    return model, scaler

# 預測
def predict_rf(model, scaler, X_test):
    X_scaled = scaler.transform(X_test)
    proba = model.predict_proba(X_scaled)[:, 1]
    return proba
```

### 5.3 XGBoost

```python
from xgboost import XGBClassifier

def build_xgb_model():
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        gamma=0.1,
        eval_metric='logloss',
        early_stopping_rounds=50,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

def train_xgb(X_train, y_train, X_val, y_val):
    """
    注意：XGBoost early stopping 需要 eval_set
    不需要 StandardScaler（tree-based 模型不需要特徵縮放）
    """
    model = build_xgb_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    print(f"[XGB] Best iteration: {model.best_iteration}")
    return model

def predict_xgb(model, X_test):
    return model.predict_proba(X_test)[:, 1]
```

### 5.4 LightGBM

```python
import lightgbm as lgb
from lightgbm import LGBMClassifier

def build_lgbm_model():
    return LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,  # 關鍵防過擬合參數
        min_gain_to_split=0.01,
        bagging_freq=5,
        verbose=-1,
        random_state=42,
        n_jobs=-1
    )

def train_lgbm(X_train, y_train, X_val, y_val):
    """
    注意：LightGBM early stopping 使用 callbacks API
    """
    model = build_lgbm_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    print(f"[LGBM] Best iteration: {model.best_iteration_}")
    return model

def predict_lgbm(model, X_test):
    return model.predict_proba(X_test)[:, 1]
```

### 5.5 CatBoost

```python
from catboost import CatBoostClassifier, Pool

def build_catboost_model():
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        border_count=64,
        od_type='Iter',
        od_wait=50,
        use_best_model=True,
        eval_metric='Logloss',
        random_seed=42,
        verbose=100
    )

def train_catboost(X_train, y_train, X_val, y_val):
    """
    注意：CatBoost 不需要 StandardScaler
    use_best_model=True 自動防止過擬合
    """
    model = build_catboost_model()
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=100
    )
    print(f"[CatBoost] Best iteration: {model.best_iteration_}")
    return model

def predict_catboost(model, X_test):
    return model.predict_proba(X_test)[:, 1]
```

### 5.6 完整對比實驗框架

```python
"""
完整對比實驗：RF vs XGBoost vs LightGBM vs CatBoost
"""
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# === 數據準備 ===
def load_and_prepare_data(symbol: str, start: str = '2020-01-01', end: str = '2025-12-31'):
    """載入並準備特徵數據"""
    data = yf.download(symbol, start=start, end=end, auto_adjust=True)
    
    X = calculate_features(data)
    y = calculate_target(data, forward_days=5)
    
    # 對齊並去除 NaN
    combined = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_clean = combined.drop('target', axis=1)
    y_clean = combined['target']
    
    return X_clean, y_clean


def strict_time_split(X, y, val_ratio=0.1, test_ratio=0.2):
    """
    嚴格時序分割：
    [------TRAIN------][--VAL--][--TEST--]
    """
    n = len(X)
    test_idx = int(n * (1 - test_ratio))
    val_idx = int(n * (1 - test_ratio - val_ratio))
    
    X_train = X.iloc[:val_idx]
    y_train = y.iloc[:val_idx]
    X_val   = X.iloc[val_idx:test_idx]
    y_val   = y.iloc[val_idx:test_idx]
    X_test  = X.iloc[test_idx:]
    y_test  = y.iloc[test_idx:]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_sharpe_ratio(y_true, y_pred_proba, prices, threshold=0.55):
    """
    計算策略 Sharpe Ratio
    - 當預測機率 > threshold → 買入
    - 持倉 5 天後賣出
    - 計算日報酬率序列的 Sharpe
    """
    daily_returns = prices.pct_change().dropna()
    
    # 對齊索引
    common_idx = y_true.index.intersection(daily_returns.index)
    y_true = y_true.loc[common_idx]
    y_pred_proba_series = pd.Series(y_pred_proba, index=y_true.index)
    
    # 生成交易信號
    signals = (y_pred_proba_series > threshold).astype(int)
    
    # 策略報酬：有信號時取實際日報酬，否則為0
    strategy_returns = signals * daily_returns.loc[common_idx]
    
    # 去除0值（未交易日）
    active_returns = strategy_returns[signals == 1]
    
    if len(active_returns) < 10:
        return np.nan
    
    # 年化 Sharpe（假設無風險利率 = 0）
    sharpe = (active_returns.mean() / active_returns.std()) * np.sqrt(252)
    return float(sharpe)


def evaluate_model(name, y_true, y_pred_proba, prices, threshold=0.55):
    """評估模型效果"""
    y_pred = (y_pred_proba > threshold).astype(int)
    
    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'sharpe_ratio': calculate_sharpe_ratio(y_true, y_pred_proba, prices, threshold),
        'n_signals': int((y_pred_proba > threshold).sum()),
        'signal_rate': float((y_pred_proba > threshold).mean()),
    }
    return metrics


# === 主實驗流程 ===
def run_comparison_experiment(symbol: str = '2330.TW'):
    """
    執行四模型對比實驗
    """
    print(f"\n{'='*60}")
    print(f"Ensemble Comparison Experiment: {symbol}")
    print(f"{'='*60}\n")
    
    # 1. 準備數據
    X, y = load_and_prepare_data(symbol)
    X_train, X_val, X_test, y_train, y_val, y_test = strict_time_split(X, y)
    
    # 取得測試期的價格（用於 Sharpe 計算）
    raw_data = yf.download(symbol, auto_adjust=True)
    test_prices = raw_data['Close'].loc[X_test.index]
    
    results = []
    
    # 2. RandomForest（需要 StandardScaler）
    print("\n[1/4] Training RandomForest...")
    rf_model, rf_scaler = train_rf(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val])
    )
    rf_proba = predict_rf(rf_model, rf_scaler, X_test)
    results.append(evaluate_model("RandomForest", y_test, rf_proba, test_prices))
    
    # 3. XGBoost
    print("\n[2/4] Training XGBoost...")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    xgb_proba = predict_xgb(xgb_model, X_test)
    results.append(evaluate_model("XGBoost", y_test, xgb_proba, test_prices))
    
    # 4. LightGBM
    print("\n[3/4] Training LightGBM...")
    lgbm_model = train_lgbm(X_train, y_train, X_val, y_val)
    lgbm_proba = predict_lgbm(lgbm_model, X_test)
    results.append(evaluate_model("LightGBM", y_test, lgbm_proba, test_prices))
    
    # 5. CatBoost
    print("\n[4/4] Training CatBoost...")
    cat_model = train_catboost(X_train, y_train, X_val, y_val)
    cat_proba = predict_catboost(cat_model, X_test)
    results.append(evaluate_model("CatBoost", y_test, cat_proba, test_prices))
    
    # 6. 結果彙總
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == '__main__':
    results = run_comparison_experiment('2330.TW')
```

---

## 6. 預期結果與假設

### 6.1 理論預測排名

基於文獻和金融特性分析：

```
預期 Sharpe Ratio 排名：
CatBoost > LightGBM > XGBoost > RandomForest

理由：
- CatBoost：Ordered boosting 防時序洩露，最適合金融時序
- LightGBM：GOSS 捕捉市場異常點，訓練快可做更多 walk-forward
- XGBoost：穩健但無法利用時序特性
- RF：Bagging 方法，缺乏學習前序錯誤的能力
```

### 6.2 準確率預期

```
目前 RF 基準：~75% (on 2330.TW)

保守預估：
- RandomForest：75% ± 2%
- XGBoost：74-78%
- LightGBM：75-79%
- CatBoost：76-80%

注意：準確率不是最重要指標
→ Precision（買入信號的正確率）才是關鍵
→ Sharpe Ratio 才是最終評判標準
```

### 6.3 潛在陷阱

1. **類別不平衡**：如果上漲比例 ≠ 50%，需要調整 `scale_pos_weight`（XGBoost）/ `is_unbalance`（LightGBM）/ `class_weights`（CatBoost）
2. **過短的訓練期**：建議最少用 1000 個交易日（約 4 年）
3. **Look-ahead bias**：確認 `calculate_target()` 使用 `.shift(-forward_days)` 正確避免未來數據洩露
4. **Transaction costs**：Sharpe 計算應考慮手續費（台股約 0.1425% + 0.3% 稅）

---

## 7. 整合進 Stock App 的建議

### 7.1 最小化改動方案

在 `server.py` 中，可以新增 `BoostingPredictor` 類：

```python
# 在 server.py 中新增
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

class BoostingPredictor(BasePredictor):
    """支援 XGBoost / LightGBM / CatBoost 的通用預測器"""
    
    def __init__(self, model_type='catboost', forward_days=5, confidence_threshold=0.55):
        super().__init__()
        self.model_type = model_type
        self.forward_days = forward_days
        self.confidence_threshold = confidence_threshold
        self.model = self._build_model()
        self.is_trained = False
        self.feature_names = []
    
    def _build_model(self):
        if self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric='logloss', early_stopping_rounds=50,
                random_state=42, n_jobs=-1
            )
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=300, num_leaves=31, max_depth=6,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, verbose=-1,
                random_state=42, n_jobs=-1
            )
        else:  # catboost（默認，推薦）
            return CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, od_type='Iter', od_wait=50,
                use_best_model=True, verbose=0, random_seed=42
            )
    
    # _calculate_features 方法與 RandomForestPredictor 相同
    # predict 方法結構也相同，只需修改 scaler 部分
    # (Boosting 模型不需要 StandardScaler)
```

### 7.2 安裝依賴

```bash
pip install xgboost lightgbm catboost
# 或更新 requirements.txt：
# xgboost>=2.0.0
# lightgbm>=4.0.0
# catboost>=1.2.0
```

---

## 8. 實驗優先順序建議

1. **第一步**：先在 Jupyter/腳本跑完整對比實驗（用本文件的代碼）
2. **第二步**：確認 CatBoost/LightGBM 在 2330.TW 上確實超越 RF
3. **第三步**：整合最佳模型進 `server.py` 的 `BoostingPredictor`
4. **第四步**：用 Walk-Forward 驗證穩定性（不只是單一測試集）
5. **第五步**：考慮 Stacking Ensemble（RF + CatBoost + LightGBM 三模型集成）

---

## 參考資料

- Shwartz-Ziv & Armon (2021). "Tabular Data: Deep Learning is Not All You Need." arXiv:2106.03253
- XGBoost Parameter Tuning Guide: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- LightGBM Parameters-Tuning: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- CatBoost Parameter Tuning: https://catboost.ai/docs/en/concepts/parameter-tuning
- "CatBoost vs. LightGBM vs. XGBoost" - Towards Data Science (2023)

---

*生成時間: 2026-02-17 | 作者: Bythos (Sub-agent: ensemble-comparison)*

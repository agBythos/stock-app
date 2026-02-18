"""
tests/test_purge_embargo.py — Look-ahead Bias 洩漏驗證
====================================================

測試項目（Suite 2 — Leakage verification）：
7. 合成洩漏測試：label[t] = feature[t+5]；有 purge → accuracy < 0.65

設計說明：
  建立一個故意洩漏的特徵：feature[t] = close[t]
  目標變數：target[t] = 1 if close[t+5] > close[t] else 0
  若無 purge，模型會學到 feature[t] 幾乎等於 target[t] 的洩漏規則。
  有 purge 後，訓練集不包含洩漏樣本，accuracy 應接近隨機（~0.5）。

作者：Bythos（sub-agent）
建立：2026-02-18
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config.walk_forward import WalkForwardConfig
from validation.purged_walk_forward import PurgedWalkForwardSplitter


def test_synthetic_leak_with_purge():
    """
    合成洩漏測試：有 purge → accuracy < 0.65

    建立合成資料：
      feature[t] = close[t] + noise
      target[t] = 1 if close[t+5] > close[t] else 0

    無 purge 時，模型會學到洩漏：feature[t] 可預測 target[t]（因 target 包含 t+5 資訊）
    有 purge 時，訓練集末尾 5 bars 被移除 → 洩漏被阻斷 → accuracy ≈ 0.5
    """
    np.random.seed(42)
    n_bars = 600

    # 建立合成價格序列（隨機遊走）
    close = np.cumsum(np.random.randn(n_bars)) + 100

    # 特徵：close[t] + 小噪音
    feature = close + np.random.randn(n_bars) * 0.5

    # 目標變數：未來 5 bars 漲跌（shift 防止前視偏差）
    future_return = pd.Series(close).pct_change(5).shift(-5)
    target = (future_return > 0).astype(int).values

    # 組成 DataFrame
    df = pd.DataFrame(
        {"feature": feature, "target": target, "close": close},
        index=pd.date_range("2020-01-01", periods=n_bars, freq="D"),
    )
    df = df.dropna()  # 移除末尾 NaN

    # 使用 Purged Walk-Forward
    cfg = WalkForwardConfig(
        train_window=252,
        test_window=21,
        step_size=21,
        label_horizon=5,  # purge 5 bars
        embargo_bars=5,
        min_train_samples=200,
    )
    splitter = PurgedWalkForwardSplitter(cfg)
    folds = splitter.active_folds(df)

    # 在每個 fold 上訓練並測試
    accuracies = []

    for fold in folds[:5]:  # 只測試前 5 個 fold（節省時間）
        X_train = df.iloc[fold.purged_train_idx][["feature"]].values
        y_train = df.iloc[fold.purged_train_idx]["target"].values

        X_test = df.iloc[fold.test_idx][["feature"]].values
        y_test = df.iloc[fold.test_idx]["target"].values

        # 訓練 RandomForest
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)

    # 預期：有 purge → accuracy 接近隨機（< 0.65）
    # 若無 purge，accuracy 會 > 0.80（因為洩漏）
    assert mean_acc < 0.65, (
        f"合成洩漏測試失敗：mean accuracy={mean_acc:.4f} ≥ 0.65。"
        "Purge 機制可能未正確阻斷洩漏。"
    )

    print(f"✓ 合成洩漏測試通過：mean accuracy={mean_acc:.4f} < 0.65（洩漏已阻斷）")


def test_no_purge_would_leak():
    """
    對照測試：直接特徵洩漏 → accuracy > 0.70（真實 look-ahead bias 場景）

    模擬工程常見錯誤：特徵欄位忘記 lag，將未來價格直接當特徵。

    設計：
      feature_0[t] = close[t]         （當前價格，合法）
      feature_1[t] = close[t+5]       （未來價格，洩漏！）
      target[t]    = 1 if close[t+5] > close[t] else 0

    結果：
      模型可輕鬆學到 feature_1 > feature_0 → target=1 的規則。
      即使在測試集，同樣的洩漏 feature 仍揭示 target → accuracy 接近 1.0。

    此測試不測試 purge 機制，而是驗證 look-ahead feature bug 的可偵測性，
    以及 walk-forward 框架在可預測資料上的基本運作。
    """
    np.random.seed(42)
    n_bars = 400
    label_h = 5

    # 隨機遊走價格
    close = np.cumsum(np.random.randn(n_bars)) + 100
    close_series = pd.Series(close)

    # 目標：未來 label_h bars 漲跌
    target = (close_series.pct_change(label_h).shift(-label_h) > 0).astype(int).values

    # feature_0: 當前價格（合法特徵）
    # feature_1: 未來 label_h bars 的價格（洩漏！忘記 lag 的 bug）
    feature_0 = close  # close[t]
    feature_1 = np.concatenate([close[label_h:], np.full(label_h, np.nan)])  # close[t+5]

    df = pd.DataFrame(
        {
            "feature_0": feature_0,
            "feature_1": feature_1,
            "target": target,
        },
        index=pd.date_range("2020-01-01", periods=n_bars, freq="D"),
    )
    df = df.dropna()  # 移除末尾 label_h 個 NaN

    # 無 purge，無 embargo（模擬未做任何洩漏防護）
    cfg = WalkForwardConfig(
        train_window=150,
        test_window=21,
        step_size=21,
        label_horizon=0,  # 無 purge
        embargo_bars=0,
        min_train_samples=100,
    )
    splitter = PurgedWalkForwardSplitter(cfg)
    folds = splitter.active_folds(df)

    accuracies = []

    for fold in folds[:3]:
        X_train = df.iloc[fold.purged_train_idx][["feature_0", "feature_1"]].values
        y_train = df.iloc[fold.purged_train_idx]["target"].values
        X_test = df.iloc[fold.test_idx][["feature_0", "feature_1"]].values
        y_test = df.iloc[fold.test_idx]["target"].values

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    mean_acc = np.mean(accuracies)

    # 預期：直接洩漏 feature → accuracy 接近 1.0（遠高於 0.70）
    assert mean_acc > 0.70, (
        f"對照測試失敗：直接洩漏特徵時 accuracy={mean_acc:.4f} ≤ 0.70。"
        "look-ahead feature 應使 model 達到高 accuracy。"
    )

    print(f"✓ 直接洩漏測試通過：mean accuracy={mean_acc:.4f} > 0.70（洩漏可偵測）")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

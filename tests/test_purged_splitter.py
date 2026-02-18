"""
tests/test_purged_splitter.py — Purged Walk-Forward Splitter 測試
================================================================

測試項目（Suite 1 — Splitter correctness）：
1. 訓練/測試集無重疊（purge 後）
2. Purged indices 的標籤不會洩漏到測試期
3. train_end 與 test_start 之間 gap ≥ embargo_bars
4. 低於 min_train_samples 的 fold 會被跳過
5. 資料不足會拋出 ValueError
6. 735-bar 資料集生成約 20 個 folds（252 train / 21 step）

作者：Bythos（sub-agent）
建立：2026-02-18
"""

import numpy as np
import pandas as pd
import pytest

from config.walk_forward import WalkForwardConfig
from validation.purged_walk_forward import PurgedWalkForwardSplitter


# ============================================================================
# Test 1: 訓練/測試無重疊
# ============================================================================


def test_no_train_test_overlap():
    """Purge 後訓練集與測試集無任何重疊"""
    cfg = WalkForwardConfig(
        train_window=252,
        test_window=21,
        step_size=21,
        label_horizon=5,
        embargo_bars=5,
        min_train_samples=200,
    )
    splitter = PurgedWalkForwardSplitter(cfg)

    # 建立假資料
    n_bars = 735  # 約 3 年資料
    df = pd.DataFrame(
        {"close": np.random.randn(n_bars)},
        index=pd.date_range("2020-01-01", periods=n_bars, freq="D"),
    )

    folds = splitter.split(df)

    for fold in folds:
        if fold.skipped:
            continue
        train_set = set(fold.purged_train_idx)
        test_set = set(fold.test_idx)
        overlap = train_set & test_set
        assert len(overlap) == 0, f"Fold {fold.fold_id}: 訓練/測試重疊 {len(overlap)} 個樣本"


# ============================================================================
# Test 2: Purged indices 不洩漏標籤
# ============================================================================


def test_purged_indices_no_leakage():
    """Purge boundary 確保訓練集標籤不包含測試期資料"""
    cfg = WalkForwardConfig(label_horizon=5, embargo_bars=5)
    splitter = PurgedWalkForwardSplitter(cfg)

    df = pd.DataFrame(
        {"close": np.random.randn(735)},
        index=pd.date_range("2020-01-01", periods=735, freq="D"),
    )

    folds = splitter.split(df)

    for fold in folds:
        if fold.skipped:
            continue
        # purged_train_idx 的最大值必須 < purge_boundary
        assert fold.purged_train_idx.max() < fold.purge_boundary, (
            f"Fold {fold.fold_id}: purged_train_idx 超出 purge_boundary"
        )
        # purge_boundary 必須 ≤ train_end_bar - label_horizon
        expected_purge = fold.train_end_bar - cfg.label_horizon
        assert fold.purge_boundary == expected_purge, (
            f"Fold {fold.fold_id}: purge_boundary={fold.purge_boundary}, "
            f"expected={expected_purge}"
        )


# ============================================================================
# Test 3: Embargo gap 檢查
# ============================================================================


def test_embargo_gap():
    """測試集起始位置與 purge boundary 之間 gap ≥ embargo_bars"""
    cfg = WalkForwardConfig(label_horizon=5, embargo_bars=5)
    splitter = PurgedWalkForwardSplitter(cfg)

    df = pd.DataFrame(
        {"close": np.random.randn(735)},
        index=pd.date_range("2020-01-01", periods=735, freq="D"),
    )

    folds = splitter.split(df)

    for fold in folds:
        if fold.skipped:
            continue
        gap = fold.test_start_bar - fold.purge_boundary
        assert gap >= cfg.embargo_bars, (
            f"Fold {fold.fold_id}: gap={gap} < embargo_bars={cfg.embargo_bars}"
        )


# ============================================================================
# Test 4: min_train_samples 跳過機制
# ============================================================================


def test_min_train_samples_skip():
    """Purge 後訓練樣本 < min_train_samples 的 fold 會被跳過"""
    cfg = WalkForwardConfig(
        train_window=100,  # 小窗口
        test_window=21,
        label_horizon=5,
        embargo_bars=5,
        min_train_samples=200,  # 高門檻（purge 後 < 100）
    )
    splitter = PurgedWalkForwardSplitter(cfg)

    df = pd.DataFrame(
        {"close": np.random.randn(500)},
        index=pd.date_range("2020-01-01", periods=500, freq="D"),
    )

    folds = splitter.split(df)

    # 所有 fold 都應該被跳過（train=100, purge=5 → 95 < 200）
    assert all(f.skipped for f in folds), "所有 fold 應被跳過（樣本不足）"


# ============================================================================
# Test 5: 資料不足拋出錯誤
# ============================================================================


def test_insufficient_data_raises():
    """資料不足以生成任何 fold 時拋出 ValueError"""
    cfg = WalkForwardConfig(
        train_window=252,
        test_window=21,
        label_horizon=5,
        embargo_bars=5,
    )
    splitter = PurgedWalkForwardSplitter(cfg)

    # 只有 100 bars（< train + gap + test = 252+10+21 = 283）
    df = pd.DataFrame(
        {"close": np.random.randn(100)},
        index=pd.date_range("2020-01-01", periods=100, freq="D"),
    )

    with pytest.raises(ValueError, match="資料不足"):
        splitter.split(df)


# ============================================================================
# Test 6: 735-bar 資料集生成約 20 folds
# ============================================================================


def test_expected_fold_count():
    """735 bars 資料集應生成約 20 個 folds（252 train / 21 step）"""
    cfg = WalkForwardConfig(
        train_window=252,
        test_window=21,
        step_size=21,
        label_horizon=5,
        embargo_bars=5,
        min_train_samples=200,
    )
    splitter = PurgedWalkForwardSplitter(cfg)

    df = pd.DataFrame(
        {"close": np.random.randn(735)},
        index=pd.date_range("2020-01-01", periods=735, freq="D"),
    )

    folds = splitter.split(df)
    active_folds = [f for f in folds if not f.skipped]

    # 理論計算：
    # first test_start = 252 + 10 = 262
    # last test_start ≤ 735 - 21 = 714
    # n_folds = (714 - 262) / 21 + 1 ≈ 22
    assert len(folds) >= 18 and len(folds) <= 24, (
        f"預期 18–24 個 folds，實際 {len(folds)}"
    )
    assert len(active_folds) >= 15, f"預期 ≥15 個 active folds，實際 {len(active_folds)}"


# ============================================================================
# Test 7: info() 方法
# ============================================================================


def test_splitter_info():
    """測試 info() 方法回傳正確的 metadata"""
    cfg = WalkForwardConfig()
    splitter = PurgedWalkForwardSplitter(cfg)

    info = splitter.info(735)
    assert "n_total_folds" in info
    assert "estimated_active_folds" in info
    assert info["n_bars"] == 735

    # 資料不足情況
    info_small = splitter.info(100)
    assert "error" in info_small
    assert info_small["n_total_folds"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

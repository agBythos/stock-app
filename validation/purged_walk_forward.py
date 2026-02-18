"""
validation/purged_walk_forward.py — Purged Walk-Forward Splitter
================================================================

實作 Marcos Lopez de Prado 的 Purged Walk-Forward 方法。

核心概念：
  - Purge：移除訓練集末尾標籤覆蓋測試期的樣本（防止前視偏差）
  - Embargo：在 purge 後額外排除 N bars，防止特徵自相關洩漏
  - 嚴格時序：訓練集任何樣本的時間均 < 測試集任何樣本的時間

關鍵保護：
  目標變數必須在訓練前 shift（在外部計算）：
    target = close.pct_change(label_horizon).shift(-label_horizon) > 0
  本模組假設 target 已正確 shift，並根據 label_horizon 進行 purge。

設計說明：
  - 以 integer position（bar index）操作，不依賴日期
  - 支援任意 pandas DataFrame（OHLCV + features + target）
  - 回傳 WalkForwardFold 物件列表，可直接用於訓練/測試

作者：Bythos（sub-agent）
建立：2026-02-18
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.walk_forward import WalkForwardConfig


# ============================================================================
# WalkForwardFold — 單一 fold 資料容器
# ============================================================================


@dataclass
class WalkForwardFold:
    """
    單一 Walk-Forward Fold

    包含訓練/測試的 integer index（相對於原始 DataFrame）。
    purged_train_idx 已排除被 purge 的樣本；embargo 邊界已保留在 gap 中。

    Attributes
    ----------
    fold_id          : fold 序號（1-indexed）
    train_idx        : 訓練集原始 indices（purge 前）
    purged_train_idx : 訓練集 indices（purge + embargo 後）
    test_idx         : 測試集 indices
    train_start_bar  : 訓練集起始 bar position
    train_end_bar    : 訓練集結束 bar position（exclusive，即 purge 前）
    test_start_bar   : 測試集起始 bar position
    test_end_bar     : 測試集結束 bar position（exclusive）
    purge_boundary   : purge 邊界（訓練集末尾 N bars 被移除）
    embargo_end      : embargo 結束位置（bar，此位置之後才是測試集）
    n_train_raw      : purge 前訓練樣本數
    n_train_purged   : purge 後訓練樣本數
    n_test           : 測試樣本數
    skipped          : True 若因樣本不足而跳過此 fold
    skip_reason      : 跳過原因說明
    """

    fold_id: int
    train_idx: np.ndarray          # purge 前
    purged_train_idx: np.ndarray   # purge + embargo 後
    test_idx: np.ndarray

    train_start_bar: int
    train_end_bar: int             # exclusive（purge 前的終點）
    test_start_bar: int
    test_end_bar: int              # exclusive

    purge_boundary: int            # 從此 bar 開始被 purge（inclusive）
    embargo_end: int               # embargo 結束位置

    n_train_raw: int = 0
    n_train_purged: int = 0
    n_test: int = 0

    skipped: bool = False
    skip_reason: str = ""

    def __post_init__(self):
        self.n_train_raw = len(self.train_idx)
        self.n_train_purged = len(self.purged_train_idx)
        self.n_test = len(self.test_idx)

    @property
    def gap_bars(self) -> int:
        """測試集起始與 purge 邊界之間的 gap"""
        return self.test_start_bar - self.purge_boundary

    def summary(self) -> str:
        """單行摘要字串"""
        if self.skipped:
            return (
                f"Fold {self.fold_id:3d} [SKIPPED] reason={self.skip_reason}"
            )
        return (
            f"Fold {self.fold_id:3d} | "
            f"train=[{self.train_start_bar},{self.train_end_bar}) "
            f"→ purged=[{self.train_start_bar},{self.purge_boundary}) "
            f"n_purged={self.n_train_purged} | "
            f"embargo=[{self.purge_boundary},{self.embargo_end}) | "
            f"test=[{self.test_start_bar},{self.test_end_bar}) "
            f"n_test={self.n_test}"
        )


# ============================================================================
# SplitEngine — 生成原始窗口邊界
# ============================================================================


class SplitEngine:
    """
    生成 Walk-Forward 窗口邊界（bar positions）

    不做任何 purge/embargo，只負責計算每個 fold 的
    (train_start, train_end, test_start, test_end)。
    """

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config

    def generate_boundaries(
        self, n_bars: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        生成所有 fold 的邊界列表

        Args:
            n_bars: 資料集總 bar 數

        Returns:
            List of (train_start, train_end, test_start, test_end)
            所有值為 integer bar positions（0-indexed, exclusive end）

        Raises:
            ValueError: 資料不足以生成任何 fold
        """
        cfg = self.cfg
        min_required = cfg.train_window + cfg.total_gap + cfg.test_window

        if n_bars < min_required:
            raise ValueError(
                f"資料不足：需要至少 {min_required} bars "
                f"(train={cfg.train_window} + gap={cfg.total_gap} + test={cfg.test_window}), "
                f"但只有 {n_bars} bars"
            )

        boundaries = []
        test_start = cfg.train_window + cfg.total_gap

        while test_start + cfg.test_window <= n_bars:
            train_end = test_start - cfg.total_gap
            train_start = train_end - cfg.train_window

            if train_start < 0:
                break

            test_end = test_start + cfg.test_window

            boundaries.append((train_start, train_end, test_start, test_end))
            test_start += cfg.step_size

        return boundaries


# ============================================================================
# PurgeEngine — 應用 Purge + Embargo
# ============================================================================


class PurgeEngine:
    """
    對訓練集索引應用 Purge + Embargo

    Purge 邏輯：
      移除訓練集末尾 label_horizon 個 bars 的樣本。
      這些樣本的目標標籤（forward return）包含測試期的資料，
      若不移除會導致前視偏差。

    Embargo 邏輯：
      在 purge 邊界後，額外保留 embargo_bars 個 bars 不用於訓練，
      防止特徵中的自相關（例如移動平均）帶入洩漏。

    注意：
      embargo_bars 在 SplitEngine.generate_boundaries 中已計入 total_gap，
      因此 test_start_bar 本身已是 embargo 後的安全位置。
      PurgeEngine 只需從 train_end_bar 往回移除 label_horizon 個 bars。
    """

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config

    def purge(
        self,
        train_idx: np.ndarray,
        train_end_bar: int,
    ) -> Tuple[np.ndarray, int]:
        """
        對訓練集索引應用 purge

        Args:
            train_idx    : 訓練集 bar positions（integer array）
            train_end_bar: 訓練集末尾位置（exclusive）

        Returns:
            (purged_idx, purge_boundary)
            purge_boundary: 從此 bar position 開始被移除（inclusive）
        """
        purge_boundary = train_end_bar - self.cfg.label_horizon
        purged_idx = train_idx[train_idx < purge_boundary]
        return purged_idx, purge_boundary


# ============================================================================
# PurgedWalkForwardSplitter — 主介面
# ============================================================================


class PurgedWalkForwardSplitter:
    """
    Purged Walk-Forward Splitter

    整合 SplitEngine + PurgeEngine，對 DataFrame 生成完整的 fold 列表。

    使用範例::

        from config.walk_forward import WalkForwardConfig
        from validation.purged_walk_forward import PurgedWalkForwardSplitter

        cfg = WalkForwardConfig()
        splitter = PurgedWalkForwardSplitter(cfg)
        folds = splitter.split(df)

        for fold in folds:
            print(fold.summary())
            X_train = df.iloc[fold.purged_train_idx][feature_cols]
            y_train = df.iloc[fold.purged_train_idx]["target"]
            X_test  = df.iloc[fold.test_idx][feature_cols]
            y_test  = df.iloc[fold.test_idx]["target"]

    防洩漏保證：
      1. 訓練集任何 index < purge_boundary（移除了包含測試期標籤的樣本）
      2. purge_boundary 到 test_start_bar 之間為 embargo（不使用）
      3. 測試集 index ≥ test_start_bar（嚴格晚於訓練集）
      4. train_idx ∩ test_idx = ∅（強制驗證）
    """

    def __init__(self, config: WalkForwardConfig):
        self.cfg = config
        self._split_engine = SplitEngine(config)
        self._purge_engine = PurgeEngine(config)

    def split(
        self, df: pd.DataFrame, verbose: bool = False
    ) -> List[WalkForwardFold]:
        """
        生成 Purged Walk-Forward fold 列表

        Args:
            df     : 包含 features + target 的 DataFrame（時間序列，已排序）
            verbose: 是否印出每個 fold 的摘要

        Returns:
            List[WalkForwardFold]（skipped fold 也包含在內，skipped=True）

        Raises:
            ValueError: 資料不足以生成任何 fold
        """
        n_bars = len(df)

        # 生成邊界
        boundaries = self._split_engine.generate_boundaries(n_bars)

        folds: List[WalkForwardFold] = []

        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(
            boundaries, start=1
        ):
            # 原始訓練/測試索引
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            # 應用 purge
            purged_train_idx, purge_boundary = self._purge_engine.purge(
                train_idx, train_end
            )

            # embargo 結束位置（在 total_gap 設計中已被納入）
            embargo_end = test_start

            # 驗證無重疊（paranoid check）
            overlap = np.intersect1d(purged_train_idx, test_idx)
            if len(overlap) > 0:
                raise RuntimeError(
                    f"Fold {fold_id}: 訓練/測試集重疊 {len(overlap)} 個樣本！"
                    "這不應該發生，請檢查 SplitEngine 邏輯。"
                )

            # 檢查最小訓練樣本數
            n_purged = len(purged_train_idx)
            skipped = False
            skip_reason = ""

            if n_purged < self.cfg.min_train_samples:
                skipped = True
                skip_reason = (
                    f"purged train size={n_purged} < "
                    f"min_train_samples={self.cfg.min_train_samples}"
                )

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_idx=train_idx,
                purged_train_idx=purged_train_idx,
                test_idx=test_idx,
                train_start_bar=train_start,
                train_end_bar=train_end,
                test_start_bar=test_start,
                test_end_bar=test_end,
                purge_boundary=purge_boundary,
                embargo_end=embargo_end,
                skipped=skipped,
                skip_reason=skip_reason,
            )

            if verbose:
                print(fold.summary())

            folds.append(fold)

        return folds

    def active_folds(self, df: pd.DataFrame, **kwargs) -> List[WalkForwardFold]:
        """只回傳未跳過的 fold"""
        return [f for f in self.split(df, **kwargs) if not f.skipped]

    def info(self, n_bars: int) -> dict:
        """
        回傳給定資料長度下的 fold 統計資訊

        Args:
            n_bars: 資料集 bar 數

        Returns:
            dict 含 n_total_folds, n_active_folds, etc.
        """
        try:
            boundaries = self._split_engine.generate_boundaries(n_bars)
        except ValueError as e:
            return {"error": str(e), "n_total_folds": 0}

        n_total = len(boundaries)
        # 估算 active folds（假設每個 fold purge 後仍有足夠樣本）
        estimated_active = sum(
            1
            for (ts, te, _, _) in boundaries
            if (te - ts) - self.cfg.label_horizon >= self.cfg.min_train_samples
        )

        return {
            "n_bars": n_bars,
            "config": self.cfg.to_dict(),
            "n_total_folds": n_total,
            "estimated_active_folds": estimated_active,
            "first_fold": {
                "train_start": boundaries[0][0],
                "train_end": boundaries[0][1],
                "test_start": boundaries[0][2],
                "test_end": boundaries[0][3],
            }
            if boundaries
            else None,
            "last_fold": {
                "train_start": boundaries[-1][0],
                "train_end": boundaries[-1][1],
                "test_start": boundaries[-1][2],
                "test_end": boundaries[-1][3],
            }
            if boundaries
            else None,
        }

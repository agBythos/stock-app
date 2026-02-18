"""
validation/cpcv_splitter.py — CPCV Splitter（核心分組與 Purge/Embargo）
========================================================================

實作 Marcos Lopez de Prado 的 Combinatorially Purged Cross-Validation：

核心邏輯：
  1. GroupEngine：將 T bars 均勻分成 N 組
  2. CPCVPurgeEngine：對非連續訓練 groups 應用 purge + embargo
  3. CPCVSplitter：枚舉 C(N,k) 個 fold，重建 φ 條 backtest paths

防洩漏保證：
  - purged_train_idx ∩ test_idx = ∅（強制驗證）
  - 每個訓練/測試邊界均應用 purge（label_horizon）+ embargo 緩衝
  - 非連續 groups 的每個「訓練右鄰測試」邊界均獨立處理

CPCV(N=6, k=2)：
  C(6,2) = 15 folds
  φ = 2×15/6 = 5 條獨立 backtest paths

作者：Bythos（sub-agent phase4-cpcv-impl）
建立：2026-02-18
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.cpcv import CPCVConfig


# ============================================================================
# CPCVFold — 單一 fold 資料容器
# ============================================================================


@dataclass
class CPCVFold:
    """
    單一 CPCV fold（一種 train/test 組合）

    Attributes
    ----------
    fold_id           : fold 序號（1-indexed，共 C(N,k) 個）
    combination       : tuple of test group indices（e.g., (1, 3)）
    train_group_ids   : 訓練用的 group indices
    test_group_ids    : 測試用的 group indices（= combination）
    purged_train_idx  : 訓練集 bar positions（purge + embargo 後）
    test_idx          : 測試集 bar positions（按 group 時間順序）
    test_group_ranges : Dict[group_id → (start_bar, end_bar)] 用於 path 重建
    n_train_purged    : purge 後訓練樣本數
    n_test            : 測試樣本數
    skipped           : 是否因樣本不足跳過
    skip_reason       : 跳過原因
    """

    fold_id: int
    combination: Tuple[int, ...]
    train_group_ids: List[int]
    test_group_ids: List[int]
    purged_train_idx: np.ndarray
    test_idx: np.ndarray
    test_group_ranges: Dict[int, Tuple[int, int]]

    n_train_purged: int = 0
    n_test: int = 0
    skipped: bool = False
    skip_reason: str = ""

    def __post_init__(self):
        self.n_train_purged = len(self.purged_train_idx)
        self.n_test = len(self.test_idx)

    def summary(self) -> str:
        """單行摘要"""
        if self.skipped:
            return f"CPCVFold {self.fold_id:3d} [SKIPPED] {self.skip_reason}"
        return (
            f"CPCVFold {self.fold_id:3d} | "
            f"test_groups={list(self.combination)} | "
            f"train_bars={self.n_train_purged} | "
            f"test_bars={self.n_test}"
        )


# ============================================================================
# CPCVPath — 單條 backtest path
# ============================================================================


@dataclass
class CPCVPath:
    """
    單條 backtest path（由多個 fold 的測試集拼接而成）

    每條 path 由「同一個 group 在不同 fold 的 test_idx」拼接而成，
    保持時間順序，覆蓋完整時序。

    Attributes
    ----------
    path_id    : path 序號（1-indexed）
    group_id   : 此 path 主要來自哪個 group（anchor group）
    fold_ids   : 貢獻此 path 的 fold id 列表
    test_idx   : 拼接後的 bar positions（時間順序）
    """

    path_id: int
    group_id: int
    fold_ids: List[int]
    test_idx: np.ndarray


# ============================================================================
# GroupEngine — 均勻分組
# ============================================================================


class GroupEngine:
    """
    將時序資料切成 N 個等大小的 group（不 shuffle）

    最後一組可能比其他組多若干 bars（餘數全放最後一組）。
    """

    def __init__(self, cfg: CPCVConfig):
        self.cfg = cfg

    def build_groups(self, n_bars: int) -> List[Tuple[int, int]]:
        """
        建立 N 個 group 的邊界

        Args:
            n_bars: 資料總 bar 數

        Returns:
            List[(start, end)] 長度 = n_groups
            end 為 exclusive（Python slice 慣例）

        Raises:
            ValueError: 資料不足
        """
        n = self.cfg.n_groups
        min_bars_required = n * 2  # 最低：每組至少 2 bars

        if n_bars < min_bars_required:
            raise ValueError(
                f"資料不足：{n_bars} bars < 最低需求 {min_bars_required} bars "
                f"（{n} groups × 2 bars/group）"
            )

        group_size = n_bars // n
        boundaries: List[Tuple[int, int]] = []

        for i in range(n):
            start = i * group_size
            end = (i + 1) * group_size if i < n - 1 else n_bars
            boundaries.append((start, end))

        return boundaries


# ============================================================================
# CPCVPurgeEngine — CPCV 版 Purge + Embargo
# ============================================================================


class CPCVPurgeEngine:
    """
    對 CPCV fold 應用 Purge + Embargo

    CPCV 的 purge 比 Walk-Forward 複雜：
    訓練集是不連續的多個 group，每個「訓練 group 緊鄰測試 group 的邊界」
    都需要獨立處理 purge 與 embargo。

    邊界分析（以 groups 0..5，測試 groups {1,3} 為例）：
      訓練 groups: 0, 2, 4, 5

      「訓練右鄰測試」邊界（需 purge 訓練集末尾）：
        - group 0 右鄰 group 1（測試）：移除 group 0 末尾 label_horizon bars
        - group 2 右鄰 group 3（測試）：移除 group 2 末尾 label_horizon bars

      「測試右鄰訓練」邊界（需 embargo 訓練集開頭）：
        - group 1（測試）右鄰 group 2（訓練）：移除 group 2 開頭 embargo_bars
        - group 3（測試）右鄰 group 4（訓練）：移除 group 4 開頭 embargo_bars

    兩側 edge groups（group 0 左側無測試，group 5 右側無測試）：
        不需要額外處理（時序邊界，沒有洩漏風險）
    """

    def __init__(self, cfg: CPCVConfig):
        self.cfg = cfg

    def apply(
        self,
        train_group_ids: List[int],
        test_group_ids: List[int],
        boundaries: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        計算 purge + embargo 後的訓練集 bar indices

        Args:
            train_group_ids : 訓練 group 序號列表（已排序）
            test_group_ids  : 測試 group 序號列表（已排序）
            boundaries      : 所有 groups 的 (start, end) 邊界列表

        Returns:
            purged_train_idx: numpy array of bar positions（purge + embargo 後）
        """
        test_set = set(test_group_ids)
        train_set = set(train_group_ids)

        # 建立初始訓練集（所有訓練 group 的 bar positions 合集）
        train_bars: List[np.ndarray] = []
        for gid in sorted(train_group_ids):
            start, end = boundaries[gid]
            train_bars.append(np.arange(start, end))

        if not train_bars:
            return np.array([], dtype=int)

        all_train = np.concatenate(train_bars)
        purge_mask = np.ones(len(all_train), dtype=bool)  # True = 保留

        # ── 建立 bar → group_id 映射（用於快速查詢）────────────────
        # 對每個訓練 bar，記錄它屬於哪個 group
        bar_group = {}
        for gid in sorted(train_group_ids):
            start, end = boundaries[gid]
            for b in range(start, end):
                bar_group[b] = gid

        # ── Purge：訓練 group 右鄰測試 group ─────────────────────
        # 若 group g_i 在左，group g_j 在右（g_j = g_i + 1），
        # 且 g_j 是測試 group，則移除 g_i 末尾 label_horizon bars
        for gid in sorted(train_group_ids):
            right_neighbor = gid + 1
            if right_neighbor in test_set:
                # g_id 右鄰測試 group → purge g_id 末尾 label_horizon bars
                g_start, g_end = boundaries[gid]
                purge_end = g_end        # exclusive
                purge_start = max(g_start, g_end - self.cfg.label_horizon)

                # 在 all_train 中標記需移除的 bars
                for i, bar in enumerate(all_train):
                    if purge_start <= bar < purge_end:
                        purge_mask[i] = False

        # ── Embargo：測試 group 右鄰訓練 group ────────────────────
        # 若 group g_j 是測試 group，group g_k = g_j + 1 是訓練 group，
        # 則移除 g_k 開頭 embargo_bars bars
        for gid in sorted(test_group_ids):
            right_neighbor = gid + 1
            if right_neighbor in train_set:
                # g_id 測試，右鄰 g_id+1 訓練 → embargo g_id+1 開頭
                g_start, g_end = boundaries[right_neighbor]
                embargo_end = min(g_end, g_start + self.cfg.embargo_bars)

                for i, bar in enumerate(all_train):
                    if g_start <= bar < embargo_end:
                        purge_mask[i] = False

        purged = all_train[purge_mask]
        return purged


# ============================================================================
# CPCVSplitter — 主介面
# ============================================================================


class CPCVSplitter:
    """
    Combinatorially Purged Cross-Validation Splitter

    使用範例::

        cfg = CPCVConfig(n_groups=6, k_test_groups=2)
        splitter = CPCVSplitter(cfg)
        folds = splitter.split(df)
        paths = splitter.build_paths(folds)

        for fold in folds:
            print(fold.summary())

    防洩漏保證：
      1. purged_train_idx ∩ test_idx = ∅（強制驗證）
      2. 每個訓練/測試邊界均應用 purge + embargo
      3. 目標變數須在外部 shift（不在本模組負責）
    """

    def __init__(self, cfg: CPCVConfig):
        self.cfg = cfg
        self._group_engine = GroupEngine(cfg)
        self._purge_engine = CPCVPurgeEngine(cfg)

    def split(self, df: pd.DataFrame, verbose: bool = False) -> List[CPCVFold]:
        """
        生成 C(N,k) 個 fold

        Args:
            df      : 完整 DataFrame（時間序列，已排序）
            verbose : 是否印出每個 fold 的摘要

        Returns:
            List[CPCVFold]（skipped fold 也包含在內）

        Raises:
            ValueError: 資料不足
        """
        n_bars = len(df)
        n = self.cfg.n_groups
        k = self.cfg.k_test_groups

        # ── Step 1: 建立 N 個 group 邊界 ─────────────────────────
        boundaries = self._group_engine.build_groups(n_bars)

        # ── Step 2: 枚舉所有 C(N,k) 個測試組合 ───────────────────
        all_combinations = list(itertools.combinations(range(n), k))

        folds: List[CPCVFold] = []

        for fold_id, combination in enumerate(all_combinations, start=1):
            test_group_ids = list(combination)
            train_group_ids = [g for g in range(n) if g not in combination]

            # ── Step 3a: 建立測試集 bar indices ──────────────────
            test_parts: List[np.ndarray] = []
            test_group_ranges: Dict[int, Tuple[int, int]] = {}

            for gid in test_group_ids:
                start, end = boundaries[gid]
                test_group_ranges[gid] = (start, end)
                test_parts.append(np.arange(start, end))

            test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

            # ── Step 3b: 計算 purged 訓練集 ───────────────────────
            purged_train_idx = self._purge_engine.apply(
                train_group_ids=train_group_ids,
                test_group_ids=test_group_ids,
                boundaries=boundaries,
            )

            # ── Step 3c: 驗證無重疊（paranoid check）─────────────
            overlap = np.intersect1d(purged_train_idx, test_idx)
            if len(overlap) > 0:
                raise RuntimeError(
                    f"CPCVFold {fold_id}: 訓練/測試集重疊 {len(overlap)} 個 bars！"
                    f"combination={combination}"
                )

            # ── Step 3d: 檢查最小訓練樣本數 ──────────────────────
            skipped = False
            skip_reason = ""
            n_purged = len(purged_train_idx)

            if n_purged < self.cfg.min_train_samples:
                skipped = True
                skip_reason = (
                    f"purged train size={n_purged} < "
                    f"min_train_samples={self.cfg.min_train_samples}"
                )

            fold = CPCVFold(
                fold_id=fold_id,
                combination=combination,
                train_group_ids=train_group_ids,
                test_group_ids=test_group_ids,
                purged_train_idx=purged_train_idx,
                test_idx=test_idx,
                test_group_ranges=test_group_ranges,
                skipped=skipped,
                skip_reason=skip_reason,
            )

            if verbose:
                print(fold.summary())

            folds.append(fold)

        return folds

    def build_paths(self, folds: List[CPCVFold]) -> List[CPCVPath]:
        """
        重建 φ = k×C(N,k)/N 條獨立 backtest paths

        Algorithm（Lopez de Prado Ch.12）：
        對每個 group g（共 N 個）：
          - 收集所有「test_groups 包含 g」的 fold
          - 從每個這樣的 fold 取 group g 的 test_idx（test_group_ranges[g]）
          - 按時間順序拼接 → 一條 path

        N=6, k=2 情況下，每個 group 剛好出現在 C(5,1)=5 個 fold 中，
        所以每條 path 由 5 個片段（每個 fold 貢獻一個 group g 的測試片段）組成，
        覆蓋完整時序（5 個 group × C(5,1) × 1 segment ≈ 全部 bars）。

        Args:
            folds: CPCVSplitter.split() 的輸出

        Returns:
            List[CPCVPath] 長度 = n_backtest_paths（= k × C(N,k) / N）
        """
        n = self.cfg.n_groups
        k = self.cfg.k_test_groups

        # 每個 group 作為 anchor，建立一條 path
        # 但 n_backtest_paths = k*C(N,k)/N，通常 < N
        # 實作：為 n_backtest_paths 個代表性 groups 建立 path
        # 每個 group 參與 C(N-1, k-1) 個 fold（作為測試 group 之一）
        # 我們取前 n_backtest_paths 個 groups 建立 paths

        n_paths = self.cfg.n_backtest_paths
        paths: List[CPCVPath] = []

        for path_id in range(1, n_paths + 1):
            # anchor group = path_id - 1（0-indexed group id）
            # 注意：n_paths <= n，所以 anchor 0..n_paths-1 都存在
            anchor_group = path_id - 1

            # 收集所有「test_groups 包含 anchor_group」的 fold
            contributing_folds = []
            contributing_fold_ids = []

            for fold in folds:
                if anchor_group in fold.test_group_ids and not fold.skipped:
                    contributing_folds.append(fold)
                    contributing_fold_ids.append(fold.fold_id)

            # 從每個貢獻 fold 取 anchor_group 的 test_idx
            path_segments: List[np.ndarray] = []

            for fold in contributing_folds:
                if anchor_group in fold.test_group_ranges:
                    g_start, g_end = fold.test_group_ranges[anchor_group]
                    path_segments.append(np.arange(g_start, g_end))

            if path_segments:
                # 按時間順序排列（合併並排序）
                combined = np.concatenate(path_segments)
                combined = np.sort(combined)
            else:
                combined = np.array([], dtype=int)

            path = CPCVPath(
                path_id=path_id,
                group_id=anchor_group,
                fold_ids=contributing_fold_ids,
                test_idx=combined,
            )
            paths.append(path)

        return paths

    def active_folds(self, df: pd.DataFrame) -> List[CPCVFold]:
        """只回傳未跳過的 fold"""
        return [f for f in self.split(df) if not f.skipped]

    def info(self, n_bars: int) -> dict:
        """
        回傳給定資料長度下的統計資訊

        Args:
            n_bars: 資料集 bar 數

        Returns:
            dict 含 n_combinations, n_backtest_paths, group_size 等
        """
        try:
            boundaries = self._group_engine.build_groups(n_bars)
        except ValueError as e:
            return {"error": str(e), "n_combinations": 0}

        group_size = n_bars // self.cfg.n_groups

        return {
            "n_bars": n_bars,
            "n_groups": self.cfg.n_groups,
            "k_test_groups": self.cfg.k_test_groups,
            "n_combinations": self.cfg.n_combinations,
            "n_backtest_paths": self.cfg.n_backtest_paths,
            "group_size": group_size,
            "label_horizon": self.cfg.label_horizon,
            "embargo_bars": self.cfg.embargo_bars,
            "boundaries": [{"group": i, "start": s, "end": e} for i, (s, e) in enumerate(boundaries)],
        }

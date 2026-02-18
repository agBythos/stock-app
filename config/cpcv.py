"""
config/cpcv.py — CPCVConfig dataclass
======================================

Combinatorially Purged Cross-Validation（CPCV）全域參數定義。

CPCV 核心概念（Lopez de Prado, AFML Chapter 12）：
  N 組資料，每次取 k 組作為測試集 → C(N,k) 個 train/test 組合
  重組後可得 φ = k×C(N,k)/N 條獨立 backtest paths

預設值（N=6, k=2）：
  C(6,2) = 15 個 fold
  φ = 2×15/6 = 5 條獨立 backtest paths

作者：Bythos（sub-agent phase4-cpcv-impl）
建立：2026-02-18
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass
class CPCVConfig:
    """
    CPCV 參數設定

    核心參數（CPCV 特有）
    ---------------------------------------------------------------
    n_groups      : 總分組數 N（預設 6）
    k_test_groups : 每次取出作測試的組數 k（預設 2）

    防洩漏（與 WalkForwardConfig 相同語意）
    ---------------------------------------------------------------
    label_horizon : 目標變數前視 bars（purge 範圍，預設 5）
    embargo_bars  : 邊界緩衝 bars（預設 5）

    樣本數門檻
    ---------------------------------------------------------------
    min_train_samples : purge 後最小訓練樣本數（預設 200）

    台灣交易成本（沿用 WalkForwardConfig 數值）
    ---------------------------------------------------------------
    commission_rate    : 手續費率（0.1425%）
    commission_discount: 折扣（0.6）
    sell_tax_rate      : 賣出證交稅（0.3%）
    initial_capital    : 初始資金（新台幣）
    random_state       : 隨機種子
    """

    # ── CPCV 核心參數 ──────────────────────────────────────────────
    n_groups: int = 6
    k_test_groups: int = 2

    # ── 防洩漏 ────────────────────────────────────────────────────
    label_horizon: int = 5
    embargo_bars: int = 5

    # ── 樣本數門檻 ─────────────────────────────────────────────────
    min_train_samples: int = 200

    # ── 台灣交易成本 ───────────────────────────────────────────────
    commission_rate: float = 0.001425
    commission_discount: float = 0.6
    sell_tax_rate: float = 0.003
    initial_capital: float = 1_000_000.0
    random_state: int = 42

    # ── 自動計算屬性 ───────────────────────────────────────────────

    @property
    def n_combinations(self) -> int:
        """C(N, k) = fold 數（組合數）"""
        return comb(self.n_groups, self.k_test_groups)

    @property
    def n_backtest_paths(self) -> int:
        """φ = k × C(N,k) / N = 獨立 backtest path 數"""
        return self.k_test_groups * self.n_combinations // self.n_groups

    @property
    def total_gap(self) -> int:
        """Purge + embargo 總 gap（bars）"""
        return self.label_horizon + self.embargo_bars

    @property
    def effective_buy_rate(self) -> float:
        """實際買入成本率"""
        return self.commission_rate * self.commission_discount

    @property
    def effective_sell_rate(self) -> float:
        """實際賣出成本率（含證交稅）"""
        return self.commission_rate * self.commission_discount + self.sell_tax_rate

    @property
    def round_trip_cost(self) -> float:
        """單趟來回成本率"""
        return self.effective_buy_rate + self.effective_sell_rate

    def validate(self) -> None:
        """
        驗證參數合法性

        Raises:
            ValueError: 若參數不合法
        """
        if self.n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {self.n_groups}")
        if not (1 <= self.k_test_groups < self.n_groups):
            raise ValueError(
                f"k_test_groups must be in [1, n_groups), "
                f"got k={self.k_test_groups}, n={self.n_groups}"
            )
        if self.label_horizon < 0:
            raise ValueError(f"label_horizon must be >= 0, got {self.label_horizon}")
        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars must be >= 0, got {self.embargo_bars}")
        if self.min_train_samples < 1:
            raise ValueError(f"min_train_samples must be >= 1, got {self.min_train_samples}")
        if not 0 < self.commission_rate < 0.05:
            raise ValueError(f"commission_rate out of range: {self.commission_rate}")
        if not 0 < self.commission_discount <= 1.0:
            raise ValueError(f"commission_discount out of range: {self.commission_discount}")
        if not 0 <= self.sell_tax_rate < 0.05:
            raise ValueError(f"sell_tax_rate out of range: {self.sell_tax_rate}")

    def to_dict(self) -> dict:
        """轉為可序列化 dict（含計算屬性）"""
        return {
            "n_groups": self.n_groups,
            "k_test_groups": self.k_test_groups,
            "n_combinations": self.n_combinations,
            "n_backtest_paths": self.n_backtest_paths,
            "label_horizon": self.label_horizon,
            "embargo_bars": self.embargo_bars,
            "total_gap": self.total_gap,
            "min_train_samples": self.min_train_samples,
            "commission_rate": self.commission_rate,
            "commission_discount": self.commission_discount,
            "sell_tax_rate": self.sell_tax_rate,
            "effective_buy_rate": round(self.effective_buy_rate, 6),
            "effective_sell_rate": round(self.effective_sell_rate, 6),
            "round_trip_cost": round(self.round_trip_cost, 6),
            "initial_capital": self.initial_capital,
            "random_state": self.random_state,
        }

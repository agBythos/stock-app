"""
config/walk_forward.py — WalkForwardConfig dataclass
=====================================================

全域 Walk-Forward 驗證框架參數定義。

台灣交易成本說明：
  買入：0.1425% × 0.6折 = 0.0855%（手續費，無稅）
  賣出：0.1425% × 0.6折 + 0.3% 證交稅 = 0.3855%
  單趟成本（非對稱）：買 0.0855% + 賣 0.3855% ≈ 0.47%

作者：Bythos（sub-agent）
建立：2026-02-18
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WalkForwardConfig:
    """
    Walk-Forward 驗證框架全域參數

    時間窗口參數（以 bars 為單位，TWSE 約 245 trading days/year）
    ---------------------------------------------------------------
    train_window     : 訓練窗口長度（bars），預設 252 ≈ 1 年
    test_window      : 測試窗口長度（bars），預設 21 ≈ 1 個月
    step_size        : 滾動步距（bars），預設與 test_window 相同（21）

    Purge / Embargo 防洩漏
    ---------------------------------------------------------------
    label_horizon    : 目標標籤前視範圍（bars）。訓練集末尾此範圍的樣本
                       會被 purge（因其標籤包含測試期資料）。預設 5。
    embargo_bars     : 訓練/測試邊界緩衝（bars）。purge 後再加 N bars
                       的 embargo，防止特徵自相關洩漏。預設 5。
    total_gap        : label_horizon + embargo_bars = 10 bars（自動計算）

    樣本數門檻
    ---------------------------------------------------------------
    min_train_samples: purge 後訓練集最小樣本數；低於此值跳過該 fold。預設 200。

    台灣交易成本（非對稱）
    ---------------------------------------------------------------
    commission_rate    : 手續費率（買賣雙向）= 0.001425（0.1425%）
    commission_discount: 網路下單折扣 = 0.6（六折）
    sell_tax_rate      : 賣出證交稅 = 0.003（0.3%，ETF 為 0.001）

    計算：
      effective_buy_rate  = commission_rate × commission_discount = 0.0855%
      effective_sell_rate = commission_rate × commission_discount + sell_tax_rate = 0.3855%
    """

    # ── 時間窗口 ──────────────────────────────────────────────────────
    train_window: int = 252       # 訓練集長度（bars）
    test_window: int = 21         # 測試集長度（bars）
    step_size: int = 21           # 滾動步距（bars）

    # ── Purge / Embargo ───────────────────────────────────────────────
    label_horizon: int = 5        # 目標變數前視 bars（= purge bars）
    embargo_bars: int = 5         # 邊界緩衝 bars

    # ── 樣本數門檻 ─────────────────────────────────────────────────────
    min_train_samples: int = 200

    # ── 台灣交易成本 ───────────────────────────────────────────────────
    commission_rate: float = 0.001425     # 手續費率
    commission_discount: float = 0.6     # 折扣
    sell_tax_rate: float = 0.003         # 賣出證交稅（股票）

    # ── 回測參數 ───────────────────────────────────────────────────────
    initial_capital: float = 1_000_000.0   # 初始資金（新台幣）
    random_state: int = 42

    # ── 自動計算屬性 ───────────────────────────────────────────────────
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
        """單趟來回成本率（買+賣）"""
        return self.effective_buy_rate + self.effective_sell_rate

    def validate(self) -> None:
        """
        驗證參數合法性

        Raises:
            ValueError: 若參數不合法
        """
        if self.train_window < 1:
            raise ValueError(f"train_window must be ≥ 1, got {self.train_window}")
        if self.train_window < 252:
            import warnings
            warnings.warn(
                f"train_window={self.train_window} < 252 bars (< 1 TWSE year). "
                "Results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )
        if self.test_window < 1:
            raise ValueError(f"test_window must be ≥ 1, got {self.test_window}")
        if self.step_size < 1:
            raise ValueError(f"step_size must be ≥ 1, got {self.step_size}")
        if self.label_horizon < 0:
            raise ValueError(f"label_horizon must be ≥ 0, got {self.label_horizon}")
        if self.embargo_bars < 0:
            raise ValueError(f"embargo_bars must be ≥ 0, got {self.embargo_bars}")
        if self.min_train_samples < 1:
            raise ValueError(f"min_train_samples must be ≥ 1, got {self.min_train_samples}")
        if not 0 < self.commission_rate < 0.05:
            raise ValueError(f"commission_rate out of range: {self.commission_rate}")
        if not 0 < self.commission_discount <= 1.0:
            raise ValueError(f"commission_discount out of range: {self.commission_discount}")
        if not 0 <= self.sell_tax_rate < 0.05:
            raise ValueError(f"sell_tax_rate out of range: {self.sell_tax_rate}")

    def to_dict(self) -> dict:
        """轉為可序列化 dict（含計算屬性）"""
        return {
            "train_window": self.train_window,
            "test_window": self.test_window,
            "step_size": self.step_size,
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

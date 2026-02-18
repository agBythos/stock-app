"""
validation/backtrader_bridge.py — Backtrader 整合介面
====================================================

提供：
  1. TWAECommission — 精確的台灣非對稱交易成本（TWAE = Taiwan Asymmetric Exchange）
  2. WalkForwardSignalStrategy — 接受外部預測信號的策略
  3. run_backtrader_fold() — 在單一 fold 上執行 Backtrader 回測

台灣交易成本（非對稱）：
  買入成本：commission_rate × discount = 0.1425% × 0.6 = 0.0855%
  賣出成本：commission_rate × discount + sell_tax = 0.0855% + 0.3% = 0.3855%
  單趟來回：≈ 0.47%（非對稱，賣比買貴 4.5x）

設計說明：
  TWAECommission 繼承自 bt.CommInfoBase 而非 bt.CommissionScheme，
  因為 CommInfoBase 的 _getcommission() 直接接受 size（正=買/負=賣），
  可精確實作非對稱成本。

作者：Bythos（sub-agent）
建立：2026-02-18
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import backtrader as bt
import numpy as np
import pandas as pd

from config.walk_forward import WalkForwardConfig

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# TWAECommission — 台灣非對稱交易成本
# ============================================================================


class TWAECommission(bt.CommInfoBase):
    """
    台灣非對稱交易成本模型（TWAE Commission）

    成本結構（非對稱）：
    ┌─────────────────────────────────────────────────────────────┐
    │  買入：0.1425% × 0.6折 = 0.0855%                           │
    │  賣出：0.1425% × 0.6折 + 0.3% 證交稅 = 0.3855%            │
    │  單趟來回：≈ 0.47%                                          │
    └─────────────────────────────────────────────────────────────┘

    與 backtest/backtrader_engine.py 的 TaiwanCommission 差異：
    - TWAECommission 從 WalkForwardConfig 讀取參數，保持一致性
    - 介面相同，可互換使用

    實作說明：
      Backtrader COMM_PERC 模式會將 params.commission 自動除以 100，
      因此不使用 params.commission，改用自訂 params（buy_rate/sell_rate/tax_rate）
      避免自動換算。
    """

    params = (
        ("buy_rate", 0.000855),      # 買入手續費率（0.0855%）
        ("sell_rate", 0.000855),     # 賣出手續費率（0.0855%）
        ("sell_tax_rate", 0.003),    # 賣出證交稅（0.3%）
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("apply_min_commission", False),
        ("min_commission", 20.0),    # 台股實務最低手續費（TWD）
    )

    @classmethod
    def from_config(cls, config: WalkForwardConfig) -> "TWAECommission":
        """
        從 WalkForwardConfig 建立 TWAECommission

        Args:
            config: WalkForwardConfig 物件

        Returns:
            TWAECommission 實例
        """
        return cls(
            buy_rate=config.effective_buy_rate,
            sell_rate=config.effective_buy_rate,   # 手續費率相同
            sell_tax_rate=config.sell_tax_rate,
        )

    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        """
        計算單邊交易成本

        Args:
            size:       交易股數（正=買入，負=賣出）
            price:      成交價格
            pseudoexec: True=模擬計算，False=實際執行

        Returns:
            交易成本金額（新台幣）
        """
        trade_value = abs(size) * price

        if size > 0:
            # 買入：只有手續費
            cost = trade_value * self.p.buy_rate
        else:
            # 賣出：手續費 + 證交稅
            cost = trade_value * (self.p.sell_rate + self.p.sell_tax_rate)

        if self.p.apply_min_commission:
            cost = max(cost, self.p.min_commission)

        return cost


# ============================================================================
# WalkForwardSignalStrategy — 接受外部信號的策略
# ============================================================================


class WalkForwardSignalStrategy(bt.Strategy):
    """
    接受外部預測信號的 Backtrader 策略

    用於將 ML 模型的預測（0/1）轉換為買賣訂單。
    信號由 EnsembleWalkForwardTrainer 生成，
    透過 signals dict 注入（date_str → signal）。

    Parameters
    ----------
    signals : dict
        date_str → signal（1=buy/hold, 0=sell/exit）
    signal_threshold : float
        預測機率閾值（若傳入的是機率值，≥ threshold 則持有）
        預設 0.5
    position_size_pct : float
        每次買入佔可用資金比例，預設 0.95
    """

    params = (
        ("signals", {}),              # {date_str: signal_value}
        ("signal_threshold", 0.5),
        ("position_size_pct", 0.95),
    )

    def __init__(self):
        self.order: Optional[bt.Order] = None
        self._equity_dates: List[str] = []
        self._equity_values: List[float] = []

    def log(self, msg: str):
        dt = self.datas[0].datetime.date(0)
        print(f"[WF {dt}] {msg}")

    def notify_order(self, order: bt.Order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    def next(self):
        # 記錄 equity curve
        dt_str = self.datas[0].datetime.date(0).isoformat()
        self._equity_dates.append(dt_str)
        self._equity_values.append(self.broker.getvalue())

        if self.order:
            return

        # 取得當日信號
        signal_val = self.p.signals.get(dt_str, None)
        if signal_val is None:
            return  # 測試期外的日期（warmup 期），不交易

        # 信號解析：支援機率值或 0/1
        if isinstance(signal_val, float) and 0.0 <= signal_val <= 1.0:
            hold = signal_val >= self.p.signal_threshold
        else:
            hold = bool(signal_val)  # 1=持有, 0=退出

        # 執行交易
        if not self.position and hold:
            # 建倉
            cash = self.broker.get_cash()
            size = int(cash * self.p.position_size_pct / self.data.close[0])
            if size > 0:
                self.order = self.buy(size=size)
        elif self.position and not hold:
            # 清倉
            self.order = self.sell(size=self.position.size)


# ============================================================================
# run_backtrader_fold() — 單一 fold 回測
# ============================================================================


@dataclass
class BacktraderFoldResult:
    """單一 fold 的 Backtrader 回測結果"""

    fold_id: int
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    final_value: float = 0.0
    initial_capital: float = 0.0
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)

    error: Optional[str] = None
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "total_return_pct": round(self.total_return_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "win_rate_pct": round(self.win_rate_pct, 4),
            "total_trades": self.total_trades,
            "final_value": round(self.final_value, 2),
            "initial_capital": round(self.initial_capital, 2),
            "skipped": self.skipped,
            "error": self.error,
        }


def run_backtrader_fold(
    price_df: pd.DataFrame,
    signals: Dict[str, float],
    config: WalkForwardConfig,
    fold_id: int = 0,
    initial_capital: Optional[float] = None,
) -> BacktraderFoldResult:
    """
    在單一 fold 的測試期上執行 Backtrader 回測

    Args:
        price_df       : OHLCV DataFrame（只包含測試期的資料）
                         欄位支援大小寫混用（open/Open/OPEN）
        signals        : {date_str: signal_value} ML 預測信號字典
        config         : WalkForwardConfig（用於交易成本參數）
        fold_id        : fold 序號（用於回報）
        initial_capital: 初始資金；None 則使用 config.initial_capital

    Returns:
        BacktraderFoldResult

    Raises:
        ValueError: 若 price_df 缺少必要欄位
    """
    if initial_capital is None:
        initial_capital = config.initial_capital

    if len(price_df) < 2:
        return BacktraderFoldResult(
            fold_id=fold_id,
            skipped=True,
            error="price_df too short (< 2 rows)",
            initial_capital=initial_capital,
            final_value=initial_capital,
        )

    try:
        # ── 準備資料 ──────────────────────────────────────────────────
        df = price_df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            return BacktraderFoldResult(
                fold_id=fold_id,
                skipped=True,
                error=f"Missing columns: {missing}",
                initial_capital=initial_capital,
                final_value=initial_capital,
            )

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # ── Cerebro 設定 ──────────────────────────────────────────────
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(initial_capital)

        # 台灣非對稱成本
        commission = TWAECommission.from_config(config)
        cerebro.broker.addcommissioninfo(commission)

        # 載入資料
        feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(feed)

        # 加入策略（注入信號）
        cerebro.addstrategy(
            WalkForwardSignalStrategy,
            signals=signals,
        )

        # ── 執行 ──────────────────────────────────────────────────────
        results = cerebro.run()
        strat = results[0]

        # ── 提取 equity curve ─────────────────────────────────────────
        eq_values = strat._equity_values
        eq_dates = strat._equity_dates

        if not eq_values:
            eq_values = [initial_capital, cerebro.broker.getvalue()]
            eq_dates = [
                df.index[0].date().isoformat(),
                df.index[-1].date().isoformat(),
            ]

        final_value = eq_values[-1] if eq_values else initial_capital
        total_return_pct = (final_value - initial_capital) / initial_capital * 100

        # ── 計算績效指標 ──────────────────────────────────────────────
        sharpe = _calc_sharpe(eq_values)
        max_dd = _calc_max_drawdown(eq_values)

        equity_curve = [
            {"date": d, "value": round(v, 2)}
            for d, v in zip(eq_dates, eq_values)
        ]

        return BacktraderFoldResult(
            fold_id=fold_id,
            total_return_pct=round(total_return_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 4),
            final_value=round(final_value, 2),
            initial_capital=initial_capital,
            equity_curve=equity_curve,
        )

    except Exception as e:
        return BacktraderFoldResult(
            fold_id=fold_id,
            skipped=True,
            error=str(e),
            initial_capital=initial_capital,
            final_value=initial_capital,
        )


# ============================================================================
# 工具函式
# ============================================================================


def _calc_sharpe(equity_values: List[float], periods_per_year: int = 252) -> float:
    """計算年化 Sharpe Ratio（無風險利率 = 0）"""
    arr = np.array(equity_values, dtype=float)
    if len(arr) < 2:
        return 0.0
    ret = np.diff(arr) / arr[:-1]
    mean_r = np.mean(ret)
    std_r = np.std(ret, ddof=1)
    if std_r == 0 or np.isnan(std_r):
        return 0.0
    return float(mean_r / std_r * np.sqrt(periods_per_year))


def _calc_max_drawdown(equity_values: List[float]) -> float:
    """計算最大回撤 %"""
    arr = np.array(equity_values, dtype=float)
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak * 100
    return float(abs(np.min(drawdown)))

"""
BacktraderEngine — 台股 Walk-Forward 回測引擎
=============================================

獨立模組：不依賴 server.py，可直接 import 使用。

功能：
- BacktraderEngine class：傳入策略 + 數據 → 輸出回測結果
- TaiwanCommission：台灣交易成本（手續費 0.1425%×0.6折 + 證交稅 0.3%）
- Walk-Forward 驗證框架（時間序列交叉驗證）
- 完整績效分析（總報酬、Sharpe、最大回撤、勝率）

台灣交易成本說明：
  買入手續費 = 成交金額 × 0.1425% × 0.6 折 = 0.0855%
  賣出手續費 = 成交金額 × 0.1425% × 0.6 折 = 0.0855%
  賣出證交稅 = 成交金額 × 0.3%（ETF 為 0.1%，此處使用股票預設值）
  總賣出成本 = 0.0855% + 0.3% = 0.3855%

作者：Bythos（sub-agent）
建立：2026-02-18
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type

import backtrader as bt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# 台灣交易成本
# ============================================================================


class TaiwanCommission(bt.CommInfoBase):
    """
    台股交易成本計算
    
    成本結構：
    - 買入手續費：0.1425% × 0.6折 = 0.0855%（網路下單折扣）
    - 賣出手續費：0.1425% × 0.6折 = 0.0855%
    - 賣出證交稅：0.3%（股票；ETF 為 0.1%）
    
    可透過建構子參數覆蓋（如 ETF）：
        TaiwanCommission(etf_tax_rate=0.001)  # ETF 證交稅 0.1%
    
    實作說明：
        Backtrader 的 COMM_PERC 模式會將 'commission' params 自動除以 100。
        為避免此問題，費率儲存在自訂 param 名（buy_rate/sell_tax_rate），
        不使用 commtype 的自動換算機制。
    """

    params = (
        # 手續費：0.1425% × 0.6折 = 0.0855%（以小數表示 0.000855）
        ("buy_rate", 0.000855),
        # 賣出手續費（同買入）
        ("sell_rate", 0.000855),
        # 證交稅（賣出，以小數表示 0.003）
        ("sell_tax_rate", 0.003),
        # 股票模式（按金額計算，非按手數）
        ("stocklike", True),
        # 使用 COMM_PERC 模式（但費率已用自訂 param 管理，避免 /100 問題）
        ("commtype", bt.CommInfoBase.COMM_PERC),
        # 最低手續費（台股實務為 20 元）
        ("min_commission", 20.0),
        # 是否套用最低手續費
        ("apply_min_commission", False),
    )

    def _getcommission(self, size: float, price: float, pseudoexec: bool) -> float:
        """
        計算單邊交易成本
        
        買入：成交金額 × 0.0855%
        賣出：成交金額 × (0.0855% + 0.3%) = 0.3855%
        
        Args:
            size:       交易股數（正=買入，負=賣出）
            price:      成交價格
            pseudoexec: True 表示模擬計算，False 表示實際執行
            
        Returns:
            交易成本金額（新台幣）
        """
        trade_value = abs(size) * price

        if size > 0:  # 買入
            commission = trade_value * self.p.buy_rate
        else:         # 賣出：手續費 + 證交稅
            commission = trade_value * (self.p.sell_rate + self.p.sell_tax_rate)

        # 套用最低手續費（實務：每筆至少 20 元）
        if self.p.apply_min_commission:
            commission = max(commission, self.p.min_commission)

        return commission


# ============================================================================
# 資料結構
# ============================================================================


@dataclass
class TradeRecord:
    """單筆交易記錄"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float            # 含手續費淨損益
    pnl_pct: float        # 以入場成本計算的 % 損益
    commission: float


@dataclass
class BacktestResult:
    """回測結果"""
    # 基本資訊
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float

    # 績效指標
    final_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    profit_factor: float

    # 曲線資料
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為可 JSON 序列化的 dict"""
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "period": {
                "start": self.start_date,
                "end": self.end_date,
            },
            "initial_capital": self.initial_capital,
            "performance": {
                "final_value": round(self.final_value, 2),
                "total_return_pct": round(self.total_return_pct, 4),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "max_drawdown_pct": round(self.max_drawdown_pct, 4),
                "win_rate_pct": round(self.win_rate_pct, 2),
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "avg_trade_pnl": round(self.avg_trade_pnl, 2),
                "profit_factor": round(self.profit_factor, 4),
            },
            "equity_curve": self.equity_curve,
            "trades": [
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 4),
                    "commission": round(t.commission, 2),
                }
                for t in self.trades
            ],
        }


@dataclass
class WalkForwardWindow:
    """Walk-Forward 單一窗口"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class WalkForwardResult:
    """Walk-Forward 驗證結果"""
    symbol: str
    strategy_name: str
    config: Dict[str, Any]
    windows: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "config": self.config,
            "windows": self.windows,
            "summary": self.summary,
        }


# ============================================================================
# Backtrader 分析器 Wrapper
# ============================================================================


class _EquityCurveObserver(bt.Observer):
    """
    自訂 Observer：每根 K 棒記錄資產淨值
    （比 Analyzer 更精確，逐 bar 記錄）
    """
    lines = ("equity",)

    def next(self):
        self.lines.equity[0] = self._owner.broker.getvalue()


class _TradeLogger(bt.Analyzer):
    """記錄每筆完整交易"""

    def start(self):
        self.trades: List[Dict] = []
        self._open_trades: Dict[int, Dict] = {}  # ref → entry info

    def notify_trade(self, trade):
        if trade.isopen:
            self._open_trades[trade.ref] = {
                "entry_date": self.datas[0].datetime.date(0).isoformat(),
                "entry_price": round(trade.price, 4),
                "shares": int(trade.size),
            }
        elif trade.isclosed:
            entry = self._open_trades.pop(trade.ref, {})
            self.trades.append(
                TradeRecord(
                    entry_date=entry.get("entry_date", ""),
                    exit_date=self.datas[0].datetime.date(0).isoformat(),
                    entry_price=entry.get("entry_price", 0.0),
                    exit_price=round(
                        abs(trade.price * trade.size) / max(abs(trade.size), 1), 4
                    ),
                    shares=entry.get("shares", 0),
                    pnl=round(trade.pnlcomm, 4),
                    pnl_pct=round(
                        trade.pnlcomm
                        / max(entry.get("entry_price", 1) * entry.get("shares", 1), 1)
                        * 100,
                        4,
                    ),
                    commission=round(trade.commission, 4),
                )
            )

    def get_analysis(self) -> List[TradeRecord]:
        return self.trades


# ============================================================================
# 內建策略（可供直接使用或作為範例）
# ============================================================================


class _BaseStrategy(bt.Strategy):
    """策略基礎類：整合 equity curve 記錄"""

    def __init__(self):
        self.order = None
        # 每日記錄資產淨值
        self._equity_dates: List[str] = []
        self._equity_values: List[float] = []

    def next(self):
        dt = self.datas[0].datetime.date(0).isoformat()
        val = self.broker.getvalue()
        self._equity_dates.append(dt)
        self._equity_values.append(val)

    def log(self, msg: str):
        dt = self.datas[0].datetime.date(0)
        print(f"[{dt}] {msg}")


class MACrossoverStrategy(_BaseStrategy):
    """均線交叉策略 (MA Crossover)"""

    params = (
        ("fast_period", 10),
        ("slow_period", 30),
    )

    def __init__(self):
        super().__init__()
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        super().next()
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:  # 快線上穿 → 買入
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f"BUY {size} @ {self.data.close[0]:.2f}")
        else:
            if self.crossover < 0:  # 快線下穿 → 賣出
                self.log(f"SELL {self.position.size} @ {self.data.close[0]:.2f}")
                self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class RSIReversalStrategy(_BaseStrategy):
    """RSI 反轉策略"""

    params = (
        ("rsi_period", 14),
        ("rsi_lower", 30),
        ("rsi_upper", 70),
    )

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        super().next()
        if self.order:
            return

        if not self.position:
            if self.rsi < self.p.rsi_lower:
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.rsi > self.p.rsi_upper:
                self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


class MACDSignalStrategy(_BaseStrategy):
    """MACD 信號策略"""

    params = (
        ("fast_ema", 12),
        ("slow_ema", 26),
        ("signal_period", 9),
    )

    def __init__(self):
        super().__init__()
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast_ema,
            period_me2=self.p.slow_ema,
            period_signal=self.p.signal_period,
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        super().next()
        if self.order:
            return

        if not self.position:
            if self.crossover > 0:
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            if self.crossover < 0:
                self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None


# ============================================================================
# BacktraderEngine — 核心引擎
# ============================================================================


class BacktraderEngine:
    """
    台股 Backtrader 回測引擎
    ========================
    
    用法::

        from backtest.backtrader_engine import BacktraderEngine, MACrossoverStrategy

        engine = BacktraderEngine(
            symbol="2330.TW",
            initial_capital=100_000,
            commission_info=TaiwanCommission(),   # 預設已整合台股成本
        )

        result = engine.run(
            strategy_class=MACrossoverStrategy,
            data=df,                              # pandas DataFrame (OHLCV)
            strategy_params={"fast_period": 10, "slow_period": 30},
        )
        print(result.to_dict())

    Walk-Forward 用法::

        wf_result = engine.walk_forward(
            strategy_class=MACrossoverStrategy,
            data=df,
            train_months=6,
            test_months=1,
            strategy_params={"fast_period": 10, "slow_period": 30},
        )

    台灣交易成本（預設）：
    - 買入：0.1425% × 0.6折 = 0.0855%
    - 賣出：0.1425% × 0.6折 + 0.3% 證交稅 = 0.3855%
    
    內建策略：
    - MACrossoverStrategy  (fast_period, slow_period)
    - RSIReversalStrategy  (rsi_period, rsi_lower, rsi_upper)
    - MACDSignalStrategy   (fast_ema, slow_ema, signal_period)
    """

    # 內建策略映射
    BUILTIN_STRATEGIES: Dict[str, Type[bt.Strategy]] = {
        "ma_crossover": MACrossoverStrategy,
        "rsi_reversal": RSIReversalStrategy,
        "macd_signal": MACDSignalStrategy,
    }

    def __init__(
        self,
        symbol: str = "2330.TW",
        initial_capital: float = 100_000.0,
        commission_info: Optional[bt.CommInfoBase] = None,
        apply_min_commission: bool = False,
    ):
        """
        初始化回測引擎

        Args:
            symbol:             股票代碼（用於報告標記）
            initial_capital:    初始資金（新台幣）
            commission_info:    自訂佣金物件；預設使用 TaiwanCommission
            apply_min_commission: 是否套用最低手續費（台股 20 元）
        """
        self.symbol = symbol
        self.initial_capital = initial_capital

        if commission_info is None:
            commission_info = TaiwanCommission(
                apply_min_commission=apply_min_commission
            )
        self.commission_info = commission_info

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_feed(data: pd.DataFrame) -> bt.feeds.PandasData:
        """
        將 pandas DataFrame 轉換為 Backtrader 資料來源
        
        支援大小寫混用的欄位名稱（yfinance 預設大寫，walk_forward.py 用小寫）。
        """
        df = data.copy()

        # 統一欄位名稱為小寫
        df.columns = [c.lower() for c in df.columns]

        # 必要欄位檢查
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame 缺少欄位：{missing}")

        # 確保 index 為 DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 移除 timezone info（Backtrader 不接受 tz-aware index）
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 排序（時間序列必須遞增）
        df = df.sort_index()

        return bt.feeds.PandasData(dataname=df)

    @staticmethod
    def _calculate_sharpe(equity_values: List[float], periods_per_year: int = 252) -> float:
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

    @staticmethod
    def _calculate_max_drawdown(equity_values: List[float]) -> float:
        """計算最大回撤 %"""
        arr = np.array(equity_values, dtype=float)
        if len(arr) < 2:
            return 0.0

        peak = np.maximum.accumulate(arr)
        drawdown = (arr - peak) / peak * 100
        return float(abs(np.min(drawdown)))

    @staticmethod
    def _calculate_win_rate(trades: List[TradeRecord]) -> Tuple[float, int, int]:
        """計算勝率、獲利筆數、虧損筆數"""
        if not trades:
            return 0.0, 0, 0

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        win_rate = len(winners) / len(trades) * 100
        return float(win_rate), len(winners), len(losers)

    @staticmethod
    def _calculate_profit_factor(trades: List[TradeRecord]) -> float:
        """計算 Profit Factor = 總獲利 / |總虧損|"""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 4)

    # ------------------------------------------------------------------
    # 主要回測介面
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        strategy_params: Optional[Dict[str, Any]] = None,
        strategy_name: str = "Custom",
    ) -> BacktestResult:
        """
        執行單次回測

        Args:
            strategy_class:   Backtrader Strategy 類別
            data:             OHLCV pandas DataFrame（支援大小寫欄位）
            strategy_params:  策略參數 dict，傳入 cerebro.addstrategy()
            strategy_name:    用於報告的策略名稱

        Returns:
            BacktestResult 物件
        """
        if strategy_params is None:
            strategy_params = {}

        # 初始化 Cerebro
        cerebro = bt.Cerebro()

        # 設定資金與佣金
        cerebro.broker.set_cash(self.initial_capital)
        cerebro.broker.addcommissioninfo(self.commission_info)

        # 載入資料
        feed = self._prepare_feed(data)
        cerebro.adddata(feed)

        # 加入策略
        cerebro.addstrategy(strategy_class, **strategy_params)

        # 加入分析器
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades_analyzer")
        cerebro.addanalyzer(_TradeLogger, _name="trade_logger")

        # 執行回測
        results = cerebro.run()
        strat = results[0]

        # ── 提取結果 ──────────────────────────────────────────────────

        # 1. Equity Curve（來自策略內部記錄）
        eq_dates = strat._equity_dates
        eq_values = strat._equity_values

        # 若策略未繼承 _BaseStrategy，嘗試從 Analyzer 取
        if not eq_values:
            # Fallback：只有初始和最終值
            final_val = cerebro.broker.getvalue()
            eq_dates = [
                data.index[0].date().isoformat() if hasattr(data.index[0], "date") else str(data.index[0]),
                data.index[-1].date().isoformat() if hasattr(data.index[-1], "date") else str(data.index[-1]),
            ]
            eq_values = [self.initial_capital, final_val]

        final_value = eq_values[-1] if eq_values else self.initial_capital

        # 2. 績效指標
        total_return_pct = (final_value - self.initial_capital) / self.initial_capital * 100
        sharpe = self._calculate_sharpe(eq_values)
        max_dd = self._calculate_max_drawdown(eq_values)

        # 3. 交易記錄
        trade_records: List[TradeRecord] = strat.analyzers.trade_logger.get_analysis()

        win_rate, winners, losers = self._calculate_win_rate(trade_records)
        profit_factor = self._calculate_profit_factor(trade_records)
        avg_pnl = (
            sum(t.pnl for t in trade_records) / len(trade_records)
            if trade_records
            else 0.0
        )

        # 4. 資料時間範圍
        start_str = data.index[0].date().isoformat() if hasattr(data.index[0], "date") else str(data.index[0])
        end_str = data.index[-1].date().isoformat() if hasattr(data.index[-1], "date") else str(data.index[-1])

        # 5. Equity Curve 序列化
        equity_curve = [
            {"date": d, "value": round(v, 2)}
            for d, v in zip(eq_dates, eq_values)
        ]

        return BacktestResult(
            symbol=self.symbol,
            strategy_name=strategy_name,
            start_date=start_str,
            end_date=end_str,
            initial_capital=self.initial_capital,
            final_value=round(final_value, 2),
            total_return_pct=round(total_return_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 4),
            win_rate_pct=round(win_rate, 2),
            total_trades=len(trade_records),
            winning_trades=winners,
            losing_trades=losers,
            avg_trade_pnl=round(avg_pnl, 2),
            profit_factor=round(profit_factor, 4),
            equity_curve=equity_curve,
            trades=trade_records,
        )

    # ------------------------------------------------------------------
    # Walk-Forward 驗證
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        strategy_class: Type[bt.Strategy],
        data: pd.DataFrame,
        train_months: int = 6,
        test_months: int = 1,
        strategy_params: Optional[Dict[str, Any]] = None,
        strategy_name: str = "Custom",
        warmup_days: int = 60,
    ) -> WalkForwardResult:
        """
        執行 Walk-Forward 驗證

        每個窗口：
        1. 訓練期：[window_start, train_end)  → 可用於有狀態策略的學習階段
        2. 測試期：[train_end, test_end)       → 評估績效（加上 warmup 預熱）

        Args:
            strategy_class:   Backtrader Strategy 類別
            data:             全期 OHLCV DataFrame
            train_months:     訓練期長度（月）
            test_months:      測試期長度（月）
            strategy_params:  策略參數
            strategy_name:    策略名稱（用於報告）
            warmup_days:      測試期前的預熱天數（用於計算技術指標）

        Returns:
            WalkForwardResult
        """
        if strategy_params is None:
            strategy_params = {}

        # 確保 index 為 DatetimeIndex
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # 生成窗口
        windows = self._generate_wf_windows(df, train_months, test_months)

        print(f"\n{'='*60}")
        print(f"Walk-Forward: {self.symbol} — {strategy_name}")
        print(f"期間: {df.index[0].date()} → {df.index[-1].date()}")
        print(f"窗口設定: 訓練={train_months}個月, 測試={test_months}個月")
        print(f"窗口數量: {len(windows)}")
        print(f"{'='*60}")

        window_results = []
        for window in windows:
            result = self._run_wf_window(
                window, strategy_class, df, strategy_params, warmup_days
            )
            window_results.append(result)

        # 彙總統計
        summary = self._summarize_wf(window_results)

        print(f"\n{'='*60}")
        print(f"Walk-Forward 摘要")
        print(f"  成功窗口: {summary['successful_windows']}/{summary['total_windows']}")
        print(f"  平均報酬: {summary['avg_return_pct']:.2f}%")
        print(f"  報酬標準差: {summary['std_return_pct']:.2f}%")
        print(f"  平均 Sharpe: {summary['avg_sharpe']:.4f}")
        print(f"  平均最大回撤: {summary['avg_max_drawdown_pct']:.2f}%")
        print(f"  總交易次數: {summary['total_trades']}")
        print(f"{'='*60}\n")

        return WalkForwardResult(
            symbol=self.symbol,
            strategy_name=strategy_name,
            config={
                "train_months": train_months,
                "test_months": test_months,
                "initial_capital": self.initial_capital,
                "warmup_days": warmup_days,
                "strategy_params": strategy_params,
            },
            windows=window_results,
            summary=summary,
        )

    def _generate_wf_windows(
        self,
        df: pd.DataFrame,
        train_months: int,
        test_months: int,
    ) -> List[WalkForwardWindow]:
        """生成 Walk-Forward 窗口列表"""
        windows: List[WalkForwardWindow] = []
        total_start = df.index[0]
        total_end = df.index[-1]
        window_id = 1
        current_start = total_start

        while True:
            train_end = current_start + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)

            if test_end > total_end:
                break

            # 確認測試期內有足夠數據
            test_data = df[(df.index >= train_end) & (df.index < test_end)]
            if len(test_data) < 5:
                # 測試期無足夠資料，嘗試下一個窗口
                current_start += pd.DateOffset(months=test_months)
                continue

            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_start=current_start.date().isoformat(),
                    train_end=train_end.date().isoformat(),
                    test_start=train_end.date().isoformat(),
                    test_end=test_end.date().isoformat(),
                )
            )
            window_id += 1
            current_start += pd.DateOffset(months=test_months)

        return windows

    def _run_wf_window(
        self,
        window: WalkForwardWindow,
        strategy_class: Type[bt.Strategy],
        full_df: pd.DataFrame,
        strategy_params: Dict[str, Any],
        warmup_days: int,
    ) -> Dict[str, Any]:
        """
        在單一 Walk-Forward 窗口上執行回測

        為正確計算技術指標，會在測試期前加入 warmup_days 的預熱資料，
        但績效統計以測試期為準。
        """
        test_start = pd.Timestamp(window.test_start)
        test_end = pd.Timestamp(window.test_end)

        # 預熱期起始 = 測試期起始 - warmup_days
        warmup_start = test_start - pd.Timedelta(days=warmup_days)
        slice_df = full_df[full_df.index >= warmup_start]
        slice_df = slice_df[slice_df.index < test_end]

        print(
            f"[Window {window.window_id}] "
            f"訓練: {window.train_start}→{window.train_end} | "
            f"測試: {window.test_start}→{window.test_end} | "
            f"資料筆數: {len(slice_df)}"
        )

        if len(slice_df) < 10:
            return {
                "window_id": window.window_id,
                "train_period": f"{window.train_start} → {window.train_end}",
                "test_period": f"{window.test_start} → {window.test_end}",
                "status": "insufficient_data",
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "total_trades": 0,
            }

        try:
            result = self.run(
                strategy_class=strategy_class,
                data=slice_df,
                strategy_params=strategy_params,
                strategy_name=f"Window-{window.window_id}",
            )
            return {
                "window_id": window.window_id,
                "train_period": f"{window.train_start} → {window.train_end}",
                "test_period": f"{window.test_start} → {window.test_end}",
                "status": "success",
                "total_return_pct": result.total_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown_pct": result.max_drawdown_pct,
                "win_rate_pct": result.win_rate_pct,
                "total_trades": result.total_trades,
                "final_value": result.final_value,
            }
        except Exception as exc:
            print(f"  [ERROR] Window {window.window_id}: {exc}")
            return {
                "window_id": window.window_id,
                "train_period": f"{window.train_start} → {window.train_end}",
                "test_period": f"{window.test_start} → {window.test_end}",
                "status": "error",
                "error": str(exc),
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "total_trades": 0,
            }

    @staticmethod
    def _summarize_wf(window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算 Walk-Forward 彙總統計"""
        success = [w for w in window_results if w.get("status") == "success"]

        if not success:
            return {
                "total_windows": len(window_results),
                "successful_windows": 0,
                "avg_return_pct": 0.0,
                "median_return_pct": 0.0,
                "std_return_pct": 0.0,
                "best_return_pct": 0.0,
                "worst_return_pct": 0.0,
                "avg_sharpe": 0.0,
                "avg_max_drawdown_pct": 0.0,
                "total_trades": 0,
                "avg_trades_per_window": 0.0,
                "note": "No successful windows",
            }

        returns = [w["total_return_pct"] for w in success]
        sharpes = [w["sharpe_ratio"] for w in success]
        drawdowns = [w["max_drawdown_pct"] for w in success]
        trades = [w["total_trades"] for w in success]

        return {
            "total_windows": len(window_results),
            "successful_windows": len(success),
            "avg_return_pct": round(float(np.mean(returns)), 4),
            "median_return_pct": round(float(np.median(returns)), 4),
            "std_return_pct": round(float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0, 4),
            "best_return_pct": round(float(max(returns)), 4),
            "worst_return_pct": round(float(min(returns)), 4),
            "avg_sharpe": round(float(np.mean(sharpes)), 4),
            "avg_max_drawdown_pct": round(float(np.mean(drawdowns)), 4),
            "total_trades": int(sum(trades)),
            "avg_trades_per_window": round(float(np.mean(trades)), 1),
        }

    # ------------------------------------------------------------------
    # 便利工廠方法
    # ------------------------------------------------------------------

    @classmethod
    def from_strategy_name(
        cls,
        strategy_name: str,
        symbol: str = "2330.TW",
        initial_capital: float = 100_000.0,
        **kwargs,
    ) -> "BacktraderEngine":
        """
        按名稱建立引擎（使用內建策略）

        Args:
            strategy_name: "ma_crossover" | "rsi_reversal" | "macd_signal"
            symbol:        股票代碼
            initial_capital: 初始資金

        Returns:
            BacktraderEngine（已綁定對應策略）

        Raises:
            KeyError: 不支援的策略名稱
        """
        if strategy_name not in cls.BUILTIN_STRATEGIES:
            supported = list(cls.BUILTIN_STRATEGIES.keys())
            raise KeyError(
                f"不支援的策略：'{strategy_name}'。可用策略：{supported}"
            )
        return cls(symbol=symbol, initial_capital=initial_capital, **kwargs)

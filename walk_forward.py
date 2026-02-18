"""
Walk-Forward Validation Framework
防止 look-ahead bias 的核心功能
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yfinance as yf
import backtrader as bt


@dataclass
class WalkForwardWindow:
    """單一 Walk-Forward 窗口"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class WalkForwardEngine:
    """Walk-Forward 驗證引擎"""
    
    def __init__(
        self,
        symbol: str,
        strategy_class: type,
        total_start: str,
        total_end: str,
        train_months: int = 6,
        test_months: int = 1,
        initial_capital: float = 100000
    ):
        """
        初始化 Walk-Forward 引擎
        
        Args:
            symbol: 股票代碼
            strategy_class: Backtrader 策略類別
            total_start: 總時間範圍起始日
            total_end: 總時間範圍結束日
            train_months: 訓練窗口大小（月）
            test_months: 測試窗口大小（月）
            initial_capital: 初始資金
        """
        self.symbol = symbol
        self.strategy_class = strategy_class
        self.total_start = datetime.strptime(total_start, "%Y-%m-%d")
        self.total_end = datetime.strptime(total_end, "%Y-%m-%d")
        self.train_months = train_months
        self.test_months = test_months
        self.initial_capital = initial_capital
        
        # 驗證參數
        self._validate_parameters()
    
    def _validate_parameters(self):
        """驗證輸入參數"""
        if self.total_start >= self.total_end:
            raise ValueError("total_start must be before total_end")
        
        total_days = (self.total_end - self.total_start).days
        min_days = (self.train_months + self.test_months) * 30  # 最少需要一個完整窗口
        
        if total_days < min_days:
            raise ValueError(
                f"Time range too short: {total_days} days. "
                f"Need at least {min_days} days for train={self.train_months}m + test={self.test_months}m"
            )
        
        if self.train_months < 1:
            raise ValueError("train_months must be at least 1")
        
        if self.test_months < 1:
            raise ValueError("test_months must be at least 1")
    
    def _generate_windows(self) -> List[WalkForwardWindow]:
        """
        生成所有 Walk-Forward 窗口
        
        策略：
        - 從 total_start 開始
        - 每次向前滾動 test_months 個月
        - 直到沒有足夠的數據為止
        
        Returns:
            WalkForwardWindow 列表
        """
        windows = []
        window_id = 1
        current_start = self.total_start
        
        while True:
            # 計算當前窗口的訓練期結束日
            train_end = current_start + timedelta(days=self.train_months * 30)
            
            # 計算測試期結束日
            test_end = train_end + timedelta(days=self.test_months * 30)
            
            # 如果測試期超出總範圍，停止
            if test_end > self.total_end:
                break
            
            # 測試期起始日 = 訓練期結束日的下一天
            test_start = train_end + timedelta(days=1)
            
            window = WalkForwardWindow(
                window_id=window_id,
                train_start=current_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d")
            )
            windows.append(window)
            
            # 滾動到下一個窗口（向前移動 test_months）
            current_start = current_start + timedelta(days=self.test_months * 30)
            window_id += 1
        
        if len(windows) == 0:
            raise ValueError(
                f"Cannot generate any windows. Time range may be too short. "
                f"Need at least {self.train_months + self.test_months} months."
            )
        
        return windows
    
    def _fetch_data(self, start: str, end: str) -> pd.DataFrame:
        """
        下載指定時間範圍的股票數據
        
        Args:
            start: 起始日期
            end: 結束日期
            
        Returns:
            股票價格 DataFrame
        """
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol} from {start} to {end}")
        
        # Backtrader 需要的欄位名稱（小寫）
        df.columns = [col.lower() for col in df.columns]
        return df
    
    def _run_single_window(
        self,
        window: WalkForwardWindow,
        strategy_params: Dict
    ) -> Dict[str, Any]:
        """
        在單一窗口上執行回測（僅在測試期）
        
        注意：為了正確計算技術指標，需要從訓練期末期開始獲取數據
        但交易只會在測試期發生（Backtrader 會自動處理）
        
        Args:
            window: Walk-Forward 窗口
            strategy_params: 策略參數
            
        Returns:
            測試期績效指標
        """
        print(f"\n[Window {window.window_id}] Testing: {window.test_start} to {window.test_end}")
        
        # 為了正確計算技術指標，需要包含訓練期的部分數據
        # 從訓練期末期 60 天開始（足夠計算 MA60 等指標）
        warmup_start = datetime.strptime(window.train_end, "%Y-%m-%d") - timedelta(days=60)
        warmup_start_str = warmup_start.strftime("%Y-%m-%d")
        
        # 獲取包含預熱期的數據
        full_data = self._fetch_data(warmup_start_str, window.test_end)
        
        # 將數據分為預熱期和測試期（用於後續績效計算）
        test_data = self._fetch_data(window.test_start, window.test_end)
        
        if len(full_data) < 10:
            print(f"[Window {window.window_id}] Insufficient data: {len(full_data)} days")
            return {
                'window_id': window.window_id,
                'train_period': f"{window.train_start} to {window.train_end}",
                'test_period': f"{window.test_start} to {window.test_end}",
                'status': 'insufficient_data',
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'test_days': len(test_data) if not test_data.empty else 0
            }
        
        # 初始化 Cerebro（使用包含預熱期的完整數據）
        cerebro = bt.Cerebro()
        data_feed = bt.feeds.PandasData(dataname=full_data)
        cerebro.adddata(data_feed)
        
        # 添加策略
        cerebro.addstrategy(self.strategy_class, **strategy_params)
        
        # 設定初始資金
        cerebro.broker.set_cash(self.initial_capital)
        
        # 設定台股交易成本（從 server.py 導入）
        from server import TaiwanStockCommission
        cerebro.broker.addcommissioninfo(TaiwanStockCommission())
        
        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 執行回測
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        strategy = results[0]
        end_value = cerebro.broker.getvalue()
        
        # 提取績效指標
        returns = strategy.analyzers.returns.get_analysis()
        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()
        
        total_return = returns.get('rtot', 0) * 100
        sharpe_ratio = sharpe.get('sharperatio', None)
        # Sharpe ratio 可能是 None（數據不足或無變化）
        if sharpe_ratio is None or np.isnan(sharpe_ratio):
            sharpe_ratio = 0.0
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        total_trades = trades.get('total', {}).get('closed', 0)
        
        print(f"[Window {window.window_id}] Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}, Trades: {total_trades}")
        
        return {
            'window_id': window.window_id,
            'train_period': f"{window.train_start} to {window.train_end}",
            'test_period': f"{window.test_start} to {window.test_end}",
            'status': 'success',
            'total_return': round(float(total_return), 2),
            'sharpe_ratio': round(float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0, 2),
            'max_drawdown': round(float(max_dd), 2),
            'total_trades': int(total_trades),
            'test_days': len(test_data) if not test_data.empty else 0,
            'total_days': len(full_data),
            'start_value': round(start_value, 2),
            'end_value': round(end_value, 2),
            'note': 'Includes warmup period for indicator calculation'
        }
    
    def run(self, strategy_params: Dict = None) -> Dict[str, Any]:
        """
        執行完整的 Walk-Forward 驗證
        
        Args:
            strategy_params: 策略參數
            
        Returns:
            完整的 Walk-Forward 驗證結果
        """
        if strategy_params is None:
            strategy_params = {}
        
        print(f"\n{'='*60}")
        print(f"Walk-Forward Validation: {self.symbol}")
        print(f"Period: {self.total_start.strftime('%Y-%m-%d')} to {self.total_end.strftime('%Y-%m-%d')}")
        print(f"Window Size: Train={self.train_months}m, Test={self.test_months}m")
        print(f"{'='*60}")
        
        # 生成窗口
        windows = self._generate_windows()
        print(f"\nGenerated {len(windows)} Walk-Forward windows")
        
        # 在每個窗口上執行回測
        window_results = []
        for window in windows:
            result = self._run_single_window(window, strategy_params)
            window_results.append(result)
        
        # 計算彙總績效
        summary = self._calculate_summary(window_results)
        
        print(f"\n{'='*60}")
        print(f"Walk-Forward Summary:")
        print(f"  Average Return: {summary['average_return']:.2f}%")
        print(f"  Average Sharpe: {summary['average_sharpe']:.2f}")
        print(f"  Return Stability (StdDev): {summary['return_std']:.2f}%")
        print(f"  Sharpe Stability (StdDev): {summary['sharpe_std']:.2f}")
        print(f"  Total Windows: {summary['total_windows']}")
        print(f"  Successful Windows: {summary['successful_windows']}")
        print(f"{'='*60}\n")
        
        return {
            'status': 'success',
            'symbol': self.symbol,
            'config': {
                'total_period': {
                    'start': self.total_start.strftime('%Y-%m-%d'),
                    'end': self.total_end.strftime('%Y-%m-%d')
                },
                'train_months': self.train_months,
                'test_months': self.test_months,
                'initial_capital': self.initial_capital
            },
            'windows': window_results,
            'summary': summary
        }
    
    def _calculate_summary(self, window_results: List[Dict]) -> Dict[str, Any]:
        """
        計算彙總績效和穩定性指標
        
        Args:
            window_results: 所有窗口的結果
            
        Returns:
            彙總統計數據
        """
        # 過濾成功的窗口
        successful = [w for w in window_results if w.get('status') == 'success']
        
        if len(successful) == 0:
            return {
                'total_windows': len(window_results),
                'successful_windows': 0,
                'average_return': 0.0,
                'average_sharpe': 0.0,
                'average_max_drawdown': 0.0,
                'return_std': 0.0,
                'sharpe_std': 0.0,
                'total_trades': 0,
                'note': 'No successful windows'
            }
        
        # 提取指標
        returns = [w['total_return'] for w in successful]
        sharpes = [w['sharpe_ratio'] for w in successful]
        drawdowns = [w['max_drawdown'] for w in successful]
        trades = [w['total_trades'] for w in successful]
        
        # 計算穩定性（標準差）
        return_std = np.std(returns) if len(returns) > 1 else 0.0
        sharpe_std = np.std(sharpes) if len(sharpes) > 1 else 0.0
        
        return {
            'total_windows': len(window_results),
            'successful_windows': len(successful),
            'average_return': round(float(np.mean(returns)), 2),
            'median_return': round(float(np.median(returns)), 2),
            'average_sharpe': round(float(np.mean(sharpes)), 2),
            'median_sharpe': round(float(np.median(sharpes)), 2),
            'average_max_drawdown': round(float(np.mean(drawdowns)), 2),
            'return_std': round(float(return_std), 2),  # 穩定性指標
            'sharpe_std': round(float(sharpe_std), 2),  # 穩定性指標
            'best_return': round(float(max(returns)), 2),
            'worst_return': round(float(min(returns)), 2),
            'total_trades': int(sum(trades)),
            'average_trades_per_window': round(float(np.mean(trades)), 1)
        }

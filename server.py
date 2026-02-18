"""
Stock Analysis Backend Server
FastAPI backend for stock data analysis with technical indicators
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import asyncio
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import uvicorn
import os
import backtrader as bt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
try:
    from catboost import CatBoostClassifier as _CatBoostClassifier  # noqa: F401 – verify install
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier as _XGBClassifier  # noqa: F401 – verify install
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier as _LGBMClassifier  # noqa: F401 – verify install
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import sqlite3
import json

# Phase 6: Import strategies from backtest/ module
try:
    from backtest.hmm_filter_strategy import HMMFilterStrategy as _HMMFilterStrategy
    from backtest.rf_strategy import RFStrategy as _BacktestRFStrategy
    from backtest.backtrader_engine import (
        BacktraderEngine as _BacktraderEngine,
        MACrossoverStrategy as _BacktestMACrossover,
    )
    _BACKTEST_MODULE_AVAILABLE = True
except ImportError as _e:
    print(f"[WARNING] backtest module import failed: {_e}")
    _BACKTEST_MODULE_AVAILABLE = False

# Phase 7 Step 5: Model persistence cache
try:
    from cache.model_cache import ModelCache as _ModelCache
    _model_cache = _ModelCache()
    _MODEL_CACHE_AVAILABLE = True
    print("[INFO] ModelCache initialised")
except Exception as _cache_err:
    print(f"[WARNING] ModelCache init failed: {_cache_err}")
    _model_cache = None
    _MODEL_CACHE_AVAILABLE = False

# Phase 7 Step 6: Scheduler
try:
    from scheduler import scheduler as _scheduler, read_alert_log as _read_alert_log
    from scheduler import append_alert_log as _append_alert_log
    _SCHEDULER_AVAILABLE = True
    print("[INFO] MonitorScheduler loaded")
except Exception as _sched_err:
    print(f"[WARNING] Scheduler load failed: {_sched_err}")
    _scheduler = None
    _SCHEDULER_AVAILABLE = False

# Phase 9: Signal Alert
try:
    from alerts.signal_alert import check_and_send_signal_alerts as _check_signals
    _SIGNAL_ALERT_AVAILABLE = True
    print("[INFO] SignalAlert loaded")
except Exception as _sa_err:
    print(f"[WARNING] SignalAlert load failed: {_sa_err}")
    _check_signals = None
    _SIGNAL_ALERT_AVAILABLE = False

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
from contextlib import contextmanager

# Initialize FastAPI app
app = FastAPI(
    title="Stock Analysis API",
    description="Stock data and technical analysis API",
    version="2.0.0"
)

def sanitize_numpy(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_numpy(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

class NumpyJSONResponse(JSONResponse):
    """JSONResponse that handles numpy types."""
    def render(self, content: Any) -> bytes:
        return json.dumps(content, cls=NumpyEncoder, ensure_ascii=False).encode("utf-8")

# Override default response class for all endpoints
app.default_response_class = NumpyJSONResponse

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event: 初始化資料庫 + 排程器
@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化模擬交易資料庫，並啟動 regime 監控排程器"""
    init_sim_trading_db()
    # Phase 7 Step 6: 啟動排程器
    if _SCHEDULER_AVAILABLE and _scheduler is not None:
        _scheduler.start()
        print("[INFO] MonitorScheduler started (daily at 15:30 Taiwan time)")

# Static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================================
# Technical Indicators Calculation
# ============================================================================

def calculate_ma(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Moving Average"""
    return data['Close'].rolling(window=period).mean()


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    return {
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }


# ============================================================================
# Prediction System
# ============================================================================

class BasePredictor(ABC):
    """Abstract base class for stock prediction strategies"""
    
    def __init__(self):
        """Initialize predictor with internal state"""
        self._historical_data = None
        self._last_prediction = None
        self._feature_scores = {}
    
    @abstractmethod
    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate prediction based on stock data
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            
        Returns:
            Dictionary with prediction results
        """
        pass
    
    def validate_input(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data quality
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Dictionary with validation results:
            - is_valid: bool
            - issues: list of detected issues
            - quality_score: 0-1 quality score
        """
        issues = []
        
        # Check if data is empty
        if data is None or len(data) == 0:
            return {
                "is_valid": False,
                "issues": ["Empty dataset"],
                "quality_score": 0.0
            }
        
        # Required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for NaN values
        nan_counts = data[required_cols].isna().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            issues.append(f"Found {total_nan} NaN values across columns")
        
        # Check for outliers (price changes > 50% in one day)
        if 'Close' in data.columns and len(data) > 1:
            pct_change = data['Close'].pct_change().abs()
            extreme_changes = (pct_change > 0.5).sum()
            if extreme_changes > 0:
                issues.append(f"Found {extreme_changes} extreme price changes (>50%)")
        
        # Check time continuity (if index is datetime)
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            gaps = (data.index[1:] - data.index[:-1]).days
            large_gaps = (gaps > 7).sum()  # More than a week
            if large_gaps > len(data) * 0.1:  # More than 10% gaps
                issues.append(f"Time series has {large_gaps} significant gaps")
        
        # Check for zero/negative prices
        if 'Close' in data.columns:
            invalid_prices = (data['Close'] <= 0).sum()
            if invalid_prices > 0:
                issues.append(f"Found {invalid_prices} zero or negative prices")
        
        # Calculate quality score
        quality_score = 1.0
        quality_score -= min(0.3, total_nan / (len(data) * len(required_cols)))  # NaN penalty
        quality_score -= min(0.2, extreme_changes / len(data) * 5)  # Outlier penalty
        quality_score -= 0.5 if missing_cols else 0  # Missing columns penalty
        quality_score = max(0.0, quality_score)
        
        return {
            "is_valid": bool(len(issues) == 0 or quality_score >= 0.6),
            "issues": issues,
            "quality_score": float(round(quality_score, 3))
        }
    
    def update(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Incrementally update model with new data
        
        Args:
            new_data: New price data to incorporate
            
        Returns:
            Dictionary with update status
        """
        validation = self.validate_input(new_data)
        
        if not validation["is_valid"]:
            return {
                "success": False,
                "reason": "Data validation failed",
                "issues": validation["issues"]
            }
        
        # Merge with existing historical data
        if self._historical_data is None:
            self._historical_data = new_data.copy()
            rows_added = len(new_data)
        else:
            # Concatenate and remove duplicates
            combined = pd.concat([self._historical_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            rows_added = len(combined) - len(self._historical_data)
            self._historical_data = combined
        
        return {
            "success": True,
            "rows_added": rows_added,
            "total_rows": len(self._historical_data),
            "date_range": {
                "start": str(self._historical_data.index[0]),
                "end": str(self._historical_data.index[-1])
            }
        }
    
    def predict_proba(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """
        Return prediction probabilities for each class
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            
        Returns:
            Dictionary with probabilities:
            - buy_probability: 0-1
            - sell_probability: 0-1
            - hold_probability: 0-1
        """
        # Get prediction
        prediction = self.predict(symbol, data)
        
        # Convert confidence to probability distribution
        confidence = prediction.get('confidence', 0) / 100.0  # Normalize to 0-1
        signal = prediction.get('signal', 'HOLD')
        
        # Create probability distribution based on signal and confidence
        if signal == 'BUY':
            buy_prob = 0.33 + confidence * 0.67  # 0.33 to 1.0
            sell_prob = (1 - buy_prob) * 0.2
            hold_prob = 1 - buy_prob - sell_prob
        elif signal == 'SELL':
            sell_prob = 0.33 + confidence * 0.67
            buy_prob = (1 - sell_prob) * 0.2
            hold_prob = 1 - buy_prob - sell_prob
        else:  # HOLD
            hold_prob = 0.5 + confidence * 0.3
            buy_prob = (1 - hold_prob) / 2
            sell_prob = (1 - hold_prob) / 2
        
        return {
            'buy_probability': float(round(buy_prob, 3)),
            'sell_probability': float(round(sell_prob, 3)),
            'hold_probability': float(round(hold_prob, 3))
        }
    
    def get_confidence(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Return confidence interval for prediction
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            
        Returns:
            Dictionary with confidence metrics:
            - confidence_level: 0-1 overall confidence
            - lower_bound: Lower price estimate
            - upper_bound: Upper price estimate
            - prediction_range: Expected price range
        """
        if len(data) < 20:
            return {
                "confidence_level": 0.0,
                "lower_bound": None,
                "upper_bound": None,
                "prediction_range": None,
                "note": "Insufficient data for confidence interval"
            }
        
        current_price = data['Close'].iloc[-1]
        
        # Calculate volatility (standard deviation)
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate prediction confidence based on data quality
        validation = self.validate_input(data)
        base_confidence = validation['quality_score']
        
        # Adjust confidence based on recent volatility
        # Higher volatility = lower confidence
        if volatility > 0.05:  # High volatility (>5% daily)
            confidence_level = base_confidence * 0.5
        elif volatility > 0.03:  # Medium volatility
            confidence_level = base_confidence * 0.75
        else:  # Low volatility
            confidence_level = base_confidence * 0.95
        
        # Calculate prediction interval (±2 std dev for ~95% confidence)
        price_std = current_price * volatility * np.sqrt(5)  # 5-day forecast
        lower_bound = current_price - 2 * price_std
        upper_bound = current_price + 2 * price_std
        
        return {
            "confidence_level": float(round(confidence_level, 3)),
            "lower_bound": float(round(lower_bound, 2)),
            "upper_bound": float(round(upper_bound, 2)),
            "prediction_range": f"${round(lower_bound, 2)} - ${round(upper_bound, 2)}",
            "volatility": float(round(volatility, 4)),
            "current_price": float(round(current_price, 2))
        }
    
    def feature_importance(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Return feature importance ranking
        
        Args:
            symbol: Stock symbol
            data: Historical price data
            
        Returns:
            List of features ranked by importance:
            - feature: Feature name
            - importance: 0-1 importance score
            - impact: Direction of impact (positive/negative)
        """
        # This is a base implementation
        # Subclasses can override with more sophisticated methods
        return [
            {
                "feature": "price_trend",
                "importance": 0.5,
                "impact": "neutral",
                "description": "Overall price trend"
            }
        ]


class TechnicalPredictor(BasePredictor):
    """Technical indicator-based prediction"""
    
    def __init__(self):
        """Initialize technical predictor"""
        super().__init__()
        self._feature_weights = {
            'ma_trend': 0.25,
            'ma_crossover': 0.20,
            'rsi': 0.25,
            'macd': 0.30
        }
    
    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate buy/sell signals based on technical indicators"""
        
        if len(data) < 60:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "Insufficient data for analysis",
                "indicators": {}
            }
        
        # Calculate indicators
        ma5 = calculate_ma(data, 5).iloc[-1]
        ma20 = calculate_ma(data, 20).iloc[-1]
        ma60 = calculate_ma(data, 60).iloc[-1]
        rsi = calculate_rsi(data, 14).iloc[-1]
        macd_data = calculate_macd(data)
        macd = macd_data['macd'].iloc[-1]
        signal = macd_data['signal'].iloc[-1]
        
        current_price = data['Close'].iloc[-1]
        
        # Signal calculation
        signals = []
        reasons = []
        
        # MA trend analysis
        if ma5 > ma20 > ma60:
            signals.append(1)  # Bullish
            reasons.append("MA5 > MA20 > MA60 (uptrend)")
        elif ma5 < ma20 < ma60:
            signals.append(-1)  # Bearish
            reasons.append("MA5 < MA20 < MA60 (downtrend)")
        
        # MA crossover
        if current_price > ma20:
            signals.append(1)
            reasons.append("Price above MA20")
        elif current_price < ma20:
            signals.append(-1)
            reasons.append("Price below MA20")
        
        # RSI analysis
        if rsi < 30:
            signals.append(1)
            reasons.append(f"RSI oversold ({rsi:.2f})")
        elif rsi > 70:
            signals.append(-1)
            reasons.append(f"RSI overbought ({rsi:.2f})")
        
        # MACD analysis
        if macd > signal and macd > 0:
            signals.append(1)
            reasons.append("MACD bullish crossover")
        elif macd < signal and macd < 0:
            signals.append(-1)
            reasons.append("MACD bearish crossover")
        
        # Calculate final signal
        if not signals:
            final_signal = "HOLD"
            confidence = 0
        else:
            avg_signal = sum(signals) / len(signals)
            confidence = abs(avg_signal) * 100
            
            if avg_signal > 0.3:
                final_signal = "BUY"
            elif avg_signal < -0.3:
                final_signal = "SELL"
            else:
                final_signal = "HOLD"
        
        return {
            "symbol": symbol,
            "signal": final_signal,
            "confidence": round(confidence, 2),
            "reason": " | ".join(reasons) if reasons else "Neutral market conditions",
            "indicators": {
                "ma5": round(ma5, 2),
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "rsi": round(rsi, 2),
                "macd": round(macd, 4),
                "signal_line": round(signal, 4),
                "current_price": round(current_price, 2)
            }
        }
    
    def feature_importance(self, symbol: str, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Return technical indicator importance ranking
        
        Calculates feature importance based on:
        - Signal strength (how strong the signal is)
        - Historical accuracy (weighted by configuration)
        - Current market conditions
        """
        if len(data) < 60:
            return [{
                "feature": "insufficient_data",
                "importance": 0.0,
                "impact": "neutral",
                "description": "Need at least 60 days of data"
            }]
        
        features = []
        
        # Calculate indicators
        ma5 = calculate_ma(data, 5).iloc[-1]
        ma20 = calculate_ma(data, 20).iloc[-1]
        ma60 = calculate_ma(data, 60).iloc[-1]
        rsi = calculate_rsi(data, 14).iloc[-1]
        macd_data = calculate_macd(data)
        macd = macd_data['macd'].iloc[-1]
        signal_line = macd_data['signal'].iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # MA Trend Analysis
        ma_trend_strength = 0.0
        ma_impact = "neutral"
        if ma5 > ma20 > ma60:
            ma_trend_strength = abs((ma5 - ma60) / ma60) * self._feature_weights['ma_trend']
            ma_impact = "positive"
        elif ma5 < ma20 < ma60:
            ma_trend_strength = abs((ma5 - ma60) / ma60) * self._feature_weights['ma_trend']
            ma_impact = "negative"
        
        features.append({
            "feature": "moving_average_trend",
            "importance": float(round(min(ma_trend_strength, 1.0), 3)),
            "impact": ma_impact,
            "description": f"MA5: {ma5:.2f}, MA20: {ma20:.2f}, MA60: {ma60:.2f}",
            "weight": self._feature_weights['ma_trend']
        })
        
        # MA Crossover
        crossover_strength = abs(current_price - ma20) / ma20 * self._feature_weights['ma_crossover']
        crossover_impact = "positive" if current_price > ma20 else "negative"
        
        features.append({
            "feature": "price_ma_crossover",
            "importance": float(round(min(crossover_strength, 1.0), 3)),
            "impact": crossover_impact,
            "description": f"Price {crossover_impact} relative to MA20",
            "weight": self._feature_weights['ma_crossover']
        })
        
        # RSI Analysis
        rsi_strength = 0.0
        rsi_impact = "neutral"
        if rsi < 30:
            rsi_strength = (30 - rsi) / 30 * self._feature_weights['rsi']
            rsi_impact = "positive"  # Oversold = buy signal
        elif rsi > 70:
            rsi_strength = (rsi - 70) / 30 * self._feature_weights['rsi']
            rsi_impact = "negative"  # Overbought = sell signal
        else:
            rsi_strength = 0.1 * self._feature_weights['rsi']  # Neutral zone
        
        features.append({
            "feature": "rsi_indicator",
            "importance": float(round(min(rsi_strength, 1.0), 3)),
            "impact": rsi_impact,
            "description": f"RSI: {rsi:.2f} {'(oversold)' if rsi < 30 else '(overbought)' if rsi > 70 else '(neutral)'}",
            "weight": self._feature_weights['rsi']
        })
        
        # MACD Analysis
        macd_strength = abs(macd - signal_line) / abs(current_price) * 100 * self._feature_weights['macd']
        macd_impact = "positive" if macd > signal_line else "negative"
        
        features.append({
            "feature": "macd_signal",
            "importance": float(round(min(macd_strength, 1.0), 3)),
            "impact": macd_impact,
            "description": f"MACD: {macd:.4f}, Signal: {signal_line:.4f}",
            "weight": self._feature_weights['macd']
        })
        
        # Sort by importance (descending)
        features.sort(key=lambda x: x['importance'], reverse=True)
        
        return features


# Predictor instance
predictor = TechnicalPredictor()


# ============================================================================
# Random Forest Predictor (ML-based)
# ============================================================================

class RandomForestPredictor(BasePredictor):
    """Random Forest 機器學習預測器"""
    
    def __init__(self, forward_days: int = 5, confidence_threshold: float = 0.55):
        """
        初始化 Random Forest 預測器
        
        Args:
            forward_days: 預測未來 N 天報酬率（default=5）
            confidence_threshold: 信心閾值（低於此值視為 HOLD）
        """
        super().__init__()
        self.forward_days = forward_days
        self.confidence_threshold = confidence_threshold
        
        # Random Forest 模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self._feature_importances = {}
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering：計算技術指標特徵
        
        Features:
        - RSI(14)
        - MACD signal
        - MA crossover (10/30, 20/60)
        - Bollinger Band %B
        - Volume ratio (vs 20-day avg)
        - Price momentum (5d, 10d, 20d returns)
        - Volatility (20d rolling std)
        
        Returns:
            DataFrame with features
        """
        df = data.copy()
        
        # 1. RSI(14)
        df['rsi_14'] = calculate_rsi(df, 14)
        
        # 2. MACD
        macd_data = calculate_macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']
        
        # 3. MA crossover (10/30, 20/60)
        df['ma_10'] = calculate_ma(df, 10)
        df['ma_30'] = calculate_ma(df, 30)
        df['ma_20'] = calculate_ma(df, 20)
        df['ma_60'] = calculate_ma(df, 60)
        
        df['ma_cross_10_30'] = (df['ma_10'] - df['ma_30']) / df['Close']
        df['ma_cross_20_60'] = (df['ma_20'] - df['ma_60']) / df['Close']
        
        # 4. Bollinger Bands %B
        bb_period = 20
        bb_std = 2
        df['bb_mid'] = df['Close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['Close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (bb_std * df['bb_std'])
        df['bb_pct_b'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 5. Volume ratio (vs 20-day avg)
        df['vol_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma_20']
        
        # 6. Price momentum (5d, 10d, 20d returns)
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_10d'] = df['Close'].pct_change(10)
        df['momentum_20d'] = df['Close'].pct_change(20)
        
        # 7. Volatility (20d rolling std)
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
        
        # 選擇最終特徵
        self.feature_names = [
            'rsi_14', 'macd_hist', 'ma_cross_10_30', 'ma_cross_20_60',
            'bb_pct_b', 'vol_ratio', 'momentum_5d', 'momentum_10d',
            'momentum_20d', 'volatility_20d'
        ]
        
        return df[self.feature_names]
    
    def _calculate_target(self, data: pd.DataFrame) -> pd.Series:
        """
        計算目標變數：未來 N 天報酬率 > 0 → 1, else 0
        
        Args:
            data: 價格數據
            
        Returns:
            Series of binary labels (0 or 1)
        """
        # 計算未來 N 天報酬率
        future_return = data['Close'].pct_change(self.forward_days).shift(-self.forward_days)
        
        # 二分類：正報酬 = 1, 負報酬 = 0
        target = (future_return > 0).astype(int)
        
        return target
    
    def train(self, data: pd.DataFrame):
        """
        訓練 Random Forest 模型
        
        Args:
            data: 歷史價格數據
        """
        if len(data) < 100:
            print("[RF] Insufficient data for training (need at least 100 days)")
            return
        
        # 計算特徵
        features = self._calculate_features(data)
        
        # 計算目標
        target = self._calculate_target(data)
        
        # 合併特徵和目標，排除 NaN
        train_data = pd.concat([features, target.rename('target')], axis=1).dropna()
        
        if len(train_data) < 50:
            print(f"[RF] Insufficient training data after removing NaN: {len(train_data)} rows")
            return
        
        X = train_data[self.feature_names]
        y = train_data['target']
        
        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)
        
        # 訓練模型
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # 儲存 feature importances
        self._feature_importances = dict(zip(self.feature_names, self.model.feature_importances_))
        
        print(f"[RF] Model trained on {len(train_data)} samples")
        print(f"[RF] Top 3 features: {sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成預測信號
        
        Returns:
            - signal: "BUY"/"SELL"/"HOLD"
            - confidence: 預測信心（0-100）
            - reason: 預測原因
        """
        # 如果模型未訓練，先訓練
        if not self.is_trained:
            self.train(data)
        
        if not self.is_trained:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "Model not trained (insufficient data)"
            }
        
        # 計算最新數據的特徵
        features = self._calculate_features(data)
        
        # 取最後一行（最新數據）
        latest_features = features.iloc[[-1]].dropna()
        
        if latest_features.empty or len(latest_features.columns) < len(self.feature_names):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "Insufficient feature data"
            }
        
        # 標準化
        X = self.scaler.transform(latest_features[self.feature_names])
        
        # 預測機率
        proba = self.model.predict_proba(X)[0]
        
        # proba[0] = 下跌機率, proba[1] = 上漲機率
        prob_up = float(proba[1])
        prob_down = float(proba[0])
        
        # 決定信號
        if prob_up >= self.confidence_threshold:
            signal = "BUY"
            confidence = prob_up * 100
        elif prob_down >= self.confidence_threshold:
            signal = "SELL"
            confidence = prob_down * 100
        else:
            signal = "HOLD"
            confidence = max(prob_up, prob_down) * 100
        
        # 找出最重要的特徵作為原因
        top_features = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        reason = f"ML prediction based on {', '.join([f[0] for f in top_features])}"
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "reason": reason,
            "probabilities": {
                "up": round(float(prob_up), 3),
                "down": round(float(prob_down), 3)
            }
        }
    
    def feature_importance(self) -> Dict[str, float]:
        """
        回傳特徵重要性
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}
        
        return {k: round(float(v), 4) for k, v in self._feature_importances.items()}


class CatBoostPredictor(BasePredictor):
    """CatBoost 梯度提升機器學習預測器"""

    def __init__(self, forward_days: int = 5, confidence_threshold: float = 0.55):
        """
        初始化 CatBoost 預測器

        Args:
            forward_days: 預測未來 N 天報酬率（default=5）
            confidence_threshold: 信心閾值（低於此值視為 HOLD）
        """
        super().__init__()
        self.forward_days = forward_days
        self.confidence_threshold = confidence_threshold

        # CatBoost 模型（延遲 import，避免未安裝時整個 server 崩潰）
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0,
            eval_metric='Accuracy',
            early_stopping_rounds=50,
        )

        self.is_trained = False
        self.feature_names = []
        self._feature_importances = {}

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering：計算技術指標特徵（與 RandomForestPredictor 相同的 10 個指標）

        Features:
        - RSI(14)
        - MACD histogram
        - MA crossover (10/30, 20/60)
        - Bollinger Band %B
        - Volume ratio (vs 20-day avg)
        - Price momentum (5d, 10d, 20d returns)
        - Volatility (20d rolling std)

        Returns:
            DataFrame with features
        """
        df = data.copy()

        # 1. RSI(14)
        df['rsi_14'] = calculate_rsi(df, 14)

        # 2. MACD
        macd_data = calculate_macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']

        # 3. MA crossover (10/30, 20/60)
        df['ma_10'] = calculate_ma(df, 10)
        df['ma_30'] = calculate_ma(df, 30)
        df['ma_20'] = calculate_ma(df, 20)
        df['ma_60'] = calculate_ma(df, 60)

        df['ma_cross_10_30'] = (df['ma_10'] - df['ma_30']) / df['Close']
        df['ma_cross_20_60'] = (df['ma_20'] - df['ma_60']) / df['Close']

        # 4. Bollinger Bands %B
        bb_period = 20
        bb_std = 2
        df['bb_mid'] = df['Close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['Close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (bb_std * df['bb_std'])
        df['bb_pct_b'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 5. Volume ratio (vs 20-day avg)
        df['vol_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma_20']

        # 6. Price momentum (5d, 10d, 20d returns)
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_10d'] = df['Close'].pct_change(10)
        df['momentum_20d'] = df['Close'].pct_change(20)

        # 7. Volatility (20d rolling std)
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()

        # 選擇最終特徵（共 10 個）
        self.feature_names = [
            'rsi_14', 'macd_hist', 'ma_cross_10_30', 'ma_cross_20_60',
            'bb_pct_b', 'vol_ratio', 'momentum_5d', 'momentum_10d',
            'momentum_20d', 'volatility_20d'
        ]

        return df[self.feature_names]

    def _calculate_target(self, data: pd.DataFrame) -> pd.Series:
        """
        計算目標變數：未來 N 天報酬率 > 0 → 1, else 0

        Args:
            data: 價格數據

        Returns:
            Series of binary labels (0 or 1)
        """
        future_return = data['Close'].pct_change(self.forward_days).shift(-self.forward_days)
        target = (future_return > 0).astype(int)
        return target

    def train(self, data: pd.DataFrame):
        """
        訓練 CatBoost 模型，使用後 20% 資料作為 validation set 觸發 early stopping

        Args:
            data: 歷史價格數據
        """
        if len(data) < 100:
            print("[CatBoost] Insufficient data for training (need at least 100 days)")
            return

        # 計算特徵與目標
        features = self._calculate_features(data)
        target = self._calculate_target(data)

        # 合併，排除 NaN
        train_data = pd.concat([features, target.rename('target')], axis=1).dropna()

        if len(train_data) < 50:
            print(f"[CatBoost] Insufficient training data after removing NaN: {len(train_data)} rows")
            return

        X = train_data[self.feature_names].values
        y = train_data['target'].values

        # 分割 train / validation（後 20% 做 early stopping）
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_val) > 0:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )
        else:
            # 資料太少，不使用 validation set
            self.model.fit(X_train, y_train, verbose=False)

        self.is_trained = True

        # 儲存 feature importances
        importances = self.model.get_feature_importance()
        self._feature_importances = dict(zip(self.feature_names, importances))

        print(f"[CatBoost] Model trained on {len(train_data)} samples "
              f"(train={split_idx}, val={len(X_val)})")
        print(f"[CatBoost] Best iteration: {self.model.best_iteration_}")
        top3 = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"[CatBoost] Top 3 features: {top3}")

    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成預測信號

        Returns:
            - signal: "BUY"/"SELL"/"HOLD"
            - confidence: 預測信心（0-100）
            - reason: 預測原因
            - probabilities: {"up": float, "down": float}
            - model: "catboost"
        """
        # 如果模型未訓練，先訓練
        if not self.is_trained:
            self.train(data)

        if not self.is_trained:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "CatBoost model not trained (insufficient data)",
                "model": "catboost",
            }

        # 計算最新數據的特徵
        features = self._calculate_features(data)
        latest_features = features.iloc[[-1]].dropna()

        if latest_features.empty or len(latest_features.columns) < len(self.feature_names):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "Insufficient feature data",
                "model": "catboost",
            }

        X = latest_features[self.feature_names].values

        # 預測機率
        proba = self.model.predict_proba(X)[0]
        prob_down, prob_up = float(proba[0]), float(proba[1])

        # 決定信號
        if prob_up >= self.confidence_threshold:
            signal = "BUY"
            confidence = prob_up * 100
        elif prob_down >= self.confidence_threshold:
            signal = "SELL"
            confidence = prob_down * 100
        else:
            signal = "HOLD"
            confidence = max(prob_up, prob_down) * 100

        # 找出最重要的特徵作為原因
        top_features = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        reason = f"CatBoost prediction based on {', '.join([f[0] for f in top_features])}"

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "reason": reason,
            "probabilities": {
                "up": round(float(prob_up), 3),
                "down": round(float(prob_down), 3),
            },
            "model": "catboost",
        }

    def feature_importance(self) -> Dict[str, float]:
        """
        回傳特徵重要性

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}

        return {k: round(float(v), 4) for k, v in self._feature_importances.items()}


# ============================================================================
# XGBoost Predictor
# ============================================================================

class XGBoostPredictor(BasePredictor):
    """XGBoost 梯度提升機器學習預測器"""

    def __init__(self, forward_days: int = 5, confidence_threshold: float = 0.55):
        """
        初始化 XGBoost 預測器

        Args:
            forward_days: 預測未來 N 天報酬率（default=5）
            confidence_threshold: 信心閾值（低於此值視為 HOLD）
        """
        super().__init__()
        self.forward_days = forward_days
        self.confidence_threshold = confidence_threshold

        # XGBoost 模型（延遲 import，避免未安裝時整個 server 崩潰）
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50,
            verbosity=0,
        )

        self.is_trained = False
        self.feature_names = []
        self._feature_importances = {}

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering：計算技術指標特徵（10 個指標，與 RandomForestPredictor 相同）

        Features:
        - RSI(14)
        - MACD histogram
        - MA crossover (10/30, 20/60)
        - Bollinger Band %B
        - Volume ratio (vs 20-day avg)
        - Price momentum (5d, 10d, 20d returns)
        - Volatility (20d rolling std)

        Returns:
            DataFrame with features
        """
        df = data.copy()

        # 1. RSI(14)
        df['rsi_14'] = calculate_rsi(df, 14)

        # 2. MACD
        macd_data = calculate_macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']

        # 3. MA crossover (10/30, 20/60)
        df['ma_10'] = calculate_ma(df, 10)
        df['ma_30'] = calculate_ma(df, 30)
        df['ma_20'] = calculate_ma(df, 20)
        df['ma_60'] = calculate_ma(df, 60)

        df['ma_cross_10_30'] = (df['ma_10'] - df['ma_30']) / df['Close']
        df['ma_cross_20_60'] = (df['ma_20'] - df['ma_60']) / df['Close']

        # 4. Bollinger Bands %B
        bb_period = 20
        bb_std = 2
        df['bb_mid'] = df['Close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['Close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (bb_std * df['bb_std'])
        df['bb_pct_b'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 5. Volume ratio (vs 20-day avg)
        df['vol_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma_20']

        # 6. Price momentum (5d, 10d, 20d returns)
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_10d'] = df['Close'].pct_change(10)
        df['momentum_20d'] = df['Close'].pct_change(20)

        # 7. Volatility (20d rolling std)
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()

        # 選擇最終特徵（共 10 個）
        self.feature_names = [
            'rsi_14', 'macd_hist', 'ma_cross_10_30', 'ma_cross_20_60',
            'bb_pct_b', 'vol_ratio', 'momentum_5d', 'momentum_10d',
            'momentum_20d', 'volatility_20d'
        ]

        return df[self.feature_names]

    def _calculate_target(self, data: pd.DataFrame) -> pd.Series:
        """
        計算目標變數：未來 N 天報酬率 > 0 → 1, else 0

        Args:
            data: 價格數據

        Returns:
            Series of binary labels (0 or 1)
        """
        future_return = data['Close'].pct_change(self.forward_days).shift(-self.forward_days)
        target = (future_return > 0).astype(int)
        return target

    def train(self, data: pd.DataFrame):
        """
        訓練 XGBoost 模型，使用後 20% 資料作為 validation set 觸發 early stopping

        Args:
            data: 歷史價格數據
        """
        if len(data) < 100:
            print("[XGBoost] Insufficient data for training (need at least 100 days)")
            return

        # 計算特徵與目標
        features = self._calculate_features(data)
        target = self._calculate_target(data)

        # 合併，排除 NaN
        train_data = pd.concat([features, target.rename('target')], axis=1).dropna()

        if len(train_data) < 50:
            print(f"[XGBoost] Insufficient training data after removing NaN: {len(train_data)} rows")
            return

        X = train_data[self.feature_names].values
        y = train_data['target'].values

        # 分割 train / validation（後 20% 做 early stopping）
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_val) > 0:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            # 資料太少，不使用 validation set（需暫時移除 early_stopping_rounds）
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='logloss',
                verbosity=0,
            )
            self.model.fit(X_train, y_train, verbose=False)

        self.is_trained = True

        # 儲存 feature importances
        importances = self.model.feature_importances_
        self._feature_importances = dict(zip(self.feature_names, importances))

        print(f"[XGBoost] Model trained on {len(train_data)} samples "
              f"(train={split_idx}, val={len(X_val)})")
        top3 = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"[XGBoost] Top 3 features: {top3}")

    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成預測信號

        Returns:
            - signal: "BUY"/"SELL"/"HOLD"
            - confidence: 預測信心（0-100）
            - reason: 預測原因
            - probabilities: {"up": float, "down": float}
            - model: "xgboost"
        """
        # 如果模型未訓練，先訓練
        if not self.is_trained:
            self.train(data)

        if not self.is_trained:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "XGBoost model not trained (insufficient data)",
                "model": "xgboost",
            }

        # 計算最新數據的特徵
        features = self._calculate_features(data)
        latest_features = features.iloc[[-1]].dropna()

        if latest_features.empty or len(latest_features.columns) < len(self.feature_names):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "Insufficient feature data",
                "model": "xgboost",
            }

        X = latest_features[self.feature_names].values

        # 預測機率
        proba = self.model.predict_proba(X)[0]
        # Convert numpy types to Python float to avoid JSON serialization errors
        prob_down, prob_up = float(proba[0]), float(proba[1])

        # 決定信號
        if prob_up >= self.confidence_threshold:
            signal = "BUY"
            confidence = prob_up * 100
        elif prob_down >= self.confidence_threshold:
            signal = "SELL"
            confidence = prob_down * 100
        else:
            signal = "HOLD"
            confidence = max(prob_up, prob_down) * 100

        # 找出最重要的特徵作為原因
        top_features = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        reason = f"XGBoost prediction based on {', '.join([f[0] for f in top_features])}"

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "reason": reason,
            "probabilities": {
                "up": round(float(prob_up), 3),
                "down": round(float(prob_down), 3),
            },
            "model": "xgboost",
        }

    def feature_importance(self) -> Dict[str, float]:
        """
        回傳特徵重要性

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}

        return {k: round(float(v), 4) for k, v in self._feature_importances.items()}


# ============================================================================
# LightGBM Predictor
# ============================================================================

class LightGBMPredictor(BasePredictor):
    """LightGBM 梯度提升機器學習預測器"""

    def __init__(self, forward_days: int = 5, confidence_threshold: float = 0.55):
        """
        初始化 LightGBM 預測器

        Args:
            forward_days: 預測未來 N 天報酬率（default=5）
            confidence_threshold: 信心閾值（低於此值視為 HOLD）
        """
        super().__init__()
        self.forward_days = forward_days
        self.confidence_threshold = confidence_threshold

        # LightGBM 模型（延遲 import，避免未安裝時整個 server 崩潰）
        from lightgbm import LGBMClassifier
        self.model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )

        self.is_trained = False
        self.feature_names = []
        self._feature_importances = {}

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering：計算技術指標特徵（10 個指標，與 RandomForestPredictor 相同）

        Features:
        - RSI(14)
        - MACD histogram
        - MA crossover (10/30, 20/60)
        - Bollinger Band %B
        - Volume ratio (vs 20-day avg)
        - Price momentum (5d, 10d, 20d returns)
        - Volatility (20d rolling std)

        Returns:
            DataFrame with features
        """
        df = data.copy()

        # 1. RSI(14)
        df['rsi_14'] = calculate_rsi(df, 14)

        # 2. MACD
        macd_data = calculate_macd(df)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']

        # 3. MA crossover (10/30, 20/60)
        df['ma_10'] = calculate_ma(df, 10)
        df['ma_30'] = calculate_ma(df, 30)
        df['ma_20'] = calculate_ma(df, 20)
        df['ma_60'] = calculate_ma(df, 60)

        df['ma_cross_10_30'] = (df['ma_10'] - df['ma_30']) / df['Close']
        df['ma_cross_20_60'] = (df['ma_20'] - df['ma_60']) / df['Close']

        # 4. Bollinger Bands %B
        bb_period = 20
        bb_std = 2
        df['bb_mid'] = df['Close'].rolling(window=bb_period).mean()
        df['bb_std'] = df['Close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (bb_std * df['bb_std'])
        df['bb_pct_b'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 5. Volume ratio (vs 20-day avg)
        df['vol_ma_20'] = df['Volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma_20']

        # 6. Price momentum (5d, 10d, 20d returns)
        df['momentum_5d'] = df['Close'].pct_change(5)
        df['momentum_10d'] = df['Close'].pct_change(10)
        df['momentum_20d'] = df['Close'].pct_change(20)

        # 7. Volatility (20d rolling std)
        df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()

        # 選擇最終特徵（共 10 個）
        self.feature_names = [
            'rsi_14', 'macd_hist', 'ma_cross_10_30', 'ma_cross_20_60',
            'bb_pct_b', 'vol_ratio', 'momentum_5d', 'momentum_10d',
            'momentum_20d', 'volatility_20d'
        ]

        return df[self.feature_names]

    def _calculate_target(self, data: pd.DataFrame) -> pd.Series:
        """
        計算目標變數：未來 N 天報酬率 > 0 → 1, else 0

        Args:
            data: 價格數據

        Returns:
            Series of binary labels (0 or 1)
        """
        future_return = data['Close'].pct_change(self.forward_days).shift(-self.forward_days)
        target = (future_return > 0).astype(int)
        return target

    def train(self, data: pd.DataFrame):
        """
        訓練 LightGBM 模型，使用後 20% 資料作為 validation set 觸發 early stopping（callbacks 方式）

        Args:
            data: 歷史價格數據
        """
        if len(data) < 100:
            print("[LightGBM] Insufficient data for training (need at least 100 days)")
            return

        # 計算特徵與目標
        features = self._calculate_features(data)
        target = self._calculate_target(data)

        # 合併，排除 NaN
        train_data = pd.concat([features, target.rename('target')], axis=1).dropna()

        if len(train_data) < 50:
            print(f"[LightGBM] Insufficient training data after removing NaN: {len(train_data)} rows")
            return

        X = train_data[self.feature_names].values
        y = train_data['target'].values

        # 分割 train / validation（後 20% 做 early stopping）
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if len(X_val) > 0:
            from lightgbm import early_stopping, log_evaluation
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=-1)],
            )
        else:
            # 資料太少，不使用 validation set
            self.model.fit(X_train, y_train)

        self.is_trained = True

        # 儲存 feature importances
        importances = self.model.feature_importances_
        self._feature_importances = dict(zip(self.feature_names, importances))

        print(f"[LightGBM] Model trained on {len(train_data)} samples "
              f"(train={split_idx}, val={len(X_val)})")
        top3 = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"[LightGBM] Top 3 features: {top3}")

    def predict(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成預測信號

        Returns:
            - signal: "BUY"/"SELL"/"HOLD"
            - confidence: 預測信心（0-100）
            - reason: 預測原因
            - probabilities: {"up": float, "down": float}
            - model: "lightgbm"
        """
        # 如果模型未訓練，先訓練
        if not self.is_trained:
            self.train(data)

        if not self.is_trained:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "LightGBM model not trained (insufficient data)",
                "model": "lightgbm",
            }

        # 計算最新數據的特徵
        features = self._calculate_features(data)
        latest_features = features.iloc[[-1]].dropna()

        if latest_features.empty or len(latest_features.columns) < len(self.feature_names):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0,
                "reason": "Insufficient feature data",
                "model": "lightgbm",
            }

        X = latest_features[self.feature_names].values

        # 預測機率
        proba = self.model.predict_proba(X)[0]
        prob_down, prob_up = float(proba[0]), float(proba[1])

        # 決定信號
        if prob_up >= self.confidence_threshold:
            signal = "BUY"
            confidence = prob_up * 100
        elif prob_down >= self.confidence_threshold:
            signal = "SELL"
            confidence = prob_down * 100
        else:
            signal = "HOLD"
            confidence = max(prob_up, prob_down) * 100

        # 找出最重要的特徵作為原因
        top_features = sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
        reason = f"LightGBM prediction based on {', '.join([f[0] for f in top_features])}"

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(float(confidence), 2),
            "reason": reason,
            "probabilities": {
                "up": round(float(prob_up), 3),
                "down": round(float(prob_down), 3),
            },
            "model": "lightgbm",
        }

    def feature_importance(self) -> Dict[str, float]:
        """
        回傳特徵重要性

        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_trained:
            return {}

        return {k: round(float(v), 4) for k, v in self._feature_importances.items()}


# ============================================================================
# Backtrader Integration - Taiwan Stock Trading
# ============================================================================

class EquityObserver(bt.Observer):
    """自定義 observer 記錄每日權益值"""
    lines = ('equity',)
    
    def next(self):
        self.lines.equity[0] = self.datas[0]._env.broker.getvalue()


# ============================================================================
# 自定義風險調整指標 Analyzers
# ============================================================================

class SortinoRatioAnalyzer(bt.Analyzer):
    """
    Sortino Ratio Analyzer（只計算下行波動率）
    
    公式：
        Sortino = (Rp - Rf) / σ_downside
        σ_downside = sqrt(mean(min(Ri - Rf, 0)²))
    
    params:
        - riskfreerate: 年化無風險利率（預設 0.0）
        - annualize: 是否年化（預設 True，使用 252 個交易日）
    """
    params = (
        ('riskfreerate', 0.0),
        ('annualize', True),
    )
    
    def __init__(self):
        self.daily_returns = []
    
    def notify_cashvalue(self, cash, value):
        self._last_value = value
    
    def start(self):
        self._prev_value = self.strategy.broker.getvalue()
        self.daily_returns = []
    
    def next(self):
        current_value = self.strategy.broker.getvalue()
        if self._prev_value and self._prev_value != 0:
            daily_return = (current_value - self._prev_value) / self._prev_value
            self.daily_returns.append(daily_return)
        self._prev_value = current_value
    
    def get_analysis(self):
        if len(self.daily_returns) < 2:
            return {'sortino': 0.0}
        
        returns = np.array(self.daily_returns)
        rf_daily = self.p.riskfreerate / 252.0
        
        # 超額報酬
        excess_returns = returns - rf_daily
        mean_excess = np.mean(excess_returns)
        
        # 下行波動率（只考慮負超額報酬）
        downside_returns = np.minimum(excess_returns, 0.0)
        downside_variance = np.mean(downside_returns ** 2)
        downside_std = np.sqrt(downside_variance)
        
        if downside_std == 0 or np.isnan(downside_std):
            sortino = 0.0
        else:
            sortino = mean_excess / downside_std
            if self.p.annualize:
                sortino *= np.sqrt(252)
        
        return {'sortino': float(round(sortino, 4)) if not np.isnan(sortino) else 0.0}


class CalmarRatioAnalyzer(bt.Analyzer):
    """
    Calmar Ratio Analyzer
    
    公式：
        Calmar = Annualized Return / Maximum Drawdown
    
    需搭配 DrawDown analyzer 一起使用；此 Analyzer 直接計算年化報酬和最大回撤。
    """
    
    def __init__(self):
        self._start_value = None
        self._peak = None
        self._max_drawdown = 0.0
        self._days = 0
    
    def start(self):
        self._start_value = self.strategy.broker.getvalue()
        self._peak = self._start_value
        self._max_drawdown = 0.0
        self._days = 0
    
    def next(self):
        current_value = self.strategy.broker.getvalue()
        self._days += 1
        
        if current_value > self._peak:
            self._peak = current_value
        
        if self._peak > 0:
            drawdown = (self._peak - current_value) / self._peak * 100
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
    
    def get_analysis(self):
        if self._start_value is None or self._start_value == 0 or self._days < 2:
            return {'calmar': 0.0, 'annualized_return': 0.0, 'max_drawdown': 0.0}
        
        end_value = self.strategy.broker.getvalue()
        total_return = (end_value - self._start_value) / self._start_value
        
        # 年化報酬（複利）
        years = self._days / 252.0
        if years > 0:
            annualized_return = ((1 + total_return) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0
        
        # Calmar Ratio
        if self._max_drawdown > 0:
            calmar = annualized_return / self._max_drawdown
        else:
            calmar = 0.0
        
        return {
            'calmar': float(round(calmar, 4)) if not np.isnan(calmar) else 0.0,
            'annualized_return': float(round(annualized_return, 4)),
            'max_drawdown': float(round(self._max_drawdown, 4))
        }


class TaiwanStockCommission(bt.CommInfoBase):
    """
    台股交易成本計算
    買入：手續費 0.1425%
    賣出：手續費 0.1425% + 證交稅 0.3% = 0.4425%
    """
    params = (
        ('commission', 0.001425),  # 0.1425% 手續費
        ('tax', 0.003),             # 0.3% 證交稅（賣出）
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """計算手續費"""
        if size > 0:  # 買入
            return abs(size) * price * self.p.commission
        else:  # 賣出（含證交稅）
            return abs(size) * price * (self.p.commission + self.p.tax)


class BaseBacktestStrategy(bt.Strategy):
    """回測策略基礎類"""
    
    def __init__(self):
        self.order = None
        self.trades_list = []  # 記錄交易明細
        self.order_history = []  # 記錄所有訂單
        
    def notify_order(self, order):
        """訂單狀態通知"""
        if order.status in [order.Completed]:
            order_info = {
                'date': self.datas[0].datetime.date(0).strftime('%Y-%m-%d'),
                'type': 'BUY' if order.isbuy() else 'SELL',
                'price': order.executed.price,
                'size': abs(order.executed.size),
                'value': order.executed.value,
                'commission': order.executed.comm
            }
            self.order_history.append(order_info)
            
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}, Cost={order.executed.value:.2f}, Comm={order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED: Price={order.executed.price:.2f}, Size={order.executed.size}, Cost={order.executed.value:.2f}, Comm={order.executed.comm:.2f}')
        self.order = None
    
    def notify_trade(self, trade):
        """交易完成通知"""
        if trade.isclosed:
            self.log(f'TRADE CLOSED: P&L Gross={trade.pnl:.2f}, Net={trade.pnlcomm:.2f}')
            
            # 從訂單歷史中找到對應的買入和賣出訂單
            # 找最近的 BUY 和 SELL 訂單配對
            buy_orders = [o for o in self.order_history if o['type'] == 'BUY']
            sell_orders = [o for o in self.order_history if o['type'] == 'SELL']
            
            if buy_orders and sell_orders:
                last_buy = buy_orders[-1]
                last_sell = sell_orders[-1]
                
                entry_value = last_buy['price'] * last_buy['size']
                
                self.trades_list.append({
                    'entry_date': last_buy['date'],
                    'entry_price': round(last_buy['price'], 2),
                    'exit_date': last_sell['date'],
                    'exit_price': round(last_sell['price'], 2),
                    'shares': int(last_buy['size']),
                    'pnl': round(trade.pnlcomm, 2),
                    'pnl_percent': round((trade.pnlcomm / entry_value) * 100, 2) if entry_value != 0 else 0,
                    'commission': round(trade.commission, 2)
                })
    
    def log(self, txt):
        """日誌輸出"""
        dt = self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')
    
    def next(self):
        """策略邏輯（子類實作）"""
        pass


class MACrossoverStrategy(BaseBacktestStrategy):
    """均線交叉策略"""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        super().__init__()
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.slow_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        if self.order:  # 有未完成訂單
            return
        
        if not self.position:  # 無持倉
            if self.crossover > 0:  # 快線上穿慢線 -> 買入
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        
        else:  # 有持倉
            if self.crossover < 0:  # 快線下穿慢線 -> 賣出
                self.order = self.sell(size=self.position.size)


class RSIReversalStrategy(BaseBacktestStrategy):
    """RSI 反轉策略"""
    
    params = (
        ('rsi_period', 14),
        ('rsi_lower', 30),
        ('rsi_upper', 70),
    )
    
    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RSI(
            self.datas[0].close, 
            period=self.params.rsi_period
        )
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.rsi < self.params.rsi_lower:  # 超賣 -> 買入
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        
        else:
            if self.rsi > self.params.rsi_upper:  # 超買 -> 賣出
                self.order = self.sell(size=self.position.size)


class MACDSignalStrategy(BaseBacktestStrategy):
    """MACD 信號策略"""
    
    params = (
        ('fast_ema', 12),
        ('slow_ema', 26),
        ('signal_period', 9),
    )
    
    def __init__(self):
        super().__init__()
        self.macd = bt.indicators.MACD(
            self.datas[0].close,
            period_me1=self.params.fast_ema,
            period_me2=self.params.slow_ema,
            period_signal=self.params.signal_period
        )
        self.crossover = bt.indicators.CrossOver(
            self.macd.macd, 
            self.macd.signal
        )
    
    def next(self):
        if self.order:
            return
        
        if not self.position:
            if self.crossover > 0:  # MACD 上穿信號線 -> 買入
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
        
        else:
            if self.crossover < 0:  # MACD 下穿信號線 -> 賣出
                self.order = self.sell(size=self.position.size)


class RFStrategy(BaseBacktestStrategy):
    """Random Forest 機器學習策略"""
    
    params = (
        ('forward_days', 5),
        ('confidence_threshold', 0.50),  # 降低閾值，增加交易機會
        ('retrain_period', 60),  # 每 N 天重新訓練
    )
    
    def __init__(self):
        super().__init__()
        self.predictor = RandomForestPredictor(
            forward_days=self.params.forward_days,
            confidence_threshold=self.params.confidence_threshold
        )
        self.days_since_train = 0
        self.trained = False
    
    def next(self):
        if self.order:
            return
        
        # 每隔一段時間重新訓練模型
        self.days_since_train += 1
        if not self.trained or self.days_since_train >= self.params.retrain_period:
            # 取得當前可用的所有歷史數據
            current_data = self.datas[0]
            
            # 將 backtrader 數據轉換為 pandas DataFrame
            hist_data = pd.DataFrame({
                'Open': [current_data.open[i] for i in range(-len(current_data)+1, 1)],
                'High': [current_data.high[i] for i in range(-len(current_data)+1, 1)],
                'Low': [current_data.low[i] for i in range(-len(current_data)+1, 1)],
                'Close': [current_data.close[i] for i in range(-len(current_data)+1, 1)],
                'Volume': [current_data.volume[i] for i in range(-len(current_data)+1, 1)]
            })
            hist_data.index = pd.DatetimeIndex([current_data.datetime.date(i) for i in range(-len(current_data)+1, 1)])
            
            # 訓練模型
            self.predictor.train(hist_data)
            self.trained = True
            self.days_since_train = 0
        
        if not self.trained:
            return
        
        # 生成預測
        # 取得最近的數據用於預測
        recent_len = min(252, len(self.datas[0]))  # 取最近1年或可用的數據
        hist_data = pd.DataFrame({
            'Open': [self.datas[0].open[i] for i in range(-recent_len+1, 1)],
            'High': [self.datas[0].high[i] for i in range(-recent_len+1, 1)],
            'Low': [self.datas[0].low[i] for i in range(-recent_len+1, 1)],
            'Close': [self.datas[0].close[i] for i in range(-recent_len+1, 1)],
            'Volume': [self.datas[0].volume[i] for i in range(-recent_len+1, 1)]
        })
        hist_data.index = pd.DatetimeIndex([self.datas[0].datetime.date(i) for i in range(-recent_len+1, 1)])
        
        prediction = self.predictor.predict("STOCK", hist_data)
        signal = prediction.get('signal', 'HOLD')
        
        # 執行交易
        if not self.position:
            if signal == 'BUY':
                size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f"RF BUY signal (confidence: {prediction.get('confidence', 0):.2f})")
        
        else:
            if signal == 'SELL':
                self.order = self.sell(size=self.position.size)
                self.log(f"RF SELL signal (confidence: {prediction.get('confidence', 0):.2f})")


class BuyHoldStrategy(bt.Strategy):
    """買入持有基準策略"""
    
    def __init__(self):
        self.order = None
        self.bought = False
    
    def next(self):
        if not self.bought and not self.order:
            size = int(self.broker.get_cash() * 0.95 / self.data.close[0])
            if size > 0:
                self.order = self.buy(size=size)
                self.bought = True


class BacktestEngine:
    """Backtrader 回測引擎"""
    
    STRATEGY_MAP = {
        'ma_crossover': MACrossoverStrategy,
        'rsi_reversal': RSIReversalStrategy,
        'macd_signal': MACDSignalStrategy,
        'rf': RFStrategy,
    }

    STRATEGY_NAMES = {
        'ma_crossover': 'MA Crossover',
        'rsi_reversal': 'RSI Reversal',
        'macd_signal': 'MACD Signal',
        'rf': 'Random Forest ML',
    }

    # Phase 6: register HMM Filter strategy if backtest module is available
    if _BACKTEST_MODULE_AVAILABLE:
        STRATEGY_MAP['hmm_filter'] = _HMMFilterStrategy
        STRATEGY_NAMES['hmm_filter'] = 'HMM Filter Strategy'
    
    def __init__(self, symbol: str, start_date: str, end_date: str, 
                 initial_capital: float = 100000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
    def _calculate_sharpe_ratio(self, equity_values: List[float]) -> float:
        """
        手動計算 Sharpe Ratio
        公式：sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
        
        Args:
            equity_values: 每日權益值列表
            
        Returns:
            Sharpe Ratio（年化）
        """
        if len(equity_values) < 2:
            return 0.0
        
        # 過濾掉 NaN 值
        equity_array = np.array([v for v in equity_values if not np.isnan(v)])
        
        if len(equity_array) < 2:
            return 0.0
        
        # 計算日報酬率
        daily_returns = np.diff(equity_array) / equity_array[:-1]
        
        if len(daily_returns) == 0:
            return 0.0
        
        # 計算平均報酬和標準差
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        # 如果標準差為 0，返回 0
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        # 年化 Sharpe Ratio（假設一年 252 個交易日）
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        return sharpe if not np.isnan(sharpe) else 0.0
    
    def _fetch_data(self) -> pd.DataFrame:
        """下載股票數據"""
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=self.start_date, end=self.end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        # Backtrader 需要的欄位名稱（小寫）
        df.columns = [col.lower() for col in df.columns]
        return df
    
    def run_backtest(self, strategy_name: str, parameters: Dict) -> Dict[str, Any]:
        """執行回測"""
        print(f"\n{'='*60}")
        print(f"Running backtest: {strategy_name} on {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"{'='*60}\n")
        
        # 1. 準備數據
        df = self._fetch_data()
        
        if len(df) < 60:
            raise ValueError(f"Insufficient data: only {len(df)} days available (need at least 60)")
        
        data_feed = bt.feeds.PandasData(dataname=df)
        
        # 2. 初始化 Cerebro（策略回測）
        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)
        
        # 3. 設定策略
        strategy_class = self.STRATEGY_MAP[strategy_name]
        cerebro.addstrategy(strategy_class, **parameters)
        
        # 4. 設定初始資金
        cerebro.broker.set_cash(self.initial_capital)
        
        # 5. 設定台股交易成本
        cerebro.broker.addcommissioninfo(TaiwanStockCommission())
        
        # 6. 添加分析器
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                            riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(SortinoRatioAnalyzer, _name='sortino')
        cerebro.addanalyzer(CalmarRatioAnalyzer, _name='calmar')
        
        # 7. 記錄每日權益（使用自定義 observer）
        cerebro.addobserver(EquityObserver)
        
        # 8. 執行回測
        print("Running strategy backtest...")
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        strategy_result = results[0]
        end_value = cerebro.broker.getvalue()
        
        print(f"Strategy - Start: ${start_value:.2f}, End: ${end_value:.2f}, Return: {((end_value-start_value)/start_value*100):.2f}%")
        
        # 9. 執行 Buy & Hold 基準
        print("\nRunning buy-and-hold benchmark...")
        benchmark_result = self._run_benchmark(df)
        
        # 10. 執行 Walk-Forward 驗證（如果數據足夠）
        folds_result = []
        if len(df) >= 150:  # 至少半年數據（約150交易日）
            print("\nRunning walk-forward validation...")
            try:
                # 準備數據（需要大寫欄位名給預測器）
                df_for_validation = df.copy()
                df_for_validation.columns = [col.capitalize() for col in df_for_validation.columns]
                
                # 使用對應的預測器
                if strategy_name == 'rf':
                    validator_predictor = RandomForestPredictor()
                else:
                    validator_predictor = TechnicalPredictor()
                
                validator = WalkForwardValidator(
                    train_window=min(100, max(60, len(df) // 3)),  # 60-100天訓練窗口
                    test_window=min(21, len(df) // 10),  # 測試窗口不超過數據的1/10
                    step=min(21, len(df) // 10)
                )
                validation_results = validator.validate(validator_predictor, df_for_validation)
                folds_result = validation_results.get('folds', [])
                print(f"Walk-Forward: {len(folds_result)} folds completed")
            except Exception as e:
                print(f"Walk-Forward validation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                folds_result = []
        
        # 11. 整理結果
        return self._format_results(strategy_name, strategy_result, benchmark_result, cerebro, folds_result)
    
    def _run_benchmark(self, df: pd.DataFrame) -> Dict:
        """執行 Buy & Hold 基準回測"""
        cerebro_bh = bt.Cerebro()
        data_feed = bt.feeds.PandasData(dataname=df)
        cerebro_bh.adddata(data_feed)
        cerebro_bh.addstrategy(BuyHoldStrategy)
        cerebro_bh.broker.set_cash(self.initial_capital)
        cerebro_bh.broker.addcommissioninfo(TaiwanStockCommission())
        cerebro_bh.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro_bh.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro_bh.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                               riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
        cerebro_bh.addanalyzer(SortinoRatioAnalyzer, _name='sortino')
        cerebro_bh.addanalyzer(CalmarRatioAnalyzer, _name='calmar')
        cerebro_bh.addobserver(EquityObserver)
        
        start_value = cerebro_bh.broker.getvalue()
        results = cerebro_bh.run()
        strategy = results[0]
        end_value = cerebro_bh.broker.getvalue()
        
        print(f"Benchmark - Start: ${start_value:.2f}, End: ${end_value:.2f}, Return: {((end_value-start_value)/start_value*100):.2f}%")
        
        # 從 EquityObserver 取得權益曲線數據
        equity_observer = None
        for obs in strategy.observers:
            if isinstance(obs, EquityObserver):
                equity_observer = obs
                break
        
        if equity_observer is None:
            raise ValueError("EquityObserver not found in benchmark observers")
        
        # 取得權益值並過濾 NaN
        equity_values = [float(v) if not np.isnan(v) else self.initial_capital 
                        for v in equity_observer.lines.equity.array]
        
        # 計算 Sharpe Ratio（手動）
        sharpe_ratio = self._calculate_sharpe_ratio(equity_values)
        
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        max_dd = drawdown_analysis.get('max', {}).get('drawdown', 0)
        
        returns_analysis = strategy.analyzers.returns.get_analysis()
        total_return = returns_analysis.get('rtot', 0) * 100
        
        # 提取 Backtrader 內建 Sharpe Ratio
        sharpe_bt_analysis = strategy.analyzers.sharpe.get_analysis()
        sharpe_bt = sharpe_bt_analysis.get('sharperatio', None)
        if sharpe_bt is None or np.isnan(float(sharpe_bt)) if sharpe_bt is not None else False:
            sharpe_bt = float(sharpe_ratio)
        else:
            sharpe_bt = float(sharpe_bt)
        
        # 提取 Sortino Ratio
        sortino_analysis = strategy.analyzers.sortino.get_analysis()
        sortino = sortino_analysis.get('sortino', 0.0)
        
        # 提取 Calmar Ratio
        calmar_analysis = strategy.analyzers.calmar.get_analysis()
        calmar = calmar_analysis.get('calmar', 0.0)
        calmar_annualized_return = calmar_analysis.get('annualized_return', 0.0)
        
        return {
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(float(sharpe_bt), 2),
            'sortino_ratio': round(float(sortino), 2),
            'calmar_ratio': round(float(calmar), 2),
            'max_drawdown': round(float(max_dd), 2),
            'annualized_return': round(float(calmar_annualized_return), 2),
            'final_portfolio_value': round(end_value, 2),
            'equity_values': equity_values  # 保存用於後續 equity curve 繪製
        }
    
    def _format_results(self, strategy_name: str, strategy, benchmark, cerebro, folds=None) -> Dict:
        """格式化回測結果"""
        # 從 EquityObserver 取得權益曲線數據
        equity_observer = None
        for obs in strategy.observers:
            if isinstance(obs, EquityObserver):
                equity_observer = obs
                break
        
        if equity_observer is None:
            raise ValueError("EquityObserver not found in strategy observers")
        
        # 取得權益值並過濾 NaN
        equity_values = [float(v) if not np.isnan(v) else self.initial_capital 
                        for v in equity_observer.lines.equity.array]
        
        # 手動計算 Sharpe Ratio（作為備援）
        sharpe_manual = self._calculate_sharpe_ratio(equity_values)
        
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
        max_dd = drawdown_analysis.get('max', {}).get('drawdown', 0)
        
        returns_analysis = strategy.analyzers.returns.get_analysis()
        total_return = returns_analysis.get('rtot', 0) * 100
        
        trades_analysis = strategy.analyzers.trades.get_analysis()
        total_trades = trades_analysis.get('total', {}).get('closed', 0)
        
        final_value = cerebro.broker.getvalue()
        
        # 計算勝率
        won_trades = trades_analysis.get('won', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # 提取 Backtrader 內建 Sharpe Ratio（優先使用）
        sharpe_bt_analysis = strategy.analyzers.sharpe.get_analysis()
        sharpe_bt = sharpe_bt_analysis.get('sharperatio', None)
        if sharpe_bt is None or (isinstance(sharpe_bt, float) and np.isnan(sharpe_bt)):
            sharpe_ratio = float(sharpe_manual)
        else:
            sharpe_ratio = float(sharpe_bt)
        
        # 提取 Sortino Ratio
        sortino_analysis = strategy.analyzers.sortino.get_analysis()
        sortino_ratio = sortino_analysis.get('sortino', 0.0)
        
        # 提取 Calmar Ratio
        calmar_analysis = strategy.analyzers.calmar.get_analysis()
        calmar_ratio = calmar_analysis.get('calmar', 0.0)
        calmar_annualized_return = calmar_analysis.get('annualized_return', 0.0)
        
        # 提取交易明細
        trades_list = strategy.trades_list if hasattr(strategy, 'trades_list') else []
        
        # 建立權益曲線（使用完整的每日數據）
        equity_curve = self._build_equity_curve(strategy, benchmark, equity_values)
        
        print(f"\n{'='*60}")
        print(f"Backtest completed: {self.STRATEGY_NAMES[strategy_name]}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {calmar_annualized_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"{'='*60}\n")
        
        result = {
            'status': 'success',
            'data': {
                'strategy_name': self.STRATEGY_NAMES[strategy_name],
                'symbol': self.symbol,
                'period': {
                    'start': self.start_date,
                    'end': self.end_date
                },
                'performance': {
                    'total_return': round(total_return, 2),
                    'annualized_return': round(float(calmar_annualized_return), 2),
                    'sharpe_ratio': round(float(sharpe_ratio), 2),
                    'sortino_ratio': round(float(sortino_ratio), 2),
                    'calmar_ratio': round(float(calmar_ratio), 2),
                    'max_drawdown': round(float(max_dd), 2),
                    'win_rate': round(win_rate, 2),
                    'total_trades': int(total_trades),
                    'final_portfolio_value': round(final_value, 2)
                },
                'benchmark': benchmark,
                'trades': trades_list,
                'equity_curve': equity_curve
            }
        }
        
        # 添加 folds 欄位（如果有）
        if folds is not None and len(folds) > 0:
            result['data']['folds'] = folds

        # P3: 添加 RF 特徵重要性（如果策略有訓練過的 RandomForest predictor）
        if hasattr(strategy, 'predictor') and hasattr(strategy.predictor, '_feature_importances'):
            fi = strategy.predictor.feature_importance()
            if fi:
                result['data']['feature_importance'] = fi

        # Survivorship Bias 警告
        result['survivorship_bias_warning'] = (
            "⚠️ 回測數據來自 yfinance，僅包含當前上市股票。"
            "已下市/退市股票不在數據中，可能導致回測報酬率虛高 2-5%。"
            "此為已知限制，非策略真實績效。"
        )

        return result
    
    def _build_equity_curve(self, strategy, benchmark, equity_values: List[float]) -> List[Dict]:
        """
        建立權益曲線（完整每日數據）
        
        Args:
            strategy: Backtrader 策略實例
            benchmark: Benchmark 結果字典
            equity_values: 策略的每日權益值列表
            
        Returns:
            每日權益曲線數據點列表
        """
        # 獲取日期索引
        dates = [strategy.datas[0].datetime.date(i).strftime('%Y-%m-%d') 
                 for i in range(-len(equity_values) + 1, 1)]
        
        # 從 benchmark 獲取權益值（如果有的話）
        benchmark_equity = benchmark.get('equity_values', [])
        
        # 如果 benchmark 沒有完整數據，使用線性插值
        if len(benchmark_equity) < len(equity_values):
            start_val = self.initial_capital
            end_val = benchmark['final_portfolio_value']
            benchmark_equity = [
                start_val + (end_val - start_val) * i / (len(equity_values) - 1)
                for i in range(len(equity_values))
            ]
        
        # 建立數據點
        equity_curve = []
        for i, (date, value) in enumerate(zip(dates, equity_values)):
            # 過濾掉 NaN 值
            if np.isnan(value):
                continue
            
            benchmark_value = benchmark_equity[i] if i < len(benchmark_equity) else benchmark['final_portfolio_value']
            
            equity_curve.append({
                'date': date,
                'value': round(float(value), 2),
                'benchmark': round(float(benchmark_value), 2)
            })
        
        return equity_curve


# ============================================================================
# ============================================================================
# 模擬交易系統 (Simulated Trading System)
# ============================================================================

# 資料庫路徑
SIM_TRADING_DB = os.path.join(os.path.dirname(__file__), "sim_trading.db")

@contextmanager
def get_db():
    """資料庫連線上下文管理器"""
    conn = sqlite3.connect(SIM_TRADING_DB)
    conn.row_factory = sqlite3.Row  # 使查詢結果可以用欄位名訪問
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_sim_trading_db():
    """初始化模擬交易資料庫"""
    print("[DB] Initializing simulated trading database...")
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. accounts 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                initial_cash REAL NOT NULL,
                cash REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. positions 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE,
                UNIQUE(account_id, symbol)
            )
        """)
        
        # 3. transactions 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                commission REAL NOT NULL,
                tax REAL DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
    
    print("[DB] Simulated trading database initialized successfully")


class SimTradingCosts:
    """台灣股票交易成本計算（模擬交易專用）"""
    
    COMMISSION_RATE = 0.001425   # 手續費 0.1425%
    COMMISSION_DISCOUNT = 0.6    # 電子下單折扣
    TAX_RATE = 0.003             # 證交稅 0.3%（賣出）
    
    @classmethod
    def calculate_buy_cost(cls, price: float, quantity: int) -> float:
        """計算買入成本（含手續費）"""
        total_value = price * quantity
        commission = total_value * cls.COMMISSION_RATE * cls.COMMISSION_DISCOUNT
        return commission
    
    @classmethod
    def calculate_sell_cost(cls, price: float, quantity: int) -> float:
        """計算賣出成本（含手續費和證交稅）"""
        total_value = price * quantity
        commission = total_value * cls.COMMISSION_RATE * cls.COMMISSION_DISCOUNT
        tax = total_value * cls.TAX_RATE
        return commission + tax


# ============================================================================
# Position Sizing
# ============================================================================

def calculate_position_size(account_value: float, risk_pct: float = 0.02,
                            entry_price: float = 0, stop_loss_price: float = 0) -> dict:
    """
    固定比例 Position Sizing
    - risk_pct: 每筆交易最大風險（預設 2%）
    - 台灣成本: 0.1425% × 0.6 買 + (0.1425% × 0.6 + 0.3%) 賣
    """
    risk_amount = account_value * risk_pct
    if stop_loss_price > 0 and entry_price > 0:
        risk_per_share = abs(entry_price - stop_loss_price)
        # 加入台灣交易成本
        buy_cost = entry_price * 0.001425 * 0.6
        sell_cost = stop_loss_price * (0.001425 * 0.6 + 0.003)
        total_cost_per_share = buy_cost + sell_cost
        risk_per_share += total_cost_per_share
        shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        # 台股一張 = 1000 股，換算張數
        lots = shares // 1000
        shares = lots * 1000
    else:
        shares = int((account_value * risk_pct) / entry_price) if entry_price > 0 else 0
        lots = shares // 1000
        shares = lots * 1000

    return {
        "shares": shares,
        "lots": lots,
        "position_value": shares * entry_price,
        "position_pct": (shares * entry_price / account_value * 100) if account_value > 0 else 0,
        "risk_amount": risk_amount,
        "cost_estimate": {
            "buy_commission": shares * entry_price * 0.001425 * 0.6,
            "sell_commission": shares * entry_price * 0.001425 * 0.6,
            "sell_tax": shares * entry_price * 0.003,
        }
    }


# Pydantic Models
class SimAccountCreate(BaseModel):
    """建立模擬帳戶請求"""
    name: str = Field(..., min_length=1, max_length=100, description="帳戶名稱")
    initial_cash: float = Field(..., gt=0, description="初始資金（必須 > 0）")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "name": "我的第一個帳戶",
                "initial_cash": 100000
            }]
        }
    }


class SimTradeRequest(BaseModel):
    """交易請求"""
    symbol: str = Field(..., description="股票代碼（例如：AAPL, 2330.TW）")
    action: str = Field(..., pattern="^(buy|sell)$", description="動作：buy 或 sell")
    quantity: int = Field(..., gt=0, description="數量（必須 > 0）")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 10
            }]
        }
    }


class SimAccount(BaseModel):
    """模擬帳戶"""
    id: int
    name: str
    initial_cash: float
    cash: float
    created_at: str
    positions: List[Dict[str, Any]] = []
    net_value: float = 0.0  # 淨值（現金 + 持倉市值）
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": 1,
                "name": "我的第一個帳戶",
                "initial_cash": 100000,
                "cash": 50000,
                "created_at": "2026-02-17 12:00:00",
                "positions": [
                    {"symbol": "AAPL", "quantity": 10, "avg_cost": 150.0, "current_price": 155.0, "market_value": 1550.0, "pnl": 50.0}
                ],
                "net_value": 51550.0
            }]
        }
    }


class SimTransaction(BaseModel):
    """交易記錄"""
    id: int
    account_id: int
    symbol: str
    action: str
    quantity: int
    price: float
    commission: float
    tax: float
    timestamp: str


class PositionSizeRequest(BaseModel):
    """Position Sizing 計算請求"""
    account_value: float = Field(..., gt=0, description="帳戶總資金")
    risk_pct: float = Field(0.02, gt=0, le=1, description="每筆交易最大風險比例（預設 2%）")
    entry_price: float = Field(..., gt=0, description="買入價格")
    stop_loss_price: float = Field(0.0, ge=0, description="停損價格（0 表示不設停損）")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "account_value": 1000000,
                "risk_pct": 0.02,
                "entry_price": 600,
                "stop_loss_price": 570
            }]
        }
    }


# Stock Search Database (expandable)
# ============================================================================

STOCK_DATABASE = {
    # Taiwan stocks
    "2330.TW": {"name": "Taiwan Semiconductor Manufacturing", "market": "TW"},
    "2317.TW": {"name": "Hon Hai Precision Industry", "market": "TW"},
    "2454.TW": {"name": "MediaTek Inc.", "market": "TW"},
    "2881.TW": {"name": "Fubon Financial Holding", "market": "TW"},
    "2882.TW": {"name": "Cathay Financial Holding", "market": "TW"},
    
    # US stocks
    "AAPL": {"name": "Apple Inc.", "market": "US"},
    "MSFT": {"name": "Microsoft Corporation", "market": "US"},
    "GOOGL": {"name": "Alphabet Inc.", "market": "US"},
    "AMZN": {"name": "Amazon.com Inc.", "market": "US"},
    "TSLA": {"name": "Tesla Inc.", "market": "US"},
    "NVDA": {"name": "NVIDIA Corporation", "market": "US"},
    "META": {"name": "Meta Platforms Inc.", "market": "US"},
    "JPM": {"name": "JPMorgan Chase & Co.", "market": "US"},
    "V": {"name": "Visa Inc.", "market": "US"},
    "WMT": {"name": "Walmart Inc.", "market": "US"},
}


def search_stocks(query: str) -> List[Dict[str, str]]:
    """Search stocks by symbol or name"""
    query_lower = query.lower()
    results = []
    
    for symbol, info in STOCK_DATABASE.items():
        if (query_lower in symbol.lower() or 
            query_lower in info['name'].lower()):
            results.append({
                "symbol": symbol,
                "name": info['name'],
                "market": info['market']
            })
    
    return results


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "ml_models": {
            "xgboost": _XGBOOST_AVAILABLE,
            "lightgbm": _LIGHTGBM_AVAILABLE,
            "catboost": _CATBOOST_AVAILABLE,
        }
    }


@app.get("/")
async def root():
    """Serve the frontend HTML"""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Stock Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "stock_data": "/api/stock/{symbol}",
            "indicators": "/api/stock/{symbol}/indicators",
            "prediction": "/api/stock/{symbol}/predict",
            "search": "/api/search"
        }
    }


RANGE_MAP = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y",
    "5Y": "5y", "10Y": "10y", "YTD": "ytd", "MAX": "max",
}

@app.get("/api/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    period: str = Query(None, description="Time period (1mo,3mo,6mo,1y,2y...)"),
    range: str = Query(None, description="Time range (1M,3M,6M,1Y,2Y...)")
):
    # Accept both 'period' and 'range' params; map range to period
    if range and not period:
        period = RANGE_MAP.get(range.upper(), range.lower())
    if not period:
        period = "1y"
    """
    Get stock OHLCV data
    
    Parameters:
    - symbol: Stock symbol (e.g., AAPL, 2330.TW)
    - period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    """
    try:
        print(f"[API] Fetching stock data: {symbol} (period: {period})")
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}"
            )
        
        # Convert to JSON-serializable format
        data = []
        for index, row in hist.iterrows():
            data.append({
                "date": index.strftime("%Y-%m-%d"),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        info = stock.info
        
        print(f"[API] Successfully fetched {len(data)} records for {symbol}")
        
        # Calculate indicators inline so frontend gets everything in one call
        indicators_data = {}
        if len(hist) >= 5:
            ma5_vals = calculate_ma(hist, 5)
            ma20_vals = calculate_ma(hist, 20)
            ma60_vals = calculate_ma(hist, 60)
            rsi_vals = calculate_rsi(hist, 14)
            macd_vals = calculate_macd(hist)
            
            import math
            def clean(series, decimals=2):
                return [round(float(v), decimals) if not (math.isnan(v) if isinstance(v, float) else True) else None for v in series.tolist()]
            
            indicators_data = {
                "ma5": clean(ma5_vals),
                "ma20": clean(ma20_vals),
                "ma60": clean(ma60_vals),
                "rsi": clean(rsi_vals),
                "macd": {
                    "macd": clean(macd_vals["macd"], 4),
                    "signal": clean(macd_vals["signal"], 4),
                    "histogram": clean(macd_vals["histogram"], 4),
                }
            }
        
        return {
            "symbol": symbol,
            "period": period,
            "data": data,
            "indicators": indicators_data,
            "info": {
                "name": info.get("longName", symbol),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "N/A")
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch stock data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/indicators")
async def get_indicators(
    symbol: str,
    period: str = Query(None),
    range: str = Query(None)
):
    if range and not period:
        period = RANGE_MAP.get(range.upper(), range.lower())
    if not period:
        period = "3mo"
    """
    Get technical indicators for a stock
    
    Indicators:
    - MA5, MA20, MA60: Moving Averages
    - RSI14: Relative Strength Index
    - MACD: Moving Average Convergence Divergence
    """
    try:
        print(f"[API] Calculating indicators: {symbol} (period: {period})")
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty or len(hist) < 60:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for indicator calculation (need 60+ days)"
            )
        
        # Calculate indicators
        ma5 = calculate_ma(hist, 5)
        ma20 = calculate_ma(hist, 20)
        ma60 = calculate_ma(hist, 60)
        rsi = calculate_rsi(hist, 14)
        macd_data = calculate_macd(hist)
        
        # Build response
        indicators = []
        for index, row in hist.iterrows():
            date_str = index.strftime("%Y-%m-%d")
            indicators.append({
                "date": date_str,
                "close": round(row['Close'], 2),
                "ma5": round(ma5[index], 2) if not pd.isna(ma5[index]) else None,
                "ma20": round(ma20[index], 2) if not pd.isna(ma20[index]) else None,
                "ma60": round(ma60[index], 2) if not pd.isna(ma60[index]) else None,
                "rsi": round(rsi[index], 2) if not pd.isna(rsi[index]) else None,
                "macd": round(macd_data['macd'][index], 4) if not pd.isna(macd_data['macd'][index]) else None,
                "signal": round(macd_data['signal'][index], 4) if not pd.isna(macd_data['signal'][index]) else None,
                "histogram": round(macd_data['histogram'][index], 4) if not pd.isna(macd_data['histogram'][index]) else None
            })
        
        print(f"[API] Successfully calculated indicators for {symbol}")
        
        return {
            "symbol": symbol,
            "period": period,
            "indicators": indicators
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to calculate indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/{symbol}/predict")
async def predict_stock(symbol: str):
    """
    Get buy/sell prediction based on technical indicators
    
    Returns:
    - signal: BUY, SELL, or HOLD
    - confidence: 0-100 confidence score
    - reason: Explanation for the signal
    - indicators: Current indicator values
    """
    try:
        print(f"[API] Generating prediction for: {symbol}")
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period="3mo")
        
        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}"
            )
        
        prediction = predictor.predict(symbol, hist)
        
        print(f"[API] Prediction: {prediction['signal']} (confidence: {prediction['confidence']}%)")
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to generate prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/catboost/{symbol}")
async def predict_catboost(symbol: str):
    """
    Get buy/sell prediction using CatBoost ML model

    Returns:
    - signal: BUY, SELL, or HOLD
    - confidence: 0-100 confidence score
    - reason: Explanation for the signal (top 3 feature names)
    - probabilities: {"up": float, "down": float}
    - model: "catboost"
    """
    if not _CATBOOST_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="CatBoost is not installed. Run: pip install catboost"
        )

    try:
        print(f"[API] CatBoost prediction for: {symbol}")

        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")  # 需要更多資料以供訓練

        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}"
            )

        cb_predictor = CatBoostPredictor(forward_days=5, confidence_threshold=0.55)
        prediction = cb_predictor.predict(symbol, hist)

        print(f"[API] CatBoost Prediction: {prediction['signal']} "
              f"(confidence: {prediction['confidence']}%)")

        return prediction

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] CatBoost prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/xgboost/{symbol}")
async def predict_xgboost(symbol: str):
    """
    Get buy/sell prediction using XGBoost ML model

    Returns:
    - signal: BUY, SELL, or HOLD
    - confidence: 0-100 confidence score
    - reason: Explanation for the signal (top 3 feature names)
    - probabilities: {"up": float, "down": float}
    - model: "xgboost"
    """
    if not _XGBOOST_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="XGBoost is not installed. Run: pip install xgboost"
        )

    try:
        print(f"[API] XGBoost prediction for: {symbol}")

        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")

        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}"
            )

        xgb_predictor = XGBoostPredictor(forward_days=5, confidence_threshold=0.55)
        prediction = xgb_predictor.predict(symbol, hist)

        print(f"[API] XGBoost Prediction: {prediction['signal']} "
              f"(confidence: {prediction['confidence']}%)")

        return prediction

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] XGBoost prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/lightgbm/{symbol}")
async def predict_lightgbm(symbol: str):
    """
    Get buy/sell prediction using LightGBM ML model

    Returns:
    - signal: BUY, SELL, or HOLD
    - confidence: 0-100 confidence score
    - reason: Explanation for the signal (top 3 feature names)
    - probabilities: {"up": float, "down": float}
    - model: "lightgbm"
    """
    if not _LIGHTGBM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LightGBM is not installed. Run: pip install lightgbm"
        )

    try:
        print(f"[API] LightGBM prediction for: {symbol}")

        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")

        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}"
            )

        lgb_predictor = LightGBMPredictor(forward_days=5, confidence_threshold=0.55)
        prediction = lgb_predictor.predict(symbol, hist)

        print(f"[API] LightGBM Prediction: {prediction['signal']} "
              f"(confidence: {prediction['confidence']}%)")

        return prediction

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] LightGBM prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/ensemble/{symbol}")
async def predict_ensemble(symbol: str):
    """
    Get buy/sell prediction using 4-model ensemble (soft voting):
    RandomForest + XGBoost + LightGBM + CatBoost

    Returns:
    - signal: BUY, SELL, or HOLD (majority soft vote)
    - confidence: 0-100 averaged confidence score
    - models: dict of individual model predictions
    - ensemble_probabilities: {"up": float, "down": float} averaged across models
    """
    available_models = []
    if not _XGBOOST_AVAILABLE:
        available_models.append("xgboost not available")
    if not _LIGHTGBM_AVAILABLE:
        available_models.append("lightgbm not available")
    if not _CATBOOST_AVAILABLE:
        available_models.append("catboost not available")

    try:
        print(f"[API] Ensemble prediction for: {symbol}")

        stock = yf.Ticker(symbol)
        hist = stock.history(period="1y")

        if hist.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {symbol}"
            )

        model_predictions = {}
        prob_up_list = []
        prob_down_list = []

        # 1. Random Forest (always available)
        rf_predictor = RandomForestPredictor(forward_days=5, confidence_threshold=0.55)
        rf_pred = rf_predictor.predict(symbol, hist)
        model_predictions["random_forest"] = rf_pred
        rf_proba = rf_pred.get("probabilities", {})
        prob_up_list.append(rf_proba.get("up", 0.5))
        prob_down_list.append(rf_proba.get("down", 0.5))

        # 2. XGBoost
        if _XGBOOST_AVAILABLE:
            xgb_predictor = XGBoostPredictor(forward_days=5, confidence_threshold=0.55)
            xgb_pred = xgb_predictor.predict(symbol, hist)
            model_predictions["xgboost"] = xgb_pred
            xgb_proba = xgb_pred.get("probabilities", {})
            prob_up_list.append(xgb_proba.get("up", 0.5))
            prob_down_list.append(xgb_proba.get("down", 0.5))

        # 3. LightGBM
        if _LIGHTGBM_AVAILABLE:
            lgb_predictor = LightGBMPredictor(forward_days=5, confidence_threshold=0.55)
            lgb_pred = lgb_predictor.predict(symbol, hist)
            model_predictions["lightgbm"] = lgb_pred
            lgb_proba = lgb_pred.get("probabilities", {})
            prob_up_list.append(lgb_proba.get("up", 0.5))
            prob_down_list.append(lgb_proba.get("down", 0.5))

        # 4. CatBoost
        if _CATBOOST_AVAILABLE:
            cb_predictor = CatBoostPredictor(forward_days=5, confidence_threshold=0.55)
            cb_pred = cb_predictor.predict(symbol, hist)
            model_predictions["catboost"] = cb_pred
            cb_proba = cb_pred.get("probabilities", {})
            prob_up_list.append(cb_proba.get("up", 0.5))
            prob_down_list.append(cb_proba.get("down", 0.5))

        # Soft voting: average probabilities across all models
        avg_prob_up = sum(prob_up_list) / len(prob_up_list) if prob_up_list else 0.5
        avg_prob_down = sum(prob_down_list) / len(prob_down_list) if prob_down_list else 0.5

        # Determine ensemble signal (threshold=0.55)
        confidence_threshold = 0.55
        if avg_prob_up >= confidence_threshold:
            ensemble_signal = "BUY"
            ensemble_confidence = avg_prob_up * 100
        elif avg_prob_down >= confidence_threshold:
            ensemble_signal = "SELL"
            ensemble_confidence = avg_prob_down * 100
        else:
            ensemble_signal = "HOLD"
            ensemble_confidence = max(avg_prob_up, avg_prob_down) * 100

        print(f"[API] Ensemble Prediction: {ensemble_signal} "
              f"(confidence: {round(ensemble_confidence, 2)}%, models={len(prob_up_list)})")

        return {
            "symbol": symbol,
            "signal": ensemble_signal,
            "confidence": round(ensemble_confidence, 2),
            "reason": f"4-model ensemble soft voting ({len(prob_up_list)} models)",
            "ensemble_probabilities": {
                "up": round(avg_prob_up, 3),
                "down": round(avg_prob_down, 3),
            },
            "models": model_predictions,
            "model_count": len(prob_up_list),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Ensemble prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search")
async def search_stock(
    q: str = Query(..., min_length=1, description="Search query (symbol or name)")
):
    """
    Search for stocks by symbol or name
    
    Parameters:
    - q: Search query string
    """
    try:
        print(f"[API] Searching stocks: '{q}'")
        
        results = search_stocks(q)
        
        print(f"[API] Found {len(results)} results")
        
        return {
            "query": q,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        print(f"[ERROR] Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Position Sizing API Endpoint
# ============================================================================

@app.post("/api/position-size")
async def api_position_size(request: PositionSizeRequest):
    """
    計算固定比例 Position Sizing（2% 風險規則）

    根據帳戶資金、風險比例、進場價格和停損價格，
    計算台股建議部位大小（以張為單位，1張=1000股）。

    Body:
    - account_value: 帳戶總資金
    - risk_pct: 每筆交易最大風險比例（預設 0.02 = 2%）
    - entry_price: 買入價格
    - stop_loss_price: 停損價格（0 表示不設停損，改以 risk_pct 直接計算）
    """
    try:
        print(f"[API] Calculating position size: account={request.account_value}, "
              f"risk={request.risk_pct*100:.1f}%, entry={request.entry_price}, "
              f"stop={request.stop_loss_price}")

        result = calculate_position_size(
            account_value=request.account_value,
            risk_pct=request.risk_pct,
            entry_price=request.entry_price,
            stop_loss_price=request.stop_loss_price,
        )

        print(f"[API] Position size: {result['lots']} lots ({result['shares']} shares), "
              f"value={result['position_value']:.2f}, risk_amount={result['risk_amount']:.2f}")

        return {
            "status": "success",
            "input": {
                "account_value": request.account_value,
                "risk_pct": request.risk_pct,
                "entry_price": request.entry_price,
                "stop_loss_price": request.stop_loss_price,
            },
            "result": result,
        }

    except Exception as e:
        print(f"[ERROR] Position size calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 模擬交易 API Endpoints
# ============================================================================

@app.post("/api/sim/accounts", status_code=201)
async def create_sim_account(account: SimAccountCreate):
    """
    建立模擬帳戶
    
    Body:
    - name: 帳戶名稱（唯一）
    - initial_cash: 初始資金
    """
    try:
        print(f"[SIM] Creating account: {account.name} with ${account.initial_cash}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO accounts (name, initial_cash, cash) VALUES (?, ?, ?)",
                (account.name, account.initial_cash, account.initial_cash)
            )
            account_id = cursor.lastrowid
        
        print(f"[SIM] Account created: ID={account_id}")
        
        return {
            "id": account_id,
            "name": account.name,
            "initial_cash": account.initial_cash,
            "cash": account.initial_cash,
            "message": "帳戶建立成功"
        }
    
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail=f"帳戶名稱「{account.name}」已存在")
    except Exception as e:
        print(f"[ERROR] Failed to create account: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sim/accounts")
async def list_sim_accounts():
    """列出所有模擬帳戶（簡要資訊）"""
    try:
        print("[SIM] Listing all accounts")
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, initial_cash, cash, created_at FROM accounts ORDER BY created_at DESC")
            rows = cursor.fetchall()
        
        accounts = [
            {
                "id": row["id"],
                "name": row["name"],
                "initial_cash": row["initial_cash"],
                "cash": row["cash"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
        
        print(f"[SIM] Found {len(accounts)} accounts")
        
        return {
            "count": len(accounts),
            "accounts": accounts
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to list accounts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sim/accounts/{account_id}")
async def get_sim_account(account_id: int):
    """
    取得帳戶詳情（含持倉和即時淨值）
    
    即時市價從 yfinance 獲取
    """
    try:
        print(f"[SIM] Fetching account details: ID={account_id}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # 取得帳戶資訊
            cursor.execute("SELECT * FROM accounts WHERE id = ?", (account_id,))
            account_row = cursor.fetchone()
            
            if not account_row:
                raise HTTPException(status_code=404, detail=f"帳戶 ID {account_id} 不存在")
            
            # 取得持倉
            cursor.execute(
                "SELECT symbol, quantity, avg_cost FROM positions WHERE account_id = ? AND quantity > 0",
                (account_id,)
            )
            position_rows = cursor.fetchall()
        
        # 計算持倉市值（使用即時市價）
        positions = []
        total_market_value = 0.0
        
        for pos in position_rows:
            symbol = pos["symbol"]
            quantity = pos["quantity"]
            avg_cost = pos["avg_cost"]
            
            # 獲取即時市價
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    current_price = avg_cost  # 無法獲取市價時使用成本價
            except Exception as e:
                print(f"[WARN] Failed to fetch price for {symbol}: {str(e)}")
                current_price = avg_cost
            
            market_value = current_price * quantity
            cost_basis = avg_cost * quantity
            pnl = market_value - cost_basis
            pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            
            positions.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_cost": round(avg_cost, 2),
                "current_price": round(current_price, 2),
                "market_value": round(market_value, 2),
                "cost_basis": round(cost_basis, 2),
                "pnl": round(pnl, 2),
                "pnl_percent": round(pnl_percent, 2)
            })
            
            total_market_value += market_value
        
        # 計算淨值
        cash = account_row["cash"]
        net_value = cash + total_market_value
        
        account_info = {
            "id": account_row["id"],
            "name": account_row["name"],
            "initial_cash": account_row["initial_cash"],
            "cash": round(cash, 2),
            "created_at": account_row["created_at"],
            "positions": positions,
            "total_market_value": round(total_market_value, 2),
            "net_value": round(net_value, 2),
            "total_pnl": round(net_value - account_row["initial_cash"], 2),
            "total_pnl_percent": round((net_value - account_row["initial_cash"]) / account_row["initial_cash"] * 100, 2)
        }
        
        print(f"[SIM] Account details: net_value=${net_value:.2f}, positions={len(positions)}")
        
        return account_info
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to fetch account details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sim/accounts/{account_id}/trade")
async def execute_sim_trade(account_id: int, trade: SimTradeRequest):
    """
    執行模擬交易（買入或賣出）
    
    Body:
    - symbol: 股票代碼
    - action: "buy" 或 "sell"
    - quantity: 數量
    
    自動從 yfinance 取得即時市價，並計算交易成本
    """
    try:
        print(f"[SIM] Trade request: account_id={account_id}, {trade.action.upper()} {trade.quantity} {trade.symbol}")
        
        # 獲取即時市價
        try:
            ticker = yf.Ticker(trade.symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                raise HTTPException(status_code=400, detail=f"無法取得股票 {trade.symbol} 的市價")
            current_price = float(hist['Close'].iloc[-1])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"無法取得股票 {trade.symbol} 的市價：{str(e)}")
        
        print(f"[SIM] Current price for {trade.symbol}: ${current_price:.2f}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # 檢查帳戶是否存在
            cursor.execute("SELECT cash FROM accounts WHERE id = ?", (account_id,))
            account_row = cursor.fetchone()
            
            if not account_row:
                raise HTTPException(status_code=404, detail=f"帳戶 ID {account_id} 不存在")
            
            cash = account_row["cash"]
            
            # 執行交易
            if trade.action.lower() == "buy":
                # 買入
                commission = SimTradingCosts.calculate_buy_cost(current_price, trade.quantity)
                total_cost = current_price * trade.quantity + commission
                
                if total_cost > cash:
                    raise HTTPException(
                        status_code=400,
                        detail=f"現金不足：需要 ${total_cost:.2f}（含手續費），現有 ${cash:.2f}"
                    )
                
                # 更新現金
                new_cash = cash - total_cost
                cursor.execute("UPDATE accounts SET cash = ? WHERE id = ?", (new_cash, account_id))
                
                # 更新持倉（計算新的平均成本）
                cursor.execute(
                    "SELECT quantity, avg_cost FROM positions WHERE account_id = ? AND symbol = ?",
                    (account_id, trade.symbol)
                )
                pos_row = cursor.fetchone()
                
                if pos_row:
                    # 已有持倉，計算新平均成本
                    old_quantity = pos_row["quantity"]
                    old_avg_cost = pos_row["avg_cost"]
                    new_quantity = old_quantity + trade.quantity
                    
                    # 新平均成本 = (舊成本總額 + 新買入成本) / 新總數量
                    old_cost_basis = old_quantity * old_avg_cost
                    new_cost_basis = trade.quantity * current_price + commission
                    new_avg_cost = (old_cost_basis + new_cost_basis) / new_quantity
                    
                    cursor.execute(
                        "UPDATE positions SET quantity = ?, avg_cost = ? WHERE account_id = ? AND symbol = ?",
                        (new_quantity, new_avg_cost, account_id, trade.symbol)
                    )
                else:
                    # 新持倉
                    avg_cost = (current_price * trade.quantity + commission) / trade.quantity
                    cursor.execute(
                        "INSERT INTO positions (account_id, symbol, quantity, avg_cost) VALUES (?, ?, ?, ?)",
                        (account_id, trade.symbol, trade.quantity, avg_cost)
                    )
                
                # 記錄交易
                cursor.execute(
                    "INSERT INTO transactions (account_id, symbol, action, quantity, price, commission, tax) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (account_id, trade.symbol, "BUY", trade.quantity, current_price, commission, 0)
                )
                
                print(f"[SIM] BUY executed: {trade.quantity} @ ${current_price:.2f}, commission=${commission:.2f}")
                
                return {
                    "success": True,
                    "action": "BUY",
                    "symbol": trade.symbol,
                    "quantity": trade.quantity,
                    "price": round(current_price, 2),
                    "commission": round(commission, 2),
                    "total_cost": round(total_cost, 2),
                    "remaining_cash": round(new_cash, 2)
                }
            
            elif trade.action.lower() == "sell":
                # 賣出
                # 檢查持倉
                cursor.execute(
                    "SELECT quantity, avg_cost FROM positions WHERE account_id = ? AND symbol = ?",
                    (account_id, trade.symbol)
                )
                pos_row = cursor.fetchone()
                
                if not pos_row or pos_row["quantity"] < trade.quantity:
                    available = pos_row["quantity"] if pos_row else 0
                    raise HTTPException(
                        status_code=400,
                        detail=f"持倉不足：持有 {available} 股，欲賣出 {trade.quantity} 股"
                    )
                
                # 計算賣出成本（手續費 + 證交稅）
                sell_cost = SimTradingCosts.calculate_sell_cost(current_price, trade.quantity)
                total_proceeds = current_price * trade.quantity - sell_cost
                commission = sell_cost * (SimTradingCosts.COMMISSION_RATE * SimTradingCosts.COMMISSION_DISCOUNT) / (SimTradingCosts.COMMISSION_RATE * SimTradingCosts.COMMISSION_DISCOUNT + SimTradingCosts.TAX_RATE)
                tax = sell_cost - commission
                
                # 更新現金
                new_cash = cash + total_proceeds
                cursor.execute("UPDATE accounts SET cash = ? WHERE id = ?", (new_cash, account_id))
                
                # 更新持倉
                new_quantity = pos_row["quantity"] - trade.quantity
                if new_quantity > 0:
                    cursor.execute(
                        "UPDATE positions SET quantity = ? WHERE account_id = ? AND symbol = ?",
                        (new_quantity, account_id, trade.symbol)
                    )
                else:
                    # 清倉
                    cursor.execute(
                        "DELETE FROM positions WHERE account_id = ? AND symbol = ?",
                        (account_id, trade.symbol)
                    )
                
                # 記錄交易
                cursor.execute(
                    "INSERT INTO transactions (account_id, symbol, action, quantity, price, commission, tax) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (account_id, trade.symbol, "SELL", trade.quantity, current_price, commission, tax)
                )
                
                print(f"[SIM] SELL executed: {trade.quantity} @ ${current_price:.2f}, commission=${commission:.2f}, tax=${tax:.2f}")
                
                return {
                    "success": True,
                    "action": "SELL",
                    "symbol": trade.symbol,
                    "quantity": trade.quantity,
                    "price": round(current_price, 2),
                    "commission": round(commission, 2),
                    "tax": round(tax, 2),
                    "total_proceeds": round(total_proceeds, 2),
                    "remaining_cash": round(new_cash, 2)
                }
            
            else:
                raise HTTPException(status_code=400, detail=f"無效的動作：{trade.action}")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Trade execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sim/accounts/{account_id}/history")
async def get_sim_trade_history(account_id: int):
    """取得交易歷史記錄"""
    try:
        print(f"[SIM] Fetching trade history for account_id={account_id}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # 檢查帳戶是否存在
            cursor.execute("SELECT id FROM accounts WHERE id = ?", (account_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"帳戶 ID {account_id} 不存在")
            
            # 取得交易記錄
            cursor.execute(
                """
                SELECT id, symbol, action, quantity, price, commission, tax, timestamp
                FROM transactions
                WHERE account_id = ?
                ORDER BY timestamp DESC
                """,
                (account_id,)
            )
            rows = cursor.fetchall()
        
        transactions = [
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "action": row["action"],
                "quantity": row["quantity"],
                "price": round(row["price"], 2),
                "commission": round(row["commission"], 2),
                "tax": round(row["tax"], 2),
                "total_value": round(row["price"] * row["quantity"], 2),
                "timestamp": row["timestamp"]
            }
            for row in rows
        ]
        
        print(f"[SIM] Found {len(transactions)} transactions")
        
        return {
            "account_id": account_id,
            "count": len(transactions),
            "transactions": transactions
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to fetch trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sim/accounts/{account_id}")
async def delete_sim_account(account_id: int):
    """刪除模擬帳戶（級聯刪除持倉和交易記錄）"""
    try:
        print(f"[SIM] Deleting account: ID={account_id}")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # 檢查帳戶是否存在
            cursor.execute("SELECT name FROM accounts WHERE id = ?", (account_id,))
            account_row = cursor.fetchone()
            
            if not account_row:
                raise HTTPException(status_code=404, detail=f"帳戶 ID {account_id} 不存在")
            
            # 刪除帳戶（CASCADE 會自動刪除相關的 positions 和 transactions）
            cursor.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
        
        print(f"[SIM] Account deleted: {account_row['name']}")
        
        return {
            "success": True,
            "message": f"帳戶「{account_row['name']}」已刪除"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to delete account: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/strategies")
async def list_strategies():
    """
    列出所有可用的回測策略及其參數

    NOTE: 所有策略類別（MACrossoverStrategy, RSIReversalStrategy,
    MACDSignalStrategy, RFStrategy）均直接定義於 server.py。
    不存在 'strategies.rf_strategy' 這個獨立模組；
    若需匯入請使用 `from server import RFStrategy`，
    或直接透過 BacktestEngine.STRATEGY_MAP['rf'] 取得類別參照。

    Returns:
        List of available strategies with parameter specifications
    """
    try:
        print("[API] Fetching available backtest strategies")
        # BacktestEngine.STRATEGY_MAP 是策略的唯一權威來源；
        # 以下清單中的 name 欄位必須與 STRATEGY_MAP 的 key 對應。
        _registered_keys = set(BacktestEngine.STRATEGY_MAP.keys())
        
        strategies = [
            {
                "name": "ma_crossover",
                "display_name": "MA Crossover",
                "description": "均線交叉策略：快線上穿慢線時買入，下穿時賣出",
                "parameters": {
                    "fast_period": {
                        "type": "int",
                        "default": 10,
                        "min": 5,
                        "max": 50,
                        "description": "快速均線週期（天）"
                    },
                    "slow_period": {
                        "type": "int",
                        "default": 30,
                        "min": 20,
                        "max": 200,
                        "description": "慢速均線週期（天）"
                    }
                }
            },
            {
                "name": "rsi_reversal",
                "display_name": "RSI Reversal",
                "description": "RSI 反轉策略：RSI < 下界時買入（超賣），RSI > 上界時賣出（超買）",
                "parameters": {
                    "rsi_period": {
                        "type": "int",
                        "default": 14,
                        "min": 5,
                        "max": 30,
                        "description": "RSI 計算週期（天）"
                    },
                    "rsi_lower": {
                        "type": "int",
                        "default": 30,
                        "min": 10,
                        "max": 40,
                        "description": "RSI 下界（超賣閾值）"
                    },
                    "rsi_upper": {
                        "type": "int",
                        "default": 70,
                        "min": 60,
                        "max": 90,
                        "description": "RSI 上界（超買閾值）"
                    }
                }
            },
            {
                "name": "macd_signal",
                "display_name": "MACD Signal",
                "description": "MACD 信號策略：MACD 上穿信號線時買入，下穿時賣出",
                "parameters": {
                    "fast_ema": {
                        "type": "int",
                        "default": 12,
                        "min": 5,
                        "max": 20,
                        "description": "快速 EMA 週期"
                    },
                    "slow_ema": {
                        "type": "int",
                        "default": 26,
                        "min": 20,
                        "max": 50,
                        "description": "慢速 EMA 週期"
                    },
                    "signal_period": {
                        "type": "int",
                        "default": 9,
                        "min": 5,
                        "max": 15,
                        "description": "信號線週期"
                    }
                }
            },
            {
                "name": "rf",
                "display_name": "Random Forest ML",
                "description": "機器學習策略：使用 Random Forest 分類器預測未來報酬，結合 RSI/MACD/均線等技術指標特徵",
                "parameters": {
                    "forward_days": {
                        "type": "int",
                        "default": 5,
                        "min": 1,
                        "max": 20,
                        "description": "預測未來 N 天報酬率"
                    },
                    "confidence_threshold": {
                        "type": "float",
                        "default": 0.50,
                        "min": 0.40,
                        "max": 0.80,
                        "description": "信心閾值（低於此值視為 HOLD）"
                    },
                    "retrain_period": {
                        "type": "int",
                        "default": 60,
                        "min": 30,
                        "max": 120,
                        "description": "每 N 天重新訓練模型"
                    }
                }
            }
        ]

        # 驗證所有列出的策略均在 STRATEGY_MAP 中有對應的類別（防止清單漂移）
        for s in strategies:
            if s["name"] not in _registered_keys:
                print(f"[WARNING] Strategy '{s['name']}' listed but missing from BacktestEngine.STRATEGY_MAP")

        print(f"[API] Returning {len(strategies)} strategies")

        return {
            "count": len(strategies),
            "strategies": strategies
        }

    except Exception as e:
        print(f"[ERROR] Failed to fetch strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel, Field
from walk_forward import WalkForwardEngine

class BacktestRequest(BaseModel):
    """回測請求模型"""
    symbol: str = Field(..., description="股票代碼（台股需加 .TW 後綴）")
    strategy: str = Field(..., description="策略名稱")
    start_date: str = Field(..., description="回測起始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="回測結束日期 (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000, ge=1000, description="初始資金（預設 100,000）")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="策略參數")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "symbol": "2330.TW",
                "strategy": "ma_crossover",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 30
                }
            }]
        }
    }


class WalkForwardRequest(BaseModel):
    """Walk-Forward 驗證請求模型"""
    symbol: str = Field(..., description="股票代碼（台股需加 .TW 後綴）")
    strategy: str = Field(..., description="策略名稱")
    total_start: str = Field(..., description="總時間範圍起始日 (YYYY-MM-DD)")
    total_end: str = Field(..., description="總時間範圍結束日 (YYYY-MM-DD)")
    train_months: int = Field(default=6, ge=1, le=24, description="訓練窗口大小（月）")
    test_months: int = Field(default=1, ge=1, le=12, description="測試窗口大小（月）")
    initial_capital: float = Field(default=100000, ge=1000, description="初始資金（預設 100,000）")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="策略參數")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "symbol": "2330.TW",
                "strategy": "ma_crossover",
                "total_start": "2023-01-01",
                "total_end": "2024-12-31",
                "train_months": 6,
                "test_months": 1,
                "initial_capital": 100000,
                "parameters": {
                    "fast_period": 10,
                    "slow_period": 30
                }
            }]
        }
    }


@app.post("/api/backtest")
async def run_backtest(request: BacktestRequest):
    """
    執行回測
    
    Parameters:
    - symbol: 股票代碼（例如：2330.TW, AAPL）
    - strategy: 策略名稱（ma_crossover, rsi_reversal, macd_signal）
    - start_date: 回測起始日期
    - end_date: 回測結束日期
    - initial_capital: 初始資金
    - parameters: 策略參數（依策略而異）
    
    Returns:
        Backtest results including performance metrics, trades, and equity curve
    """
    try:
        print(f"[API] Backtest request: {request.strategy} on {request.symbol}")
        print(f"[API] Period: {request.start_date} to {request.end_date}")
        print(f"[API] Initial capital: ${request.initial_capital}")
        
        # 驗證日期格式和順序
        try:
            start = datetime.strptime(request.start_date, "%Y-%m-%d")
            end = datetime.strptime(request.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
            )
        
        if start >= end:
            raise HTTPException(
                status_code=400,
                detail="start_date must be before end_date"
            )
        
        # 檢查日期範圍是否合理（至少 60 天）
        days_diff = (end - start).days
        if days_diff < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Date range too short: {days_diff} days (need at least 60 days for meaningful backtest)"
            )
        
        # 驗證策略名稱
        if request.strategy not in BacktestEngine.STRATEGY_MAP:
            available = list(BacktestEngine.STRATEGY_MAP.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy '{request.strategy}'. Available strategies: {available}"
            )
        
        # 驗證初始資金
        if request.initial_capital < 1000:
            raise HTTPException(
                status_code=400,
                detail="Initial capital must be at least $1,000"
            )
        
        # 執行回測
        engine = BacktestEngine(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )
        
        result = engine.run_backtest(request.strategy, request.parameters)
        
        print(f"[API] Backtest completed successfully")
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        # 處理數據獲取錯誤（例如無效的 ticker）
        error_msg = str(e)
        if "No data found" in error_msg:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {request.symbol}. Check if the symbol is correct and data is available for the specified period."
            )
        elif "Insufficient data" in error_msg:
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        print(f"[ERROR] Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Backtest execution failed: {str(e)}"
        )


@app.post("/api/backtest/walk-forward")
async def run_walk_forward(request: WalkForwardRequest):
    """
    執行 Walk-Forward 驗證
    
    Walk-Forward 驗證是防止 look-ahead bias 的關鍵技術。
    它將時間範圍分成多個滾動窗口，每個窗口包含：
    - 訓練期：用於策略開發/參數優化（本實作中假設參數已優化）
    - 測試期：用於驗證策略表現（out-of-sample）
    
    窗口滾動前進，確保每個測試期都是真正的未來數據。
    
    Parameters:
    - symbol: 股票代碼（例如：2330.TW, AAPL）
    - strategy: 策略名稱（ma_crossover, rsi_reversal, macd_signal）
    - total_start: 總時間範圍起始日
    - total_end: 總時間範圍結束日
    - train_months: 訓練窗口大小（月）
    - test_months: 測試窗口大小（月）
    - initial_capital: 初始資金
    - parameters: 策略參數
    
    Returns:
        Walk-Forward 驗證結果，包含：
        - windows: 每個窗口的績效（return, sharpe, trades）
        - summary: 整體彙總績效
        - stability: 穩定性指標（各窗口績效的標準差）
    """
    try:
        print(f"[API] Walk-Forward request: {request.strategy} on {request.symbol}")
        print(f"[API] Total period: {request.total_start} to {request.total_end}")
        print(f"[API] Window: train={request.train_months}m, test={request.test_months}m")
        
        # 驗證日期格式和順序
        try:
            total_start = datetime.strptime(request.total_start, "%Y-%m-%d")
            total_end = datetime.strptime(request.total_end, "%Y-%m-%d")
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
            )
        
        if total_start >= total_end:
            raise HTTPException(
                status_code=400,
                detail="total_start must be before total_end"
            )
        
        # 檢查時間範圍是否足夠
        days_diff = (total_end - total_start).days
        min_days = (request.train_months + request.test_months) * 30
        
        if days_diff < min_days:
            raise HTTPException(
                status_code=400,
                detail=f"Time range too short: {days_diff} days. "
                       f"Need at least {min_days} days for train={request.train_months}m + test={request.test_months}m"
            )
        
        # 驗證策略名稱
        if request.strategy not in BacktestEngine.STRATEGY_MAP:
            available = list(BacktestEngine.STRATEGY_MAP.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy '{request.strategy}'. Available strategies: {available}"
            )
        
        # 驗證初始資金
        if request.initial_capital < 1000:
            raise HTTPException(
                status_code=400,
                detail="Initial capital must be at least $1,000"
            )
        
        # 獲取策略類別
        strategy_class = BacktestEngine.STRATEGY_MAP[request.strategy]
        
        # 執行 Walk-Forward 驗證
        engine = WalkForwardEngine(
            symbol=request.symbol,
            strategy_class=strategy_class,
            total_start=request.total_start,
            total_end=request.total_end,
            train_months=request.train_months,
            test_months=request.test_months,
            initial_capital=request.initial_capital
        )
        
        result = engine.run(strategy_params=request.parameters)
        
        print(f"[API] Walk-Forward validation completed successfully")
        print(f"[API] Total windows: {result['summary']['total_windows']}")
        print(f"[API] Average return: {result['summary']['average_return']:.2f}%")
        print(f"[API] Return stability: {result['summary']['return_std']:.2f}%")
        
        # Survivorship Bias 警告
        result['survivorship_bias_warning'] = (
            "⚠️ 回測數據來自 yfinance，僅包含當前上市股票。"
            "已下市/退市股票不在數據中，可能導致回測報酬率虛高 2-5%。"
            "此為已知限制，非策略真實績效。"
        )
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        if "No data found" in error_msg:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol: {request.symbol}. Check if the symbol is correct."
            )
        else:
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        print(f"[ERROR] Walk-Forward validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Walk-Forward validation failed: {str(e)}"
        )


# ============================================================================
# Walk-Forward Validation Framework
# ============================================================================

class WalkForwardValidator:
    """時間序列專用的前向驗證框架，防止 look-ahead bias"""
    
    def __init__(self, train_window: int = 252, test_window: int = 21, step: int = 21):
        """
        初始化 Walk-Forward Validator
        
        Args:
            train_window: 訓練窗口大小（交易日，約1年 = 252天）
            test_window: 測試窗口大小（交易日，約1個月 = 21天）
            step: 滾動步長（交易日）
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
    
    def split(self, data: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        產生 (train, test) splits，嚴格時間順序
        
        Args:
            data: 完整的時間序列數據（必須按時間排序）
            
        Returns:
            List of (train_df, test_df) tuples
        """
        splits = []
        
        # 確保數據按時間排序
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        data = data.sort_index()
        
        # 滾動窗口分割
        for i in range(self.train_window, len(data) - self.test_window + 1, self.step):
            train = data.iloc[i - self.train_window:i]
            test = data.iloc[i:i + self.test_window]
            
            if len(train) == self.train_window and len(test) == self.test_window:
                splits.append((train, test))
        
        return splits
    
    def validate(self, predictor: BasePredictor, data: pd.DataFrame) -> dict:
        """
        對 predictor 執行完整 walk-forward 驗證，回傳各 fold 結果
        
        Args:
            predictor: 預測器實例（實現 BasePredictor 接口）
            data: 完整的歷史數據
            
        Returns:
            Dictionary with validation results:
            - folds: 每個 fold 的詳細結果
            - aggregate: 彙總統計（sharpe, max_drawdown, win_rate, avg_return）
        """
        splits = self.split(data)
        
        if len(splits) == 0:
            return {
                "folds": [],
                "aggregate": {
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "avg_return": 0.0,
                    "total_folds": 0
                }
            }
        
        folds = []
        all_returns = []
        all_sharpes = []
        all_drawdowns = []
        all_predictions = []
        
        for fold_idx, (train, test) in enumerate(splits):
            # 在訓練集上更新預測器（如果支持）
            if hasattr(predictor, 'update'):
                predictor.update(train)
            
            # 在測試集上進行預測
            predictions = []
            for date_idx in range(len(test)):
                current_data = pd.concat([train, test.iloc[:date_idx+1]])
                pred = predictor.predict("UNKNOWN", current_data)
                predictions.append({
                    "date": test.index[date_idx].strftime("%Y-%m-%d"),
                    "signal": pred.get("signal", "HOLD"),
                    "confidence": pred.get("confidence", 0)
                })
            
            # 計算該 fold 的績效指標
            test_returns = test['Close'].pct_change().dropna()
            fold_return = (test['Close'].iloc[-1] / test['Close'].iloc[0] - 1) * 100
            
            # 計算 Sharpe ratio
            if len(test_returns) > 1:
                sharpe = (test_returns.mean() / test_returns.std()) * np.sqrt(252) if test_returns.std() > 0 else 0.0
            else:
                sharpe = 0.0
            
            # 計算 Max Drawdown
            cumulative = (1 + test_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative / running_max - 1) * 100
            max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
            
            # 計算勝率（基於日收益）
            wins = (test_returns > 0).sum()
            total = len(test_returns)
            win_rate = (wins / total * 100) if total > 0 else 0.0
            
            fold_result = {
                "fold": fold_idx + 1,
                "train_period": {
                    "start": train.index[0].strftime("%Y-%m-%d"),
                    "end": train.index[-1].strftime("%Y-%m-%d")
                },
                "test_period": {
                    "start": test.index[0].strftime("%Y-%m-%d"),
                    "end": test.index[-1].strftime("%Y-%m-%d")
                },
                "metrics": {
                    "return": round(float(fold_return), 2),
                    "sharpe": round(float(sharpe), 2),
                    "max_drawdown": round(float(max_dd), 2),
                    "win_rate": round(float(win_rate), 2)
                },
                "predictions": predictions
            }
            
            folds.append(fold_result)
            all_returns.append(fold_return)
            all_sharpes.append(sharpe)
            all_drawdowns.append(max_dd)
        
        # 彙總統計
        aggregate = {
            "avg_return": round(float(np.mean(all_returns)), 2) if all_returns else 0.0,
            "sharpe": round(float(np.mean(all_sharpes)), 2) if all_sharpes else 0.0,
            "max_drawdown": round(float(np.min(all_drawdowns)), 2) if all_drawdowns else 0.0,
            "win_rate": round(float(np.mean([f["metrics"]["win_rate"] for f in folds])), 2) if folds else 0.0,
            "return_std": round(float(np.std(all_returns)), 2) if all_returns else 0.0,
            "total_folds": len(folds)
        }
        
        return {
            "folds": folds,
            "aggregate": aggregate
        }


# ============================================================================
# Transaction Cost Simulator (Taiwan-specific)
# ============================================================================

class TaiwanTradingCosts:
    """台灣股市交易成本模擬"""
    
    COMMISSION_RATE = 0.001425   # 券商手續費 0.1425%
    COMMISSION_DISCOUNT = 0.6    # 常見電子下單折扣
    TAX_RATE = 0.003             # 證交稅 0.3%（賣出時收）
    ETF_TAX_RATE = 0.001         # ETF 證交稅 0.1%
    
    def round_trip_cost(self, is_etf: bool = False) -> float:
        """
        計算一次完整買賣的成本率
        
        Args:
            is_etf: 是否為 ETF（稅率不同）
            
        Returns:
            成本率（小數形式，例如 0.004 = 0.4%）
        """
        # 買入手續費 + 賣出手續費
        commission = self.COMMISSION_RATE * self.COMMISSION_DISCOUNT * 2
        
        # 賣出證交稅
        tax = self.ETF_TAX_RATE if is_etf else self.TAX_RATE
        
        return commission + tax
    
    def apply_costs(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        將交易成本套用到交易記錄上
        
        Args:
            trades: 交易記錄 DataFrame，必須包含以下欄位：
                   - date: 交易日期
                   - action: 'buy' 或 'sell'
                   - price: 交易價格
                   - shares: 股數
                   - ticker: 股票代碼（用於判斷是否為 ETF）
        
        Returns:
            加上 cost 和 net_return 欄位的 DataFrame
        """
        if trades.empty:
            return trades
        
        # 確保必要欄位存在
        required_cols = ['date', 'action', 'price', 'shares', 'ticker']
        missing_cols = [col for col in required_cols if col not in trades.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        trades = trades.copy()
        
        # 計算每筆交易的成本
        costs = []
        for idx, row in trades.iterrows():
            trade_value = row['price'] * row['shares']
            is_etf = 'ETF' in row['ticker'].upper() or row['ticker'].startswith('00')
            
            if row['action'].lower() == 'buy':
                # 買入只有手續費
                commission = trade_value * self.COMMISSION_RATE * self.COMMISSION_DISCOUNT
                cost = commission
            elif row['action'].lower() == 'sell':
                # 賣出有手續費 + 證交稅
                commission = trade_value * self.COMMISSION_RATE * self.COMMISSION_DISCOUNT
                tax = trade_value * (self.ETF_TAX_RATE if is_etf else self.TAX_RATE)
                cost = commission + tax
            else:
                cost = 0.0
            
            costs.append(cost)
        
        trades['cost'] = costs
        
        # 計算淨收益（需要配對買賣）
        if 'net_return' not in trades.columns:
            net_returns = [np.nan] * len(trades)
            
            # 簡單配對：假設每個 sell 對應前一個 buy
            buy_stack = []
            for idx, row in trades.iterrows():
                if row['action'].lower() == 'buy':
                    buy_stack.append({
                        'index': idx,
                        'price': row['price'],
                        'shares': row['shares'],
                        'cost': row['cost']
                    })
                elif row['action'].lower() == 'sell' and buy_stack:
                    buy_info = buy_stack.pop()
                    
                    # 計算淨收益
                    buy_value = buy_info['price'] * buy_info['shares']
                    sell_value = row['price'] * row['shares']
                    gross_profit = sell_value - buy_value
                    total_cost = buy_info['cost'] + row['cost']
                    net_profit = gross_profit - total_cost
                    
                    net_returns[idx] = net_profit
            
            trades['net_return'] = net_returns
        
        return trades


# ============================================================================
# Buy-and-Hold Baseline
# ============================================================================

class BuyAndHoldBaseline:
    """買入持有基準線，所有策略必須打敗這個才有意義"""
    
    def __init__(self, cost_model: TaiwanTradingCosts = None):
        """
        初始化基準線計算器
        
        Args:
            cost_model: 交易成本模型（默認使用台灣股市成本）
        """
        self.cost_model = cost_model or TaiwanTradingCosts()
    
    def calculate(self, data: pd.DataFrame, initial_capital: float = 1000000) -> dict:
        """
        計算買入持有的績效指標
        
        Args:
            data: 價格數據 DataFrame（必須有 'Close' 欄位和 DatetimeIndex）
            initial_capital: 初始資金（默認 1,000,000）
        
        Returns:
            Dictionary with metrics:
            - total_return: 總報酬率 (%)
            - annualized_return: 年化報酬率 (%)
            - sharpe: Sharpe ratio（年化）
            - max_drawdown: 最大回撤 (%)
            - final_value: 最終資產價值
        """
        if data.empty or 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # 確保數據按時間排序
        data = data.sort_index()
        
        # 計算可買入股數（扣除買入手續費）
        first_price = data['Close'].iloc[0]
        is_etf = False  # 可以從 ticker 判斷，這裡簡化
        buy_commission_rate = self.cost_model.COMMISSION_RATE * self.cost_model.COMMISSION_DISCOUNT
        
        # 買入後剩餘資金
        shares = int((initial_capital * (1 - buy_commission_rate)) / first_price)
        buy_cost = shares * first_price
        buy_commission = buy_cost * buy_commission_rate
        
        # 計算每日資產價值
        portfolio_values = data['Close'] * shares
        
        # 最終賣出（扣除賣出成本）
        final_price = data['Close'].iloc[-1]
        sell_value = shares * final_price
        sell_commission = sell_value * buy_commission_rate
        sell_tax = sell_value * (self.cost_model.ETF_TAX_RATE if is_etf else self.cost_model.TAX_RATE)
        
        final_value = sell_value - sell_commission - sell_tax
        
        # 計算總報酬率
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        
        # 計算年化報酬率
        days = (data.index[-1] - data.index[0]).days
        years = days / 365.25
        if years > 0:
            annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0.0
        
        # 計算 Sharpe ratio
        daily_returns = data['Close'].pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # 計算 Max Drawdown
        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values / cumulative_max - 1) * 100
        max_drawdown = drawdown.min()
        
        return {
            "total_return": round(float(total_return), 2),
            "annualized_return": round(float(annualized_return), 2),
            "sharpe": round(float(sharpe), 2),
            "max_drawdown": round(float(max_drawdown), 2),
            "final_value": round(float(final_value), 2),
            "initial_capital": round(float(initial_capital), 2),
            "shares_bought": int(shares),
            "buy_price": round(float(first_price), 2),
            "sell_price": round(float(final_price), 2),
            "total_costs": round(float(buy_commission + sell_commission + sell_tax), 2)
        }


# ============================================================================
# Enhanced Backtest Endpoint with Walk-Forward & Baseline
# ============================================================================

class EnhancedBacktestRequest(BaseModel):
    """增強版回測請求（包含 Walk-Forward 和基準線比較）"""
    ticker: str = Field(..., description="股票代碼（台股需加 .TW 後綴）")
    start: str = Field(..., description="回測起始日期 (YYYY-MM-DD)")
    end: str = Field(..., description="回測結束日期 (YYYY-MM-DD)")
    strategy: str = Field(default="technical", description="策略類型（technical/ma_crossover/rsi_reversal/macd_signal）")
    initial_capital: float = Field(default=1000000, ge=1000, description="初始資金")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "ticker": "2330.TW",
                "start": "2024-01-01",
                "end": "2025-12-31",
                "strategy": "technical",
                "initial_capital": 1000000
            }]
        }
    }


@app.post("/backtest")
async def enhanced_backtest(request: EnhancedBacktestRequest):
    """
    增強版回測 API（包含 Walk-Forward 驗證和 Buy-and-Hold 基準線）
    
    POST body: {
        "ticker": "2330.TW",
        "start": "2024-01-01",
        "end": "2025-12-31",
        "strategy": "technical",
        "initial_capital": 1000000
    }
    
    回傳: {
        "strategy_results": {...},  # 策略績效
        "baseline": {...},          # Buy-and-Hold 基準
        "comparison": {...},        # 策略 vs 基準比較
        "folds": [...]              # Walk-Forward 各 fold 結果
    }
    """
    try:
        print(f"[API] Enhanced backtest: {request.strategy} on {request.ticker}")
        print(f"[API] Period: {request.start} to {request.end}")
        
        # 1. 驗證日期
        try:
            start_dt = pd.Timestamp(request.start)
            end_dt = pd.Timestamp(request.end)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
            )
        
        if start_dt >= end_dt:
            raise HTTPException(
                status_code=400,
                detail="start date must be before end date"
            )
        
        # 2. 下載數據
        print(f"[API] Fetching data for {request.ticker}...")
        ticker = yf.Ticker(request.ticker)
        data = ticker.history(start=request.start, end=request.end)
        
        if data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {request.ticker}"
            )
        
        if len(data) < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: only {len(data)} days (need at least 60)"
            )
        
        print(f"[API] Fetched {len(data)} days of data")
        
        # 3. 執行 Buy-and-Hold 基準線
        print("[API] Calculating buy-and-hold baseline...")
        baseline_calculator = BuyAndHoldBaseline()
        baseline_results = baseline_calculator.calculate(data, request.initial_capital)
        
        # 4. 選擇預測器（根據 strategy 參數）
        if request.strategy == "technical":
            strategy_predictor = TechnicalPredictor()
        else:
            # 其他策略類型可以在這裡擴展
            strategy_predictor = TechnicalPredictor()
        
        # 5. 執行 Walk-Forward 驗證
        print("[API] Running walk-forward validation...")
        validator = WalkForwardValidator(
            train_window=252,  # 約1年訓練
            test_window=21,    # 約1個月測試
            step=21            # 每月滾動
        )
        
        validation_results = validator.validate(strategy_predictor, data)
        
        # 6. 計算策略整體績效（使用 aggregate 結果）
        aggregate = validation_results.get("aggregate", {})
        strategy_results = {
            "total_return": aggregate.get("avg_return", 0.0),
            "sharpe_ratio": aggregate.get("sharpe", 0.0),
            "max_drawdown": aggregate.get("max_drawdown", 0.0),
            "win_rate": aggregate.get("win_rate", 0.0),
            "return_volatility": aggregate.get("return_std", 0.0),
            "total_folds": aggregate.get("total_folds", 0)
        }
        
        # 7. 比較策略 vs 基準
        comparison = {
            "return_difference": round(strategy_results["total_return"] - baseline_results["total_return"], 2),
            "sharpe_difference": round(strategy_results["sharpe_ratio"] - baseline_results["sharpe"], 2),
            "outperformance": strategy_results["total_return"] > baseline_results["total_return"],
            "risk_adjusted_outperformance": strategy_results["sharpe_ratio"] > baseline_results["sharpe"]
        }
        
        print(f"[API] Backtest completed")
        print(f"[API] Strategy return: {strategy_results['total_return']:.2f}%")
        print(f"[API] Baseline return: {baseline_results['total_return']:.2f}%")
        print(f"[API] Outperformance: {comparison['return_difference']:.2f}%")
        
        return {
            "status": "success",
            "ticker": request.ticker,
            "period": {
                "start": request.start,
                "end": request.end
            },
            "strategy_results": strategy_results,
            "baseline": baseline_results,
            "comparison": comparison,
            "folds": validation_results["folds"],
            "survivorship_bias_warning": (
                "⚠️ 回測數據來自 yfinance，僅包含當前上市股票。"
                "已下市/退市股票不在數據中，可能導致回測報酬率虛高 2-5%。"
                "此為已知限制，非策略真實績效。"
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Enhanced backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed: {str(e)}"
        )


# ============================================================================
# Phase 3.5 — Purged Walk-Forward Validation Endpoint
# ============================================================================


@app.post("/api/validate/walk-forward")
async def validate_walk_forward(
    symbol: str = Query(..., description="Stock symbol e.g. 2330.TW"),
    period: str = Query("3y", description="Data period"),
    train_window: int = Query(252),
    test_window: int = Query(21),
    label_horizon: int = Query(5),
    embargo_bars: int = Query(5),
):
    """
    Purged Walk-Forward Validation (Phase 3.5)

    使用 Purge + Embargo 機制防止 look-ahead bias 的嚴格 Walk-Forward 驗證。
    每個 fold 包含：ML 模型評估（ensemble accuracy, ROC-AUC）+ Backtrader 回測績效。

    Parameters:
    - symbol: 股票代碼（例如：2330.TW, AAPL）
    - period: 資料期間（例如：1y, 2y, 3y）
    - train_window: 訓練窗口長度（bars，預設 252 = 1年）
    - test_window: 測試窗口長度（bars，預設 21 = 1個月）
    - label_horizon: 標籤預測範圍（bars，預設 5）
    - embargo_bars: Purge 後的緩衝 bars（預設 5）

    Returns:
        WalkForwardReport.to_dict() 包含：
        - symbol, run_timestamp, config
        - folds: 每個 fold 的 ML 指標 + Backtrader 績效
        - summary: 彙總統計
    """
    try:
        # 延遲 import，避免修改現有 import 區塊
        from validation.run_walk_forward import PurgedWalkForwardRunner
        from config.walk_forward import WalkForwardConfig

        print(
            f"[API] validate_walk_forward: symbol={symbol}, period={period}, "
            f"train={train_window}, test={test_window}, "
            f"label_horizon={label_horizon}, embargo={embargo_bars}"
        )

        # 1. 建立設定
        cfg = WalkForwardConfig(
            train_window=train_window,
            test_window=test_window,
            step_size=test_window,       # step = test_window（滾動一個測試窗口）
            label_horizon=label_horizon,
            embargo_bars=embargo_bars,
        )
        cfg.validate()

        # 2. 下載資料
        print(f"[API] Downloading {symbol} data (period={period})...")
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=period)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol '{symbol}'"
            )

        min_bars = train_window + test_window + label_horizon + embargo_bars + 50
        if len(df) < min_bars:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Insufficient data: {len(df)} bars for {symbol} (period={period}). "
                    f"Need at least {min_bars} bars for the given configuration."
                )
            )

        print(
            f"[API] Data fetched: {len(df)} bars  "
            f"({df.index[0].date()} → {df.index[-1].date()})"
        )

        # 3. 執行 Purged Walk-Forward
        runner = PurgedWalkForwardRunner(cfg)
        report = runner.run(df, symbol=symbol, verbose=True, run_backtrader=True)

        # 4. 序列化並回傳
        result = report.to_dict()
        result = sanitize_numpy(result)   # 確保 numpy 型態轉為 Python 原生型態

        print(
            f"[API] validate_walk_forward done: {len(report.folds)} folds, "
            f"summary={report.summary}"
        )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Walk-forward validation failed: {str(e)}"
        )


# ============================================================================
# Phase 4 — CPCV（Combinatorially Purged Cross-Validation）Endpoint
# ============================================================================


@app.post("/api/validate/cpcv")
async def validate_cpcv(
    symbol: str = Query(..., description="Stock symbol e.g. 2330.TW"),
    period: str = Query("5y", description="Data period（建議 5y，CPCV 需要更多資料）"),
    n_groups: int = Query(6, description="Total groups N（預設 6）"),
    k_test_groups: int = Query(2, description="Test groups per fold k（預設 2）"),
    label_horizon: int = Query(5, description="Label forward-look bars（purge 範圍）"),
    embargo_bars: int = Query(5, description="Embargo buffer bars"),
):
    """
    Combinatorially Purged Cross-Validation (Phase 4)

    CPCV 透過組合學生成 C(N,k) 個 fold 與 φ=k×C(N,k)/N 條獨立 backtest paths，
    可計算 Sharpe Ratio 分佈（均值、標準差、信賴區間、PBO）。

    預設 CPCV(6,2)：
    - C(6,2) = 15 個 fold
    - φ = 5 條獨立 backtest paths

    Returns:
        CPCVReport.to_dict() 包含：
        - folds: C(N,k) 個 fold 的 ML 指標 + Backtrader 績效
        - paths: φ 條獨立 backtest paths 的績效
        - summary: Sharpe 分佈統計（mean, std, CI-95, PBO）
    """
    try:
        from validation.cpcv_runner import CPCVRunner
        from config.cpcv import CPCVConfig

        print(
            f"[API] validate_cpcv: symbol={symbol}, period={period}, "
            f"N={n_groups}, k={k_test_groups}, "
            f"label_horizon={label_horizon}, embargo={embargo_bars}"
        )

        # 1. 建立設定
        cfg = CPCVConfig(
            n_groups=n_groups,
            k_test_groups=k_test_groups,
            label_horizon=label_horizon,
            embargo_bars=embargo_bars,
        )
        cfg.validate()

        print(
            f"[API] CPCV config: C({n_groups},{k_test_groups})={cfg.n_combinations} folds, "
            f"φ={cfg.n_backtest_paths} paths"
        )

        # 2. 下載資料
        print(f"[API] Downloading {symbol} data (period={period})...")
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=period)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol '{symbol}'"
            )

        min_bars = n_groups * 50 + label_horizon + embargo_bars
        if len(df) < min_bars:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Insufficient data: {len(df)} bars for {symbol} (period={period}). "
                    f"Need at least {min_bars} bars for N={n_groups} groups."
                )
            )

        print(
            f"[API] Data fetched: {len(df)} bars  "
            f"({df.index[0].date()} → {df.index[-1].date()})"
        )

        # 3. 執行 CPCV
        runner = CPCVRunner(cfg)
        report = runner.run(df, symbol=symbol, verbose=True, run_backtrader=True)

        # 4. 序列化並回傳
        result = report.to_dict()
        result = sanitize_numpy(result)

        print(
            f"[API] validate_cpcv done: {report.n_active_folds}/{report.n_total_folds} folds active, "
            f"{report.n_backtest_paths} paths, "
            f"sharpe_mean={report.summary_sharpe_mean:.4f}"
        )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Configuration error: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"CPCV validation failed: {str(e)}"
        )


# ============================================================================
# Testing Block (Random Forest Predictor)
# ============================================================================

def test_rf_predictor():
    """測試 Random Forest 預測器"""
    print("\n" + "=" * 60)
    print("Testing Random Forest Predictor")
    print("=" * 60)
    
    try:
        # 1. 下載測試數據
        print("\n[TEST] Downloading test data (AAPL, 1 year)...")
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1y")
        
        if data.empty:
            print("[TEST] Failed: No data downloaded")
            return
        
        print(f"[TEST] Downloaded {len(data)} days of data")
        
        # 2. 初始化 Random Forest 預測器
        print("\n[TEST] Initializing RandomForestPredictor...")
        rf_predictor = RandomForestPredictor(forward_days=5, confidence_threshold=0.55)
        
        # 3. 訓練模型
        print("\n[TEST] Training model...")
        rf_predictor.train(data)
        
        if not rf_predictor.is_trained:
            print("[TEST] Failed: Model not trained")
            return
        
        # 4. 進行預測
        print("\n[TEST] Making prediction...")
        prediction = rf_predictor.predict("AAPL", data)
        
        print(f"\n[TEST] Prediction Results:")
        print(f"  Symbol: {prediction['symbol']}")
        print(f"  Signal: {prediction['signal']}")
        print(f"  Confidence: {prediction['confidence']:.2f}%")
        print(f"  Reason: {prediction['reason']}")
        print(f"  Probabilities: Up={prediction['probabilities']['up']:.3f}, Down={prediction['probabilities']['down']:.3f}")
        
        # 5. 查看特徵重要性
        print("\n[TEST] Feature Importances:")
        importances = rf_predictor.feature_importance()
        for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
        
        print("\n[TEST] [OK] All tests passed!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n[TEST] [FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60 + "\n")


# ============================================================================
# Phase 6 — API 整合：新 Endpoints
# ============================================================================

# ── Pydantic models for Phase 6 ────────────────────────────────────────────

class BacktestRunRequest(BaseModel):
    """Phase 6 回測請求（簡化版，對應新 /api/backtest/run）"""
    symbol: str = Field(..., description="股票代碼（例如：2330.TW, AAPL）")
    strategy: str = Field(..., description="策略名稱：ma_crossover | rf | hmm_filter")
    start_date: str = Field(..., description="回測起始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="回測結束日期 (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000, ge=1000, description="初始資金")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="策略參數")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "symbol": "AAPL",
                "strategy": "ma_crossover",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "initial_capital": 100000,
                "params": {"fast_period": 10, "slow_period": 30},
            }]
        }
    }


class HMMValidateRequest(BaseModel):
    """HMM + CPCV 驗證請求"""
    symbol: str = Field(..., description="股票代碼")
    period: str = Field(default="3y", description="數據期間（建議 2y 以上）")
    hmm_n_states: int = Field(default=3, ge=2, le=5, description="HMM 隱藏狀態數")
    hmm_window: int = Field(default=252, ge=60, description="HMM 訓練窗口（天）")
    n_groups: int = Field(default=6, ge=3, le=10, description="CPCV 組數 N")
    k_test_groups: int = Field(default=2, ge=1, le=4, description="CPCV 測試組數 k")


# ── /api/strategies ────────────────────────────────────────────────────────

@app.get("/api/strategies")
async def get_strategies():
    """
    Phase 6: 列出可用回測策略

    Returns 三個核心策略（MA Crossover / RF / HMM Filter）及其參數規格，
    供前端選單使用。
    """
    strategies = [
        {
            "id": "ma_crossover",
            "name": "MA Crossover",
            "description": "均線交叉策略：快線上穿慢線時買入，下穿時賣出",
            "params": {
                "fast_period": {"type": "int", "default": 10, "min": 5, "max": 50,
                                "label": "快速均線週期"},
                "slow_period": {"type": "int", "default": 30, "min": 20, "max": 200,
                                "label": "慢速均線週期"},
            },
        },
        {
            "id": "rf",
            "name": "Random Forest ML",
            "description": "機器學習策略：Random Forest 預測未來 N 天報酬，結合技術指標特徵",
            "params": {
                "forward_days": {"type": "int", "default": 5, "min": 1, "max": 20,
                                 "label": "預測天數"},
                "confidence_threshold": {"type": "float", "default": 0.50, "min": 0.40,
                                         "max": 0.80, "label": "信心閾值"},
                "retrain_period": {"type": "int", "default": 60, "min": 30, "max": 120,
                                   "label": "重訓週期（天）"},
            },
        },
        {
            "id": "hmm_filter",
            "name": "HMM Filter Strategy",
            "description": "HMM 市場狀態過濾策略：Bull 狀態允許 RF 買入，Bear 強制平倉",
            "available": _BACKTEST_MODULE_AVAILABLE,
            "params": {
                "forward_days": {"type": "int", "default": 5, "min": 1, "max": 20,
                                 "label": "RF 預測天數"},
                "confidence_threshold": {"type": "float", "default": 0.50, "min": 0.40,
                                         "max": 0.80, "label": "RF 信心閾值"},
                "hmm_n_states": {"type": "int", "default": 3, "min": 2, "max": 5,
                                 "label": "HMM 狀態數"},
                "hmm_window": {"type": "int", "default": 252, "min": 60, "max": 504,
                               "label": "HMM 訓練窗口（天）"},
            },
        },
    ]
    return {"count": len(strategies), "strategies": strategies}


# ── /api/backtest/run ──────────────────────────────────────────────────────

# Phase 6 strategy mapping (using backtest/ module classes)
_PHASE6_STRATEGY_MAP: Dict[str, Any] = {}
_PHASE6_STRATEGY_NAMES: Dict[str, str] = {
    "ma_crossover": "MA Crossover",
    "rf": "Random Forest ML",
    "hmm_filter": "HMM Filter Strategy",
}

if _BACKTEST_MODULE_AVAILABLE:
    _PHASE6_STRATEGY_MAP = {
        "ma_crossover": _BacktestMACrossover,
        "rf": _BacktestRFStrategy,
        "hmm_filter": _HMMFilterStrategy,
    }


@app.post("/api/backtest/run")
async def backtest_run(request: BacktestRunRequest):
    """
    Phase 6: 執行策略回測（使用 backtest/ 模組的 BacktraderEngine）

    支援策略：ma_crossover | rf | hmm_filter

    Args:
        symbol:          股票代碼
        strategy:        策略 ID
        start_date:      起始日期 (YYYY-MM-DD)
        end_date:        結束日期 (YYYY-MM-DD)
        initial_capital: 初始資金
        params:          策略參數（可選）

    Returns:
        {
          symbol, strategy_name, period,
          performance: {total_return_pct, max_drawdown_pct, sharpe_ratio, total_trades, ...},
          equity_curve: [...],
          trades: [...],
        }
    """
    if not _BACKTEST_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Backtest module unavailable. Check server logs.",
        )

    # 驗證策略
    if request.strategy not in _PHASE6_STRATEGY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{request.strategy}'. Available: {list(_PHASE6_STRATEGY_MAP.keys())}",
        )

    # 驗證日期
    try:
        start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="start_date must be before end_date")

    days_diff = (end_dt - start_dt).days
    if days_diff < 60:
        raise HTTPException(
            status_code=400,
            detail=f"Date range too short ({days_diff} days). Need ≥ 60 days.",
        )

    try:
        print(f"[Phase6/run] {request.strategy} on {request.symbol} "
              f"{request.start_date} ~ {request.end_date}")

        # ── Phase 7 Step 5: Model Cache check ─────────────────────────────
        # For ML strategies (rf, hmm_filter) we cache the full backtest result
        # (output dict) for the day. The cache key encodes symbol + strategy +
        # date range so different date ranges are never conflated.
        _CACHEABLE_STRATEGIES = {"rf", "hmm_filter"}
        _cache_hit = False
        _cache_model_type = None

        if _MODEL_CACHE_AVAILABLE and request.strategy in _CACHEABLE_STRATEGIES:
            # Build a model_type that is unique per (strategy, start_date, end_date)
            # The date range is encoded in the model_type string so the generic
            # {symbol}_{model_type}_{today}.pkl scheme works correctly.
            _safe_start = request.start_date.replace("-", "")
            _safe_end = request.end_date.replace("-", "")
            _cache_model_type = f"{request.strategy}_{_safe_start}_{_safe_end}"

            # Try to load cached backtest result
            cached_result = _model_cache.load_model(
                symbol=request.symbol,
                model_type=_cache_model_type,
                max_age_days=1,
            )

            if cached_result is not None:
                print(f"[Phase6/run] Cache HIT for {request.symbol}/{request.strategy} "
                      f"({request.start_date}~{request.end_date}) — skipping training")
                _cache_hit = True
                cached_result["cache_hit"] = True
                return cached_result

        # ── Cache miss: run full backtest ──────────────────────────────────

        # 下載數據
        ticker_obj = yf.Ticker(request.symbol)
        raw_df = ticker_obj.history(start=request.start_date, end=request.end_date)

        if raw_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for symbol '{request.symbol}'",
            )

        if len(raw_df) < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: only {len(raw_df)} bars (need ≥ 60)",
            )

        # BacktraderEngine 需要小寫欄位
        df = raw_df.copy()
        df.columns = [c.lower() for c in df.columns]

        # 執行回測
        engine = _BacktraderEngine(
            symbol=request.symbol,
            initial_capital=request.initial_capital,
        )

        strategy_class = _PHASE6_STRATEGY_MAP[request.strategy]
        strategy_params = request.params or {}

        result = engine.run(
            strategy_class=strategy_class,
            data=df,
            strategy_params=strategy_params,
            strategy_name=_PHASE6_STRATEGY_NAMES[request.strategy],
        )

        output = result.to_dict()
        output = sanitize_numpy(output)

        print(f"[Phase6/run] Done: return={output['performance']['total_return_pct']:.2f}% "
              f"trades={output['performance']['total_trades']}")

        # ── Phase 7 Step 5: Save to cache ─────────────────────────────────
        if _MODEL_CACHE_AVAILABLE and _cache_model_type is not None:
            try:
                _model_cache.save_model(
                    symbol=request.symbol,
                    model_type=_cache_model_type,
                    model_obj=output,
                    metadata={
                        "strategy": request.strategy,
                        "start_date": request.start_date,
                        "end_date": request.end_date,
                        "initial_capital": request.initial_capital,
                        "params": request.params or {},
                        "total_return_pct": output.get("performance", {}).get("total_return_pct"),
                        "total_trades": output.get("performance", {}).get("total_trades"),
                    },
                )
                output["cache_hit"] = False
                print(f"[Phase6/run] Backtest result cached for {request.symbol}/{request.strategy}")
            except Exception as _save_err:
                print(f"[Phase6/run] Cache save failed (non-fatal): {_save_err}")

        return output

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")


# ── /api/validate/hmm ─────────────────────────────────────────────────────

@app.post("/api/validate/hmm")
async def validate_hmm(request: HMMValidateRequest):
    """
    Phase 6: HMM Filter Strategy 驗證（HMM 狀態分析 + 回測績效）

    步驟：
    1. 下載股票數據
    2. 訓練 MarketHMM，分析市場狀態分佈
    3. 用 HMMFilterStrategy 執行全段回測
    4. 回傳：狀態分佈 + 回測績效摘要

    Returns:
        {
          symbol, period,
          hmm: {n_states, regime_distribution, transitions},
          backtest: {total_return_pct, max_drawdown_pct, sharpe_ratio, total_trades},
        }
    """
    if not _BACKTEST_MODULE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backtest module unavailable.")

    try:
        from hmm.market_hmm import MarketHMM

        print(f"[Phase6/validate_hmm] symbol={request.symbol}, period={request.period}, "
              f"n_states={request.hmm_n_states}")

        # 1. 下載數據
        ticker_obj = yf.Ticker(request.symbol)
        raw_df = ticker_obj.history(period=request.period)

        if raw_df.empty:
            raise HTTPException(status_code=404, detail=f"No data for '{request.symbol}'")

        min_bars = request.hmm_window + 60
        if len(raw_df) < min_bars:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(raw_df)} bars (need ≥ {min_bars})",
            )

        print(f"[Phase6/validate_hmm] Fetched {len(raw_df)} bars "
              f"({raw_df.index[0].date()} → {raw_df.index[-1].date()})")

        # 2. 訓練 MarketHMM → 取得狀態序列
        hmm_model = MarketHMM(n_states=request.hmm_n_states)
        hmm_model.fit(raw_df)
        regimes = hmm_model.predict(raw_df)

        regime_labels = {0: "Bull", 1: "Sideways", 2: "Bear"}
        total_bars = len(regimes)
        regime_counts: Dict[str, int] = {}
        for r in regimes:
            label = regime_labels.get(int(r), f"State{r}")
            regime_counts[label] = regime_counts.get(label, 0) + 1

        regime_distribution = {
            label: {
                "count": count,
                "pct": round(count / total_bars * 100, 2),
            }
            for label, count in regime_counts.items()
        }

        # 3. 執行 HMMFilterStrategy 回測（使用全段數據）
        df = raw_df.copy()
        df.columns = [c.lower() for c in df.columns]

        engine = _BacktraderEngine(symbol=request.symbol, initial_capital=100_000)
        result = engine.run(
            strategy_class=_HMMFilterStrategy,
            data=df,
            strategy_params={
                "hmm_n_states": request.hmm_n_states,
                "hmm_window": request.hmm_window,
            },
            strategy_name="HMM Filter Strategy",
        )

        perf = result.to_dict()["performance"]
        perf = sanitize_numpy(perf)

        output = {
            "symbol": request.symbol,
            "period": request.period,
            "data_bars": total_bars,
            "hmm": {
                "n_states": request.hmm_n_states,
                "window": request.hmm_window,
                "regime_distribution": regime_distribution,
            },
            "backtest": {
                "total_return_pct": perf.get("total_return_pct"),
                "max_drawdown_pct": perf.get("max_drawdown_pct"),
                "sharpe_ratio": perf.get("sharpe_ratio"),
                "total_trades": perf.get("total_trades"),
                "win_rate_pct": perf.get("win_rate_pct"),
            },
        }

        print(f"[Phase6/validate_hmm] Done: return={perf.get('total_return_pct', 0):.2f}%")
        return output

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"HMM validation failed: {e}")


# ── Backtest HTML page ─────────────────────────────────────────────────────

@app.get("/backtest")
async def serve_backtest_page():
    """Serve the strategy backtest UI page."""
    backtest_html = os.path.join(os.path.dirname(__file__), "static", "backtest.html")
    if os.path.exists(backtest_html):
        return FileResponse(backtest_html)
    raise HTTPException(status_code=404, detail="backtest.html not found")


# ============================================================================
# Phase 7: Portfolio Analysis
# ============================================================================

# Portfolio module import with graceful fallback
try:
    from portfolio.portfolio_analyzer import PortfolioAnalyzer as _PortfolioAnalyzer
    _PORTFOLIO_MODULE_AVAILABLE = True
except ImportError as _pe:
    print(f"[WARNING] portfolio module import failed: {_pe}")
    _PORTFOLIO_MODULE_AVAILABLE = False

# PortfolioManager import (Phase 7 Step 6)
try:
    from portfolio.portfolio_manager import PortfolioManager as _PortfolioManager
    _PORTFOLIO_MANAGER_AVAILABLE = True
except ImportError as _pme:
    print(f"[WARNING] portfolio_manager import failed: {_pme}")
    _PORTFOLIO_MANAGER_AVAILABLE = False

# Default portfolio for /api/portfolio/summary endpoint
_DEFAULT_PORTFOLIO = [
    {"symbol": "2330.TW", "weight": 0.50, "strategy_type": "hmm_rf"},
    {"symbol": "0050.TW", "weight": 0.30, "strategy_type": "rf"},
    {"symbol": "2317.TW", "weight": 0.20, "strategy_type": "rf"},
]


class PortfolioAnalyzeRequest(BaseModel):
    """Request body for POST /api/portfolio/analyze"""
    symbols: List[str] = Field(
        ...,
        min_length=2,
        description="List of ticker symbols (2–10)",
        example=["AAPL", "MSFT", "GOOG"],
    )
    period: str = Field(
        default="1y",
        description="yfinance period string (1mo, 3mo, 6mo, 1y, 2y, 5y)",
        example="1y",
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Portfolio weights (must match symbol count; normalised to sum=1). "
                    "Omit for equal weighting.",
        example=None,
    )


@app.post("/api/portfolio/analyze")
async def portfolio_analyze(req: PortfolioAnalyzeRequest):
    """
    **Phase 7** — Analyse a portfolio of multiple stocks simultaneously.

    Returns:
    - Per-asset metrics: total return, annualised return/volatility, Sharpe, Sortino,
      max drawdown, Calmar ratio, win rate, beta vs portfolio
    - Pearson correlation matrix
    - Combined portfolio metrics (weighted)
    - Normalised price curves (base=100) for charting
    """
    if not _PORTFOLIO_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Portfolio module not available. Check server logs for import errors.",
        )
    if len(req.symbols) < 2:
        raise HTTPException(status_code=422, detail="At least 2 symbols required.")
    if len(req.symbols) > 10:
        raise HTTPException(status_code=422, detail="Maximum 10 symbols allowed.")

    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
    if req.period not in valid_periods:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid period '{req.period}'. Valid: {sorted(valid_periods)}",
        )

    try:
        analyzer = _PortfolioAnalyzer()
        result = analyzer.analyze(
            symbols=req.symbols,
            period=req.period,
            weights=req.weights,
        )
        return sanitize_numpy(result.to_dict())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {e}")


@app.get("/api/portfolio/summary")
async def portfolio_summary(
    include_correlation: bool = Query(
        default=False,
        description="Set true to include correlation matrix (slower, requires network)",
    ),
):
    """
    **Phase 7 Step 6** — Get real-time portfolio summary for the default portfolio.

    Default portfolio:
      - 2330.TW (TSMC)       50% — HMM-Filtered RF strategy
      - 0050.TW (Taiwan ETF) 30% — RF strategy
      - 2317.TW (Foxconn)    20% — RF strategy

    Returns:
      - Per-symbol regime status (Bull / Sideways / Bear via MarketHMM)
      - Portfolio-level KPI estimates (buy-and-hold, 6 months)
      - Optionally: correlation matrix (when include_correlation=true)

    Note: Regime detection requires network access to fetch recent price data.
          If unavailable, regime defaults to "Unknown".
    """
    if not _PORTFOLIO_MANAGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PortfolioManager module not available. Check server logs.",
        )

    try:
        pm = _PortfolioManager(initial_capital=1_000_000)
        for entry in _DEFAULT_PORTFOLIO:
            pm.add_position(
                symbol=entry["symbol"],
                weight=entry["weight"],
                strategy_type=entry["strategy_type"],
            )

        summary = pm.get_portfolio_summary()
        summary = sanitize_numpy(summary)

        # Optionally include correlation matrix
        if include_correlation:
            corr = pm.calculate_correlation_matrix(lookback_days=252)
            summary["correlation_matrix"] = sanitize_numpy(corr)

        return summary

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Portfolio summary failed: {e}")


@app.get("/api/portfolio/info")
async def portfolio_info():
    """Return portfolio module availability and supported periods."""
    return {
        "available": _PORTFOLIO_MODULE_AVAILABLE,
        "max_symbols": 10,
        "min_symbols": 2,
        "supported_periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        "metrics": {
            "asset_level": [
                "total_return_pct", "annualized_return_pct", "annualized_volatility_pct",
                "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio",
                "win_rate_pct", "beta_vs_portfolio",
            ],
            "portfolio_level": [
                "total_return_pct", "annualized_return_pct", "annualized_volatility_pct",
                "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "calmar_ratio",
                "diversification_ratio", "portfolio_volatility_cov_pct",
            ],
        },
    }


# ============================================================================
# Phase 7 Step 4: /api/monitor/check — Regime Monitor Endpoint
# ============================================================================

_MONITOR_SYMBOLS = ["2330.TW", "2317.TW", "0050.TW"]


@app.get("/api/monitor/check")
async def monitor_check():
    """
    Phase 7 Step 4: 定期觸發 regime monitor 檢查。

    對監控清單 ["2330.TW", "2317.TW", "0050.TW"] 執行：
      1. 下載近期股票數據（6 個月）
      2. 訓練 MarketHMM 並預測當前 regime
      3. 呼叫 regime_monitor.check_regime_change 偵測切換並發送 alert

    若 HMM 未初始化（無數據或數據不足），graceful skip。

    Returns:
        {
            "checked":       ["2330.TW", ...],
            "skipped":       ["2317.TW"],
            "alerts_sent":   0,
            "regime_status": {
                "2330.TW": {"regime": "Bull", "since": "2026-02-18", ...},
                ...
            }
        }
    """
    try:
        from hmm.market_hmm import MarketHMM
        from alerts.regime_monitor import check_regime_change, get_all_regime_status
    except ImportError as ie:
        raise HTTPException(status_code=503, detail=f"Required module unavailable: {ie}")

    checked: list = []
    skipped: list = []
    alerts_sent: int = 0

    for symbol in _MONITOR_SYMBOLS:
        try:
            ticker_obj = yf.Ticker(symbol)
            raw_df = ticker_obj.history(period="6mo")

            if raw_df.empty or len(raw_df) < 60:
                print(f"[monitor/check] {symbol}: insufficient data ({len(raw_df)} bars), skipping.")
                skipped.append(symbol)
                continue

            # 訓練 HMM
            hmm_model = MarketHMM(n_states=3)
            hmm_model.fit(raw_df)

            # 預測最新一根的 regime
            regimes = hmm_model.predict(raw_df)
            latest_regime_idx = int(regimes[-1])
            regime_label_map = {0: "Bull", 1: "Sideways", 2: "Bear"}
            current_regime = regime_label_map.get(latest_regime_idx, f"State{latest_regime_idx}")

            # 取最新收盤價
            close_col = "Close" if "Close" in raw_df.columns else "close"
            latest_price = float(raw_df[close_col].iloc[-1])

            # 讀取切換前的 regime（用於 alert log）
            from alerts.regime_monitor import _load_state as _rm_load_state
            from pathlib import Path as _Path
            _state_p = _Path(__file__).resolve().parent / "alerts" / "state.json"
            _state_before = _rm_load_state(_state_p)
            _old_regime = _state_before.get(symbol, {}).get("regime", None)

            # HMM 信心度
            _confidence = 0.0
            try:
                _proba_df = hmm_model.predict_proba(raw_df)
                if current_regime in _proba_df.columns:
                    _confidence = float(_proba_df[current_regime].iloc[-1])
            except Exception:
                pass

            # Regime 切換偵測
            changed = check_regime_change(
                symbol=symbol,
                current_regime=current_regime,
                price=latest_price,
            )
            if changed:
                alerts_sent += 1
                # Phase 7 Step 6: 寫入 alert log
                if _SCHEDULER_AVAILABLE and _old_regime is not None:
                    from datetime import datetime, timezone
                    _append_alert_log({
                        "timestamp":  datetime.now(timezone.utc).isoformat(),
                        "symbol":     symbol,
                        "old_regime": _old_regime,
                        "new_regime": current_regime,
                        "confidence": round(_confidence, 4),
                    })

            checked.append(symbol)
            print(f"[monitor/check] {symbol}: regime={current_regime}, price={latest_price:.2f}, changed={changed}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[monitor/check] {symbol} error: {exc}")
            skipped.append(symbol)

    regime_status = get_all_regime_status()

    return {
        "checked":       checked,
        "skipped":       skipped,
        "alerts_sent":   alerts_sent,
        "regime_status": regime_status,
    }


# ── Phase 7 Step 6: Alert Log Endpoint ────────────────────────────────────

@app.get("/api/monitor/alert-log")
async def get_alert_log(limit: int = Query(default=50, ge=1, le=200)):
    """
    Phase 7 Step 6: 回傳 alert-log.json 最近 N 條記錄。

    Query Parameters:
        limit: 最多回傳幾條（預設 50，上限 200）

    Returns:
        {
            "alerts": [...],
            "total":  int,
        }

    每條記錄格式：
        {
            "timestamp":  "2026-02-18T07:30:00+00:00",
            "symbol":     "2330.TW",
            "old_regime": "Bull",
            "new_regime": "Bear",
            "confidence": 0.82,
        }
    """
    if not _SCHEDULER_AVAILABLE:
        return {"alerts": [], "total": 0, "note": "Scheduler not available"}

    alerts = _read_alert_log(limit=limit)
    return {"alerts": alerts, "total": len(alerts)}


# ── Phase 7 Step 6: Scheduler Status Endpoint ─────────────────────────────

@app.get("/api/monitor/scheduler-status")
async def get_scheduler_status():
    """
    Phase 7 Step 6: 回傳排程器狀態。

    Returns:
        {
            "running":     bool,
            "last_run":    str or null,
            "next_run":    str,
            "last_result": dict or null,
        }
    """
    if not _SCHEDULER_AVAILABLE or _scheduler is None:
        return {"running": False, "last_run": None, "next_run": None, "last_result": None}
    return _scheduler.get_status()


# ── Phase 9: Signal Check Endpoint ────────────────────────────────────────

@app.get("/api/monitor/signal-check")
async def signal_check(
    symbols: Optional[str] = Query(None, description="逗號分隔股票代碼，預設 2330.TW,0050.TW,2317.TW"),
    confidence_threshold: float = Query(60.0, description="信心度門檻（%），預設 60.0"),
    send_discord: bool = Query(False, description="是否發送 Discord 通知（需 DISCORD_WEBHOOK_URL）"),
):
    """
    Phase 9: 對指定股票執行 ML 預測，可選擇是否推播 Discord 訊號通知。

    - 使用 Random Forest 預測 5 日信號（BUY / SELL / HOLD）
    - 信心度 >= confidence_threshold 且 send_discord=true 時發送 Discord alert
    - 不需 DISCORD_WEBHOOK_URL 也可使用（send_discord=false 純查詢模式）

    Returns:
        {
            "timestamp":    str,
            "checked":      [symbol, ...],
            "skipped":      [symbol, ...],
            "alerts_sent":  int,
            "results":      [{symbol, signal, confidence, price, regime, alerted, error}, ...]
        }
    """
    if not _SIGNAL_ALERT_AVAILABLE or _check_signals is None:
        raise HTTPException(status_code=503, detail="SignalAlert module not available")

    sym_list = None
    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    try:
        result = _check_signals(
            symbols=sym_list,
            confidence_threshold=confidence_threshold,
            webhook_url=None if send_discord else "",  # "" → skip discord in send_alert
            alert_on_signals=["BUY", "SELL"] if send_discord else [],
        )
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Portfolio HTML page ────────────────────────────────────────────────────

@app.get("/portfolio")
async def serve_portfolio_page():
    """Serve the portfolio analysis UI page."""
    portfolio_html = os.path.join(os.path.dirname(__file__), "static", "portfolio.html")
    if os.path.exists(portfolio_html):
        return FileResponse(portfolio_html)
    raise HTTPException(status_code=404, detail="portfolio.html not found")


# ============================================================================
# Phase 8D: Ensemble Strategy Comparison Panel
# ============================================================================

def _make_data_fetcher(df: pd.DataFrame):
    """Return a zero-arg callable that always returns a copy of df.
    Used to monkey-patch BacktestEngine._fetch_data so all strategies
    share the same downloaded DataFrame without extra yfinance calls.
    """
    def _fetch():
        return df.copy()
    return _fetch


@app.get("/api/backtest/compare")
async def compare_strategies(
    symbol: str = Query("2330.TW", description="Stock symbol"),
    start: str = Query("2024-01-01", description="Start date YYYY-MM-DD"),
    end: str = Query("2024-12-31", description="End date YYYY-MM-DD"),
    interval: str = Query("1d", description="Data interval (informational; yfinance always uses daily)"),
):
    """
    Phase 8D: Ensemble Strategy Comparison

    Runs MA Crossover, RF Strategy, and HMM Filter Strategy in parallel,
    plus a Buy & Hold benchmark.  Returns equity curves and KPIs for all
    four, with a recommendation based on the highest Sharpe ratio.

    Parameters
    ----------
    symbol   : Stock ticker (e.g. ``2330.TW``, ``AAPL``)
    start    : ISO date string for backtest start
    end      : ISO date string for backtest end
    interval : Data interval hint (default ``1d``; engine always uses daily)

    Returns
    -------
    JSON with keys: symbol, period, strategies, benchmark, recommended,
    recommendation_reason
    """
    try:
        # ── Validate dates ────────────────────────────────────────────────
        try:
            dt_start = datetime.strptime(start, "%Y-%m-%d")
            dt_end   = datetime.strptime(end,   "%Y-%m-%d")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date format: {exc}")

        if dt_start >= dt_end:
            raise HTTPException(status_code=400, detail="start must be before end")

        days_diff = (dt_end - dt_start).days
        if days_diff < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Date range too short ({days_diff} days). Need at least 60 days.",
            )

        # ── Fetch market data once ────────────────────────────────────────
        print(f"[COMPARE] Fetching {symbol} {start}→{end}")
        ticker = yf.Ticker(symbol)
        raw_df = ticker.history(start=start, end=end)

        if raw_df is None or raw_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")

        # Normalise column names to lower-case (BacktestEngine expects that)
        raw_df.columns = [c.lower() for c in raw_df.columns]

        if len(raw_df) < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: only {len(raw_df)} rows (need ≥ 60).",
            )

        initial_capital: float = 100_000.0
        years = max(days_diff / 365.25, 0.01)

        # ── Helper: run one strategy on cached data ────────────────────────
        def _run_strategy(strat_key: str, strat_display_name: str) -> Dict[str, Any]:
            """Execute a single backtest and return normalised result dict."""
            if strat_key not in BacktestEngine.STRATEGY_MAP:
                return {
                    "name": strat_display_name,
                    "equity": [],
                    "kpi": {
                        "sharpe": 0.0, "max_drawdown": 0.0,
                        "win_rate": 0.0, "cagr": 0.0, "total_return": 0.0,
                    },
                    "error": f"Strategy '{strat_key}' not registered",
                }

            engine = BacktestEngine(
                symbol=symbol,
                start_date=start,
                end_date=end,
                initial_capital=initial_capital,
            )
            # Inject cached data — bypasses internal yfinance call
            engine._fetch_data = _make_data_fetcher(raw_df)

            try:
                result = engine.run_backtest(strat_key, {})
                perf   = result["data"]["performance"]
                eq_raw = result["data"].get("equity_curve", [])

                # Equity curve: keep only {date, value}
                equity = [
                    {"date": e["date"], "value": e["value"]}
                    for e in eq_raw
                    if not np.isnan(e.get("value", float("nan")))
                ]

                total_ret_frac = perf.get("total_return", 0.0) / 100.0
                cagr = (1.0 + total_ret_frac) ** (1.0 / years) - 1.0

                return {
                    "name": strat_display_name,
                    "equity": equity,
                    "kpi": {
                        "sharpe":       round(float(perf.get("sharpe_ratio",  0.0)), 4),
                        "max_drawdown": round(-abs(float(perf.get("max_drawdown", 0.0))) / 100.0, 4),
                        "win_rate":     round(float(perf.get("win_rate",      0.0)) / 100.0, 4),
                        "cagr":         round(float(cagr), 4),
                        "total_return": round(float(total_ret_frac), 4),
                    },
                }
            except Exception as exc:
                print(f"[COMPARE] Strategy '{strat_key}' error: {exc}")
                return {
                    "name": strat_display_name,
                    "equity": [],
                    "kpi": {
                        "sharpe": 0.0, "max_drawdown": 0.0,
                        "win_rate": 0.0, "cagr": 0.0, "total_return": 0.0,
                    },
                    "error": str(exc),
                }

        # ── Run 3 strategies ──────────────────────────────────────────────
        strategy_configs = [
            ("ma_crossover", "MA Crossover"),
            ("rf",           "RF Strategy"),
            ("hmm_filter",   "HMM Filter Strategy"),
        ]

        strategies_output: List[Dict[str, Any]] = []
        for strat_key, strat_name in strategy_configs:
            print(f"[COMPARE] Running {strat_name} …")
            strategies_output.append(_run_strategy(strat_key, strat_name))

        # ── Buy & Hold benchmark ──────────────────────────────────────────
        print("[COMPARE] Running Buy & Hold benchmark …")
        try:
            bh_engine = BacktestEngine(
                symbol=symbol,
                start_date=start,
                end_date=end,
                initial_capital=initial_capital,
            )
            bh_engine._fetch_data = _make_data_fetcher(raw_df)
            bh_raw = bh_engine._run_benchmark(raw_df.copy())

            bh_equity_vals = bh_raw.get("equity_values", [])
            # Reconstruct date sequence from raw_df
            bh_dates = [
                d.date().isoformat() if hasattr(d, "date") else str(d)
                for d in raw_df.index
            ]

            # Align lengths
            min_len = min(len(bh_dates), len(bh_equity_vals))
            bh_equity = [
                {"date": bh_dates[i], "value": round(float(bh_equity_vals[i]), 2)}
                for i in range(min_len)
                if not np.isnan(bh_equity_vals[i])
            ]

            bh_total_ret_frac = bh_raw.get("total_return", 0.0) / 100.0
            bh_cagr = (1.0 + bh_total_ret_frac) ** (1.0 / years) - 1.0

            benchmark_output: Dict[str, Any] = {
                "name": "Buy & Hold",
                "equity": bh_equity,
                "kpi": {
                    "sharpe":       round(float(bh_raw.get("sharpe_ratio",  0.0)), 4),
                    "max_drawdown": round(-abs(float(bh_raw.get("max_drawdown", 0.0))) / 100.0, 4),
                    "win_rate":     0.0,
                    "cagr":         round(float(bh_cagr), 4),
                    "total_return": round(float(bh_total_ret_frac), 4),
                },
            }
        except Exception as exc:
            print(f"[COMPARE] Benchmark error: {exc}")
            benchmark_output = {
                "name": "Buy & Hold",
                "equity": [],
                "kpi": {
                    "sharpe": 0.0, "max_drawdown": 0.0,
                    "win_rate": 0.0, "cagr": 0.0, "total_return": 0.0,
                },
                "error": str(exc),
            }

        # ── Determine recommended strategy (highest Sharpe) ────────────────
        valid_strategies = [
            s for s in strategies_output
            if s.get("equity") and not s.get("error")
        ]

        if valid_strategies:
            best = max(valid_strategies, key=lambda s: s["kpi"]["sharpe"])
            recommended      = best["name"]
            best_sharpe      = best["kpi"]["sharpe"]
            recommendation_reason = f"最高 Sharpe ratio ({best_sharpe:.4f})"
        else:
            recommended           = None
            recommendation_reason = "無法完成任何策略回測"

        return sanitize_numpy({
            "symbol":   symbol,
            "period":   {"start": start, "end": end},
            "strategies": strategies_output,
            "benchmark":  benchmark_output,
            "recommended": recommended,
            "recommendation_reason": recommendation_reason,
        })

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[COMPARE] Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Compare failed: {exc}")


# ── Compare HTML page ──────────────────────────────────────────────────────

@app.get("/compare")
async def serve_compare_page():
    """Serve the Phase 8D ensemble strategy comparison UI."""
    compare_html = os.path.join(os.path.dirname(__file__), "static", "compare.html")
    if os.path.exists(compare_html):
        return FileResponse(compare_html)
    raise HTTPException(status_code=404, detail="compare.html not found")


# ============================================================================
# Phase 8B: WebSocket Real-time Price Dashboard
# ============================================================================

MAX_WS_CONNECTIONS = 10


class WebSocketConnectionManager:
    """Manages active WebSocket connections with a max-connection cap."""

    def __init__(self, max_connections: int = MAX_WS_CONNECTIONS):
        self.max_connections = max_connections
        # symbol -> set of WebSocket objects
        self.active: Dict[str, set] = {}

    def count(self) -> int:
        return sum(len(ws_set) for ws_set in self.active.values())

    async def connect(self, websocket: WebSocket, symbol: str) -> bool:
        """Accept connection; return False if cap is reached."""
        if self.count() >= self.max_connections:
            await websocket.close(code=1008, reason="Max connections reached")
            return False
        await websocket.accept()
        self.active.setdefault(symbol.upper(), set()).add(websocket)
        print(f"[WS] Connected: {symbol}  total={self.count()}")
        return True

    def disconnect(self, websocket: WebSocket, symbol: str):
        sym = symbol.upper()
        if sym in self.active:
            self.active[sym].discard(websocket)
            if not self.active[sym]:
                del self.active[sym]
        print(f"[WS] Disconnected: {symbol}  total={self.count()}")

    async def broadcast(self, symbol: str, message: dict):
        """Send JSON payload to all clients subscribed to symbol."""
        sym = symbol.upper()
        if sym not in self.active:
            return
        dead = set()
        for ws in list(self.active[sym]):
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active[sym].discard(ws)


ws_manager = WebSocketConnectionManager(max_connections=MAX_WS_CONNECTIONS)


def _fetch_realtime_quote(symbol: str) -> Dict[str, Any]:
    """
    Fetch latest real-time quote via yfinance.
    Uses 1-minute interval (last 2 days) to get the most recent bar.
    Falls back to 1-day history if intraday data is unavailable.
    """
    ticker = yf.Ticker(symbol)

    # Try 1-min intraday first
    df_1m = ticker.history(period="1d", interval="1m")
    if not df_1m.empty:
        latest = df_1m.iloc[-1]
        price   = float(latest["Close"])
        volume  = int(latest["Volume"]) if not np.isnan(latest["Volume"]) else 0
        # Day change: compare to previous day close
        df_1d = ticker.history(period="5d", interval="1d")
        if len(df_1d) >= 2:
            prev_close = float(df_1d.iloc[-2]["Close"])
            day_open   = float(df_1d.iloc[-1]["Open"]) if not np.isnan(df_1d.iloc[-1]["Open"]) else prev_close
        elif len(df_1d) == 1:
            prev_close = float(df_1d.iloc[0]["Open"])
            day_open   = prev_close
        else:
            prev_close = price
            day_open   = price
        change_pct = round((price - prev_close) / prev_close * 100, 4) if prev_close else 0.0
        return {
            "symbol":     symbol.upper(),
            "price":      round(price, 4),
            "volume":     volume,
            "change_pct": change_pct,
            "prev_close": round(prev_close, 4),
            "timestamp":  datetime.utcnow().isoformat() + "Z",
            "source":     "1m",
        }

    # Fallback: daily data
    df_1d = ticker.history(period="5d", interval="1d")
    if df_1d.empty:
        raise ValueError(f"No data available for symbol '{symbol}'")

    latest   = df_1d.iloc[-1]
    price    = float(latest["Close"])
    volume   = int(latest["Volume"]) if not np.isnan(latest["Volume"]) else 0
    prev_close = float(df_1d.iloc[-2]["Close"]) if len(df_1d) >= 2 else price
    change_pct = round((price - prev_close) / prev_close * 100, 4) if prev_close else 0.0
    return {
        "symbol":     symbol.upper(),
        "price":      round(price, 4),
        "volume":     volume,
        "change_pct": change_pct,
        "prev_close": round(prev_close, 4),
        "timestamp":  datetime.utcnow().isoformat() + "Z",
        "source":     "1d",
    }


@app.websocket("/ws/price/{symbol}")
async def websocket_price_feed(websocket: WebSocket, symbol: str):
    """
    Phase 8B WebSocket endpoint — real-time price feed.

    Protocol:
    - Client connects to  ws://<host>/ws/price/AAPL
    - Server pushes JSON every 5 seconds:
      {
        "symbol":     "AAPL",
        "price":      182.45,
        "volume":     12345678,
        "change_pct": 0.73,
        "prev_close": 181.13,
        "timestamp":  "2026-02-18T09:45:00Z",
        "source":     "1m"
      }
    - On disconnect the connection is cleaned up automatically.
    - Max 10 simultaneous connections across all symbols.
    """
    connected = await ws_manager.connect(websocket, symbol)
    if not connected:
        return  # already closed inside connect()

    try:
        while True:
            try:
                payload = await asyncio.get_event_loop().run_in_executor(
                    None, _fetch_realtime_quote, symbol
                )
                await websocket.send_json(payload)
            except Exception as fetch_err:
                error_payload = {
                    "symbol":    symbol.upper(),
                    "error":     str(fetch_err),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                try:
                    await websocket.send_json(error_payload)
                except Exception:
                    break  # client gone

            # Wait 5 seconds, but also check for disconnect messages
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                pass  # normal — just keep looping
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, symbol)


@app.websocket("/ws/realtime/{symbol}")
async def websocket_realtime_feed(websocket: WebSocket, symbol: str):
    """
    Phase 8B alias endpoint — mirrors /ws/price/{symbol}.
    Provided for spec compatibility; delegates to the same handler logic.
    """
    await websocket_price_feed(websocket, symbol)


@app.get("/api/ws/status")
async def websocket_status():
    """Return current WebSocket connection counts (diagnostic endpoint)."""
    return {
        "total_connections": ws_manager.count(),
        "max_connections":   MAX_WS_CONNECTIONS,
        "by_symbol": {
            sym: len(ws_set)
            for sym, ws_set in ws_manager.active.items()
        },
    }


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Stock Analysis Backend Server")
    print("=" * 60)
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"API Documentation: http://0.0.0.0:{port}/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=port)

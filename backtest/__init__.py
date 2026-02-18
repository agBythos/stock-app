"""
Stock App Backtest Module
========================
Standalone Backtrader integration with Taiwan stock market support.
"""

from .backtrader_engine import BacktraderEngine, TaiwanCommission, WalkForwardResult
from .rf_strategy import RFStrategy, RandomForestPredictor
from .hmm_filter_strategy import HMMFilterStrategy, REGIME_BULL, REGIME_SIDEWAYS, REGIME_BEAR

__all__ = [
    "BacktraderEngine",
    "TaiwanCommission",
    "WalkForwardResult",
    "RFStrategy",
    "RandomForestPredictor",
    "HMMFilterStrategy",
    "REGIME_BULL",
    "REGIME_SIDEWAYS",
    "REGIME_BEAR",
]

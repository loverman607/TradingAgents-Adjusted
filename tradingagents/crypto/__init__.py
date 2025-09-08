"""
Crypto Trading Module

This module provides cryptocurrency-specific optimizations for the TradingAgents system.
It includes specialized trading engines, indicators, and strategies designed for BTC and ETH.
"""

from .crypto_engine import CryptoTradingEngine
from .crypto_indicators import CryptoIndicators
from .crypto_strategies import CryptoStrategies
from .crypto_data import CryptoDataProvider
from .crypto_risk import CryptoRiskManager

__all__ = [
    'CryptoTradingEngine',
    'CryptoIndicators', 
    'CryptoStrategies',
    'CryptoDataProvider',
    'CryptoRiskManager'
]

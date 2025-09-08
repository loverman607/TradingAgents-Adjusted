"""
Crypto-Specific Trading Strategies

This module provides trading strategies specifically designed for cryptocurrency markets,
including momentum, mean reversion, and trend-following strategies optimized for BTC and ETH.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

from .crypto_indicators import CryptoIndicators

logger = logging.getLogger(__name__)


class CryptoStrategies:
    """Crypto-specific trading strategies"""
    
    def __init__(self):
        self.indicators = CryptoIndicators()
    
    def momentum_strategy(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Crypto momentum strategy optimized for volatile markets
        
        Args:
            data: OHLCV data
            ticker: Crypto ticker
            
        Returns:
            Strategy signal and metadata
        """
        prices = data['close']
        volume = data['volume']
        
        # Get crypto indicators
        signals = self.indicators.get_crypto_signals(prices, volume, data.get('high'), data.get('low'))
        
        # Generate trading signal
        trading_signal = self.indicators.generate_crypto_trading_signal(signals, prices.iloc[-1])
        
        # Momentum-specific logic
        momentum_score = 0
        momentum_reasoning = []
        
        # Price momentum
        price_change_5d = (prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6] if len(prices) >= 6 else 0
        price_change_20d = (prices.iloc[-1] - prices.iloc[-21]) / prices.iloc[-21] if len(prices) >= 21 else 0
        
        if price_change_5d > 0.05:  # 5% in 5 days
            momentum_score += 2
            momentum_reasoning.append(f"Strong 5-day momentum: {price_change_5d:.2%}")
        elif price_change_5d > 0.02:  # 2% in 5 days
            momentum_score += 1
            momentum_reasoning.append(f"Positive 5-day momentum: {price_change_5d:.2%}")
        elif price_change_5d < -0.05:
            momentum_score -= 2
            momentum_reasoning.append(f"Negative 5-day momentum: {price_change_5d:.2%}")
        elif price_change_5d < -0.02:
            momentum_score -= 1
            momentum_reasoning.append(f"Negative 5-day momentum: {price_change_5d:.2%}")
        
        # Volume confirmation
        volume_avg = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
        
        if volume_ratio > 1.5:
            momentum_score += 1
            momentum_reasoning.append(f"High volume confirmation: {volume_ratio:.1f}x")
        elif volume_ratio < 0.5:
            momentum_score -= 1
            momentum_reasoning.append(f"Low volume: {volume_ratio:.1f}x")
        
        # Trend alignment
        if price_change_20d > 0.1:  # 10% in 20 days
            momentum_score += 1
            momentum_reasoning.append(f"Strong trend: {price_change_20d:.2%}")
        elif price_change_20d < -0.1:
            momentum_score -= 1
            momentum_reasoning.append(f"Downtrend: {price_change_20d:.2%}")
        
        # Combine with technical signals
        final_score = trading_signal['score'] + momentum_score
        final_confidence = min(0.95, trading_signal['confidence'] + abs(momentum_score) * 0.1)
        
        # Determine final signal
        if final_score >= 4:
            signal = "STRONG_BUY"
        elif final_score >= 2:
            signal = "BUY"
        elif final_score <= -4:
            signal = "STRONG_SELL"
        elif final_score <= -2:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            "strategy": "momentum",
            "signal": signal,
            "confidence": final_confidence,
            "score": final_score,
            "reasoning": trading_signal['reasoning'] + momentum_reasoning,
            "metadata": {
                "price_change_5d": price_change_5d,
                "price_change_20d": price_change_20d,
                "volume_ratio": volume_ratio,
                "technical_indicators": trading_signal['indicators']
            }
        }
    
    def mean_reversion_strategy(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Crypto mean reversion strategy for ranging markets
        
        Args:
            data: OHLCV data
            ticker: Crypto ticker
            
        Returns:
            Strategy signal and metadata
        """
        prices = data['close']
        volume = data['volume']
        
        # Get crypto indicators
        signals = self.indicators.get_crypto_signals(prices, volume, data.get('high'), data.get('low'))
        
        # Mean reversion specific logic
        reversion_score = 0
        reversion_reasoning = []
        
        # Bollinger Bands mean reversion
        bb_upper = signals['bollinger']['upper'].iloc[-1]
        bb_lower = signals['bollinger']['lower'].iloc[-1]
        bb_middle = signals['bollinger']['middle'].iloc[-1]
        current_price = prices.iloc[-1]
        
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        if bb_position < 0.2:  # Near lower band
            reversion_score += 2
            reversion_reasoning.append(f"Price near lower Bollinger Band ({bb_position:.2f})")
        elif bb_position < 0.3:
            reversion_score += 1
            reversion_reasoning.append(f"Price below middle of Bollinger Bands ({bb_position:.2f})")
        elif bb_position > 0.8:  # Near upper band
            reversion_score -= 2
            reversion_reasoning.append(f"Price near upper Bollinger Band ({bb_position:.2f})")
        elif bb_position > 0.7:
            reversion_score -= 1
            reversion_reasoning.append(f"Price above middle of Bollinger Bands ({bb_position:.2f})")
        
        # RSI mean reversion
        rsi = signals['rsi'].iloc[-1] if not signals['rsi'].empty else 50
        
        if rsi < 30:
            reversion_score += 2
            reversion_reasoning.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            reversion_score += 1
            reversion_reasoning.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            reversion_score -= 2
            reversion_reasoning.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            reversion_score -= 1
            reversion_reasoning.append(f"RSI high ({rsi:.1f})")
        
        # Support/Resistance levels
        levels = signals['levels']
        support_levels = levels['support']
        resistance_levels = levels['resistance']
        
        # Check distance to nearest support/resistance
        if support_levels:
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
            support_distance = abs(current_price - nearest_support) / current_price
            
            if support_distance < 0.02:  # Within 2% of support
                reversion_score += 1
                reversion_reasoning.append(f"Near support level: ${nearest_support:.2f}")
        
        if resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            resistance_distance = abs(current_price - nearest_resistance) / current_price
            
            if resistance_distance < 0.02:  # Within 2% of resistance
                reversion_score -= 1
                reversion_reasoning.append(f"Near resistance level: ${nearest_resistance:.2f}")
        
        # Market regime check (mean reversion works best in ranging markets)
        market_regime = signals['market_regime'].iloc[-1] if not signals['market_regime'].empty else 0
        
        if market_regime == 0:  # Ranging market
            reversion_score *= 1.5  # Boost confidence
            reversion_reasoning.append("Ranging market - ideal for mean reversion")
        elif market_regime == 1:  # Trending market
            reversion_score *= 0.5  # Reduce confidence
            reversion_reasoning.append("Trending market - mean reversion less effective")
        
        # Determine final signal
        if reversion_score >= 3:
            signal = "STRONG_BUY"
            confidence = min(0.9, 0.6 + reversion_score * 0.1)
        elif reversion_score >= 1:
            signal = "BUY"
            confidence = min(0.8, 0.4 + reversion_score * 0.1)
        elif reversion_score <= -3:
            signal = "STRONG_SELL"
            confidence = min(0.9, 0.6 + abs(reversion_score) * 0.1)
        elif reversion_score <= -1:
            signal = "SELL"
            confidence = min(0.8, 0.4 + abs(reversion_score) * 0.1)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return {
            "strategy": "mean_reversion",
            "signal": signal,
            "confidence": confidence,
            "score": reversion_score,
            "reasoning": reversion_reasoning,
            "metadata": {
                "bb_position": bb_position,
                "rsi": rsi,
                "market_regime": market_regime,
                "support_levels": support_levels,
                "resistance_levels": resistance_levels
            }
        }
    
    def trend_following_strategy(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Crypto trend following strategy for trending markets
        
        Args:
            data: OHLCV data
            ticker: Crypto ticker
            
        Returns:
            Strategy signal and metadata
        """
        prices = data['close']
        volume = data['volume']
        
        # Get crypto indicators
        signals = self.indicators.get_crypto_signals(prices, volume, data.get('high'), data.get('low'))
        
        # Trend following specific logic
        trend_score = 0
        trend_reasoning = []
        
        # Multiple timeframe trend analysis
        ma_10 = prices.rolling(10).mean()
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        
        current_price = prices.iloc[-1]
        ma_10_val = ma_10.iloc[-1]
        ma_20_val = ma_20.iloc[-1]
        ma_50_val = ma_50.iloc[-1]
        
        # Trend alignment
        if current_price > ma_10_val > ma_20_val > ma_50_val:
            trend_score += 3
            trend_reasoning.append("Strong uptrend: Price > MA10 > MA20 > MA50")
        elif current_price > ma_10_val > ma_20_val:
            trend_score += 2
            trend_reasoning.append("Uptrend: Price > MA10 > MA20")
        elif current_price > ma_10_val:
            trend_score += 1
            trend_reasoning.append("Weak uptrend: Price > MA10")
        elif current_price < ma_10_val < ma_20_val < ma_50_val:
            trend_score -= 3
            trend_reasoning.append("Strong downtrend: Price < MA10 < MA20 < MA50")
        elif current_price < ma_10_val < ma_20_val:
            trend_score -= 2
            trend_reasoning.append("Downtrend: Price < MA10 < MA20")
        elif current_price < ma_10_val:
            trend_score -= 1
            trend_reasoning.append("Weak downtrend: Price < MA10")
        
        # MACD trend confirmation
        macd = signals['macd']['macd'].iloc[-1] if not signals['macd']['macd'].empty else 0
        macd_signal = signals['macd']['signal'].iloc[-1] if not signals['macd']['signal'].empty else 0
        macd_histogram = signals['macd']['histogram'].iloc[-1] if not signals['macd']['histogram'].empty else 0
        
        if macd > macd_signal and macd_histogram > 0:
            trend_score += 1
            trend_reasoning.append("MACD bullish trend confirmation")
        elif macd < macd_signal and macd_histogram < 0:
            trend_score -= 1
            trend_reasoning.append("MACD bearish trend confirmation")
        
        # Trend strength
        trend_strength = signals['trend_strength'].iloc[-1] if not signals['trend_strength'].empty else 0
        
        if trend_strength > 0.7:
            trend_score += 2
            trend_reasoning.append(f"Very strong trend: {trend_strength:.2f}")
        elif trend_strength > 0.5:
            trend_score += 1
            trend_reasoning.append(f"Strong trend: {trend_strength:.2f}")
        elif trend_strength < -0.7:
            trend_score -= 2
            trend_reasoning.append(f"Very strong downtrend: {trend_strength:.2f}")
        elif trend_strength < -0.5:
            trend_score -= 1
            trend_reasoning.append(f"Strong downtrend: {trend_strength:.2f}")
        
        # Volume confirmation
        volume_avg = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
        
        if volume_ratio > 1.2:
            trend_score += 1
            trend_reasoning.append(f"Volume confirmation: {volume_ratio:.1f}x")
        elif volume_ratio < 0.8:
            trend_score -= 1
            trend_reasoning.append(f"Low volume: {volume_ratio:.1f}x")
        
        # Market regime check (trend following works best in trending markets)
        market_regime = signals['market_regime'].iloc[-1] if not signals['market_regime'].empty else 0
        
        if market_regime == 1:  # Trending market
            trend_score *= 1.5  # Boost confidence
            trend_reasoning.append("Trending market - ideal for trend following")
        elif market_regime == 0:  # Ranging market
            trend_score *= 0.7  # Reduce confidence
            trend_reasoning.append("Ranging market - trend following less effective")
        
        # Determine final signal
        if trend_score >= 4:
            signal = "STRONG_BUY"
            confidence = min(0.95, 0.7 + trend_score * 0.05)
        elif trend_score >= 2:
            signal = "BUY"
            confidence = min(0.9, 0.5 + trend_score * 0.1)
        elif trend_score <= -4:
            signal = "STRONG_SELL"
            confidence = min(0.95, 0.7 + abs(trend_score) * 0.05)
        elif trend_score <= -2:
            signal = "SELL"
            confidence = min(0.9, 0.5 + abs(trend_score) * 0.1)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return {
            "strategy": "trend_following",
            "signal": signal,
            "confidence": confidence,
            "score": trend_score,
            "reasoning": trend_reasoning,
            "metadata": {
                "ma_alignment": {
                    "price": current_price,
                    "ma_10": ma_10_val,
                    "ma_20": ma_20_val,
                    "ma_50": ma_50_val
                },
                "macd": {
                    "macd": macd,
                    "signal": macd_signal,
                    "histogram": macd_histogram
                },
                "trend_strength": trend_strength,
                "volume_ratio": volume_ratio,
                "market_regime": market_regime
            }
        }
    
    def breakout_strategy(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Crypto breakout strategy for volatile markets
        
        Args:
            data: OHLCV data
            ticker: Crypto ticker
            
        Returns:
            Strategy signal and metadata
        """
        prices = data['close']
        volume = data['volume']
        high = data.get('high', prices)
        low = data.get('low', prices)
        
        # Get crypto indicators
        signals = self.indicators.get_crypto_signals(prices, volume, high, low)
        
        # Breakout specific logic
        breakout_score = 0
        breakout_reasoning = []
        
        # Price breakout analysis
        current_price = prices.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        
        # Recent high/low levels
        recent_high = high.rolling(20).max().iloc[-1]
        recent_low = low.rolling(20).min().iloc[-1]
        
        # Breakout thresholds
        high_breakout_threshold = recent_high * 1.01  # 1% above recent high
        low_breakout_threshold = recent_low * 0.99   # 1% below recent low
        
        if current_price > high_breakout_threshold:
            breakout_score += 3
            breakout_reasoning.append(f"Upward breakout: ${current_price:.2f} > ${high_breakout_threshold:.2f}")
        elif current_price > recent_high:
            breakout_score += 2
            breakout_reasoning.append(f"Above recent high: ${current_price:.2f} > ${recent_high:.2f}")
        elif current_price < low_breakout_threshold:
            breakout_score -= 3
            breakout_reasoning.append(f"Downward breakout: ${current_price:.2f} < ${low_breakout_threshold:.2f}")
        elif current_price < recent_low:
            breakout_score -= 2
            breakout_reasoning.append(f"Below recent low: ${current_price:.2f} < ${recent_low:.2f}")
        
        # Volume confirmation
        volume_avg = volume.rolling(20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
        
        if volume_ratio > 2.0:
            breakout_score += 2
            breakout_reasoning.append(f"High volume breakout: {volume_ratio:.1f}x")
        elif volume_ratio > 1.5:
            breakout_score += 1
            breakout_reasoning.append(f"Volume confirmation: {volume_ratio:.1f}x")
        elif volume_ratio < 0.8:
            breakout_score -= 1
            breakout_reasoning.append(f"Low volume: {volume_ratio:.1f}x")
        
        # Volatility check (breakouts work better in volatile markets)
        volatility = signals['volatility'].iloc[-1] if not signals['volatility'].empty else 0
        
        if volatility > 0.5:  # High volatility
            breakout_score += 1
            breakout_reasoning.append(f"High volatility environment: {volatility:.2f}")
        elif volatility < 0.2:  # Low volatility
            breakout_score -= 1
            breakout_reasoning.append(f"Low volatility: {volatility:.2f}")
        
        # ATR-based breakout confirmation
        atr = signals.get('atr', pd.Series([0])).iloc[-1] if 'atr' in signals else 0
        
        if atr > 0:
            atr_breakout_threshold = atr * 0.5  # Half ATR
            price_change = abs(current_price - prices.iloc[-2]) if len(prices) > 1 else 0
            
            if price_change > atr_breakout_threshold:
                breakout_score += 1
                breakout_reasoning.append(f"ATR breakout confirmation: {price_change:.2f} > {atr_breakout_threshold:.2f}")
        
        # Market regime check
        market_regime = signals['market_regime'].iloc[-1] if not signals['market_regime'].empty else 0
        
        if market_regime == 2:  # Volatile market
            breakout_score *= 1.3  # Boost confidence
            breakout_reasoning.append("Volatile market - ideal for breakouts")
        elif market_regime == 0:  # Ranging market
            breakout_score *= 0.8  # Reduce confidence
            breakout_reasoning.append("Ranging market - breakouts less reliable")
        
        # Determine final signal
        if breakout_score >= 4:
            signal = "STRONG_BUY"
            confidence = min(0.95, 0.6 + breakout_score * 0.05)
        elif breakout_score >= 2:
            signal = "BUY"
            confidence = min(0.9, 0.4 + breakout_score * 0.1)
        elif breakout_score <= -4:
            signal = "STRONG_SELL"
            confidence = min(0.95, 0.6 + abs(breakout_score) * 0.05)
        elif breakout_score <= -2:
            signal = "SELL"
            confidence = min(0.9, 0.4 + abs(breakout_score) * 0.1)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return {
            "strategy": "breakout",
            "signal": signal,
            "confidence": confidence,
            "score": breakout_score,
            "reasoning": breakout_reasoning,
            "metadata": {
                "recent_high": recent_high,
                "recent_low": recent_low,
                "breakout_thresholds": {
                    "high": high_breakout_threshold,
                    "low": low_breakout_threshold
                },
                "volume_ratio": volume_ratio,
                "volatility": volatility,
                "atr": atr,
                "market_regime": market_regime
            }
        }
    
    def adaptive_strategy(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Adaptive strategy that selects the best strategy based on market conditions
        
        Args:
            data: OHLCV data
            ticker: Crypto ticker
            
        Returns:
            Strategy signal and metadata
        """
        # Run all strategies
        momentum_result = self.momentum_strategy(data, ticker)
        mean_reversion_result = self.mean_reversion_strategy(data, ticker)
        trend_following_result = self.trend_following_strategy(data, ticker)
        breakout_result = self.breakout_strategy(data, ticker)
        
        # Get market regime
        prices = data['close']
        volume = data['volume']
        signals = self.indicators.get_crypto_signals(prices, volume, data.get('high'), data.get('low'))
        market_regime = signals['market_regime'].iloc[-1] if not signals['market_regime'].empty else 0
        
        # Strategy weights based on market regime
        if market_regime == 0:  # Ranging
            weights = {
                'momentum': 0.2,
                'mean_reversion': 0.5,
                'trend_following': 0.1,
                'breakout': 0.2
            }
        elif market_regime == 1:  # Trending
            weights = {
                'momentum': 0.3,
                'mean_reversion': 0.1,
                'trend_following': 0.5,
                'breakout': 0.1
            }
        else:  # Volatile
            weights = {
                'momentum': 0.2,
                'mean_reversion': 0.2,
                'trend_following': 0.2,
                'breakout': 0.4
            }
        
        # Calculate weighted scores
        weighted_score = 0
        total_confidence = 0
        all_reasoning = []
        
        strategies = {
            'momentum': momentum_result,
            'mean_reversion': mean_reversion_result,
            'trend_following': trend_following_result,
            'breakout': breakout_result
        }
        
        for strategy_name, result in strategies.items():
            weight = weights[strategy_name]
            score = result['score']
            confidence = result['confidence']
            
            weighted_score += score * weight
            total_confidence += confidence * weight
            all_reasoning.extend([f"{strategy_name}: {reason}" for reason in result['reasoning']])
        
        # Determine final signal
        if weighted_score >= 3:
            signal = "STRONG_BUY"
        elif weighted_score >= 1:
            signal = "BUY"
        elif weighted_score <= -3:
            signal = "STRONG_SELL"
        elif weighted_score <= -1:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            "strategy": "adaptive",
            "signal": signal,
            "confidence": total_confidence,
            "score": weighted_score,
            "reasoning": all_reasoning,
            "metadata": {
                "market_regime": market_regime,
                "strategy_weights": weights,
                "individual_strategies": {
                    name: {
                        "signal": result['signal'],
                        "score": result['score'],
                        "confidence": result['confidence']
                    }
                    for name, result in strategies.items()
                }
            }
        }
    
    def get_strategy_signal(self, data: pd.DataFrame, ticker: str, strategy: str = "adaptive") -> Dict[str, Any]:
        """
        Get trading signal from specified strategy
        
        Args:
            data: OHLCV data
            ticker: Crypto ticker
            strategy: Strategy name (momentum, mean_reversion, trend_following, breakout, adaptive)
            
        Returns:
            Strategy signal and metadata
        """
        if strategy == "momentum":
            return self.momentum_strategy(data, ticker)
        elif strategy == "mean_reversion":
            return self.mean_reversion_strategy(data, ticker)
        elif strategy == "trend_following":
            return self.trend_following_strategy(data, ticker)
        elif strategy == "breakout":
            return self.breakout_strategy(data, ticker)
        elif strategy == "adaptive":
            return self.adaptive_strategy(data, ticker)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

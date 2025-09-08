"""
Crypto-Specific Technical Indicators

This module provides technical indicators specifically optimized for cryptocurrency analysis,
including volatility indicators, momentum oscillators, and trend-following tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CryptoIndicators:
    """Crypto-specific technical indicators"""
    
    @staticmethod
    def crypto_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Crypto-optimized RSI with faster response to volatility
        
        Args:
            prices: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Crypto-specific adjustments
        # More sensitive to rapid price changes
        rsi = rsi * 1.1  # Slight amplification
        rsi = rsi.clip(0, 100)  # Ensure bounds
        
        return rsi
    
    @staticmethod
    def crypto_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Crypto-optimized MACD with enhanced signal detection
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        # Crypto-specific enhancements
        # Amplify signals during high volatility
        volatility = prices.pct_change().rolling(20).std()
        volatility_multiplier = 1 + (volatility * 2)  # Amplify during high vol
        histogram = histogram * volatility_multiplier
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    @staticmethod
    def crypto_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Crypto-optimized Bollinger Bands with dynamic width
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        # Dynamic standard deviation based on volatility
        volatility = prices.pct_change().rolling(20).std()
        dynamic_std = std_dev * (1 + volatility * 0.5)  # Wider bands during high vol
        
        upper_band = sma + (std * dynamic_std)
        lower_band = sma - (std * dynamic_std)
        
        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band
        }
    
    @staticmethod
    def crypto_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Crypto-optimized Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR series
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        
        # Crypto-specific: More responsive to gaps
        atr = atr * 1.2  # Slight amplification for crypto volatility
        
        return atr
    
    @staticmethod
    def crypto_volume_profile(volume: pd.Series, prices: pd.Series, bins: int = 20) -> Dict[str, np.ndarray]:
        """
        Crypto volume profile analysis
        
        Args:
            volume: Volume series
            prices: Price series
            bins: Number of price bins
            
        Returns:
            Volume profile data
        """
        # Create price bins
        price_min = prices.min()
        price_max = prices.max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Calculate volume in each bin
        volume_profile = np.zeros(bins)
        for i in range(len(prices)):
            bin_idx = np.digitize(prices.iloc[i], price_bins) - 1
            bin_idx = max(0, min(bin_idx, bins - 1))
            volume_profile[bin_idx] += volume.iloc[i]
        
        return {
            "volume_profile": volume_profile,
            "price_levels": (price_bins[:-1] + price_bins[1:]) / 2,
            "poc": price_bins[np.argmax(volume_profile)]  # Point of Control
        }
    
    @staticmethod
    def crypto_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Crypto momentum indicator
        
        Args:
            prices: Price series
            period: Momentum period
            
        Returns:
            Momentum series
        """
        momentum = prices.pct_change(period) * 100  # As percentage
        
        # Crypto-specific: Amplify during trending markets
        trend_strength = abs(prices.pct_change(period).rolling(5).mean())
        momentum = momentum * (1 + trend_strength)
        
        return momentum
    
    @staticmethod
    def crypto_volatility(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Crypto volatility indicator
        
        Args:
            prices: Price series
            period: Volatility period
            
        Returns:
            Volatility series (annualized)
        """
        returns = prices.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(365)  # Annualized
        
        return volatility
    
    @staticmethod
    def crypto_support_resistance(prices: pd.Series, window: int = 20, threshold: float = 0.02) -> Dict[str, List[float]]:
        """
        Identify crypto support and resistance levels
        
        Args:
            prices: Price series
            window: Window for local extrema
            threshold: Minimum price change threshold
            
        Returns:
            Support and resistance levels
        """
        # Find local maxima and minima
        highs = prices.rolling(window=window, center=True).max() == prices
        lows = prices.rolling(window=window, center=True).min() == prices
        
        # Extract levels
        resistance_levels = prices[highs].tolist()
        support_levels = prices[lows].tolist()
        
        # Filter by threshold
        resistance_levels = [level for level in resistance_levels if not pd.isna(level)]
        support_levels = [level for level in support_levels if not pd.isna(level)]
        
        # Group similar levels
        def group_levels(levels, threshold):
            if not levels:
                return []
            
            levels = sorted(levels)
            grouped = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - grouped[-1]) / grouped[-1] > threshold:
                    grouped.append(level)
                else:
                    # Average with existing level
                    grouped[-1] = (grouped[-1] + level) / 2
            
            return grouped
        
        resistance_levels = group_levels(resistance_levels, threshold)
        support_levels = group_levels(support_levels, threshold)
        
        return {
            "resistance": resistance_levels,
            "support": support_levels
        }
    
    @staticmethod
    def crypto_trend_strength(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Crypto trend strength indicator
        
        Args:
            prices: Price series
            period: Period for trend calculation
            
        Returns:
            Trend strength series (-1 to 1)
        """
        # Calculate multiple timeframes
        short_ma = prices.rolling(window=period//2).mean()
        long_ma = prices.rolling(window=period).mean()
        
        # Price position relative to moving averages
        price_position = (prices - short_ma) / (long_ma - short_ma)
        
        # Trend direction
        trend_direction = np.where(short_ma > long_ma, 1, -1)
        
        # Trend strength
        trend_strength = price_position * trend_direction
        
        # Normalize to -1 to 1
        trend_strength = np.clip(trend_strength, -1, 1)
        
        return pd.Series(trend_strength, index=prices.index)
    
    @staticmethod
    def crypto_market_regime(prices: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Identify crypto market regime (trending, ranging, volatile)
        
        Args:
            prices: Price series
            volume: Volume series
            period: Analysis period
            
        Returns:
            Market regime series (0: ranging, 1: trending, 2: volatile)
        """
        # Calculate metrics
        returns = prices.pct_change()
        volatility = returns.rolling(window=period).std()
        trend_strength = abs(prices.rolling(window=period).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        volume_ratio = volume / volume.rolling(window=period).mean()
        
        # Define regimes
        regime = np.zeros(len(prices))
        
        # Volatile regime (high volatility)
        volatile_mask = volatility > volatility.quantile(0.8)
        regime[volatile_mask] = 2
        
        # Trending regime (strong trend, normal volatility)
        trending_mask = (trend_strength > trend_strength.quantile(0.7)) & (~volatile_mask)
        regime[trending_mask] = 1
        
        # Ranging regime (everything else)
        ranging_mask = (regime == 0)
        regime[ranging_mask] = 0
        
        return pd.Series(regime, index=prices.index)
    
    @staticmethod
    def get_crypto_signals(prices: pd.Series, volume: pd.Series, 
                          high: pd.Series = None, low: pd.Series = None) -> Dict[str, pd.Series]:
        """
        Generate comprehensive crypto trading signals
        
        Args:
            prices: Close prices
            volume: Volume data
            high: High prices (optional)
            low: Low prices (optional)
            
        Returns:
            Dictionary of all crypto signals
        """
        signals = {}
        
        # Basic indicators
        signals['rsi'] = CryptoIndicators.crypto_rsi(prices)
        signals['macd'] = CryptoIndicators.crypto_macd(prices)
        signals['bollinger'] = CryptoIndicators.crypto_bollinger_bands(prices)
        signals['momentum'] = CryptoIndicators.crypto_momentum(prices)
        signals['volatility'] = CryptoIndicators.crypto_volatility(prices)
        signals['trend_strength'] = CryptoIndicators.crypto_trend_strength(prices)
        signals['market_regime'] = CryptoIndicators.crypto_market_regime(prices, volume)
        
        # Volume analysis
        signals['volume_profile'] = CryptoIndicators.crypto_volume_profile(volume, prices)
        
        # Support/Resistance
        signals['levels'] = CryptoIndicators.crypto_support_resistance(prices)
        
        # ATR if high/low available
        if high is not None and low is not None:
            signals['atr'] = CryptoIndicators.crypto_atr(high, low, prices)
        
        return signals
    
    @staticmethod
    def generate_crypto_trading_signal(signals: Dict[str, any], current_price: float) -> Dict[str, any]:
        """
        Generate final crypto trading signal based on all indicators
        
        Args:
            signals: Dictionary of all crypto signals
            current_price: Current price
            
        Returns:
            Trading signal with confidence and reasoning
        """
        # Extract current values
        rsi = signals['rsi'].iloc[-1] if not signals['rsi'].empty else 50
        macd = signals['macd']['macd'].iloc[-1] if not signals['macd']['macd'].empty else 0
        macd_signal = signals['macd']['signal'].iloc[-1] if not signals['macd']['signal'].empty else 0
        momentum = signals['momentum'].iloc[-1] if not signals['momentum'].empty else 0
        trend_strength = signals['trend_strength'].iloc[-1] if not signals['trend_strength'].empty else 0
        market_regime = signals['market_regime'].iloc[-1] if not signals['market_regime'].empty else 0
        
        # Bollinger Bands
        bb_upper = signals['bollinger']['upper'].iloc[-1] if not signals['bollinger']['upper'].empty else current_price * 1.02
        bb_lower = signals['bollinger']['lower'].iloc[-1] if not signals['bollinger']['lower'].empty else current_price * 0.98
        bb_middle = signals['bollinger']['middle'].iloc[-1] if not signals['bollinger']['middle'].empty else current_price
        
        # Signal logic
        signal_score = 0
        reasoning = []
        
        # RSI signals
        if rsi < 30:
            signal_score += 2  # Strong buy
            reasoning.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            signal_score += 1  # Weak buy
            reasoning.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            signal_score -= 2  # Strong sell
            reasoning.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            signal_score -= 1  # Weak sell
            reasoning.append(f"RSI high ({rsi:.1f})")
        
        # MACD signals
        if macd > macd_signal and macd > 0:
            signal_score += 1
            reasoning.append("MACD bullish crossover")
        elif macd < macd_signal and macd < 0:
            signal_score -= 1
            reasoning.append("MACD bearish crossover")
        
        # Bollinger Bands
        if current_price <= bb_lower:
            signal_score += 2
            reasoning.append("Price at lower Bollinger Band")
        elif current_price >= bb_upper:
            signal_score -= 2
            reasoning.append("Price at upper Bollinger Band")
        
        # Momentum
        if momentum > 5:
            signal_score += 1
            reasoning.append(f"Strong momentum ({momentum:.1f}%)")
        elif momentum < -5:
            signal_score -= 1
            reasoning.append(f"Negative momentum ({momentum:.1f}%)")
        
        # Trend strength
        if trend_strength > 0.5:
            signal_score += 1
            reasoning.append("Strong uptrend")
        elif trend_strength < -0.5:
            signal_score -= 1
            reasoning.append("Strong downtrend")
        
        # Market regime adjustment
        if market_regime == 2:  # Volatile
            signal_score *= 0.5  # Reduce confidence in volatile markets
            reasoning.append("High volatility - reduced confidence")
        elif market_regime == 1:  # Trending
            signal_score *= 1.2  # Increase confidence in trending markets
            reasoning.append("Trending market - increased confidence")
        
        # Determine final signal
        if signal_score >= 3:
            signal = "STRONG_BUY"
            confidence = min(0.9, 0.5 + signal_score * 0.1)
        elif signal_score >= 1:
            signal = "BUY"
            confidence = min(0.8, 0.3 + signal_score * 0.1)
        elif signal_score <= -3:
            signal = "STRONG_SELL"
            confidence = min(0.9, 0.5 + abs(signal_score) * 0.1)
        elif signal_score <= -1:
            signal = "SELL"
            confidence = min(0.8, 0.3 + abs(signal_score) * 0.1)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return {
            "signal": signal,
            "confidence": confidence,
            "score": signal_score,
            "reasoning": reasoning,
            "indicators": {
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "momentum": momentum,
                "trend_strength": trend_strength,
                "market_regime": market_regime,
                "bb_position": (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            }
        }

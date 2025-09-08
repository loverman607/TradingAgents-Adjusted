"""
Crypto Risk Management Module

This module provides specialized risk management tools for cryptocurrency trading,
including volatility-based position sizing, dynamic stop losses, and portfolio risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CryptoRiskManager:
    """Crypto-specific risk management system"""
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk per trade
                 max_daily_loss: float = 0.05,     # 5% max daily loss
                 max_correlation: float = 0.7,     # Max correlation between positions
                 volatility_lookback: int = 30,    # Days for volatility calculation
                 enable_dynamic_sizing: bool = True):
        """
        Initialize crypto risk manager
        
        Args:
            max_portfolio_risk: Maximum portfolio risk per trade
            max_daily_loss: Maximum daily loss as percentage of portfolio
            max_correlation: Maximum correlation between positions
            volatility_lookback: Days for volatility calculation
            enable_dynamic_sizing: Enable dynamic position sizing
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_loss = max_daily_loss
        self.max_correlation = max_correlation
        self.volatility_lookback = volatility_lookback
        self.enable_dynamic_sizing = enable_dynamic_sizing
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.position_correlations = {}
        self.volatility_cache = {}
        
        # Crypto-specific risk parameters
        self.crypto_risk_params = {
            "BTC": {
                "base_volatility": 0.03,
                "max_position_size": 0.3,
                "stop_loss_multiplier": 2.0,
                "take_profit_multiplier": 3.0
            },
            "ETH": {
                "base_volatility": 0.04,
                "max_position_size": 0.25,
                "stop_loss_multiplier": 2.5,
                "take_profit_multiplier": 4.0
            }
        }
    
    def reset_daily_limits(self):
        """Reset daily limits if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info("Daily risk limits reset")
    
    def calculate_volatility(self, prices: pd.Series, ticker: str) -> float:
        """
        Calculate crypto volatility with caching
        
        Args:
            prices: Price series
            ticker: Crypto ticker
            
        Returns:
            Volatility (annualized)
        """
        if ticker in self.volatility_cache:
            cache_time, volatility = self.volatility_cache[ticker]
            if datetime.now() - cache_time < timedelta(hours=1):
                return volatility
        
        # Calculate volatility
        returns = prices.pct_change().dropna()
        if len(returns) < 10:
            # Use base volatility if insufficient data
            volatility = self.crypto_risk_params.get(ticker, {}).get("base_volatility", 0.05)
        else:
            volatility = returns.std() * np.sqrt(365)  # Annualized
        
        # Cache the result
        self.volatility_cache[ticker] = (datetime.now(), volatility)
        
        return volatility
    
    def calculate_position_size(self, 
                              ticker: str, 
                              price: float, 
                              portfolio_value: float,
                              stop_loss_price: float,
                              signal_strength: float = 1.0,
                              historical_prices: pd.Series = None) -> Dict[str, Any]:
        """
        Calculate optimal position size for crypto trading
        
        Args:
            ticker: Crypto ticker
            price: Current price
            portfolio_value: Total portfolio value
            stop_loss_price: Stop loss price
            signal_strength: Signal strength (0.0 to 1.0)
            historical_prices: Historical prices for volatility calculation
            
        Returns:
            Position sizing information
        """
        self.reset_daily_limits()
        
        # Get risk parameters for ticker
        risk_params = self.crypto_risk_params.get(ticker, {
            "base_volatility": 0.05,
            "max_position_size": 0.2,
            "stop_loss_multiplier": 2.0,
            "take_profit_multiplier": 3.0
        })
        
        # Calculate volatility
        if historical_prices is not None and len(historical_prices) >= 10:
            volatility = self.calculate_volatility(historical_prices, ticker)
        else:
            volatility = risk_params["base_volatility"]
        
        # Calculate risk per unit
        risk_per_unit = abs(price - stop_loss_price)
        risk_per_unit_pct = risk_per_unit / price
        
        # Calculate position size based on portfolio risk
        max_risk_amount = portfolio_value * self.max_portfolio_risk
        base_position_size = max_risk_amount / risk_per_unit
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = min(1.0, risk_params["base_volatility"] / volatility)
        
        # Adjust for signal strength
        signal_adjustment = signal_strength
        
        # Calculate final position size
        position_size = base_position_size * volatility_adjustment * signal_adjustment
        
        # Apply maximum position size limit
        max_position_value = portfolio_value * risk_params["max_position_size"]
        max_position_size = max_position_value / price
        
        position_size = min(position_size, max_position_size)
        
        # Calculate position value and risk
        position_value = position_size * price
        position_risk = position_size * risk_per_unit
        position_risk_pct = position_risk / portfolio_value
        
        return {
            "position_size": position_size,
            "position_value": position_value,
            "position_risk": position_risk,
            "position_risk_pct": position_risk_pct,
            "volatility": volatility,
            "volatility_adjustment": volatility_adjustment,
            "signal_adjustment": signal_adjustment,
            "max_position_size": max_position_size,
            "risk_per_unit": risk_per_unit,
            "risk_per_unit_pct": risk_per_unit_pct
        }
    
    def calculate_dynamic_stop_loss(self, 
                                  ticker: str, 
                                  entry_price: float, 
                                  position_type: str = "long",
                                  historical_prices: pd.Series = None) -> Dict[str, Any]:
        """
        Calculate dynamic stop loss for crypto positions
        
        Args:
            ticker: Crypto ticker
            entry_price: Entry price
            position_type: Position type (long/short)
            historical_prices: Historical prices for ATR calculation
            
        Returns:
            Stop loss information
        """
        # Get risk parameters
        risk_params = self.crypto_risk_params.get(ticker, {
            "base_volatility": 0.05,
            "stop_loss_multiplier": 2.0
        })
        
        # Calculate ATR if historical data available
        if historical_prices is not None and len(historical_prices) >= 14:
            atr = self._calculate_atr(historical_prices)
        else:
            # Use volatility-based ATR
            volatility = self.calculate_volatility(historical_prices, ticker) if historical_prices is not None else risk_params["base_volatility"]
            atr = entry_price * volatility / np.sqrt(365)  # Daily volatility
        
        # Calculate stop loss distance
        stop_loss_multiplier = risk_params["stop_loss_multiplier"]
        stop_loss_distance = atr * stop_loss_multiplier
        
        # Calculate stop loss price
        if position_type == "long":
            stop_loss_price = entry_price - stop_loss_distance
        else:  # short
            stop_loss_price = entry_price + stop_loss_distance
        
        # Ensure stop loss is reasonable (not too close or too far)
        min_stop_distance = entry_price * 0.01  # 1% minimum
        max_stop_distance = entry_price * 0.20  # 20% maximum
        
        stop_loss_distance = max(min_stop_distance, min(stop_loss_distance, max_stop_distance))
        
        if position_type == "long":
            stop_loss_price = entry_price - stop_loss_distance
        else:
            stop_loss_price = entry_price + stop_loss_distance
        
        return {
            "stop_loss_price": stop_loss_price,
            "stop_loss_distance": stop_loss_distance,
            "stop_loss_pct": stop_loss_distance / entry_price,
            "atr": atr,
            "atr_multiplier": stop_loss_multiplier
        }
    
    def calculate_dynamic_take_profit(self, 
                                    ticker: str, 
                                    entry_price: float, 
                                    position_type: str = "long",
                                    historical_prices: pd.Series = None) -> Dict[str, Any]:
        """
        Calculate dynamic take profit for crypto positions
        
        Args:
            ticker: Crypto ticker
            entry_price: Entry price
            position_type: Position type (long/short)
            historical_prices: Historical prices for ATR calculation
            
        Returns:
            Take profit information
        """
        # Get risk parameters
        risk_params = self.crypto_risk_params.get(ticker, {
            "base_volatility": 0.05,
            "take_profit_multiplier": 3.0
        })
        
        # Calculate ATR if historical data available
        if historical_prices is not None and len(historical_prices) >= 14:
            atr = self._calculate_atr(historical_prices)
        else:
            # Use volatility-based ATR
            volatility = self.calculate_volatility(historical_prices, ticker) if historical_prices is not None else risk_params["base_volatility"]
            atr = entry_price * volatility / np.sqrt(365)
        
        # Calculate take profit distance
        take_profit_multiplier = risk_params["take_profit_multiplier"]
        take_profit_distance = atr * take_profit_multiplier
        
        # Calculate take profit price
        if position_type == "long":
            take_profit_price = entry_price + take_profit_distance
        else:  # short
            take_profit_price = entry_price - take_profit_distance
        
        return {
            "take_profit_price": take_profit_price,
            "take_profit_distance": take_profit_distance,
            "take_profit_pct": take_profit_distance / entry_price,
            "atr": atr,
            "atr_multiplier": take_profit_multiplier
        }
    
    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return prices.std() * 0.1  # Fallback
        
        # For simplicity, use price range as proxy for true range
        price_ranges = []
        for i in range(1, len(prices)):
            price_range = abs(prices.iloc[i] - prices.iloc[i-1])
            price_ranges.append(price_range)
        
        atr = np.mean(price_ranges[-period:]) if price_ranges else prices.std() * 0.1
        return atr
    
    def check_portfolio_risk(self, 
                           current_positions: Dict[str, Any], 
                           new_position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check portfolio risk before adding new position
        
        Args:
            current_positions: Current portfolio positions
            new_position: New position to add
            
        Returns:
            Risk check results
        """
        self.reset_daily_limits()
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return {
                "can_add_position": False,
                "reason": "Daily loss limit exceeded",
                "daily_pnl": self.daily_pnl,
                "max_daily_loss": self.max_daily_loss
            }
        
        # Check position correlation
        if len(current_positions) > 0:
            correlation_risk = self._check_correlation_risk(current_positions, new_position)
            if not correlation_risk["can_add"]:
                return {
                    "can_add_position": False,
                    "reason": f"High correlation with existing positions: {correlation_risk['max_correlation']:.2f}",
                    "correlation_details": correlation_risk
                }
        
        # Check total portfolio risk
        total_risk = sum(pos.get("position_risk_pct", 0) for pos in current_positions.values())
        total_risk += new_position.get("position_risk_pct", 0)
        
        if total_risk > self.max_portfolio_risk * 3:  # 3x max single position risk
            return {
                "can_add_position": False,
                "reason": f"Total portfolio risk too high: {total_risk:.2%}",
                "total_risk": total_risk,
                "max_total_risk": self.max_portfolio_risk * 3
            }
        
        return {
            "can_add_position": True,
            "reason": "Risk checks passed",
            "daily_pnl": self.daily_pnl,
            "total_risk": total_risk,
            "correlation_risk": correlation_risk if len(current_positions) > 0 else None
        }
    
    def _check_correlation_risk(self, 
                               current_positions: Dict[str, Any], 
                               new_position: Dict[str, Any]) -> Dict[str, Any]:
        """Check correlation risk between positions"""
        # This is a simplified correlation check
        # In practice, you would calculate actual price correlations
        
        new_ticker = new_position.get("ticker", "")
        
        # Check for same ticker
        if new_ticker in current_positions:
            return {
                "can_add": False,
                "max_correlation": 1.0,
                "reason": "Same ticker already in portfolio"
            }
        
        # Check for highly correlated pairs
        crypto_correlations = {
            "BTC": ["ETH"],  # BTC and ETH are somewhat correlated
            "ETH": ["BTC"]
        }
        
        correlated_tickers = crypto_correlations.get(new_ticker, [])
        for ticker in correlated_tickers:
            if ticker in current_positions:
                return {
                    "can_add": False,
                    "max_correlation": 0.8,  # Estimated correlation
                    "reason": f"High correlation with {ticker}"
                }
        
        return {
            "can_add": True,
            "max_correlation": 0.0,
            "reason": "No significant correlations found"
        }
    
    def calculate_portfolio_metrics(self, 
                                  positions: Dict[str, Any], 
                                  current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            positions: Current positions
            current_prices: Current prices for all tickers
            
        Returns:
            Portfolio risk metrics
        """
        if not positions:
            return {
                "total_positions": 0,
                "total_value": 0,
                "total_risk": 0,
                "diversification_score": 1.0,
                "concentration_risk": 0.0,
                "correlation_risk": 0.0
            }
        
        # Calculate basic metrics
        total_value = sum(pos.get("position_value", 0) for pos in positions.values())
        total_risk = sum(pos.get("position_risk", 0) for pos in positions.values())
        
        # Calculate concentration risk (largest position as % of total)
        if total_value > 0:
            position_values = [pos.get("position_value", 0) for pos in positions.values()]
            max_position_value = max(position_values)
            concentration_risk = max_position_value / total_value
        else:
            concentration_risk = 0.0
        
        # Calculate diversification score
        num_positions = len(positions)
        diversification_score = min(1.0, num_positions / 5.0)  # Optimal at 5 positions
        
        # Calculate correlation risk (simplified)
        correlation_risk = self._calculate_portfolio_correlation_risk(positions)
        
        return {
            "total_positions": num_positions,
            "total_value": total_value,
            "total_risk": total_risk,
            "total_risk_pct": total_risk / total_value if total_value > 0 else 0,
            "diversification_score": diversification_score,
            "concentration_risk": concentration_risk,
            "correlation_risk": correlation_risk,
            "daily_pnl": self.daily_pnl,
            "max_daily_loss": self.max_daily_loss,
            "risk_utilization": total_risk / (self.max_daily_loss * total_value) if total_value > 0 else 0
        }
    
    def _calculate_portfolio_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio correlation risk (simplified)"""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation risk based on crypto pairs
        crypto_correlations = {
            ("BTC", "ETH"): 0.7,
            ("ETH", "BTC"): 0.7
        }
        
        tickers = list(positions.keys())
        max_correlation = 0.0
        
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                correlation = crypto_correlations.get((ticker1, ticker2), 0.0)
                max_correlation = max(max_correlation, correlation)
        
        return max_correlation
    
    def get_risk_recommendations(self, 
                               positions: Dict[str, Any], 
                               portfolio_metrics: Dict[str, Any]) -> List[str]:
        """
        Get risk management recommendations
        
        Args:
            positions: Current positions
            portfolio_metrics: Portfolio risk metrics
            
        Returns:
            List of risk recommendations
        """
        recommendations = []
        
        # Check concentration risk
        if portfolio_metrics["concentration_risk"] > 0.4:
            recommendations.append("High concentration risk - consider diversifying positions")
        
        # Check correlation risk
        if portfolio_metrics["correlation_risk"] > 0.6:
            recommendations.append("High correlation risk - consider reducing correlated positions")
        
        # Check total risk
        if portfolio_metrics["total_risk_pct"] > 0.1:
            recommendations.append("High total portfolio risk - consider reducing position sizes")
        
        # Check diversification
        if portfolio_metrics["diversification_score"] < 0.6:
            recommendations.append("Low diversification - consider adding more positions")
        
        # Check daily P&L
        if self.daily_pnl < -self.max_daily_loss * 0.8:
            recommendations.append("Approaching daily loss limit - consider reducing risk")
        
        # Check risk utilization
        if portfolio_metrics["risk_utilization"] > 0.8:
            recommendations.append("High risk utilization - consider taking profits or reducing positions")
        
        if not recommendations:
            recommendations.append("Portfolio risk levels are acceptable")
        
        return recommendations

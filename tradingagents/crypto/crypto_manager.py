"""
Crypto Trading Manager

This module provides the main interface for crypto-optimized trading,
integrating all crypto-specific components for BTC and ETH trading.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .crypto_engine import CryptoTradingEngine, CryptoOrder, CryptoOrderSide, CryptoOrderType
from .crypto_indicators import CryptoIndicators
from .crypto_strategies import CryptoStrategies
from .crypto_data import CryptoDataProvider
from .crypto_risk import CryptoRiskManager

logger = logging.getLogger(__name__)


class CryptoTradingManager:
    """
    Main crypto trading manager that integrates all crypto-specific components
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.2,
                 enable_24_7: bool = True,
                 enable_leverage: bool = True,
                 max_leverage: float = 3.0,
                 state_file: str = "crypto_trading_state.json"):
        """
        Initialize crypto trading manager
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as percentage of portfolio
            enable_24_7: Enable 24/7 trading
            enable_leverage: Enable leveraged trading
            max_leverage: Maximum leverage allowed
            state_file: State file for persistence
        """
        # Initialize components
        self.engine = CryptoTradingEngine(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            enable_24_7=enable_24_7,
            enable_leverage=enable_leverage,
            max_leverage=max_leverage,
            state_file=state_file
        )
        
        self.indicators = CryptoIndicators()
        self.strategies = CryptoStrategies()
        self.data_provider = CryptoDataProvider()
        self.risk_manager = CryptoRiskManager()
        
        # Trading configuration
        self.tickers = ["BTC", "ETH"]
        self.strategy = "adaptive"  # Default strategy
        self.analysis_frequency = "1h"  # 1 hour analysis frequency
        
        logger.info(f"Crypto trading manager initialized with ${initial_capital:,.2f} capital")
    
    def analyze_crypto_market(self, ticker: str, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Perform comprehensive crypto market analysis
        
        Args:
            ticker: Crypto ticker (BTC, ETH)
            lookback_days: Days of historical data to analyze
            
        Returns:
            Comprehensive market analysis
        """
        try:
            logger.info(f"Analyzing crypto market for {ticker}")
            
            # Get historical data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            historical_data = self.data_provider.get_historical_data(
                ticker, start_date, end_date, interval="1h"
            )
            
            if historical_data.empty:
                logger.warning(f"No historical data available for {ticker}")
                return self._get_fallback_analysis(ticker)
            
            # Get real-time data
            realtime_data = self.data_provider.get_realtime_price(ticker)
            
            # Get fundamentals
            fundamentals = self.data_provider.get_crypto_fundamentals(ticker)
            
            # Get sentiment
            sentiment = self.data_provider.get_market_sentiment(ticker)
            
            # Generate trading signal
            signal_result = self.strategies.get_strategy_signal(
                historical_data, ticker, self.strategy
            )
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_portfolio_metrics(
                self.engine.positions, {ticker: realtime_data["price"]}
            )
            
            # Generate technical analysis
            technical_analysis = self._generate_technical_analysis(historical_data, ticker)
            
            # Generate market outlook
            market_outlook = self._generate_market_outlook(
                ticker, signal_result, technical_analysis, fundamentals, sentiment
            )
            
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "realtime_data": realtime_data,
                "fundamentals": fundamentals,
                "sentiment": sentiment,
                "trading_signal": signal_result,
                "technical_analysis": technical_analysis,
                "market_outlook": market_outlook,
                "risk_metrics": risk_metrics,
                "historical_data_points": len(historical_data),
                "analysis_quality": "high" if len(historical_data) > 100 else "medium"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing crypto market for {ticker}: {e}")
            return self._get_fallback_analysis(ticker)
    
    def _generate_technical_analysis(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Generate technical analysis"""
        try:
            # Get all crypto indicators
            signals = self.indicators.get_crypto_signals(
                data['close'], data['volume'], data.get('high'), data.get('low')
            )
            
            # Calculate additional metrics
            current_price = data['close'].iloc[-1]
            price_change_24h = ((current_price - data['close'].iloc[-25]) / 
                               data['close'].iloc[-25] * 100) if len(data) >= 25 else 0
            
            # Volatility analysis
            volatility_7d = data['close'].pct_change().rolling(7*24).std().iloc[-1] * np.sqrt(365) * 100
            volatility_30d = data['close'].pct_change().rolling(30*24).std().iloc[-1] * np.sqrt(365) * 100
            
            # Trend analysis
            sma_20 = data['close'].rolling(20*24).mean().iloc[-1]
            sma_50 = data['close'].rolling(50*24).mean().iloc[-1] if len(data) >= 50*24 else sma_20
            
            trend_direction = "bullish" if current_price > sma_20 > sma_50 else "bearish" if current_price < sma_20 < sma_50 else "neutral"
            
            return {
                "current_price": current_price,
                "price_change_24h": price_change_24h,
                "volatility_7d": volatility_7d,
                "volatility_30d": volatility_30d,
                "trend_direction": trend_direction,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": signals['rsi'].iloc[-1] if not signals['rsi'].empty else 50,
                "macd": {
                    "macd": signals['macd']['macd'].iloc[-1] if not signals['macd']['macd'].empty else 0,
                    "signal": signals['macd']['signal'].iloc[-1] if not signals['macd']['signal'].empty else 0,
                    "histogram": signals['macd']['histogram'].iloc[-1] if not signals['macd']['histogram'].empty else 0
                },
                "bollinger_bands": {
                    "upper": signals['bollinger']['upper'].iloc[-1] if not signals['bollinger']['upper'].empty else current_price * 1.02,
                    "middle": signals['bollinger']['middle'].iloc[-1] if not signals['bollinger']['middle'].empty else current_price,
                    "lower": signals['bollinger']['lower'].iloc[-1] if not signals['bollinger']['lower'].empty else current_price * 0.98
                },
                "support_resistance": signals['levels'],
                "market_regime": signals['market_regime'].iloc[-1] if not signals['market_regime'].empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating technical analysis: {e}")
            return {"error": str(e)}
    
    def _generate_market_outlook(self, 
                               ticker: str, 
                               signal_result: Dict[str, Any], 
                               technical_analysis: Dict[str, Any],
                               fundamentals: Dict[str, Any],
                               sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market outlook"""
        try:
            # Combine all analysis components
            signal = signal_result.get("signal", "HOLD")
            confidence = signal_result.get("confidence", 0.5)
            
            # Technical outlook
            trend_direction = technical_analysis.get("trend_direction", "neutral")
            volatility = technical_analysis.get("volatility_30d", 0)
            
            # Fundamental outlook
            market_cap_rank = fundamentals.get("market_cap_rank", 100)
            price_change_30d = fundamentals.get("price_change_30d", 0)
            
            # Sentiment outlook
            fear_greed_index = sentiment.get("fear_greed_index", 50)
            overall_sentiment = sentiment.get("overall_sentiment", 0.5)
            
            # Generate outlook score
            outlook_score = 0
            
            # Signal contribution
            if signal in ["STRONG_BUY", "BUY"]:
                outlook_score += 2 if signal == "STRONG_BUY" else 1
            elif signal in ["STRONG_SELL", "SELL"]:
                outlook_score -= 2 if signal == "STRONG_SELL" else 1
            
            # Technical contribution
            if trend_direction == "bullish":
                outlook_score += 1
            elif trend_direction == "bearish":
                outlook_score -= 1
            
            # Fundamental contribution
            if price_change_30d > 10:
                outlook_score += 1
            elif price_change_30d < -10:
                outlook_score -= 1
            
            # Sentiment contribution
            if fear_greed_index < 30:  # Fear
                outlook_score += 1  # Contrarian bullish
            elif fear_greed_index > 70:  # Greed
                outlook_score -= 1  # Contrarian bearish
            
            # Determine outlook
            if outlook_score >= 3:
                outlook = "Very Bullish"
                outlook_color = "green"
            elif outlook_score >= 1:
                outlook = "Bullish"
                outlook_color = "lightgreen"
            elif outlook_score <= -3:
                outlook = "Very Bearish"
                outlook_color = "red"
            elif outlook_score <= -1:
                outlook = "Bearish"
                outlook_color = "lightcoral"
            else:
                outlook = "Neutral"
                outlook_color = "gray"
            
            return {
                "outlook": outlook,
                "outlook_color": outlook_color,
                "outlook_score": outlook_score,
                "confidence": confidence,
                "key_factors": {
                    "signal": signal,
                    "trend": trend_direction,
                    "volatility": volatility,
                    "sentiment": overall_sentiment,
                    "fear_greed": fear_greed_index,
                    "fundamentals": price_change_30d
                },
                "recommendations": self._generate_recommendations(signal, outlook_score, volatility)
            }
            
        except Exception as e:
            logger.error(f"Error generating market outlook: {e}")
            return {"outlook": "Unknown", "error": str(e)}
    
    def _generate_recommendations(self, signal: str, outlook_score: int, volatility: float) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        if signal in ["STRONG_BUY", "BUY"]:
            recommendations.append("Consider entering long position")
            if volatility > 50:
                recommendations.append("High volatility - use smaller position size")
        elif signal in ["STRONG_SELL", "SELL"]:
            recommendations.append("Consider entering short position or closing longs")
        else:
            recommendations.append("Wait for clearer signals")
        
        if outlook_score >= 3:
            recommendations.append("Strong bullish momentum - consider adding to positions")
        elif outlook_score <= -3:
            recommendations.append("Strong bearish momentum - consider reducing positions")
        
        if volatility > 60:
            recommendations.append("Extremely high volatility - consider reducing risk")
        elif volatility < 20:
            recommendations.append("Low volatility - potential for breakout")
        
        return recommendations
    
    def execute_crypto_trade(self, 
                           ticker: str, 
                           signal: str, 
                           analysis: Dict[str, Any],
                           custom_position_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute crypto trade based on analysis
        
        Args:
            ticker: Crypto ticker
            signal: Trading signal
            analysis: Market analysis
            custom_position_size: Custom position size (optional)
            
        Returns:
            Trade execution results
        """
        try:
            logger.info(f"Executing crypto trade for {ticker}: {signal}")
            
            # Get current price
            realtime_data = analysis.get("realtime_data", {})
            current_price = realtime_data.get("price", 0)
            
            if current_price <= 0:
                return {"success": False, "error": "Invalid price data"}
            
            # Get portfolio value
            portfolio_summary = self.engine.get_crypto_portfolio_summary()
            portfolio_value = portfolio_summary["total_value"]
            
            # Calculate position size
            if custom_position_size:
                position_size = custom_position_size
            else:
                # Use risk manager to calculate position size
                historical_data = self._get_historical_data_for_risk(ticker)
                stop_loss_price = self._calculate_stop_loss_price(ticker, current_price, signal)
                
                position_info = self.risk_manager.calculate_position_size(
                    ticker=ticker,
                    price=current_price,
                    portfolio_value=portfolio_value,
                    stop_loss_price=stop_loss_price,
                    signal_strength=analysis.get("trading_signal", {}).get("confidence", 0.5),
                    historical_prices=historical_data
                )
                
                position_size = position_info["position_size"]
            
            # Create order
            if signal in ["BUY", "STRONG_BUY"]:
                order_side = CryptoOrderSide.BUY
            elif signal in ["SELL", "STRONG_SELL"]:
                order_side = CryptoOrderSide.SELL
            else:
                return {"success": False, "error": f"Invalid signal for trading: {signal}"}
            
            order = CryptoOrder(
                ticker=ticker,
                side=order_side,
                order_type=CryptoOrderType.MARKET,
                quantity=position_size,
                price=current_price
            )
            
            # Execute order
            success = self.engine.place_crypto_order(order)
            
            if success:
                # Set stop loss and take profit
                self._set_risk_management(ticker, current_price, signal, analysis)
                
                return {
                    "success": True,
                    "ticker": ticker,
                    "signal": signal,
                    "quantity": position_size,
                    "price": current_price,
                    "order_id": len(self.engine.orders),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Order execution failed",
                    "ticker": ticker,
                    "signal": signal
                }
                
        except Exception as e:
            logger.error(f"Error executing crypto trade: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_historical_data_for_risk(self, ticker: str) -> pd.Series:
        """Get historical data for risk calculations"""
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            historical_data = self.data_provider.get_historical_data(
                ticker, start_date, end_date, interval="1d"
            )
            
            return historical_data['close'] if not historical_data.empty else pd.Series()
            
        except Exception as e:
            logger.error(f"Error getting historical data for risk: {e}")
            return pd.Series()
    
    def _calculate_stop_loss_price(self, ticker: str, current_price: float, signal: str) -> float:
        """Calculate stop loss price"""
        try:
            historical_data = self._get_historical_data_for_risk(ticker)
            position_type = "long" if signal in ["BUY", "STRONG_BUY"] else "short"
            
            stop_loss_info = self.risk_manager.calculate_dynamic_stop_loss(
                ticker=ticker,
                entry_price=current_price,
                position_type=position_type,
                historical_prices=historical_data
            )
            
            return stop_loss_info["stop_loss_price"]
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Fallback to 5% stop loss
            if signal in ["BUY", "STRONG_BUY"]:
                return current_price * 0.95
            else:
                return current_price * 1.05
    
    def _set_risk_management(self, ticker: str, entry_price: float, signal: str, analysis: Dict[str, Any]):
        """Set stop loss and take profit for position"""
        try:
            historical_data = self._get_historical_data_for_risk(ticker)
            position_type = "long" if signal in ["BUY", "STRONG_BUY"] else "short"
            
            # Calculate stop loss
            stop_loss_info = self.risk_manager.calculate_dynamic_stop_loss(
                ticker=ticker,
                entry_price=entry_price,
                position_type=position_type,
                historical_prices=historical_data
            )
            
            # Calculate take profit
            take_profit_info = self.risk_manager.calculate_dynamic_take_profit(
                ticker=ticker,
                entry_price=entry_price,
                position_type=position_type,
                historical_prices=historical_data
            )
            
            # Set stop loss and take profit orders
            # Note: This would integrate with the actual order management system
            logger.info(f"Risk management set for {ticker}: SL=${stop_loss_info['stop_loss_price']:.2f}, TP=${take_profit_info['take_profit_price']:.2f}")
            
        except Exception as e:
            logger.error(f"Error setting risk management: {e}")
    
    def _get_fallback_analysis(self, ticker: str) -> Dict[str, Any]:
        """Get fallback analysis when data is unavailable"""
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "realtime_data": {"price": 50000.0 if ticker == "BTC" else 3000.0},
            "fundamentals": {},
            "sentiment": {},
            "trading_signal": {"signal": "HOLD", "confidence": 0.5},
            "technical_analysis": {},
            "market_outlook": {"outlook": "Unknown"},
            "risk_metrics": {},
            "historical_data_points": 0,
            "analysis_quality": "low"
        }
    
    def run_crypto_analysis_cycle(self) -> Dict[str, Any]:
        """
        Run complete crypto analysis cycle for all tickers
        
        Returns:
            Analysis results for all tickers
        """
        results = {}
        
        for ticker in self.tickers:
            try:
                logger.info(f"Running analysis cycle for {ticker}")
                
                # Analyze market
                analysis = self.analyze_crypto_market(ticker)
                
                # Execute trades if signal is strong enough
                signal = analysis.get("trading_signal", {}).get("signal", "HOLD")
                confidence = analysis.get("trading_signal", {}).get("confidence", 0.5)
                
                trade_result = None
                if signal in ["STRONG_BUY", "STRONG_SELL"] and confidence > 0.7:
                    trade_result = self.execute_crypto_trade(ticker, signal, analysis)
                
                results[ticker] = {
                    "analysis": analysis,
                    "trade_result": trade_result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error in analysis cycle for {ticker}: {e}")
                results[ticker] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return results
    
    def get_crypto_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive crypto portfolio summary"""
        portfolio_summary = self.engine.get_crypto_portfolio_summary()
        
        # Add risk metrics
        risk_metrics = self.risk_manager.calculate_portfolio_metrics(
            self.engine.positions, 
            {ticker: 50000.0 if ticker == "BTC" else 3000.0 for ticker in self.tickers}
        )
        
        # Add recommendations
        recommendations = self.risk_manager.get_risk_recommendations(
            self.engine.positions, risk_metrics
        )
        
        portfolio_summary.update({
            "risk_metrics": risk_metrics,
            "recommendations": recommendations,
            "trading_config": {
                "strategy": self.strategy,
                "analysis_frequency": self.analysis_frequency,
                "tickers": self.tickers
            }
        })
        
        return portfolio_summary

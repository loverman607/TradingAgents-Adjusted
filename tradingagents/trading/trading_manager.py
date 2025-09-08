"""
Trading Manager

This module integrates the trading execution engine with the existing analysis pipeline.
It processes agent recommendations and executes trades accordingly.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .execution_engine import TradingExecutionEngine, Order, OrderSide, OrderType, create_order_from_signal

logger = logging.getLogger(__name__)


class TradingManager:
    """Manages trading operations and integrates with the analysis pipeline"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.1,
                 state_file: str = "trading_state.json",
                 enable_short_selling: bool = True,
                 margin_requirement: float = 0.5):
        """
        Initialize the trading manager
        
        Args:
            initial_capital: Starting capital amount
            max_position_size: Maximum position size as percentage of portfolio
            state_file: File to save/load trading state
            enable_short_selling: Whether to allow short selling
            margin_requirement: Margin requirement for short positions
        """
        self.engine = TradingExecutionEngine(
            initial_capital, 
            max_position_size,
            enable_short_selling=enable_short_selling,
            margin_requirement=margin_requirement
        )
        self.state_file = state_file
        self.analysis_history: List[Dict] = []
        
        # Load existing state if available
        if Path(self.state_file).exists():
            try:
                self.engine.load_state(self.state_file)
                logger.info("Loaded existing trading state")
            except Exception as e:
                logger.warning(f"Failed to load trading state: {e}")
    
    def process_analysis_result(self, 
                              ticker: str, 
                              analysis_result: Dict[str, Any],
                              trade_date: str) -> Dict[str, Any]:
        """
        Process analysis result and execute trades
        
        Args:
            ticker: The ticker symbol
            analysis_result: Result from the trading agents analysis
            trade_date: Date of the analysis
            
        Returns:
            Dictionary containing trading actions taken and results
        """
        logger.info(f"Processing analysis result for {ticker} on {trade_date}")
        
        # Extract trading signal
        signal = self._extract_signal(analysis_result)
        logger.info(f"Extracted signal: {signal}")
        
        # Get current price (mock for now)
        current_price = self._get_current_price(ticker)
        
        # Calculate position size
        position_size = self._calculate_position_size(ticker, current_price, analysis_result)
        
        # Create trading actions
        actions = []
        
        if signal == "BUY":
            action = self._execute_buy(ticker, position_size, current_price, analysis_result)
            if action:
                actions.append(action)
        
        elif signal == "SELL":
            action = self._execute_sell(ticker, position_size, current_price, analysis_result)
            if action:
                actions.append(action)
        
        elif signal == "SHORT":
            action = self._execute_short(ticker, position_size, current_price, analysis_result)
            if action:
                actions.append(action)
        
        elif signal == "HOLD":
            # Check if we should set stop loss or take profit
            stop_loss_action = self._check_stop_loss_setting(ticker, analysis_result)
            if stop_loss_action:
                actions.append(stop_loss_action)
        
        # Update prices and check for triggered orders
        self.engine.update_prices({ticker: current_price})
        
        # Save state
        self.engine.save_state(self.state_file)
        
        # Record analysis
        analysis_record = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "trade_date": trade_date,
            "signal": signal,
            "current_price": current_price,
            "position_size": position_size,
            "actions_taken": actions,
            "analysis_result": analysis_result
        }
        self.analysis_history.append(analysis_record)
        
        # Get portfolio summary
        portfolio_summary = self.engine.get_portfolio_summary()
        
        return {
            "signal": signal,
            "actions_taken": actions,
            "portfolio_summary": portfolio_summary,
            "analysis_record": analysis_record
        }
    
    def _extract_signal(self, analysis_result: Dict[str, Any]) -> str:
        """Extract trading signal from analysis result"""
        # Try to get signal from final_trade_decision
        final_decision = analysis_result.get("final_trade_decision", "")
        
        # Look for signal keywords
        final_decision_upper = final_decision.upper()
        
        # Check for short selling signals
        if any(phrase in final_decision_upper for phrase in ["SHORT", "SHORT SELL", "SHORT-SELL"]):
            return "SHORT"
        elif "BUY" in final_decision_upper and "SELL" not in final_decision_upper and "SHORT" not in final_decision_upper:
            return "BUY"
        elif "SELL" in final_decision_upper and "BUY" not in final_decision_upper and "SHORT" not in final_decision_upper:
            return "SELL"
        elif "HOLD" in final_decision_upper:
            return "HOLD"
        
        # Fallback: check trader investment plan
        trader_plan = analysis_result.get("trader_investment_plan", "")
        trader_plan_upper = trader_plan.upper()
        
        if any(phrase in trader_plan_upper for phrase in ["SHORT", "SHORT SELL", "SHORT-SELL"]):
            return "SHORT"
        elif "FINAL TRANSACTION PROPOSAL: **BUY**" in trader_plan_upper:
            return "BUY"
        elif "FINAL TRANSACTION PROPOSAL: **SELL**" in trader_plan_upper:
            return "SELL"
        elif "FINAL TRANSACTION PROPOSAL: **HOLD**" in trader_plan_upper:
            return "HOLD"
        
        # Default to HOLD if no clear signal
        logger.warning("No clear trading signal found, defaulting to HOLD")
        return "HOLD"
    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for ticker (mock implementation)"""
        # In a real implementation, this would fetch from a market data API
        mock_prices = {
            "BTC": 50000.0,
            "ETH": 3000.0,
            "AAPL": 150.0,
            "TSLA": 200.0,
            "SPY": 400.0
        }
        return mock_prices.get(ticker, 100.0)
    
    def _calculate_position_size(self, ticker: str, price: float, analysis_result: Dict[str, Any]) -> float:
        """Calculate position size based on analysis and risk management"""
        # Get current position
        current_position = self.engine.positions.get(ticker)
        current_quantity = current_position.quantity if current_position else 0
        
        # Calculate desired position size based on signal strength
        signal_strength = self._assess_signal_strength(analysis_result)
        
        # Base position size as percentage of portfolio
        portfolio_value = self.engine.get_portfolio_value()
        base_position_value = portfolio_value * self.engine.max_position_size
        
        # Adjust based on signal strength
        adjusted_position_value = base_position_value * signal_strength
        
        # Calculate quantity
        desired_quantity = adjusted_position_value / price if price > 0 else 0
        
        # For BUY signals, return additional quantity to buy
        if self._extract_signal(analysis_result) == "BUY":
            return max(0, desired_quantity - current_quantity)
        
        # For SELL signals, return quantity to sell
        elif self._extract_signal(analysis_result) == "SELL":
            return min(current_quantity, desired_quantity)
        
        return 0
    
    def _assess_signal_strength(self, analysis_result: Dict[str, Any]) -> float:
        """Assess the strength of the trading signal (0.0 to 1.0)"""
        # Analyze the confidence level in the decision
        final_decision = analysis_result.get("final_trade_decision", "")
        
        # Look for confidence indicators
        confidence_indicators = [
            "strong", "high confidence", "very confident", "definitely",
            "clear", "obvious", "certain", "convinced"
        ]
        
        uncertainty_indicators = [
            "uncertain", "unclear", "mixed", "caution", "careful",
            "monitor", "watch", "wait", "hesitant"
        ]
        
        final_decision_lower = final_decision.lower()
        
        confidence_score = 0.5  # Base score
        
        # Increase confidence for positive indicators
        for indicator in confidence_indicators:
            if indicator in final_decision_lower:
                confidence_score += 0.1
        
        # Decrease confidence for uncertainty indicators
        for indicator in uncertainty_indicators:
            if indicator in final_decision_lower:
                confidence_score -= 0.1
        
        # Clamp between 0.1 and 1.0
        return max(0.1, min(1.0, confidence_score))
    
    def _execute_buy(self, ticker: str, quantity: float, price: float, analysis_result: Dict[str, Any]) -> Optional[Dict]:
        """Execute buy order"""
        if quantity <= 0:
            return None
        
        # Create order
        order = Order(
            id=f"buy_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ticker=ticker,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            reason=f"Agent recommendation: BUY - {self._get_reason_summary(analysis_result)}"
        )
        
        # Place order
        success = self.engine.place_order(order)
        
        if success:
            # Set stop loss based on analysis
            stop_loss_price = self._calculate_stop_loss(ticker, price, analysis_result)
            if stop_loss_price:
                self.engine.set_stop_loss(ticker, stop_loss_price, "Risk management")
            
            # Set take profit if mentioned in analysis
            take_profit_price = self._calculate_take_profit(ticker, price, analysis_result)
            if take_profit_price:
                self.engine.set_take_profit(ticker, take_profit_price, "Profit taking")
            
            return {
                "action": "BUY",
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
                "total_value": quantity * price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "success": True
            }
        
        return {
            "action": "BUY",
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "success": False,
            "reason": "Order rejected"
        }
    
    def _execute_sell(self, ticker: str, quantity: float, price: float, analysis_result: Dict[str, Any]) -> Optional[Dict]:
        """Execute sell order"""
        if quantity <= 0:
            return None
        
        # Create order
        order = Order(
            id=f"sell_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            reason=f"Agent recommendation: SELL - {self._get_reason_summary(analysis_result)}"
        )
        
        # Place order
        success = self.engine.place_order(order)
        
        return {
            "action": "SELL",
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "total_value": quantity * price,
            "success": success
        }
    
    def _execute_short(self, ticker: str, quantity: float, price: float, analysis_result: Dict[str, Any]) -> Optional[Dict]:
        """Execute short sell order"""
        if quantity <= 0:
            return None
        
        # Create order
        order = Order(
            id=f"short_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            reason=f"Agent recommendation: SHORT - {self._get_reason_summary(analysis_result)}"
        )
        
        # Place order
        success = self.engine.place_order(order)
        
        if success:
            # Set stop loss for short position (above entry price)
            stop_loss_price = self._calculate_short_stop_loss(ticker, price, analysis_result)
            if stop_loss_price:
                self.engine.set_stop_loss(ticker, stop_loss_price, "Short position risk management")
            
            # Set take profit for short position (below entry price)
            take_profit_price = self._calculate_short_take_profit(ticker, price, analysis_result)
            if take_profit_price:
                self.engine.set_take_profit(ticker, take_profit_price, "Short position profit taking")
            
            return {
                "action": "SHORT",
                "ticker": ticker,
                "quantity": quantity,
                "price": price,
                "total_value": quantity * price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "success": True
            }
        
        return {
            "action": "SHORT",
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "success": False,
            "reason": "Order rejected"
        }
    
    def _check_stop_loss_setting(self, ticker: str, analysis_result: Dict[str, Any]) -> Optional[Dict]:
        """Check if stop loss should be set for HOLD positions"""
        current_position = self.engine.positions.get(ticker)
        if not current_position:
            return None
        
        # Check if stop loss is already set
        if current_position.stop_loss_price:
            return None
        
        # Look for stop loss recommendations in analysis
        final_decision = analysis_result.get("final_trade_decision", "")
        if "stop loss" in final_decision.lower() or "stop-loss" in final_decision.lower():
            current_price = current_position.current_price
            stop_loss_price = self._calculate_stop_loss(ticker, current_price, analysis_result)
            
            if stop_loss_price:
                self.engine.set_stop_loss(ticker, stop_loss_price, "Risk management from analysis")
                return {
                    "action": "SET_STOP_LOSS",
                    "ticker": ticker,
                    "stop_loss_price": stop_loss_price,
                    "current_price": current_price
                }
        
        return None
    
    def _calculate_stop_loss(self, ticker: str, price: float, analysis_result: Dict[str, Any]) -> Optional[float]:
        """Calculate stop loss price based on analysis"""
        # Look for specific stop loss recommendations in the analysis
        final_decision = analysis_result.get("final_trade_decision", "")
        
        # Try to extract stop loss percentage from text
        import re
        
        # Look for patterns like "15-20% below" or "15% stop loss"
        stop_loss_patterns = [
            r'(\d+)[-–](\d+)%\s*(?:below|stop)',
            r'(\d+)%\s*(?:below|stop)',
            r'stop[-\s]?loss[:\s]*(\d+)%'
        ]
        
        for pattern in stop_loss_patterns:
            match = re.search(pattern, final_decision, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    # Range like "15-20%"
                    min_pct = int(match.group(1)) / 100
                    max_pct = int(match.group(2)) / 100
                    avg_pct = (min_pct + max_pct) / 2
                else:
                    # Single percentage
                    avg_pct = int(match.group(1)) / 100
                
                return price * (1 - avg_pct)
        
        # Default stop loss: 10% below current price
        return price * 0.9
    
    def _calculate_take_profit(self, ticker: str, price: float, analysis_result: Dict[str, Any]) -> Optional[float]:
        """Calculate take profit price based on analysis"""
        final_decision = analysis_result.get("final_trade_decision", "")
        
        # Look for take profit recommendations
        import re
        
        take_profit_patterns = [
            r'(\d+)[-–](\d+)%\s*(?:above|target)',
            r'(\d+)%\s*(?:above|target)',
            r'take[-\s]?profit[:\s]*(\d+)%'
        ]
        
        for pattern in take_profit_patterns:
            match = re.search(pattern, final_decision, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    min_pct = int(match.group(1)) / 100
                    max_pct = int(match.group(2)) / 100
                    avg_pct = (min_pct + max_pct) / 2
                else:
                    avg_pct = int(match.group(1)) / 100
                
                return price * (1 + avg_pct)
        
        # Default take profit: 20% above current price
        return price * 1.2
    
    def _calculate_short_stop_loss(self, ticker: str, price: float, analysis_result: Dict[str, Any]) -> Optional[float]:
        """Calculate stop loss price for short position (above entry price)"""
        final_decision = analysis_result.get("final_trade_decision", "")
        
        # Look for stop loss recommendations in the analysis
        import re
        
        # Look for patterns like "15-20% above" or "15% stop loss"
        stop_loss_patterns = [
            r'(\d+)[-–](\d+)%\s*(?:above|stop)',
            r'(\d+)%\s*(?:above|stop)',
            r'stop[-\s]?loss[:\s]*(\d+)%'
        ]
        
        for pattern in stop_loss_patterns:
            match = re.search(pattern, final_decision, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    # Range like "15-20%"
                    min_pct = int(match.group(1)) / 100
                    max_pct = int(match.group(2)) / 100
                    avg_pct = (min_pct + max_pct) / 2
                else:
                    # Single percentage
                    avg_pct = int(match.group(1)) / 100
                
                return price * (1 + avg_pct)  # Above price for short position
        
        # Default stop loss: 10% above current price for short position
        return price * 1.1
    
    def _calculate_short_take_profit(self, ticker: str, price: float, analysis_result: Dict[str, Any]) -> Optional[float]:
        """Calculate take profit price for short position (below entry price)"""
        final_decision = analysis_result.get("final_trade_decision", "")
        
        # Look for take profit recommendations
        import re
        
        take_profit_patterns = [
            r'(\d+)[-–](\d+)%\s*(?:below|target)',
            r'(\d+)%\s*(?:below|target)',
            r'take[-\s]?profit[:\s]*(\d+)%'
        ]
        
        for pattern in take_profit_patterns:
            match = re.search(pattern, final_decision, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    min_pct = int(match.group(1)) / 100
                    max_pct = int(match.group(2)) / 100
                    avg_pct = (min_pct + max_pct) / 2
                else:
                    avg_pct = int(match.group(1)) / 100
                
                return price * (1 - avg_pct)  # Below price for short position
        
        # Default take profit: 15% below current price for short position
        return price * 0.85
    
    def _get_reason_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Get a brief summary of the reasoning for the trade"""
        final_decision = analysis_result.get("final_trade_decision", "")
        
        # Extract key points (first 100 characters)
        if len(final_decision) > 100:
            return final_decision[:100] + "..."
        return final_decision
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        return self.engine.get_portfolio_summary()
    
    def get_trading_history(self) -> List[Dict]:
        """Get trading history"""
        return self.engine.trade_history
    
    def get_analysis_history(self) -> List[Dict]:
        """Get analysis history"""
        return self.analysis_history
    
    def save_analysis_report(self, filepath: str):
        """Save analysis and trading history to file"""
        report = {
            "portfolio_summary": self.get_portfolio_summary(),
            "trading_history": self.get_trading_history(),
            "analysis_history": self.get_analysis_history(),
            "generated_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {filepath}")
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.engine = TradingExecutionEngine(
            self.engine.initial_capital, 
            self.engine.max_position_size
        )
        self.analysis_history = []
        logger.info("Portfolio reset to initial state")

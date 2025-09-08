"""
Trading Execution Engine

This module provides the core trading functionality including buy, sell, and stop loss orders.
It integrates with the existing analysis pipeline to execute trades based on agent recommendations.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of trading orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Represents a trading order"""
    id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    reason: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Position:
    """Represents a trading position"""
    ticker: str
    quantity: float
    average_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss_reason: str = ""
    take_profit_reason: str = ""
    created_at: datetime = None
    last_updated: datetime = None
    position_type: PositionType = PositionType.LONG
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()
        
        if self.quantity > 0:
            self.position_type = PositionType.LONG
        elif self.quantity < 0:
            self.position_type = PositionType.SHORT
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.average_price
    
    @property
    def absolute_quantity(self) -> float:
        return abs(self.quantity)
    
    def update_price(self, new_price: float):
        self.current_price = new_price
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity
        else:
            self.unrealized_pnl = (self.average_price - self.current_price) * abs(self.quantity)
        self.last_updated = datetime.now()
    
    def update_market_value(self):
        """Update market value and unrealized P&L"""
        self.unrealized_pnl = self.calculate_unrealized_pnl()
        self.last_updated = datetime.now()
    
    def calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.average_price) * self.quantity
        else:  # short
            return (self.average_price - self.current_price) * abs(self.quantity)


class TradingExecutionEngine:
    """Main trading execution engine"""
    
    def __init__(self, initial_capital: float = 100000.0, max_position_size: float = 0.1, 
                 enable_short_selling: bool = True, margin_requirement: float = 0.5):
        self.initial_capital = initial_capital
        self.cash = initial_capital  # Available cash
        self.available_capital = initial_capital
        self.max_position_size = max_position_size
        self.enable_short_selling = enable_short_selling
        self.margin_requirement = margin_requirement
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_history: List[Order] = []
        self.trade_history: List[Dict] = []
        
        self.max_daily_loss = 0.05
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        self.short_borrowed_shares: Dict[str, float] = {}
        self.margin_used: float = 0.0
        
        logger.info(f"Trading engine initialized with ${initial_capital:,.2f} capital")
        if enable_short_selling:
            logger.info(f"Short selling enabled with {margin_requirement:.0%} margin requirement")

    # [ALL methods unchanged … skipping for brevity until save_state]

    def save_state(self, filepath: str):
        """Save current state to file"""
        state = {
            "initial_capital": self.initial_capital,
            "available_capital": self.available_capital,
            "max_position_size": self.max_position_size,
            "positions": {ticker: asdict(pos) for ticker, pos in self.positions.items()},
            "orders": [asdict(order) for order in self.orders],
            "order_history": [asdict(order) for order in self.order_history],
            "trade_history": self.trade_history,
            "daily_pnl": self.daily_pnl,
            "last_reset_date": self.last_reset_date.isoformat()
        }

        # Always resolve path relative to this file’s directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(base_dir, filepath)

        dir_path = os.path.dirname(abs_path)
        os.makedirs(dir_path, exist_ok=True)

        logger.info(f"[DEBUG] Saving state. Filepath: {abs_path}, Dir: {dir_path}, Exists? {os.path.exists(dir_path)}")

        with open(abs_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Trading state saved to {abs_path}")
    
    def load_state(self, filepath: str):
        """Load state from file"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(base_dir, filepath)

        if not os.path.exists(abs_path):
            logger.warning(f"State file {abs_path} does not exist. Starting fresh.")
            return

        with open(abs_path, 'r') as f:
            state = json.load(f)
        
        self.initial_capital = state["initial_capital"]
        self.available_capital = state["available_capital"]
        self.max_position_size = state["max_position_size"]
        
        self.positions = {}
        for ticker, pos_data in state["positions"].items():
            pos_data["created_at"] = datetime.fromisoformat(pos_data["created_at"])
            pos_data["last_updated"] = datetime.fromisoformat(pos_data["last_updated"])
            self.positions[ticker] = Position(**pos_data)
        
        self.orders = []
        for order_data in state["orders"]:
            order_data["side"] = OrderSide(order_data["side"])
            order_data["order_type"] = OrderType(order_data["order_type"])
            order_data["status"] = OrderStatus(order_data["status"])
            order_data["created_at"] = datetime.fromisoformat(order_data["created_at"])
            if order_data["filled_at"]:
                order_data["filled_at"] = datetime.fromisoformat(order_data["filled_at"])
            self.orders.append(Order(**order_data))
        
        self.order_history = []
        for order_data in state["order_history"]:
            order_data["side"] = OrderSide(order_data["side"])
            order_data["order_type"] = OrderType(order_data["order_type"])
            order_data["status"] = OrderStatus(order_data["status"])
            order_data["created_at"] = datetime.fromisoformat(order_data["created_at"])
            if order_data["filled_at"]:
                order_data["filled_at"] = datetime.fromisoformat(order_data["filled_at"])
            self.order_history.append(Order(**order_data))
        
        self.trade_history = state["trade_history"]
        self.daily_pnl = state["daily_pnl"]
        self.last_reset_date = datetime.fromisoformat(state["last_reset_date"]).date()
        
        logger.info(f"Trading state loaded from {abs_path}")

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value including cash and positions"""
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position.market_value
        
        return total_value

    def get_portfolio_summary(self) -> dict:
        """Get comprehensive portfolio summary"""
        total_value = self.get_portfolio_value()
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            "total_value": total_value,
            "cash": self.cash,
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0,
            "positions_count": len(self.positions),
            "daily_pnl": self.daily_pnl,
            "max_position_size": self.max_position_size
        }

    def update_prices(self, prices: dict) -> None:
        """Update current prices for all positions"""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
                self.positions[ticker].update_market_value()

    def set_stop_loss(self, ticker: str, stop_price: float, reason: str = "") -> bool:
        """Set stop loss for a position"""
        if ticker not in self.positions:
            logger.warning(f"Cannot set stop loss: No position found for {ticker}")
            return False
        
        position = self.positions[ticker]
        position.stop_loss = stop_price
        position.stop_loss_reason = reason
        
        logger.info(f"Stop loss set for {ticker} at ${stop_price:.2f}: {reason}")
        return True

    def set_take_profit(self, ticker: str, take_profit_price: float, reason: str = "") -> bool:
        """Set take profit for a position"""
        if ticker not in self.positions:
            logger.warning(f"Cannot set take profit: No position found for {ticker}")
            return False
        
        position = self.positions[ticker]
        position.take_profit = take_profit_price
        position.take_profit_reason = reason
        
        logger.info(f"Take profit set for {ticker} at ${take_profit_price:.2f}: {reason}")
        return True

    def check_stop_losses(self) -> None:
        """Check and execute stop loss orders"""
        for ticker, position in self.positions.items():
            if position.stop_loss and position.current_price <= position.stop_loss:
                logger.info(f"Stop loss triggered for {ticker} at ${position.current_price:.2f}")
                # Execute stop loss sell order
                self._execute_sell(ticker, position.quantity, position.current_price, 
                                 f"Stop loss triggered at ${position.stop_loss:.2f}")

    def check_take_profits(self) -> None:
        """Check and execute take profit orders"""
        for ticker, position in self.positions.items():
            if position.take_profit and position.current_price >= position.take_profit:
                logger.info(f"Take profit triggered for {ticker} at ${position.current_price:.2f}")
                # Execute take profit sell order
                self._execute_sell(ticker, position.quantity, position.current_price, 
                                 f"Take profit triggered at ${position.take_profit:.2f}")

    def _execute_sell(self, ticker: str, quantity: float, price: float, reason: str = "") -> bool:
        """Execute a sell order"""
        if ticker not in self.positions:
            logger.warning(f"Cannot sell: No position found for {ticker}")
            return False
        
        position = self.positions[ticker]
        
        # Check if we have enough quantity to sell
        if abs(quantity) > abs(position.quantity):
            logger.warning(f"Cannot sell {abs(quantity)} shares: Only {abs(position.quantity)} available")
            return False
        
        # Calculate proceeds
        proceeds = abs(quantity) * price
        
        # Update position
        if position.position_type == PositionType.LONG:
            # Selling long position
            position.quantity -= quantity
            if position.quantity == 0:
                # Position closed
                realized_pnl = (price - position.average_price) * abs(quantity)
                self.cash += proceeds + realized_pnl
                del self.positions[ticker]
                logger.info(f"Long position closed for {ticker}: P&L = ${realized_pnl:.2f}")
            else:
                # Partial sell
                realized_pnl = (price - position.average_price) * abs(quantity)
                self.cash += proceeds + realized_pnl
                position.update_price(price)
                logger.info(f"Partial sell for {ticker}: P&L = ${realized_pnl:.2f}")
        else:
            # Covering short position
            position.quantity += quantity
            if position.quantity == 0:
                # Position closed
                realized_pnl = (position.average_price - price) * abs(quantity)
                self.cash += proceeds + realized_pnl
                del self.positions[ticker]
                logger.info(f"Short position closed for {ticker}: P&L = ${realized_pnl:.2f}")
            else:
                # Partial cover
                realized_pnl = (position.average_price - price) * abs(quantity)
                self.cash += proceeds + realized_pnl
                position.update_price(price)
                logger.info(f"Partial cover for {ticker}: P&L = ${realized_pnl:.2f}")
        
        # Record trade
        trade = {
            "ticker": ticker,
            "side": "SELL",
            "quantity": abs(quantity),
            "price": price,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }
        self.trade_history.append(trade)
        
        return True


def create_order_from_signal(ticker: str, signal: str, quantity: float, 
                           current_price: float, reason: str = "") -> Optional[Order]:
    """Create an order from a trading signal"""
    signal = signal.upper().strip()
    
    if signal == "BUY":
        return Order(
            id=str(uuid.uuid4()),
            ticker=ticker,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            reason=reason
        )
    elif signal == "SELL":
        return Order(
            id=str(uuid.uuid4()),
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            reason=reason
        )
    elif signal == "HOLD":
        return None
    
    return None

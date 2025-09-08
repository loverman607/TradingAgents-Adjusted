"""
Crypto-Optimized Trading Engine

This module provides a specialized trading engine optimized for cryptocurrency trading,
with 24/7 support, volatility-based position sizing, and crypto-specific risk management.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CryptoOrderSide(Enum):
    """Crypto order sides"""
    BUY = "BUY"
    SELL = "SELL"
    LONG = "LONG"
    SHORT = "SHORT"


class CryptoOrderType(Enum):
    """Crypto order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


@dataclass
class CryptoOrder:
    """Crypto trading order"""
    ticker: str
    side: CryptoOrderSide
    order_type: CryptoOrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Canceled
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class CryptoPosition:
    """Crypto trading position"""
    ticker: str
    quantity: float
    avg_price: float
    position_type: str = "long"  # long, short
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    leverage: float = 1.0
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        
        # Set position type based on quantity
        if self.quantity > 0:
            self.position_type = "long"
        elif self.quantity < 0:
            self.position_type = "short"
            self.quantity = abs(self.quantity)  # Store as positive for short positions


class CryptoTradingEngine:
    """
    Crypto-optimized trading engine with 24/7 support and volatility-based risk management
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.2,  # 20% max position for crypto
                 max_daily_loss: float = 0.05,    # 5% max daily loss
                 volatility_threshold: float = 0.05,  # 5% volatility threshold
                 enable_24_7: bool = True,
                 enable_leverage: bool = True,
                 max_leverage: float = 3.0,
                 state_file: str = "crypto_trading_state.json"):
        """
        Initialize crypto trading engine
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as percentage of portfolio
            max_daily_loss: Maximum daily loss as percentage of capital
            volatility_threshold: Volatility threshold for position sizing
            enable_24_7: Enable 24/7 trading (crypto markets)
            enable_leverage: Enable leveraged trading
            max_leverage: Maximum leverage allowed
            state_file: File to save/load state
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.volatility_threshold = volatility_threshold
        self.enable_24_7 = enable_24_7
        self.enable_leverage = enable_leverage
        self.max_leverage = max_leverage
        self.state_file = state_file
        
        # Trading state
        self.cash = initial_capital
        self.positions: Dict[str, CryptoPosition] = {}
        self.orders: List[CryptoOrder] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Risk management
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.volatility_cache: Dict[str, float] = {}
        
        # Crypto-specific settings
        self.crypto_assets = ["BTC", "ETH", "BNB", "ADA", "SOL", "DOT", "MATIC", "AVAX"]
        self.crypto_volatility = {
            "BTC": 0.03,  # 3% daily volatility
            "ETH": 0.04,  # 4% daily volatility
            "BNB": 0.05,  # 5% daily volatility
            "ADA": 0.06,  # 6% daily volatility
            "SOL": 0.07,  # 7% daily volatility
            "DOT": 0.06,  # 6% daily volatility
            "MATIC": 0.08,  # 8% daily volatility
            "AVAX": 0.08   # 8% daily volatility
        }
        
        # Load existing state
        self.load_state()
        
        logger.info(f"Crypto trading engine initialized with ${initial_capital:,.2f} capital")
    
    def reset_daily_limits(self):
        """Reset daily limits if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            logger.info("Daily limits reset")
    
    def get_volatility(self, ticker: str, lookback_days: int = 30) -> float:
        """Get historical volatility for position sizing"""
        if ticker in self.volatility_cache:
            return self.volatility_cache[ticker]
        
        # Use default volatility if not available
        default_vol = self.crypto_volatility.get(ticker, 0.05)
        self.volatility_cache[ticker] = default_vol
        return default_vol
    
    def calculate_crypto_position_size(self, ticker: str, price: float, 
                                     signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on crypto volatility and signal strength
        
        Args:
            ticker: Crypto ticker symbol
            price: Current price
            signal_strength: Signal strength (0.0 to 1.0)
            
        Returns:
            Position size in units
        """
        # Get volatility
        volatility = self.get_volatility(ticker)
        
        # Base position size from max position size
        base_size = self.max_position_size * self.cash / price
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = min(1.0, self.volatility_threshold / volatility)
        
        # Adjust for signal strength
        signal_adjustment = signal_strength
        
        # Calculate final position size
        position_size = base_size * volatility_adjustment * signal_adjustment
        
        # Ensure minimum and maximum limits
        min_size = 0.001  # Minimum 0.1% of portfolio
        max_size = self.max_position_size * self.cash / price
        
        position_size = max(min_size, min(position_size, max_size))
        
        logger.info(f"Position size for {ticker}: {position_size:.6f} (vol: {volatility:.3f}, signal: {signal_strength:.3f})")
        
        return position_size
    
    def can_place_crypto_order(self, order: CryptoOrder) -> Tuple[bool, str]:
        """Check if a crypto order can be placed"""
        self.reset_daily_limits()
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * self.initial_capital:
            return False, "Daily loss limit exceeded"
        
        # Check if ticker is supported
        if order.ticker not in self.crypto_assets:
            return False, f"Unsupported crypto asset: {order.ticker}"
        
        # Check capital requirements
        if order.side in [CryptoOrderSide.BUY, CryptoOrderSide.LONG]:
            required_capital = order.quantity * (order.price or 0)
            if required_capital > self.cash:
                return False, f"Insufficient capital. Required: ${required_capital:,.2f}, Available: ${self.cash:,.2f}"
        
        # Check position limits
        if order.side in [CryptoOrderSide.SELL, CryptoOrderSide.SHORT]:
            if order.ticker not in self.positions:
                return False, f"No position to sell for {order.ticker}"
            
            position = self.positions[order.ticker]
            if order.quantity > position.quantity:
                return False, f"Insufficient position. Available: {position.quantity:.6f}, Requested: {order.quantity:.6f}"
        
        return True, "Order can be placed"
    
    def place_crypto_order(self, order: CryptoOrder) -> bool:
        """Place a crypto order"""
        can_place, reason = self.can_place_crypto_order(order)
        
        if not can_place:
            logger.warning(f"Cannot place order: {reason}")
            order.status = "REJECTED"
            return False
        
        # Add order to list
        self.orders.append(order)
        logger.info(f"Placed {order.side.value} order for {order.quantity:.6f} {order.ticker} @ ${order.price:.2f}")
        
        # For market orders, execute immediately
        if order.order_type == CryptoOrderType.MARKET:
            return self.execute_crypto_order(order)
        
        return True
    
    def execute_crypto_order(self, order: CryptoOrder) -> bool:
        """Execute a crypto order"""
        try:
            if order.side in [CryptoOrderSide.BUY, CryptoOrderSide.LONG]:
                return self._execute_crypto_buy(order)
            elif order.side in [CryptoOrderSide.SELL, CryptoOrderSide.SHORT]:
                return self._execute_crypto_sell(order)
            else:
                logger.error(f"Unknown order side: {order.side}")
                return False
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            order.status = "REJECTED"
            return False
    
    def _execute_crypto_buy(self, order: CryptoOrder) -> bool:
        """Execute crypto buy order"""
        cost = order.quantity * order.price
        commission = cost * 0.001  # 0.1% commission
        total_cost = cost + commission
        
        # Update cash
        self.cash -= total_cost
        
        # Update or create position
        if order.ticker in self.positions:
            position = self.positions[order.ticker]
            # Average down/up
            old_quantity = position.quantity
            old_avg_price = position.avg_price
            new_quantity = old_quantity + order.quantity
            new_avg_price = (old_quantity * old_avg_price + cost) / new_quantity
            
            position.quantity = new_quantity
            position.avg_price = new_avg_price
            position.updated_at = datetime.now()
        else:
            self.positions[order.ticker] = CryptoPosition(
                ticker=order.ticker,
                quantity=order.quantity,
                avg_price=order.price,
                position_type="long"
            )
        
        # Update order status
        order.status = "FILLED"
        order.filled_at = datetime.now()
        order.filled_quantity = order.quantity
        order.filled_price = order.price
        order.commission = commission
        
        # Record trade
        self.trades.append({
            "timestamp": datetime.now(),
            "ticker": order.ticker,
            "side": "BUY",
            "quantity": order.quantity,
            "price": order.price,
            "value": cost,
            "commission": commission
        })
        
        logger.info(f"Executed BUY: {order.quantity:.6f} {order.ticker} @ ${order.price:.2f} (${cost:,.2f})")
        return True
    
    def _execute_crypto_sell(self, order: CryptoOrder) -> bool:
        """Execute crypto sell order"""
        if order.ticker not in self.positions:
            logger.error(f"No position to sell for {order.ticker}")
            return False
        
        position = self.positions[order.ticker]
        proceeds = order.quantity * order.price
        commission = proceeds * 0.001  # 0.1% commission
        net_proceeds = proceeds - commission
        
        # Update cash
        self.cash += net_proceeds
        
        # Calculate P&L
        cost_basis = order.quantity * position.avg_price
        pnl = proceeds - cost_basis - commission
        
        # Update position
        position.quantity -= order.quantity
        position.realized_pnl += pnl
        position.updated_at = datetime.now()
        
        # Remove position if quantity is zero
        if position.quantity <= 0.001:  # Small threshold for floating point
            del self.positions[order.ticker]
        
        # Update order status
        order.status = "FILLED"
        order.filled_at = datetime.now()
        order.filled_quantity = order.quantity
        order.filled_price = order.price
        order.commission = commission
        
        # Record trade
        self.trades.append({
            "timestamp": datetime.now(),
            "ticker": order.ticker,
            "side": "SELL",
            "quantity": order.quantity,
            "price": order.price,
            "value": proceeds,
            "commission": commission,
            "pnl": pnl
        })
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        logger.info(f"Executed SELL: {order.quantity:.6f} {order.ticker} @ ${order.price:.2f} (${proceeds:,.2f}, P&L: ${pnl:,.2f})")
        return True
    
    def update_crypto_prices(self, prices: Dict[str, float]):
        """Update crypto prices and calculate unrealized P&L"""
        for ticker, price in prices.items():
            if ticker in self.positions:
                position = self.positions[ticker]
                if position.position_type == "long":
                    position.unrealized_pnl = (price - position.avg_price) * position.quantity
                else:  # short
                    position.unrealized_pnl = (position.avg_price - price) * position.quantity
                position.updated_at = datetime.now()
    
    def get_crypto_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive crypto portfolio summary"""
        total_value = self.cash
        total_unrealized_pnl = 0.0
        total_realized_pnl = 0.0
        
        for position in self.positions.values():
            # Calculate position value (simplified - would need current prices)
            total_unrealized_pnl += position.unrealized_pnl
            total_realized_pnl += position.realized_pnl
        
        return {
            "cash": self.cash,
            "total_value": total_value + total_unrealized_pnl,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_realized_pnl": total_realized_pnl,
            "daily_pnl": self.daily_pnl,
            "positions": {ticker: asdict(pos) for ticker, pos in self.positions.items()},
            "total_trades": len(self.trades),
            "active_orders": len([o for o in self.orders if o.status == "PENDING"]),
            "crypto_assets": self.crypto_assets,
            "volatility_cache": self.volatility_cache
        }
    
    def save_state(self):
        """Save trading state to file"""
        state = {
            "cash": self.cash,
            "positions": {ticker: asdict(pos) for ticker, pos in self.positions.items()},
            "orders": [asdict(order) for order in self.orders],
            "trades": self.trades,
            "daily_pnl": self.daily_pnl,
            "last_reset_date": self.last_reset_date.isoformat(),
            "volatility_cache": self.volatility_cache
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Trading state saved to {self.state_file}")
    
    def load_state(self):
        """Load trading state from file"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.cash = state.get("cash", self.initial_capital)
            self.daily_pnl = state.get("daily_pnl", 0.0)
            self.last_reset_date = datetime.fromisoformat(state.get("last_reset_date", datetime.now().date().isoformat()))
            self.volatility_cache = state.get("volatility_cache", {})
            
            # Load positions
            self.positions = {}
            for ticker, pos_data in state.get("positions", {}).items():
                self.positions[ticker] = CryptoPosition(**pos_data)
            
            # Load trades
            self.trades = state.get("trades", [])
            
            logger.info(f"Trading state loaded from {self.state_file}")
            
        except FileNotFoundError:
            logger.info("No existing state file found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load trading state: {e}")
    
    def get_crypto_performance_metrics(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate crypto-specific performance metrics"""
        self.update_crypto_prices(current_prices)
        
        total_value = self.cash
        total_unrealized_pnl = 0.0
        total_realized_pnl = 0.0
        
        for position in self.positions.values():
            total_unrealized_pnl += position.unrealized_pnl
            total_realized_pnl += position.realized_pnl
        
        total_value += total_unrealized_pnl
        
        # Calculate returns
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        # Calculate crypto-specific metrics
        btc_exposure = 0.0
        eth_exposure = 0.0
        
        if "BTC" in self.positions:
            btc_exposure = self.positions["BTC"].quantity * current_prices.get("BTC", 0) / total_value
        
        if "ETH" in self.positions:
            eth_exposure = self.positions["ETH"].quantity * current_prices.get("ETH", 0) / total_value
        
        return {
            "total_return": total_return,
            "total_value": total_value,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "daily_pnl": self.daily_pnl,
            "btc_exposure": btc_exposure,
            "eth_exposure": eth_exposure,
            "total_exposure": btc_exposure + eth_exposure,
            "cash_ratio": self.cash / total_value,
            "volatility_adjusted_return": total_return / np.mean(list(self.volatility_cache.values())) if self.volatility_cache else total_return
        }

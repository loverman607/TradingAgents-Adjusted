"""
Crypto-Specific Backtesting Framework

This module provides backtesting capabilities specifically optimized for cryptocurrency trading,
with 24/7 support, volatility-based testing, and crypto-specific metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from pathlib import Path

from .crypto_engine import CryptoTradingEngine
from .crypto_strategies import CryptoStrategies
from .crypto_indicators import CryptoIndicators
from .crypto_risk import CryptoRiskManager

logger = logging.getLogger(__name__)


class CryptoBacktestEngine:
    """Crypto-optimized backtesting engine"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.2,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0005,  # 0.05% slippage
                 enable_24_7: bool = True):
        """
        Initialize crypto backtest engine
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size
            commission: Commission rate per trade
            slippage: Slippage rate
            enable_24_7: Enable 24/7 trading
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission = commission
        self.slippage = slippage
        self.enable_24_7 = enable_24_7
        
        # Initialize components
        self.engine = CryptoTradingEngine(
            initial_capital=initial_capital,
            max_position_size=max_position_size,
            enable_24_7=enable_24_7
        )
        
        self.strategies = CryptoStrategies()
        self.indicators = CryptoIndicators()
        self.risk_manager = CryptoRiskManager()
        
        # Backtest results
        self.results = {}
        self.trades = []
        self.portfolio_values = []
        self.drawdowns = []
        
    def run_crypto_backtest(self, 
                          historical_data: Dict[str, pd.DataFrame],
                          strategy: str = "adaptive",
                          start_date: str = None,
                          end_date: str = None) -> Dict[str, Any]:
        """
        Run crypto backtest
        
        Args:
            historical_data: Dict of ticker -> OHLCV DataFrame
            strategy: Trading strategy to use
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting crypto backtest with {strategy} strategy")
        
        # Prepare data
        if start_date:
            for ticker in historical_data:
                historical_data[ticker] = historical_data[ticker][historical_data[ticker].index >= start_date]
        
        if end_date:
            for ticker in historical_data:
                historical_data[ticker] = historical_data[ticker][historical_data[ticker].index <= end_date]
        
        # Get analysis dates (hourly for crypto)
        all_dates = set()
        for ticker, data in historical_data.items():
            all_dates.update(data.index)
        
        analysis_dates = sorted(list(all_dates))
        
        logger.info(f"Running backtest on {len(analysis_dates)} data points")
        
        # Run backtest
        for i, date in enumerate(analysis_dates):
            if i % 100 == 0:
                logger.info(f"Processing date {i+1}/{len(analysis_dates)}: {date}")
            
            # Update prices
            current_prices = {}
            for ticker, data in historical_data.items():
                if date in data.index:
                    current_prices[ticker] = data.loc[date, 'close']
            
            self.engine.update_crypto_prices(current_prices)
            
            # Analyze each ticker
            for ticker, data in historical_data.items():
                if date not in data.index:
                    continue
                
                # Get data up to current date
                data_up_to_date = data.loc[:date]
                
                if len(data_up_to_date) < 24:  # Need at least 24 hours of data
                    continue
                
                # Generate trading signal
                try:
                    signal_result = self.strategies.get_strategy_signal(
                        data_up_to_date, ticker, strategy
                    )
                    
                    signal = signal_result.get("signal", "HOLD")
                    confidence = signal_result.get("confidence", 0.5)
                    
                    # Execute trade if signal is strong enough
                    if signal in ["STRONG_BUY", "STRONG_SELL"] and confidence > 0.6:
                        self._execute_backtest_trade(ticker, signal, current_prices[ticker], date)
                    elif signal in ["BUY", "SELL"] and confidence > 0.7:
                        self._execute_backtest_trade(ticker, signal, current_prices[ticker], date)
                        
                except Exception as e:
                    logger.warning(f"Error analyzing {ticker} on {date}: {e}")
                    continue
            
            # Record portfolio value
            portfolio_value = self.engine.get_crypto_portfolio_summary()["total_value"]
            self.portfolio_values.append({
                "date": date,
                "value": portfolio_value
            })
            
            # Calculate drawdown
            if len(self.portfolio_values) > 1:
                peak_value = max(pv["value"] for pv in self.portfolio_values)
                current_drawdown = (peak_value - portfolio_value) / peak_value
                self.drawdowns.append({
                    "date": date,
                    "drawdown": current_drawdown
                })
        
        # Calculate final results
        results = self._calculate_backtest_results(historical_data)
        
        logger.info("Crypto backtest completed")
        return results
    
    def _execute_backtest_trade(self, ticker: str, signal: str, price: float, date: datetime):
        """Execute trade in backtest"""
        try:
            # Calculate position size
            portfolio_summary = self.engine.get_crypto_portfolio_summary()
            portfolio_value = portfolio_summary["total_value"]
            
            # Use risk manager for position sizing
            position_info = self.risk_manager.calculate_position_size(
                ticker=ticker,
                price=price,
                portfolio_value=portfolio_value,
                stop_loss_price=price * 0.95 if signal in ["BUY", "STRONG_BUY"] else price * 1.05,
                signal_strength=0.7
            )
            
            position_size = position_info["position_size"]
            
            # Apply slippage
            if signal in ["BUY", "STRONG_BUY"]:
                execution_price = price * (1 + self.slippage)
                order_side = "BUY"
            else:
                execution_price = price * (1 - self.slippage)
                order_side = "SELL"
            
            # Create and execute order
            from .crypto_engine import CryptoOrder, CryptoOrderSide, CryptoOrderType
            
            order = CryptoOrder(
                ticker=ticker,
                side=CryptoOrderSide.BUY if order_side == "BUY" else CryptoOrderSide.SELL,
                order_type=CryptoOrderType.MARKET,
                quantity=position_size,
                price=execution_price
            )
            
            success = self.engine.place_crypto_order(order)
            
            if success:
                # Record trade
                self.trades.append({
                    "date": date,
                    "ticker": ticker,
                    "side": order_side,
                    "quantity": position_size,
                    "price": execution_price,
                    "value": position_size * execution_price,
                    "commission": position_size * execution_price * self.commission
                })
                
        except Exception as e:
            logger.warning(f"Error executing backtest trade: {e}")
    
    def _calculate_backtest_results(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        if not self.portfolio_values:
            return {"error": "No portfolio values recorded"}
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = self.portfolio_values[-1]["value"]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.portfolio_values)):
            prev_value = self.portfolio_values[i-1]["value"]
            curr_value = self.portfolio_values[i]["value"]
            daily_return = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_return)
        
        # Risk metrics
        if daily_returns:
            volatility = np.std(daily_returns) * np.sqrt(365)  # Annualized
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
            max_drawdown = max(dd["drawdown"] for dd in self.drawdowns) if self.drawdowns else 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade analysis
        total_trades = len(self.trades)
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        
        # Analyze trades
        trade_analysis = {}
        for ticker in historical_data.keys():
            ticker_trades = [t for t in self.trades if t["ticker"] == ticker]
            trade_analysis[ticker] = {
                "total_trades": len(ticker_trades),
                "buy_trades": len([t for t in ticker_trades if t["side"] == "BUY"]),
                "sell_trades": len([t for t in ticker_trades if t["side"] == "SELL"])
            }
        
        # Crypto-specific metrics
        crypto_metrics = self._calculate_crypto_metrics(historical_data)
        
        # Performance comparison
        performance_comparison = self._calculate_performance_comparison(historical_data)
        
        return {
            "backtest_summary": {
                "initial_capital": initial_value,
                "final_value": final_value,
                "total_return": total_return,
                "total_trades": total_trades,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0
            },
            "crypto_metrics": crypto_metrics,
            "trade_analysis": trade_analysis,
            "performance_comparison": performance_comparison,
            "portfolio_values": self.portfolio_values,
            "trades": self.trades,
            "drawdowns": self.drawdowns
        }
    
    def _calculate_crypto_metrics(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate crypto-specific metrics"""
        metrics = {}
        
        for ticker, data in historical_data.items():
            if data.empty:
                continue
            
            # Price metrics
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            price_return = (end_price - start_price) / start_price
            
            # Volatility metrics
            returns = data['close'].pct_change().dropna()
            volatility_7d = returns.rolling(7).std().iloc[-1] * np.sqrt(365) if len(returns) >= 7 else 0
            volatility_30d = returns.rolling(30).std().iloc[-1] * np.sqrt(365) if len(returns) >= 30 else 0
            
            # Volume metrics
            avg_volume = data['volume'].mean()
            volume_volatility = data['volume'].std() / avg_volume if avg_volume > 0 else 0
            
            # Technical metrics
            sma_20 = data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else end_price
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else end_price
            
            metrics[ticker] = {
                "price_return": price_return,
                "volatility_7d": volatility_7d,
                "volatility_30d": volatility_30d,
                "avg_volume": avg_volume,
                "volume_volatility": volume_volatility,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "trend_strength": (end_price - sma_20) / sma_20 if sma_20 > 0 else 0
            }
        
        return metrics
    
    def _calculate_performance_comparison(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate performance comparison with buy-and-hold"""
        comparison = {}
        
        for ticker, data in historical_data.items():
            if data.empty:
                continue
            
            # Buy and hold performance
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            buy_hold_return = (end_price - start_price) / start_price
            
            # Strategy performance (simplified)
            strategy_return = (self.portfolio_values[-1]["value"] - self.initial_capital) / self.initial_capital if self.portfolio_values else 0
            
            comparison[ticker] = {
                "buy_hold_return": buy_hold_return,
                "strategy_return": strategy_return,
                "outperformance": strategy_return - buy_hold_return,
                "relative_performance": strategy_return / buy_hold_return if buy_hold_return != 0 else 0
            }
        
        return comparison
    
    def generate_crypto_backtest_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive crypto backtest report"""
        if "error" in results:
            return f"Backtest Error: {results['error']}"
        
        summary = results["backtest_summary"]
        crypto_metrics = results["crypto_metrics"]
        performance_comparison = results["performance_comparison"]
        
        report = f"""
CRYPTO BACKTEST REPORT
{'='*50}

PERFORMANCE SUMMARY
{'-'*20}
Initial Capital: ${summary['initial_capital']:,.2f}
Final Value: ${summary['final_value']:,.2f}
Total Return: {summary['total_return']:.2%}
Total Trades: {summary['total_trades']}
Volatility: {summary['volatility']:.2%}
Sharpe Ratio: {summary['sharpe_ratio']:.2f}
Max Drawdown: {summary['max_drawdown']:.2%}
Win Rate: {summary['win_rate']:.2%}

CRYPTO METRICS
{'-'*20}
"""
        
        for ticker, metrics in crypto_metrics.items():
            report += f"""
{ticker}:
  Price Return: {metrics['price_return']:.2%}
  7-Day Volatility: {metrics['volatility_7d']:.2%}
  30-Day Volatility: {metrics['volatility_30d']:.2%}
  Avg Volume: {metrics['avg_volume']:,.0f}
  Trend Strength: {metrics['trend_strength']:.2%}
"""
        
        report += f"""
PERFORMANCE COMPARISON
{'-'*20}
"""
        
        for ticker, comparison in performance_comparison.items():
            report += f"""
{ticker}:
  Buy & Hold Return: {comparison['buy_hold_return']:.2%}
  Strategy Return: {comparison['strategy_return']:.2%}
  Outperformance: {comparison['outperformance']:.2%}
  Relative Performance: {comparison['relative_performance']:.2f}x
"""
        
        return report
    
    def save_backtest_results(self, results: Dict[str, Any], filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_backtest_results_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results
        clean_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                clean_results[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                clean_results[key] = convert_numpy(value)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        logger.info(f"Crypto backtest results saved to {filename}")
        return filename


def create_crypto_historical_data(ticker: str, start_date: str, end_date: str, 
                                initial_price: float = None) -> pd.DataFrame:
    """
    Create realistic crypto historical data for backtesting
    
    Args:
        ticker: Crypto ticker (BTC, ETH)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_price: Starting price
        
    Returns:
        OHLCV DataFrame with hourly data
    """
    # Set realistic initial prices
    if initial_price is None:
        if ticker == "BTC":
            initial_price = 45000.0
        elif ticker == "ETH":
            initial_price = 3000.0
        else:
            initial_price = 100.0
    
    # Generate hourly data
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.date_range(start=start, end=end, freq='H')
    
    # Generate realistic price movements
    np.random.seed(42)
    
    # Crypto-specific parameters
    if ticker == "BTC":
        daily_volatility = 0.03
        daily_drift = 0.0005
    elif ticker == "ETH":
        daily_volatility = 0.04
        daily_drift = 0.0003
    else:
        daily_volatility = 0.05
        daily_drift = 0.0002
    
    # Convert to hourly
    hourly_volatility = daily_volatility / np.sqrt(24)
    hourly_drift = daily_drift / 24
    
    # Generate price series
    returns = np.random.normal(hourly_drift, hourly_volatility, len(dates))
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = hourly_volatility * 0.5
        
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        open_price = prices[i-1] if i > 0 else price
        
        # Ensure OHLC relationships
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        # Generate volume (higher during volatile periods)
        base_volume = 1000000 if ticker in ["BTC", "ETH"] else 100000
        volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume during big moves
        volume = base_volume * volume_multiplier * (1 + np.random.uniform(-0.3, 0.3))
        
        data.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(price, 2),
            'volume': int(volume)
        })
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date'
    
    logger.info(f"Created crypto historical data for {ticker}: {len(df)} hours, price range ${df['close'].min():.2f}-${df['close'].max():.2f}")
    
    return df

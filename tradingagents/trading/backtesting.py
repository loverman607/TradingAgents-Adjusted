"""
Backtesting Module

This module provides backtesting capabilities for the trading strategy.
It allows testing the strategy against historical data to validate performance.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from .execution_engine import TradingExecutionEngine, Order, OrderSide, OrderType
from .trading_manager import TradingManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.1,
                 commission: float = 0.001):  # 0.1% commission
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as percentage of portfolio
            commission: Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission = commission
        
        # Results storage
        self.backtest_results = []
        self.performance_metrics = {}
        
    def run_backtest(self, 
                    historical_data: Dict[str, pd.DataFrame],
                    analysis_dates: List[str],
                    strategy_function,
                    **strategy_kwargs) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            historical_data: Dict of ticker -> DataFrame with OHLCV data
            analysis_dates: List of dates to run analysis on
            strategy_function: Function that takes (ticker, date, data) and returns analysis result
            **strategy_kwargs: Additional arguments for strategy function
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest with {len(analysis_dates)} analysis dates")
        
        # Initialize trading engine for backtest
        engine = TradingExecutionEngine(self.initial_capital, self.max_position_size)
        
        # Track results
        daily_values = []
        trades = []
        analysis_results = []
        
        for date in analysis_dates:
            logger.info(f"Processing date: {date}")
            
            # Run analysis for each ticker
            for ticker, data in historical_data.items():
                # Get data up to current date
                current_data = data[data.index <= date]
                if current_data.empty:
                    continue
                
                # Run strategy analysis
                try:
                    analysis_result = strategy_function(ticker, date, current_data, **strategy_kwargs)
                    analysis_results.append({
                        'date': date,
                        'ticker': ticker,
                        'analysis': analysis_result
                    })
                    
                    # Extract signal and execute trades
                    signal = self._extract_signal(analysis_result)
                    current_price = current_data['close'].iloc[-1]
                    
                    if signal == "BUY":
                        self._execute_backtest_buy(engine, ticker, current_price, date, analysis_result)
                    elif signal == "SELL":
                        self._execute_backtest_sell(engine, ticker, current_price, date, analysis_result)
                    
                    # Update prices
                    engine.update_prices({ticker: current_price})
                    
                except Exception as e:
                    logger.warning(f"Error processing {ticker} on {date}: {e}")
                    continue
            
            # Record daily portfolio value
            portfolio_value = engine.get_portfolio_value()
            daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'available_capital': engine.available_capital,
                'positions': {ticker: pos.quantity for ticker, pos in engine.positions.items()}
            })
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(daily_values, engine.trade_history)
        
        # Store results
        self.backtest_results = {
            'daily_values': daily_values,
            'trades': engine.trade_history,
            'analysis_results': analysis_results,
            'performance_metrics': performance_metrics,
            'final_portfolio': engine.get_portfolio_summary()
        }
        
        logger.info("Backtest completed")
        return self.backtest_results
    
    def _extract_signal(self, analysis_result: Dict[str, Any]) -> str:
        """Extract trading signal from analysis result"""
        # Similar to TradingManager._extract_signal
        final_decision = analysis_result.get("final_trade_decision", "")
        final_decision_upper = final_decision.upper()
        
        if "BUY" in final_decision_upper and "SELL" not in final_decision_upper:
            return "BUY"
        elif "SELL" in final_decision_upper and "BUY" not in final_decision_upper:
            return "SELL"
        elif "HOLD" in final_decision_upper:
            return "HOLD"
        
        # Fallback: check trader investment plan
        trader_plan = analysis_result.get("trader_investment_plan", "")
        trader_plan_upper = trader_plan.upper()
        
        if "FINAL TRANSACTION PROPOSAL: **BUY**" in trader_plan_upper:
            return "BUY"
        elif "FINAL TRANSACTION PROPOSAL: **SELL**" in trader_plan_upper:
            return "SELL"
        elif "FINAL TRANSACTION PROPOSAL: **HOLD**" in trader_plan_upper:
            return "HOLD"
        
        return "HOLD"
    
    def _execute_backtest_buy(self, engine: TradingExecutionEngine, ticker: str, 
                            price: float, date: str, analysis_result: Dict[str, Any]):
        """Execute buy order in backtest"""
        # Calculate position size
        portfolio_value = engine.get_portfolio_value()
        max_position_value = portfolio_value * self.max_position_size
        quantity = max_position_value / price if price > 0 else 0
        
        if quantity <= 0:
            return
        
        # Create and execute order
        order = Order(
            id=f"backtest_buy_{ticker}_{date}",
            ticker=ticker,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            reason=f"Backtest BUY signal on {date}"
        )
        
        # Apply commission
        commission_cost = quantity * price * self.commission
        if engine.available_capital >= (quantity * price + commission_cost):
            engine.place_order(order)
            engine.available_capital -= commission_cost  # Deduct commission
    
    def _execute_backtest_sell(self, engine: TradingExecutionEngine, ticker: str, 
                             price: float, date: str, analysis_result: Dict[str, Any]):
        """Execute sell order in backtest"""
        position = engine.positions.get(ticker)
        if not position or position.quantity <= 0:
            return
        
        # Sell entire position
        quantity = position.quantity
        
        order = Order(
            id=f"backtest_sell_{ticker}_{date}",
            ticker=ticker,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
            reason=f"Backtest SELL signal on {date}"
        )
        
        # Apply commission
        commission_cost = quantity * price * self.commission
        engine.place_order(order)
        engine.available_capital -= commission_cost  # Deduct commission
    
    def _calculate_performance_metrics(self, daily_values: List[Dict], trades: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not daily_values:
            return {}
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Basic metrics
        initial_value = df['portfolio_value'].iloc[0]
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Risk metrics
        volatility = df['daily_return'].std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (df['daily_return'].mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        df['cumulative_max'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max']
        max_drawdown = df['drawdown'].min()
        
        # Trade analysis
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('realized_pnl', 0) > 0])
        losing_trades = len([t for t in trades if t.get('realized_pnl', 0) < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average win/loss
        wins = [t['realized_pnl'] for t in trades if t.get('realized_pnl', 0) > 0]
        losses = [t['realized_pnl'] for t in trades if t.get('realized_pnl', 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = total_return * (252 / len(df))
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'initial_value': initial_value,
            'final_value': final_value
        }
    
    def generate_backtest_report(self) -> str:
        """Generate formatted backtest report"""
        if not self.backtest_results:
            return "No backtest results available"
        
        metrics = self.backtest_results['performance_metrics']
        
        report = []
        report.append("=" * 60)
        report.append("BACKTEST PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Initial Capital: ${metrics['initial_value']:,.2f}")
        report.append(f"Final Value: ${metrics['final_value']:,.2f}")
        report.append(f"Total Return: {metrics['total_return']:.2%}")
        report.append(f"Annual Return: {metrics['annual_return']:.2%}")
        report.append("")
        
        report.append("RISK METRICS:")
        report.append("-" * 30)
        report.append(f"Volatility: {metrics['volatility']:.2%}")
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        report.append(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        report.append("")
        
        report.append("TRADING METRICS:")
        report.append("-" * 30)
        report.append(f"Total Trades: {metrics['total_trades']}")
        report.append(f"Winning Trades: {metrics['winning_trades']}")
        report.append(f"Losing Trades: {metrics['losing_trades']}")
        report.append(f"Win Rate: {metrics['win_rate']:.1%}")
        report.append(f"Average Win: ${metrics['average_win']:.2f}")
        report.append(f"Average Loss: ${metrics['average_loss']:.2f}")
        report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        return "\n".join(report)
    
    def save_backtest_results(self, filepath: str):
        """Save backtest results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.backtest_results, f, indent=2, default=str)
        logger.info(f"Backtest results saved to {filepath}")
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot backtest performance (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return
        
        if not self.backtest_results:
            logger.warning("No backtest results to plot")
            return
        
        df = pd.DataFrame(self.backtest_results['daily_values'])
        df['date'] = pd.to_datetime(df['date'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Portfolio value over time
        ax1.plot(df['date'], df['portfolio_value'], label='Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Drawdown
        df['cumulative_max'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['cumulative_max']) / df['cumulative_max']
        ax2.fill_between(df['date'], df['drawdown'], 0, alpha=0.3, color='red')
        ax2.plot(df['date'], df['drawdown'], color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Performance plot saved to {save_path}")
        else:
            plt.show()


def create_mock_historical_data(ticker: str, start_date: str, end_date: str, 
                               initial_price: float = 100.0) -> pd.DataFrame:
    """Create mock historical data for backtesting"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random walk price data
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Simple OHLCV generation
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


def run_strategy_backtest(ticker: str, start_date: str, end_date: str, 
                         strategy_function, **strategy_kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a complete backtest
    
    Args:
        ticker: Ticker symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        strategy_function: Function that implements the trading strategy
        **strategy_kwargs: Additional arguments for strategy function
        
    Returns:
        Backtest results
    """
    # Create mock data
    historical_data = {
        ticker: create_mock_historical_data(ticker, start_date, end_date)
    }
    
    # Generate analysis dates (weekly)
    analysis_dates = pd.date_range(start=start_date, end=end_date, freq='W').strftime('%Y-%m-%d').tolist()
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run_backtest(historical_data, analysis_dates, strategy_function, **strategy_kwargs)
    
    return results

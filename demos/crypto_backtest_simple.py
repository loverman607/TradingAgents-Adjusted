"""
Simplified Crypto Backtest System

This is a simplified version that works with your current conda environment
without requiring additional dependencies beyond what you already have.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_backtest_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleCryptoBacktest:
    """Simplified crypto backtest system"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.2,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize simplified crypto backtest
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size
            commission: Commission rate per trade
            slippage: Slippage rate
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission = commission
        self.slippage = slippage
        
        # Results storage
        self.results = {}
        self.trades = []
        self.portfolio_values = []
        self.drawdowns = []
        self.returns = []
        
    def create_crypto_data(self, 
                          ticker: str, 
                          start_date: str, 
                          end_date: str,
                          timeframe: str = "1H") -> pd.DataFrame:
        """
        Create realistic crypto data for backtesting
        
        Args:
            ticker: Crypto ticker (BTC, ETH)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe (1H, 4H, 1D)
            
        Returns:
            OHLCV DataFrame
        """
        # Set realistic initial prices and parameters
        if ticker == "BTC":
            initial_price = 45000.0
            daily_volatility = 0.03
            daily_drift = 0.0005
        elif ticker == "ETH":
            initial_price = 3000.0
            daily_volatility = 0.04
            daily_drift = 0.0003
        else:
            initial_price = 100.0
            daily_volatility = 0.05
            daily_drift = 0.0002
        
        # Generate date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if timeframe == "1H":
            freq = "h"  # Use 'h' instead of 'H'
            periods_per_day = 24
        elif timeframe == "4H":
            freq = "4h"
            periods_per_day = 6
        elif timeframe == "1D":
            freq = "D"
            periods_per_day = 1
        else:
            freq = "h"
            periods_per_day = 24
        
        dates = pd.date_range(start=start, end=end, freq=freq)
        
        # Generate realistic price movements
        np.random.seed(42)  # For reproducible results
        
        # Convert to appropriate timeframe
        period_volatility = daily_volatility / np.sqrt(periods_per_day)
        period_drift = daily_drift / periods_per_day
        
        # Generate returns with some autocorrelation for realism
        returns = np.random.normal(period_drift, period_volatility, len(dates))
        
        # Add some autocorrelation and volatility clustering
        for i in range(1, len(returns)):
            # GARCH-like volatility clustering
            vol_multiplier = 1 + 0.1 * abs(returns[i-1]) / period_volatility
            returns[i] *= vol_multiplier
            
            # Some momentum
            if abs(returns[i-1]) > 2 * period_volatility:
                returns[i] += 0.1 * returns[i-1]
        
        # Generate price series
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            period_vol = period_volatility * 0.5
            
            high = price * (1 + np.random.uniform(0, period_vol))
            low = price * (1 - np.random.uniform(0, period_vol))
            open_price = prices[i-1] if i > 0 else price
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            # Generate volume (higher during volatile periods)
            base_volume = 1000000 if ticker in ["BTC", "ETH"] else 100000
            volatility_multiplier = 1 + abs(returns[i]) * 10
            volume = base_volume * volatility_multiplier * (1 + np.random.uniform(-0.3, 0.3))
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': int(volume)
            })
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        logger.info(f"Created {ticker} data: {len(df)} periods, price range ${df['close'].min():.2f}-${df['close'].max():.2f}")
        
        return df
    
    def run_backtest(self, 
                    tickers: list,
                    start_date: str,
                    end_date: str,
                    timeframe: str = "1H",
                    strategy: str = "adaptive") -> dict:
        """
        Run crypto backtest
        
        Args:
            tickers: List of crypto tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            strategy: Trading strategy
            
        Returns:
            Backtest results
        """
        logger.info(f"Starting crypto backtest for {tickers} from {start_date} to {end_date}")
        
        # Create historical data
        historical_data = {}
        for ticker in tickers:
            logger.info(f"Generating data for {ticker}...")
            historical_data[ticker] = self.create_crypto_data(
                ticker, start_date, end_date, timeframe
            )
        
        # Initialize trading state
        cash = self.initial_capital
        positions = {}
        trades = []
        portfolio_values = []
        drawdowns = []
        returns = []
        
        # Get all unique timestamps
        all_timestamps = set()
        for data in historical_data.values():
            all_timestamps.update(data.index)
        
        timestamps = sorted(list(all_timestamps))
        logger.info(f"Running backtest on {len(timestamps)} data points")
        
        # Run backtest
        for i, timestamp in enumerate(timestamps):
            if i % 100 == 0:
                logger.info(f"Processing {i+1}/{len(timestamps)}: {timestamp}")
            
            # Update prices
            current_prices = {}
            for ticker, data in historical_data.items():
                if timestamp in data.index:
                    current_prices[ticker] = data.loc[timestamp, 'close']
            
            # Analyze each ticker
            for ticker, data in historical_data.items():
                if timestamp not in data.index:
                    continue
                
                # Get data up to current timestamp
                data_up_to_timestamp = data.loc[:timestamp]
                
                if len(data_up_to_timestamp) < 24:  # Need sufficient data
                    continue
                
                # Generate trading signal using simplified strategy
                signal = self._generate_trading_signal(data_up_to_timestamp, ticker)
                
                # Execute trade if signal is strong enough
                if signal['signal'] in ['STRONG_BUY', 'STRONG_SELL'] and signal['confidence'] > 0.6:
                    self._execute_trade(
                        ticker, signal, current_prices[ticker], 
                        timestamp, cash, positions, trades
                    )
                elif signal['signal'] in ['BUY', 'SELL'] and signal['confidence'] > 0.7:
                    self._execute_trade(
                        ticker, signal, current_prices[ticker], 
                        timestamp, cash, positions, trades
                    )
            
            # Calculate portfolio value
            portfolio_value = cash
            for ticker, position in positions.items():
                if ticker in current_prices:
                    portfolio_value += position['quantity'] * current_prices[ticker]
            
            portfolio_values.append({
                'timestamp': timestamp,
                'value': portfolio_value,
                'cash': cash,
                'positions': len(positions)
            })
            
            # Calculate returns
            if len(portfolio_values) > 1:
                prev_value = portfolio_values[-2]['value']
                current_return = (portfolio_value - prev_value) / prev_value
                returns.append(current_return)
            
            # Calculate drawdown
            if portfolio_values:
                peak_value = max(pv['value'] for pv in portfolio_values)
                current_drawdown = (peak_value - portfolio_value) / peak_value
                drawdowns.append({
                    'timestamp': timestamp,
                    'drawdown': current_drawdown
                })
        
        # Calculate final results
        results = self._calculate_results(
            portfolio_values, trades, drawdowns, returns, historical_data
        )
        
        # Store results
        self.results = results
        self.trades = trades
        self.portfolio_values = portfolio_values
        self.drawdowns = drawdowns
        self.returns = returns
        
        logger.info("Crypto backtest completed")
        return results
    
    def _generate_trading_signal(self, data: pd.DataFrame, ticker: str) -> dict:
        """Generate trading signal using simplified strategy"""
        if len(data) < 20:
            return {"signal": "HOLD", "confidence": 0.5, "reasoning": ["Insufficient data"]}
        
        prices = data['close']
        current_price = prices.iloc[-1]
        
        # Calculate indicators
        rsi = self._calculate_rsi(prices)
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else sma_20
        
        # Calculate volatility
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(365)  # Annualized
        
        # Generate signal
        signal_score = 0
        reasoning = []
        
        # RSI signals
        if rsi < 30:
            signal_score += 2
            reasoning.append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            signal_score += 1
            reasoning.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            signal_score -= 2
            reasoning.append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            signal_score -= 1
            reasoning.append(f"RSI high ({rsi:.1f})")
        
        # Trend signals
        if current_price > sma_20 > sma_50:
            signal_score += 2
            reasoning.append("Strong uptrend")
        elif current_price > sma_20:
            signal_score += 1
            reasoning.append("Weak uptrend")
        elif current_price < sma_20 < sma_50:
            signal_score -= 2
            reasoning.append("Strong downtrend")
        elif current_price < sma_20:
            signal_score -= 1
            reasoning.append("Weak downtrend")
        
        # Volatility adjustment
        if volatility > 0.5:  # High volatility
            signal_score *= 0.8
            reasoning.append("High volatility - reduced confidence")
        
        # Determine final signal
        if signal_score >= 3:
            signal = "STRONG_BUY"
            confidence = min(0.9, 0.6 + signal_score * 0.1)
        elif signal_score >= 1:
            signal = "BUY"
            confidence = min(0.8, 0.4 + signal_score * 0.1)
        elif signal_score <= -3:
            signal = "STRONG_SELL"
            confidence = min(0.9, 0.6 + abs(signal_score) * 0.1)
        elif signal_score <= -1:
            signal = "SELL"
            confidence = min(0.8, 0.4 + abs(signal_score) * 0.1)
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
                "sma_20": sma_20,
                "sma_50": sma_50,
                "volatility": volatility
            }
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 1
        rsi = 100 - (100 / (1 + rs))
        
        return min(100, max(0, rsi))
    
    def _execute_trade(self, 
                      ticker: str, 
                      signal: dict, 
                      price: float, 
                      timestamp: pd.Timestamp,
                      cash: float, 
                      positions: dict, 
                      trades: list):
        """Execute trade"""
        try:
            # Calculate position size (10% of portfolio)
            portfolio_value = cash + sum(pos['quantity'] * price for pos in positions.values())
            position_size = 0.1 * portfolio_value / price
            
            # Apply slippage
            if signal['signal'] in ['BUY', 'STRONG_BUY']:
                execution_price = price * (1 + self.slippage)
                order_side = 'BUY'
            else:
                execution_price = price * (1 - self.slippage)
                order_side = 'SELL'
            
            # Check if we can execute
            if order_side == 'BUY' and execution_price * position_size > cash:
                return  # Insufficient cash
            elif order_side == 'SELL' and (ticker not in positions or positions[ticker]['quantity'] < position_size):
                return  # No position to sell
            
            # Execute trade
            if order_side == 'BUY':
                cost = execution_price * position_size
                cash -= cost
                
                if ticker in positions:
                    # Average down/up
                    old_quantity = positions[ticker]['quantity']
                    old_avg_price = positions[ticker]['avg_price']
                    new_quantity = old_quantity + position_size
                    new_avg_price = (old_quantity * old_avg_price + cost) / new_quantity
                    positions[ticker] = {'quantity': new_quantity, 'avg_price': new_avg_price}
                else:
                    positions[ticker] = {'quantity': position_size, 'avg_price': execution_price}
            else:  # SELL
                proceeds = execution_price * position_size
                cash += proceeds
                
                positions[ticker]['quantity'] -= position_size
                if positions[ticker]['quantity'] <= 0.001:
                    del positions[ticker]
            
            # Record trade
            trades.append({
                'timestamp': timestamp,
                'ticker': ticker,
                'side': order_side,
                'quantity': position_size,
                'price': execution_price,
                'value': execution_price * position_size,
                'commission': execution_price * position_size * self.commission,
                'signal': signal['signal'],
                'confidence': signal['confidence']
            })
            
            logger.info(f"{order_side}: {position_size:.6f} {ticker} @ ${execution_price:.2f}")
            
        except Exception as e:
            logger.warning(f"Error executing trade: {e}")
    
    def _calculate_results(self, 
                          portfolio_values: list, 
                          trades: list, 
                          drawdowns: list, 
                          returns: list,
                          historical_data: dict) -> dict:
        """Calculate backtest results"""
        if not portfolio_values:
            return {"error": "No portfolio values recorded"}
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = portfolio_values[-1]['value']
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate additional metrics
        if returns:
            volatility = np.std(returns) * np.sqrt(365)  # Annualized
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
            max_drawdown = max(dd['drawdown'] for dd in drawdowns) if drawdowns else 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade analysis
        total_trades = len(trades)
        buy_trades = len([t for t in trades if t['side'] == 'BUY'])
        sell_trades = len([t for t in trades if t['side'] == 'SELL'])
        
        # Calculate win rate
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        
        # Analyze trade pairs
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                if buy_trade['side'] == 'BUY' and sell_trade['side'] == 'SELL':
                    pnl = sell_trade['value'] - buy_trade['value']
                    total_pnl += pnl
                    if pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        # Crypto-specific metrics
        crypto_metrics = {}
        for ticker, data in historical_data.items():
            if data.empty:
                continue
            
            start_price = data['close'].iloc[0]
            end_price = data['close'].iloc[-1]
            price_return = (end_price - start_price) / start_price
            
            # Calculate buy and hold performance
            buy_hold_return = price_return
            
            crypto_metrics[ticker] = {
                "price_return": price_return,
                "buy_hold_return": buy_hold_return,
                "outperformance": total_return - buy_hold_return,
                "volatility": data['close'].pct_change().std() * np.sqrt(365),
                "max_price": data['close'].max(),
                "min_price": data['close'].min()
            }
        
        return {
            "backtest_summary": {
                "initial_capital": initial_value,
                "final_value": final_value,
                "total_return": total_return,
                "total_trades": total_trades,
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_pnl": total_pnl
            },
            "crypto_metrics": crypto_metrics,
            "portfolio_values": portfolio_values,
            "trades": trades,
            "drawdowns": drawdowns,
            "returns": returns
        }
    
    def generate_report(self, results: dict) -> str:
        """Generate backtest report"""
        if "error" in results:
            return f"Backtest Error: {results['error']}"
        
        summary = results["backtest_summary"]
        crypto_metrics = results["crypto_metrics"]
        
        report = f"""
CRYPTO BACKTEST REPORT
{'='*50}

PERFORMANCE SUMMARY
{'-'*20}
Initial Capital: ${summary['initial_capital']:,.2f}
Final Value: ${summary['final_value']:,.2f}
Total Return: {summary['total_return']:.2%}
Total Trades: {summary['total_trades']}
Buy Trades: {summary['buy_trades']}
Sell Trades: {summary['sell_trades']}
Win Rate: {summary['win_rate']:.2%}
Total P&L: ${summary['total_pnl']:,.2f}

RISK METRICS
{'-'*20}
Volatility: {summary['volatility']:.2%}
Sharpe Ratio: {summary['sharpe_ratio']:.2f}
Max Drawdown: {summary['max_drawdown']:.2%}

CRYPTO PERFORMANCE
{'-'*20}
"""
        
        for ticker, metrics in crypto_metrics.items():
            report += f"""
{ticker}:
  Price Return: {metrics['price_return']:.2%}
  Buy & Hold Return: {metrics['buy_hold_return']:.2%}
  Outperformance: {metrics['outperformance']:.2%}
  Volatility: {metrics['volatility']:.2%}
  Max Price: ${metrics['max_price']:,.2f}
  Min Price: ${metrics['min_price']:,.2f}
"""
        
        return report
    
    def create_charts(self, results: dict, save_path: str = "crypto_backtest_charts"):
        """Create visualization charts"""
        if "error" in results:
            logger.error("Cannot create charts due to backtest error")
            return
        
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Portfolio Value Over Time
        self._create_portfolio_chart(results, save_path)
        
        # 2. Drawdown Chart
        self._create_drawdown_chart(results, save_path)
        
        # 3. Returns Distribution
        self._create_returns_chart(results, save_path)
        
        logger.info(f"Charts saved to {save_path}/")
    
    def _create_portfolio_chart(self, results: dict, save_path: str):
        """Create portfolio value chart"""
        portfolio_values = results.get("portfolio_values", [])
        if not portfolio_values:
            return
        
        df = pd.DataFrame(portfolio_values)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 8))
        plt.plot(df['timestamp'], df['value'], linewidth=2, label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/portfolio_value.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_drawdown_chart(self, results: dict, save_path: str):
        """Create drawdown chart"""
        drawdowns = results.get("drawdowns", [])
        if not drawdowns:
            return
        
        df = pd.DataFrame(drawdowns)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['timestamp'], df['drawdown'] * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        plt.plot(df['timestamp'], df['drawdown'] * 100, color='red', linewidth=1)
        plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/drawdown.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_returns_chart(self, results: dict, save_path: str):
        """Create returns distribution chart"""
        returns = results.get("returns", [])
        if not returns:
            return
        
        returns = np.array(returns) * 100  # Convert to percentage
        
        plt.figure(figsize=(12, 6))
        plt.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}%')
        plt.title('Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Returns (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}/returns_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: dict, filename: str = None):
        """Save backtest results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_backtest_simple_{timestamp}.json"
        
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
        
        logger.info(f"Results saved to {filename}")
        return filename


def main():
    """Main function to run crypto backtests"""
    print("CRYPTO BACKTEST SYSTEM - SIMPLIFIED")
    print("=" * 50)
    print("Comprehensive backtesting for crypto-optimized TradingAgents")
    print("=" * 50)
    
    # Configuration
    tickers = ["BTC", "ETH"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    initial_capital = 100000.0
    
    try:
        # Initialize backtest
        backtest = SimpleCryptoBacktest(
            initial_capital=initial_capital,
            max_position_size=0.2,
            commission=0.001,
            slippage=0.0005
        )
        
        # Run backtest
        print(f"\nRunning backtest for {', '.join(tickers)}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        
        results = backtest.run_backtest(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            timeframe="1H",
            strategy="adaptive"
        )
        
        # Generate report
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        report = backtest.generate_report(results)
        print(report)
        
        # Create charts
        print("\nCreating visualization charts...")
        backtest.create_charts(results)
        
        # Save results
        filename = backtest.save_results(results)
        
        print(f"\n{'='*60}")
        print("CRYPTO BACKTEST COMPLETED! 🚀")
        print(f"{'='*60}")
        print(f"\n✅ Backtest completed successfully!")
        print(f"📊 Results saved to: {filename}")
        print(f"📈 Charts saved to: crypto_backtest_charts/")
        print(f"📋 Log saved to: crypto_backtest_simple.log")
        
        return True
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

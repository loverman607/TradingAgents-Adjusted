"""
Crypto Backtest Launcher

This script provides a comprehensive interface for running crypto backtests
with your conda environment. It includes multiple backtest scenarios and
automated result analysis.
"""

import sys
import os
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import subprocess
import webbrowser
import time
import pandas as pd

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from crypto_backtest_runner import CryptoBacktestRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_backtest_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryptoBacktestLauncher:
    """Comprehensive crypto backtest launcher"""
    
    def __init__(self, config_file: str = "crypto_backtest_config.yaml"):
        """
        Initialize backtest launcher
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
        self.results_dir = Path(self.config['output']['results_dir'])
        self.charts_dir = Path(self.config['output']['charts_dir'])
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        
    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_file} not found, using defaults")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'general': {
                'initial_capital': 100000.0,
                'max_position_size': 0.2,
                'commission': 0.001,
                'slippage': 0.0005
            },
            'backtest': {
                'start_date': '2024-01-01',
                'end_date': '2024-12-31',
                'timeframe': '1H',
                'tickers': ['BTC', 'ETH'],
                'strategy': 'adaptive'
            },
            'output': {
                'save_results': True,
                'save_charts': True,
                'results_dir': 'backtest_results',
                'charts_dir': 'backtest_charts'
            }
        }
    
    def run_single_backtest(self, 
                          tickers: list = None,
                          start_date: str = None,
                          end_date: str = None,
                          strategy: str = None,
                          initial_capital: float = None) -> dict:
        """
        Run a single backtest
        
        Args:
            tickers: List of tickers to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            strategy: Trading strategy
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        # Use config defaults if not provided
        tickers = tickers or self.config['backtest']['tickers']
        start_date = start_date or self.config['backtest']['start_date']
        end_date = end_date or self.config['backtest']['end_date']
        strategy = strategy or self.config['backtest']['strategy']
        initial_capital = initial_capital or self.config['general']['initial_capital']
        
        logger.info(f"Running backtest for {tickers} from {start_date} to {end_date}")
        logger.info(f"Strategy: {strategy}, Capital: ${initial_capital:,.2f}")
        
        # Initialize runner
        runner = CryptoBacktestRunner(
            initial_capital=initial_capital,
            max_position_size=self.config['general']['max_position_size'],
            commission=self.config['general']['commission'],
            slippage=self.config['general']['slippage']
        )
        
        # Run backtest
        results = runner.run_crypto_backtest(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            timeframe=self.config['backtest']['timeframe'],
            strategy=strategy
        )
        
        # Save results
        if self.config['output']['save_results']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.results_dir}/crypto_backtest_{timestamp}.json"
            runner.save_results(results, filename)
            logger.info(f"Results saved to {filename}")
        
        # Create charts
        if self.config['output']['save_charts']:
            runner.create_visualization_charts(results, str(self.charts_dir))
            logger.info(f"Charts saved to {self.charts_dir}")
        
        return results
    
    def run_multiple_backtests(self, scenarios: list) -> dict:
        """
        Run multiple backtest scenarios
        
        Args:
            scenarios: List of backtest scenarios
            
        Returns:
            Combined results
        """
        logger.info(f"Running {len(scenarios)} backtest scenarios")
        
        all_results = {}
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"Running scenario {i+1}/{len(scenarios)}: {scenario.get('name', f'Scenario {i+1}')}")
            
            try:
                results = self.run_single_backtest(**scenario)
                all_results[scenario.get('name', f'scenario_{i+1}')] = results
            except Exception as e:
                logger.error(f"Error in scenario {i+1}: {e}")
                all_results[scenario.get('name', f'scenario_{i+1}')] = {"error": str(e)}
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/multiple_backtests_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Combined results saved to {filename}")
        return all_results
    
    def run_strategy_comparison(self, 
                              tickers: list = None,
                              start_date: str = None,
                              end_date: str = None) -> dict:
        """
        Run backtest comparison across different strategies
        
        Args:
            tickers: List of tickers to test
            start_date: Start date
            end_date: End date
            
        Returns:
            Strategy comparison results
        """
        strategies = ['momentum', 'mean_reversion', 'trend_following', 'breakout', 'adaptive']
        
        logger.info(f"Running strategy comparison for {tickers}")
        
        scenarios = []
        for strategy in strategies:
            scenarios.append({
                'name': f'{strategy}_strategy',
                'tickers': tickers,
                'start_date': start_date,
                'end_date': end_date,
                'strategy': strategy
            })
        
        return self.run_multiple_backtests(scenarios)
    
    def run_timeframe_comparison(self, 
                               tickers: list = None,
                               start_date: str = None,
                               end_date: str = None) -> dict:
        """
        Run backtest comparison across different timeframes
        
        Args:
            tickers: List of tickers to test
            start_date: Start date
            end_date: End date
            
        Returns:
            Timeframe comparison results
        """
        timeframes = ['1H', '4H', '1D']
        
        logger.info(f"Running timeframe comparison for {tickers}")
        
        scenarios = []
        for timeframe in timeframes:
            scenarios.append({
                'name': f'{timeframe}_timeframe',
                'tickers': tickers,
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe
            })
        
        return self.run_multiple_backtests(scenarios)
    
    def run_period_comparison(self, 
                            tickers: list = None,
                            periods: list = None) -> dict:
        """
        Run backtest comparison across different time periods
        
        Args:
            tickers: List of tickers to test
            periods: List of period dictionaries with start_date and end_date
            
        Returns:
            Period comparison results
        """
        if periods is None:
            # Default periods
            periods = [
                {'name': 'Q1_2024', 'start_date': '2024-01-01', 'end_date': '2024-03-31'},
                {'name': 'Q2_2024', 'start_date': '2024-04-01', 'end_date': '2024-06-30'},
                {'name': 'Q3_2024', 'start_date': '2024-07-01', 'end_date': '2024-09-30'},
                {'name': 'Q4_2024', 'start_date': '2024-10-01', 'end_date': '2024-12-31'}
            ]
        
        logger.info(f"Running period comparison for {tickers}")
        
        scenarios = []
        for period in periods:
            scenarios.append({
                'name': period['name'],
                'tickers': tickers,
                'start_date': period['start_date'],
                'end_date': period['end_date']
            })
        
        return self.run_multiple_backtests(scenarios)
    
    def generate_comparison_report(self, results: dict) -> str:
        """Generate comprehensive comparison report"""
        if not results:
            return "No results to compare"
        
        report = f"""
CRYPTO BACKTEST COMPARISON REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Summary table
        summary_data = []
        for name, result in results.items():
            if 'error' in result:
                summary_data.append({
                    'Scenario': name,
                    'Status': 'ERROR',
                    'Total Return': 'N/A',
                    'Sharpe Ratio': 'N/A',
                    'Max Drawdown': 'N/A',
                    'Total Trades': 'N/A'
                })
            else:
                summary = result.get('backtest_summary', {})
                summary_data.append({
                    'Scenario': name,
                    'Status': 'SUCCESS',
                    'Total Return': f"{summary.get('total_return', 0):.2%}",
                    'Sharpe Ratio': f"{summary.get('sharpe_ratio', 0):.2f}",
                    'Max Drawdown': f"{summary.get('max_drawdown', 0):.2%}",
                    'Total Trades': summary.get('total_trades', 0)
                })
        
        # Create summary table
        df = pd.DataFrame(summary_data)
        report += df.to_string(index=False)
        report += "\n\n"
        
        # Detailed analysis
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if successful_results:
            # Find best performing scenario
            best_return = max(successful_results.items(), 
                            key=lambda x: x[1].get('backtest_summary', {}).get('total_return', 0))
            best_sharpe = max(successful_results.items(), 
                            key=lambda x: x[1].get('backtest_summary', {}).get('sharpe_ratio', 0))
            
            report += f"""
BEST PERFORMING SCENARIOS
{'-'*30}
Best Return: {best_return[0]} ({best_return[1]['backtest_summary']['total_return']:.2%})
Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['backtest_summary']['sharpe_ratio']:.2f})

"""
        
        return report
    
    def launch_dashboard(self, results_file: str = None):
        """Launch the interactive dashboard"""
        try:
            logger.info("Launching crypto backtest dashboard...")
            
            # Start dashboard in background
            dashboard_process = subprocess.Popen([
                sys.executable, 'crypto_backtest_dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for dashboard to start
            time.sleep(3)
            
            # Open browser
            webbrowser.open('http://127.0.0.1:8050')
            
            logger.info("Dashboard launched at http://127.0.0.1:8050")
            logger.info("Press Ctrl+C to stop the dashboard")
            
            # Wait for user to stop
            try:
                dashboard_process.wait()
            except KeyboardInterrupt:
                logger.info("Stopping dashboard...")
                dashboard_process.terminate()
                
        except Exception as e:
            logger.error(f"Error launching dashboard: {e}")
    
    def run_comprehensive_analysis(self, 
                                 tickers: list = None,
                                 start_date: str = None,
                                 end_date: str = None) -> dict:
        """
        Run comprehensive analysis including multiple comparisons
        
        Args:
            tickers: List of tickers to test
            start_date: Start date
            end_date: End date
            
        Returns:
            Comprehensive analysis results
        """
        logger.info("Running comprehensive crypto backtest analysis")
        
        # Use config defaults if not provided
        tickers = tickers or self.config['backtest']['tickers']
        start_date = start_date or self.config['backtest']['start_date']
        end_date = end_date or self.config['backtest']['end_date']
        
        all_results = {}
        
        # 1. Single backtest
        logger.info("1. Running single backtest...")
        single_result = self.run_single_backtest(tickers, start_date, end_date)
        all_results['single_backtest'] = single_result
        
        # 2. Strategy comparison
        logger.info("2. Running strategy comparison...")
        strategy_results = self.run_strategy_comparison(tickers, start_date, end_date)
        all_results['strategy_comparison'] = strategy_results
        
        # 3. Timeframe comparison
        logger.info("3. Running timeframe comparison...")
        timeframe_results = self.run_timeframe_comparison(tickers, start_date, end_date)
        all_results['timeframe_comparison'] = timeframe_results
        
        # 4. Period comparison
        logger.info("4. Running period comparison...")
        period_results = self.run_period_comparison(tickers)
        all_results['period_comparison'] = period_results
        
        # Generate comprehensive report
        logger.info("5. Generating comprehensive report...")
        report = self.generate_comparison_report(all_results)
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.results_dir}/comprehensive_analysis_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Comprehensive report saved to {report_file}")
        
        # Save all results
        results_file = f"{self.results_dir}/comprehensive_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"All results saved to {results_file}")
        
        return all_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Crypto Backtest Launcher')
    parser.add_argument('--mode', choices=['single', 'strategy', 'timeframe', 'period', 'comprehensive', 'dashboard'], 
                       default='single', help='Backtest mode')
    parser.add_argument('--tickers', nargs='+', default=['BTC', 'ETH'], help='Tickers to test')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', default='adaptive', help='Trading strategy')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--config', default='crypto_backtest_config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = CryptoBacktestLauncher(args.config)
    
    try:
        if args.mode == 'single':
            results = launcher.run_single_backtest(
                tickers=args.tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                strategy=args.strategy,
                initial_capital=args.capital
            )
            
        elif args.mode == 'strategy':
            results = launcher.run_strategy_comparison(
                tickers=args.tickers,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
        elif args.mode == 'timeframe':
            results = launcher.run_timeframe_comparison(
                tickers=args.tickers,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
        elif args.mode == 'period':
            results = launcher.run_period_comparison(tickers=args.tickers)
            
        elif args.mode == 'comprehensive':
            results = launcher.run_comprehensive_analysis(
                tickers=args.tickers,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
        elif args.mode == 'dashboard':
            launcher.launch_dashboard()
            return
        
        print(f"\n{'='*60}")
        print("CRYPTO BACKTEST COMPLETED! 🚀")
        print(f"{'='*60}")
        print(f"Mode: {args.mode}")
        print(f"Tickers: {', '.join(args.tickers)}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Results saved to: {launcher.results_dir}")
        print(f"Charts saved to: {launcher.charts_dir}")
        
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

"""
Trading CLI Interface

Command-line interface for the trading system.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from .trading_manager import TradingManager
from .dashboard import TradingDashboard
from .backtesting import BacktestEngine, create_mock_historical_data

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_analysis_command(args):
    """Run trading analysis for a single ticker"""
    print(f"Running analysis for {args.ticker} on {args.date}")
    
    # Initialize trading manager
    trading_manager = TradingManager(
        initial_capital=args.capital,
        max_position_size=args.max_position / 100,  # Convert percentage to decimal
        state_file=args.state_file
    )
    
    # Import here to avoid circular imports
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    
    # Create config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.llm_provider
    config["backend_url"] = args.backend_url
    config["deep_think_llm"] = args.model
    config["quick_think_llm"] = args.model
    config["max_debate_rounds"] = args.debate_rounds
    config["online_tools"] = args.online_tools
    
    # Initialize analysis system
    ta = TradingAgentsGraph(debug=args.debug, config=config)
    
    # Run analysis
    final_state, decision = ta.propagate(args.ticker, args.date)
    
    # Process and execute trades
    trading_result = trading_manager.process_analysis_result(
        ticker=args.ticker,
        analysis_result=final_state,
        trade_date=args.date
    )
    
    # Display results
    dashboard = TradingDashboard(trading_manager)
    
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"Signal: {trading_result['signal']}")
    print(f"Actions Taken: {len(trading_result['actions_taken'])}")
    
    for action in trading_result['actions_taken']:
        print(f"  - {action['action']}: {action.get('quantity', 0):.2f} {args.ticker} @ ${action.get('price', 0):.2f}")
        if action.get('stop_loss'):
            print(f"    Stop Loss: ${action['stop_loss']:.2f}")
        if action.get('take_profit'):
            print(f"    Take Profit: ${action['take_profit']:.2f}")
    
    print("\n" + dashboard.display_portfolio_summary())
    
    # Save reports if requested
    if args.save_reports:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trading_manager.save_analysis_report(f"trading_report_{args.ticker}_{timestamp}.json")
        dashboard.save_dashboard_report(f"dashboard_report_{args.ticker}_{timestamp}.json")
        print(f"\nReports saved with timestamp: {timestamp}")


def run_portfolio_command(args):
    """Display portfolio status"""
    if not Path(args.state_file).exists():
        print(f"State file {args.state_file} not found. Run analysis first.")
        return
    
    # Load trading manager
    trading_manager = TradingManager(state_file=args.state_file)
    dashboard = TradingDashboard(trading_manager)
    
    print(dashboard.display_portfolio_summary())
    
    if args.show_trades:
        print("\n" + dashboard.display_recent_trades(limit=args.trade_limit))
    
    if args.show_analysis:
        print("\n" + dashboard.display_analysis_history(limit=args.analysis_limit))
    
    if args.show_performance:
        print("\n" + dashboard.display_performance_report())


def run_backtest_command(args):
    """Run backtest on historical data"""
    print(f"Running backtest for {args.ticker} from {args.start_date} to {args.end_date}")
    
    # Create mock historical data
    historical_data = {
        args.ticker: create_mock_historical_data(
            args.ticker, 
            args.start_date, 
            args.end_date,
            initial_price=args.initial_price
        )
    }
    
    # Generate analysis dates
    import pandas as pd
    analysis_dates = pd.date_range(
        start=args.start_date, 
        end=args.end_date, 
        freq=args.frequency
    ).strftime('%Y-%m-%d').tolist()
    
    # Mock strategy function (in real implementation, this would use the actual analysis)
    def mock_strategy(ticker, date, data, **kwargs):
        # Simple mock strategy based on price momentum
        if len(data) < 20:
            return {"final_trade_decision": "HOLD"}
        
        # Calculate 20-day moving average
        ma20 = data['close'].rolling(20).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        if current_price > ma20 * 1.02:  # 2% above MA
            return {"final_trade_decision": "BUY"}
        elif current_price < ma20 * 0.98:  # 2% below MA
            return {"final_trade_decision": "SELL"}
        else:
            return {"final_trade_decision": "HOLD"}
    
    # Run backtest
    engine = BacktestEngine(
        initial_capital=args.capital,
        max_position_size=args.max_position / 100,
        commission=args.commission / 100
    )
    
    results = engine.run_backtest(
        historical_data, 
        analysis_dates, 
        mock_strategy
    )
    
    # Display results
    print("\n" + engine.generate_backtest_report())
    
    # Save results if requested
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        engine.save_backtest_results(f"backtest_results_{args.ticker}_{timestamp}.json")
        print(f"\nBacktest results saved with timestamp: {timestamp}")


def run_reset_command(args):
    """Reset portfolio to initial state"""
    trading_manager = TradingManager(
        initial_capital=args.capital,
        max_position_size=args.max_position / 100,
        state_file=args.state_file
    )
    
    trading_manager.reset_portfolio()
    print(f"Portfolio reset to initial state with ${args.capital:,.2f} capital")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Trading Agents CLI")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Set logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analysis command
    analysis_parser = subparsers.add_parser("analyze", help="Run trading analysis")
    analysis_parser.add_argument("ticker", help="Ticker symbol to analyze")
    analysis_parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                               help="Analysis date (YYYY-MM-DD)")
    analysis_parser.add_argument("--capital", type=float, default=100000.0,
                               help="Initial capital amount")
    analysis_parser.add_argument("--max-position", type=float, default=10.0,
                               help="Maximum position size as percentage of portfolio")
    analysis_parser.add_argument("--state-file", default="trading_state.json",
                               help="State file to save/load trading state")
    analysis_parser.add_argument("--llm-provider", default="google",
                               choices=["google", "openai", "anthropic"],
                               help="LLM provider")
    analysis_parser.add_argument("--backend-url", 
                               default="https://generativelanguage.googleapis.com/v1",
                               help="Backend URL for LLM")
    analysis_parser.add_argument("--model", default="gemini-2.0-flash",
                               help="LLM model name")
    analysis_parser.add_argument("--debate-rounds", type=int, default=1,
                               help="Number of debate rounds")
    analysis_parser.add_argument("--online-tools", action="store_true", default=True,
                               help="Enable online tools")
    analysis_parser.add_argument("--debug", action="store_true",
                               help="Enable debug mode")
    analysis_parser.add_argument("--save-reports", action="store_true",
                               help="Save analysis and dashboard reports")
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Display portfolio status")
    portfolio_parser.add_argument("--state-file", default="trading_state.json",
                                help="State file to load")
    portfolio_parser.add_argument("--show-trades", action="store_true",
                                help="Show recent trades")
    portfolio_parser.add_argument("--show-analysis", action="store_true",
                                help="Show recent analysis")
    portfolio_parser.add_argument("--show-performance", action="store_true",
                                help="Show performance metrics")
    portfolio_parser.add_argument("--trade-limit", type=int, default=10,
                                help="Limit number of recent trades to show")
    portfolio_parser.add_argument("--analysis-limit", type=int, default=5,
                                help="Limit number of recent analyses to show")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("ticker", help="Ticker symbol to backtest")
    backtest_parser.add_argument("--start-date", required=True,
                               help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", required=True,
                               help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--capital", type=float, default=100000.0,
                               help="Initial capital amount")
    backtest_parser.add_argument("--max-position", type=float, default=10.0,
                               help="Maximum position size as percentage")
    backtest_parser.add_argument("--commission", type=float, default=0.1,
                               help="Commission rate as percentage")
    backtest_parser.add_argument("--initial-price", type=float, default=100.0,
                               help="Initial price for mock data")
    backtest_parser.add_argument("--frequency", default="W",
                               choices=["D", "W", "M"],
                               help="Analysis frequency")
    backtest_parser.add_argument("--save-results", action="store_true",
                               help="Save backtest results")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset portfolio")
    reset_parser.add_argument("--capital", type=float, default=100000.0,
                            help="Initial capital amount")
    reset_parser.add_argument("--max-position", type=float, default=10.0,
                            help="Maximum position size as percentage")
    reset_parser.add_argument("--state-file", default="trading_state.json",
                            help="State file to reset")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "analyze":
        run_analysis_command(args)
    elif args.command == "portfolio":
        run_portfolio_command(args)
    elif args.command == "backtest":
        run_backtest_command(args)
    elif args.command == "reset":
        run_reset_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

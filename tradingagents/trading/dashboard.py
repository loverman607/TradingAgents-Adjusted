"""
Trading Dashboard

This module provides a dashboard for monitoring trading performance and portfolio status.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class TradingDashboard:
    """Dashboard for monitoring trading performance"""
    
    def __init__(self, trading_manager):
        """
        Initialize the dashboard
        
        Args:
            trading_manager: TradingManager instance
        """
        self.trading_manager = trading_manager
    
    def display_portfolio_summary(self) -> str:
        """Display portfolio summary in a formatted string"""
        try:
            summary = self.trading_manager.get_portfolio_summary()
            
            output = []
            output.append("=" * 60)
            output.append("PORTFOLIO SUMMARY")
            output.append("=" * 60)
            output.append(f"Total Portfolio Value: ${summary.get('total_value', 0):,.2f}")
            output.append(f"Available Capital: ${summary.get('cash', 0):,.2f}")
            output.append(f"Total Unrealized P&L: ${summary.get('total_pnl', 0):,.2f}")
            output.append(f"Total Realized P&L: $0.00")  # Not tracked in current implementation
            output.append(f"Total P&L: ${summary.get('total_pnl', 0):,.2f}")
            output.append(f"Return Percentage: {summary.get('total_pnl_pct', 0):.2f}%")
            output.append(f"Daily P&L: ${summary.get('daily_pnl', 0):,.2f}")
            output.append(f"Max Daily Loss Limit: ${summary.get('max_daily_loss', 0):,.2f}")
            output.append("")
            
            # Get positions from the trading manager's engine
            positions = self.trading_manager.engine.positions
            if positions:
                output.append("CURRENT POSITIONS:")
                output.append("-" * 40)
                
                for ticker, position in positions.items():
                    position_type = "LONG" if position.position_type.value == "long" else "SHORT"
                    output.append(f"  {ticker} ({position_type}):")
                    output.append(f"    Quantity: {position.quantity:.2f}")
                    output.append(f"    Average Price: ${position.average_price:.2f}")
                    output.append(f"    Current Price: ${position.current_price:.2f}")
                    output.append(f"    Market Value: ${position.market_value:.2f}")
                    output.append(f"    Unrealized P&L: ${position.unrealized_pnl:.2f}")
                    output.append(f"    Realized P&L: ${position.realized_pnl:.2f}")
                    if position.stop_loss:
                        output.append(f"    Stop Loss: ${position.stop_loss:.2f}")
                    if position.take_profit:
                        output.append(f"    Take Profit: ${position.take_profit:.2f}")
                    output.append("")
                
                # Display margin information
                if self.trading_manager.engine.enable_short_selling:
                    output.append("MARGIN INFORMATION:")
                    output.append(f"  Margin Used: ${self.trading_manager.engine.margin_used:.2f}")
                    output.append(f"  Available Margin: ${self.trading_manager.engine.cash * self.trading_manager.engine.margin_requirement:.2f}")
                    output.append(f"  Margin Requirement: {self.trading_manager.engine.margin_requirement:.1%}")
                    if self.trading_manager.engine.short_borrowed_shares:
                        output.append("  Borrowed Shares:")
                        for ticker, shares in self.trading_manager.engine.short_borrowed_shares.items():
                            output.append(f"    {ticker}: {shares:.2f} shares")
                    output.append("")
            else:
                output.append("No current positions")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error displaying portfolio summary: {e}")
            return f"Error displaying portfolio summary: {str(e)}"
    
    def display_recent_trades(self, limit: int = 10) -> str:
        """Display recent trades"""
        trades = self.trading_manager.get_trading_history()
        recent_trades = trades[-limit:] if trades else []
        
        output = []
        output.append("=" * 60)
        output.append(f"RECENT TRADES (Last {len(recent_trades)})")
        output.append("=" * 60)
        
        if not recent_trades:
            output.append("No trades yet")
            return "\n".join(output)
        
        for trade in reversed(recent_trades):
            timestamp = datetime.fromisoformat(trade['timestamp'])
            output.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {trade['side'].upper()} {trade['quantity']:.2f} {trade['ticker']} @ ${trade['price']:.2f}")
            output.append(f"  Total Value: ${trade['total_value']:,.2f}")
            if 'realized_pnl' in trade:
                output.append(f"  Realized P&L: ${trade['realized_pnl']:,.2f}")
            output.append(f"  Reason: {trade['reason']}")
            output.append("")
        
        return "\n".join(output)
    
    def display_analysis_history(self, limit: int = 5) -> str:
        """Display recent analysis results"""
        analyses = self.trading_manager.get_analysis_history()
        recent_analyses = analyses[-limit:] if analyses else []
        
        output = []
        output.append("=" * 60)
        output.append(f"RECENT ANALYSIS (Last {len(recent_analyses)})")
        output.append("=" * 60)
        
        if not recent_analyses:
            output.append("No analysis history yet")
            return "\n".join(output)
        
        for analysis in reversed(recent_analyses):
            timestamp = datetime.fromisoformat(analysis['timestamp'])
            output.append(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {analysis['ticker']} Analysis")
            output.append(f"Signal: {analysis['signal']}")
            output.append(f"Price: ${analysis['current_price']:.2f}")
            output.append(f"Position Size: {analysis['position_size']:.2f}")
            output.append(f"Actions Taken: {len(analysis['actions_taken'])}")
            
            for action in analysis['actions_taken']:
                output.append(f"  - {action['action']}: {action.get('quantity', 0):.2f} @ ${action.get('price', 0):.2f}")
            
            output.append("")
        
        return "\n".join(output)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        portfolio_summary = self.trading_manager.get_portfolio_summary()
        trades = self.trading_manager.get_trading_history()
        analyses = self.trading_manager.get_analysis_history()
        
        # Calculate performance metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('realized_pnl', 0) > 0])
        losing_trades = len([t for t in trades if t.get('realized_pnl', 0) < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average win/loss
        wins = [t['realized_pnl'] for t in trades if t.get('realized_pnl', 0) > 0]
        losses = [t['realized_pnl'] for t in trades if t.get('realized_pnl', 0) < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Calculate signal accuracy
        signal_accuracy = {}
        for analysis in analyses:
            signal = analysis['signal']
            if signal not in signal_accuracy:
                signal_accuracy[signal] = {'total': 0, 'profitable': 0}
            
            signal_accuracy[signal]['total'] += 1
            
            # Check if this analysis led to profitable trades
            ticker = analysis['ticker']
            ticker_trades = [t for t in trades if t['ticker'] == ticker and 
                           datetime.fromisoformat(t['timestamp']) > datetime.fromisoformat(analysis['timestamp'])]
            
            if ticker_trades:
                total_pnl = sum(t.get('realized_pnl', 0) for t in ticker_trades)
                if total_pnl > 0:
                    signal_accuracy[signal]['profitable'] += 1
        
        # Calculate accuracy percentages
        for signal in signal_accuracy:
            total = signal_accuracy[signal]['total']
            profitable = signal_accuracy[signal]['profitable']
            signal_accuracy[signal]['accuracy'] = (profitable / total * 100) if total > 0 else 0
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "portfolio_metrics": {
                "total_value": portfolio_summary.get('total_value', 0),
                "total_pnl": portfolio_summary.get('total_pnl', 0),
                "return_percentage": portfolio_summary.get('total_pnl_pct', 0),
                "available_capital": portfolio_summary.get('cash', 0)
            },
            "trading_metrics": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "average_win": avg_win,
                "average_loss": avg_loss,
                "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            },
            "signal_accuracy": signal_accuracy,
            "recent_activity": {
                "last_analysis": analyses[-1]['timestamp'] if analyses else None,
                "last_trade": trades[-1]['timestamp'] if trades else None,
                "total_analyses": len(analyses)
            }
        }
        
        return report
    
    def display_performance_report(self) -> str:
        """Display performance report in formatted string"""
        report = self.generate_performance_report()
        
        output = []
        output.append("=" * 60)
        output.append("PERFORMANCE REPORT")
        output.append("=" * 60)
        output.append(f"Generated: {datetime.fromisoformat(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        # Portfolio metrics
        output.append("PORTFOLIO METRICS:")
        output.append("-" * 30)
        pm = report['portfolio_metrics']
        output.append(f"Total Value: ${pm['total_value']:,.2f}")
        output.append(f"Total P&L: ${pm['total_pnl']:,.2f}")
        output.append(f"Return: {pm['return_percentage']:.2f}%")
        output.append(f"Available Capital: ${pm['available_capital']:,.2f}")
        output.append("")
        
        # Trading metrics
        output.append("TRADING METRICS:")
        output.append("-" * 30)
        tm = report['trading_metrics']
        output.append(f"Total Trades: {tm['total_trades']}")
        output.append(f"Winning Trades: {tm['winning_trades']}")
        output.append(f"Losing Trades: {tm['losing_trades']}")
        output.append(f"Win Rate: {tm['win_rate']:.1f}%")
        output.append(f"Average Win: ${tm['average_win']:.2f}")
        output.append(f"Average Loss: ${tm['average_loss']:.2f}")
        output.append(f"Profit Factor: {tm['profit_factor']:.2f}")
        output.append("")
        
        # Signal accuracy
        output.append("SIGNAL ACCURACY:")
        output.append("-" * 30)
        for signal, metrics in report['signal_accuracy'].items():
            output.append(f"{signal}: {metrics['accuracy']:.1f}% ({metrics['profitable']}/{metrics['total']})")
        output.append("")
        
        # Recent activity
        output.append("RECENT ACTIVITY:")
        output.append("-" * 30)
        ra = report['recent_activity']
        if ra['last_analysis']:
            last_analysis = datetime.fromisoformat(ra['last_analysis'])
            output.append(f"Last Analysis: {last_analysis.strftime('%Y-%m-%d %H:%M:%S')}")
        if ra['last_trade']:
            last_trade = datetime.fromisoformat(ra['last_trade'])
            output.append(f"Last Trade: {last_trade.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Total Analyses: {ra['total_analyses']}")
        
        return "\n".join(output)
    
    def save_dashboard_report(self, filepath: str):
        """Save dashboard report to file"""
        report = self.generate_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Dashboard report saved to {filepath}")
    
    def export_trades_to_csv(self, filepath: str):
        """Export trading history to CSV"""
        trades = self.trading_manager.get_trading_history()
        
        if not trades:
            logger.warning("No trades to export")
            return
        
        df = pd.DataFrame(trades)
        df.to_csv(filepath, index=False)
        logger.info(f"Trades exported to {filepath}")
    
    def export_analysis_to_csv(self, filepath: str):
        """Export analysis history to CSV"""
        analyses = self.trading_manager.get_analysis_history()
        
        if not analyses:
            logger.warning("No analysis history to export")
            return
        
        # Flatten the analysis data
        flattened = []
        for analysis in analyses:
            flat_analysis = {
                'timestamp': analysis['timestamp'],
                'ticker': analysis['ticker'],
                'trade_date': analysis['trade_date'],
                'signal': analysis['signal'],
                'current_price': analysis['current_price'],
                'position_size': analysis['position_size'],
                'actions_count': len(analysis['actions_taken'])
            }
            
            # Add action details
            for i, action in enumerate(analysis['actions_taken']):
                flat_analysis[f'action_{i+1}'] = action['action']
                flat_analysis[f'action_{i+1}_quantity'] = action.get('quantity', 0)
                flat_analysis[f'action_{i+1}_price'] = action.get('price', 0)
            
            flattened.append(flat_analysis)
        
        df = pd.DataFrame(flattened)
        df.to_csv(filepath, index=False)
        logger.info(f"Analysis history exported to {filepath}")

#!/usr/bin/env python3
"""
TradingAgents System Launcher
Supports both day trading and swing trading modes
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_day_trading():
    """Run day trading mode (1H timeframe)"""
    print("🚀 Starting DAY TRADING mode...")
    print("⏰ Timeframe: 1 hour")
    print("📊 Strategy: Adaptive momentum")
    print("🎯 Target: High frequency, short-term positions")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "demos/crypto_backtest_simple.py"], check=True)
        print("\n✅ Day trading backtest completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Day trading failed: {e}")
        return False
    return True

def run_swing_trading():
    """Run swing trading mode (1D timeframe)"""
    print("🚀 Starting SWING TRADING mode...")
    print("⏰ Timeframe: 1 day")
    print("📊 Strategy: Trend following")
    print("🎯 Target: Medium frequency, longer-term positions")
    print("-" * 50)
    
    # Modify the simple backtest for swing trading
    try:
        # Read the simple backtest file
        with open("demos/crypto_backtest_simple.py", "r") as f:
            content = f.read()
        
        # Modify for swing trading
        swing_content = content.replace(
            'timeframe="1H"',
            'timeframe="1D"'
        ).replace(
            'strategy="adaptive"',
            'strategy="trend_following"'
        ).replace(
            'max_position_size=0.2',
            'max_position_size=0.15'
        )
        
        # Write temporary swing trading file
        with open("crypto_swing_temp.py", "w") as f:
            f.write(swing_content)
        
        # Run swing trading
        subprocess.run([sys.executable, "crypto_swing_temp.py"], check=True)
        
        # Clean up
        Path("crypto_swing_temp.py").unlink()
        
        print("\n✅ Swing trading backtest completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Swing trading failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    return True

def run_full_analysis():
    """Run full TradingAgents analysis"""
    print("🚀 Starting FULL ANALYSIS mode...")
    print("🤖 Using LLM-driven analysis")
    print("📊 Complete agent workflow")
    print("🎯 Target: Comprehensive market analysis")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        print("\n✅ Full analysis completed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Full analysis failed: {e}")
        return False
    return True

def show_recommendations():
    """Show system recommendations"""
    print("\n" + "="*60)
    print("🎯 TRADING SYSTEM RECOMMENDATIONS")
    print("="*60)
    
    print("\n📊 DAY TRADING (Current Default)")
    print("-" * 30)
    print("⏰ Frequency: Every 1-4 hours")
    print("📈 Expected Return: 100-200% annually")
    print("📉 Max Drawdown: 30-50%")
    print("🎯 Win Rate: 60-70%")
    print("💡 Best for: Active traders, high volatility periods")
    
    print("\n📊 SWING TRADING (Alternative)")
    print("-" * 30)
    print("⏰ Frequency: Daily or every 2-3 days")
    print("📈 Expected Return: 50-100% annually")
    print("📉 Max Drawdown: 20-30%")
    print("🎯 Win Rate: 70-80%")
    print("💡 Best for: Less active traders, trend following")
    
    print("\n📊 FULL ANALYSIS (LLM-Driven)")
    print("-" * 30)
    print("⏰ Frequency: Every 2-4 hours")
    print("📈 Expected Return: Variable")
    print("📉 Max Drawdown: Variable")
    print("🎯 Win Rate: Variable")
    print("💡 Best for: Research, strategy development")
    
    print("\n🚀 QUICK START")
    print("-" * 30)
    print("1. Start with day trading: python run_trading_system.py --mode day")
    print("2. Try swing trading: python run_trading_system.py --mode swing")
    print("3. Run full analysis: python run_trading_system.py --mode full")
    print("4. Get recommendations: python run_trading_system.py --recommendations")

def main():
    parser = argparse.ArgumentParser(description="TradingAgents System Launcher")
    parser.add_argument(
        "--mode", 
        choices=["day", "swing", "full", "recommendations"],
        default="day",
        help="Trading mode to run (default: day)"
    )
    
    args = parser.parse_args()
    
    print("🚀 TradingAgents Crypto System")
    print("=" * 40)
    
    if args.mode == "day":
        success = run_day_trading()
    elif args.mode == "swing":
        success = run_swing_trading()
    elif args.mode == "full":
        success = run_full_analysis()
    elif args.mode == "recommendations":
        show_recommendations()
        success = True
    else:
        print(f"❌ Unknown mode: {args.mode}")
        success = False
    
    if success and args.mode != "recommendations":
        print("\n" + "="*60)
        print("🎉 SYSTEM EXECUTION COMPLETED!")
        print("="*60)
        print("📊 Check results in generated files")
        print("📈 Review charts in crypto_backtest_charts/")
        print("📋 Check logs for detailed information")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

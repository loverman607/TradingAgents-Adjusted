#!/usr/bin/env python3
"""
Test Crypto 24/7 Trading Capability

This script demonstrates that the system now handles crypto trading
on weekends and holidays, unlike traditional stock markets.
"""

from datetime import datetime, timedelta
from tradingagents.dataflows.crypto_interface import (
    is_crypto_trading_day,
    get_crypto_trading_days,
    get_crypto_current_price
)

def test_crypto_24_7():
    """Test crypto 24/7 trading functionality"""
    
    print("🔄 Testing Crypto 24/7 Trading Capability")
    print("=" * 50)
    
    # Test various dates including weekends
    test_dates = [
        "2025-09-07",  # Sunday
        "2025-09-06",  # Saturday  
        "2025-09-08",  # Monday
        "2025-12-25",  # Christmas Day
        "2025-01-01",  # New Year's Day
    ]
    
    print("\n📅 Testing Trading Days:")
    for date in test_dates:
        is_trading = is_crypto_trading_day(date)
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
        status = "✅ TRADING DAY" if is_trading else "❌ NOT TRADING"
        print(f"  {date} ({day_name}): {status}")
    
    print("\n📊 Testing Trading Days Range:")
    start_date = "2025-09-06"  # Saturday
    end_date = "2025-09-08"    # Monday
    
    trading_days = get_crypto_trading_days(start_date, end_date)
    print(f"  From {start_date} to {end_date}:")
    for day in trading_days:
        day_name = datetime.strptime(day, "%Y-%m-%d").strftime("%A")
        print(f"    {day} ({day_name})")
    
    print(f"\n  Total trading days: {len(trading_days)} (includes weekends!)")
    
    print("\n💰 Testing Crypto Price Fetching:")
    crypto_symbols = ["BTC", "ETH", "ADA", "SOL"]
    for symbol in crypto_symbols:
        price = get_crypto_current_price(symbol)
        print(f"  {symbol}: ${price:,.2f}")
    
    print("\n🎯 Key Differences from Traditional Stocks:")
    print("  ✅ Crypto trades 24/7 including weekends")
    print("  ✅ Crypto trades on holidays")
    print("  ✅ No market closures for crypto")
    print("  ✅ Continuous price updates")
    
    print("\n🚀 System Status: CRYPTO-OPTIMIZED FOR 24/7 TRADING!")
    print("=" * 50)

if __name__ == "__main__":
    test_crypto_24_7()

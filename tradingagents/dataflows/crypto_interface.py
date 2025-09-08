"""
Crypto-Optimized Data Interface

This module provides crypto-specific data handling that accounts for 24/7 trading,
replacing traditional stock market weekend/holiday restrictions.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Annotated
from .crypto_stockstats_utils import CryptoStockstatsUtils

def get_crypto_stockstats_indicators_report_online(
    symbol: Annotated[str, "crypto ticker symbol (e.g., BTC, ETH)"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "current date in YYYY-mm-dd format"],
) -> str:
    """
    Get crypto technical indicators report with 24/7 support.
    This replaces the traditional stock market version that excludes weekends.
    """
    try:
        # For crypto, we can analyze any date including weekends
        data_dir = os.path.join(os.getcwd(), "tradingagents", "dataflows", "data_cache")
        
        # Get the indicator value for the specific date
        indicator_value = CryptoStockstatsUtils.get_crypto_stock_stats(
            symbol=symbol,
            indicator=indicator,
            curr_date=curr_date,
            data_dir=data_dir,
            online=True
        )
        
        # Format the response
        if "Error" in str(indicator_value) or "not found" in str(indicator_value):
            return f"## {indicator} value for {symbol} on {curr_date}:\n\n{curr_date}: N/A: {indicator_value}\n\nCrypto trades 24/7, but data may not be available for this specific date."
        else:
            return f"## {indicator} value for {symbol} on {curr_date}:\n\n{curr_date}: {indicator_value}\n\nCrypto trades 24/7, including weekends and holidays."
            
    except Exception as e:
        return f"## {indicator} value for {symbol} on {curr_date}:\n\n{curr_date}: N/A: Error fetching crypto data - {str(e)}\n\nNote: Crypto markets trade 24/7."

def get_crypto_stockstats_indicators_window(
    symbol: Annotated[str, "crypto ticker symbol"],
    indicator: Annotated[str, "technical indicator name"],
    curr_date: Annotated[str, "current date in YYYY-mm-dd format"],
    look_back_days: Annotated[int, "number of days to look back"],
) -> str:
    """
    Get crypto indicators for a window of days with 24/7 support.
    This includes weekends and holidays for crypto markets.
    """
    try:
        data_dir = os.path.join(os.getcwd(), "tradingagents", "dataflows", "data_cache")
        
        return CryptoStockstatsUtils.get_crypto_indicators_window(
            symbol=symbol,
            indicator=indicator,
            curr_date=curr_date,
            look_back_days=look_back_days,
            data_dir=data_dir
        )
        
    except Exception as e:
        return f"Error generating crypto indicators window: {str(e)}"

def is_crypto_trading_day(date_str: str) -> bool:
    """Check if a date is a valid crypto trading day (always True for crypto)"""
    return CryptoStockstatsUtils.is_crypto_trading_day(date_str)

def get_crypto_trading_days(start_date: str, end_date: str) -> list:
    """Get all crypto trading days between start and end dates (includes weekends)"""
    return CryptoStockstatsUtils.get_crypto_trading_days(start_date, end_date)

def get_crypto_data_online(
    symbol: Annotated[str, "crypto ticker symbol"],
    start_date: Annotated[str, "start date in YYYY-mm-dd format"],
    end_date: Annotated[str, "end date in YYYY-mm-dd format"],
) -> str:
    """
    Get crypto data online with 24/7 support.
    This handles weekends and holidays as valid trading days.
    """
    try:
        # For crypto, we can get data for any date range including weekends
        # This is a simplified version - in production, you'd use crypto APIs like CoinGecko, Binance, etc.
        
        # Generate a message indicating crypto 24/7 trading
        start_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate the number of days
        days_diff = (end_obj - start_obj).days + 1
        
        # Check if the range includes weekends
        weekend_days = 0
        current = start_obj
        while current <= end_obj:
            if current.weekday() >= 5:  # Saturday = 5, Sunday = 6
                weekend_days += 1
            current += timedelta(days=1)
        
        message = f"""# Crypto data for {symbol} from {start_date} to {end_date}
# Total days: {days_diff} (including {weekend_days} weekend days)
# Crypto markets trade 24/7, including weekends and holidays
# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Note: This is crypto data with 24/7 trading support.
Unlike traditional stock markets, crypto trades continuously including weekends.
"""
        
        return message
        
    except Exception as e:
        return f"Error fetching crypto data: {str(e)}"

def get_crypto_current_price(symbol: str) -> float:
    """Get current crypto price (placeholder for real API integration)"""
    try:
        # This is a placeholder - in production, you'd integrate with:
        # - CoinGecko API
        # - Binance API
        # - Coinbase API
        # - CryptoCompare API
        
        # For now, return a reasonable default based on symbol
        default_prices = {
            "BTC": 65000.0,
            "ETH": 3200.0,
            "ADA": 0.45,
            "SOL": 100.0
        }
        
        return default_prices.get(symbol.upper(), 100.0)
        
    except Exception as e:
        return 0.0

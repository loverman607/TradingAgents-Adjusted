"""
Crypto-Optimized StockStats Utils

This module provides crypto-specific technical indicator calculations
that account for 24/7 trading markets (no weekends/holidays).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Annotated
import os
from pathlib import Path

class CryptoStockstatsUtils:
    """Crypto-optimized technical indicators for 24/7 markets"""
    
    @staticmethod
    def get_crypto_stock_stats(
        symbol: Annotated[str, "crypto ticker symbol (e.g., BTC, ETH)"],
        indicator: Annotated[str, "technical indicator name"],
        curr_date: Annotated[str, "current date in YYYY-mm-dd format"],
        data_dir: Annotated[str, "directory where crypto data is stored"],
        online: Annotated[bool, "whether to fetch data online"] = False,
    ):
        """
        Get crypto technical indicators with 24/7 support.
        Unlike traditional stocks, crypto trades 24/7 including weekends.
        """
        
        if online:
            return CryptoStockstatsUtils._get_online_crypto_data(symbol, indicator, curr_date)
        else:
            return CryptoStockstatsUtils._get_offline_crypto_data(symbol, indicator, curr_date, data_dir)
    
    @staticmethod
    def _get_online_crypto_data(symbol: str, indicator: str, curr_date: str):
        """Get crypto data online with 24/7 support"""
        try:
            # For crypto, we can get data for any date including weekends
            # This is a simplified version - in production, you'd use crypto APIs
            
            # Check if it's a weekend
            date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
            is_weekend = date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6
            
            if is_weekend:
                # For crypto, weekends are valid trading days
                return f"Crypto trades 24/7 - {curr_date} is a valid trading day for {symbol}"
            else:
                return f"Regular trading day for {symbol} on {curr_date}"
                
        except Exception as e:
            return f"Error fetching crypto data: {str(e)}"
    
    @staticmethod
    def _get_offline_crypto_data(symbol: str, indicator: str, curr_date: str, data_dir: str):
        """Get crypto data from offline sources with 24/7 support"""
        try:
            # Look for crypto data files
            crypto_data_files = [
                f"{symbol}-YFin-data-2010-09-07-2025-09-07.csv",
                f"{symbol}-crypto-data.csv",
                f"{symbol}-24h-data.csv"
            ]
            
            data_file = None
            for file in crypto_data_files:
                file_path = os.path.join(data_dir, "data_cache", file)
                if os.path.exists(file_path):
                    data_file = file_path
                    break
            
            if not data_file:
                # Fallback to regular data directory
                for file in crypto_data_files:
                    file_path = os.path.join(data_dir, file)
                    if os.path.exists(file_path):
                        data_file = file_path
                        break
            
            if not data_file:
                return f"No crypto data file found for {symbol}"
            
            # Read the data
            df = pd.read_csv(data_file)
            
            # Ensure Date column exists and is properly formatted
            if 'Date' not in df.columns:
                return f"No Date column found in {data_file}"
            
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # For crypto, we need to handle 24/7 data
            # Look for the exact date or the most recent available date
            target_date = pd.to_datetime(curr_date)
            
            # Find exact match first
            exact_match = df[df['Date'].dt.date == target_date.date()]
            
            if not exact_match.empty:
                # Found exact date match
                if indicator in exact_match.columns:
                    indicator_value = exact_match[indicator].iloc[0]
                    return indicator_value
                else:
                    return f"Indicator '{indicator}' not found in data"
            else:
                # For crypto, if exact date not found, get the most recent data
                # This handles cases where data might be missing for specific dates
                recent_data = df[df['Date'] <= target_date].tail(1)
                
                if not recent_data.empty:
                    if indicator in recent_data.columns:
                        indicator_value = recent_data[indicator].iloc[0]
                        return f"{indicator_value} (from {recent_data['Date'].iloc[0].strftime('%Y-%m-%d')})"
                    else:
                        return f"Indicator '{indicator}' not found in recent data"
                else:
                    return f"No data available for {curr_date} or earlier dates"
                    
        except Exception as e:
            return f"Error processing crypto data: {str(e)}"
    
    @staticmethod
    def is_crypto_trading_day(date_str: str) -> bool:
        """Check if a date is a valid crypto trading day (always True for crypto)"""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            # Crypto trades 24/7, so every day is a trading day
            return True
        except:
            return False
    
    @staticmethod
    def get_crypto_trading_days(start_date: str, end_date: str) -> list:
        """Get all crypto trading days between start and end dates (includes weekends)"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            trading_days = []
            current = start
            
            while current <= end:
                trading_days.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            
            return trading_days
        except:
            return []
    
    @staticmethod
    def get_crypto_indicators_window(
        symbol: str,
        indicator: str,
        curr_date: str,
        look_back_days: int,
        data_dir: str
    ) -> str:
        """Get crypto indicators for a window of days (includes weekends)"""
        try:
            from dateutil.relativedelta import relativedelta
            
            curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
            before = curr_date_obj - relativedelta(days=look_back_days)
            
            # Get all trading days (including weekends for crypto)
            trading_days = CryptoStockstatsUtils.get_crypto_trading_days(
                before.strftime("%Y-%m-%d"),
                curr_date
            )
            
            ind_string = ""
            for trading_day in reversed(trading_days):  # Most recent first
                indicator_value = CryptoStockstatsUtils.get_crypto_stock_stats(
                    symbol, indicator, trading_day, data_dir, online=False
                )
                
                # Format the output
                if "Error" in str(indicator_value) or "not found" in str(indicator_value):
                    ind_string += f"{trading_day}: N/A: {indicator_value}\n"
                else:
                    ind_string += f"{trading_day}: {indicator_value}\n"
            
            # Get indicator description
            best_ind_params = {
                "close_50_sma": "50-day Simple Moving Average: Shows medium-term trend. Usage: Price above SMA = bullish, below = bearish.",
                "close_200_sma": "200-day Simple Moving Average: Shows long-term trend. Usage: Price above SMA = long-term bullish trend.",
                "close_10_ema": "10-day Exponential Moving Average: Shows short-term momentum. Usage: More responsive than SMA for short-term signals.",
                "macd": "MACD (Moving Average Convergence Divergence): Shows momentum changes. Usage: MACD above signal line = bullish, below = bearish.",
                "rsi": "RSI (Relative Strength Index): Shows overbought/oversold conditions. Usage: RSI > 70 = overbought, RSI < 30 = oversold.",
                "atr": "ATR (Average True Range): Shows volatility. Usage: Higher ATR = more volatile, use for stop-loss positioning.",
                "boll_ub": "Bollinger Upper Band: Shows potential resistance. Usage: Price touching upper band may indicate overbought conditions.",
                "boll_lb": "Bollinger Lower Band: Shows potential support. Usage: Price touching lower band may indicate oversold conditions."
            }
            
            result_str = (
                f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date} (Crypto 24/7):\n\n"
                + ind_string
                + "\n\n"
                + best_ind_params.get(indicator, "No description available.")
            )
            
            return result_str
            
        except Exception as e:
            return f"Error generating crypto indicators window: {str(e)}"

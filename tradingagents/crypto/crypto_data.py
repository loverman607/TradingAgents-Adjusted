"""
Crypto Market Data Provider

This module provides real-time and historical cryptocurrency data integration
optimized for BTC and ETH trading.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CryptoDataProvider:
    """Crypto market data provider with multiple sources"""
    
    def __init__(self, cache_dir: str = "crypto_data_cache"):
        """
        Initialize crypto data provider
        
        Args:
            cache_dir: Directory to cache data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # API endpoints and configurations
        self.apis = {
            "coinbase": {
                "base_url": "https://api.exchange.coinbase.com",
                "rate_limit": 10,  # requests per second
                "last_request": 0
            },
            "binance": {
                "base_url": "https://api.binance.com",
                "rate_limit": 10,
                "last_request": 0
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com",
                "rate_limit": 1,
                "last_request": 0,
                "api_key": None  # Set your API key
            }
        }
        
        # Crypto symbols mapping
        self.crypto_symbols = {
            "BTC": {
                "coinbase": "BTC-USD",
                "binance": "BTCUSDT",
                "coinmarketcap": "1"
            },
            "ETH": {
                "coinbase": "ETH-USD",
                "binance": "ETHUSDT",
                "coinmarketcap": "1027"
            }
        }
        
        # Cache settings
        self.cache_duration = {
            "realtime": 60,  # 1 minute
            "historical": 3600,  # 1 hour
            "fundamental": 86400  # 24 hours
        }
    
    def _rate_limit(self, api_name: str):
        """Apply rate limiting for API calls"""
        api_config = self.apis[api_name]
        current_time = time.time()
        time_since_last = current_time - api_config["last_request"]
        
        if time_since_last < 1.0 / api_config["rate_limit"]:
            sleep_time = (1.0 / api_config["rate_limit"]) - time_since_last
            time.sleep(sleep_time)
        
        api_config["last_request"] = time.time()
    
    def _get_cached_data(self, cache_key: str, max_age: int) -> Optional[Dict]:
        """Get cached data if not expired"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is still valid
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > max_age:
            return None
        
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def _save_cached_data(self, cache_key: str, data: Dict):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def get_realtime_price(self, ticker: str, source: str = "coinbase") -> Dict[str, Any]:
        """
        Get real-time crypto price
        
        Args:
            ticker: Crypto ticker (BTC, ETH)
            source: Data source (coinbase, binance)
            
        Returns:
            Real-time price data
        """
        cache_key = f"realtime_{ticker}_{source}"
        cached_data = self._get_cached_data(cache_key, self.cache_duration["realtime"])
        
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit(source)
            
            if source == "coinbase":
                data = self._get_coinbase_price(ticker)
            elif source == "binance":
                data = self._get_binance_price(ticker)
            else:
                raise ValueError(f"Unsupported source: {source}")
            
            # Cache the data
            self._save_cached_data(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get real-time price for {ticker} from {source}: {e}")
            return self._get_fallback_price(ticker)
    
    def _get_coinbase_price(self, ticker: str) -> Dict[str, Any]:
        """Get price from Coinbase API"""
        symbol = self.crypto_symbols[ticker]["coinbase"]
        url = f"{self.apis['coinbase']['base_url']}/products/{symbol}/ticker"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            "ticker": ticker,
            "price": float(data["price"]),
            "bid": float(data["bid"]),
            "ask": float(data["ask"]),
            "volume": float(data["volume"]),
            "timestamp": datetime.now().isoformat(),
            "source": "coinbase"
        }
    
    def _get_binance_price(self, ticker: str) -> Dict[str, Any]:
        """Get price from Binance API"""
        symbol = self.crypto_symbols[ticker]["binance"]
        url = f"{self.apis['binance']['base_url']}/api/v3/ticker/24hr"
        
        params = {"symbol": symbol}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            "ticker": ticker,
            "price": float(data["lastPrice"]),
            "bid": float(data["bidPrice"]),
            "ask": float(data["askPrice"]),
            "volume": float(data["volume"]),
            "high_24h": float(data["highPrice"]),
            "low_24h": float(data["lowPrice"]),
            "change_24h": float(data["priceChangePercent"]),
            "timestamp": datetime.now().isoformat(),
            "source": "binance"
        }
    
    def _get_fallback_price(self, ticker: str) -> Dict[str, Any]:
        """Get fallback price when APIs fail"""
        # Use mock data as fallback
        mock_prices = {
            "BTC": 50000.0,
            "ETH": 3000.0
        }
        
        return {
            "ticker": ticker,
            "price": mock_prices.get(ticker, 100.0),
            "bid": mock_prices.get(ticker, 100.0) * 0.999,
            "ask": mock_prices.get(ticker, 100.0) * 1.001,
            "volume": 1000000.0,
            "timestamp": datetime.now().isoformat(),
            "source": "fallback"
        }
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: str, 
                          interval: str = "1d", source: str = "coinbase") -> pd.DataFrame:
        """
        Get historical crypto data
        
        Args:
            ticker: Crypto ticker (BTC, ETH)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 1h, 1d)
            source: Data source (coinbase, binance)
            
        Returns:
            Historical OHLCV data
        """
        cache_key = f"historical_{ticker}_{start_date}_{end_date}_{interval}_{source}"
        cached_data = self._get_cached_data(cache_key, self.cache_duration["historical"])
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            self._rate_limit(source)
            
            if source == "coinbase":
                data = self._get_coinbase_historical(ticker, start_date, end_date, interval)
            elif source == "binance":
                data = self._get_binance_historical(ticker, start_date, end_date, interval)
            else:
                raise ValueError(f"Unsupported source: {source}")
            
            # Cache the data
            self._save_cached_data(cache_key, data.to_dict('records'))
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {ticker} from {source}: {e}")
            return self._get_fallback_historical(ticker, start_date, end_date)
    
    def _get_coinbase_historical(self, ticker: str, start_date: str, end_date: str, 
                                interval: str) -> pd.DataFrame:
        """Get historical data from Coinbase"""
        symbol = self.crypto_symbols[ticker]["coinbase"]
        
        # Convert interval to Coinbase format
        interval_map = {
            "1m": "60",
            "5m": "300",
            "1h": "3600",
            "1d": "86400"
        }
        
        granularity = interval_map.get(interval, "86400")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        
        url = f"{self.apis['coinbase']['base_url']}/products/{symbol}/candles"
        params = {
            "start": start_ts,
            "end": end_ts,
            "granularity": granularity
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('date')
        df = df.sort_index()
        
        # Select and rename columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        return df
    
    def _get_binance_historical(self, ticker: str, start_date: str, end_date: str, 
                               interval: str) -> pd.DataFrame:
        """Get historical data from Binance"""
        symbol = self.crypto_symbols[ticker]["binance"]
        
        # Convert interval to Binance format
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "1h": "1h",
            "1d": "1d"
        }
        
        binance_interval = interval_map.get(interval, "1d")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        url = f"{self.apis['binance']['base_url']}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": binance_interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date')
        df = df.sort_index()
        
        # Select and convert columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    def _get_fallback_historical(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get fallback historical data when APIs fail"""
        # Generate mock historical data
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Generate realistic price movements
        np.random.seed(42)
        
        if ticker == "BTC":
            initial_price = 45000.0
            daily_volatility = 0.03
        elif ticker == "ETH":
            initial_price = 3000.0
            daily_volatility = 0.04
        else:
            initial_price = 100.0
            daily_volatility = 0.05
        
        # Generate price series
        returns = np.random.normal(0.0005, daily_volatility, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = daily_volatility * 0.5
            
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            open_price = prices[i-1] if i > 0 else price
            
            # Ensure OHLC relationships
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            volume = 1000000 * (1 + np.random.uniform(-0.5, 0.5))
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': int(volume)
            })
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        return df
    
    def get_crypto_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """
        Get crypto fundamental data
        
        Args:
            ticker: Crypto ticker (BTC, ETH)
            
        Returns:
            Fundamental data
        """
        cache_key = f"fundamentals_{ticker}"
        cached_data = self._get_cached_data(cache_key, self.cache_duration["fundamental"])
        
        if cached_data:
            return cached_data
        
        try:
            # This would integrate with CoinMarketCap API for real data
            # For now, return mock fundamental data
            fundamentals = self._get_mock_fundamentals(ticker)
            
            # Cache the data
            self._save_cached_data(cache_key, fundamentals)
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Failed to get fundamentals for {ticker}: {e}")
            return self._get_mock_fundamentals(ticker)
    
    def _get_mock_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Get mock fundamental data"""
        if ticker == "BTC":
            return {
                "ticker": "BTC",
                "name": "Bitcoin",
                "market_cap": 1000000000000,  # $1T
                "circulating_supply": 19000000,
                "max_supply": 21000000,
                "total_supply": 19000000,
                "market_cap_rank": 1,
                "price_change_24h": 2.5,
                "price_change_7d": 5.2,
                "price_change_30d": 15.8,
                "volume_24h": 25000000000,
                "market_cap_dominance": 45.2,
                "timestamp": datetime.now().isoformat()
            }
        elif ticker == "ETH":
            return {
                "ticker": "ETH",
                "name": "Ethereum",
                "market_cap": 400000000000,  # $400B
                "circulating_supply": 120000000,
                "max_supply": None,
                "total_supply": 120000000,
                "market_cap_rank": 2,
                "price_change_24h": 1.8,
                "price_change_7d": 3.5,
                "price_change_30d": 12.3,
                "volume_24h": 15000000000,
                "market_cap_dominance": 18.5,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "ticker": ticker,
                "name": ticker,
                "market_cap": 1000000000,
                "circulating_supply": 1000000000,
                "max_supply": 1000000000,
                "total_supply": 1000000000,
                "market_cap_rank": 100,
                "price_change_24h": 0.0,
                "price_change_7d": 0.0,
                "price_change_30d": 0.0,
                "volume_24h": 100000000,
                "market_cap_dominance": 0.1,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_market_sentiment(self, ticker: str) -> Dict[str, Any]:
        """
        Get crypto market sentiment data
        
        Args:
            ticker: Crypto ticker (BTC, ETH)
            
        Returns:
            Sentiment data
        """
        # This would integrate with sentiment analysis APIs
        # For now, return mock sentiment data
        return {
            "ticker": ticker,
            "fear_greed_index": np.random.randint(20, 80),
            "social_sentiment": np.random.uniform(0.3, 0.7),
            "news_sentiment": np.random.uniform(0.4, 0.6),
            "reddit_sentiment": np.random.uniform(0.2, 0.8),
            "twitter_sentiment": np.random.uniform(0.3, 0.7),
            "overall_sentiment": np.random.uniform(0.3, 0.7),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_crypto_data_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive crypto data summary
        
        Args:
            ticker: Crypto ticker (BTC, ETH)
            
        Returns:
            Comprehensive data summary
        """
        try:
            # Get real-time price
            price_data = self.get_realtime_price(ticker)
            
            # Get historical data (last 30 days)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            historical_data = self.get_historical_data(ticker, start_date, end_date)
            
            # Get fundamentals
            fundamentals = self.get_crypto_fundamentals(ticker)
            
            # Get sentiment
            sentiment = self.get_market_sentiment(ticker)
            
            # Calculate additional metrics
            if not historical_data.empty:
                price_change_24h = ((price_data["price"] - historical_data["close"].iloc[-2]) / 
                                   historical_data["close"].iloc[-2] * 100) if len(historical_data) > 1 else 0
                
                volatility_30d = historical_data["close"].pct_change().std() * np.sqrt(365) * 100
                
                # Technical indicators
                sma_20 = historical_data["close"].rolling(20).mean().iloc[-1]
                sma_50 = historical_data["close"].rolling(50).mean().iloc[-1] if len(historical_data) >= 50 else sma_20
                
                rsi = self._calculate_rsi(historical_data["close"])
                
                technical_indicators = {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "rsi": rsi.iloc[-1] if not rsi.empty else 50,
                    "volatility_30d": volatility_30d
                }
            else:
                price_change_24h = 0
                technical_indicators = {}
            
            return {
                "ticker": ticker,
                "price_data": price_data,
                "price_change_24h": price_change_24h,
                "fundamentals": fundamentals,
                "sentiment": sentiment,
                "technical_indicators": technical_indicators,
                "historical_data_points": len(historical_data),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get data summary for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

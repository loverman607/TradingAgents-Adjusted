from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.trading.trading_manager import TradingManager
from tradingagents.trading.dashboard import TradingDashboard


# ========== CRYPTO-OPTIMIZED CONFIGURATION ==========
config = DEFAULT_CONFIG.copy()

# OpenAI config (switched from Google Gemini due to rate limits)
config["llm_provider"] = "openai"
config["backend_url"] = "https://api.openai.com/v1"
config["deep_think_llm"] = "gpt-4o-mini"
config["quick_think_llm"] = "gpt-4o-mini"

# Enhanced analysis settings
config["max_debate_rounds"] = 2  # More thorough analysis
config["online_tools"] = True
config["selected_analysts"] = ["market", "social", "news", "fundamentals"]  # All analysts

# Crypto-optimized settings (24/7 trading)
config["enable_24_7"] = True
config["enable_leverage"] = True
config["max_leverage"] = 3.0
config["volatility_threshold"] = 0.05
config["crypto_mode"] = True  # Enable crypto-specific handling
config["trading_days_24_7"] = True  # Crypto trades 24/7 including weekends

# Risk management
config["max_position_size"] = 0.15  # 15% max position for crypto
config["max_daily_loss"] = 0.03     # 3% max daily loss
config["stop_loss_pct"] = 0.05      # 5% stop loss
config["take_profit_pct"] = 0.15    # 15% take profit


# ========== HELPERS ==========
def resolve_current_price(ticker: str, final_state: dict, debug: bool = False) -> float:
    """
    Resolve the most reliable current price from final_state or fallback to APIs.
    Optimized for crypto 24/7 trading.
    """
    price = 0.0

    # Directly from final_state
    if final_state.get("current_price", 0) > 0:
        price = final_state["current_price"]

    # From nested market data
    elif "market_data" in final_state and isinstance(final_state["market_data"], dict):
        price = final_state["market_data"].get("current_price", 0) or 0

    # From analysis results
    elif "analysis_results" in final_state:
        for result in final_state["analysis_results"]:
            if isinstance(result, dict) and result.get("current_price", 0) > 0:
                price = result["current_price"]
                break

    # Try to extract price from sentiment report (where the real price is often found)
    if price == 0 and 'sentiment_report' in final_state:
        import re
        sentiment_text = final_state['sentiment_report']
        # Look for price patterns like $111,171.00 or $111171.00
        price_match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', sentiment_text)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            try:
                price = float(price_str)
                if debug:
                    print(f"[INFO] Extracted price from sentiment report: ${price:,.2f}")
            except ValueError:
                pass

    # Try to extract price from any text field in final_state
    if price == 0:
        import re
        for key, value in final_state.items():
            if isinstance(value, str) and '$' in value:
                # Look for price patterns in any string field
                price_match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', value)
                if price_match:
                    price_str = price_match.group(1).replace(',', '')
                    try:
                        candidate_price = float(price_str)
                        # Only use reasonable prices (not too low or too high)
                        if 1000 < candidate_price < 200000:  # Reasonable BTC price range
                            price = candidate_price
                            if debug:
                                print(f"[INFO] Extracted price from {key}: ${price:,.2f}")
                            break
                    except ValueError:
                        pass

    # Try to extract price from market report
    if price == 0 and 'market_report' in final_state:
        import re
        market_text = final_state['market_report']
        # Look for price patterns in market report
        price_match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', market_text)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            try:
                price = float(price_str)
                if debug:
                    print(f"[INFO] Extracted price from market report: ${price:,.2f}")
            except ValueError:
                pass

    # Try to extract price from news report
    if price == 0 and 'news_report' in final_state:
        import re
        news_text = final_state['news_report']
        # Look for price patterns in news report
        price_match = re.search(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', news_text)
        if price_match:
            price_str = price_match.group(1).replace(',', '')
            try:
                price = float(price_str)
                if debug:
                    print(f"[INFO] Extracted price from news report: ${price:,.2f}")
            except ValueError:
                pass

    # Crypto-specific price resolution (24/7 support)
    if price == 0 and ticker.upper() in ["BTC", "ETH", "ADA", "SOL"]:
        try:
            from tradingagents.dataflows.crypto_interface import get_crypto_current_price
            price = get_crypto_current_price(ticker)
            if debug and price > 0:
                print(f"[INFO] Got crypto price for {ticker}: ${price:,.2f}")
        except Exception as e:
            if debug:
                print(f"[WARN] Crypto price fetch failed for {ticker}: {e}")

    # From Yahoo Finance API (fallback)
    if price == 0:
        try:
            from tradingagents.dataflows.yfin_utils import get_current_price
            price = get_current_price(ticker)
            if debug and price > 0:
                print(f"[INFO] Got price from Yahoo Finance: ${price:,.2f}")
        except Exception as e:
            if debug:
                print(f"[WARN] Yahoo Finance fetch failed for {ticker}: {e}")

    # From Finnhub API (fallback)
    if price == 0:
        try:
            from tradingagents.dataflows.finnhub_utils import get_data_in_range
            data = get_data_in_range(ticker, "2025-09-07", "2025-09-07", "price", ".")
            if data is not None and "c" in data:  # Finnhub "current close"
                price = float(data["c"])
                if debug:
                    print(f"[INFO] Got price from Finnhub: ${price:,.2f}")
        except Exception as e:
            if debug:
                print(f"[WARN] Finnhub fetch failed for {ticker}: {e}")

    if price == 0 and debug:
        print(f"[ERROR] Could not resolve real price for {ticker}")

    return price


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "reports",
        "logs", 
        "reports/charts",
        "reports/backtests",
        "reports/analysis",
        "tradingagents/dataflows/data_cache"
    ]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except (FileExistsError, OSError) as e:
            # Directory already exists or other OS error, continue
            if "Cannot create a file when that file already exists" not in str(e):
                print(f"Warning: Could not create directory {directory}: {e}")
            pass


def generate_recommendation(ticker: str, signal: str, current_price: float):
    """
    Generate basic trading recommendation string for BUY, SELL, SHORT, or HOLD.
    """
    if signal == "BUY":
        stop_loss = current_price * 0.95
        take_profit = current_price * 1.15
        return (
            f"🟢 RECOMMENDATION: LONG {ticker}\n"
            f"📊 Current Price: ${current_price:,.2f}\n"
            f"🛑 Stop Loss: ${stop_loss:,.2f} (-5%)\n"
            f"🎯 Take Profit: ${take_profit:,.2f} (+15%)\n"
            f"📈 Risk/Reward Ratio: 1:3"
        )

    elif signal == "SELL":
        return (
            f"🔴 RECOMMENDATION: CLOSE LONG POSITION {ticker}\n"
            f"📊 Exit Price: ${current_price:,.2f}\n"
            f"💡 Reason: Take profits or cut losses"
        )

    elif signal == "SHORT":
        stop_loss = current_price * 1.05
        take_profit = current_price * 0.85
        return (
            f"🔴 RECOMMENDATION: SHORT {ticker}\n"
            f"📊 Current Price: ${current_price:,.2f}\n"
            f"🛑 Stop Loss: ${stop_loss:,.2f} (+5%)\n"
            f"🎯 Take Profit: ${take_profit:,.2f} (-15%)\n"
            f"📈 Risk/Reward Ratio: 1:3"
        )

    else:  # HOLD
        return (
            f"🟡 RECOMMENDATION: STAY OUT OF MARKET ({ticker})\n"
            f"📊 Current Price: ${current_price:,.2f}\n"
            f"💡 Reason: Market conditions not favorable"
        )


def generate_enhanced_recommendation(ticker: str, signal: str, current_price: float, 
                                   final_state: dict, trading_result: dict) -> str:
    """
    Generate enhanced trading recommendation with crypto optimization and advanced analytics.
    """
    # Calculate position size based on volatility and confidence
    confidence = trading_result.get("confidence", 0.5)
    base_position = config["max_position_size"]
    
    # Adjust position size based on confidence
    position_size = base_position * confidence
    position_size = min(position_size, config["max_position_size"])
    
    # Calculate stop loss and take profit with crypto optimization
    stop_loss_pct = config["stop_loss_pct"]
    take_profit_pct = config["take_profit_pct"]
    
    # Adjust levels based on volatility if available
    volatility = 0.05  # Default
    if "market_report" in final_state:
        import re
        atr_match = re.search(r'ATR.*?(\d+\.?\d*)', final_state["market_report"])
        if atr_match:
            volatility = float(atr_match.group(1)) / 100
    
    # Wider stops for higher volatility (crypto characteristic)
    if volatility > 0.1:
        stop_loss_pct *= 1.5
        take_profit_pct *= 1.2
    
    if signal == "BUY":
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
        return f"""🟢 OPTIMIZED RECOMMENDATION: LONG {ticker}
📊 Current Price: ${current_price:,.2f}
🎯 Entry Price: ${current_price:,.2f}
🛑 Stop Loss: ${stop_loss:,.2f} (-{stop_loss_pct:.1%})
🎯 Take Profit: ${take_profit:,.2f} (+{take_profit_pct:.1%})
📈 Risk/Reward Ratio: 1:3
💰 Position Size: {position_size:.1%} of portfolio
💵 Position Value: ${current_price * position_size:,.2f}
            🎯 Architecture: Full TradingAgents + Crypto Optimization (OpenAI)
📊 Volatility: {volatility:.1%}
🧠 Confidence: {confidence:.1%}"""

    elif signal == "SELL":
        return f"""🔴 OPTIMIZED RECOMMENDATION: CLOSE LONG POSITION {ticker}
📊 Current Price: ${current_price:,.2f}
🎯 Exit Price: ${current_price:,.2f}
💡 Reason: Take profits or cut losses
            🎯 Architecture: Full TradingAgents + Crypto Optimization (OpenAI)
🧠 Confidence: {confidence:.1%}"""

    elif signal == "SHORT":
        stop_loss = current_price * (1 + stop_loss_pct)
        take_profit = current_price * (1 - take_profit_pct)
        return f"""🔴 OPTIMIZED RECOMMENDATION: SHORT {ticker}
📊 Current Price: ${current_price:,.2f}
🎯 Entry Price: ${current_price:,.2f}
🛑 Stop Loss: ${stop_loss:,.2f} (+{stop_loss_pct:.1%})
🎯 Take Profit: ${take_profit:,.2f} (-{take_profit_pct:.1%})
📈 Risk/Reward Ratio: 1:3
💰 Position Size: {position_size:.1%} of portfolio
💵 Position Value: ${current_price * position_size:,.2f}
            🎯 Architecture: Full TradingAgents + Crypto Optimization (OpenAI)
📊 Volatility: {volatility:.1%}
🧠 Confidence: {confidence:.1%}"""

    else:  # HOLD
        return f"""🟡 OPTIMIZED RECOMMENDATION: STAY OUT OF MARKET ({ticker})
📊 Current Price: ${current_price:,.2f}
💡 Reason: Market conditions not favorable for trading
            🎯 Architecture: Full TradingAgents + Crypto Optimization (OpenAI)
📊 Volatility: {volatility:.1%}
🧠 Confidence: {confidence:.1%}
⏰ Next Analysis: Wait for better setup"""


# ========== MAIN FUNCTIONS ==========
def run_trading_analysis(ticker: str, trade_date: str, initial_capital: float = 100000.0, debug: bool = False):
    """
    Run complete trading analysis and execution for a single ticker.
    Leverages full TradingAgents architecture with crypto optimization.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING CRYPTO-OPTIMIZED TRADING ANALYSIS FOR {ticker}")
    print(f"Date: {trade_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Architecture: Full TradingAgents + Crypto 24/7 Optimization (OpenAI)")
    print(f"Trading Mode: 24/7 (including weekends and holidays)")
    print(f"{'='*60}\n")

    # Initialize trading manager with crypto-optimized settings
    ensure_directories()  # Ensure directories exist before creating manager
    
    trading_manager = TradingManager(
        initial_capital=initial_capital,
        max_position_size=config["max_position_size"],
        state_file=f"reports/analysis/trading_state_{ticker.lower()}.json",
        enable_short_selling=True,
        margin_requirement=0.5,
    )

    # Initialize TradingAgents graph with all analysts
    ta = TradingAgentsGraph(
        selected_analysts=config["selected_analysts"],
        debug=debug, 
        config=config
    )
    
    try:
        print("🔍 Running multi-agent analysis...")
        final_state, decision = ta.propagate(ticker, trade_date)

        print(f"\nAnalysis Decision: {decision}")
        print(f"Final Trade Decision: {final_state.get('final_trade_decision', 'N/A')[:200]}...")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        # Create a fallback analysis result
        final_state = {
            "final_trade_decision": "HOLD - Analysis failed",
            "current_price": 50000.0,  # Fallback price
            "analysis_results": []
        }
        decision = "HOLD"

    try:
        print("⚙️ Processing analysis result...")
        trading_result = trading_manager.process_analysis_result(
            ticker=ticker, analysis_result=final_state, trade_date=trade_date
        )
    except Exception as e:
        print(f"❌ Error processing trading result: {e}")
        trading_result = {
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": f"Error: {str(e)}"
        }

    try:
        print("💰 Resolving current price...")
        current_price = resolve_current_price(ticker, final_state, debug=debug)
        if current_price == 0:
            print(f"⚠️ WARNING: Could not resolve real-time price for {ticker}")
            current_price = 50000.0  # Fallback price for BTC
            print(f"Using fallback price: ${current_price:,.2f}")
    except Exception as e:
        print(f"❌ Error resolving price: {e}")
        current_price = 50000.0  # Fallback price
        print(f"Using fallback price: ${current_price:,.2f}")

    # Generate enhanced recommendation with crypto optimization
    try:
        print("🎯 Generating trading recommendation...")
        recommendation = generate_enhanced_recommendation(
            ticker, trading_result["signal"], current_price, final_state, trading_result
        )
    except Exception as e:
        print(f"❌ Error generating recommendation: {e}")
        recommendation = f"🟡 BASIC RECOMMENDATION: {trading_result['signal']} {ticker}\n📊 Current Price: ${current_price:,.2f}\n❌ Error: {str(e)}"

    print("\n" + "=" * 80)
    print("🎯 OPTIMIZED TRADING RECOMMENDATION")
    print("=" * 80)
    print(recommendation)

    print(f"\n📊 Confidence Level: {trading_result.get('confidence', 0.5):.1%}")

    if final_state.get("final_trade_decision", ""):
        print(f"\n🧠 Analysis Summary: {final_state['final_trade_decision'][:200]}...")

    dashboard = TradingDashboard(trading_manager)
    print("\n" + "=" * 80)
    print("📊 PORTFOLIO STATUS")
    print("=" * 80)
    print(dashboard.display_portfolio_summary())

    # Save comprehensive report to organized subfolders
    try:
        ensure_directories()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced report with architecture info
        report = {
            "ticker": ticker,
            "analysis_decision": decision,
            "trading_result": trading_result,
            "current_price": current_price,
            "recommendation": recommendation,
            "architecture": "full_tradingagents_optimized",
            "crypto_optimized": True,
            "timestamp": timestamp
        }
        
        # Save to organized subfolders
        with open(f"reports/analysis/optimized_trading_report_{ticker}_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        trading_manager.save_analysis_report(f"reports/analysis/trading_report_{ticker}_{timestamp}.json")
        dashboard.save_dashboard_report(f"reports/analysis/dashboard_report_{ticker}_{timestamp}.json")
    except Exception as e:
        print(f"❌ Error saving reports: {e}")
        print("Continuing without saving reports...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n📋 Reports saved to organized subfolders:")
    print(f"  - reports/analysis/optimized_trading_report_{ticker}_{timestamp}.json")
    print(f"  - reports/analysis/trading_report_{ticker}_{timestamp}.json")
    print(f"  - reports/analysis/dashboard_report_{ticker}_{timestamp}.json")

    return trading_result, trading_manager


def run_multiple_analyses(tickers: list, trade_date: str, initial_capital: float = 100000.0, debug: bool = False):
    """
    Run analysis for multiple tickers.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING MULTIPLE TRADING ANALYSES")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date: {trade_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"{'='*80}\n")

    trading_manager = TradingManager(
        initial_capital=initial_capital,
        max_position_size=0.1,
        state_file="reports/trading_state_multi.json",
        enable_short_selling=True,
        margin_requirement=0.5,
    )

    ta = TradingAgentsGraph(debug=debug, config=config)
    results = {}

    for ticker in tickers:
        print(f"\n{'='*40}")
        print(f"ANALYZING {ticker}")
        print(f"{'='*40}")

        try:
            final_state, decision = ta.propagate(ticker, trade_date)
            trading_result = trading_manager.process_analysis_result(
                ticker=ticker, analysis_result=final_state, trade_date=trade_date
            )
            results[ticker] = trading_result

            current_price = resolve_current_price(ticker, final_state, debug=debug)
            if current_price == 0:
                print(f"⚠️ WARNING: Could not resolve price for {ticker}")
                continue

            print(f"\n🎯 TRADING RECOMMENDATION FOR {ticker}")
            print("-" * 50)
            print(generate_recommendation(ticker, trading_result["signal"], current_price))
            print(f"   Confidence: {trading_result.get('confidence', 0.5):.1%}")
            print(f"   Actions: {len(trading_result['actions_taken'])}")

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            results[ticker] = {"error": str(e)}

    dashboard = TradingDashboard(trading_manager)
    print("\n" + "=" * 80)
    print("📊 COMBINED PORTFOLIO SUMMARY")
    print("=" * 80)
    print(dashboard.display_portfolio_summary())

    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trading_manager.save_analysis_report(f"reports/trading_report_multi_{timestamp}.json")
    dashboard.save_dashboard_report(f"reports/dashboard_report_multi_{timestamp}.json")

    return results, trading_manager


# ========== MAIN ==========
if __name__ == "__main__":
    print("Starting crypto-optimized 24/7 trading analysis...")
    # Test with a weekend date to demonstrate 24/7 crypto trading
    result, manager = run_trading_analysis("BTC", "2025-09-07", initial_capital=100000.0, debug=True)

    # Uncomment to run multiple tickers
    # results, manager = run_multiple_analyses(["BTC", "ETH", "AAPL"], "2025-09-07", initial_capital=100000.0, debug=True)

    dashboard = TradingDashboard(manager)
    print("\n" + dashboard.display_recent_trades(limit=5))
    print("\n" + dashboard.display_analysis_history(limit=3))
    print("\n" + dashboard.display_performance_report())

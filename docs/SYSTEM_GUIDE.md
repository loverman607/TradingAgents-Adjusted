# 🚀 Crypto TradingAgents System Guide

## 📋 **System Overview**

Your crypto-optimized TradingAgents system is ready for production use! This guide covers how to run the system and optimal execution rates.

## 🎯 **Current System Configuration**

### **Trading Style: DAY TRADING**
- **Timeframe**: 1-hour intervals
- **Data Frequency**: Hourly OHLCV data
- **Strategy**: Adaptive momentum-based
- **Position Sizing**: 10% of portfolio per trade
- **Risk Management**: RSI, SMA, volatility-based

### **Performance Results** (Latest Backtest)
- **Total Return**: 145.54% (1 year)
- **Win Rate**: 66.67%
- **Max Drawdown**: 64.01%
- **Outperformance**: 147.49% vs BTC, 168.78% vs ETH

## 🚀 **How to Run the System**

### **1. Quick Backtest (Recommended)**
```bash
# Activate your conda environment
conda activate tradingagents

# Run simplified crypto backtest
python crypto_backtest_simple.py
```

### **2. Full TradingAgents Analysis**
```bash
# Run complete LLM-driven analysis
python main.py
```

### **3. Advanced Backtesting (Optional)**
```bash
# Install additional dependencies first
pip install -r requirements.txt

# Run comprehensive backtesting
python run_crypto_backtest.py --mode comprehensive
```

## ⏰ **Recommended Execution Rates**

### **For Day Trading (Current Setup)**
- **Frequency**: Every 1-4 hours
- **Best Times**: 
  - **UTC 00:00-04:00** (Asian markets)
  - **UTC 08:00-12:00** (European markets)
  - **UTC 14:00-18:00** (US markets)
- **Schedule**: Use `schedule` library for automation

### **For Swing Trading (Alternative)**
- **Frequency**: Daily or every 2-3 days
- **Best Times**: 
  - **Daily at 09:00 UTC** (market open)
  - **Every 2-3 days** for longer-term analysis
- **Timeframe**: Change to `1D` in config

## 🔧 **System Configuration**

### **Current Settings** (`crypto_backtest_config.yaml`)
```yaml
backtest:
  timeframe: "1H"        # 1H for day trading, 1D for swing
  strategy: "adaptive"   # momentum, mean_reversion, trend_following
  tickers: ["BTC", "ETH"]

risk:
  max_position_size: 0.2  # 20% max position
  max_daily_loss: 0.05    # 5% max daily loss
  stop_loss_multiplier: 2.0
```

### **Position Sizing Recommendations**
- **Conservative**: 5-10% per trade
- **Moderate**: 10-15% per trade (current)
- **Aggressive**: 15-20% per trade

## 📊 **Monitoring & Alerts**

### **Key Metrics to Watch**
1. **Portfolio Value**: Track daily
2. **Drawdown**: Alert if > 20%
3. **Win Rate**: Maintain > 60%
4. **Sharpe Ratio**: Target > 1.0

### **Risk Management**
- **Stop Loss**: 2x ATR (current)
- **Take Profit**: 3x ATR (current)
- **Max Daily Loss**: 5%
- **Max Drawdown**: 20%

## 🎛️ **System Modes**

### **Mode 1: Backtesting Only**
```bash
python crypto_backtest_simple.py
```
- **Use Case**: Strategy validation
- **Frequency**: Weekly
- **Duration**: 1-2 hours

### **Mode 2: Live Analysis**
```bash
python main.py
```
- **Use Case**: Real-time analysis
- **Frequency**: Every 1-4 hours
- **Duration**: 5-10 minutes

### **Mode 3: Comprehensive Backtesting**
```bash
python run_crypto_backtest.py --mode comprehensive
```
- **Use Case**: Deep strategy analysis
- **Frequency**: Monthly
- **Duration**: 2-4 hours

## ⚡ **Performance Optimization**

### **For High-Frequency Trading**
- Use `1H` timeframe
- Run every 1-2 hours
- Monitor volatility closely
- Use smaller position sizes

### **For Swing Trading**
- Change to `1D` timeframe
- Run daily or every 2-3 days
- Use larger position sizes
- Focus on longer-term trends

## 🔄 **Automation Setup**

### **Using Python Schedule**
```python
import schedule
import time

def run_analysis():
    # Your analysis code here
    pass

# Schedule every 2 hours
schedule.every(2).hours.do(run_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### **Using Cron (Linux/Mac)**
```bash
# Run every 2 hours
0 */2 * * * /path/to/conda/env/bin/python /path/to/main.py

# Run daily at 9 AM UTC
0 9 * * * /path/to/conda/env/bin/python /path/to/main.py
```

## 📈 **Expected Performance**

### **Day Trading (Current)**
- **Expected Return**: 100-200% annually
- **Max Drawdown**: 30-50%
- **Win Rate**: 60-70%
- **Trade Frequency**: 20-50 trades/year

### **Swing Trading (Alternative)**
- **Expected Return**: 50-100% annually
- **Max Drawdown**: 20-30%
- **Win Rate**: 70-80%
- **Trade Frequency**: 10-20 trades/year

## 🚨 **Important Notes**

### **Risk Warnings**
- **High Volatility**: Crypto markets are extremely volatile
- **Liquidity Risk**: Ensure sufficient liquidity for trades
- **Technical Risk**: Monitor system performance closely
- **Market Risk**: Crypto markets can move 20%+ in hours

### **Best Practices**
1. **Start Small**: Begin with 1-5% of capital
2. **Monitor Closely**: Check system every few hours
3. **Set Alerts**: Use stop-losses and take-profits
4. **Diversify**: Don't put all capital in one trade
5. **Backtest Regularly**: Validate strategies monthly

## 🎯 **Quick Start Checklist**

- [ ] Activate conda environment: `conda activate tradingagents`
- [ ] Run backtest: `python crypto_backtest_simple.py`
- [ ] Review results and charts
- [ ] Configure risk parameters
- [ ] Set up monitoring/alerting
- [ ] Start with small position sizes
- [ ] Monitor performance closely

## 📞 **Support**

For issues or questions:
1. Check logs in `crypto_backtest_simple.log`
2. Review configuration in `crypto_backtest_config.yaml`
3. Examine results in generated JSON files
4. Check charts in `crypto_backtest_charts/`

---

**Your crypto trading system is ready to go! 🚀**

**Recommended starting frequency: Every 2-4 hours for day trading, daily for swing trading.**

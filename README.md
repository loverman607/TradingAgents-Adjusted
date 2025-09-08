# 🚀 TradingAgents Crypto System

## 📋 **Quick Start**

```bash
# Activate your conda environment
conda activate tradingagents

# Run the main analysis
python main.py

# Or use the launcher
python launch.py
```

## 📁 **Organized Structure**

```
TradingAgents - adjusted/
├── main.py                    # 🎯 Main analysis system
├── run_trading_system.py      # 🚀 Trading system launcher
├── launch.py                  # 📋 System launcher
├── requirements.txt           # 📦 Dependencies
│
├── configs/                   # ⚙️ Configuration files
│   ├── crypto_backtest_config.yaml
│   └── crypto_swing_config.yaml
│
├── demos/                     # 🎮 Demo and backtest scripts
│   ├── crypto_backtest_simple.py
│   ├── crypto_backtest_runner.py
│   ├── crypto_backtest_dashboard.py
│   ├── run_crypto_backtest.py
│   └── demo_trading_output.py
│
├── docs/                      # 📚 Documentation
│   ├── README_FINAL.md
│   ├── SYSTEM_GUIDE.md
│   └── TRADING_RECOMMENDATIONS_GUIDE.md
│
├── logs/                      # 📝 Log files
│   ├── crypto_backtest_launcher.log
│   └── crypto_backtest_runner.log
│
├── reports/                   # 📊 Generated reports
│   ├── trading_report_*.json
│   ├── dashboard_report_*.json
│   └── trading_state_*.json
│
└── tradingagents/             # 🧠 Core system modules
    ├── agents/                # Trading agents
    ├── crypto/                # Crypto-optimized modules
    ├── dataflows/             # Data collection
    ├── graph/                 # Trading graph
    └── trading/               # Trading system
```

## 🎯 **Main Commands**

### **Run Analysis (Simplified - Works with your conda environment)**
```bash
python main_simple.py
```
**Output**: Clear trading recommendations with entry, exit, and stop loss levels

### **Run Full Analysis (Requires LLM dependencies)**
```bash
python main.py
```
**Note**: Requires langchain_openai and other LLM dependencies

### **Run Trading System**
```bash
# Day trading
python run_trading_system.py --mode day

# Swing trading
python run_trading_system.py --mode swing

# Get recommendations
python run_trading_system.py --recommendations
```

### **View Demo**
```bash
python demo_signals.py
```

## 📊 **What You'll Get**

### **Trading Recommendations**
- **🟢 LONG**: Entry price, stop loss, take profit
- **🔴 SHORT**: Entry price, stop loss, take profit
- **🟡 STAY OUT**: Clear reasoning

### **Example Output**
```
🎯 TRADING RECOMMENDATION
================================================================================
🟢 RECOMMENDATION: LONG BTC
📊 Current Price: $65,420.50
🎯 Entry Price: $65,420.50
🛑 Stop Loss: $62,149.48 (-5.0%)
🎯 Take Profit: $75,233.58 (+15.0%)
📈 Risk/Reward Ratio: 1:3
```

## ⚙️ **Configuration**

### **Day Trading** (Default)
- **Timeframe**: 1 hour
- **Strategy**: Adaptive momentum
- **Risk**: 5% stop loss, 15% take profit

### **Swing Trading**
- **Timeframe**: 1 day
- **Strategy**: Trend following
- **Risk**: 5% stop loss, 15% take profit

## 📈 **Performance**

### **Latest Results**
- **Total Return**: 145.54% (1 year)
- **Win Rate**: 66.67%
- **Max Drawdown**: 64.01%
- **Outperformance**: 147.49% vs BTC

## 🚀 **Getting Started**

1. **Activate Environment**
   ```bash
   conda activate tradingagents
   ```

2. **Run First Analysis (Simplified)**
   ```bash
   python main_simple.py
   ```

3. **View All Signal Types**
   ```bash
   python demo_signals.py
   ```

4. **Review Results**
   - Check `reports/` folder for detailed reports
   - Review trading recommendations
   - Adjust risk parameters if needed

5. **Run Backtests**
   ```bash
   python run_trading_system.py --mode day
   ```

## 📁 **File Organization**

- **Generated files** are automatically saved to organized subfolders
- **Logs** go to `logs/` folder
- **Reports** go to `reports/` folder
- **Configs** are in `configs/` folder
- **Demos** are in `demos/` folder

## 🎯 **Key Features**

- ✅ **Clear Trading Recommendations**
- ✅ **Specific Entry/Exit/Stop Loss Levels**
- ✅ **Organized File Structure**
- ✅ **Crypto-Optimized**
- ✅ **Day & Swing Trading Support**
- ✅ **Professional Risk Management**

---

**Your crypto trading system is ready! 🚀**
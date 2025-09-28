# 📈 Algorithmic Trading System


A sophisticated algorithmic trading system implementing momentum-reversal strategies with machine learning predictions for Indian equity markets (NSE).

## 🌟 Features

- **Multi-Strategy Trading**: RSI-based mean reversion combined with SMA crossover trends
- **Machine Learning Integration**: Logistic regression and Random Forest models for price movement prediction
- **Risk Management**: ATR-based stop-losses, position sizing, and portfolio risk controls
- **Comprehensive Backtesting**: Transaction costs, slippage modeling, and multiple performance metrics
- **Real Market Data**: Integration with Yahoo Finance for NSE stocks
- **Portfolio Management**: Multi-asset support with dynamic weight allocation

## 📊 Performance Metrics

The system calculates comprehensive performance metrics including:
- **Returns**: Total, Annual, Risk-Adjusted
- **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Drawdown Analysis**: Maximum Drawdown, Duration, Recovery
- **Trade Statistics**: Win Rate, Profit Factor, Trade Count
- **Risk Measures**: VaR, CVaR, Skewness, Kurtosis


## 📁 Project Structure

```
algo-trading/
│
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management
│   ├── data.py               # Data fetching and preprocessing
│   ├── indicators.py         # Technical indicators library
│   ├── strategy.py           # Trading strategy implementation
│   ├── backtest.py           # Backtesting engine
│   ├── ml.py                 # Machine learning models
│   └── run.py                # Main execution script
│
├── tests/                    # Unit tests
│   ├── test_indicators.py
│   ├── test_strategy.py
│   └── test_backtest.py
│
├── notebooks/                # Jupyter notebooks for analysis
│   ├── strategy_analysis.ipynb
│   └── ml_experiments.ipynb
│
├── data/                     # Data directory (auto-created)
├── logs/                     # Logging directory
├── reports/                  # Backtest reports
│
├── requirements.txt          # Python dependencies
├── config.json              # Configuration file
├── README.md                # This file
```

## 🔧 Configuration

Create a `config.json` file to customize strategy parameters:

```json
{
  "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
  "period": "1y",
  "interval": "1d",
  "rsi_oversold": 30,
  "rsi_overbought": 70,
  "sma_fast": 20,
  "sma_slow": 50,
  "max_position_size": 0.1,
  "stop_loss_atr_multiplier": 2.0,
  "fee_bp": 5.0,
  "slippage_bp": 2.0
}
```

## 📈 Strategy Details

### Core Strategy: RSI Mean Reversion + Trend Following

**Entry Signals:**
- RSI < 30 (oversold condition)
- SMA(20) > SMA(50) (upward trend)
- Volume confirmation (optional)

**Exit Signals:**
- RSI > 50 (momentum exhaustion)
- SMA(20) < SMA(50) (trend reversal)
- ATR-based stop-loss hit

### Machine Learning Enhancement

The ML module uses the following features:
- Technical Indicators: RSI, MACD, SMA slopes
- Price Patterns: 1/5/20-day returns
- Volume Metrics: Volume ratio, trend
- Volatility: ATR, rolling standard deviation

## 📊 Sample Backtest Results

```
=== PORTFOLIO BACKTEST RESULTS ===
Total Return: 23.45%
Annual Return: 18.32%
Sharpe Ratio: 1.24
Max Drawdown: -12.8%
Win Rate: 58.3%
Number of Trades: 42
```


## 🙏 Acknowledgments

- Yahoo Finance for providing market data
- The open-source Python community
- NSE for Indian market data access
- Contributors and testers

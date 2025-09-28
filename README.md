# 📈 Algorithmic Trading System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-orange)](https://www.python.org/dev/peps/pep-0008/)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen)](https://github.com/yourusername/algo-trading)

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

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
pip >= 20.0
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/algo-trading.git
cd algo-trading
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
# Run the basic strategy backtest
python src/run.py

# Run with custom configuration
python src/run.py --config config.json

# Run ML predictions
python -c "from src.run import run_ml_bonus; run_ml_bonus('RELIANCE.NS')"
```

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
└── LICENSE                  # MIT License
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

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_strategy.py

# Run with coverage
pytest --cov=src tests/
```

## 📝 Usage Examples

### 1. Basic Strategy Backtest

```python
from src.data import fetch_ohlcv
from src.strategy import generate_signals
from src.backtest import backtest_long_only

# Fetch data
df = fetch_ohlcv("RELIANCE.NS", period="6mo")

# Generate signals
signals = generate_signals(df)

# Run backtest
results = backtest_long_only(signals, fee_bp=5.0)
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
```

### 2. ML Predictions

```python
from src.ml import make_features, train_logreg, time_split_train_test

# Prepare features
X, y = make_features(df)

# Split data
X_train, X_test, y_train, y_test = time_split_train_test(X, y)

# Train model
model = train_logreg(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### 3. Custom Indicator

```python
from src.indicators import rsi, sma

# Calculate indicators
df['RSI'] = rsi(df['Close'], window=14)
df['SMA_20'] = sma(df['Close'], window=20)
df['Signal'] = (df['RSI'] < 30) & (df['SMA_20'] > df['SMA_20'].shift(1))
```

## 🔍 Advanced Features (Roadmap)

- [ ] **Live Trading**: Integration with broker APIs (Zerodha, IBKR)
- [ ] **Advanced Strategies**: Pairs trading, Options strategies
- [ ] **Portfolio Optimization**: Markowitz optimization, Risk parity
- [ ] **Sentiment Analysis**: News and social media integration
- [ ] **Web Dashboard**: Real-time monitoring and control panel
- [ ] **Cloud Deployment**: AWS/GCP deployment scripts
- [ ] **Alert System**: Email/SMS notifications for signals

## ⚠️ Risk Disclaimer

**IMPORTANT**: This software is for educational purposes only. 

- Past performance does not guarantee future results
- Algorithmic trading carries substantial risk of loss
- Always validate strategies with paper trading first
- Never trade with money you cannot afford to lose
- Consult with financial advisors before live trading

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows PEP8 standards and includes appropriate tests.

## 📚 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
yfinance>=0.2.18
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=7.0.0
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGithub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Yahoo Finance for providing market data
- The open-source Python community
- NSE for Indian market data access
- Contributors and testers

## 📞 Support

For support, email your.email@example.com or create an issue in the GitHub repository.

## 📊 Performance Visualization

![Strategy Performance](https://via.placeholder.com/800x400?text=Strategy+Performance+Chart)

*Sample equity curve showing strategy performance vs buy-and-hold benchmark*

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

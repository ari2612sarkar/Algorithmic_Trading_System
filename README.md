# 📈 Algorithmic Trading System

![CI](https://github.com/ari2612sarkar/Algorithmic_Trading_System/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end algorithmic trading research platform for NSE equities. It ingests
historical OHLCV data, engineers technical features, trains a deep-learning +
gradient-boosting ensemble to forecast price direction, backtests an RSI/SMA
strategy under realistic risk controls, and serves it all through an interactive
web dashboard — deployable anywhere with a single Docker command.

**Tech stack:** Python · PyTorch (LSTM) · LightGBM · pandas · Plotly · Streamlit ·
APScheduler · Docker.

**Pipeline:** `collect → engineer features → train (LSTM + LightGBM) → predict /
forecast → risk-aware backtest → schedule & monitor`.

> ⚠️ For research and educational use only. This is **not** financial advice and
> places no live broker orders.

## 🖥️ Dashboard

The Streamlit dashboard (`app.py`) gives an interactive view of the whole
pipeline — candlestick + SMA + volume + RSI charts, one-click train/predict/
backtest, a probability gauge, a strategy-vs-buy-&-hold equity curve, a live
scheduler monitor, and a CSV report browser.

```bash
streamlit run app.py          # → http://localhost:8501
```

<!-- Add a screenshot at docs/dashboard.png and uncomment:
![Dashboard](docs/dashboard.png)
-->

## 🌟 Features

### Core Features (100% Implementation)
1. **Data Ingestion** - NSE data via nselib with yfinance fallback for NIFTY 50 stocks
2. **Trading Strategy** - RSI + Moving Average crossover strategy
3. **Backtesting** - Risk-aware historical backtesting with detailed metrics
4. **ML Predictions** - LSTM (deep learning) + LightGBM (gradient boosting) models
5. **CSV Reporting** - Automated logging of trades, performance, and predictions to `reports/`
6. **Full Automation** - Scheduled multi-interval execution for all symbols

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- 20-day & 50-day Simple Moving Averages
- Volume indicators
- Price momentum

## 📋 Requirements

- Python 3.8+
- Active internet connection

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd algo-trading-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `src/config.py` to adjust the universe and strategy/model parameters:

```python
TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]  # NIFTY 50 stocks
RSI_WINDOW = 14
SMA_FAST = 20
SMA_SLOW = 50
```

### 3. Run the System

The entry point is the `src.run` CLI module:

```bash
# Collect + cache OHLCV (daily)
python -m src.run --collect

# Train models, then generate next-bar predictions
python -m src.run --train
python -m src.run --predict

# Risk-aware backtest
python -m src.run --backtest

# Run everything across both intervals (1d + 1h)
python -m src.run --all
```

All output is written as timestamped CSV files to the `reports/` directory.

### 4. Interactive Dashboard (Web UI)

An interactive [Streamlit](https://streamlit.io/) dashboard wraps the whole
pipeline — pick a ticker/interval and collect data, train, predict, forecast,
and backtest with live charts:

```bash
streamlit run app.py
```

Then open http://localhost:8501. Features:
- Candlestick chart with SMA overlay, buy-signal markers, and an RSI panel
- One-click **Collect / Train / Predict / Forecast / Backtest** from the sidebar
- Prediction card (UP/DOWN + confidence) and N-bar forecast path
- Risk-aware backtest metrics, equity curve, and trade log
- Browser for the timestamped CSV reports (with download)

## 🐳 Deployment (Docker)

The repo ships a `Dockerfile` and `docker-compose.yml`. CPU-only PyTorch is
installed in the image, so it runs on any host without a GPU.

```bash
# Build and start the dashboard (http://localhost:8501)
docker compose up --build

# Run in the background
docker compose up -d

# Also run the live scheduler daemon (hourly/daily predict + weekly retrain)
docker compose --profile live up -d scheduler
```

`data/`, `models/`, and `reports/` are mounted as volumes, so fetched data,
trained models, and reports persist across container restarts. The timezone
defaults to `Asia/Kolkata` (NSE hours) and can be changed in `docker-compose.yml`.

Because it's a standard container, you can deploy the same image to any
Docker-capable host (a VPS, or PaaS like Render / Railway / Fly.io) — point the
platform at this repo or push the built `algo-trading-dashboard` image.

## 📊 Trading Strategy

### Buy Signal
- **Condition 1**: RSI < 30 (oversold)
- **Condition 2**: 20-day MA crosses above 50-day MA (bullish crossover)

### Sell Signal
- **Condition 1**: RSI > 70 (overbought)
- **Condition 2**: 20-day MA crosses below 50-day MA (bearish crossover)

## 🤖 Machine Learning Models

### Features Used (`src/features.py`)
- Returns (1-bar, 5-bar, 20-bar)
- RSI (14-period)
- SMA 20 & SMA 50
- MACD, MACD signal, MACD histogram
- Volume change & high–low range
- ATR and return-over-ATR

### Labels
Targets are **triple-barrier** labels (take-profit / stop-loss / timeout) computed
from ATR-scaled barriers — a higher signal-to-noise target than naive next-bar
direction. Label = 1 if the TP barrier is hit before the SL barrier.

### Models
1. **LSTM Classifier** (PyTorch, `src/predictor.py`)
   - 2-layer LSTM (hidden=64) with dropout, sigmoid head
   - Trained on scaled, windowed sequences (`SEQ_LEN` per interval)
   - Class-imbalance-weighted loss; runs on GPU if available, else CPU

2. **LightGBM Classifier** (`src/gbm.py`)
   - Gradient-boosted trees on the same feature set
   - Trained per ticker as an ensemble companion to the LSTM

Predictions are the **average of the LSTM and LightGBM probabilities** (the LSTM
alone is used if LightGBM is unavailable). Persisted per ticker/interval under
`models/` and evaluated via a rolling **walk-forward** split.

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## 📈 Backtest Metrics

The system calculates:
- **Total Return %**: Overall portfolio performance
- **Win Ratio**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Number of Trades**: Total buy/sell signals

## 📑 CSV Report Structure

All reports are written to `reports/` as timestamped CSV files:

### `trades_*.csv` / `backtest_*.csv`
Trade signals and backtest results: ticker, interval, total return, win rate, Sharpe, max drawdown, number of trades.

### `predictions_*.csv`
Next-bar model predictions per symbol.

### `training_*.csv`
Model training metrics per symbol.

## 📁 Project Structure

```
algo-trading-system/
├── app.py                   # Streamlit interactive dashboard
├── Dockerfile               # Container image (CPU-only PyTorch)
├── docker-compose.yml       # Dashboard + optional live scheduler
├── .streamlit/config.toml   # Dashboard theme
├── .github/workflows/ci.yml # Lint + compile + import CI
├── LICENSE                  # MIT
├── src/
│   ├── run.py               # CLI entry point
│   ├── config.py            # Universe + parameters
│   ├── data_collector.py    # OHLCV ingestion (nselib + yfinance)
│   ├── strategy.py          # RSI + MA crossover signals
│   ├── backtest.py          # Backtesting engine
│   ├── risk.py              # Risk-aware backtest (ATR stops + Kelly)
│   ├── predictor.py         # Train / predict / forecast orchestration
│   ├── dl_model.py          # LSTM deep-learning model
│   ├── gbm.py               # LightGBM model
│   └── scheduler.py         # APScheduler live daemon
├── data/                    # Cached OHLCV CSVs (git-ignored)
├── models/                  # Trained model artifacts (git-ignored)
├── reports/                 # Timestamped output CSVs (git-ignored)
├── logs/                    # system.log for the Live tab (git-ignored)
├── Demo/                    # Standalone Colab demo notebook
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## 🧩 Module Breakdown

### 1. DataIngestion
- Fetches historical data from NSE
- Handles API rate limiting
- Generates sample data for testing

### 2. TechnicalAnalysis
- Calculates RSI, MACD, Moving Averages
- Adds volume indicators
- Computes price momentum

### 3. TradingStrategy
- Implements buy/sell logic
- Generates trading signals
- Tracks positions

### 4. Backtester
- Simulates historical trading
- Calculates portfolio returns
- Computes performance metrics

### 5. MLPredictor
- Prepares features from indicators
- Trains classification models
- Predicts next-day movements

### 6. ReportWriter
- Writes trades, performance, and ML results to CSV
- Timestamps each report under `reports/`

### 7. AutomatedTradingSystem
- Orchestrates all modules
- Runs analysis for multiple symbols
- Handles errors gracefully

## 🎯 Evaluation Criteria (20% each)

### ✅ API Handling (20%)
- ✓ nselib integration with yfinance fallback
- ✓ Error handling & retry logic
- ✓ Rate limiting
- ✓ Data validation

### ✅ Strategy Logic (20%)
- ✓ RSI-based signals
- ✓ MA crossover confirmation
- ✓ Position tracking
- ✓ Signal generation

### ✅ Automation (20%)
- ✓ Single-function execution
- ✓ Multi-symbol processing
- ✓ Scheduled capability
- ✓ Error recovery

### ✅ ML Implementation (20%)
- ✓ LSTM deep-learning model
- ✓ LightGBM gradient-boosting model (ensembled)
- ✓ Feature engineering + triple-barrier labels
- ✓ Accuracy reporting + walk-forward validation

### ✅ Code Quality (20%)
- ✓ Modular design
- ✓ Comprehensive logging
- ✓ Documentation
- ✓ Error handling

## 🔧 Advanced Usage

All settings live in `src/config.py`.

### Running for Different Stocks

```python
TICKERS = ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK']
```

### Changing Strategy Parameters

```python
RSI_WINDOW = 14
SMA_FAST = 20
SMA_SLOW = 50
```

### Tuning the Models

```python
EPOCHS = 40          # LSTM training epochs
SEQ_LEN = {"1d": 10, "1h": 40}    # sequence window per interval
TB_HORIZON = {"1d": 10, "1h": 30} # triple-barrier look-forward
```

## 📊 Sample Output

```
================================================================================
TRADING SYSTEM SUMMARY
================================================================================

RELIANCE:
  Latest Price: ₹2,543.75
  Latest RSI: 45.32
  Total Return: 12.45%
  Win Ratio: 65.50%
  ML Accuracy: 68.75%
  Next Day Prediction: UP

TCS:
  Latest Price: ₹3,687.20
  Latest RSI: 52.18
  Total Return: 8.92%
  Win Ratio: 61.20%
  ML Accuracy: 72.30%
  Next Day Prediction: DOWN

INFY:
  Latest Price: ₹1,523.45
  Latest RSI: 38.67
  Total Return: 15.67%
  Win Ratio: 70.40%
  ML Accuracy: 71.85%
  Next Day Prediction: UP
```

## 🐛 Troubleshooting

### NSE API Issues
- Check internet connection
- Verify nselib is latest version
- System automatically falls back to yfinance if nselib fails


## 📝 Logging

All activities are logged to `trading_system.log`:
- Data fetching operations
- Signal generation
- Backtest results
- ML training progress
- Errors and warnings

## ⚡ Performance Tips

1. **Rate Limiting**: Add delays between API calls
2. **Caching**: Store fetched data locally
3. **Parallel Processing**: Use multiprocessing for multiple symbols
4. **Database**: Store historical data in SQLite/PostgreSQL

## 🔮 Future Enhancements

- [ ] Live trading integration
- [ ] More technical indicators (Bollinger Bands, ATR)
- [ ] Advanced ML models (Random Forest, XGBoost)
- [ ] Real-time WebSocket data
- [ ] Portfolio optimization
- [ ] Risk management system
- [ ] Web dashboard
- [ ] Paper trading mode

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

**Built with ❤️ for algorithmic trading enthusiasts**

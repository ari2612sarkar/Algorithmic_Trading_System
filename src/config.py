# src/config.py
import logging

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
PERIOD = "6mo"        # last 6 months
INTERVAL = "1d"       # daily bars
RSI_WINDOW = 14
SMA_FAST = 20
SMA_SLOW = 50

# simple logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("mini-algo")

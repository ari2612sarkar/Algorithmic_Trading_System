# src/config.py
import logging
import os
from datetime import datetime, timedelta

# ---- Universe ----
TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]

# ---- Intervals ----
SUPPORTED_INTERVALS = ("1d", "1h")

# ---- Lookbacks per interval ----
# Daily: 5 years. Hourly: 720 days (yfinance hard cap is ~730 for 1h).
LOOKBACK_DAYS = {"1d": 1825, "1h": 720}

# ---- Strategy / indicator params ----
RSI_WINDOW = 14
SMA_FAST = 20
SMA_SLOW = 50

# ---- ML / DL sequence length per interval ----
# Hourly: ~6 bars/day x 7 days ≈ 40 bars window
SEQ_LEN = {"1d": 10, "1h": 40}

# ---- Triple-barrier horizons per interval ----
TB_HORIZON = {"1d": 10, "1h": 30}  # ~5 trading days hourly

# Training hyper-params
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TEST_FRAC = 0.2

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
for _p in (DATA_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR):
    os.makedirs(_p, exist_ok=True)

# Shared log file the dashboard can tail to show live scheduler/CLI activity.
LOG_FILE = os.path.join(LOG_DIR, "system.log")


def date_range(lookback_days: int):
    """Return (start, end) as dd-mm-YYYY strings expected by nselib."""
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    return start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")


# ---- Logger ----
# Log to both stdout and a rotating file so the dashboard's Live tab can tail it.
from logging.handlers import RotatingFileHandler  # noqa: E402

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=2,
                                    encoding="utf-8")
_file_handler.setFormatter(_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger("algo")

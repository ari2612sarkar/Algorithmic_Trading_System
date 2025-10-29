import pandas as pd
import logging
from nselib import capital_market

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def fetch_ohlcv(symbol: str, start: str, end: str):
    """
    Fetch daily OHLCV data for NSE-listed equity from nselib API.

    Args:
        symbol (str): Stock symbol, e.g. "RELIANCE"
        start (str): Start date, format "dd-mm-yyyy"
        end (str): End date, format "dd-mm-yyyy"

    Returns:
        pd.DataFrame: Cleaned OHLCV DataFrame with Date index.
    """

    log.info(f"📥 Fetching NSE data for {symbol} from {start} to {end}...")

    try:
        df = capital_market.price_volume_and_deliverable_position_data(
            symbol=symbol,
            from_date=start,
            to_date=end
        )
    except Exception as e:
        log.error(f"❌ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()

    if df.empty:
        log.warning(f"⚠️ No data returned for {symbol}")
        return df

    log.info(f"🔹 Raw columns returned: {list(df.columns)}")

    # Rename columns to consistent format
    rename_map = {
        "OpenPrice": "Open",
        "HighPrice": "High",
        "LowPrice": "Low",
        "ClosePrice": "Close",
        "TotalTradedQuantity": "Volume"
    }

    df = df.rename(columns=rename_map)

    # Convert date and numeric columns
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date")

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in num_cols:
        df[c] = (
            df[c].astype(str)
            .str.replace(",", "")
            .astype(float)
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    log.info(f"✅ Cleaned data for {symbol} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    data = fetch_ohlcv("RELIANCE", "01-04-2025", "30-10-2025")
    if not data.empty:
        log.info("\n" + str(data.tail(5)))
    else:
        log.warning("No data fetched.")

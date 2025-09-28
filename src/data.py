# src/data.py
import pandas as pd
import yfinance as yf
from src.config import log, PERIOD, INTERVAL

def fetch_ohlcv(ticker: str, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    log.info(f"Downloading {ticker} {period} {interval}")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df

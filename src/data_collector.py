# src/data_collector.py
"""Automated OHLCV collector with per-interval caching + retry + yfinance fallback.

Supported intervals: '1d' (5y default) and '1h' (~2y, yfinance cap).
- 1d: tries nselib first, falls back to yfinance.
- 1h: forces yfinance (nselib doesn't reliably serve hourly history).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.config import (
    DATA_DIR, LOOKBACK_DAYS, SUPPORTED_INTERVALS, date_range, log as root_log,
)

log = root_log.getChild("collector")


def _nselib_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    from nselib import capital_market  # type: ignore
    raw = capital_market.price_volume_and_deliverable_position_data(
        symbol=symbol, from_date=start, to_date=end
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    rename_map = {
        "OpenPrice": "Open", "HighPrice": "High", "LowPrice": "Low",
        "ClosePrice": "Close", "TotalTradedQuantity": "Volume",
    }
    df = raw.rename(columns=rename_map)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def _yfinance_fetch(symbol: str, start_dt: datetime, end_dt: datetime,
                    interval: str = "1d") -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        log.warning("yfinance not installed; skipping fallback")
        return pd.DataFrame()
    yf_symbol = symbol if "." in symbol else f"{symbol}.NS"

    # yfinance hourly cap: ~730 days. Period works better than start/end for intraday.
    if interval == "1h":
        days = min((end_dt - start_dt).days + 1, 729)
        df = yf.download(yf_symbol, period=f"{days}d", interval="1h",
                         progress=False, auto_adjust=False)
    else:
        df = yf.download(
            yf_symbol,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval, progress=False, auto_adjust=False,
        )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None) if df.index.tz else df.index.tz_localize(None)
    return df.sort_index()


def _cache_path(symbol: str, interval: str) -> str:
    safe = symbol.replace("/", "_").replace(".", "_")
    return os.path.join(DATA_DIR, f"{safe}_{interval}.csv")


def _load_cache(symbol: str, interval: str) -> pd.DataFrame:
    p = _cache_path(symbol, interval)
    if not os.path.exists(p):
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["Date"], index_col="Date")
        return df.sort_index()
    except Exception as e:
        log.warning(f"cache read failed for {symbol} {interval}: {e}")
        return pd.DataFrame()


def _save_cache(symbol: str, interval: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_convert(None) if out.index.tz else out.index.tz_localize(None)
    out.to_csv(_cache_path(symbol, interval), index_label="Date")


def fetch_with_retry(symbol: str, start: str, end: str, interval: str = "1d",
                     retries: int = 3) -> pd.DataFrame:
    """For 1d: try nselib then yfinance. For 1h: yfinance only."""
    start_dt = datetime.strptime(start, "%d-%m-%Y")
    end_dt = datetime.strptime(end, "%d-%m-%Y")

    if interval == "1d":
        for attempt in range(1, retries + 1):
            try:
                log.info(f"[nselib] {symbol} {start}->{end} 1d (attempt {attempt})")
                df = _nselib_daily(symbol, start, end)
                if not df.empty:
                    return df
                log.warning(f"[nselib] empty result for {symbol}")
            except Exception as e:
                log.warning(f"[nselib] {symbol} failed: {e}")
            time.sleep(1.5 * attempt)
        log.info(f"[yfinance] fallback for {symbol} 1d")
        try:
            return _yfinance_fetch(symbol, start_dt, end_dt, "1d")
        except Exception as e:
            log.error(f"[yfinance] {symbol} 1d failed: {e}")
            return pd.DataFrame()

    # interval == "1h"
    log.info(f"[yfinance] {symbol} 1h {start}->{end}")
    try:
        return _yfinance_fetch(symbol, start_dt, end_dt, "1h")
    except Exception as e:
        log.error(f"[yfinance] {symbol} 1h failed: {e}")
        return pd.DataFrame()


def collect(symbol: str, interval: str = "1d",
            lookback_days: Optional[int] = None,
            force_refresh: bool = False) -> pd.DataFrame:
    """Incremental cached OHLCV. Per-interval cache file."""
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"interval must be one of {SUPPORTED_INTERVALS}")
    if lookback_days is None:
        lookback_days = LOOKBACK_DAYS[interval]

    cached = pd.DataFrame() if force_refresh else _load_cache(symbol, interval)

    if cached.empty:
        start, end = date_range(lookback_days)
        fresh = fetch_with_retry(symbol, start, end, interval=interval)
        _save_cache(symbol, interval, fresh)
        return fresh

    last_ts = pd.Timestamp(cached.index.max())
    # Normalize to tz-naive — yfinance hourly returns tz-aware, nselib daily doesn't
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert(None) if hasattr(last_ts, "tz_convert") else last_ts.tz_localize(None)
    today = pd.Timestamp(datetime.today())
    tolerance = timedelta(hours=20) if interval == "1d" else timedelta(hours=1)
    if (today - last_ts) < tolerance:
        log.info(f"cache up-to-date for {symbol} {interval} (last={last_ts})")
        return cached

    start_ts = last_ts + (timedelta(days=1) if interval == "1d" else timedelta(hours=1))
    start = start_ts.strftime("%d-%m-%Y")
    end = today.strftime("%d-%m-%Y")
    log.info(f"incremental update for {symbol} {interval}: {start} -> {end}")
    new_part = fetch_with_retry(symbol, start, end, interval=interval)

    if new_part.empty:
        return cached

    merged = pd.concat([cached, new_part])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _save_cache(symbol, interval, merged)
    return merged


def collect_all(symbols: list[str], interval: str = "1d",
                lookback_days: Optional[int] = None,
                force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        out[s] = collect(s, interval=interval, lookback_days=lookback_days,
                         force_refresh=force_refresh)
        time.sleep(0.5)
    return out


if __name__ == "__main__":
    from src.config import TICKERS
    for itv in ("1d", "1h"):
        print(f"\n--- {itv} ---")
        data = collect_all(TICKERS, interval=itv)
        for sym, df in data.items():
            print(f"{sym}: {len(df)} rows" + (f" last={df.index.max()}" if not df.empty else ""))

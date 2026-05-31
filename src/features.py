# src/features.py
"""Feature engineering + triple-barrier labels.

Two label schemes available:
  - simple next-day up/down  -> 'Target'
  - triple-barrier (TP / SL / timeout) -> 'TB_Target', far better signal-to-noise
"""
import pandas as pd
import numpy as np
from src.indicators import rsi, sma, macd, atr

FEATURE_COLS = [
    "Return", "Return_5", "Return_20",
    "RSI", "SMA20", "SMA50",
    "MACD", "MACD_SIG", "MACD_HIST",
    "Vol_Chg", "HL_Range",
    "ATR", "Ret_over_ATR",
]


def create_features(df: pd.DataFrame, drop_na: bool = True,
                    tb_horizon: int = 10, tb_tp: float = 1.5, tb_sl: float = 1.5) -> pd.DataFrame:
    """Build feature matrix + both target columns.

    Args:
        df: OHLCV DataFrame (Open/High/Low/Close/Volume) indexed by Date.
        tb_horizon: bars to look forward for triple-barrier labels.
        tb_tp, tb_sl: take-profit / stop-loss expressed as multiples of ATR.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["Return_5"] = d["Close"].pct_change(5)
    d["Return_20"] = d["Close"].pct_change(20)
    d["RSI"] = rsi(d["Close"], 14)
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    line, sig, hist = macd(d["Close"])
    d["MACD"] = line
    d["MACD_SIG"] = sig
    d["MACD_HIST"] = hist
    d["Vol_Chg"] = d["Volume"].pct_change().replace([np.inf, -np.inf], 0)
    d["HL_Range"] = (d["High"] - d["Low"]) / d["Close"].replace(0, np.nan)
    d["ATR"] = atr(d["High"], d["Low"], d["Close"], 14)
    d["Ret_over_ATR"] = d["Return"] / (d["ATR"] / d["Close"]).replace(0, np.nan)

    d["Target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d["TB_Target"] = _triple_barrier(d["High"], d["Low"], d["Close"], d["ATR"],
                                     horizon=tb_horizon, tp_mult=tb_tp, sl_mult=tb_sl)

    if drop_na:
        d = d.dropna(subset=FEATURE_COLS + ["Target"])
    return d


def _triple_barrier(high: pd.Series, low: pd.Series, close: pd.Series, atr_s: pd.Series,
                    horizon: int, tp_mult: float, sl_mult: float) -> pd.Series:
    """For each bar i, look at the next `horizon` bars.

    Label = 1 if TP barrier (close[i] + tp_mult*ATR[i]) is hit first,
            0 if SL barrier hit first or neither hit (timeout treated as 0).
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    cl = close.values
    hi = high.values
    lo = low.values
    a = atr_s.values
    for i in range(n - horizon):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        tp = cl[i] + tp_mult * a[i]
        sl = cl[i] - sl_mult * a[i]
        hit = 0  # timeout default
        for j in range(i + 1, i + 1 + horizon):
            if hi[j] >= tp:
                hit = 1
                break
            if lo[j] <= sl:
                hit = 0
                break
        out[i] = hit
    return pd.Series(out, index=close.index)

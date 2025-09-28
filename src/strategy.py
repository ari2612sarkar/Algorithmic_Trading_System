# src/strategy.py
import pandas as pd
from src.indicators import rsi, sma
from src.config import RSI_WINDOW, SMA_FAST, SMA_SLOW

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RSI"] = rsi(out["Close"], RSI_WINDOW)
    out["SMA_F"] = sma(out["Close"], SMA_FAST)
    out["SMA_S"] = sma(out["Close"], SMA_SLOW)
    return out

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: RSI, SMA_F, SMA_S, signal (1 buy today), position (1 long/0 flat) executed next day."""
    d = add_indicators(df)
    # entry rule (today): RSI<30 and SMA20>SMA50
    d["buy_rule"] = (d["RSI"] < 30) & (d["SMA_F"] > d["SMA_S"])
    # exit rule (today): SMA20<SMA50 OR RSI>50
    d["exit_rule"] = (d["SMA_F"] < d["SMA_S"]) | (d["RSI"] > 50)

    # stateful position (execute next day -> shift to avoid lookahead)
    position = []
    in_pos = False
    for i in range(len(d)):
        if not in_pos and d["buy_rule"].iloc[i]:
            in_pos = True
        elif in_pos and d["exit_rule"].iloc[i]:
            in_pos = False
        position.append(1 if in_pos else 0)
    d["position"] = pd.Series(position, index=d.index).shift(1).fillna(0)  # trade next bar
    d["signal"] = d["buy_rule"].astype(int)
    return d

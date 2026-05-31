# src/risk.py
"""Risk management: ATR stops, Kelly sizing, drawdown circuit breaker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import atr


def kelly_fraction(prob_up: float, payoff_ratio: float = 2.0,
                   cap: float = 0.25) -> float:
    """Kelly: f* = p - (1-p)/b. Capped to avoid over-leverage."""
    b = max(payoff_ratio, 1e-6)
    f = prob_up - (1 - prob_up) / b
    return float(np.clip(f, 0.0, cap))


def position_size(equity: float, last_close: float, prob_up: float,
                  atr_val: float, atr_stop_mult: float = 1.0,
                  payoff_ratio: float = 2.0, max_frac: float = 0.25) -> int:
    """Volatility-targeted Kelly sizing -> integer share count.

    risk_per_share = atr_stop_mult * ATR
    risk_budget   = equity * kelly_fraction
    shares        = floor(risk_budget / risk_per_share)
    """
    if prob_up <= 0.5 or atr_val <= 0 or last_close <= 0:
        return 0
    k = kelly_fraction(prob_up, payoff_ratio, max_frac)
    if k <= 0:
        return 0
    risk_per_share = atr_stop_mult * atr_val
    if risk_per_share <= 0:
        return 0
    budget = equity * k
    shares = int(budget // risk_per_share)
    # cap so notional doesn't exceed equity
    max_shares = int(equity // last_close)
    return max(0, min(shares, max_shares))


def backtest_with_risk(df: pd.DataFrame,
                       prob_series: pd.Series | None = None,
                       prob_threshold: float = 0.55,
                       atr_window: int = 14,
                       tp_mult: float = 2.0,
                       sl_mult: float = 1.0,
                       max_hold: int = 10,
                       initial_capital: float = 100_000.0,
                       fee_bp: float = 5.0,
                       max_drawdown_stop: float = 0.20) -> dict:
    """Event-driven backtest with ATR stops, take-profits, and DD circuit breaker.

    Entry: when prob_series > prob_threshold (or, if no prob series, when
    Close > SMA20 > SMA50 — simple momentum baseline).
    Exit:  whichever of [TP, SL, max_hold bars] fires first.
    """
    d = df.copy()
    d["ATR"] = atr(d["High"], d["Low"], d["Close"], atr_window)

    if prob_series is None:
        prob_series = pd.Series(
            ((d["Close"] > d["Close"].rolling(20).mean()) &
             (d["Close"].rolling(20).mean() > d["Close"].rolling(50).mean())).astype(float),
            index=d.index,
        )
    p = prob_series.reindex(d.index).fillna(0.0)

    equity = initial_capital
    peak = initial_capital
    halted = False
    in_pos = False
    entry_idx = None
    entry_px = 0.0
    shares = 0
    tp_px = sl_px = 0.0

    equity_curve = []
    trades = []
    fee = fee_bp / 10000.0

    for i in range(len(d)):
        bar = d.iloc[i]
        if pd.isna(bar["ATR"]) or bar["ATR"] <= 0:
            equity_curve.append(equity)
            continue

        if halted:
            equity_curve.append(equity)
            continue

        if in_pos:
            hit_tp = bar["High"] >= tp_px
            hit_sl = bar["Low"] <= sl_px
            held = i - entry_idx
            exit_px = None
            reason = None
            if hit_sl and hit_tp:
                exit_px, reason = sl_px, "SL"  # conservative: assume SL hits first
            elif hit_tp:
                exit_px, reason = tp_px, "TP"
            elif hit_sl:
                exit_px, reason = sl_px, "SL"
            elif held >= max_hold:
                exit_px, reason = float(bar["Close"]), "TIMEOUT"

            if exit_px is not None:
                gross = (exit_px - entry_px) * shares
                cost = (entry_px + exit_px) * shares * fee
                pnl = gross - cost
                equity += pnl
                trades.append({
                    "entry_date": d.index[entry_idx], "exit_date": d.index[i],
                    "entry": entry_px, "exit": exit_px, "shares": shares,
                    "pnl": pnl, "reason": reason,
                })
                in_pos = False
                shares = 0
                peak = max(peak, equity)
                if (equity / peak) - 1.0 <= -max_drawdown_stop:
                    halted = True

        if (not in_pos) and (not halted) and p.iloc[i] > prob_threshold:
            atr_v = float(bar["ATR"])
            shares = position_size(equity, float(bar["Close"]), float(p.iloc[i]),
                                   atr_v, atr_stop_mult=sl_mult,
                                   payoff_ratio=tp_mult / sl_mult)
            if shares > 0:
                in_pos = True
                entry_idx = i
                entry_px = float(bar["Close"])
                tp_px = entry_px + tp_mult * atr_v
                sl_px = entry_px - sl_mult * atr_v

        equity_curve.append(equity)

    eq = pd.Series(equity_curve, index=d.index)
    rets = eq.pct_change().fillna(0)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    pnls = [t["pnl"] for t in trades]
    win_rate = float(np.mean([1 if x > 0 else 0 for x in pnls])) if pnls else 0.0

    return {
        "total_return": float(eq.iloc[-1] / initial_capital - 1),
        "final_equity": float(eq.iloc[-1]),
        "sharpe": float(sharpe),
        "max_drawdown": float(dd),
        "n_trades": len(trades),
        "win_rate": win_rate,
        "halted": halted,
        "equity_curve": eq,
        "trades": pd.DataFrame(trades),
    }

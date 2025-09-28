# src/backtest.py
import pandas as pd
import numpy as np

def backtest_long_only(df: pd.DataFrame, fee_bp: float = 5.0) -> dict:
    """
    Long-only, enter when position goes 0->1, exit when 1->0. Fee in basis points per trade side.
    Expects columns: Close, position
    """
    d = df.copy()
    d["pos_prev"] = d["position"].shift(1).fillna(0)
    d["entry"] = (d["position"] == 1) & (d["pos_prev"] == 0)
    d["exit"]  = (d["position"] == 0) & (d["pos_prev"] == 1)

    # daily returns
    d["ret"] = d["Close"].pct_change().fillna(0)
    strat_ret = d["ret"] * d["position"]

    # fees: apply when toggling position
    turn = (d["entry"] | d["exit"]).astype(float)
    fee = (fee_bp / 10000.0) * turn  # subtract on those days
    strat_ret_net = strat_ret - fee

    # equity curve
    d["equity"] = (1 + strat_ret_net).cumprod()

    # trade stats
    entries = d.index[d["entry"]].tolist()
    exits   = d.index[d["exit"]].tolist()
    # align trades (ignore open trade at end)
    pairs = []
    e_ix = 0
    for t_in in entries:
        while e_ix < len(exits) and exits[e_ix] <= t_in:
            e_ix += 1
        if e_ix < len(exits):
            pairs.append((t_in, exits[e_ix]))
            e_ix += 1
    trades = []
    for t_in, t_out in pairs:
        px_in  = d.at[t_in, "Close"]
        px_out = d.at[t_out, "Close"]
        gross = (px_out/px_in) - 1.0
        gross -= 2 * (fee_bp/10000.0)  # entry + exit cost
        trades.append(gross)

    # metrics
    total_return = d["equity"].iloc[-1] - 1.0
    win_rate = float(np.mean([1 if x > 0 else 0 for x in trades])) if trades else 0.0
    sharpe = _sharpe(strat_ret_net)
    max_dd = _max_drawdown(d["equity"])
    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": len(trades),
        "equity_curve": d["equity"],
        "daily_returns": strat_ret_net
    }

def _sharpe(r: pd.Series, risk_free_daily: float = 0.0):
    excess = r - risk_free_daily
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)

def _max_drawdown(equity: pd.Series):
    roll_max = equity.cummax()
    dd = (equity/roll_max) - 1.0
    return dd.min()

# src/run.py
import pandas as pd
from src.config import log, TICKERS
from src.data import fetch_ohlcv
from  src.strategy import generate_signals
from src.backtest import backtest_long_only
from src.ml import make_features, time_split_train_test, train_logreg, evaluate

def run_core():
    summary = []
    for t in TICKERS:
        df = fetch_ohlcv(t)
        ds = generate_signals(df)
        res = backtest_long_only(ds)
        log.info(f"{t} | ret: {res['total_return']:.2%} | win: {res['win_rate']:.1%} | sharpe: {res['sharpe']:.2f} | maxDD: {res['max_drawdown']:.2%} | trades: {res['n_trades']}")
        summary.append({"ticker": t, **{k: v for k, v in res.items() if k not in ('equity_curve','daily_returns')}})
    return pd.DataFrame(summary)

def run_ml_bonus(ticker: str):
    df = fetch_ohlcv(ticker)
    X, y = make_features(df)
    if len(X) < 60:
        raise ValueError("Not enough samples for ML.")
    X_tr, X_te, y_tr, y_te = time_split_train_test(X, y)
    model = train_logreg(X_tr, y_tr)
    acc, rep = evaluate(model, X_te, y_te)
    log.info(f"[ML] {ticker} accuracy: {acc:.3f}\n{rep}")

if __name__ == "__main__":
    core = run_core()
    print("\n=== CORE SUMMARY ===")
    print(core.to_string(index=False))
    # ML bonus on one ticker
    run_ml_bonus("RELIANCE.NS")

# src/run.py
"""CLI entry point for the trading system.

Usage:
    python -m src.run --collect                      # daily
    python -m src.run --collect --interval 1h        # hourly
    python -m src.run --train --interval 1h
    python -m src.run --predict
    python -m src.run --backtest                     # backtest uses 1d
    python -m src.run --forecast 5 --interval 1h     # next 5 hours
    python -m src.run --all                          # both intervals
    python -m src.run --schedule
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

from src.config import TICKERS, REPORT_DIR, SUPPORTED_INTERVALS, log
from src.data_collector import collect, collect_all
from src.strategy import generate_signals
from src.backtest import backtest_long_only
from src.predictor import train_all, predict_all, forecast_path, walk_forward
from src.risk import backtest_with_risk


def _report(prefix: str, interval: str = "1d") -> str:
    stamp = datetime.today().strftime("%Y-%m-%d")
    return os.path.join(REPORT_DIR, f"{prefix}_{interval}_{stamp}.csv")


def cmd_collect(interval: str = "1d"):
    data = collect_all(TICKERS, interval=interval)
    for s, df in data.items():
        if df.empty:
            log.warning(f"{s} {interval}: NO DATA")
        else:
            log.info(f"{s} {interval}: {len(df)} rows ({df.index.min()} -> {df.index.max()})")


def cmd_train(interval: str = "1d"):
    df = train_all(TICKERS, interval=interval)
    if not df.empty:
        out = _report("training", interval)
        df.to_csv(out, index=False)
        print(df.to_string(index=False))
        log.info(f"wrote {out}")


def cmd_predict(interval: str = "1d"):
    df = predict_all(TICKERS, interval=interval)
    if df.empty:
        log.warning(f"no predictions produced ({interval})")
        return
    out = _report("predictions", interval)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    log.info(f"wrote {out}")


def cmd_backtest(risk_aware: bool = True, interval: str = "1d"):
    rows = []
    for t in TICKERS:
        df = collect(t, interval=interval)
        if df.empty:
            log.warning(f"{t}: no data, skipping backtest")
            continue
        if risk_aware:
            r = backtest_with_risk(df)
            log.info(
                f"{t} | ret: {r['total_return']:.2%} | win: {r['win_rate']:.1%} | "
                f"sharpe: {r['sharpe']:.2f} | maxDD: {r['max_drawdown']:.2%} | "
                f"trades: {r['n_trades']} | halted: {r['halted']}"
            )
            rows.append({"ticker": t, "interval": interval,
                         "total_return": r["total_return"], "win_rate": r["win_rate"],
                         "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"],
                         "n_trades": r["n_trades"], "halted": r["halted"]})
        else:
            ds = generate_signals(df)
            r = backtest_long_only(ds)
            log.info(f"{t} | ret: {r['total_return']:.2%} | trades: {r['n_trades']}")
            rows.append({"ticker": t, "interval": interval,
                         "total_return": r["total_return"], "win_rate": r["win_rate"],
                         "sharpe": r["sharpe"], "max_drawdown": r["max_drawdown"],
                         "n_trades": r["n_trades"]})
    if rows:
        out = _report("backtest", interval)
        pd.DataFrame(rows).to_csv(out, index=False)
        log.info(f"wrote {out}")


def cmd_walkforward(interval: str = "1d"):
    rows = []
    for t in TICKERS:
        try:
            r = walk_forward(t, interval=interval)
            rows.append({k: v for k, v in r.items() if k != "per_fold"})
        except Exception as e:
            log.warning(f"walk-forward failed for {t}: {e}")
    if rows:
        out = _report("walkforward", interval)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        log.info(f"wrote {out}")


def cmd_forecast(horizon: int, interval: str = "1d"):
    rows = []
    for t in TICKERS:
        try:
            rows.extend(forecast_path(t, interval=interval, horizon=horizon))
        except Exception as e:
            log.warning(f"forecast failed for {t}: {e}")
    if rows:
        df = pd.DataFrame(rows)
        out = _report(f"forecast_{horizon}", interval)
        df.to_csv(out, index=False)
        print(df.to_string(index=False))
        log.info(f"wrote {out}")


def cmd_schedule():
    from src.scheduler import start
    start()


def main():
    p = argparse.ArgumentParser(description="Algorithmic trading system")
    p.add_argument("--interval", choices=list(SUPPORTED_INTERVALS), default="1d",
                   help="bar interval (default 1d)")
    p.add_argument("--collect", action="store_true", help="fetch + cache OHLCV")
    p.add_argument("--train", action="store_true", help="train LSTM+GBM per ticker")
    p.add_argument("--predict", action="store_true", help="next-bar predictions")
    p.add_argument("--backtest", action="store_true", help="risk-aware backtest (ATR stops + Kelly)")
    p.add_argument("--backtest-simple", action="store_true", help="legacy long-only backtest")
    p.add_argument("--walkforward", action="store_true", help="walk-forward CV evaluation")
    p.add_argument("--forecast", type=int, metavar="N", help="N-step iterative forecast")
    p.add_argument("--all", action="store_true",
                   help="run both intervals: collect+train+backtest+predict")
    p.add_argument("--schedule", action="store_true", help="run APScheduler daemon")
    args = p.parse_args()

    if not any([args.collect, args.train, args.predict, args.backtest,
                args.backtest_simple, args.walkforward, args.forecast,
                args.all, args.schedule]):
        p.print_help()
        return

    if args.all:
        for itv in SUPPORTED_INTERVALS:
            log.info(f"========== {itv} ==========")
            cmd_collect(itv)
            cmd_train(itv)
            cmd_backtest(risk_aware=True, interval=itv)
            cmd_predict(itv)
        return

    itv = args.interval
    if args.collect:          cmd_collect(itv)
    if args.train:            cmd_train(itv)
    if args.backtest:         cmd_backtest(risk_aware=True, interval=itv)
    if args.backtest_simple:  cmd_backtest(risk_aware=False, interval=itv)
    if args.walkforward:      cmd_walkforward(itv)
    if args.predict:          cmd_predict(itv)
    if args.forecast:         cmd_forecast(args.forecast, itv)
    if args.schedule:         cmd_schedule()


if __name__ == "__main__":
    main()

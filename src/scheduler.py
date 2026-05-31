# src/scheduler.py
"""Automated multi-interval fetch + predict; weekly retrain. Writes CSV reports."""
from __future__ import annotations

import os
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import TICKERS, REPORT_DIR, SUPPORTED_INTERVALS, log as root_log
from src.data_collector import collect_all
from src.predictor import predict_all, train_all

log = root_log.getChild("scheduler")


def _report_path(prefix: str, interval: str) -> str:
    stamp = datetime.today().strftime("%Y-%m-%d-%H%M")
    return os.path.join(REPORT_DIR, f"{prefix}_{interval}_{stamp}.csv")


def _run(interval: str):
    collect_all(TICKERS, interval=interval)
    df = predict_all(TICKERS, interval=interval)
    if not df.empty:
        out = _report_path("predictions", interval)
        df.to_csv(out, index=False)
        log.info(f"wrote {out}")
        log.info("\n" + df.to_string(index=False))


def job_hourly():
    log.info("=== HOURLY JOB START ===")
    _run("1h")
    log.info("=== HOURLY JOB END ===")


def job_daily():
    log.info("=== DAILY JOB START ===")
    _run("1d")
    log.info("=== DAILY JOB END ===")


def job_weekly_retrain():
    log.info("=== WEEKLY RETRAIN START ===")
    for itv in SUPPORTED_INTERVALS:
        df = train_all(TICKERS, interval=itv)
        if not df.empty:
            df.to_csv(_report_path("training", itv), index=False)
    log.info("=== WEEKLY RETRAIN END ===")


def start():
    """Hourly during NSE session, daily after close, weekly retrain Sun night."""
    sched = BlockingScheduler(timezone="Asia/Kolkata")
    sched.add_job(job_hourly, CronTrigger(day_of_week="mon-fri",
                                          hour="9-15", minute=35),
                  id="hourly_predict", replace_existing=True)
    sched.add_job(job_daily, CronTrigger(hour=18, minute=30),
                  id="daily_predict", replace_existing=True)
    sched.add_job(job_weekly_retrain, CronTrigger(day_of_week="sun", hour=20),
                  id="weekly_retrain", replace_existing=True)
    log.info("scheduler armed: hourly 9-15:35 IST, daily 18:30 IST, retrain Sun 20:00 IST")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopped")


if __name__ == "__main__":
    start()

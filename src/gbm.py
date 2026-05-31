# src/gbm.py
"""LightGBM classifier on flat features. Cheap, strong baseline for daily bars."""
from __future__ import annotations

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from src.config import MODEL_DIR, log as root_log
from src.features import FEATURE_COLS

log = root_log.getChild("gbm")

try:
    import lightgbm as lgb  # type: ignore
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False
    log.warning("lightgbm not installed — GBM model unavailable")


def _path(symbol: str, interval: str = "1d") -> str:
    safe = symbol.replace("/", "_").replace(".", "_")
    return os.path.join(MODEL_DIR, f"{safe}_{interval}_gbm.joblib")


def train_gbm(symbol: str, feat: pd.DataFrame, label_col: str = "TB_Target",
              interval: str = "1d") -> dict:
    if not HAVE_LGB:
        raise RuntimeError("lightgbm not installed")
    df = feat.dropna(subset=FEATURE_COLS + [label_col]).copy()
    if len(df) < 100:
        raise ValueError(f"not enough rows for GBM: {len(df)}")

    X = df[FEATURE_COLS].values
    y = df[label_col].astype(int).values

    split = int(len(df) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=31,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob > 0.5).astype(int)
    acc = float((pred == y_te).mean())

    joblib.dump({"model": model, "features": FEATURE_COLS, "interval": interval},
                _path(symbol, interval))
    log.info(f"[{symbol} {interval}] GBM acc={acc:.3f}")
    return {"symbol": symbol, "interval": interval, "accuracy": acc, "n_test": len(y_te)}


def load_gbm(symbol: str, interval: str = "1d"):
    p = _path(symbol, interval)
    if not os.path.exists(p):
        return None
    return joblib.load(p)


def predict_gbm(symbol: str, feat: pd.DataFrame, interval: str = "1d") -> Optional[float]:
    blob = load_gbm(symbol, interval)
    if blob is None:
        return None
    model = blob["model"]
    X = feat[FEATURE_COLS].iloc[[-1]].values
    return float(model.predict_proba(X)[0, 1])

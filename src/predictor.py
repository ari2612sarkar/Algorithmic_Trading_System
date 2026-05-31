# src/predictor.py
"""LSTM-based stock direction predictor with persistent models per ticker.

Pipeline per ticker:
  1. collect cached OHLCV
  2. build features (src/features.py)
  3. scale + window into sequences of SEQ_LEN
  4. train LSTM (binary next-day up classifier)
  5. persist model + scaler to models/
  6. predict_next() -> direction + probability
  7. forecast_path() -> N-day iterative directional forecast
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from src.config import (
    MODEL_DIR, SEQ_LEN, TB_HORIZON, EPOCHS, BATCH_SIZE, LEARNING_RATE, TEST_FRAC,
    log as root_log,
)
from src.features import create_features, FEATURE_COLS
from src.data_collector import collect
from src.gbm import train_gbm, predict_gbm, HAVE_LGB

log = root_log.getChild("predictor")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- model ----------
class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ---------- helpers ----------
def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _paths(symbol: str, interval: str = "1d"):
    safe = symbol.replace("/", "_").replace(".", "_")
    tag = f"{safe}_{interval}"
    return (
        os.path.join(MODEL_DIR, f"{tag}_lstm.pt"),
        os.path.join(MODEL_DIR, f"{tag}_scaler.joblib"),
        os.path.join(MODEL_DIR, f"{tag}_meta.json"),
    )


# ---------- training ----------
@dataclass
class TrainResult:
    symbol: str
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def train(symbol: str, interval: str = "1d", df: Optional[pd.DataFrame] = None,
          epochs: int = EPOCHS, seq_len: Optional[int] = None,
          label_col: str = "TB_Target", train_gbm_too: bool = True) -> TrainResult:
    """Train LSTM (and optionally LightGBM) on triple-barrier labels by default."""
    if seq_len is None:
        seq_len = SEQ_LEN[interval]
    tb_h = TB_HORIZON[interval]
    if df is None:
        df = collect(symbol, interval=interval)
    if df.empty or len(df) < seq_len + 60:
        raise ValueError(f"not enough rows for {symbol} {interval}: {len(df)}")

    feat = create_features(df, tb_horizon=tb_h)
    if feat.empty or len(feat) < seq_len + 30:
        raise ValueError(f"not enough feature rows for {symbol} {interval}: {len(feat)}")

    # Drop tail rows where TB label couldn't be computed
    feat_lbl = feat.dropna(subset=[label_col]).copy()

    X = feat_lbl[FEATURE_COLS].values
    y = feat_lbl[label_col].astype(int).values

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    X_seq, y_seq = _make_sequences(Xs, y, seq_len)
    split = int(len(X_seq) * (1 - TEST_FRAC))
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE).unsqueeze(1)
    Xv = torch.tensor(X_te, device=DEVICE)
    yv = torch.tensor(y_te, device=DEVICE).unsqueeze(1)

    model = LSTMClassifier(n_features=X_seq.shape[2]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos = max(int(y_tr.sum()), 1)
    neg = max(len(y_tr) - pos, 1)
    pos_weight = torch.tensor([neg / pos], device=DEVICE, dtype=torch.float32)
    def loss_fn(pred, target):
        eps = 1e-7
        pred_c = pred.clamp(eps, 1 - eps)
        w = target * pos_weight + (1 - target)
        return -(w * (target * torch.log(pred_c) + (1 - target) * torch.log(1 - pred_c))).mean()

    n = len(Xt)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                v_pred = model(Xv)
                v_loss = loss_fn(v_pred, yv).item()
            log.info(f"[{symbol}] epoch {epoch}/{epochs} train={total/n:.4f} val={v_loss:.4f}")

    # eval
    model.eval()
    with torch.no_grad():
        probs = model(Xv).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)
    res = TrainResult(
        symbol=symbol,
        n_samples=len(X_seq),
        accuracy=float(accuracy_score(y_te, preds)),
        precision=float(precision_score(y_te, preds, zero_division=0)),
        recall=float(recall_score(y_te, preds, zero_division=0)),
        f1=float(f1_score(y_te, preds, zero_division=0)),
    )

    # persist
    mpath, spath, jpath = _paths(symbol, interval)
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": X_seq.shape[2],
        "seq_len": seq_len,
        "interval": interval,
    }, mpath)
    joblib.dump(scaler, spath)
    with open(jpath, "w") as f:
        json.dump({**asdict(res), "interval": interval}, f, indent=2)
    log.info(f"[{symbol} {interval}] saved model | acc={res.accuracy:.3f} f1={res.f1:.3f}")

    # Companion LightGBM for ensembling
    if train_gbm_too and HAVE_LGB:
        try:
            train_gbm(symbol, feat, label_col=label_col, interval=interval)
        except Exception as e:
            log.warning(f"[{symbol} {interval}] GBM training skipped: {e}")
    return res


# ---------- walk-forward CV ----------
def walk_forward(symbol: str, interval: str = "1d",
                 df: Optional[pd.DataFrame] = None,
                 train_window: int = 200, test_window: int = 20,
                 epochs: int = 20, seq_len: Optional[int] = None,
                 label_col: str = "TB_Target") -> dict:
    """Rolling walk-forward evaluation. Trains LSTM on `train_window` bars,
    predicts the next `test_window` bars, slides forward. Returns aggregate metrics."""
    if seq_len is None:
        seq_len = SEQ_LEN[interval]
    tb_h = TB_HORIZON[interval]
    if df is None:
        df = collect(symbol, interval=interval)
    feat = create_features(df, tb_horizon=tb_h).dropna(subset=[label_col]).copy()
    X = feat[FEATURE_COLS].values
    y = feat[label_col].astype(int).values

    folds = []
    all_y, all_p = [], []
    start = 0
    fold_id = 0
    while start + train_window + test_window <= len(X):
        fold_id += 1
        tr_end = start + train_window
        te_end = tr_end + test_window
        scaler = StandardScaler().fit(X[start:tr_end])
        Xs = scaler.transform(X[start:te_end])
        X_seq, y_seq = _make_sequences(Xs, y[start:te_end], seq_len)
        cut = train_window - seq_len
        if cut <= 0 or len(X_seq) <= cut:
            start += test_window
            continue
        X_tr, X_te = X_seq[:cut], X_seq[cut:]
        y_tr, y_te = y_seq[:cut], y_seq[cut:]

        model = LSTMClassifier(n_features=X_seq.shape[2]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.BCELoss()
        Xt = torch.tensor(X_tr, device=DEVICE)
        yt = torch.tensor(y_tr, device=DEVICE).unsqueeze(1)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(X_te, device=DEVICE)).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        acc = float((preds == y_te).mean()) if len(y_te) else 0.0
        folds.append({"fold": fold_id, "n_test": len(y_te), "accuracy": acc})
        all_y.extend(y_te.tolist())
        all_p.extend(preds.tolist())
        start += test_window

    if not folds:
        return {"symbol": symbol, "folds": 0, "accuracy": None}
    overall = {
        "symbol": symbol,
        "folds": len(folds),
        "accuracy": float(accuracy_score(all_y, all_p)),
        "precision": float(precision_score(all_y, all_p, zero_division=0)),
        "recall": float(recall_score(all_y, all_p, zero_division=0)),
        "f1": float(f1_score(all_y, all_p, zero_division=0)),
        "per_fold": folds,
    }
    log.info(f"[{symbol} {interval}] walk-forward {len(folds)} folds | acc={overall['accuracy']:.3f} f1={overall['f1']:.3f}")
    overall["interval"] = interval
    return overall


# ---------- inference ----------
def _load(symbol: str, interval: str = "1d"):
    mpath, spath, jpath = _paths(symbol, interval)
    if not (os.path.exists(mpath) and os.path.exists(spath)):
        return None
    ckpt = torch.load(mpath, map_location=DEVICE, weights_only=False)
    model = LSTMClassifier(n_features=ckpt["n_features"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = joblib.load(spath)
    return model, scaler, ckpt["seq_len"]


def predict_next(symbol: str, interval: str = "1d",
                 df: Optional[pd.DataFrame] = None) -> dict:
    """Ensemble next-bar prediction: average LSTM + LightGBM probabilities."""
    loaded = _load(symbol, interval)
    if loaded is None:
        raise FileNotFoundError(f"no trained model for {symbol} {interval}; run train() first")
    model, scaler, seq_len = loaded

    if df is None:
        df = collect(symbol, interval=interval)
    feat = create_features(df, tb_horizon=TB_HORIZON[interval])
    if len(feat) < seq_len:
        raise ValueError(f"not enough feature rows ({len(feat)}) for seq_len={seq_len}")

    X = scaler.transform(feat[FEATURE_COLS].values)
    window = torch.tensor(X[-seq_len:][None, :, :].astype(np.float32), device=DEVICE)
    with torch.no_grad():
        lstm_prob = float(model(window).cpu().numpy().flatten()[0])

    gbm_prob = predict_gbm(symbol, feat, interval=interval)
    if gbm_prob is None:
        prob = lstm_prob
        members = ["lstm"]
    else:
        prob = 0.5 * lstm_prob + 0.5 * gbm_prob
        members = ["lstm", "gbm"]

    last_close = float(feat["Close"].iloc[-1]) if "Close" in feat.columns else float(df["Close"].iloc[-1])
    return {
        "symbol": symbol,
        "interval": interval,
        "as_of": str(feat.index[-1]),
        "last_close": round(last_close, 2),
        "lstm_prob_up": round(lstm_prob, 4),
        "gbm_prob_up": round(gbm_prob, 4) if gbm_prob is not None else None,
        "prob_up": round(prob, 4),
        "direction": "UP" if prob > 0.5 else "DOWN",
        "confidence": round(abs(prob - 0.5) * 2, 4),
        "ensemble": "+".join(members),
    }


def forecast_path(symbol: str, interval: str = "1d", horizon: int = 5,
                  df: Optional[pd.DataFrame] = None) -> list[dict]:
    """Iterative N-step directional forecast (`step` size = interval bar)."""
    loaded = _load(symbol, interval)
    if loaded is None:
        raise FileNotFoundError(f"no trained model for {symbol} {interval}")
    model, scaler, seq_len = loaded

    if df is None:
        df = collect(symbol, interval=interval)
    feat = create_features(df, tb_horizon=TB_HORIZON[interval])
    X = scaler.transform(feat[FEATURE_COLS].values).astype(np.float32)
    window = X[-seq_len:].copy()

    step_delta = pd.Timedelta(days=1) if interval == "1d" else pd.Timedelta(hours=1)
    results = []
    last_ts = feat.index[-1]
    last_close = float(feat["Close"].iloc[-1])
    drift = 0.005 if interval == "1d" else 0.001  # smaller hourly drift
    for step in range(1, horizon + 1):
        w = torch.tensor(window[None, :, :], device=DEVICE)
        with torch.no_grad():
            prob = float(model(w).cpu().numpy().flatten()[0])
        direction = "UP" if prob > 0.5 else "DOWN"
        projected = last_close * (1 + drift if direction == "UP" else 1 - drift)
        results.append({
            "symbol": symbol,
            "interval": interval,
            "step": step,
            "ts": str(last_ts + step * step_delta),
            "prob_up": round(prob, 4),
            "direction": direction,
            "projected_close": round(projected, 2),
        })
        window = np.vstack([window[1:], window[-1:]])
        last_close = projected
    return results


def predict_all(symbols: list[str], interval: str = "1d") -> pd.DataFrame:
    rows = []
    for s in symbols:
        try:
            rows.append(predict_next(s, interval=interval))
        except Exception as e:
            log.warning(f"predict failed for {s} {interval}: {e}")
    return pd.DataFrame(rows)


def train_all(symbols: list[str], interval: str = "1d") -> pd.DataFrame:
    rows = []
    for s in symbols:
        try:
            r = train(s, interval=interval)
            rows.append({**asdict(r), "interval": interval})
        except Exception as e:
            log.warning(f"train failed for {s} {interval}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.config import TICKERS
    for itv in ("1d", "1h"):
        print(train_all(TICKERS, interval=itv))
        print(predict_all(TICKERS, interval=itv))

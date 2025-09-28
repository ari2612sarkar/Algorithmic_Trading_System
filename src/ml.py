# src/ml.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.indicators import rsi, sma, macd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["RSI"] = rsi(d["Close"], 14)
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    macd_line, macd_sig, macd_hist = macd(d["Close"])
    d["MACD"] = macd_line
    d["MACD_SIG"] = macd_sig
    d["VOL_CHG"] = d["Volume"].pct_change().fillna(0)
    # target: next-day up/down
    d["y"] = (d["Close"].shift(-1) > d["Close"]).astype(int)
    d = d.dropna()
    X = d[["RSI", "SMA20", "SMA50", "MACD", "MACD_SIG", "VOL_CHG"]]
    y = d["y"]
    return X, y

def time_split_train_test(X, y, test_frac=0.25):
    n = len(X)
    cut = int(n * (1 - test_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

def train_logreg(X_tr, y_tr):
    model = LogisticRegression(max_iter=200)
    model.fit(X_tr, y_tr)
    return model

def evaluate(model, X_te, y_te):
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    rep = classification_report(y_te, pred, digits=3)
    return acc, rep

# app.py
"""Interactive Streamlit dashboard for the Algorithmic Trading System.

Run locally:
    streamlit run app.py
Or via Docker:
    docker compose up

The dashboard wraps the existing `src/` pipeline:
    collect -> chart/signals -> train -> predict/forecast -> backtest
and lets you browse the timestamped CSV reports in reports/.
"""
from __future__ import annotations

import os
import glob
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import TICKERS, SUPPORTED_INTERVALS, REPORT_DIR, MODEL_DIR, LOG_FILE
from src.data_collector import collect
from src.strategy import generate_signals
from src.risk import backtest_with_risk
from src.predictor import train, predict_next, forecast_path

# ----------------------------- theme constants -----------------------------
ACCENT = "#00C49A"
UP = "#00C49A"
DOWN = "#FF5C5C"
SMA_F_COLOR = "#4C9AFF"
SMA_S_COLOR = "#FFAB4C"
GRID = "rgba(255,255,255,0.06)"

st.set_page_config(page_title="Algo Trading Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------- styling -----------------------------
st.markdown(
    """
    <style>
      [data-testid="stToolbar"], [data-testid="stDecoration"], footer {display: none;}
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}

      /* metric cards */
      [data-testid="stMetric"] {
        background: #161A23;
        border: 1px solid #232A38;
        border-radius: 14px;
        padding: 14px 18px;
      }
      [data-testid="stMetricLabel"] p {font-size: .78rem; opacity: .65;
        text-transform: uppercase; letter-spacing: .04em;}

      /* hero banner */
      .hero {
        background: linear-gradient(120deg, #11161F 0%, #0E1117 60%);
        border: 1px solid #232A38; border-radius: 16px;
        padding: 18px 24px; margin-bottom: 18px;
        display: flex; align-items: center; justify-content: space-between;
      }
      .hero h1 {margin: 0; font-size: 1.7rem; font-weight: 800;
        background: linear-gradient(90deg, #E6E9EF, #00C49A);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
      .hero .sub {opacity: .6; font-size: .85rem; margin-top: 2px;}
      .pill {display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: .8rem; font-weight: 600; border: 1px solid #2A3142; background: #161A23;}

      /* tabs */
      button[data-baseweb="tab"] {font-weight: 600;}

      /* buttons */
      .stButton > button {border-radius: 10px; font-weight: 600;}
      section[data-testid="stSidebar"] {border-right: 1px solid #1c222e;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- helpers -----------------------------
def _model_path(symbol: str, interval: str) -> str:
    safe = symbol.replace("/", "_").replace(".", "_")
    return os.path.join(MODEL_DIR, f"{safe}_{interval}_lstm.pt")


def has_model(symbol: str, interval: str) -> bool:
    return os.path.exists(_model_path(symbol, interval))


@st.cache_data(show_spinner=False, ttl=900)
def load_data(symbol: str, interval: str, force: bool) -> pd.DataFrame:
    """Cached OHLCV fetch. `force` busts the cache to trigger a refresh."""
    return collect(symbol, interval=interval, force_refresh=force)


def latest_report(prefix: str) -> str | None:
    """Most recently modified reports/<prefix>_*.csv, or None."""
    matches = glob.glob(os.path.join(REPORT_DIR, f"{prefix}_*.csv"))
    return max(matches, key=os.path.getmtime) if matches else None


def tail_log(n: int = 40) -> str:
    """Last n lines of the shared system log."""
    if not os.path.exists(LOG_FILE):
        return ""
    try:
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception as e:
        return f"(could not read log: {e})"


def style_fig(fig: go.Figure, height: int = 600) -> go.Figure:
    """Apply the dashboard's dark, transparent theme to a plotly figure."""
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E9EF"),
        margin=dict(t=40, b=10, l=10, r=10),
        legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def price_chart(df: pd.DataFrame, symbol: str, show_signals: bool,
                bars: int) -> go.Figure:
    d = generate_signals(df).tail(bars)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.58, 0.18, 0.24],
        vertical_spacing=0.04,
        subplot_titles=(f"{symbol} — price &amp; SMA", "Volume", "RSI (14)"),
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="OHLC", increasing_line_color=UP, decreasing_line_color=DOWN,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA_F"], name="SMA fast",
                             line=dict(color=SMA_F_COLOR, width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA_S"], name="SMA slow",
                             line=dict(color=SMA_S_COLOR, width=1.4)), row=1, col=1)
    if show_signals:
        buys = d[d["signal"] == 1]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["Low"] * 0.99, mode="markers", name="buy signal",
                marker=dict(symbol="triangle-up", size=11, color=ACCENT,
                            line=dict(width=1, color="#0E1117"))), row=1, col=1)

    vol_colors = [UP if c >= o else DOWN for o, c in zip(d["Open"], d["Close"])]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                         marker_color=vol_colors, opacity=0.5,
                         showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
                             line=dict(color="#B388FF", width=1.3),
                             showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=DOWN, opacity=0.6, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=UP, opacity=0.6, row=3, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#161A23", activecolor=ACCENT, font=dict(size=11),
        ),
        row=1, col=1,
    )
    return style_fig(fig, height=680)


def prob_gauge(prob_up: float) -> go.Figure:
    up = prob_up >= 0.5
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_up * 100,
        number={"suffix": "%", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": UP if up else DOWN},
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#241B22"},
                {"range": [50, 100], "color": "#16241F"},
            ],
            "threshold": {"line": {"color": "#E6E9EF", "width": 2}, "value": 50},
        },
    ))
    return style_fig(fig, height=240)


# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.markdown("### 📈 Algo Trading")
    st.caption("LSTM + LightGBM signal engine")
    st.divider()

    symbol = st.selectbox("Ticker", TICKERS, index=0)
    custom = st.text_input("…or custom symbol", value="",
                           placeholder="e.g. SBIN")
    if custom.strip():
        symbol = custom.strip().upper()
    interval = st.radio("Interval", list(SUPPORTED_INTERVALS), horizontal=True)
    horizon = st.slider("Forecast horizon (bars)", 1, 30, 5)
    bars = st.slider("Chart history (bars)", 60, 500, 250, step=10)

    st.divider()
    do_refresh = st.button("🔄 Collect / refresh data", use_container_width=True)
    do_train = st.button("🧠 Train model", use_container_width=True, type="primary")

    st.divider()
    trained = has_model(symbol, interval)
    st.markdown(
        f"<span class='pill'>{'🟢 model ready' if trained else '🔴 no model'}"
        f" · {symbol} {interval}</span>",
        unsafe_allow_html=True,
    )
    with st.expander("ℹ️ About"):
        st.markdown(
            "- **Strategy:** RSI + SMA crossover\n"
            "- **ML:** LSTM + LightGBM ensemble on triple-barrier labels\n"
            "- **Risk:** ATR stops, Kelly sizing, drawdown breaker\n\n"
            "Train a model, then use **Prediction** and **Backtest**."
        )


# ----------------------------- data load -----------------------------
try:
    df = load_data(symbol, interval, force=do_refresh)
    if do_refresh:
        st.toast(f"Refreshed {symbol} {interval}: {len(df):,} rows", icon="✅")
except Exception as e:
    st.error(f"Data load failed for {symbol} {interval}: {e}")
    st.stop()

if df is None or df.empty:
    st.warning(f"No data available for {symbol} {interval}. "
               "Try 'Collect / refresh data', or a different symbol.")
    st.stop()

# ----------------------------- hero + KPIs -----------------------------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
chg_pct = (last_close / prev_close - 1) * 100 if prev_close else 0.0
sig = generate_signals(df).iloc[-1]
trend = "Bullish ▲" if sig["SMA_F"] >= sig["SMA_S"] else "Bearish ▼"

st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>Algorithmic Trading System</h1>
        <div class="sub">{symbol} · {interval} · {len(df):,} bars ·
          last bar {str(df.index.max()).split(' ')[0]}</div>
      </div>
      <div class="pill">{'🟢 model ready' if trained else '🔴 untrained'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Last close", f"₹{last_close:,.2f}", f"{chg_pct:+.2f}%")
k2.metric("RSI (14)", f"{sig['RSI']:.1f}")
k3.metric("Trend (SMA20/50)", trend)
k4.metric("History", f"{len(df):,} bars")


# ----------------------------- training -----------------------------
if do_train:
    with st.spinner(f"Training LSTM + LightGBM for {symbol} {interval}…"):
        try:
            res = train(symbol, interval=interval, df=df)
            st.success(
                f"Trained {symbol} {interval} — acc {res.accuracy:.3f} · "
                f"precision {res.precision:.3f} · recall {res.recall:.3f} · "
                f"f1 {res.f1:.3f}"
            )
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Training failed: {e}")


# ----------------------------- tabs -----------------------------
tab_chart, tab_pred, tab_bt, tab_live, tab_reports = st.tabs(
    ["📊 Chart", "🔮 Prediction", "🧪 Backtest", "📡 Live", "📁 Reports"]
)

with tab_chart:
    show_signals = st.toggle("Show buy signals", value=True)
    st.plotly_chart(price_chart(df, symbol, show_signals, bars),
                    use_container_width=True)

with tab_pred:
    if not trained:
        st.info("No trained model yet — click **🧠 Train model** in the sidebar first.")
    else:
        c_run, c_fc = st.columns(2)
        with c_run:
            if st.button("Predict next bar", use_container_width=True, type="primary"):
                with st.spinner("Predicting…"):
                    try:
                        st.session_state["pred"] = predict_next(
                            symbol, interval=interval, df=df)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        with c_fc:
            if st.button(f"Forecast {horizon} bars", use_container_width=True):
                with st.spinner(f"Forecasting {horizon} bars…"):
                    try:
                        st.session_state["fc"] = pd.DataFrame(forecast_path(
                            symbol, interval=interval, horizon=horizon, df=df))
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")

        pred = st.session_state.get("pred")
        if pred and pred.get("symbol") == symbol and pred.get("interval") == interval:
            up = pred["direction"] == "UP"
            g_col, m_col = st.columns([1, 1])
            with g_col:
                st.plotly_chart(prob_gauge(pred["prob_up"]), use_container_width=True)
            with m_col:
                st.markdown(
                    f"<h2 style='color:{UP if up else DOWN};margin-bottom:0'>"
                    f"{'▲ UP' if up else '▼ DOWN'}</h2>", unsafe_allow_html=True)
                mm1, mm2 = st.columns(2)
                mm1.metric("Confidence", f"{pred['confidence']*100:.1f}%")
                mm2.metric("Last close", f"₹{pred['last_close']:,.2f}")
                st.caption(
                    f"LSTM {pred['lstm_prob_up']:.3f} · "
                    f"GBM {pred['gbm_prob_up'] if pred['gbm_prob_up'] is not None else '—'} · "
                    f"ensemble `{pred['ensemble']}` · as of {pred['as_of']}"
                )

        fc = st.session_state.get("fc")
        if fc is not None and not fc.empty:
            st.subheader("Forecast path")
            colors = [UP if d == "UP" else DOWN for d in fc["direction"]]
            fig = go.Figure(go.Scatter(
                x=fc["step"], y=fc["projected_close"], mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                marker=dict(color=colors, size=9), name="projected close"))
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)
            st.dataframe(fc, use_container_width=True, hide_index=True)

with tab_bt:
    with st.expander("⚙️ Backtest parameters", expanded=False):
        bc1, bc2, bc3 = st.columns(3)
        capital = bc1.number_input("Initial capital (₹)", 10_000, 10_000_000,
                                   100_000, step=10_000)
        thresh = bc2.slider("Entry prob threshold", 0.50, 0.80, 0.55, 0.01)
        max_dd = bc3.slider("Max drawdown stop", 0.05, 0.50, 0.20, 0.05)
        bc4, bc5, bc6 = st.columns(3)
        tp = bc4.slider("Take-profit (×ATR)", 1.0, 4.0, 2.0, 0.5)
        sl = bc5.slider("Stop-loss (×ATR)", 0.5, 3.0, 1.0, 0.5)
        hold = bc6.slider("Max hold (bars)", 3, 40, 10)

    if st.button("Run risk-aware backtest", type="primary"):
        with st.spinner("Backtesting…"):
            try:
                st.session_state["bt"] = backtest_with_risk(
                    df, prob_threshold=thresh, tp_mult=tp, sl_mult=sl,
                    max_hold=hold, initial_capital=capital,
                    max_drawdown_stop=max_dd)
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    bt = st.session_state.get("bt")
    if bt:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total return", f"{bt['total_return']*100:.2f}%")
        m2.metric("Win rate", f"{bt['win_rate']*100:.1f}%")
        m3.metric("Sharpe", f"{bt['sharpe']:.2f}")
        m4.metric("Max drawdown", f"{bt['max_drawdown']*100:.1f}%")
        m5.metric("Trades", bt["n_trades"])
        if bt["halted"]:
            st.warning("⚠️ Drawdown circuit breaker halted trading during this run.")

        eq = bt["equity_curve"]
        cap0 = float(eq.iloc[0])
        benchmark = cap0 * (df["Close"].reindex(eq.index).ffill() /
                            df["Close"].reindex(eq.index).ffill().iloc[0])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy",
                                 line=dict(color=ACCENT, width=2),
                                 fill="tozeroy", fillcolor="rgba(0,196,154,0.08)"))
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values,
                                 name="Buy &amp; hold",
                                 line=dict(color="#7A8699", width=1.4, dash="dot")))
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

        trades = bt["trades"]
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            st.subheader("Trades")
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download trades", trades.to_csv(index=False),
                               file_name=f"trades_{symbol}_{interval}.csv",
                               mime="text/csv")
    else:
        st.caption("Set parameters above, then run a backtest to see the equity "
                   "curve, metrics, and trade log.")

with tab_live:
    st.caption(
        "Auto-refreshing view of the latest scheduler / CLI output. "
        "Start the scheduler in a separate terminal with "
        "`python -m src.run --schedule`."
    )
    auto = st.checkbox("Auto-refresh", value=True)
    every = st.select_slider("Refresh interval (s)", options=[2, 5, 10, 30],
                             value=5) if auto else None

    @st.fragment(run_every=every)
    def live_panel():
        now_ts = datetime.now().timestamp()
        if os.path.exists(LOG_FILE):
            mtime = os.path.getmtime(LOG_FILE)
            age = int(now_ts - mtime)
            status = "🟢 active" if age < 120 else ("🟡 idle" if age < 3600 else "⚪ quiet")
            st.markdown(
                f"**Activity:** {status} — last log write {age}s ago "
                f"({datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M:%S})"
            )
        else:
            st.info("No system.log yet — start the scheduler or run any CLI command.")

        lp = latest_report("predictions")
        if lp:
            st.subheader("Latest predictions")
            st.caption(f"{os.path.basename(lp)} · "
                       f"{datetime.fromtimestamp(os.path.getmtime(lp)):%Y-%m-%d %H:%M}")
            st.dataframe(pd.read_csv(lp), use_container_width=True, hide_index=True)
        else:
            st.caption("No predictions report yet — they appear here once a "
                       "predict job runs.")

        st.subheader("Recent log")
        st.code(tail_log(40) or "(log is empty)", language="log")

    live_panel()

with tab_reports:
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "*.csv")),
                   key=os.path.getmtime, reverse=True)
    if not files:
        st.caption("No reports yet. Run training/prediction/backtest from the CLI "
                   "or this dashboard to generate them.")
    else:
        names = [os.path.basename(f) for f in files]
        choice = st.selectbox("Report file", names)
        path = os.path.join(REPORT_DIR, choice)
        st.caption(f"Modified {datetime.fromtimestamp(os.path.getmtime(path)):%Y-%m-%d %H:%M}")
        rep = pd.read_csv(path)
        st.dataframe(rep, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", rep.to_csv(index=False),
                           file_name=choice, mime="text/csv")

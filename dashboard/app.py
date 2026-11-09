"""
dashboard/app.py
────────────────
Streamlit dashboard — live signals + historical prediction performance.

Run:
    streamlit run dashboard/app.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DB_PATH, MODEL_DIR, RECENT_PREDICTIONS_SHOWN,
    DASHBOARD_REFRESH_SECONDS, BINANCE_BASE_URL, SYMBOL,
)
from live.recorder import PredictionRecorder

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Candle AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def get_recent_predictions(n: int = RECENT_PREDICTIONS_SHOWN) -> pd.DataFrame:
    rec = PredictionRecorder(DB_PATH)
    return rec.get_recent(n)


@st.cache_data(ttl=DASHBOARD_REFRESH_SECONDS)
def get_stats() -> dict:
    rec = PredictionRecorder(DB_PATH)
    return rec.get_stats()


@st.cache_data(ttl=30)
def get_current_price() -> float:
    import requests
    try:
        r = requests.get(f"{BINANCE_BASE_URL}/api/v3/ticker/price",
                         params={"symbol": SYMBOL}, timeout=5)
        return float(r.json()["price"])
    except Exception:
        return 0.0


@st.cache_data(ttl=30)
def get_recent_5m_candles(n: int = 20) -> pd.DataFrame:
    import requests
    try:
        r = requests.get(f"{BINANCE_BASE_URL}/api/v3/klines",
                         params={"symbol": SYMBOL, "interval": "5m", "limit": n},
                         timeout=5)
        raw = r.json()
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "nt", "tbv", "tbqv", "ign"]
        df = pd.DataFrame(raw, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open_time", "open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


def load_model_info() -> dict:
    ptr = MODEL_DIR / "current_model.json"
    if ptr.exists():
        return json.loads(ptr.read_text())
    return {}


def load_registry() -> dict:
    reg = MODEL_DIR / "model_registry.json"
    if reg.exists():
        return json.loads(reg.read_text())
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Signal colour helpers
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {"UP": "#00c853", "DOWN": "#d50000", "SKIP": "#888888"}

def signal_badge(signal: str) -> str:
    c = SIGNAL_COLORS.get(signal, "#888888")
    return f'<span style="background:{c};color:white;padding:2px 10px;border-radius:4px;font-weight:bold;">{signal}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("₿ BTC Candle AI")
    st.caption("5-minute candle direction predictor")
    st.divider()

    model_info = load_model_info()
    if model_info:
        st.info(f"Active model: **{model_info.get('model_name', '—')}**")
    else:
        st.warning("No trained model found.\nRun: `python -m models.train`")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh every (s)", 5, 60, DASHBOARD_REFRESH_SECONDS)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("BTC/USDT 5-Minute Candle Prediction")

# ── Row 1: KPI tiles ──────────────────────────────────────────────────────────
price  = get_current_price()
stats  = get_stats()
recent = get_recent_predictions()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("BTC Price", f"${price:,.2f}" if price else "—")

with col2:
    total = stats.get("total", 0) or 0
    st.metric("Total Predictions", total)

with col3:
    decided = stats.get("decided", 0) or 0
    st.metric("Decided (non-SKIP)", decided)

with col4:
    acc = stats.get("overall_accuracy")
    st.metric("Accuracy", f"{acc*100:.1f}%" if acc is not None else "—")

with col5:
    skip_count = stats.get("skipped", 0) or 0
    skip_pct = (skip_count / total * 100) if total else 0
    st.metric("Skip Rate", f"{skip_pct:.1f}%")

st.divider()

# ── Row 2: Latest signal + 5-min candle chart ─────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Latest Signal")
    if not recent.empty:
        last = recent.iloc[0]
        signal = last["signal"]
        prob   = last["prob_up"]
        cot    = last["candle_open_time"]
        resolved = last["actual_direction"] if pd.notna(last.get("actual_direction")) else None

        st.markdown(f"**Candle:** {cot[:16]} UTC")
        st.markdown(signal_badge(signal), unsafe_allow_html=True)
        st.metric("P(UP)", f"{prob:.2%}")

        if resolved:
            match = signal == resolved
            result_label = "✅ Correct" if match else ("❌ Wrong" if signal != "SKIP" else "—")
            st.info(f"Actual: **{resolved}**  |  {result_label}")
        else:
            st.caption("Awaiting candle close…")

        # Countdown to next prediction
        now = datetime.now(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        elapsed = (now - epoch).total_seconds()
        candle_start_s = (elapsed // 300) * 300
        secs_in_candle = elapsed - candle_start_s
        to_4min = 240 - secs_in_candle
        if to_4min > 0:
            st.info(f"⏱ Next prediction in **{to_4min:.0f}s**")
        else:
            to_close = 300 - secs_in_candle + 2
            st.success(f"📡 Prediction active — candle closes in **{max(to_close,0):.0f}s**")
    else:
        st.info("No predictions yet. Start the live predictor:\n\n`python -m live.predictor`")

with right:
    st.subheader("Recent 5-min Candles (BTCUSDT)")
    df_candles = get_recent_5m_candles(20)
    if not df_candles.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_candles["open_time"],
            open=df_candles["open"],
            high=df_candles["high"],
            low=df_candles["low"],
            close=df_candles["close"],
            increasing_line_color="#00c853",
            decreasing_line_color="#d50000",
        )])
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=250,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            xaxis=dict(color="white"),
            yaxis=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Unable to fetch live candles.")

st.divider()

# ── Row 3: Model performance ───────────────────────────────────────────────────
st.subheader("Model Performance")

registry = load_registry()
if registry:
    perf_rows = []
    for name, info in registry.items():
        m = info.get("test_metrics", {})
        perf_rows.append({
            "Model":     name,
            "Accuracy":  m.get("accuracy"),
            "Precision": m.get("precision"),
            "Recall":    m.get("recall"),
            "F1":        m.get("f1"),
            "ROC-AUC":   m.get("roc_auc"),
            "Brier":     m.get("brier_score"),
            "Skip %":    m.get("skip_pct"),
            "Trained at": info.get("trained_at", "")[:16],
        })
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True)

    # Bar chart: accuracy comparison
    fig2 = px.bar(
        perf_df.dropna(subset=["Accuracy"]),
        x="Model", y="Accuracy",
        color="Model",
        color_discrete_map={"logistic": "#4e79a7", "lightgbm": "#f28e2b", "xgboost": "#59a14f"},
        text_auto=".3f",
        title="Test Set Accuracy by Model",
    )
    fig2.update_layout(
        showlegend=False,
        height=300,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No trained models found. Run `python -m models.train` first.")

st.divider()

# ── Row 4: UP/DOWN accuracy breakdown ────────────────────────────────────────
if stats.get("decided", 0):
    st.subheader("Signal Accuracy Breakdown")
    col_a, col_b = st.columns(2)

    with col_a:
        up_total   = stats.get("up_total", 0) or 0
        up_correct = stats.get("up_correct", 0) or 0
        up_acc = up_correct / up_total if up_total else 0
        st.metric("UP Accuracy", f"{up_acc:.1%}", f"{up_correct}/{up_total}")

    with col_b:
        dn_total   = stats.get("down_total", 0) or 0
        dn_correct = stats.get("down_correct", 0) or 0
        dn_acc = dn_correct / dn_total if dn_total else 0
        st.metric("DOWN Accuracy", f"{dn_acc:.1%}", f"{dn_correct}/{dn_total}")

    st.divider()

# ── Row 5: Recent predictions table ──────────────────────────────────────────
st.subheader(f"Last {RECENT_PREDICTIONS_SHOWN} Predictions")

if not recent.empty:
    display = recent[[
        "candle_open_time", "signal", "prob_up",
        "actual_direction", "correct",
        "model_version", "predicted_at",
    ]].copy()

    display["prob_up"] = display["prob_up"].map("{:.4f}".format)
    display["candle_open_time"] = display["candle_open_time"].str[:16]
    display["predicted_at"]     = display["predicted_at"].str[:16]

    # actual_direction: UP / DOWN / DOJI / (pending)
    def _fmt_direction(x):
        if not x or not isinstance(x, str) or x != x:  # None / NaN / empty
            return "(pending)"
        return x  # UP, DOWN, DOJI as-is

    display["actual_direction"] = display["actual_direction"].apply(_fmt_direction)

    # Result: compare signal vs actual_direction explicitly
    def _fmt_result(row):
        sig = row["signal"]
        act = row["actual_direction"]
        if act == "(pending)":
            return "⏳"
        if sig == "SKIP" or act == "DOJI":
            return "—"
        return "✅" if sig == act else "❌"

    display["result"] = display.apply(_fmt_result, axis=1)

    # Drop the raw correct column (replaced by result)
    display = display.drop(columns=["correct"])

    # Rename for readability
    display = display.rename(columns={
        "candle_open_time": "Candle (UTC)",
        "signal":           "Prediction",
        "prob_up":          "P(UP)",
        "actual_direction": "Actual (Binance 5m)",
        "result":           "Result",
        "model_version":    "Model",
        "predicted_at":     "Predicted At",
    })

    st.dataframe(display, use_container_width=True, height=400)

    # Rolling accuracy chart
    df_resolved = recent[recent["correct"].notna()].copy()
    if len(df_resolved) >= 5:
        df_resolved = df_resolved.sort_values("candle_open_time")
        df_resolved["rolling_acc"] = df_resolved["correct"].rolling(min(10, len(df_resolved))).mean()
        fig3 = px.line(
            df_resolved,
            x="candle_open_time",
            y="rolling_acc",
            title="Rolling Accuracy (10-candle window)",
            labels={"rolling_acc": "Accuracy", "candle_open_time": "Time"},
        )
        fig3.add_hline(y=0.5, line_dash="dash", line_color="grey", annotation_text="50% baseline")
        fig3.update_layout(
            height=250,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No predictions recorded yet.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

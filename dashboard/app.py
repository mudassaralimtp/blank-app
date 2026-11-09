"""
dashboard/app.py
────────────────
Streamlit dashboard — live BTCUSDT 5-min candle signals + performance history.

Cloud-deployment design
───────────────────────
• A single background thread runs the prediction loop so Streamlit reruns
  never trigger duplicate predictions.  The thread is started once and stored
  in st.session_state under the key "_predictor_thread".

• The thread communicates with the dashboard via a shared PredictionRecorder
  instance stored in st.session_state["_recorder"].  All writes in the thread
  are protected by the recorder's internal SQLite connection (thread-safe) or
  the in-memory lock.

• The dashboard auto-refreshes every DASHBOARD_REFRESH_SECONDS via
  st.rerun(); the thread keeps running transparently between reruns.

• Historical data and models are NOT downloaded or retrained on startup.
  They are expected to be committed to the repo (models/saved/).
  If a model is missing the dashboard degrades gracefully.

• A manual "Refresh Market Data" button clears @st.cache_data caches so the
  candlestick chart and price feed update immediately.

Run locally:
    streamlit run dashboard/app.py

Deploy:
    Push to GitHub → connect repo at share.streamlit.io → entry point = streamlit_app.py
"""

import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Path bootstrap (needed when launched via streamlit_app.py shim) ───────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config import (
    MODEL_DIR, RECENT_PREDICTIONS_SHOWN,
    DASHBOARD_REFRESH_SECONDS, BINANCE_BASE_URL, SYMBOL,
    DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD,
    PREDICTION_OFFSET_SECONDS, CANDLE_DURATION_SECONDS,
)
from live.recorder import PredictionRecorder, RECORDER_WARN

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Candle AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers (cached — never re-load unless cache expires)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_model_cached():
    """
    Load the trained model once per app lifetime.
    Returns (model_name, model_object) or (None, None) if no model exists.

    The path stored in current_model.json may be absolute (local dev) or
    relative to the project root (cloud deployment).  We try both.
    """
    import joblib

    ptr = MODEL_DIR / "current_model.json"
    if not ptr.exists():
        return None, None
    try:
        info = json.loads(ptr.read_text())
        model_name = info["model_name"]
        raw_path   = info["path"]

        # Resolve: absolute path first, then relative to project root
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = _ROOT / raw_path
        if not candidate.exists():
            # Last resort: look directly in MODEL_DIR
            candidate = MODEL_DIR / f"{model_name}.joblib"
        if not candidate.exists():
            log.error("Model file not found: %s", raw_path)
            return None, None

        model = joblib.load(candidate)
        return model_name, model
    except Exception as exc:
        log.error("Model load failed: %s", exc)
        return None, None


@st.cache_resource
def _get_recorder() -> PredictionRecorder:
    """One recorder instance shared across all reruns (SQLite or in-memory)."""
    return PredictionRecorder()


def _load_thresholds(model_name: Optional[str]) -> tuple:
    if model_name is None:
        return DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD
    reg_path = MODEL_DIR / "model_registry.json"
    if not reg_path.exists():
        return DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD
    try:
        reg = json.loads(reg_path.read_text())
        info = reg.get(model_name, {})
        return (
            info.get("up_threshold",   DEFAULT_UP_THRESHOLD),
            info.get("down_threshold", DEFAULT_DOWN_THRESHOLD),
        )
    except Exception:
        return DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD


def _load_model_info() -> dict:
    ptr = MODEL_DIR / "current_model.json"
    if ptr.exists():
        try:
            return json.loads(ptr.read_text())
        except Exception:
            pass
    return {}


def _load_registry() -> dict:
    reg = MODEL_DIR / "model_registry.json"
    if reg.exists():
        try:
            return json.loads(reg.read_text())
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Binance market data (short TTL caches)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def _get_current_price() -> float:
    try:
        r = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/ticker/price",
            params={"symbol": SYMBOL}, timeout=5,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return 0.0


@st.cache_data(ttl=30)
def _get_recent_5m_candles(n: int = 20) -> pd.DataFrame:
    try:
        r = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/klines",
            params={"symbol": SYMBOL, "interval": "5m", "limit": n},
            timeout=5,
        )
        r.raise_for_status()
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


# ─────────────────────────────────────────────────────────────────────────────
# Prediction loop (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

def _current_candle_open(now: datetime) -> datetime:
    epoch   = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed = (now - epoch).total_seconds()
    return datetime.fromtimestamp(
        (elapsed // CANDLE_DURATION_SECONDS) * CANDLE_DURATION_SECONDS,
        tz=timezone.utc,
    )


def _seconds_until_4min_mark(now: datetime) -> float:
    open_ = _current_candle_open(now)
    return open_.timestamp() + PREDICTION_OFFSET_SECONDS - now.timestamp()


def _seconds_until_candle_close(now: datetime) -> float:
    open_ = _current_candle_open(now)
    return open_.timestamp() + CANDLE_DURATION_SECONDS + 2 - now.timestamp()


def _prediction_loop(model_name: str, model, up_thr: float, down_thr: float,
                     recorder: PredictionRecorder, stop_event: threading.Event) -> None:
    """
    Background thread: sleep → predict at 4-min mark → record → sleep → resolve.
    Never blocks the Streamlit main thread.
    """
    from live.predictor import make_prediction, resolve_candle

    log.info("[thread] Prediction loop started: model=%s  up≥%.2f  down≤%.2f",
             model_name, up_thr, 1 - down_thr)

    while not stop_event.is_set():
        try:
            now  = datetime.now(timezone.utc)
            wait = _seconds_until_4min_mark(now)

            if wait > 0:
                log.info("[thread] Sleeping %.1f s until 4-min mark", wait)
                # Sleep in small chunks so the stop_event is checked regularly
                _interruptible_sleep(wait, stop_event)
                if stop_event.is_set():
                    break

            # ── Make prediction ───────────────────────────────────────────────
            signal, prob_up, candle_open = make_prediction(
                model, model_name, up_thr, down_thr
            )

            recorder.record_prediction(
                candle_open_time=candle_open,
                model_version=model_name,
                prob_up=prob_up,
                signal=signal,
                up_threshold=up_thr,
                down_threshold=down_thr,
            )

            # ── Sleep until candle close ──────────────────────────────────────
            now2  = datetime.now(timezone.utc)
            wait2 = _seconds_until_candle_close(now2)
            if wait2 > 0:
                log.info("[thread] Signal: %s | Waiting %.1f s for candle close", signal, wait2)
                _interruptible_sleep(wait2, stop_event)
                if stop_event.is_set():
                    break

            # ── Resolve outcome ───────────────────────────────────────────────
            resolve_candle(candle_open, recorder)

            # Small gap before the next candle's 4-min mark
            _interruptible_sleep(3, stop_event)

        except Exception as exc:
            log.error("[thread] Prediction loop error: %s", exc, exc_info=True)
            # Back off to avoid hammering the API on repeated failures
            _interruptible_sleep(30, stop_event)


def _interruptible_sleep(seconds: float, stop_event: threading.Event,
                         chunk: float = 2.0) -> None:
    """Sleep for `seconds` but wake early if stop_event is set."""
    remaining = seconds
    while remaining > 0 and not stop_event.is_set():
        time.sleep(min(chunk, remaining))
        remaining -= chunk


def _ensure_predictor_running() -> None:
    """
    Start the background prediction thread exactly once per Streamlit session.
    Subsequent reruns find the running thread in session_state and skip.
    """
    model_name, model = _load_model_cached()
    if model is None:
        # No model available — show a warning but don't crash
        if "predictor_no_model" not in st.session_state:
            st.session_state["predictor_no_model"] = True
        return

    # Already running?
    thread: Optional[threading.Thread] = st.session_state.get("_predictor_thread")
    if thread is not None and thread.is_alive():
        return

    up_thr, down_thr = _load_thresholds(model_name)
    recorder         = _get_recorder()
    stop_event       = threading.Event()

    t = threading.Thread(
        target=_prediction_loop,
        args=(model_name, model, up_thr, down_thr, recorder, stop_event),
        daemon=True,
        name="btc-predictor",
    )
    t.start()

    st.session_state["_predictor_thread"] = t
    st.session_state["_predictor_stop"]   = stop_event
    log.info("Predictor thread started (daemon).")


# ─────────────────────────────────────────────────────────────────────────────
# Signal colour helpers
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {"UP": "#00c853", "DOWN": "#d50000", "SKIP": "#888888"}

def _signal_badge(signal: str) -> str:
    c = SIGNAL_COLORS.get(signal, "#888888")
    return (
        f'<span style="background:{c};color:white;padding:3px 12px;'
        f'border-radius:4px;font-weight:bold;font-size:1.1em;">{signal}</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard entry point — everything below runs on every Streamlit rerun
# ─────────────────────────────────────────────────────────────────────────────

# 1. Start/check the background predictor (no-op after the first call)
_ensure_predictor_running()

recorder = _get_recorder()

# ── Global banners ────────────────────────────────────────────────────────────
if RECORDER_WARN:
    st.warning(RECORDER_WARN)

if st.session_state.get("predictor_no_model"):
    st.error(
        "No trained model found in `models/saved/`. "
        "Live predictions are disabled. "
        "Commit your trained models to the repository before deploying."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("₿ BTC Candle AI")
    st.caption("5-minute candle direction predictor")
    st.divider()

    model_info = _load_model_info()
    if model_info:
        st.info(f"Active model: **{model_info.get('model_name', '—')}**")
    else:
        st.warning("No trained model found.\nCommit models to `models/saved/` before deploying.")

    # Thread status indicator
    thread: Optional[threading.Thread] = st.session_state.get("_predictor_thread")
    if thread is not None and thread.is_alive():
        st.success("🟢 Predictor running")
    elif model_info:
        st.error("🔴 Predictor stopped")

    st.divider()

    auto_refresh = st.checkbox("Auto-refresh dashboard", value=True)
    refresh_interval = st.slider("Refresh every (s)", 5, 60, DASHBOARD_REFRESH_SECONDS)

    if st.button("🔄 Refresh Market Data"):
        _get_current_price.clear()
        _get_recent_5m_candles.clear()
        st.rerun()

    st.divider()
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    st.caption("Data: Binance public REST API — no keys required")
    st.caption("⚠️ No trades are placed by this system")


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("BTC/USDT 5-Minute Candle Prediction")

# ── Row 1: KPI tiles ──────────────────────────────────────────────────────────
price  = _get_current_price()
stats  = recorder.get_stats()
recent = recorder.get_recent(RECENT_PREDICTIONS_SHOWN)

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
    skip_pct   = (skip_count / total * 100) if total else 0
    st.metric("Skip Rate", f"{skip_pct:.1f}%")

st.divider()

# ── Row 2: Latest signal + candlestick chart ───────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Latest Signal")

    # Countdown clock (purely cosmetic — computed from wall clock)
    now_utc    = datetime.now(timezone.utc)
    epoch      = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed    = (now_utc - epoch).total_seconds()
    candle_s   = (elapsed // CANDLE_DURATION_SECONDS) * CANDLE_DURATION_SECONDS
    secs_in    = elapsed - candle_s
    to_4min    = PREDICTION_OFFSET_SECONDS - secs_in

    if to_4min > 0:
        mins, secs = divmod(int(to_4min), 60)
        countdown_txt = f"⏱ Next prediction in **{mins}m {secs:02d}s**"
    else:
        to_close = CANDLE_DURATION_SECONDS - secs_in + 2
        countdown_txt = f"📡 Prediction active — candle closes in **{max(int(to_close), 0)}s**"

    if not recent.empty:
        last     = recent.iloc[0]
        signal   = last["signal"]
        prob_up  = last["prob_up"]
        cot      = last["candle_open_time"]
        resolved = last["actual_direction"] if pd.notna(last.get("actual_direction")) else None

        st.markdown(f"**Candle:** {str(cot)[:16]} UTC")
        st.markdown(_signal_badge(signal), unsafe_allow_html=True)
        st.metric("P(UP)", f"{float(prob_up):.2%}")

        if resolved:
            match        = signal == resolved
            result_label = (
                "✅ Correct" if match else
                ("❌ Wrong"  if signal != "SKIP" else "—")
            )
            st.info(f"Actual: **{resolved}**  |  {result_label}")
        else:
            st.caption("Awaiting candle close…")

        st.info(countdown_txt)
    else:
        st.info("No predictions recorded yet.\nThe predictor will begin at the next 4-minute mark.")
        st.info(countdown_txt)

with right:
    st.subheader("Recent 5-min Candles (BTCUSDT)")
    df_candles = _get_recent_5m_candles(20)
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
            height=260,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            xaxis=dict(color="white"),
            yaxis=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Unable to fetch live candles — check your internet connection.")

st.divider()

# ── Row 3: Model performance ───────────────────────────────────────────────────
st.subheader("Model Performance")

registry = _load_registry()
if registry:
    perf_rows = []
    for name, info in registry.items():
        m = info.get("test_metrics", {})
        perf_rows.append({
            "Model":      name,
            "Accuracy":   m.get("accuracy"),
            "Precision":  m.get("precision"),
            "Recall":     m.get("recall"),
            "F1":         m.get("f1"),
            "ROC-AUC":    m.get("roc_auc"),
            "Brier":      m.get("brier_score"),
            "Skip %":     m.get("skip_pct"),
            "Trained at": info.get("trained_at", "")[:16],
        })
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True)

    fig2 = px.bar(
        perf_df.dropna(subset=["Accuracy"]),
        x="Model", y="Accuracy",
        color="Model",
        color_discrete_map={
            "logistic":  "#4e79a7",
            "lightgbm":  "#f28e2b",
            "xgboost":   "#59a14f",
        },
        text_auto=".3f",
        title="Test-Set Accuracy by Model",
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
    st.info(
        "No trained models found. "
        "Run `python run.py train` locally, commit `models/saved/`, then redeploy."
    )

st.divider()

# ── Row 4: UP/DOWN accuracy breakdown ────────────────────────────────────────
if stats.get("decided", 0):
    st.subheader("Signal Accuracy Breakdown")
    col_a, col_b = st.columns(2)

    with col_a:
        up_total   = stats.get("up_total",   0) or 0
        up_correct = stats.get("up_correct", 0) or 0
        up_acc     = up_correct / up_total if up_total else 0
        st.metric("UP Accuracy",   f"{up_acc:.1%}", f"{up_correct}/{up_total}")

    with col_b:
        dn_total   = stats.get("down_total",   0) or 0
        dn_correct = stats.get("down_correct", 0) or 0
        dn_acc     = dn_correct / dn_total if dn_total else 0
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

    display["prob_up"]          = display["prob_up"].map("{:.4f}".format)
    display["candle_open_time"] = display["candle_open_time"].astype(str).str[:16]
    display["predicted_at"]     = display["predicted_at"].astype(str).str[:16]

    def _fmt_direction(x):
        if x is None or (isinstance(x, float) and pd.isna(x)) or x == "":
            return "(pending)"
        return x

    display["actual_direction"] = display["actual_direction"].apply(_fmt_direction)

    def _fmt_result(row):
        sig = row["signal"]
        act = row["actual_direction"]
        if act == "(pending)":
            return "⏳"
        if sig == "SKIP" or act == "DOJI":
            return "—"
        return "✅" if sig == act else "❌"

    display["result"] = display.apply(_fmt_result, axis=1)
    display = display.drop(columns=["correct"])
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
        window = min(10, len(df_resolved))
        df_resolved["rolling_acc"] = (
            df_resolved["correct"].astype(float).rolling(window).mean()
        )
        fig3 = px.line(
            df_resolved,
            x="candle_open_time",
            y="rolling_acc",
            title=f"Rolling Accuracy ({window}-candle window)",
            labels={"rolling_acc": "Accuracy", "candle_open_time": "Time"},
        )
        fig3.add_hline(y=0.5, line_dash="dash", line_color="grey",
                       annotation_text="50% baseline")
        fig3.update_layout(
            height=250,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No predictions recorded yet. The first prediction will appear here at the 4-minute mark.")

# ── Auto-refresh (non-blocking) ───────────────────────────────────────────────
if auto_refresh:
    # st.rerun() after a sleep would block. We use time.sleep here only for the
    # brief rerun cadence (≤ 60 s); the prediction loop itself runs in the
    # background thread and is completely independent of this sleep.
    time.sleep(refresh_interval)
    st.rerun()

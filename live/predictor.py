"""
predictor.py
────────────
Live prediction service for BTC/USDT 5-minute candle direction.

Workflow (runs in a loop):
  1. Calculate time remaining until the 4-minute mark of the current 5-min candle.
  2. Sleep until that mark.
  3. Fetch the last 4 completed 1-min bars (m1–m4) of the current candle.
  4. Fetch recent history for rolling indicators.
  5. Compute features.
  6. Predict UP / DOWN / SKIP.
  7. Log the prediction.
  8. Sleep until candle close.
  9. Fetch the official Binance 5-min candle close.
  10. Resolve the prediction (compare with actual).
  11. Repeat.

Safety guarantees:
  - Only closed 1-min bars are used (Binance kline with close_time < now).
  - The 5th-minute bar is fetched only AFTER the candle closes.
  - SKIP is returned if fewer than 4 closed bars are available or data is stale.

Usage:
    python -m live.predictor
    python -m live.predictor --model lightgbm
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BINANCE_BASE_URL, SYMBOL,
    KLINE_LIMIT, MODEL_DIR,
    DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD,
    PREDICTION_OFFSET_SECONDS, CANDLE_DURATION_SECONDS,
)
from data.features import build_live_features
from live.recorder import PredictionRecorder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Binance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_klines(interval: str, limit: int = 100) -> pd.DataFrame:
    """Fetch the most-recent `limit` klines for SYMBOL/interval."""
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_volume"]:
        df[c] = df[c].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    return df.drop(columns=["ignore"]).sort_values("open_time").reset_index(drop=True)


def _get_closed_klines_before(interval: str, before_ts: datetime, limit: int) -> pd.DataFrame:
    """Return up to `limit` klines whose close_time < before_ts."""
    df = _get_klines(interval, limit=limit + 5)   # fetch a few extra
    now_utc = before_ts.replace(tzinfo=timezone.utc) if before_ts.tzinfo is None else before_ts
    return df[df["close_time"] < now_utc].tail(limit).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _current_candle_open(now: datetime) -> datetime:
    """Return the open_time of the current 5-min candle."""
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed = (now - epoch).total_seconds()
    candle_start = (elapsed // CANDLE_DURATION_SECONDS) * CANDLE_DURATION_SECONDS
    return datetime.fromtimestamp(candle_start, tz=timezone.utc)


def _seconds_until_4min_mark(now: datetime) -> float:
    """Seconds to sleep until the 4-minute mark of the current candle."""
    candle_open = _current_candle_open(now)
    target = candle_open.timestamp() + PREDICTION_OFFSET_SECONDS
    remaining = target - now.timestamp()
    return remaining


def _seconds_until_candle_close(now: datetime) -> float:
    """Seconds to sleep until the current 5-min candle closes (+2s buffer)."""
    candle_open = _current_candle_open(now)
    target = candle_open.timestamp() + CANDLE_DURATION_SECONDS + 2
    return target - now.timestamp()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: Optional[str] = None):
    """Load a trained model. Defaults to the current best model."""
    if model_name is None:
        ptr = MODEL_DIR / "current_model.json"
        if not ptr.exists():
            raise FileNotFoundError("No trained model found. Run: python -m models.train")
        with open(ptr) as f:
            info = json.load(f)
        model_name = info["model_name"]
        model_path = Path(info["path"])
    else:
        model_path = MODEL_DIR / f"{model_name}.joblib"

    log.info("Loading model: %s from %s", model_name, model_path)
    return model_name, joblib.load(model_path)


def load_thresholds(model_name: str) -> tuple:
    reg_path = MODEL_DIR / "model_registry.json"
    if not reg_path.exists():
        return DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD
    with open(reg_path) as f:
        reg = json.load(f)
    info = reg.get(model_name, {})
    return (
        info.get("up_threshold",   DEFAULT_UP_THRESHOLD),
        info.get("down_threshold", DEFAULT_DOWN_THRESHOLD),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single prediction
# ─────────────────────────────────────────────────────────────────────────────

def make_prediction(
    model,
    model_name:     str,
    up_threshold:   float,
    down_threshold: float,
) -> tuple:
    """
    Collect live data, build features, return (signal, prob_up, candle_open_time).
    Returns ("SKIP", 0.5, candle_open_time) if data is unavailable.
    """
    now = datetime.now(timezone.utc)
    candle_open = _current_candle_open(now)

    log.info("Predicting for candle: %s", candle_open.isoformat())

    # Fetch 4 closed 1-min bars belonging to the current 5-min candle
    # (their open_time must be >= candle_open)
    try:
        df_1m_recent = _get_closed_klines_before("1m", before_ts=now, limit=60)
        df_15m = _get_closed_klines_before("15m", before_ts=now, limit=50)
        df_1h  = _get_closed_klines_before("1h",  before_ts=now, limit=50)
    except Exception as exc:
        log.error("Data fetch failed: %s", exc)
        return "SKIP", 0.5, candle_open

    # Filter 1-min bars that belong to the current 5-min candle
    partial = df_1m_recent[df_1m_recent["open_time"] >= candle_open].copy()

    if len(partial) < 4:
        log.warning("Only %d closed 1-min bars available — SKIP", len(partial))
        return "SKIP", 0.5, candle_open

    # Use exactly the first 4 bars of the current candle
    partial = partial.iloc[:4].reset_index(drop=True)

    # History for rolling indicators (exclude current candle bars)
    history = df_1m_recent[df_1m_recent["open_time"] < candle_open].copy()
    if len(history) < 30:
        log.warning("Insufficient history (%d rows) for rolling features — SKIP", len(history))
        return "SKIP", 0.5, candle_open

    # Build features
    try:
        X_live = build_live_features(partial, history, df_15m, df_1h)
    except Exception as exc:
        log.error("Feature build failed: %s", exc)
        return "SKIP", 0.5, candle_open

    # Any NaN → SKIP
    if X_live.isnull().any().any():
        log.warning("NaN features detected — SKIP")
        return "SKIP", 0.5, candle_open

    # Predict
    prob_up = float(model.predict_proba(X_live)[0, 1])

    if prob_up >= up_threshold:
        signal = "UP"
    elif prob_up <= 1 - down_threshold:
        signal = "DOWN"
    else:
        signal = "SKIP"

    log.info("  prob_up=%.4f  →  %s  (thresholds: up≥%.2f  down≤%.2f)",
             prob_up, signal, up_threshold, 1 - down_threshold)
    return signal, prob_up, candle_open


# ─────────────────────────────────────────────────────────────────────────────
# Resolve outcome after candle closes
# ─────────────────────────────────────────────────────────────────────────────

def resolve_candle(candle_open: datetime, recorder: PredictionRecorder):
    """Fetch the closed 5-min Binance candle and resolve the stored prediction."""
    try:
        df = _get_closed_klines_before("5m", before_ts=datetime.now(timezone.utc), limit=5)
        row = df[df["open_time"] == candle_open]
        if row.empty:
            log.warning("5-min candle not found for %s", candle_open.isoformat())
            return
        actual_open  = float(row.iloc[0]["open"])
        actual_close = float(row.iloc[0]["close"])
        recorder.resolve_prediction(candle_open, actual_open, actual_close)
    except Exception as exc:
        log.error("Failed to resolve candle %s: %s", candle_open.isoformat(), exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_live(model_name: Optional[str] = None):
    model_name, model = load_model(model_name)
    up_threshold, down_threshold = load_thresholds(model_name)
    recorder = PredictionRecorder()

    log.info("Live predictor started — model=%s  up≥%.2f  down≤%.2f",
             model_name, up_threshold, 1 - down_threshold)

    while True:
        now = datetime.now(timezone.utc)

        # Sleep until the 4-minute mark
        wait = _seconds_until_4min_mark(now)
        if wait > 0:
            log.info("Sleeping %.1f s until 4-min mark …", wait)
            time.sleep(wait)

        # Make prediction
        signal, prob_up, candle_open = make_prediction(
            model, model_name, up_threshold, down_threshold
        )

        # Record it
        recorder.record_prediction(
            candle_open_time=candle_open,
            model_version=model_name,
            prob_up=prob_up,
            signal=signal,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
        )

        # Sleep until candle closes (remaining ~60 s + 2 s buffer)
        now2 = datetime.now(timezone.utc)
        wait2 = _seconds_until_candle_close(now2)
        if wait2 > 0:
            log.info("Signal: %s  |  Waiting %.1f s for candle to close …", signal, wait2)
            time.sleep(wait2)

        # Resolve actual outcome
        resolve_candle(candle_open, recorder)

        # Small buffer before the next candle's 4-min mark
        time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live BTC candle predictor")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (logistic / lightgbm / xgboost). Defaults to best model.")
    args = parser.parse_args()
    run_live(args.model)

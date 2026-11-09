"""
features.py
───────────
Builds the full feature matrix from the 4-minute snapshot dataset.

All features are computed using ONLY the data available at the 4-minute mark.
Higher-timeframe (15m, 1h) indicators are joined via an asof-merge so the
system only uses bars whose close_time is strictly before the prediction
timestamp — no future information ever leaks in.

Features produced
─────────────────
  Intra-candle (from the 4 partial 1-min bars):
    net_volume          – taker-buy - taker-sell proxy
    body_ratio          – |close - open| / (high - low + ε)
    upper_wick_ratio    – upper wick / total range
    lower_wick_ratio    – lower wick / total range
    intra_momentum      – m4_close / m1_open - 1
    m1m2_ret, m2m3_ret, m3m4_ret  – minute-by-minute returns
    volume_accel        – (m3+m4 vol) / (m1+m2 vol) ratio

  Rolling (computed on the running 1-min series up to the snap_close bar):
    obv                 – on-balance volume (last value)
    rsi_14              – RSI(14)
    ema_9, ema_21       – EMA crossover signal
    ema_cross           – ema_9 - ema_21
    vwap_dev            – snap_close deviation from rolling VWAP
    volume_spike_z      – z-score of current 4-min volume vs rolling window
    atr_14              – ATR(14) on 1-min bars

  Higher-timeframe:
    ema9_15m, ema21_15m, ema_cross_15m
    ema9_1h,  ema21_1h,  ema_cross_1h
    rsi_15m, rsi_1h

Usage:
    from data.features import build_features
    X, y = build_features(snapshots_df, df_1m, df_15m, df_1h)
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EMA_SHORT, EMA_LONG, RSI_PERIOD, ATR_PERIOD,
    VWAP_WINDOW, VOLUME_SPIKE_WINDOW, FEATURE_WARMUP_ROWS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level indicator helpers (pure numpy / pandas, no TA library dependency
# at the core so behaviour is fully transparent)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    tp = close  # simplified: use close as typical price proxy on 1-min bars
    return (tp * volume).rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)


def _volume_spike_z(volume: pd.Series, window: int) -> pd.Series:
    mu  = volume.rolling(window).mean()
    std = volume.rolling(window).std().replace(0, np.nan)
    return (volume - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
# Rolling feature builder (runs on the complete 1-min series)
# ─────────────────────────────────────────────────────────────────────────────

def _build_rolling_features_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling indicators on the 1-min series.
    Returns a DataFrame indexed by 1-min open_time containing:
        obv, rsi_14, ema_9, ema_21, ema_cross, vwap, volume_spike_z, atr_14
    """
    d = df_1m.copy().sort_values("open_time").reset_index(drop=True)

    d["obv"]            = _obv(d["close"], d["volume"])
    d["rsi_14"]         = _rsi(d["close"], RSI_PERIOD)
    d["ema_9"]          = _ema(d["close"], EMA_SHORT)
    d["ema_21"]         = _ema(d["close"], EMA_LONG)
    d["ema_cross"]      = d["ema_9"] - d["ema_21"]
    d["vwap"]           = _rolling_vwap(d["close"], d["volume"], VWAP_WINDOW)
    d["volume_spike_z"] = _volume_spike_z(d["volume"], VOLUME_SPIKE_WINDOW)
    d["atr_14"]         = _atr(d["high"], d["low"], d["close"], ATR_PERIOD)

    keep = ["open_time", "close", "obv", "rsi_14", "ema_9", "ema_21",
            "ema_cross", "vwap", "volume_spike_z", "atr_14"]
    return d[keep].set_index("open_time")


def _build_rolling_features_htf(df_htf: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Compute EMA and RSI on a higher-timeframe DataFrame."""
    d = df_htf.copy().sort_values("open_time").reset_index(drop=True)
    # Use close_time as the timestamp: the bar is fully known only after it closes
    d["ema_9"]       = _ema(d["close"], EMA_SHORT)
    d["ema_21"]      = _ema(d["close"], EMA_LONG)
    d["ema_cross"]   = d["ema_9"] - d["ema_21"]
    d[f"rsi_{suffix}"] = _rsi(d["close"], RSI_PERIOD)

    d = d.rename(columns={
        "ema_9":     f"ema9_{suffix}",
        "ema_21":    f"ema21_{suffix}",
        "ema_cross": f"ema_cross_{suffix}",
    })
    keep = ["close_time", f"ema9_{suffix}", f"ema21_{suffix}",
            f"ema_cross_{suffix}", f"rsi_{suffix}"]
    return d[keep].sort_values("close_time").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    snapshots: pd.DataFrame,
    df_1m:    pd.DataFrame,
    df_15m:   pd.DataFrame,
    df_1h:    pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the full feature matrix X and label series y.

    Parameters
    ----------
    snapshots : output of build_4min_snapshots()
    df_1m     : raw 1-min kline DataFrame
    df_15m    : raw 15-min kline DataFrame
    df_1h     : raw 1-hour kline DataFrame

    Returns
    -------
    X : pd.DataFrame  – feature matrix (no label columns)
    y : pd.Series     – binary labels (1=UP, 0=DOWN)
    """
    snaps = snapshots.copy().sort_values("candle_open_time").reset_index(drop=True)

    # ── 1. Intra-candle features ──────────────────────────────────────────────
    eps = 1e-10
    total_range = snaps["snap_high"] - snaps["snap_low"] + eps

    snaps["net_volume"]       = snaps["snap_taker_buy_vol"] - (snaps["snap_volume"] - snaps["snap_taker_buy_vol"])
    snaps["body_ratio"]       = (snaps["snap_close"] - snaps["snap_open"]).abs() / total_range
    snaps["upper_wick_ratio"] = (snaps["snap_high"]  - snaps[["snap_open", "snap_close"]].max(axis=1)) / total_range
    snaps["lower_wick_ratio"] = (snaps[["snap_open", "snap_close"]].min(axis=1) - snaps["snap_low"]) / total_range
    snaps["intra_momentum"]   = snaps["snap_close"] / (snaps["snap_open"] + eps) - 1

    # Minute-by-minute returns
    snaps["m1m2_ret"] = snaps["m2_close"] / (snaps["m1_close"] + eps) - 1
    snaps["m2m3_ret"] = snaps["m3_close"] / (snaps["m2_close"] + eps) - 1
    snaps["m3m4_ret"] = snaps["m4_close"] / (snaps["m3_close"] + eps) - 1

    # Volume acceleration: (m3+m4) / (m1+m2)
    snaps["volume_accel"] = (snaps["m3_volume"] + snaps["m4_volume"]) / (
        snaps["m1_volume"] + snaps["m2_volume"] + eps
    )

    # ── 2. Rolling 1-min features (joined by the 4th-minute open_time) ────────
    roll1m = _build_rolling_features_1m(df_1m)

    # The 4th minute bar's open_time is candle_open_time + 3 minutes
    snaps["m4_open_time"] = snaps["candle_open_time"] + pd.Timedelta(minutes=3)

    # Ensure timezone consistency: coerce both sides to UTC
    def _to_utc(s: pd.Series) -> pd.Series:
        if s.dt.tz is None:
            return s.dt.tz_localize("UTC")
        return s.dt.tz_convert("UTC")

    snaps["m4_open_time"] = _to_utc(snaps["m4_open_time"])

    # asof merge: take the most-recent 1-min row whose timestamp ≤ m4_open_time
    snaps_sorted = snaps.sort_values("m4_open_time")
    roll_sorted  = roll1m.sort_index()

    roll_reset = roll_sorted.reset_index().rename(columns={"open_time": "m4_open_time"})
    roll_reset["m4_open_time"] = _to_utc(roll_reset["m4_open_time"])

    merged_roll = pd.merge_asof(
        snaps_sorted.reset_index(),
        roll_reset,
        on="m4_open_time",
        direction="backward",
    ).set_index("index").sort_index()

    # vwap deviation (normalised)
    merged_roll["vwap_dev"] = (merged_roll["snap_close"] - merged_roll["vwap"]) / (merged_roll["vwap"] + eps)

    # ── 3. Higher-timeframe features ──────────────────────────────────────────
    htf_15m = _build_rolling_features_htf(df_15m, "15m")
    htf_1h  = _build_rolling_features_htf(df_1h,  "1h")

    # Prediction timestamp = candle_open_time + 4 minutes
    snaps["pred_time"] = _to_utc(snaps["candle_open_time"] + pd.Timedelta(minutes=4))

    # Merge 15m: use last 15m bar whose close_time ≤ pred_time
    base = merged_roll.reset_index(drop=True)
    # Use pd.Series assignment (not .values) to preserve UTC timezone
    base["pred_time"] = snaps["pred_time"].reset_index(drop=True)
    base["pred_time"] = _to_utc(base["pred_time"])

    htf_15m_r = htf_15m.rename(columns={"close_time": "pred_time"}).copy()
    htf_15m_r["pred_time"] = _to_utc(htf_15m_r["pred_time"])

    htf_1h_r = htf_1h.rename(columns={"close_time": "pred_time"}).copy()
    htf_1h_r["pred_time"] = _to_utc(htf_1h_r["pred_time"])

    base_sorted = base.sort_values("pred_time")

    base_sorted = pd.merge_asof(
        base_sorted,
        htf_15m_r,
        on="pred_time",
        direction="backward",
    )
    base_sorted = pd.merge_asof(
        base_sorted.sort_values("pred_time"),
        htf_1h_r,
        on="pred_time",
        direction="backward",
    )

    # ── 4. Assemble final feature list ────────────────────────────────────────
    feature_cols = [
        # intra-candle
        "net_volume", "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
        "intra_momentum", "m1m2_ret", "m2m3_ret", "m3m4_ret", "volume_accel",
        # rolling 1-min
        "obv", "rsi_14", "ema_9", "ema_21", "ema_cross", "vwap_dev",
        "volume_spike_z", "atr_14",
        # higher timeframe
        "ema9_15m", "ema21_15m", "ema_cross_15m", "rsi_15m",
        "ema9_1h",  "ema21_1h",  "ema_cross_1h",  "rsi_1h",
    ]

    # Keep only rows where ALL features are present
    df_full = base_sorted.sort_values("candle_open_time").reset_index(drop=True)
    df_full = df_full.dropna(subset=feature_cols)

    # Drop warmup rows to let rolling indicators stabilise
    df_full = df_full.iloc[FEATURE_WARMUP_ROWS:].reset_index(drop=True)

    X = df_full[feature_cols].copy()
    y = df_full["label"].astype(int)

    # Attach candle_open_time as index for time-aware splitting
    X.index = df_full["candle_open_time"]
    y.index = df_full["candle_open_time"]

    log.info("Feature matrix: %d rows × %d features  (UP=%d, DOWN=%d)",
             len(X), X.shape[1], y.sum(), (y == 0).sum())
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Live feature builder (single prediction)
# ─────────────────────────────────────────────────────────────────────────────

def build_live_features(
    partial_bars: pd.DataFrame,   # exactly 4 rows of 1-min bars (m1–m4)
    history_1m:   pd.DataFrame,   # recent history for rolling indicators
    df_15m:       pd.DataFrame,
    df_1h:        pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for live prediction.

    partial_bars : 4 rows with columns open/high/low/close/volume/
                   taker_buy_volume from the current partial 5-min candle.
    history_1m   : enough recent 1-min bars (≥ max(RSI_PERIOD, ATR_PERIOD, VWAP_WINDOW) rows)
                   NOT including the current partial candle's 5th bar.
    """
    # Combine history + partial bars for rolling calculations
    combined = pd.concat([history_1m, partial_bars]).sort_values("open_time").reset_index(drop=True)
    roll_features = _build_rolling_features_1m(combined).iloc[-1]  # last row = m4

    row: dict = {}

    # Intra-candle
    b = partial_bars.sort_values("open_time").reset_index(drop=True)
    snap_open  = float(b.iloc[0]["open"])
    snap_close = float(b.iloc[-1]["close"])
    snap_high  = float(b["high"].max())
    snap_low   = float(b["low"].min())
    snap_vol   = float(b["volume"].sum())
    snap_tbv   = float(b["taker_buy_volume"].sum())
    eps = 1e-10

    total_range = snap_high - snap_low + eps
    row["net_volume"]       = snap_tbv - (snap_vol - snap_tbv)
    row["body_ratio"]       = abs(snap_close - snap_open) / total_range
    row["upper_wick_ratio"] = (snap_high - max(snap_open, snap_close)) / total_range
    row["lower_wick_ratio"] = (min(snap_open, snap_close) - snap_low) / total_range
    row["intra_momentum"]   = snap_close / (snap_open + eps) - 1
    row["m1m2_ret"] = float(b.iloc[1]["close"]) / (float(b.iloc[0]["close"]) + eps) - 1
    row["m2m3_ret"] = float(b.iloc[2]["close"]) / (float(b.iloc[1]["close"]) + eps) - 1
    row["m3m4_ret"] = float(b.iloc[3]["close"]) / (float(b.iloc[2]["close"]) + eps) - 1
    vol_early = float(b.iloc[0]["volume"]) + float(b.iloc[1]["volume"]) + eps
    vol_late  = float(b.iloc[2]["volume"]) + float(b.iloc[3]["volume"])
    row["volume_accel"] = vol_late / vol_early

    # Rolling 1-min
    row["obv"]            = float(roll_features["obv"])
    row["rsi_14"]         = float(roll_features["rsi_14"])
    row["ema_9"]          = float(roll_features["ema_9"])
    row["ema_21"]         = float(roll_features["ema_21"])
    row["ema_cross"]      = float(roll_features["ema_cross"])
    vwap_val              = float(roll_features["vwap"]) if not np.isnan(roll_features["vwap"]) else snap_close
    row["vwap_dev"]       = (snap_close - vwap_val) / (vwap_val + eps)
    row["volume_spike_z"] = float(roll_features["volume_spike_z"]) if not np.isnan(roll_features["volume_spike_z"]) else 0.0
    row["atr_14"]         = float(roll_features["atr_14"])

    # Higher-timeframe (use latest completed bar)
    pred_time = b.iloc[-1]["open_time"]  # ~4-min mark

    def _latest_htf_indicators(df_htf: pd.DataFrame, suffix: str) -> dict:
        d = df_htf[df_htf["close_time"] <= pred_time].sort_values("close_time")
        if d.empty:
            return {f"ema9_{suffix}": np.nan, f"ema21_{suffix}": np.nan,
                    f"ema_cross_{suffix}": np.nan, f"rsi_{suffix}": np.nan}
        e9  = _ema(d["close"], EMA_SHORT).iloc[-1]
        e21 = _ema(d["close"], EMA_LONG).iloc[-1]
        r   = _rsi(d["close"], RSI_PERIOD).iloc[-1]
        return {f"ema9_{suffix}": float(e9), f"ema21_{suffix}": float(e21),
                f"ema_cross_{suffix}": float(e9 - e21), f"rsi_{suffix}": float(r)}

    row.update(_latest_htf_indicators(df_15m, "15m"))
    row.update(_latest_htf_indicators(df_1h,  "1h"))

    return pd.DataFrame([row])

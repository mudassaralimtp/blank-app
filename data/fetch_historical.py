"""
fetch_historical.py
───────────────────
Downloads Binance 1-min klines and builds 4-minute partial-candle snapshots
that mimic exactly what the live predictor will see at the 4-minute mark.

No future (5th-minute) data leaks into any feature row.

Usage:
    python -m data.fetch_historical          # downloads and caches to data/cache/
    python -m data.fetch_historical --days 30
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BINANCE_BASE_URL, SYMBOL, INTERVAL_1M, INTERVAL_15M, INTERVAL_1H,
    KLINE_LIMIT, HISTORICAL_DAYS, DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Binance REST helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch all klines between start_ms and end_ms in paginated batches."""
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": KLINE_LIMIT,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        # next batch starts 1 ms after the last close time
        current_start = rows[-1][6] + 1
        if len(rows) < KLINE_LIMIT:
            break
        time.sleep(0.1)   # gentle rate-limiting

    return all_rows


def _klines_to_df(raw: list) -> pd.DataFrame:
    """Convert raw Binance kline list to a clean DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = df[col].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df = df.drop(columns=["ignore"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public fetch functions
# ─────────────────────────────────────────────────────────────────────────────

def fetch_1m(days: int = HISTORICAL_DAYS) -> pd.DataFrame:
    """Return ~`days` days of 1-min BTCUSDT klines."""
    cache_file = DATA_DIR / f"btcusdt_1m_{days}d.parquet"
    if cache_file.exists():
        log.info("Loading 1-min klines from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    log.info("Fetching %d days of 1-min klines from Binance …", days)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    raw = _fetch_klines(SYMBOL, INTERVAL_1M, start_ms, now_ms)
    df = _klines_to_df(raw)
    df.to_parquet(cache_file, index=False)
    log.info("Saved %d rows → %s", len(df), cache_file)
    return df


def fetch_15m(days: int = HISTORICAL_DAYS) -> pd.DataFrame:
    """Return ~`days` days of 15-min klines (for trend features)."""
    cache_file = DATA_DIR / f"btcusdt_15m_{days}d.parquet"
    if cache_file.exists():
        log.info("Loading 15-min klines from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    log.info("Fetching %d days of 15-min klines …", days)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    raw = _fetch_klines(SYMBOL, INTERVAL_15M, start_ms, now_ms)
    df = _klines_to_df(raw)
    df.to_parquet(cache_file, index=False)
    log.info("Saved %d rows → %s", len(df), cache_file)
    return df


def fetch_1h(days: int = HISTORICAL_DAYS) -> pd.DataFrame:
    """Return ~`days` days of 1-hour klines (for trend features)."""
    cache_file = DATA_DIR / f"btcusdt_1h_{days}d.parquet"
    if cache_file.exists():
        log.info("Loading 1-hour klines from cache: %s", cache_file)
        return pd.read_parquet(cache_file)

    log.info("Fetching %d days of 1-hour klines …", days)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    raw = _fetch_klines(SYMBOL, INTERVAL_1H, start_ms, now_ms)
    df = _klines_to_df(raw)
    df.to_parquet(cache_file, index=False)
    log.info("Saved %d rows → %s", len(df), cache_file)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4-minute snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

def build_4min_snapshots(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Group 1-min candles into 5-min buckets and build a snapshot using only
    the FIRST FOUR minutes of each bucket.  The 5th-minute close is recorded
    separately as the label source (never used as a feature).

    Returns a DataFrame with one row per complete 5-min candle that has a
    full 4-minute partial window available.

    Columns returned:
        candle_open_time   – open_time of the 5-min candle
        snap_open          – price at start of the 4-min window (= 5-min open)
        snap_high          – highest high across the 4 minutes
        snap_low           – lowest low across the 4 minutes
        snap_close         – last close in the 4-min window (m4 close price)
        snap_volume        – total volume over 4 minutes
        snap_quote_volume  – total quote volume over 4 minutes
        snap_num_trades    – total trades over 4 minutes
        snap_taker_buy_vol – taker buy volume over 4 minutes
        m1..m4_*           – individual minute bars (open/high/low/close/vol)
        final_close        – ACTUAL 5-min candle close (label only, not a feature)
        label              – 1 (UP) or 0 (DOWN)
    """
    df = df_1m.copy()

    # Align each 1-min bar to its parent 5-min candle open
    df["candle_open_time"] = df["open_time"].dt.floor("5min")

    groups = []
    for candle_ts, grp in df.groupby("candle_open_time"):
        grp = grp.sort_values("open_time").reset_index(drop=True)

        # Need exactly 5 bars to have a complete candle + label
        if len(grp) < 5:
            continue

        # The first 4 bars are what the predictor sees at the 4-min mark
        partial = grp.iloc[:4]
        fifth   = grp.iloc[4]

        row = {
            "candle_open_time":   candle_ts,
            "snap_open":          float(partial.iloc[0]["open"]),
            "snap_high":          float(partial["high"].max()),
            "snap_low":           float(partial["low"].min()),
            "snap_close":         float(partial.iloc[-1]["close"]),
            "snap_volume":        float(partial["volume"].sum()),
            "snap_quote_volume":  float(partial["quote_volume"].sum()),
            "snap_num_trades":    int(partial["num_trades"].sum()),
            "snap_taker_buy_vol": float(partial["taker_buy_volume"].sum()),
        }

        # Store each individual minute bar (useful for intra-candle momentum)
        for i, (_, bar) in enumerate(partial.iterrows(), start=1):
            row[f"m{i}_open"]   = float(bar["open"])
            row[f"m{i}_high"]   = float(bar["high"])
            row[f"m{i}_low"]    = float(bar["low"])
            row[f"m{i}_close"]  = float(bar["close"])
            row[f"m{i}_volume"] = float(bar["volume"])

        # Label: does the candle close ABOVE its opening price?
        final_close = float(fifth["close"])
        candle_open = float(partial.iloc[0]["open"])
        row["final_close"] = final_close
        row["label"] = 1 if final_close > candle_open else 0

        groups.append(row)

    result = pd.DataFrame(groups).sort_values("candle_open_time").reset_index(drop=True)
    log.info("Built %d 4-min snapshots from %d 1-min bars", len(result), len(df_1m))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical BTC klines")
    parser.add_argument("--days", type=int, default=HISTORICAL_DAYS)
    args = parser.parse_args()

    df1m  = fetch_1m(args.days)
    df15m = fetch_15m(args.days)
    df1h  = fetch_1h(args.days)

    snaps = build_4min_snapshots(df1m)
    out   = DATA_DIR / f"snapshots_4min_{args.days}d.parquet"
    snaps.to_parquet(out, index=False)
    log.info("Snapshots saved → %s  (%d rows)", out, len(snaps))

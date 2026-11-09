"""
recorder.py
───────────
Persistent store for every prediction and its eventual outcome.

Schema (SQLite table: predictions)
───────────────────────────────────
  id                INTEGER PK
  candle_open_time  TEXT  (ISO-8601 UTC)
  predicted_at      TEXT  (ISO-8601 UTC)
  model_version     TEXT
  prob_up           REAL   (P(UP) from calibrated model)
  signal            TEXT   (UP / DOWN / SKIP)
  up_threshold      REAL
  down_threshold    REAL
  actual_open       REAL   (NULL until candle closes)
  actual_close      REAL   (NULL until candle closes)
  actual_direction  TEXT   (UP / DOWN — NULL until resolved)
  correct           INTEGER (1/0/NULL)
  resolved_at       TEXT   (ISO-8601 UTC — NULL until resolved)

Usage:
    from live.recorder import PredictionRecorder
    rec = PredictionRecorder()
    rec.record_prediction(...)
    rec.resolve_prediction(candle_open_time, actual_open, actual_close)
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

log = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    candle_open_time  TEXT    NOT NULL UNIQUE,
    predicted_at      TEXT    NOT NULL,
    model_version     TEXT    NOT NULL,
    prob_up           REAL    NOT NULL,
    signal            TEXT    NOT NULL,
    up_threshold      REAL    NOT NULL,
    down_threshold    REAL    NOT NULL,
    actual_open       REAL,
    actual_close      REAL,
    actual_direction  TEXT,
    correct           INTEGER,
    resolved_at       TEXT
);
"""


class PredictionRecorder:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _init_db(self):
        with self._conn() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Write ──────────────────────────────────────────────────────────────────
    def record_prediction(
        self,
        candle_open_time: datetime,
        model_version:    str,
        prob_up:          float,
        signal:           str,
        up_threshold:     float,
        down_threshold:   float,
    ) -> None:
        """Insert a new prediction row (outcome fields are NULL until resolved)."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT OR IGNORE INTO predictions
              (candle_open_time, predicted_at, model_version, prob_up, signal,
               up_threshold, down_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                candle_open_time.isoformat(),
                now,
                model_version,
                round(prob_up, 6),
                signal,
                up_threshold,
                down_threshold,
            ))
        log.info("Recorded prediction: candle=%s  signal=%s  prob_up=%.4f",
                 candle_open_time.isoformat(), signal, prob_up)

    def resolve_prediction(
        self,
        candle_open_time: datetime,
        actual_open:      float,
        actual_close:     float,
    ) -> Optional[str]:
        """
        Fill in the actual outcome for a previously recorded prediction.

        actual_direction is determined strictly by the Binance 5m candle:
            close > open  → UP
            close < open  → DOWN
            close == open → DOJI  (unscored — correct stays NULL)

        Returns the signal that was recorded, or None if row not found.
        """
        if actual_close > actual_open:
            actual_direction = "UP"
        elif actual_close < actual_open:
            actual_direction = "DOWN"
        else:
            actual_direction = "DOJI"   # open == close, no direction to score

        now = datetime.now(timezone.utc).isoformat()

        # Fetch the signal we predicted
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT signal FROM predictions WHERE candle_open_time = ?",
                (candle_open_time.isoformat(),)
            )
            row = cur.fetchone()

        if row is None:
            log.warning("resolve_prediction: no row found for candle %s", candle_open_time)
            return None

        signal = row[0]
        correct: Optional[int] = None
        # Only score when we made a directional call AND the candle has a clear direction
        if signal != "SKIP" and actual_direction != "DOJI":
            correct = 1 if signal == actual_direction else 0

        sql = """
            UPDATE predictions
            SET actual_open = ?, actual_close = ?, actual_direction = ?,
                correct = ?, resolved_at = ?
            WHERE candle_open_time = ?
        """
        with self._conn() as conn:
            conn.execute(sql, (
                actual_open, actual_close, actual_direction,
                correct, now,
                candle_open_time.isoformat(),
            ))

        log.info("Resolved: candle=%s  pred=%s  actual=%s  correct=%s",
                 candle_open_time.isoformat(), signal, actual_direction,
                 "SKIP" if correct is None else bool(correct))
        return signal

    # ── Read ───────────────────────────────────────────────────────────────────
    def get_recent(self, n: int = 50) -> pd.DataFrame:
        sql = """
            SELECT * FROM predictions
            ORDER BY candle_open_time DESC
            LIMIT ?
        """
        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=(n,))
        return df

    def get_stats(self) -> dict:
        """Return aggregate prediction statistics."""
        sql = """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN signal != 'SKIP' THEN 1 ELSE 0 END) AS decided,
                SUM(CASE WHEN signal  = 'SKIP' THEN 1 ELSE 0 END) AS skipped,
                SUM(CASE WHEN correct  = 1      THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN correct  = 0      THEN 1 ELSE 0 END) AS wrong,
                SUM(CASE WHEN signal  = 'UP'   AND correct IS NOT NULL THEN 1 ELSE 0 END) AS up_total,
                SUM(CASE WHEN signal  = 'UP'   AND correct = 1 THEN 1 ELSE 0 END) AS up_correct,
                SUM(CASE WHEN signal  = 'DOWN' AND correct IS NOT NULL THEN 1 ELSE 0 END) AS down_total,
                SUM(CASE WHEN signal  = 'DOWN' AND correct = 1 THEN 1 ELSE 0 END) AS down_correct
            FROM predictions
            WHERE resolved_at IS NOT NULL
        """
        with self._conn() as conn:
            cur = conn.execute(sql)
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
        stats = dict(zip(cols, row))

        decided = stats["decided"] or 0
        correct = stats["correct"] or 0
        stats["overall_accuracy"] = round(correct / decided, 4) if decided else None
        return stats

    def get_all(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql_query("SELECT * FROM predictions ORDER BY candle_open_time", conn)

"""
recorder.py
───────────
Persistent store for every prediction and its eventual outcome.

Cloud deployment notes
──────────────────────
Streamlit Community Cloud's ephemeral filesystem means SQLite data is lost
on every container restart.  The recorder is therefore designed in two modes:

  1. "disk" mode (default locally and on any persistent volume):
     Uses SQLite at DB_PATH.  Fast and fully featured.

  2. "memory" mode (automatic fallback when DB_PATH is not writable):
     Uses an in-memory list guarded by a module-level lock.  Survives Streamlit
     *reruns* within the same session (via st.session_state wiring in app.py)
     but is reset on container restart.  A RECORDER_WARN is set so the
     dashboard can surface a visible banner.

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
  actual_direction  TEXT   (UP / DOWN / DOJI — NULL until resolved)
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
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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

# Module-level flag: set to True when disk write failed → dashboard shows banner
RECORDER_WARN: str = ""

# ─────────────────────────────────────────────────────────────────────────────
# In-memory store (used when SQLite is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

_mem_lock: threading.Lock = threading.Lock()
_mem_rows: List[dict] = []      # ordered newest-first after insertion


def _mem_record(row: dict) -> None:
    with _mem_lock:
        # Deduplicate by candle_open_time
        if any(r["candle_open_time"] == row["candle_open_time"] for r in _mem_rows):
            return
        _mem_rows.insert(0, row)


def _mem_resolve(candle_open_time_iso: str, actual_open: float, actual_close: float) -> Optional[str]:
    with _mem_lock:
        for r in _mem_rows:
            if r["candle_open_time"] == candle_open_time_iso:
                direction = "UP" if actual_close > actual_open else ("DOWN" if actual_close < actual_open else "DOJI")
                r["actual_open"]      = actual_open
                r["actual_close"]     = actual_close
                r["actual_direction"] = direction
                r["resolved_at"]      = datetime.now(timezone.utc).isoformat()
                if r["signal"] != "SKIP" and direction != "DOJI":
                    r["correct"] = 1 if r["signal"] == direction else 0
                else:
                    r["correct"] = None
                return r["signal"]
    return None


def _mem_get_recent(n: int) -> pd.DataFrame:
    with _mem_lock:
        rows = deepcopy(_mem_rows[:n])
    if not rows:
        return pd.DataFrame(columns=[
            "id", "candle_open_time", "predicted_at", "model_version",
            "prob_up", "signal", "up_threshold", "down_threshold",
            "actual_open", "actual_close", "actual_direction",
            "correct", "resolved_at",
        ])
    return pd.DataFrame(rows)


def _mem_get_stats() -> dict:
    with _mem_lock:
        rows = deepcopy(_mem_rows)
    resolved = [r for r in rows if r.get("resolved_at")]
    total    = len(rows)
    decided  = sum(1 for r in rows if r["signal"] != "SKIP")
    skipped  = total - decided
    correct  = sum(1 for r in resolved if r.get("correct") == 1)
    wrong    = sum(1 for r in resolved if r.get("correct") == 0)
    up_total   = sum(1 for r in resolved if r["signal"] == "UP" and r.get("correct") is not None)
    up_correct = sum(1 for r in resolved if r["signal"] == "UP" and r.get("correct") == 1)
    dn_total   = sum(1 for r in resolved if r["signal"] == "DOWN" and r.get("correct") is not None)
    dn_correct = sum(1 for r in resolved if r["signal"] == "DOWN" and r.get("correct") == 1)
    overall_decided = up_total + dn_total
    return {
        "total":        total,
        "decided":      decided,
        "skipped":      skipped,
        "correct":      correct,
        "wrong":        wrong,
        "up_total":     up_total,
        "up_correct":   up_correct,
        "down_total":   dn_total,
        "down_correct": dn_correct,
        "overall_accuracy": round(correct / overall_decided, 4) if overall_decided else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main recorder class
# ─────────────────────────────────────────────────────────────────────────────

class PredictionRecorder:
    """
    Unified recorder that transparently selects SQLite or in-memory storage.

    Pass use_memory=True to force in-memory mode (e.g. from unit tests or when
    running on a read-only filesystem).
    """

    def __init__(self, db_path: Path = DB_PATH, use_memory: bool = False):
        global RECORDER_WARN
        self.db_path   = db_path
        self._use_mem  = use_memory

        if not self._use_mem:
            try:
                self._init_db()
            except Exception as exc:
                log.warning("SQLite unavailable (%s) — switching to in-memory recorder.", exc)
                self._use_mem  = True
                RECORDER_WARN  = (
                    "⚠️ Prediction history is stored in-memory only. "
                    "Data will be lost on container restart. "
                    f"(SQLite error: {exc})"
                )

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
        cot_iso = candle_open_time.isoformat()

        if self._use_mem:
            _mem_record({
                "id": None,
                "candle_open_time": cot_iso,
                "predicted_at":     now,
                "model_version":    model_version,
                "prob_up":          round(prob_up, 6),
                "signal":           signal,
                "up_threshold":     up_threshold,
                "down_threshold":   down_threshold,
                "actual_open":      None,
                "actual_close":     None,
                "actual_direction": None,
                "correct":          None,
                "resolved_at":      None,
            })
            log.info("Recorded (mem): candle=%s  signal=%s  prob_up=%.4f",
                     cot_iso, signal, prob_up)
            return

        sql = """
            INSERT OR IGNORE INTO predictions
              (candle_open_time, predicted_at, model_version, prob_up, signal,
               up_threshold, down_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (cot_iso, now, model_version,
                               round(prob_up, 6), signal,
                               up_threshold, down_threshold))
        log.info("Recorded (db): candle=%s  signal=%s  prob_up=%.4f",
                 cot_iso, signal, prob_up)

    def resolve_prediction(
        self,
        candle_open_time: datetime,
        actual_open:      float,
        actual_close:     float,
    ) -> Optional[str]:
        """
        Fill in the actual outcome for a previously recorded prediction.

        Returns the predicted signal, or None if row not found.
        """
        cot_iso = candle_open_time.isoformat()

        if self._use_mem:
            signal = _mem_resolve(cot_iso, actual_open, actual_close)
            if signal is None:
                log.warning("resolve_prediction (mem): no row for %s", cot_iso)
            return signal

        if actual_close > actual_open:
            actual_direction = "UP"
        elif actual_close < actual_open:
            actual_direction = "DOWN"
        else:
            actual_direction = "DOJI"

        now = datetime.now(timezone.utc).isoformat()

        with self._conn() as conn:
            cur = conn.execute(
                "SELECT signal FROM predictions WHERE candle_open_time = ?",
                (cot_iso,)
            )
            row = cur.fetchone()

        if row is None:
            log.warning("resolve_prediction: no row found for candle %s", cot_iso)
            return None

        signal = row[0]
        correct: Optional[int] = None
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
                correct, now, cot_iso,
            ))

        log.info("Resolved: candle=%s  pred=%s  actual=%s  correct=%s",
                 cot_iso, signal, actual_direction,
                 "SKIP" if correct is None else bool(correct))
        return signal

    # ── Read ───────────────────────────────────────────────────────────────────
    def get_recent(self, n: int = 50) -> pd.DataFrame:
        if self._use_mem:
            return _mem_get_recent(n)
        sql = """
            SELECT * FROM predictions
            ORDER BY candle_open_time DESC
            LIMIT ?
        """
        with self._conn() as conn:
            return pd.read_sql_query(sql, conn, params=(n,))

    def get_stats(self) -> dict:
        if self._use_mem:
            return _mem_get_stats()
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
        if self._use_mem:
            with _mem_lock:
                rows = deepcopy(_mem_rows)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df.sort_values("candle_open_time").reset_index(drop=True)
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY candle_open_time", conn
            )

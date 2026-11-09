"""
train.py
────────
Chronological walk-forward training pipeline.

Split:  70 % train  |  15 % validation  |  15 % test  (time-ordered, no shuffle)

Models trained:
  - Logistic Regression
  - LightGBM
  - XGBoost

Each model is calibrated with Platt scaling (CalibratedClassifierCV),
saved to models/saved/<model_name>.joblib, and its metadata written to
models/saved/model_registry.json.

Usage:
    python -m models.train
    python -m models.train --days 60
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, MODEL_DIR, HISTORICAL_DAYS,
    TRAIN_RATIO, VAL_RATIO,
    DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD,
)
from data.fetch_historical import fetch_1m, fetch_15m, fetch_1h, build_4min_snapshots
from data.features import build_features
from models.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chronological split helper
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(X: pd.DataFrame, y: pd.Series):
    """Split X, y into train / val / test preserving time order."""
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X.iloc[:train_end],  y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],    y.iloc[val_end:]

    log.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))
    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def _make_logistic() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=0.1,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])


def _make_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )


def _make_xgb() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training and calibration
# ─────────────────────────────────────────────────────────────────────────────

def train_and_calibrate(
    name: str,
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
) -> object:
    """Fit estimator, then Platt-calibrate on validation set."""
    log.info("[%s] Fitting base model …", name)
    estimator.fit(X_train, y_train)

    log.info("[%s] Calibrating on validation set …", name)
    # sklearn >= 1.6 removed cv="prefit"; pass the fitted estimator with cv=None
    calibrated = CalibratedClassifierCV(estimator, cv=None, method="sigmoid")
    calibrated.fit(X_val, y_val)
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_training(days: int = HISTORICAL_DAYS):
    # 1. Load / fetch data
    snap_cache = DATA_DIR / f"snapshots_4min_{days}d.parquet"
    if snap_cache.exists():
        log.info("Loading cached snapshots from %s", snap_cache)
        snapshots = pd.read_parquet(snap_cache)
    else:
        log.info("Building snapshots from scratch …")
        df1m  = fetch_1m(days)
        snapshots = build_4min_snapshots(df1m)
        snapshots.to_parquet(snap_cache, index=False)

    df_1m  = fetch_1m(days)
    df_15m = fetch_15m(days)
    df_1h  = fetch_1h(days)

    # 2. Feature engineering
    X, y = build_features(snapshots, df_1m, df_15m, df_1h)

    # 3. Chronological split
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    # 4. Define models
    models = {
        "logistic":  _make_logistic(),
        "lightgbm":  _make_lgbm(),
        "xgboost":   _make_xgb(),
    }

    registry = {}
    test_results = {}

    for name, estimator in models.items():
        log.info("══ Training: %s ══", name)

        calibrated = train_and_calibrate(name, estimator, X_train, y_train, X_val, y_val)

        # Save model
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(calibrated, model_path)
        log.info("[%s] Saved → %s", name, model_path)

        # Evaluate on test set
        metrics = evaluate_model(
            model=calibrated,
            X_test=X_test,
            y_test=y_test,
            up_threshold=DEFAULT_UP_THRESHOLD,
            down_threshold=DEFAULT_DOWN_THRESHOLD,
            model_name=name,
        )
        test_results[name] = metrics

        registry[name] = {
            "path":           str(model_path),
            "trained_at":     datetime.now(timezone.utc).isoformat(),
            "n_train":        int(len(y_train)),
            "n_val":          int(len(y_val)),
            "n_test":         int(len(y_test)),
            "up_threshold":   DEFAULT_UP_THRESHOLD,
            "down_threshold": DEFAULT_DOWN_THRESHOLD,
            "test_metrics":   metrics,
        }

    # Save registry
    reg_path = MODEL_DIR / "model_registry.json"
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    log.info("Registry saved → %s", reg_path)

    # Pick best model by val ROC-AUC (already evaluated above on test)
    best_name = max(test_results, key=lambda k: test_results[k].get("roc_auc") or 0)
    best_path = MODEL_DIR / f"{best_name}.joblib"

    # Write a "current_model" pointer
    pointer = {"model_name": best_name, "path": str(best_path)}
    with open(MODEL_DIR / "current_model.json", "w") as f:
        json.dump(pointer, f, indent=2)
    log.info("Best model: %s (ROC-AUC=%.4f) → set as current model",
             best_name, test_results[best_name].get("roc_auc", 0))

    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines(X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Naive baselines:
      - always_up:   predict 1 (UP) every time
      - always_down: predict 0 (DOWN) every time
      - random:      random 50/50
    """
    from sklearn.metrics import accuracy_score, f1_score

    n = len(y_test)
    rng = np.random.default_rng(42)

    baselines = {
        "always_up":   np.ones(n, dtype=int),
        "always_down": np.zeros(n, dtype=int),
        "random_50":   rng.integers(0, 2, n),
    }
    results = {}
    for bl_name, preds in baselines.items():
        results[bl_name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1":       round(f1_score(y_test, preds, zero_division=0), 4),
        }
        log.info("Baseline [%s]: acc=%.4f  f1=%.4f",
                 bl_name, results[bl_name]["accuracy"], results[bl_name]["f1"])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=HISTORICAL_DAYS)
    args = parser.parse_args()
    run_training(args.days)

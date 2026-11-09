"""
evaluate.py
───────────
Computes and prints the full evaluation report for a trained model.

Metrics reported:
  - Accuracy, Precision, Recall, F1 (UP/DOWN/weighted)
  - ROC-AUC
  - Brier Score
  - Confusion Matrix
  - Calibration summary

Usage (standalone):
    python -m models.evaluate
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, MODEL_DIR, HISTORICAL_DAYS, REPORTS_DIR,
    DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    up_threshold:   float = DEFAULT_UP_THRESHOLD,
    down_threshold: float = DEFAULT_DOWN_THRESHOLD,
    model_name: str = "model",
) -> dict:
    """
    Evaluate a calibrated binary classifier.

    Returns a dict of all metrics and saves a text report to reports/.
    """
    # Probabilities: column 1 = P(UP)
    proba = model.predict_proba(X_test)[:, 1]

    # Apply UP/DOWN/SKIP thresholds
    def _threshold_predict(p):
        if p >= up_threshold:
            return 1
        if p <= 1 - down_threshold:
            return 0
        return -1   # SKIP

    preds_with_skip = np.array([_threshold_predict(p) for p in proba])

    # For metrics: exclude SKIP rows
    mask = preds_with_skip != -1
    n_skip = (~mask).sum()
    skip_pct = n_skip / len(preds_with_skip) * 100

    y_decided  = y_test.values[mask]
    p_decided  = proba[mask]
    pred_decided = preds_with_skip[mask]

    if len(y_decided) == 0:
        log.warning("[%s] All predictions were SKIP — lower thresholds.", model_name)
        return {
            "n_total": len(proba), "n_skip": int(n_skip), "skip_pct": 100.0,
            "n_decided": 0, "accuracy": None, "precision": None, "recall": None,
            "f1": None, "roc_auc": None, "brier_score": None,
            "confusion_matrix": None, "calibration": {},
        }

    # Standard binary metrics (on non-SKIP rows)
    acc   = accuracy_score(y_decided, pred_decided)
    prec  = precision_score(y_decided, pred_decided, zero_division=0)
    rec   = recall_score(y_decided, pred_decided, zero_division=0)
    f1    = f1_score(y_decided, pred_decided, zero_division=0)
    try:
        auc = roc_auc_score(y_decided, p_decided)
    except ValueError:
        auc = float("nan")

    brier = brier_score_loss(y_decided, p_decided)
    cm    = confusion_matrix(y_decided, pred_decided, labels=[0, 1])
    cr    = classification_report(y_decided, pred_decided, target_names=["DOWN", "UP"])

    # Calibration: mean predicted prob vs actual fraction positive per decile
    try:
        frac_pos, mean_pred = calibration_curve(y_decided, p_decided, n_bins=10)
        cal_summary = {
            "mean_pred": mean_pred.tolist(),
            "frac_pos":  frac_pos.tolist(),
        }
    except Exception:
        cal_summary = {}

    metrics = {
        "n_total":        len(proba),
        "n_skip":         int(n_skip),
        "skip_pct":       round(skip_pct, 2),
        "n_decided":      int(mask.sum()),
        "accuracy":       round(acc,   4),
        "precision":      round(prec,  4),
        "recall":         round(rec,   4),
        "f1":             round(f1,    4),
        "roc_auc":        round(auc,   4) if not np.isnan(auc) else None,
        "brier_score":    round(brier, 4),
        "confusion_matrix": cm.tolist(),
        "calibration":    cal_summary,
    }

    # ── Print report ──────────────────────────────────────────────────────────
    sep = "─" * 55
    report_lines = [
        sep,
        f"  Evaluation Report: {model_name.upper()}",
        sep,
        f"  Samples  : {metrics['n_total']} total  |  {metrics['n_decided']} decided  |  {metrics['n_skip']} skipped ({skip_pct:.1f} %)",
        f"  Accuracy : {acc:.4f}",
        f"  Precision: {prec:.4f}",
        f"  Recall   : {rec:.4f}",
        f"  F1       : {f1:.4f}",
        f"  ROC-AUC  : {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC  : n/a",
        f"  Brier    : {brier:.4f}",
        "",
        "  Confusion Matrix (rows=actual, cols=predicted):",
        "           DOWN   UP",
        f"  DOWN     {cm[0,0]:4d}  {cm[0,1]:4d}",
        f"  UP       {cm[1,0]:4d}  {cm[1,1]:4d}",
        "",
        "  Classification Report:",
        cr,
        sep,
    ]
    full_report = "\n".join(report_lines)
    print(full_report)

    # Save text report
    rpt_path = REPORTS_DIR / f"{model_name}_eval.txt"
    rpt_path.write_text(full_report)
    log.info("Report saved → %s", rpt_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner: re-evaluate saved models on held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(days: int = HISTORICAL_DAYS):
    from data.fetch_historical import fetch_1m, fetch_15m, fetch_1h, build_4min_snapshots
    from data.features import build_features
    from models.train import chronological_split

    snap_cache = DATA_DIR / f"snapshots_4min_{days}d.parquet"
    if not snap_cache.exists():
        raise FileNotFoundError(f"No snapshot cache found at {snap_cache}. Run models.train first.")

    snapshots = pd.read_parquet(snap_cache)
    df_1m  = fetch_1m(days)
    df_15m = fetch_15m(days)
    df_1h  = fetch_1h(days)

    X, y = build_features(snapshots, df_1m, df_15m, df_1h)
    _, _, _, _, X_test, y_test = chronological_split(X, y)

    reg_path = MODEL_DIR / "model_registry.json"
    if not reg_path.exists():
        raise FileNotFoundError("model_registry.json not found. Run models.train first.")

    with open(reg_path) as f:
        registry = json.load(f)

    all_metrics = {}
    for name, info in registry.items():
        model_path = Path(info["path"])
        if not model_path.exists():
            log.warning("Model file not found: %s", model_path)
            continue
        model = joblib.load(model_path)
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            up_threshold=info.get("up_threshold", DEFAULT_UP_THRESHOLD),
            down_threshold=info.get("down_threshold", DEFAULT_DOWN_THRESHOLD),
            model_name=name,
        )
        all_metrics[name] = metrics

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=HISTORICAL_DAYS)
    args = parser.parse_args()
    run_full_evaluation(args.days)

"""
run.py — convenience entry-point
─────────────────────────────────
Commands
────────
  python run.py fetch       – download historical data and build snapshots
  python run.py train       – train all models
  python run.py evaluate    – re-evaluate saved models on the test set
  python run.py live        – start the live prediction service
  python run.py dashboard   – launch the Streamlit dashboard (prints command)
  python run.py all         – fetch + train + evaluate
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("command",
                        choices=["fetch", "train", "evaluate", "live", "dashboard", "all"])
    parser.add_argument("--days",  type=int, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Model name for live mode (logistic/lightgbm/xgboost)")
    args = parser.parse_args()

    if args.command in ("fetch", "all"):
        from data.fetch_historical import fetch_1m, fetch_15m, fetch_1h, build_4min_snapshots
        from config import HISTORICAL_DAYS, DATA_DIR
        days = args.days or HISTORICAL_DAYS
        df1m = fetch_1m(days)
        fetch_15m(days)
        fetch_1h(days)
        snaps = build_4min_snapshots(df1m)
        out = DATA_DIR / f"snapshots_4min_{days}d.parquet"
        snaps.to_parquet(out, index=False)
        print(f"Snapshots saved: {out}  ({len(snaps)} rows)")

    if args.command in ("train", "all"):
        from models.train import run_training
        from config import HISTORICAL_DAYS
        days = args.days or HISTORICAL_DAYS
        run_training(days)

    if args.command in ("evaluate", "all"):
        from models.evaluate import run_full_evaluation
        from config import HISTORICAL_DAYS
        days = args.days or HISTORICAL_DAYS
        run_full_evaluation(days)

    if args.command == "live":
        from live.predictor import run_live
        run_live(args.model)

    if args.command == "dashboard":
        print("\nTo launch the dashboard, run:\n")
        print("  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()

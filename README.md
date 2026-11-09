# BTC Candle AI — 5-Minute Candle Direction Predictor

Predicts whether the current BTC/USDT 5-minute candle will close **UP** or **DOWN** relative to
its opening price. The prediction is made at the **4-minute mark** — 60 seconds before the candle
closes.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download historical data and build 4-minute snapshots
python run.py fetch

# 3. Train all three models (Logistic Regression, LightGBM, XGBoost)
python run.py train

# 4. (Optional) Re-evaluate on the held-out test set
python run.py evaluate

# 5. Start the live predictor
python run.py live

# 6. Open the dashboard (in a second terminal)
streamlit run dashboard/app.py
```

Or run everything at once:

```bash
python run.py all
```

---

## Architecture

```
btc-candle-ai/
├── config.py                  Central config
├── requirements.txt
├── run.py                     CLI entry-point
├── data/
│   ├── fetch_historical.py    Binance kline downloader + 4-min snapshot builder
│   └── features.py            Feature engineering (26 features)
├── models/
│   ├── train.py               Walk-forward train / validate / test pipeline
│   ├── evaluate.py            Full metrics report
│   └── saved/                 Trained model files + registry
├── live/
│   ├── predictor.py           Live prediction loop
│   └── recorder.py            SQLite prediction log
├── dashboard/
│   └── app.py                 Streamlit dashboard
├── data/cache/                Cached Binance kline parquet files
├── reports/                   Evaluation text reports
└── predictions.db             All predictions + outcomes
```

---

## Prediction rules

| Signal | Condition |
|--------|-----------|
| **UP** | P(UP) ≥ `up_threshold` (default 0.55) |
| **DOWN** | P(UP) ≤ `1 − down_threshold` (default 0.45) |
| **SKIP** | Model confidence insufficient |

---

## Features (26 total)

**Intra-candle** (from the 4 partial 1-min bars):
- Net volume, body ratio, upper/lower wick ratios
- Intra-candle momentum, minute-by-minute returns (m1→m2, m2→m3, m3→m4)
- Volume acceleration (late vs early half)

**Rolling 1-min**:
- OBV, RSI(14), EMA(9), EMA(21), EMA cross
- VWAP deviation, volume spike z-score, ATR(14)

**Higher-timeframe** (15m and 1h):
- EMA(9), EMA(21), EMA cross, RSI(14) — joined via asof merge (no look-ahead)

---

## No look-ahead bias

- Only **closed** 1-min bars are used (close_time < prediction timestamp).
- Higher-timeframe bars are joined with `direction="backward"` — only bars fully closed before the prediction timestamp are used.
- The 5th-minute bar is fetched **after** the candle closes, solely for labelling.
- SKIP is returned if fewer than 4 closed bars are available or any feature is NaN.

---

## Configuration

Edit `config.py` to change:
- `HISTORICAL_DAYS` — training window
- `DEFAULT_UP_THRESHOLD` / `DEFAULT_DOWN_THRESHOLD` — prediction thresholds
- `EMA_SHORT`, `EMA_LONG`, `RSI_PERIOD`, etc. — indicator parameters

---

## Data source

Binance Spot public REST API — no API key required.

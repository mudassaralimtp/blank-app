"""
Central configuration for the BTC 5-min candle prediction system.
No API keys are required for public market data endpoints.
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "cache"
MODEL_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = BASE_DIR / "predictions.db"

for _d in [DATA_DIR, MODEL_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Binance public REST ───────────────────────────────────────────────────────
BINANCE_BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL_1M = "1m"
INTERVAL_5M = "5m"
INTERVAL_15M = "15m"
INTERVAL_1H = "1h"

# ── Data collection ───────────────────────────────────────────────────────────
# How many 1-min candles to fetch per request (Binance max = 1000)
KLINE_LIMIT = 1000

# Training window: last N days of 1-min candles
HISTORICAL_DAYS = 90          # ~90 days of 1-min data ≈ 129 600 rows
FEATURE_WARMUP_ROWS = 50      # rows to drop before the rolling indicators stabilise

# ── Feature engineering ───────────────────────────────────────────────────────
EMA_SHORT = 9
EMA_LONG = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
VWAP_WINDOW = 20              # rolling VWAP window (in 1-min bars)
VOLUME_SPIKE_WINDOW = 20      # rolling window for volume-spike z-score

# ── Model training ────────────────────────────────────────────────────────────
# Walk-forward: 70 % train, 15 % val, 15 % test (chronological, no shuffle)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO  (implicitly 0.15)

# ── Prediction thresholds ────────────────────────────────────────────────────
# Tuned on validation set; default values used if tuning hasn't run yet.
DEFAULT_UP_THRESHOLD   = 0.55   # P(UP)   > this  → predict UP
DEFAULT_DOWN_THRESHOLD = 0.55   # P(DOWN) > this  → predict DOWN
# Otherwise → SKIP

# ── Live prediction service ───────────────────────────────────────────────────
# How many seconds into a 5-min candle to fire the prediction (240 = 4 min)
PREDICTION_OFFSET_SECONDS = 240
CANDLE_DURATION_SECONDS   = 300   # 5 min

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SECONDS = 3
RECENT_PREDICTIONS_SHOWN  = 50

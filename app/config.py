"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

UNUSUAL_WHALES_API_KEY = os.environ["UNUSUAL_WHALES_API_KEY"]

OHLCV_LOOKBACK_DAYS = 365  # ~252 trading days

EMA_SHORT = 20
EMA_LONG = 50
ATR_PERIOD = 14

WATCHLIST = ["AAPL", "NVDA", "MSFT", "AMZN", "META", "TSLA"]

WATCHLIST_TTL_DAYS = 5

ATR_TRAIL_MULT = 3.0
HYBRID_TRAIL_MULT = 2.0
PARTIAL_EXIT_PCT = 0.5
MAX_HOLD_DAYS = 5

MIN_FINAL_SCORE = 7.0
PORTFOLIO_CAPITAL = 10_000.0
MAX_POSITIONS = 10
MAX_PORTFOLIO_HEAT = 0.15
ROTATION_SCORE_MARGIN = 1.0
SIZING_TIERS = [
    (9.0, 0.030),   # score >= 9.0 → risk 3.0% of capital
    (8.0, 0.020),   # score >= 8.0 → risk 2.0%
    (7.0, 0.015),   # score >= 7.0 → risk 1.5%
]

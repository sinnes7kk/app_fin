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

ATR_TRAIL_MULT = 2.5
HYBRID_TRAIL_MULT = 2.0
POST_T1_TRAIL_ATR = 1.0
PARTIAL_EXIT_PCT = 0.5
# Conviction-scaled partial sizing: (min_score, partial_pct)
# Higher conviction → smaller partial exit at T1 to let runners develop
PARTIAL_TIERS = [
    (9.0, 0.33),  # high conviction: only take 1/3 at T1
    (8.0, 0.40),
    (7.0, 0.50),  # default: half at T1
]
MAX_HOLD_DAYS = 5

GAMMA_NEGATIVE_TARGET_MULT = 1.15
GAMMA_POSITIVE_TARGET_MULT = 0.85
GAMMA_NEGATIVE_TRAIL_MULT = 1.15
GAMMA_POSITIVE_TRAIL_MULT = 0.85
WALL_PROXIMITY_REJECT_PCT = 0.02
WALL_PROXIMITY_REJECT_ATR = 1.0
WALL_PROXIMITY_WARNING_PCT = 0.05
WALL_PROXIMITY_WARNING_ATR = 2.0

BID_SIDE_FLOW_CONFIDENCE = 0.5

# Market regime (VIX thresholds and sizing adjustments)
VIX_ELEVATED = 20.0
VIX_HIGH = 30.0
VIX_SIZING_ELEVATED = 0.75
VIX_SIZING_HIGH = 0.50

# Regime-based directional threshold boost (max added to MIN_FINAL_SCORE for
# counter-regime trades).  At regime_score=0.5 both sides get half the boost.
REGIME_THRESHOLD_BOOST = 1.0

# Counter-trend trades need higher conviction to pass
COUNTER_TREND_PREMIUM = 1.0

# Relative strength
RS_LOOKBACK_DAYS = 20
RS_LONG_MIN = -0.02   # longs must not be lagging SPY by more than 2%
RS_SHORT_MAX = 0.05   # shorts: demote to watchlist if leading SPY by more than 5%

# Multi-day flow persistence
FLOW_PERSISTENCE_DAYS = 3    # look back this many calendar days
FLOW_PERSISTENCE_BONUS = 0.5 # max score bonus for persistent flow (conservative)

MIN_FINAL_SCORE = 7.0
PORTFOLIO_CAPITAL = 10_000.0
MAX_POSITIONS = 10
MAX_PORTFOLIO_HEAT = 0.15
ROTATION_SCORE_MARGIN = 1.0
MAX_SECTOR_PER_DIRECTION = 2

# Drawdown circuit breaker
DRAWDOWN_THROTTLE_PCT = 0.05   # reduce sizing by 50% when drawdown exceeds this
DRAWDOWN_HALT_PCT = 0.10       # halt all new entries when drawdown exceeds this
DRAWDOWN_SIZING_MULT = 0.50    # sizing multiplier when throttled
SIZING_TIERS = [
    (9.0, 0.030),   # score >= 9.0 → risk 3.0% of capital
    (8.0, 0.020),   # score >= 8.0 → risk 2.0%
    (7.0, 0.015),   # score >= 7.0 → risk 1.5%
]

# Multi-day flow tracker
FLOW_TRACKER_LOOKBACK_DAYS = 5
FLOW_TRACKER_MIN_ACTIVE_DAYS = 2
FLOW_TRACKER_MIN_PREMIUM = 250_000
FLOW_TRACKER_MIN_MCAP = 500_000_000
FLOW_TRACKER_MIN_PREM_MCAP_BPS = 0.10
FLOW_TRACKER_MAX_RESULTS = 30

FLOW_TRACKER_ETF_EXCLUDE = {
    "SPY", "QQQ", "IWM", "DIA", "RSP",
    "GDX", "GDXJ", "GLD", "SLV", "AGQ",
    "XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLC", "XLB",
    "XBI", "XOP", "XHB", "XME",
    "KRE", "KBE",
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "SPXS", "TNA", "TZA",
    "UCO", "SCO",
    "TLT", "TBT", "TMF", "TMV", "SHY", "IEF", "BND",
    "HYG", "JNK", "LQD",
    "EEM", "EWJ", "EWZ", "FXI", "KWEB", "INDA", "EWW", "EPI",
    "ARKK", "ARKW", "ARKF", "ARKG",
    "SMH", "SOXX", "VGT", "IGV",
    "VTI", "VOO", "VEA", "VWO", "VNQ", "VNQI",
    "USO", "UNG", "WEAT", "CORN",
}

# Multi-day dark pool tracker
DP_TRACKER_LOOKBACK_DAYS = 5
DP_TRACKER_MIN_ACTIVE_DAYS = 2

# Earnings proximity
EARNINGS_HOLD_PENALTY = 0.5   # score penalty when ER falls within hold window
EARNINGS_WARN_DAYS = 10       # badge threshold: "Earnings in X days"

# Insider transactions
INSIDER_BUY_BONUS = 0.3       # score bonus when insider buys align with direction

# Sentiment APIs (optional — gracefully skipped if missing)
STOCKTWITS_API_KEY = os.environ.get("STOCKTWITS_API_KEY", "")
REDDIT_USER_AGENT = "flow_tracker/1.0 by app_fin"
REDDIT_SUBREDDITS = ["wallstreetbets", "options", "stocks", "investing"]

# Position health and rotation
HEALTH_STRONG_THRESHOLD = 7
HEALTH_WEAK_THRESHOLD = 4
HEALTH_FAILING_THRESHOLD = 2
ROTATION_HEALTH_MARGIN = 2.0   # new_score - health must exceed this for NEUTRAL rotation
ROTATION_COOLDOWN_DAYS = 2     # anti-whipsaw: min days between rotations in same slot

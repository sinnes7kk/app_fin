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

# Rolling z-score baselines for flow components (see app/features/flow_stats.py).
# When USE_ZSCORE_FLOW is True, each directional flow component is scored as
# (value - median_30d) / MAD_30d instead of against the absolute thresholds
# in flow_features._FLOW_THRESHOLDS. Four-tier fallback ladder handles tickers
# with insufficient history.
USE_ZSCORE_FLOW = False           # master switch; False keeps current absolute-threshold scoring
ZSCORE_LOOKBACK_DAYS = 30         # trailing window for per-ticker stats
ZSCORE_MIN_N_FULL = 20            # Tier 1 minimum valid observations
ZSCORE_MIN_N_SHRUNK = 5           # Tier 2 minimum (below → Tier 3 cross-sectional)
ZSCORE_SHRINKAGE_K = 10           # Bayesian shrinkage strength for Tier 2 MAD
ZSCORE_CLIP = 3.5                 # clamp |z| to this before logistic
ZSCORE_MIN_COHORT_SIZE = 10       # Tier 3/cross-section needs at least this many tickers

# Delta-weighted directional premium (see app/features/flow_features.py::add_delta_weights).
# When USE_DELTA_WEIGHTED_FLOW is True, the `flow_intensity` component in the
# scorer reads from bullish/bearish_delta_intensity (premium × |delta| / mcap)
# instead of raw bullish/bearish premium over mcap.  The flag ships OFF so we
# can shadow-log the new intensity distribution for ~1 week, recalibrate the
# ceiling in _FLOW_THRESHOLDS, then flip on.
USE_DELTA_WEIGHTED_FLOW = False        # master switch — shadow mode OFF
DELTA_PROXY_VOL = 0.35                 # annualized vol used by the BS moneyness fallback
DELTA_FETCH_TIMEOUT = 5                # per-call timeout (seconds) for UW greek-exposure
DELTA_MAX_UNIQUE_PER_SCAN = 250        # above this, skip UW enrichment and fall back to proxy
DELTA_MIN_PROXY_COVERAGE = 0.3         # warn if >30% of premium is backed by proxy deltas

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

# Wave 0.5 C1 — snapshot retention + horizon toggle.
#
# Retention is how long we keep historical rows in screener_snapshots.csv.
# Old logic was `LOOKBACK_DAYS + 3` (8 days) which broke the 15d horizon and
# starved the B3 relative-PCR check of per-ticker history.  21 days is
# enough for the 15d horizon + a 6-day buffer for weekends/holidays, and
# still short enough that the CSV stays small.
FLOW_TRACKER_RETENTION_DAYS = 21

# Horizons offered to the UI as a [5d][15d] toggle.  5d is the default
# (matches FLOW_TRACKER_LOOKBACK_DAYS); 15d confirms persistence over a
# longer window.  Horizon config is `{key: {label, lookback_days,
# min_active_days}}`.
FLOW_TRACKER_HORIZONS = {
    "5d": {
        "label": "5d",
        "lookback_days": 5,
        "min_active_days": 2,
    },
    "15d": {
        "label": "15d",
        "lookback_days": 15,
        "min_active_days": 4,
    },
}
FLOW_TRACKER_HORIZON_DEFAULT = "5d"

# Flow Tracker mode-aware gating (Wave 0 — accumulation-first filtering).
#
# `all` is the legacy loose gate (today's behaviour).  `accumulation` is the
# new default: most days active, strongly one-sided, non-fading, material
# vs market cap.  `strong_accumulation` is the purest swing pattern: every
# day active, still rising, A-tier grade.
#
# The composite conviction score weights shift when mode != "all" so that
# one-sidedness + acceleration carry more weight (the actual accumulation
# signature) at the expense of raw intensity.  Ladder grades stay on the
# same 0-10 scale so `grade_stats.json` remains compatible.
FLOW_TRACKER_MODE_DEFAULT = "accumulation"          # UI default on first load
FLOW_TRACKER_AUTO_WIDEN_MIN = 5                     # auto-widen notice threshold

FLOW_TRACKER_MODES = {
    "all": {
        "label": "All",
        "min_active_days": 2,
        "min_cum_premium": 250_000,
        "min_prem_mcap_bps": 0.10,
        "min_consistency": 0.0,
        "min_accel_t": -99.0,
        "exclude_hedging": False,
        "min_grade_rank": 0,                        # C and above
        "intro": "All tickers clearing the base multi-day persistence gate.",
    },
    "accumulation": {
        "label": "Accumulation",
        "min_active_days": 4,
        "min_cum_premium": 1_000_000,
        "min_prem_mcap_bps": 2.0,
        "min_consistency": 0.55,
        "min_accel_t": -0.5,
        "exclude_hedging": True,
        "min_grade_rank": 3,                        # B+ and above
        "intro": "Names with consistent multi-day one-sided options flow — the classic accumulation pattern.",
    },
    "strong_accumulation": {
        "label": "Strong",
        "min_active_days": 5,
        "min_cum_premium": 2_000_000,
        "min_prem_mcap_bps": 3.0,
        "min_consistency": 0.65,
        "min_accel_t": 0.5,
        "exclude_hedging": True,
        "min_grade_rank": 4,                        # A- and above
        "intro": "Names with unusual options activity on every day of the last 5, strongly one-sided, and still rising.",
    },
}

# Use accumulation-mode weights for conviction scoring on every row (even
# when the user picks "All" mode) so grades stay comparable across modes
# and the accumulation signature always drives ranking.  The ladder (A+/A/
# A-/B+/B/B-/C) is unchanged; this is a re-weighting within 0-10.
#
# Wave 0.5 A7: added `oi_change` (0.05) reclaimed from `mass` (0.10→0.05).
# Open-interest change is a direct readout of whether positions were opened
# vs closed and adds a structural-conviction signal that mass alone can't
# capture.  Weights still sum to 1.0.
FLOW_TRACKER_WEIGHTS_ACCUM = {
    "persistence": 0.25,
    "intensity":   0.20,
    "consistency": 0.25,
    "accel":       0.20,
    "mass":        0.05,
    "oi_change":   0.05,
}
FLOW_TRACKER_WEIGHTS_LEGACY = {
    "persistence": 0.30,
    "intensity":   0.30,
    "consistency": 0.20,
    "accel":       0.10,
    "mass":        0.10,
}

# Wave 0.5 A1: dominant DTE bucket.  Labels are ordered by "near-term weak
# structure" to "long-term commitment".  `0-7d` is the softest: short-term
# lottery flow that resolves quickly and often in market-maker hedging
# patterns — we apply a small haircut when the bucket dominates on
# accumulation-mode rows.  `91+d` = LEAPs, softly boosted as structural.
FLOW_TRACKER_DTE_BUCKETS = [
    ("0-7",   0,   7,   0.90),   # bucket label, dte_min, dte_max, score_multiplier
    ("8-30",  8,   30,  1.00),
    ("31-90", 31,  90,  1.05),
    ("91+",   91,  9999, 1.05),
]

# Wave 0.5 A6: require perc_3_day_total above this percentile for the
# accumulation/strong-accumulation gates.  `perc_3_day_total` is UW's
# normalized unusualness score (0..1) across the universe — values above
# 0.70 mean "top 30% of unusual 3-day activity".
FLOW_TRACKER_MIN_3D_PERCENTILE = 0.70

# Wave 0.5 A3: when True, flow alert fetches include UW's
# `size_greater_oi=true` flag so we only count prints that likely opened a
# position (trade size > current OI).  Default OFF so behaviour is unchanged
# until you explicitly cut over.
FLOW_OPENING_ONLY = False

# Wave 0.5 A4: window return pill thresholds.  Return is computed over the
# tracker lookback window using snapshot closes.
FLOW_TRACKER_RETURN_BONUS = 0.25   # score bonus when return aligns with direction
FLOW_TRACKER_RETURN_DRAG = 0.25    # score drag when return is fighting direction

# Wave 0.5 A8: dark-pool alignment bonus.  When DP flow direction agrees
# with options flow direction AND DP notional is material (>3 bps of
# market cap), we multiply the conviction score by (1 + bonus).  Max 0.30
# (30% uplift) scaled by strength of alignment.
DP_ALIGNMENT_MAX_BONUS = 0.30
DP_ALIGNMENT_MIN_NOTIONAL_BPS = 3.0

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

# ──────────────────────────────────────────────────────────────────────────
# Wave 8 — Risk Regime (market-aware sizing adjustments).
#
# Aggregates SPY-trend / VIX / VIX-term-structure / portfolio-heat / economic
# calendar into a single multiplier that downstream sizing (trader-card
# notional + structure-tab caveats) consumes.
#
# Tiers are ``calm`` / ``elevated`` / ``panic`` / ``halt`` — the multiplier
# falls off geometrically so an ``elevated`` regime doubles the thinking
# time before sizing up, while ``panic`` effectively stops new fresh
# positions.
# ──────────────────────────────────────────────────────────────────────────
# VIX tiers driving the regime score.  Lower bound inclusive, upper exclusive.
VIX_TIERS = [
    ("calm",     0.0,  18.0, 1.00),
    ("elevated", 18.0, 24.0, 0.75),
    ("panic",    24.0, 35.0, 0.50),
    ("halt",     35.0, 999.0, 0.25),
]

# SPY RSI bounds that flag overextension / oversold regimes.  Anything in
# between is considered "in range".  Crossings trigger regime caveats.
SPY_RSI_OVERBOUGHT = 70.0
SPY_RSI_OVERSOLD = 30.0

# Heat-based sizing checks.  Portfolio-wide risk % at which we start
# halving (clamp) or halting (freeze) new risk.
HEAT_CLAMP_PCT = 3.0
HEAT_FREEZE_PCT = 5.0

# Sector / direction concentration caps.  Checked against open positions +
# proposed entry.  ``None`` disables the check.
MAX_SAME_SECTOR = 3
MAX_SAME_DIRECTION = 6

# Event windows that flip us to halt regardless of other checks.  Values
# in days (0 = today).
HALT_ON_FOMC_WINDOW_DAYS = 1
HALT_ON_CPI_WINDOW_DAYS = 1
HALT_ON_NFP_WINDOW_DAYS = 1

# VIX3M / VIX term-structure flag.  When VIX3M / VIX < threshold we're in
# backwardation → risk-off.  ``None`` disables the check.
VIX_BACKWARDATION_THRESHOLD = 1.0  # ratio below this = backwardation

# Path to the hand-curated economic calendar (FOMC / CPI / NFP dates).
# Kept out of source as a JSON blob to make it easy to update.
ECONOMIC_CALENDAR_PATH = Path(__file__).resolve().parent / "data" / "economic_calendar.json"

# Path used by the Wave 8 market-data cache.
MARKET_INDICATORS_CACHE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "market_indicators.json"
)
MARKET_INDICATORS_CACHE_TTL_HOURS = 12  # cache validity

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

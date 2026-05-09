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

# ──────────────────────────────────────────────────────────────────────────
# Per-DTE-bucket hold and trail config.
#
# `MAX_HOLD_DAYS` (legacy, 5d) was a single global cap that was too tight
# for `position` (31-90d DTE) trades and too loose for `lottery` (0-7d DTE)
# trades. The replay backtest (`scripts/build_replay_backtest.py` →
# `data/diagnostic_replay_*.md`) showed:
#   - position bucket: median time-to-MFE ~1d, p75 ~3d, but 13.8% of plans
#     hit T2 with 53.6% P(+2R | +1R) — capping at 5d clipped the runners.
#   - swing/leap: too few rows to trust the recommendation (LOW confidence).
#   - unknown: the catch-all from rows without DTE enrichment.
#
# These per-bucket constants replace the single global. The legacy aliases
# `MAX_HOLD_DAYS` and `ATR_TRAIL_MULT` stay defined (resolved against
# `position` / `unknown` to preserve back-compat) for one release while
# all call sites migrate to `resolve_hold_config()` /
# `resolve_trail_config()` from `app/signals/hold_config.py`.
#
# `EARNINGS_RISK_WINDOW_DAYS` is decoupled from MAX_HOLD: even if a leap
# trade has a 25-day hold, we still want to skip / warn on earnings within
# the next 5 trading days because earnings volatility is the dominant
# overnight-gap risk regardless of intended hold.
# ──────────────────────────────────────────────────────────────────────────
MAX_HOLD_DAYS_BY_BUCKET = {
    "lottery":  3,    # 0-7 DTE — fast resolution, fast exit
    "swing":    7,    # 8-30 DTE — LOW confidence; conservative
    "position": 10,   # 31-90 DTE — HIGH confidence from replay
    "leap":     15,   # 91+ DTE — LOW confidence; conservative
    "unknown":  5,    # fallback when DTE enrichment unavailable
}

ATR_TRAIL_MULT_BY_BUCKET = {
    "lottery":  1.5,  # tight; lottery-vol expansions are short-lived
    "swing":    2.1,  # from replay
    "position": 2.3,  # from replay
    "leap":     2.1,  # from replay (LOW confidence)
    "unknown":  2.4,  # from replay
}

# Time-stop minimum-R to keep a position past its bucket MAX_HOLD. If the
# position has not reached this many R after MAX_HOLD bars, time-stop
# fires. Lower for buckets where +0.5R/3d hit-rates are weaker.
TIME_STOP_MIN_R_BY_BUCKET = {
    "lottery":  0.5,
    "swing":    0.5,
    "position": 1.0,
    "leap":     0.5,
    "unknown":  1.0,
}

# Earnings filter window (decoupled from MAX_HOLD). Skip / warn entries
# where ER falls within this many trading days regardless of intended hold.
EARNINGS_RISK_WINDOW_DAYS = 5

# Legacy aliases. Kept so any code still importing `MAX_HOLD_DAYS` or
# `ATR_TRAIL_MULT` directly keeps working until migrated. New code should
# use `resolve_hold_config(dominant_dte_bucket)` from
# `app/signals/hold_config.py` instead.
MAX_HOLD_DAYS = MAX_HOLD_DAYS_BY_BUCKET["unknown"]

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

# ──────────────────────────────────────────────────────────────────────────
# T/A gating — extension-cap policy.
#
# The dashboard is used for decision support (no live broker), so hiding a
# high-flow setup is more costly than surfacing one we'd size down on. Two
# levers:
#
#   USE_SOFT_EXTENSION_GATE  (Option B)
#       When True, an "extended" price (close > N×ATR from EMA20) no longer
#       hard-rejects the setup in score_long_setup / score_short_setup.
#       Instead the row is promoted to WATCHLIST so it stays visible with
#       the `extended` flag set on the result. Final SIGNAL state still
#       requires the gate to pass — the soft path only converts what would
#       have been a REJECT into a WATCHLIST so the human reviewer can see
#       it. Score still reflects extension via the continuous component.
#
#   USE_SUSTAINED_TREND_EXTENSION  (Option C)
#       When True, sustained multi-day trends get a wider extension cap
#       (SUSTAINED_TREND_MAX_DISTANCE_ATR, default 5.0) instead of the
#       default 2.5 / breakout 4.0. "Sustained" is detected from price
#       alone (clean trend ≥6/10 directional bars + EMA20 stack on the
#       trend side + last close on the trend side of EMA20), so the
#       widening kicks in exactly when the setup is "buy strength, not
#       wait for a pullback that may never come".
#
# Both flags ship True for the dashboard. Disable to revert to the strict
# pullback-style gating.
# ──────────────────────────────────────────────────────────────────────────
USE_SOFT_EXTENSION_GATE = True
USE_SUSTAINED_TREND_EXTENSION = True
EXTENSION_MAX_DISTANCE_ATR = 2.5            # default cap (was 2.0)
EXTENSION_BREAKOUT_MAX_DISTANCE_ATR = 4.0   # SR-breakout cap
EXTENSION_CONSOLIDATION_MAX_DISTANCE_ATR = 3.0  # range-break cap
EXTENSION_SUSTAINED_TREND_MAX_DISTANCE_ATR = 5.0  # multi-day clean-trend cap

# ──────────────────────────────────────────────────────────────────────────
# Watchlist persistence — Layer 2 (streak bonus) + Layer 3 (freight train).
#
# Layer 2 turns the per-day watchlist streak (see app/signals/watchlist.py)
# into a final-score bonus so multi-day rising flow can pull a borderline
# setup over the regime threshold instead of just being a visual hint.
#
#   bonus = base × trend_mult, only applied when mean_flow_5d ≥ floor
#     base       = min((streak_days - MIN + 1) × STEP, MAX_BONUS)
#                  → 3d=0.25, 4d=0.5, 5d=0.75, 6d+=1.0 with defaults
#     trend_mult = rising=1.0 / flat=0.5 / falling=0.0 / n/a=0.5
#
# Layer 3 marks "freight train" candidates — multi-day, rising-flow,
# sector-hot setups that we want to surface even when T/A is borderline.
# v1 stamps a flag on the row (consumed by the dashboard + saw-couldn't-
# trade panel) and does not auto-promote rejects into the SIGNAL list.
# ──────────────────────────────────────────────────────────────────────────
USE_WATCHLIST_STREAK_BONUS = True
WATCHLIST_STREAK_MIN_DAYS = 3
WATCHLIST_STREAK_STEP = 0.25
WATCHLIST_STREAK_MAX_BONUS = 1.0
WATCHLIST_STREAK_MEAN_FLOW_FLOOR = 0.4

USE_FREIGHT_TRAIN_FLAG = True
FREIGHT_TRAIN_MIN_STREAK = 4
FREIGHT_TRAIN_MIN_MEAN_FLOW = 0.5
FREIGHT_TRAIN_REQUIRE_RISING_TREND = True
FREIGHT_TRAIN_SECTOR_HEAT_FLOOR = 5.0  # 0-10 scale from app/features/sector_heat.py

# ──────────────────────────────────────────────────────────────────────────
# Stage E — Flow Tracker auto-promotion.
#
# When enabled, ``app/signals/flow_promote.py`` synthesizes an ATR-based
# trade plan for any Flow Tracker Grade A entry that:
#   - Has conviction_score >= FLOW_PROMOTE_MIN_SCORE.
#   - Has close on the right side of EMA20 (long: above; short: below)
#     when FLOW_PROMOTE_REQUIRE_EMA20_TREND_CONFIRM is True.
#   - Is NOT already in the main signal list.
#
# Promoted signals are tagged with ``promoted_from_flow_tracker=True``
# and carry ``source="flow_promoted"`` so the dashboard can render a
# distinct badge and the "auto-promoted" filter chip targets them.
# ──────────────────────────────────────────────────────────────────────────
FLOW_PROMOTE_ENABLED = True
FLOW_PROMOTE_MIN_SCORE = 8.0
FLOW_PROMOTE_REQUIRE_EMA20_TREND_CONFIRM = True

# Multi-day flow persistence
FLOW_PERSISTENCE_DAYS = 3    # look back this many calendar days
FLOW_PERSISTENCE_BONUS = 0.5 # max score bonus for persistent flow (conservative)

# Rolling z-score baselines for flow components (see app/features/flow_stats.py).
# When USE_ZSCORE_FLOW is True, each listed component in ZSCORE_COMPONENTS is
# scored as (value - median_30d) / MAD_30d instead of against the absolute
# thresholds in flow_features._FLOW_THRESHOLDS. Four-tier fallback ladder
# handles tickers with insufficient history.
#
# History for the z-baseline is sourced from UW's
# /stock/{ticker}/options-volume?limit=30 endpoint (see
# app/features/uw_history.py), cached on disk 24h per ticker. Only components
# in ZSCORE_COMPONENTS that UW can hydrate are z-scored; the rest stay on
# absolute-threshold scoring until hydrated separately.
USE_ZSCORE_FLOW = True            # master switch (UW-backed baseline, intensity only)
ZSCORE_LOOKBACK_DAYS = 30         # trailing window for per-ticker stats
ZSCORE_MIN_N_FULL = 20            # Tier 1 minimum valid observations
ZSCORE_MIN_N_SHRUNK = 5           # Tier 2 minimum (below → Tier 3 cross-sectional)
ZSCORE_SHRINKAGE_K = 10           # Bayesian shrinkage strength for Tier 2 MAD
ZSCORE_CLIP = 3.5                 # clamp |z| to this before logistic
ZSCORE_MIN_COHORT_SIZE = 10       # Tier 3/cross-section needs at least this many tickers
ZSCORE_COMPONENTS = ["flow_intensity"]  # components whose z-baseline is UW-backed;
                                        # others stay on absolute thresholds until
                                        # hydrated separately

# Extended z-score path — hydrates vol_oi and unusual_premium_share in addition
# to flow_intensity via the widened UW baseline (``load_uw_baselines``). Kept
# behind a flag so the wider set can be shadow-logged before cutover; when
# ``USE_ZSCORE_FLOW_EXTENDED`` is True, ``ZSCORE_COMPONENTS_EXTENDED`` is used
# in place of ``ZSCORE_COMPONENTS``. ``unusual_premium_share`` is emitted as a
# shadow component (z + tier columns attach to the feature table) but does not
# feed the weighted ``bullish_score`` / ``bearish_score`` yet — that cutover
# is a separate, weight-aware change.
USE_ZSCORE_FLOW_EXTENDED = False
ZSCORE_COMPONENTS_EXTENDED = ["flow_intensity", "vol_oi", "unusual_premium_share"]

UW_HISTORY_CACHE_TTL_HOURS = 24   # per-ticker UW options-volume history cache TTL

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
FLOW_TRACKER_MAX_RESULTS = 15

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
FLOW_TRACKER_MODE_DEFAULT = "activity"              # UI default on first load
FLOW_TRACKER_AUTO_WIDEN_MIN = 3                     # auto-widen notice threshold
# Flow-Tracker-Swing-Radar: when True, ``compute_multi_day_flow`` hard-
# filters out rows that fail the requested mode (legacy behaviour
# returned all rows tagged with per-mode flags).  Also filters out
# ILLIQUID liquidity tiers in ``_build_flow_tracker``.
FLOW_TRACKER_HARD_MODE_FILTER = True
FLOW_TRACKER_HARD_ILLIQUID_FILTER = True

# Premium-Taxonomy plan — after scripts/purge_snapshot_history.py wipes
# the old screener_snapshots.csv we need at least ~3 trading days of
# fresh rows before the multi-day regression has anything useful to
# say.  While ramping, the Action Bar renders a "Rebuilding history
# (day X of N)" banner sourced from these constants.  Tune N if you
# need a longer warm-up; 3 matches the ``min_active_days=2`` default
# with a one-day buffer.
FLOW_TRACKER_WARMUP_DAYS = 3
FLOW_TRACKER_WARMUP_BANNER_ENABLED = True

FLOW_TRACKER_MODES = {
    # Flow-Tracker bucket redesign (2026-04-30 → 2026-05-05):
    # the old ``accumulation`` (cons≥0.55) / ``strong_accumulation``
    # (cons≥0.65) gates fired 0–1 times across 10 trading days of real
    # screener data because total_{bull,bear}_premium mixes conviction
    # with routine hedging on both sides. Lowering the threshold would
    # admit names with barely-biased noise, so we redesigned the modes
    # around two distinct goals:
    #
    #   - ``activity`` (replaces ``accumulation``) drops the directional-
    #     purity gate entirely and surfaces names with sustained heavy
    #     premium regardless of the bull/bear split. Useful most days.
    #     Backed by dollar floors + acceleration only.
    #
    #   - ``strong_accumulation`` keeps a strict definition but is now
    #     primarily gated on day-LEVEL persistence (≥60% of days clearly
    #     directional, same direction, no flips), with the aggregate
    #     consistency floor as a secondary check. Allowed to be empty
    #     in two-sided regimes — that empty state is informative.
    #
    # Mode gates are hard filters: rows failing the gate are dropped
    # before the 15-row cap so the radar stays tight.
    "all": {
        "label": "All",
        "min_active_days": 2,
        "min_cum_premium": 2_000_000,
        "min_prem_mcap_bps": 0.50,
        "min_consistency": 0.0,
        "min_day_persistence": 0.0,
        "require_no_flips": False,
        "min_accel_t": -99.0,
        "exclude_hedging": False,
        "min_grade_rank": 0,                        # C and above
        "intro": "All tickers clearing the base multi-day persistence gate.",
    },
    "activity": {
        "label": "Activity",
        "min_active_days": 4,
        "min_cum_premium": 25_000_000,
        "min_prem_mcap_bps": 3.0,
        "min_consistency": 0.0,                     # no purity requirement
        "min_day_persistence": 0.0,
        "require_no_flips": False,
        "min_accel_t": 0.0,                         # require flat-or-rising flow
        "exclude_hedging": True,
        "min_grade_rank": 3,                        # B+ and above
        "intro": "Names with sustained multi-day options flow — heavy persistent activity, any direction.",
    },
    "strong_accumulation": {
        "label": "Strong",
        # Empirically calibrated 2026-05-09 against
        # ``data/snapshots_archive.csv.gz`` (15 trading days). The
        # previous gates (``min_active_days=5``, ``min_consistency=0.30``,
        # ``min_accel_t=0.5``, ``DAY_SKEW_FLOOR=0.20``) fired 0/15 days
        # by arithmetic alone — see
        # ``data/diagnostic_strong_calibration_2026-05-09.md`` for the
        # sweep that picked these values. The new gates fire ~0.7/day on
        # average across the same window with a clean shortlist (WULF,
        # MUSA, SATS, POET, FSLR, AXSM, PTCT, QCOM).
        "min_active_days": 4,
        "min_cum_premium": 25_000_000,
        "min_prem_mcap_bps": 5.0,
        # 0.10 ≈ 55/45 cumulative split. 0.30 was unrealistic for liquid
        # mega-caps where puts and calls trade in similar absolute size;
        # the day-level persistence + no-flips gates carry the directional
        # purity work.
        "min_consistency": 0.10,
        "min_day_persistence": 0.60,                # ≥60% of days clearly directional
        "require_no_flips": True,                   # no opposite-direction days
        # 0.5 was too tight on a 4-bar regression (t-stat is noisy with
        # so few points). 0.0 still requires the log-linear regression
        # of daily premium to be flat-or-rising.
        "min_accel_t": 0.0,
        "exclude_hedging": True,
        "min_grade_rank": 4,                        # A- and above
        "intro": "Rare: 4+ active days, every active day leans the same direction, no flips, flat-or-rising premium, A-tier grade.",
    },
    # Early — added 2026-05-09 to bridge the gap between 1-day flow
    # (noise) and the multi-day Activity / Strong gates. Captures a
    # 2-day same-direction confirmation pattern: both active days have
    # to lean the same way (``min_day_persistence=1.0``) with zero
    # opposite-direction days (``require_no_flips``). Calibrated to
    # produce ~6 names/day across the empirical sweep window — small
    # enough to manually skim, big enough to surface emerging flow
    # before it matures into a Strong / Activity hit.
    #
    # Note: this mode operates on whatever lookback the active horizon
    # uses (5d / 15d). The 2-day pattern carries directional meaning
    # because of ``min_day_persistence=1.0`` — *every* active day in
    # the window must lean the same direction, not just a window-
    # average preference.
    "early_accumulation": {
        "label": "Early",
        "min_active_days": 2,
        "min_cum_premium": 5_000_000,
        "min_prem_mcap_bps": 2.0,
        "min_consistency": 0.10,
        "min_day_persistence": 1.00,                # both active days same direction
        "require_no_flips": True,                   # absolute purity — any flip kills the row
        "min_accel_t": -99.0,                       # acceleration t-stat meaningless on 2 points
        "exclude_hedging": True,
        "min_grade_rank": 3,                        # B+ and above
        "intro": "2-day same-direction confirmation: at least 2 active days, every one leaning the same way, no flips. The early-radar between 1-day noise and Strong's multi-day confirmation.",
    },
}

# Day-level skew floor used by ``_compute_day_persistence`` to classify
# a single day as "clearly directional". 0.20 was the original value but
# proved too tight: liquid mega-caps where puts and calls trade in
# similar absolute size rarely cross 60/40 on single-day flow even when
# the cumulative bias is unmistakable (QCOM 2026-05-04→07: bull > bear
# every day, every daily skew under 0.12, all classified FLAT under
# 0.20). Empirically calibrated to **0.10** (≈55/45) on 2026-05-09 —
# strong-purity gates (``require_no_flips``, ``min_day_persistence``)
# do the directional-classification work; this floor only defines what
# counts as a directional vs flat day.
FLOW_TRACKER_DAY_SKEW_FLOOR = 0.10

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

# Wire-through for the walk-forward weight refit (Stage D).
#
# When ``USE_RECALIBRATED_WEIGHTS`` is True, ``app.features.flow_tracker``
# checks ``data/conviction_recalibration.json`` at import time. If the
# global fit reports ``accept=True`` AND its OOS Spearman is positive,
# the weights from that fit replace ``FLOW_TRACKER_WEIGHTS_ACCUM`` for
# scoring. Otherwise the legacy weights stay in effect — the scripts/
# pipeline never silently downgrades to a fit that didn't validate.
USE_RECALIBRATED_WEIGHTS = True
RECALIBRATED_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "conviction_recalibration.json"
)
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

# Premium-taxonomy DTE buckets.  These are screening-friendly bucket
# definitions that feed the Trader Card Premium-Mix panel and the
# per-bucket `lottery_*_premium` / `swing_*_premium` / `leap_*_premium`
# columns persisted to screener_snapshots.csv.  Kept separate from
# FLOW_TRACKER_DTE_BUCKETS above so the dominant-bucket grade-explainer
# logic is unaffected.
#
# Buckets:
#   lottery (0-14d)   — event speculation / earnings bets / squeezes
#   swing   (30-120d) — institutional swing window (signal pipeline home)
#   leap    (180d+)   — deep positional commitment
#
# DTEs that fall in the gaps (15-29d, 121-179d) land in the "other"
# bucket and are tracked implicitly as total - sum(buckets).
FLOW_TRACKER_PREMIUM_BUCKETS = [
    # (label, dte_min, dte_max, tooltip)
    ("lottery", 0,   14,   "0-14d - event speculation, earnings bets, squeezes"),
    ("swing",   30,  120,  "30-120d - institutional swing window"),
    ("leap",    180, 9999, "180d+ - deep positional commitment"),
]

# Wave 0.5 A6: require perc_3_day_total above this percentile for the
# accumulation/strong-accumulation gates.  `perc_3_day_total` is UW's
# normalized unusualness score (0..1) across the universe — values above
# 0.70 mean "top 30% of unusual 3-day activity".
FLOW_TRACKER_MIN_3D_PERCENTILE = 0.70

# Wave 0.5 A3: when True, flow alert fetches include UW's
# `size_greater_oi=true` flag so we only count prints that likely opened a
# position (trade size > current OI).  Closing-trades flow the same way on
# the tape as opening-trades, so leaving this OFF silently dilutes the
# directional signal with profit-taking / short-covering activity.  Cut
# over to True as the default; the UI's `/api/alerts?opening_only=0` knob
# still allows discretionary exploration without mutating this flag.
FLOW_OPENING_ONLY = True

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

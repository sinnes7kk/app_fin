"""Position health scoring for open positions.

Computes a 0-10 health score that measures whether an open position's thesis
is still valid, the trade is progressing, and the risk profile is acceptable.
This is fundamentally different from the entry conviction score — it is
path-dependent and incorporates realized trade behavior.

Health components (max 10 points):
  - Trend intact:        3 pts
  - Structure intact:    2 pts
  - Momentum:            2 pts
  - Distance to target:  1 pt
  - Options context:     2 pts (market agreement + structural risk)

Penalties applied after base score:
  - Conviction decay for stagnant trades
  - Stop proximity penalty
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.config import (
    HEALTH_FAILING_THRESHOLD,
    HEALTH_STRONG_THRESHOLD,
    HEALTH_WEAK_THRESHOLD,
)
from app.features.options_context import fetch_options_context
from app.features.price_features import clean_ohlcv, compute_features, fetch_ohlcv
from app.rules.continuation_rules import detect_trend


def _trend_score(df: pd.DataFrame, direction: str) -> float:
    """Trend intact component (0-3).

    Full credit if trend direction matches position. Partial credit if NEUTRAL
    but EMA slope is favorable.
    """
    trend = detect_trend(df)
    trend_dir = trend.get("trend", "NEUTRAL")
    strength = trend.get("strength", 0.0)

    if direction == "LONG" and trend_dir == "LONG":
        return 2.0 + min(strength, 1.0)
    if direction == "SHORT" and trend_dir == "SHORT":
        return 2.0 + min(strength, 1.0)

    # Partial credit: NEUTRAL trend but slope favors direction
    last = df.iloc[-1]
    slope = float(last["ema20_slope"]) if "ema20_slope" in last.index and not pd.isna(last.get("ema20_slope")) else 0.0
    if trend_dir == "NEUTRAL":
        if (direction == "LONG" and slope > 0) or (direction == "SHORT" and slope < 0):
            return 1.5
        return 0.5

    return 0.0


def _structure_score(pos: dict, df: pd.DataFrame) -> float:
    """Structure intact component (0-2).

    1 pt: price still on correct side of support/resistance.
    1 pt: stop has comfortable distance (> 0.5 ATR from price).
    """
    last = df.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr14"]) if "atr14" in last.index else 0.0

    pts = 0.0

    if pos["direction"] == "LONG":
        if close > pos.get("initial_stop", 0):
            pts += 1.0
    else:
        if close < pos.get("initial_stop", float("inf")):
            pts += 1.0

    active_stop = pos.get("active_stop", pos.get("initial_stop", 0))
    if atr > 0:
        if pos["direction"] == "LONG":
            stop_dist = (close - active_stop) / atr
        else:
            stop_dist = (active_stop - close) / atr
        if stop_dist > 0.5:
            pts += 1.0

    return pts


def _momentum_score(pos: dict, df: pd.DataFrame) -> float:
    """Momentum component (0-2).

    1 pt: EMA20 slope aligned with direction.
    1 pt: last 3 bars making progress (higher highs/lows for long, inverse for short).
    """
    last = df.iloc[-1]
    slope = float(last["ema20_slope"]) if "ema20_slope" in last.index and not pd.isna(last.get("ema20_slope")) else 0.0

    pts = 0.0
    if (pos["direction"] == "LONG" and slope > 0) or (pos["direction"] == "SHORT" and slope < 0):
        pts += 1.0

    if len(df) >= 4:
        recent = df.iloc[-3:]
        if pos["direction"] == "LONG":
            highs_rising = all(
                float(recent.iloc[i]["high"]) >= float(recent.iloc[i - 1]["high"])
                for i in range(1, len(recent))
            )
            if highs_rising:
                pts += 1.0
        else:
            lows_falling = all(
                float(recent.iloc[i]["low"]) <= float(recent.iloc[i - 1]["low"])
                for i in range(1, len(recent))
            )
            if lows_falling:
                pts += 1.0

    return pts


def _target_distance_score(pos: dict, close: float) -> float:
    """Distance to target component (0-1). Closer to T2 = higher health."""
    entry = pos["entry_price"]
    t2 = pos.get("target_2", entry)
    if pos["direction"] == "LONG":
        total_range = t2 - entry
        progress = close - entry
    else:
        total_range = entry - t2
        progress = entry - close

    if total_range <= 0:
        return 0.5
    return round(max(0.0, min(progress / total_range, 1.0)), 3)


def _options_context_score(pos: dict, opts_ctx: dict | None, enrichment: dict | None = None) -> float:
    """Options context component (0-2): market agreement + structural risk.

    Each sub-score is 0-1. Returns neutral 1.0 when data is unavailable.
    When enrichment data (net premium ticks, dark pool, flow-recent) is
    available, it improves the market agreement sub-score.
    """
    if not opts_ctx or not opts_ctx.get("options_context_available"):
        return 1.0

    direction = pos["direction"]

    # Sub-score 1: Market agreement (premium bias + volume aggressiveness)
    bullish_p = opts_ctx.get("daily_bullish_premium")
    bearish_p = opts_ctx.get("daily_bearish_premium")
    if bullish_p is not None and bearish_p is not None and (bullish_p + bearish_p) > 0:
        total = bullish_p + bearish_p
        agreement = bullish_p / total if direction == "LONG" else bearish_p / total
    else:
        agreement = 0.5

    # Refine agreement with live enrichment data when available
    if enrichment:
        npt = enrichment.get("net_prem_ticks")
        if npt:
            prem_dir = npt.get("intraday_premium_direction", 0.5)
            live_agree = prem_dir if direction == "LONG" else 1.0 - prem_dir
            agreement = 0.6 * agreement + 0.4 * live_agree

        dp = enrichment.get("dark_pool")
        if dp:
            dp_bias = dp.get("dark_pool_bias", 0.5)
            dp_agree = dp_bias if direction == "LONG" else 1.0 - dp_bias
            agreement = 0.8 * agreement + 0.2 * dp_agree

    # Sub-score 2: Structural risk (wall proximity + gamma alignment)
    gamma = opts_ctx.get("gamma_regime", "NEUTRAL")

    if direction == "LONG":
        wall_dist = opts_ctx.get("distance_to_call_wall_pct")
        gamma_favorable = gamma == "POSITIVE"
        gamma_adverse = gamma == "NEGATIVE"
    else:
        wall_dist = opts_ctx.get("distance_to_put_wall_pct")
        gamma_favorable = gamma == "NEGATIVE"
        gamma_adverse = gamma == "POSITIVE"

    wall_component = 0.5
    if wall_dist is not None:
        wall_component = max(0.0, min((wall_dist - 1.0) / 4.0, 1.0))

    gamma_component = 0.5
    if gamma_favorable:
        gamma_component = 1.0
    elif gamma_adverse:
        gamma_component = 0.0

    structural = (wall_component + gamma_component) / 2.0

    return round(agreement + structural, 3)


def _conviction_decay(pos: dict, close: float) -> float:
    """Penalty for stagnant trades that aren't making progress."""
    days = pos.get("days_held", 0)
    risk = pos.get("risk_per_share", 0)
    if risk <= 0:
        return 0.0

    entry = pos["entry_price"]
    if pos["direction"] == "LONG":
        ur = (close - entry) / risk
    else:
        ur = (entry - close) / risk

    penalty = 0.0
    if days > 3 and ur < 0.5:
        penalty += 1.0
    if days > 5 and ur < 1.0:
        penalty += 1.0
    return penalty


def _stop_proximity_penalty(pos: dict, close: float, atr: float) -> float:
    """Penalty when price is within 0.5 ATR of the active stop."""
    if atr <= 0:
        return 0.0
    active_stop = pos.get("active_stop", pos.get("initial_stop", 0))
    if pos["direction"] == "LONG":
        dist = (close - active_stop) / atr
    else:
        dist = (active_stop - close) / atr
    if dist < 0.5:
        return 2.0
    return 0.0


# ---------------------------------------------------------------------------
# Stage F.1 — flow-aware health components.
#
# These three penalties give the existing trade-management framework an
# *exit signal* sourced from Flow Tracker / sector heat / unusual flow
# rather than purely from price action. Rationale: a position that
# entered on Grade A flow but whose flow has decayed materially is in a
# worse state than the price chart alone suggests — we want to start
# tightening the trail / preferring rotation before the price-based
# trail catches up.
#
# All three penalties read from per-position fields stamped by Stage F.2
# (``update_positions`` enriches each open position daily with
# ``current_grade``, ``current_sector_heat``, ``current_unusual_flow_dir``).
# When the enrichment is missing, the penalty is 0 (no opinion).
# ---------------------------------------------------------------------------


_GRADE_RANK = {"C": 0, "B-": 1, "B": 2, "B+": 3, "A-": 4, "A": 5, "A+": 6}


def _grade_rank(label: str | None) -> int | None:
    if not label:
        return None
    s = str(label).strip()
    return _GRADE_RANK.get(s)


def _grade_decay_penalty(pos: dict) -> float:
    """Penalty when current Flow Tracker grade has dropped from entry.

    Compares ``pos['current_grade']`` against ``pos['conviction_grade']``
    (stamped at entry). 1 tier drop → 0.5 penalty, 2 tiers → 1.0,
    3+ tiers → 1.5. Caps at 1.5 so a single slow-bleed position can't
    completely zero out health.
    """
    entry_grade = pos.get("conviction_grade") or pos.get("entry_conviction_grade")
    cur_grade = pos.get("current_grade")
    er = _grade_rank(entry_grade)
    cr = _grade_rank(cur_grade)
    if er is None or cr is None:
        return 0.0
    drop = er - cr
    if drop <= 0:
        return 0.0
    return min(0.5 * drop, 1.5)


def _sector_heat_reversal_penalty(pos: dict) -> float:
    """Penalty when sector heat has flipped against the position direction.

    ``pos['current_sector_heat']`` is the live sector-heat aggregate
    (``-10..+10`` signed scale; positive = bullish basket, negative =
    bearish basket). For a LONG position, a heat reading of < -2 is a
    notable reversal. For SHORT, > +2 is.
    """
    heat = pos.get("current_sector_heat")
    if heat is None:
        return 0.0
    try:
        h = float(heat)
    except (TypeError, ValueError):
        return 0.0
    direction = pos.get("direction", "LONG")
    if direction == "LONG" and h <= -2.0:
        return min(abs(h) / 5.0, 1.5)  # at h=-7.5 → cap 1.5
    if direction == "SHORT" and h >= 2.0:
        return min(h / 5.0, 1.5)
    return 0.0


def _unusual_flow_flip_penalty(pos: dict) -> float:
    """Penalty when the ticker's unusual options flow has flipped direction.

    ``pos['current_unusual_flow_dir']`` is one of ``"BULLISH"``,
    ``"BEARISH"`` or ``None`` (insufficient flow). Flip = opposite to
    position direction. Flat penalty (1.0) since this is binary.
    """
    cur = pos.get("current_unusual_flow_dir")
    if not cur:
        return 0.0
    direction = pos.get("direction", "LONG")
    if direction == "LONG" and str(cur).upper() == "BEARISH":
        return 1.0
    if direction == "SHORT" and str(cur).upper() == "BULLISH":
        return 1.0
    return 0.0


def _flow_decay_factor(pos: dict) -> float:
    """Compute a 0-1 factor used by Stage F.3 to tighten trailing stops.

    Returns 1.0 = no tightening, 0.0 = maximum tightening (stop moves to
    its post-T1 / hybrid target). Linear blend of the three flow-aware
    penalties capped to a sensible range.
    """
    total_pen = (
        _grade_decay_penalty(pos)
        + _sector_heat_reversal_penalty(pos)
        + _unusual_flow_flip_penalty(pos)
    )
    # Total possible: 1.5 + 1.5 + 1.0 = 4.0. Normalize and invert.
    return float(max(0.0, min(1.0, 1.0 - total_pen / 4.0)))


def _classify_state(health: float) -> str:
    if health >= HEALTH_STRONG_THRESHOLD:
        return "STRONG"
    if health >= HEALTH_WEAK_THRESHOLD:
        return "NEUTRAL"
    if health >= HEALTH_FAILING_THRESHOLD:
        return "WEAK"
    return "FAILING"


def compute_position_health(pos: dict, enrichment: dict | None = None) -> dict:
    """Compute the health score and state for an open position.

    Parameters
    ----------
    pos : dict
        The position dict.
    enrichment : dict, optional
        Live enrichment data from net_prem_ticks, dark_pool, flow_recent.

    Returns a dict with:
      - health: float (0-10)
      - health_state: STRONG | NEUTRAL | WEAK | FAILING
      - health_at_entry: float (set on first call, preserved thereafter)
      - health_prev: float (previous health before this call)
      - health_delta: float (change since last computation)
      - components: dict breakdown for debugging
    """
    ticker = pos["ticker"]

    try:
        df = compute_features(clean_ohlcv(fetch_ohlcv(ticker)))
    except Exception:
        prev = pos.get("health", 5.0)
        return {
            "health": prev,
            "health_state": _classify_state(prev),
            "health_at_entry": pos.get("health_at_entry", prev),
            "health_prev": prev,
            "health_delta": 0.0,
            "components": {},
        }

    close = float(df.iloc[-1]["close"])
    atr = float(df.iloc[-1]["atr14"]) if "atr14" in df.columns else 0.0

    try:
        opts_ctx = fetch_options_context(ticker, close)
    except Exception:
        opts_ctx = None

    trend = _trend_score(df, pos["direction"])
    structure = _structure_score(pos, df)
    momentum = _momentum_score(pos, df)
    target_dist = _target_distance_score(pos, close)
    options = _options_context_score(pos, opts_ctx, enrichment=enrichment)

    base = trend + structure + momentum + target_dist + options
    decay = _conviction_decay(pos, close)
    stop_pen = _stop_proximity_penalty(pos, close, atr)
    # Stage F.1 — flow-aware penalties (max 4.0 total, capped per-component).
    grade_decay_pen = _grade_decay_penalty(pos)
    sector_pen = _sector_heat_reversal_penalty(pos)
    unusual_pen = _unusual_flow_flip_penalty(pos)
    flow_total_pen = grade_decay_pen + sector_pen + unusual_pen
    health = max(0.0, min(base - decay - stop_pen - flow_total_pen, 10.0))

    prev_health = pos.get("health")
    health_at_entry = pos.get("health_at_entry")

    if prev_health is None:
        health_prev = health
        health_delta = 0.0
        h_at_entry = health
    else:
        health_prev = prev_health
        health_delta = round(health - prev_health, 2)
        h_at_entry = health_at_entry if health_at_entry is not None else health

    return {
        "health": round(health, 2),
        "health_state": _classify_state(health),
        "health_at_entry": round(h_at_entry, 2),
        "health_prev": round(health_prev, 2),
        "health_delta": health_delta,
        "components": {
            "trend": round(trend, 2),
            "structure": round(structure, 2),
            "momentum": round(momentum, 2),
            "target_distance": round(target_dist, 3),
            "options": round(options, 3),
            "decay_penalty": round(decay, 2),
            "stop_penalty": round(stop_pen, 2),
            "grade_decay_penalty": round(grade_decay_pen, 2),
            "sector_heat_reversal_penalty": round(sector_pen, 2),
            "unusual_flow_flip_penalty": round(unusual_pen, 2),
        },
    }

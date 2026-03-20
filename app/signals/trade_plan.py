"""Trade plan generation and trailing stop computation."""

from __future__ import annotations

import pandas as pd

from app.config import ATR_TRAIL_MULT, HYBRID_TRAIL_MULT, MAX_HOLD_DAYS

MAX_STOP_PCT = 0.12
MIN_RR = 2.0


def _cap_stop(entry: float, raw_stop: float, direction: str) -> float:
    """Clamp stop so it never exceeds MAX_STOP_PCT from entry."""
    if direction == "LONG":
        floor = entry * (1 - MAX_STOP_PCT)
        return max(raw_stop, floor)
    ceil = entry * (1 + MAX_STOP_PCT)
    return min(raw_stop, ceil)


def build_long_trade_plan(df: pd.DataFrame, scored_signal: dict) -> dict:
    """Build a long trade plan using S/R-based targets with R:R filtering."""
    if not scored_signal.get("is_valid", False):
        raise ValueError("Cannot build long trade plan for invalid signal")

    last = df.iloc[-1]

    entry_price = float(last["close"])
    atr = float(last["atr14"])
    latest_low = float(last["low"])
    support = float(scored_signal["support"])
    resistance = float(scored_signal["resistance"])

    entry_zone_low = entry_price - 0.25 * atr
    entry_zone_high = entry_price + 0.25 * atr

    raw_stop = min(latest_low, support) - 0.25 * atr
    stop_price = _cap_stop(entry_price, raw_stop, "LONG")
    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        raise ValueError("Invalid long trade plan: non-positive risk per share")

    t1_rmultiple = entry_price + 2.0 * risk_per_share
    if resistance > entry_price + MIN_RR * risk_per_share:
        target_1 = resistance
    else:
        target_1 = t1_rmultiple
    target_2 = entry_price + 3.0 * risk_per_share

    rr_ratio = (target_1 - entry_price) / risk_per_share

    return {
        "ticker": scored_signal["ticker"],
        "direction": "LONG",
        "score": scored_signal["score"],
        "entry_price": round(entry_price, 2),
        "entry_zone_low": round(entry_zone_low, 2),
        "entry_zone_high": round(entry_zone_high, 2),
        "stop_price": round(stop_price, 2),
        "risk_per_share": round(risk_per_share, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "rr_ratio": round(rr_ratio, 2),
        "time_stop_days": MAX_HOLD_DAYS,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "reasons": scored_signal["reasons"],
    }


def build_short_trade_plan(df: pd.DataFrame, scored_signal: dict) -> dict:
    """Build a short trade plan using S/R-based targets with R:R filtering."""
    if not scored_signal.get("is_valid", False):
        raise ValueError("Cannot build short trade plan for invalid signal")

    last = df.iloc[-1]

    entry_price = float(last["close"])
    atr = float(last["atr14"])
    latest_high = float(last["high"])
    support = float(scored_signal["support"])
    resistance = float(scored_signal["resistance"])

    entry_zone_low = entry_price - 0.25 * atr
    entry_zone_high = entry_price + 0.25 * atr

    raw_stop = max(latest_high, resistance) + 0.25 * atr
    stop_price = _cap_stop(entry_price, raw_stop, "SHORT")
    risk_per_share = stop_price - entry_price

    if risk_per_share <= 0:
        raise ValueError("Invalid short trade plan: non-positive risk per share")

    t1_rmultiple = entry_price - 2.0 * risk_per_share
    if support < entry_price - MIN_RR * risk_per_share:
        target_1 = support
    else:
        target_1 = t1_rmultiple
    target_2 = entry_price - 3.0 * risk_per_share

    rr_ratio = (entry_price - target_1) / risk_per_share

    return {
        "ticker": scored_signal["ticker"],
        "direction": "SHORT",
        "score": scored_signal["score"],
        "entry_price": round(entry_price, 2),
        "entry_zone_low": round(entry_zone_low, 2),
        "entry_zone_high": round(entry_zone_high, 2),
        "stop_price": round(stop_price, 2),
        "risk_per_share": round(risk_per_share, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "rr_ratio": round(rr_ratio, 2),
        "time_stop_days": MAX_HOLD_DAYS,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "reasons": scored_signal["reasons"],
    }


def compute_trailing_stops(pos: dict, df: pd.DataFrame) -> dict:
    """Compute all three trailing stop levels for an open position.

    Returns a dict with updated best_price, trail values, active_stop,
    and unrealized_r. The caller merges these into the position record.
    """
    last = df.iloc[-1]
    atr = float(last["atr14"])
    ema20 = float(last["ema20"])
    close = float(last["close"])

    direction = pos["direction"]
    entry = pos["entry_price"]
    initial_stop = pos["initial_stop"]
    risk = pos["risk_per_share"]
    best = pos["best_price"]
    days = pos["days_held"]

    is_long = direction == "LONG"

    if is_long:
        best = max(best, float(last["high"]))
        unrealized_r = (close - entry) / risk if risk > 0 else 0.0
    else:
        best = min(best, float(last["low"]))
        unrealized_r = (entry - close) / risk if risk > 0 else 0.0

    # ATR Chandelier
    if is_long:
        trail_atr = best - ATR_TRAIL_MULT * atr
    else:
        trail_atr = best + ATR_TRAIL_MULT * atr

    # EMA Trail
    trail_ema = ema20

    # Hybrid (breakeven + tighter ATR)
    if unrealized_r < 1.0:
        trail_hybrid = initial_stop
    elif unrealized_r < 2.0:
        trail_hybrid = entry
    else:
        if is_long:
            trail_hybrid = best - HYBRID_TRAIL_MULT * atr
        else:
            trail_hybrid = best + HYBRID_TRAIL_MULT * atr

    # Time-based tightening: weak P&L late in hold → force breakeven
    if days >= 4 and unrealized_r < 0.5:
        trail_hybrid = entry

    # Never let a trail be worse than the initial stop
    if is_long:
        trail_atr = max(trail_atr, initial_stop)
        trail_hybrid = max(trail_hybrid, initial_stop)
        active_stop = max(trail_atr, trail_ema, trail_hybrid)
    else:
        trail_atr = min(trail_atr, initial_stop)
        trail_hybrid = min(trail_hybrid, initial_stop)
        active_stop = min(trail_atr, trail_ema, trail_hybrid)

    return {
        "best_price": round(best, 2),
        "trail_atr": round(trail_atr, 2),
        "trail_ema": round(trail_ema, 2),
        "trail_hybrid": round(trail_hybrid, 2),
        "active_stop": round(active_stop, 2),
        "unrealized_r": round(unrealized_r, 2),
    }

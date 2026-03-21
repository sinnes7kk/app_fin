"""Trade plan generation and trailing stop computation."""

from __future__ import annotations

import pandas as pd

from app.config import (
    ATR_TRAIL_MULT,
    GAMMA_NEGATIVE_TARGET_MULT,
    GAMMA_NEGATIVE_TRAIL_MULT,
    GAMMA_POSITIVE_TARGET_MULT,
    GAMMA_POSITIVE_TRAIL_MULT,
    HYBRID_TRAIL_MULT,
    MAX_HOLD_DAYS,
    WALL_PROXIMITY_WARNING_ATR,
    WALL_PROXIMITY_WARNING_PCT,
)

MAX_STOP_PCT = 0.12
MIN_RR = 2.0

WALL_TIGHTEN_ATR = 1.5


def _cap_stop(entry: float, raw_stop: float, direction: str) -> float:
    """Clamp stop so it never exceeds MAX_STOP_PCT from entry."""
    if direction == "LONG":
        floor = entry * (1 - MAX_STOP_PCT)
        return max(raw_stop, floor)
    ceil = entry * (1 + MAX_STOP_PCT)
    return min(raw_stop, ceil)


def _gamma_target_mult(ctx: dict | None) -> float:
    """Return a target multiplier based on gamma regime, defaulting to 1.0."""
    if not ctx or not ctx.get("options_context_available"):
        return 1.0
    regime = ctx.get("gamma_regime", "NEUTRAL")
    if regime == "NEGATIVE":
        return GAMMA_NEGATIVE_TARGET_MULT
    if regime == "POSITIVE":
        return GAMMA_POSITIVE_TARGET_MULT
    return 1.0


def _apply_wall_cap_long(
    entry: float,
    atr: float,
    target_1: float,
    target_2: float,
    ctx: dict | None,
) -> tuple[float, float, dict]:
    """Cap long targets at the call wall (regime-dependent) and return options metadata."""
    meta: dict = {
        "call_wall": None,
        "put_wall": None,
        "gamma_regime": None,
        "gamma_multiplier": 1.0,
        "wall_capped_t1": False,
        "wall_capped_t2": False,
        "wall_dist_pct": None,
        "wall_dist_atr": None,
        "wall_proximity_warning": False,
    }
    notes: list[str] = []

    if not ctx or not ctx.get("options_context_available"):
        return target_1, target_2, {"options_context": meta, "notes": notes}

    regime = ctx.get("gamma_regime", "NEUTRAL")
    mult = _gamma_target_mult(ctx)
    call_wall = ctx.get("nearest_call_wall")
    put_wall = ctx.get("nearest_put_wall")

    meta["gamma_regime"] = regime
    meta["gamma_multiplier"] = mult
    meta["call_wall"] = call_wall
    meta["put_wall"] = put_wall

    if mult != 1.0:
        notes.append(f"{regime.lower()} gamma: target multiplier {mult:.2f}")

    if call_wall is not None and call_wall > entry and atr > 0:
        dist_pct = (call_wall - entry) / entry
        dist_atr = (call_wall - entry) / atr
        meta["wall_dist_pct"] = round(dist_pct, 4)
        meta["wall_dist_atr"] = round(dist_atr, 2)

        if dist_pct < WALL_PROXIMITY_WARNING_PCT and dist_atr < WALL_PROXIMITY_WARNING_ATR:
            meta["wall_proximity_warning"] = True
            notes.append(f"wall proximity warning: call wall {dist_pct:.1%} / {dist_atr:.1f} ATR away")

        if target_1 > call_wall:
            old_t1 = target_1
            target_1 = call_wall
            meta["wall_capped_t1"] = True
            notes.append(f"call wall capped target_1: {old_t1:.2f} -> {call_wall:.2f}")

        if regime == "POSITIVE" and target_2 > call_wall:
            old_t2 = target_2
            target_2 = call_wall
            meta["wall_capped_t2"] = True
            notes.append(f"positive gamma: capped target_2 at call wall {call_wall:.2f} (was {old_t2:.2f})")
        elif regime != "POSITIVE" and target_2 > call_wall:
            notes.append(f"{regime.lower()} gamma: runner target_2 allowed beyond call wall")

    return target_1, target_2, {"options_context": meta, "notes": notes}


def _apply_wall_cap_short(
    entry: float,
    atr: float,
    target_1: float,
    target_2: float,
    ctx: dict | None,
) -> tuple[float, float, dict]:
    """Cap short targets at the put wall (regime-dependent) and return options metadata."""
    meta: dict = {
        "call_wall": None,
        "put_wall": None,
        "gamma_regime": None,
        "gamma_multiplier": 1.0,
        "wall_capped_t1": False,
        "wall_capped_t2": False,
        "wall_dist_pct": None,
        "wall_dist_atr": None,
        "wall_proximity_warning": False,
    }
    notes: list[str] = []

    if not ctx or not ctx.get("options_context_available"):
        return target_1, target_2, {"options_context": meta, "notes": notes}

    regime = ctx.get("gamma_regime", "NEUTRAL")
    mult = _gamma_target_mult(ctx)
    call_wall = ctx.get("nearest_call_wall")
    put_wall = ctx.get("nearest_put_wall")

    meta["gamma_regime"] = regime
    meta["gamma_multiplier"] = mult
    meta["call_wall"] = call_wall
    meta["put_wall"] = put_wall

    if mult != 1.0:
        notes.append(f"{regime.lower()} gamma: target multiplier {mult:.2f}")

    if put_wall is not None and put_wall < entry and atr > 0:
        dist_pct = (entry - put_wall) / entry
        dist_atr = (entry - put_wall) / atr
        meta["wall_dist_pct"] = round(dist_pct, 4)
        meta["wall_dist_atr"] = round(dist_atr, 2)

        if dist_pct < WALL_PROXIMITY_WARNING_PCT and dist_atr < WALL_PROXIMITY_WARNING_ATR:
            meta["wall_proximity_warning"] = True
            notes.append(f"wall proximity warning: put wall {dist_pct:.1%} / {dist_atr:.1f} ATR away")

        if target_1 < put_wall:
            old_t1 = target_1
            target_1 = put_wall
            meta["wall_capped_t1"] = True
            notes.append(f"put wall capped target_1: {old_t1:.2f} -> {put_wall:.2f}")

        if regime == "POSITIVE" and target_2 < put_wall:
            old_t2 = target_2
            target_2 = put_wall
            meta["wall_capped_t2"] = True
            notes.append(f"positive gamma: capped target_2 at put wall {put_wall:.2f} (was {old_t2:.2f})")
        elif regime != "POSITIVE" and target_2 < put_wall:
            notes.append(f"{regime.lower()} gamma: runner target_2 allowed beyond put wall")

    return target_1, target_2, {"options_context": meta, "notes": notes}


def build_long_trade_plan(
    df: pd.DataFrame,
    scored_signal: dict,
    options_ctx: dict | None = None,
) -> dict:
    """Build a long trade plan using structural S/R-based targets with R:R filtering.

    When options_ctx is available, targets are adjusted by gamma regime and
    capped at the nearest call wall (T1 always, T2 only in positive gamma).
    """
    if not scored_signal.get("is_valid", False):
        raise ValueError("Cannot build long trade plan for invalid signal")

    last = df.iloc[-1]

    entry_price = float(last["close"])
    atr = float(last["atr14"])
    latest_low = float(last["low"])
    support = float(scored_signal["support"])
    structural_resistance = float(scored_signal.get("structural_resistance", scored_signal["resistance"]))

    entry_zone_low = entry_price - 0.25 * atr
    entry_zone_high = entry_price + 0.25 * atr

    raw_stop = min(latest_low, support) - 0.25 * atr
    stop_price = _cap_stop(entry_price, raw_stop, "LONG")
    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        raise ValueError("Invalid long trade plan: non-positive risk per share")

    gamma_mult = _gamma_target_mult(options_ctx)

    t1_rmultiple = entry_price + 2.0 * risk_per_share * gamma_mult
    if structural_resistance > entry_price + MIN_RR * risk_per_share:
        target_1 = structural_resistance
    else:
        target_1 = t1_rmultiple
    target_2 = entry_price + 3.0 * risk_per_share * gamma_mult

    target_1, target_2, opts_block = _apply_wall_cap_long(
        entry_price, atr, target_1, target_2, options_ctx,
    )

    rr_ratio = (target_1 - entry_price) / risk_per_share

    plan = {
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
        "resistance": round(structural_resistance, 2),
        "reasons": scored_signal["reasons"],
    }
    plan.update(opts_block)
    return plan


def build_short_trade_plan(
    df: pd.DataFrame,
    scored_signal: dict,
    options_ctx: dict | None = None,
) -> dict:
    """Build a short trade plan using structural S/R-based targets with R:R filtering.

    When options_ctx is available, targets are adjusted by gamma regime and
    capped at the nearest put wall (T1 always, T2 only in positive gamma).
    """
    if not scored_signal.get("is_valid", False):
        raise ValueError("Cannot build short trade plan for invalid signal")

    last = df.iloc[-1]

    entry_price = float(last["close"])
    atr = float(last["atr14"])
    latest_high = float(last["high"])
    resistance = float(scored_signal["resistance"])
    structural_support = float(scored_signal.get("structural_support", scored_signal["support"]))

    entry_zone_low = entry_price - 0.25 * atr
    entry_zone_high = entry_price + 0.25 * atr

    raw_stop = max(latest_high, resistance) + 0.25 * atr
    stop_price = _cap_stop(entry_price, raw_stop, "SHORT")
    risk_per_share = stop_price - entry_price

    if risk_per_share <= 0:
        raise ValueError("Invalid short trade plan: non-positive risk per share")

    gamma_mult = _gamma_target_mult(options_ctx)

    t1_rmultiple = entry_price - 2.0 * risk_per_share * gamma_mult
    if structural_support < entry_price - MIN_RR * risk_per_share:
        target_1 = structural_support
    else:
        target_1 = t1_rmultiple
    target_2 = entry_price - 3.0 * risk_per_share * gamma_mult

    target_1, target_2, opts_block = _apply_wall_cap_short(
        entry_price, atr, target_1, target_2, options_ctx,
    )

    rr_ratio = (entry_price - target_1) / risk_per_share

    plan = {
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
        "support": round(structural_support, 2),
        "resistance": round(resistance, 2),
        "reasons": scored_signal["reasons"],
    }
    plan.update(opts_block)
    return plan


def _gamma_trail_mult(ctx: dict | None) -> float:
    """Return a trailing-stop ATR multiplier adjustment based on gamma regime."""
    if not ctx or not ctx.get("options_context_available"):
        return 1.0
    regime = ctx.get("gamma_regime", "NEUTRAL")
    if regime == "NEGATIVE":
        return GAMMA_NEGATIVE_TRAIL_MULT
    if regime == "POSITIVE":
        return GAMMA_POSITIVE_TRAIL_MULT
    return 1.0


def _wall_proximity_blend(
    price: float,
    wall: float | None,
    atr: float,
    is_long: bool,
) -> float:
    """Return a blend factor 0..1 indicating how much to tighten toward the hybrid
    multiplier when price approaches a wall.  0 = no tightening, 1 = full tighten."""
    if wall is None or atr <= 0:
        return 0.0
    if is_long:
        dist = wall - price
    else:
        dist = price - wall
    if dist <= 0:
        return 0.0
    dist_atr = dist / atr
    if dist_atr >= WALL_TIGHTEN_ATR:
        return 0.0
    return 1.0 - (dist_atr / WALL_TIGHTEN_ATR)


def compute_trailing_stops(
    pos: dict,
    df: pd.DataFrame,
    options_ctx: dict | None = None,
) -> dict:
    """Compute all three trailing stop levels for an open position.

    When options_ctx is provided, gamma regime scales the ATR multipliers
    and wall proximity blends in tighter trailing behavior.
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

    g_mult = _gamma_trail_mult(options_ctx)
    eff_atr_mult = ATR_TRAIL_MULT * g_mult
    eff_hybrid_mult = HYBRID_TRAIL_MULT * g_mult

    wall = None
    if options_ctx and options_ctx.get("options_context_available"):
        wall = options_ctx.get("nearest_call_wall") if is_long else options_ctx.get("nearest_put_wall")
        if wall is not None:
            if is_long and wall <= entry:
                wall = None
            elif not is_long and wall >= entry:
                wall = None

    blend = _wall_proximity_blend(close, wall, atr, is_long)
    if blend > 0:
        eff_atr_mult = eff_atr_mult * (1.0 - blend) + eff_hybrid_mult * blend

    # ATR Chandelier
    if is_long:
        trail_atr = best - eff_atr_mult * atr
    else:
        trail_atr = best + eff_atr_mult * atr

    # EMA Trail
    trail_ema = ema20

    # Hybrid (breakeven + tighter ATR)
    if unrealized_r < 1.0:
        trail_hybrid = initial_stop
    elif unrealized_r < 2.0:
        trail_hybrid = entry
    else:
        if is_long:
            trail_hybrid = best - eff_hybrid_mult * atr
        else:
            trail_hybrid = best + eff_hybrid_mult * atr

    # Time-based tightening: weak P&L late in hold -> force breakeven
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

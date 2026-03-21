"""Conviction scoring for continuation setups."""

from __future__ import annotations

import pandas as pd

from app.rules.continuation_rules import (
    detect_trend,
    find_support_resistance,
    has_room_to_target_long,
    has_room_to_target_short,
    is_bounce_and_fail_long,
    is_bounce_and_fail_short,
    is_flag_breakout,
    is_flag_breakdown,
    is_pullback_to_support,
    is_pullback_to_resistance,
    is_structural_breakout,
    is_structural_breakdown,
    bullish_strong_close,
    bullish_rejection_wick,
    bearish_strong_close,
    bearish_rejection_wick,
    is_healthy_pullback_volume,
    has_volume_confirmation,
)

MAX_DISTANCE_FROM_EMA20_ATR = 2.0
BREAKOUT_MAX_DISTANCE_ATR = 4.0

# Options context thresholds
CALL_WALL_CLOSE_PCT = 2.0   # penalize longs if call wall within this %
CALL_WALL_CLEAR_PCT = 5.0   # boost longs if call wall farther than this %
PUT_WALL_CLOSE_PCT = 2.0    # penalize shorts if put wall within this %
PUT_WALL_CLEAR_PCT = 5.0    # boost shorts if put wall farther than this %


def apply_options_context_adjustment(
    score: int,
    direction: str,
    ctx: dict,
    spot: float | None = None,
) -> tuple[int, list[str]]:
    """Apply up to +2 / -2 score points based on options gamma/OI context.

    Returns (adjusted_score, list_of_reasons).  Skips entirely when the
    context is unavailable or all-None, returning the score unchanged.
    """
    if not ctx.get("options_context_available"):
        return score, []

    adj = 0
    reasons: list[str] = []
    regime = ctx.get("gamma_regime", "NEUTRAL")
    flip = ctx.get("gamma_flip_level_estimate")
    dist_call = ctx.get("distance_to_call_wall_pct")
    dist_put = ctx.get("distance_to_put_wall_pct")
    near_oi = ctx.get("near_term_oi") or 0
    swing_oi = ctx.get("swing_dte_oi") or 0

    if direction == "LONG":
        # Boost: negative gamma or above flip -> dealer hedging amplifies moves up
        if regime == "NEGATIVE" or (flip is not None and spot is not None and spot > flip):
            adj += 1
            reasons.append("gamma_tailwind")

        # Boost: call wall far above -> room to run
        if dist_call is not None and dist_call > CALL_WALL_CLEAR_PCT:
            adj += 1
            reasons.append("call_wall_clear")

        # Penalize: positive gamma and chasing (dealer hedging dampens moves)
        if regime == "POSITIVE" and (flip is not None and spot is not None and spot > flip):
            adj -= 1
            reasons.append("gamma_headwind")

        # Penalize: call wall very close overhead
        if dist_call is not None and dist_call < CALL_WALL_CLOSE_PCT:
            adj -= 1
            reasons.append("call_wall_near")

        # DTE structure: swing > near-term is positive conviction
        if swing_oi > near_oi and near_oi > 0:
            adj += 1
            reasons.append("swing_dte_dominant")
        elif near_oi > swing_oi and swing_oi > 0:
            adj -= 1
            reasons.append("short_dated_noise")

    else:  # SHORT
        # Boost: negative gamma -> downside can extend
        if regime == "NEGATIVE":
            adj += 1
            reasons.append("gamma_tailwind")

        # Boost: put wall far below -> room to fall
        if dist_put is not None and dist_put > PUT_WALL_CLEAR_PCT:
            adj += 1
            reasons.append("put_wall_clear")

        # Penalize: positive gamma -> price likely to pin/mean-revert
        if regime == "POSITIVE":
            adj -= 1
            reasons.append("gamma_headwind")

        # Penalize: strong put wall just below price
        if dist_put is not None and dist_put < PUT_WALL_CLOSE_PCT:
            adj -= 1
            reasons.append("put_wall_near")

        # DTE structure
        if swing_oi > near_oi and near_oi > 0:
            adj += 1
            reasons.append("swing_dte_dominant")
        elif near_oi > swing_oi and swing_oi > 0:
            adj -= 1
            reasons.append("short_dated_noise")

    adj = max(-2, min(2, adj))
    adjusted = max(0, min(10, score + adj))
    return adjusted, reasons


def is_not_extended(df: pd.DataFrame, max_distance_atr: float = MAX_DISTANCE_FROM_EMA20_ATR) -> bool:
    """
    Reject setups that are too far from the 20 EMA in ATR units.
    """
    if df.empty:
        return False

    last = df.iloc[-1]
    atr = last.get("atr14")
    ema20 = last.get("ema20")
    close = last.get("close")

    if pd.isna(atr) or pd.isna(ema20) or pd.isna(close) or atr <= 0:
        return False

    distance = abs(close - ema20) / atr
    return bool(distance <= max_distance_atr)


def _build_result(
    *,
    direction: str,
    score: int,
    support: float,
    resistance: float,
    structural_support: float,
    structural_resistance: float,
    state: str,
    reasons: list[str],
    checks_passed: list[str],
    checks_failed: list[str],
) -> dict:
    return {
        "direction": direction,
        "score": score,
        "max_score": 10,
        "is_valid": state == "SIGNAL",
        "state": state,
        "reasons": reasons,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "support": support,
        "resistance": resistance,
        "structural_support": structural_support,
        "structural_resistance": structural_resistance,
    }


def score_long_setup(df: pd.DataFrame, options_ctx: dict | None = None) -> dict:
    """Score a long continuation setup on a 0-10 scale with SIGNAL/WATCHLIST/REJECT state."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    trend_ok = trend["trend"] == "LONG"
    not_extended_ok = is_not_extended(df)
    room_ok = has_room_to_target_long(df, levels["structural_resistance"])

    bounce_fail = is_bounce_and_fail_long(df)
    flag = is_flag_breakout(df)
    pullback_to_level = is_pullback_to_support(df, levels["support"])
    structural_bo = is_structural_breakout(df, levels["structural_resistance"])
    continuation_ok = bounce_fail or flag or pullback_to_level or structural_bo

    if structural_bo and not not_extended_ok:
        not_extended_ok = is_not_extended(df, max_distance_atr=BREAKOUT_MAX_DISTANCE_ATR)

    strong_close = bullish_strong_close(df)
    rejection_wick = bullish_rejection_wick(df)
    candle_ok = strong_close or rejection_wick
    pullback_volume_ok = is_healthy_pullback_volume(df)
    confirmation_volume_ok = has_volume_confirmation(df)

    score = 0
    reasons: list[str] = []
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    if trend_ok:
        score += 3
        reasons.append("trend_aligned")
        checks_passed.append("trend_aligned")
    else:
        checks_failed.append("trend_aligned")

    if not_extended_ok:
        score += 1
        reasons.append("not_extended")
        checks_passed.append("not_extended")
    else:
        checks_failed.append("not_extended")

    if room_ok or structural_bo:
        score += 1
        reasons.append("room_to_target")
        checks_passed.append("room_to_target")
    else:
        checks_failed.append("room_to_target")

    if continuation_ok:
        score += 2
        if structural_bo:
            label = "structural_breakout"
        elif bounce_fail:
            label = "bounce_and_fail"
        elif flag:
            label = "flag_breakout"
        else:
            label = "pullback_to_support"
        reasons.append(label)
        checks_passed.append(label)
    else:
        checks_failed.append("continuation_pattern")

    candle_points = 0
    if strong_close:
        candle_points += 1
        reasons.append("bullish_strong_close")
        checks_passed.append("bullish_strong_close")
    else:
        checks_failed.append("bullish_strong_close")

    if rejection_wick:
        candle_points += 1
        reasons.append("bullish_rejection_wick")
        checks_passed.append("bullish_rejection_wick")
    else:
        checks_failed.append("bullish_rejection_wick")

    score += min(candle_points, 1)

    if pullback_volume_ok:
        score += 1
        reasons.append("healthy_pullback_volume")
        checks_passed.append("healthy_pullback_volume")
    else:
        checks_failed.append("healthy_pullback_volume")

    if confirmation_volume_ok:
        score += 1
        reasons.append("confirmation_volume")
        checks_passed.append("confirmation_volume")
    else:
        checks_failed.append("confirmation_volume")

    if options_ctx:
        spot = float(df.iloc[-1]["close"])
        score, gamma_reasons = apply_options_context_adjustment(score, "LONG", options_ctx, spot)
        reasons.extend(gamma_reasons)
        checks_passed.extend(gamma_reasons)

    soft_count = sum([
        room_ok or structural_bo,
        continuation_ok,
        candle_ok,
        pullback_volume_ok,
        confirmation_volume_ok,
    ])

    if not trend_ok or not not_extended_ok:
        state = "REJECT"
    elif structural_bo:
        state = "SIGNAL" if soft_count >= 3 and score >= 7 else "WATCHLIST"
    elif not room_ok:
        state = "WATCHLIST" if soft_count >= 2 and score >= 5 else "REJECT"
    elif soft_count >= 3 and score >= 7:
        state = "SIGNAL"
    elif soft_count >= 2 and score >= 5:
        state = "WATCHLIST"
    else:
        state = "REJECT"

    return _build_result(
        direction="LONG",
        score=score,
        support=levels["support"],
        resistance=levels["resistance"],
        structural_support=levels["structural_support"],
        structural_resistance=levels["structural_resistance"],
        state=state,
        reasons=reasons,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )


def score_short_setup(df: pd.DataFrame, options_ctx: dict | None = None) -> dict:
    """Score a short continuation setup on a 0-10 scale with SIGNAL/WATCHLIST/REJECT state."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    trend_ok = trend["trend"] == "SHORT"
    not_extended_ok = is_not_extended(df)
    room_ok = has_room_to_target_short(df, levels["structural_support"])

    bounce_fail = is_bounce_and_fail_short(df)
    flag = is_flag_breakdown(df)
    pullback_to_level = is_pullback_to_resistance(df, levels["resistance"])
    structural_bd = is_structural_breakdown(df, levels["structural_support"])
    continuation_ok = bounce_fail or flag or pullback_to_level or structural_bd

    if structural_bd and not not_extended_ok:
        not_extended_ok = is_not_extended(df, max_distance_atr=BREAKOUT_MAX_DISTANCE_ATR)

    strong_close = bearish_strong_close(df)
    rejection_wick = bearish_rejection_wick(df)
    candle_ok = strong_close or rejection_wick
    pullback_volume_ok = is_healthy_pullback_volume(df)
    confirmation_volume_ok = has_volume_confirmation(df)

    score = 0
    reasons: list[str] = []
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    if trend_ok:
        score += 3
        reasons.append("trend_aligned")
        checks_passed.append("trend_aligned")
    else:
        checks_failed.append("trend_aligned")

    if not_extended_ok:
        score += 1
        reasons.append("not_extended")
        checks_passed.append("not_extended")
    else:
        checks_failed.append("not_extended")

    if room_ok or structural_bd:
        score += 1
        reasons.append("room_to_target")
        checks_passed.append("room_to_target")
    else:
        checks_failed.append("room_to_target")

    if continuation_ok:
        score += 2
        if structural_bd:
            label = "structural_breakdown"
        elif bounce_fail:
            label = "bounce_and_fail"
        elif flag:
            label = "flag_breakdown"
        else:
            label = "pullback_to_resistance"
        reasons.append(label)
        checks_passed.append(label)
    else:
        checks_failed.append("continuation_pattern")

    candle_points = 0
    if strong_close:
        candle_points += 1
        reasons.append("bearish_strong_close")
        checks_passed.append("bearish_strong_close")
    else:
        checks_failed.append("bearish_strong_close")

    if rejection_wick:
        candle_points += 1
        reasons.append("bearish_rejection_wick")
        checks_passed.append("bearish_rejection_wick")
    else:
        checks_failed.append("bearish_rejection_wick")

    score += min(candle_points, 1)

    if pullback_volume_ok:
        score += 1
        reasons.append("healthy_pullback_volume")
        checks_passed.append("healthy_pullback_volume")
    else:
        checks_failed.append("healthy_pullback_volume")

    if confirmation_volume_ok:
        score += 1
        reasons.append("confirmation_volume")
        checks_passed.append("confirmation_volume")
    else:
        checks_failed.append("confirmation_volume")

    if options_ctx:
        spot = float(df.iloc[-1]["close"])
        score, gamma_reasons = apply_options_context_adjustment(score, "SHORT", options_ctx, spot)
        reasons.extend(gamma_reasons)
        checks_passed.extend(gamma_reasons)

    soft_count = sum([
        room_ok or structural_bd,
        continuation_ok,
        candle_ok,
        pullback_volume_ok,
        confirmation_volume_ok,
    ])

    if not trend_ok or not not_extended_ok:
        state = "REJECT"
    elif structural_bd:
        state = "SIGNAL" if soft_count >= 3 and score >= 7 else "WATCHLIST"
    elif not room_ok:
        state = "WATCHLIST" if soft_count >= 2 and score >= 5 else "REJECT"
    elif soft_count >= 3 and score >= 7:
        state = "SIGNAL"
    elif soft_count >= 2 and score >= 5:
        state = "WATCHLIST"
    else:
        state = "REJECT"

    return _build_result(
        direction="SHORT",
        score=score,
        support=levels["support"],
        resistance=levels["resistance"],
        structural_support=levels["structural_support"],
        structural_resistance=levels["structural_resistance"],
        state=state,
        reasons=reasons,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )
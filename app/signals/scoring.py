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
    bullish_strong_close,
    bullish_rejection_wick,
    bearish_strong_close,
    bearish_rejection_wick,
    is_healthy_pullback_volume,
    has_volume_confirmation,
)

MAX_DISTANCE_FROM_EMA20_ATR = 2.0


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
    }


def score_long_setup(df: pd.DataFrame) -> dict:
    """Score a long continuation setup on a 0-10 scale with SIGNAL/WATCHLIST/REJECT state."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    trend_ok = trend["trend"] == "LONG"
    not_extended_ok = is_not_extended(df)
    room_ok = has_room_to_target_long(df, levels["resistance"])

    bounce_fail = is_bounce_and_fail_long(df)
    flag = is_flag_breakout(df)
    pullback_to_level = is_pullback_to_support(df, levels["support"])
    continuation_ok = bounce_fail or flag or pullback_to_level

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

    if room_ok:
        score += 1
        reasons.append("room_to_target")
        checks_passed.append("room_to_target")
    else:
        checks_failed.append("room_to_target")

    if continuation_ok:
        score += 2
        if bounce_fail:
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

    soft_count = sum([
        room_ok,
        continuation_ok,
        candle_ok,
        pullback_volume_ok,
        confirmation_volume_ok,
    ])

    if not trend_ok or not not_extended_ok:
        state = "REJECT"
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
        state=state,
        reasons=reasons,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )


def score_short_setup(df: pd.DataFrame) -> dict:
    """Score a short continuation setup on a 0-10 scale with SIGNAL/WATCHLIST/REJECT state."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    trend_ok = trend["trend"] == "SHORT"
    not_extended_ok = is_not_extended(df)
    room_ok = has_room_to_target_short(df, levels["support"])

    bounce_fail = is_bounce_and_fail_short(df)
    flag = is_flag_breakdown(df)
    pullback_to_level = is_pullback_to_resistance(df, levels["resistance"])
    continuation_ok = bounce_fail or flag or pullback_to_level

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

    if room_ok:
        score += 1
        reasons.append("room_to_target")
        checks_passed.append("room_to_target")
    else:
        checks_failed.append("room_to_target")

    if continuation_ok:
        score += 2
        if bounce_fail:
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

    soft_count = sum([
        room_ok,
        continuation_ok,
        candle_ok,
        pullback_volume_ok,
        confirmation_volume_ok,
    ])

    if not trend_ok or not not_extended_ok:
        state = "REJECT"
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
        state=state,
        reasons=reasons,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )
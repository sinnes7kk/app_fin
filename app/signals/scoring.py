"""Conviction scoring for continuation setups."""

from __future__ import annotations

import pandas as pd

from app.rules.continuation_rules import (
    detect_trend,
    find_support_resistance,
    is_pullback_to_support,
    is_pullback_to_resistance,
    bullish_strong_close,
    bullish_rejection_wick,
    bearish_strong_close,
    bearish_rejection_wick,
    is_healthy_pullback_volume,
    has_volume_confirmation,
)


def score_long_setup(df: pd.DataFrame) -> dict:
    """Score a long continuation setup on a 0-10 scale."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    pullback_ok = is_pullback_to_support(df, levels["support"])
    strong_close = bullish_strong_close(df)
    rejection_wick = bullish_rejection_wick(df)
    pullback_volume_ok = is_healthy_pullback_volume(df)
    confirmation_volume_ok = has_volume_confirmation(df)

    score = 0
    reasons: list[str] = []

    # Trend quality
    if trend["trend"] == "LONG":
        score += 2
        reasons.append("trend_aligned")

    # Pullback location
    if pullback_ok:
        score += 2
        reasons.append("pullback_to_support")

    # Candle confirmation
    candle_points = 0
    if strong_close:
        candle_points += 1
        reasons.append("bullish_strong_close")
    if rejection_wick:
        candle_points += 1
        reasons.append("bullish_rejection_wick")
    score += min(candle_points, 2)

    # Pullback volume quality
    if pullback_volume_ok:
        score += 2
        reasons.append("healthy_pullback_volume")

    # Confirmation volume
    if confirmation_volume_ok:
        score += 2
        reasons.append("confirmation_volume")

    candle_ok = strong_close or rejection_wick
    is_valid = (
        trend["trend"] == "LONG"
        and pullback_ok
        and candle_ok
        and pullback_volume_ok
        and confirmation_volume_ok
    )

    return {
        "direction": "LONG",
        "score": score,
        "max_score": 10,
        "is_valid": is_valid,
        "reasons": reasons,
        "support": levels["support"],
        "resistance": levels["resistance"],
    }


def score_short_setup(df: pd.DataFrame) -> dict:
    """Score a short continuation setup on a 0-10 scale."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    pullback_ok = is_pullback_to_resistance(df, levels["resistance"])
    strong_close = bearish_strong_close(df)
    rejection_wick = bearish_rejection_wick(df)
    pullback_volume_ok = is_healthy_pullback_volume(df)
    confirmation_volume_ok = has_volume_confirmation(df)

    score = 0
    reasons: list[str] = []

    # Trend quality
    if trend["trend"] == "SHORT":
        score += 2
        reasons.append("trend_aligned")

    # Pullback location
    if pullback_ok:
        score += 2
        reasons.append("pullback_to_resistance")

    # Candle confirmation
    candle_points = 0
    if strong_close:
        candle_points += 1
        reasons.append("bearish_strong_close")
    if rejection_wick:
        candle_points += 1
        reasons.append("bearish_rejection_wick")
    score += min(candle_points, 2)

    # Pullback volume quality
    if pullback_volume_ok:
        score += 2
        reasons.append("healthy_pullback_volume")

    # Confirmation volume
    if confirmation_volume_ok:
        score += 2
        reasons.append("confirmation_volume")

    candle_ok = strong_close or rejection_wick
    is_valid = (
        trend["trend"] == "SHORT"
        and pullback_ok
        and candle_ok
        and pullback_volume_ok
        and confirmation_volume_ok
    )

    return {
        "direction": "SHORT",
        "score": score,
        "max_score": 10,
        "is_valid": is_valid,
        "reasons": reasons,
        "support": levels["support"],
        "resistance": levels["resistance"],
    }
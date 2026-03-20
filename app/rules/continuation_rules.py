"""Rule engine for daily swing continuation setups."""

from __future__ import annotations

import pandas as pd


def detect_trend(df: pd.DataFrame) -> dict:
    """Determine simple daily trend state from price and moving averages."""
    if len(df) < 2:
        return {"trend": "NEUTRAL", "is_valid": False}

    last = df.iloc[-1]

    if last["close"] > last["ema20"] and last["ema20"] > last["ema50"]:
        return {"trend": "LONG", "is_valid": True}

    if last["close"] < last["ema20"] and last["ema20"] < last["ema50"]:
        return {"trend": "SHORT", "is_valid": True}

    return {"trend": "NEUTRAL", "is_valid": False}


def find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Find simple rolling support and resistance levels.

    For V1, this uses prior bars only so the current candle does not define
    today's support/resistance level too heavily.
    """
    if len(df) < lookback + 1:
        raise ValueError(
            f"Need at least {lookback + 1} rows to compute support/resistance"
        )

    recent = df.iloc[-(lookback + 1) : -1]
    support = recent["low"].min()
    resistance = recent["high"].max()

    return {
        "support": float(support),
        "resistance": float(resistance),
    }


def is_pullback_to_support(
    df: pd.DataFrame,
    support_level: float,
    atr_multiple: float = 0.25,
) -> bool:
    """
    Check whether the latest candle pulled back into the support zone and held it.
    """
    if df.empty:
        return False

    last = df.iloc[-1]
    atr = last["atr14"]

    if pd.isna(atr) or atr <= 0:
        return False

    zone_low = support_level - atr_multiple * atr
    zone_high = support_level + atr_multiple * atr

    touched_zone = last["low"] <= zone_high
    closed_above_zone_low = last["close"] > zone_low

    return bool(touched_zone and closed_above_zone_low)


def bullish_strong_close(df: pd.DataFrame, threshold: float = 0.75) -> bool:
    """
    Check whether the latest candle closes strongly near its high.
    Threshold 0.75 means the close is in the top 25% of the candle range.
    """
    if df.empty:
        return False

    last = df.iloc[-1]
    candle_range = last["high"] - last["low"]

    if candle_range <= 0:
        return False

    if last["close"] <= last["open"]:
        return False

    close_position = (last["close"] - last["low"]) / candle_range
    return bool(close_position >= threshold)


def bullish_rejection_wick(df: pd.DataFrame, wick_body_ratio: float = 1.5) -> bool:
    """Check whether the latest candle has a bullish lower rejection wick."""
    if df.empty:
        return False

    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    candle_range = last["high"] - last["low"]

    if candle_range <= 0:
        return False

    if body == 0:
        body = 1e-9

    close_above_mid = last["close"] > (last["low"] + candle_range * 0.5)

    return bool(
        lower_wick >= wick_body_ratio * body
        and lower_wick > upper_wick
        and close_above_mid
    )


def is_healthy_pullback_volume(
    df: pd.DataFrame,
    lookback: int = 3,
    max_avg_rel_volume: float = 1.1,
) -> bool:
    """
    Check that recent pullback volume is relatively light.

    Uses the bars before the latest bar, because the latest bar is treated as
    the possible confirmation/signal bar.
    """
    if len(df) < lookback + 1:
        return False

    if "rel_volume" not in df.columns:
        raise ValueError("Expected 'rel_volume' column in DataFrame")

    pullback_bars = df.iloc[-(lookback + 1) : -1]

    if pullback_bars["rel_volume"].isna().all():
        return False

    avg_rel_volume = pullback_bars["rel_volume"].mean()
    return bool(avg_rel_volume <= max_avg_rel_volume)


def has_volume_confirmation(
    df: pd.DataFrame,
    min_rel_volume: float = 1.2,
) -> bool:
    """Check whether the latest bar has stronger-than-normal volume."""
    if df.empty:
        return False

    if "rel_volume" not in df.columns:
        raise ValueError("Expected 'rel_volume' column in DataFrame")

    last = df.iloc[-1]
    rel_volume = last["rel_volume"]

    if pd.isna(rel_volume):
        return False

    return bool(rel_volume >= min_rel_volume)


def evaluate_long_setup(df: pd.DataFrame) -> dict:
    """
    Evaluate a simple long swing continuation setup.

    Conditions:
    - daily uptrend
    - pullback to support
    - healthy pullback volume
    - bullish confirmation candle
    - confirmation bar volume expansion
    """
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    pullback_ok = is_pullback_to_support(df, levels["support"])
    pullback_volume_ok = is_healthy_pullback_volume(df)

    candle_ok = bullish_strong_close(df) or bullish_rejection_wick(df)
    confirmation_volume_ok = has_volume_confirmation(df)

    is_valid = (
        trend["trend"] == "LONG"
        and pullback_ok
        and pullback_volume_ok
        and candle_ok
        and confirmation_volume_ok
    )

    return {
        "is_valid": is_valid,
        "trend": trend["trend"],
        "support": levels["support"],
        "resistance": levels["resistance"],
        "pullback_ok": pullback_ok,
        "pullback_volume_ok": pullback_volume_ok,
        "candle_ok": candle_ok,
        "confirmation_volume_ok": confirmation_volume_ok,
    }


def is_pullback_to_resistance(
    df: pd.DataFrame,
    resistance_level: float,
    atr_multiple: float = 0.25,
) -> bool:
    """
    Check whether the latest candle pulled back into the resistance zone and failed there.
    """
    if df.empty:
        return False

    last = df.iloc[-1]
    atr = last["atr14"]

    if pd.isna(atr) or atr <= 0:
        return False

    zone_low = resistance_level - atr_multiple * atr
    zone_high = resistance_level + atr_multiple * atr

    touched_zone = last["high"] >= zone_low
    closed_below_zone_high = last["close"] < zone_high

    return bool(touched_zone and closed_below_zone_high)


def bearish_strong_close(df: pd.DataFrame, threshold: float = 0.25) -> bool:
    """
    Check whether the latest candle closes strongly near its low.
    Threshold 0.25 means the close is in the bottom 25% of the candle range.
    """
    if df.empty:
        return False

    last = df.iloc[-1]
    candle_range = last["high"] - last["low"]

    if candle_range <= 0:
        return False

    if last["close"] >= last["open"]:
        return False

    close_position = (last["close"] - last["low"]) / candle_range
    return bool(close_position <= threshold)


def bearish_rejection_wick(df: pd.DataFrame, wick_body_ratio: float = 1.5) -> bool:
    """
    Check whether the latest candle has a bearish upper rejection wick.
    """
    if df.empty:
        return False

    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["open"], last["close"])
    lower_wick = min(last["open"], last["close"]) - last["low"]
    candle_range = last["high"] - last["low"]

    if candle_range <= 0:
        return False

    if body == 0:
        body = 1e-9

    close_below_mid = last["close"] < (last["low"] + candle_range * 0.5)

    return bool(
        upper_wick >= wick_body_ratio * body
        and upper_wick > lower_wick
        and close_below_mid
    )


def evaluate_short_setup(df: pd.DataFrame) -> dict:
    """
    Evaluate a simple short swing continuation setup.

    Conditions:
    - daily downtrend
    - pullback to resistance
    - healthy pullback volume
    - bearish confirmation candle
    - confirmation bar volume expansion
    """
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    pullback_ok = is_pullback_to_resistance(df, levels["resistance"])
    pullback_volume_ok = is_healthy_pullback_volume(df)

    candle_ok = bearish_strong_close(df) or bearish_rejection_wick(df)
    confirmation_volume_ok = has_volume_confirmation(df)

    is_valid = (
        trend["trend"] == "SHORT"
        and pullback_ok
        and pullback_volume_ok
        and candle_ok
        and confirmation_volume_ok
    )

    return {
        "is_valid": is_valid,
        "trend": trend["trend"],
        "support": levels["support"],
        "resistance": levels["resistance"],
        "pullback_ok": pullback_ok,
        "pullback_volume_ok": pullback_volume_ok,
        "candle_ok": candle_ok,
        "confirmation_volume_ok": confirmation_volume_ok,
    }
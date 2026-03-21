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


STRUCTURAL_LOOKBACK = 60


def find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Find tactical (20-bar) and structural (60-bar) S/R levels.

    Tactical levels are used for pullback detection and entry timing.
    Structural levels capture major swing highs/lows that institutional
    traders watch, used for room-to-target checks and trade plan targets.
    Both exclude the current bar to avoid self-referencing.
    """
    if len(df) < lookback + 1:
        raise ValueError(
            f"Need at least {lookback + 1} rows to compute support/resistance"
        )

    recent = df.iloc[-(lookback + 1) : -1]
    support = recent["low"].min()
    resistance = recent["high"].max()

    struct_bars = min(STRUCTURAL_LOOKBACK, len(df) - 1)
    structural = df.iloc[-(struct_bars + 1) : -1]
    structural_support = structural["low"].min()
    structural_resistance = structural["high"].max()

    return {
        "support": float(support),
        "resistance": float(resistance),
        "structural_support": float(structural_support),
        "structural_resistance": float(structural_resistance),
    }


def is_pullback_to_ema(
    df: pd.DataFrame,
    atr_multiple: float = 1.0,
    lookback: int = 5,
) -> bool:
    """Check whether any of the last ``lookback`` bars pulled back near the
    EMA20 or EMA50.

    Each bar is evaluated against its own EMA values (not today's), since the
    EMA shifts daily.  Passes if any bar's low came within
    ``atr_multiple * ATR`` of either EMA while closing above it.
    """
    if len(df) < lookback:
        return False

    window = df.iloc[-lookback:]

    for _, bar in window.iterrows():
        atr = bar.get("atr14")
        ema20 = bar.get("ema20")
        ema50 = bar.get("ema50")
        low = bar.get("low")
        close = bar.get("close")

        if any(pd.isna(v) for v in (atr, ema20, ema50, low, close)) or atr <= 0:
            continue

        band = atr_multiple * atr
        for ema in (ema20, ema50):
            if low <= ema + band and close >= ema - band:
                return True

    return False


def is_pullback_to_support(
    df: pd.DataFrame,
    support_level: float,
    atr_multiple: float = 0.25,
    lookback: int = 5,
) -> bool:
    """Check whether any of the last ``lookback`` bars pulled back into the
    support zone and held it.
    """
    if len(df) < lookback:
        return False

    window = df.iloc[-lookback:]

    for _, bar in window.iterrows():
        atr = bar["atr14"]
        if pd.isna(atr) or atr <= 0:
            continue

        zone_low = support_level - atr_multiple * atr
        zone_high = support_level + atr_multiple * atr

        if bar["low"] <= zone_high and bar["close"] > zone_low:
            return True

    return False


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
    min_rel_volume: float = 1.0,
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


def is_bounce_and_fail_long(
    df: pd.DataFrame,
    atr_multiple: float = 1.0,
    lookback: int = 5,
    min_red_bars: int = 1,
) -> bool:
    """Detect a dip-and-recovery pattern for long continuation.

    Requires:
    1. At least ``min_red_bars`` red bars in the lookback window (evidence of
       a pullback dip).
    2. One of those red bars reached within ``atr_multiple * ATR`` of EMA20 or
       EMA50 (dip touched dynamic support).
    3. The final bar is green (close > open), confirming the dip was bought.
    """
    if len(df) < lookback + 1:
        return False

    window = df.iloc[-(lookback + 1):-1]
    last = df.iloc[-1]

    if last["close"] <= last["open"]:
        return False

    red_bars_near_ema = 0
    for _, bar in window.iterrows():
        if bar["close"] >= bar["open"]:
            continue
        atr = bar.get("atr14")
        ema20 = bar.get("ema20")
        ema50 = bar.get("ema50")
        low = bar.get("low")
        if any(pd.isna(v) for v in (atr, ema20, ema50, low)) or atr <= 0:
            continue
        band = atr_multiple * atr
        for ema in (ema20, ema50):
            if low <= ema + band:
                red_bars_near_ema += 1
                break

    return red_bars_near_ema >= min_red_bars


def is_bounce_and_fail_short(
    df: pd.DataFrame,
    atr_multiple: float = 1.0,
    lookback: int = 5,
    min_green_bars: int = 1,
) -> bool:
    """Detect a rally-and-fail pattern for short continuation.

    Requires:
    1. At least ``min_green_bars`` green bars in the lookback window (evidence
       of a counter-trend rally).
    2. One of those green bars reached within ``atr_multiple * ATR`` of EMA20
       or EMA50 (rally touched dynamic resistance).
    3. The final bar is red (close < open), confirming the rally failed.
    """
    if len(df) < lookback + 1:
        return False

    window = df.iloc[-(lookback + 1):-1]
    last = df.iloc[-1]

    if last["close"] >= last["open"]:
        return False

    green_bars_near_ema = 0
    for _, bar in window.iterrows():
        if bar["close"] <= bar["open"]:
            continue
        atr = bar.get("atr14")
        ema20 = bar.get("ema20")
        ema50 = bar.get("ema50")
        high = bar.get("high")
        if any(pd.isna(v) for v in (atr, ema20, ema50, high)) or atr <= 0:
            continue
        band = atr_multiple * atr
        for ema in (ema20, ema50):
            if high >= ema - band:
                green_bars_near_ema += 1
                break

    return green_bars_near_ema >= min_green_bars


def is_flag_breakout(
    df: pd.DataFrame,
    flag_bars: int = 4,
    max_range_atr: float = 1.5,
) -> bool:
    """Detect a bullish flag breakout — tight consolidation followed by an
    upside break with volume.

    1. The ``flag_bars`` bars before the current bar have a narrow total range
       (high-low spread < ``max_range_atr * ATR``).
    2. Today's bar breaks above the flag high.
    3. Today's volume exceeds the 20-day average (rel_volume >= 1.0).
    """
    needed = flag_bars + 1
    if len(df) < needed:
        return False

    last = df.iloc[-1]
    flag = df.iloc[-needed:-1]

    atr = last.get("atr14")
    rel_vol = last.get("rel_volume")
    if pd.isna(atr) or atr <= 0 or pd.isna(rel_vol):
        return False

    flag_high = flag["high"].max()
    flag_low = flag["low"].min()
    flag_range = flag_high - flag_low

    if flag_range > max_range_atr * atr:
        return False

    return bool(last["close"] > flag_high and rel_vol >= 1.0)


def is_flag_breakdown(
    df: pd.DataFrame,
    flag_bars: int = 4,
    max_range_atr: float = 1.5,
) -> bool:
    """Detect a bearish flag breakdown — tight consolidation followed by a
    downside break with volume.

    1. The ``flag_bars`` bars before the current bar have a narrow total range
       (high-low spread < ``max_range_atr * ATR``).
    2. Today's bar breaks below the flag low.
    3. Today's volume exceeds the 20-day average (rel_volume >= 1.0).
    """
    needed = flag_bars + 1
    if len(df) < needed:
        return False

    last = df.iloc[-1]
    flag = df.iloc[-needed:-1]

    atr = last.get("atr14")
    rel_vol = last.get("rel_volume")
    if pd.isna(atr) or atr <= 0 or pd.isna(rel_vol):
        return False

    flag_high = flag["high"].max()
    flag_low = flag["low"].min()
    flag_range = flag_high - flag_low

    if flag_range > max_range_atr * atr:
        return False

    return bool(last["close"] < flag_low and rel_vol >= 1.0)


def is_structural_breakout(
    df: pd.DataFrame,
    structural_resistance: float,
    lookback: int = 2,
) -> bool:
    """Detect a confirmed close above 60-bar structural resistance with volume.

    A breakout is valid when any of the last ``lookback`` bars closed above
    the structural resistance level on above-average volume.
    """
    if len(df) < lookback:
        return False

    window = df.iloc[-lookback:]
    for _, bar in window.iterrows():
        rel_vol = bar.get("rel_volume")
        if pd.isna(rel_vol):
            continue
        if bar["close"] > structural_resistance and rel_vol >= 1.0:
            return True
    return False


def is_structural_breakdown(
    df: pd.DataFrame,
    structural_support: float,
    lookback: int = 2,
) -> bool:
    """Detect a confirmed close below 60-bar structural support with volume.

    A breakdown is valid when any of the last ``lookback`` bars closed below
    the structural support level on above-average volume.
    """
    if len(df) < lookback:
        return False

    window = df.iloc[-lookback:]
    for _, bar in window.iterrows():
        rel_vol = bar.get("rel_volume")
        if pd.isna(rel_vol):
            continue
        if bar["close"] < structural_support and rel_vol >= 1.0:
            return True
    return False


MIN_ROOM_ATR = 1.0


def has_room_to_target_long(df: pd.DataFrame, resistance: float) -> bool:
    """Reject longs that are already within MIN_ROOM_ATR of structural resistance."""
    if df.empty:
        return False
    last = df.iloc[-1]
    atr = last.get("atr14")
    close = last.get("close")
    if pd.isna(atr) or pd.isna(close) or atr <= 0:
        return False
    return bool((resistance - close) / atr >= MIN_ROOM_ATR)


def has_room_to_target_short(df: pd.DataFrame, support: float) -> bool:
    """Reject shorts that are already within MIN_ROOM_ATR of structural support."""
    if df.empty:
        return False
    last = df.iloc[-1]
    atr = last.get("atr14")
    close = last.get("close")
    if pd.isna(atr) or pd.isna(close) or atr <= 0:
        return False
    return bool((close - support) / atr >= MIN_ROOM_ATR)


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
    lookback: int = 5,
) -> bool:
    """Check whether any of the last ``lookback`` bars pulled back into the
    resistance zone and failed there.
    """
    if len(df) < lookback:
        return False

    window = df.iloc[-lookback:]

    for _, bar in window.iterrows():
        atr = bar["atr14"]
        if pd.isna(atr) or atr <= 0:
            continue

        zone_low = resistance_level - atr_multiple * atr
        zone_high = resistance_level + atr_multiple * atr

        if bar["high"] >= zone_low and bar["close"] < zone_high:
            return True

    return False


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
"""Rule engine for daily swing continuation setups."""

from __future__ import annotations

import pandas as pd


def detect_trend(df: pd.DataFrame) -> dict:
    """Determine daily trend state and strength from price, MAs, and ADX.

    Returns a trend direction (LONG/SHORT/NEUTRAL) plus a ``strength`` value
    between 0.0 and 1.0 derived from ADX and EMA slope.  A strength of 0.0
    means no trend; 1.0 means a very strong trend.

    Primary path: EMA20 > EMA50 (classic golden/death cross alignment).
    Alternative path: close beyond EMA50 with a rising/falling EMA50 slope,
    capturing recovering trends where the 20/50 crossover hasn't completed yet.
    Alternative-path strength is capped at 70% of normal to reflect the
    less-confirmed nature.
    """
    if len(df) < 2:
        return {"trend": "NEUTRAL", "is_valid": False, "strength": 0.0}

    last = df.iloc[-1]

    adx = float(last["adx"]) if "adx" in last.index and not pd.isna(last.get("adx")) else 0.0
    slope20 = float(last["ema20_slope"]) if "ema20_slope" in last.index and not pd.isna(last.get("ema20_slope")) else 0.0
    slope50 = float(last["ema50_slope"]) if "ema50_slope" in last.index and not pd.isna(last.get("ema50_slope")) else 0.0

    adx_norm = max(0.0, min((adx - 15.0) / 25.0, 1.0))
    slope_norm = max(0.0, min(abs(slope20) / 2.0, 1.0))
    strength = round(0.6 * adx_norm + 0.4 * slope_norm, 3)

    # Primary: full EMA alignment
    if last["close"] > last["ema20"] and last["ema20"] > last["ema50"]:
        return {"trend": "LONG", "is_valid": True, "strength": strength}

    if last["close"] < last["ema20"] and last["ema20"] < last["ema50"]:
        return {"trend": "SHORT", "is_valid": True, "strength": strength}

    # Alternative: recovering trend — price beyond EMA50 with EMA50 moving
    # in the same direction (crossover hasn't completed yet after a pullback)
    recovering_strength = round(strength * 0.7, 3)

    if last["close"] > last["ema50"] and slope50 > 0:
        return {"trend": "LONG", "is_valid": True, "strength": recovering_strength}

    if last["close"] < last["ema50"] and slope50 < 0:
        return {"trend": "SHORT", "is_valid": True, "strength": recovering_strength}

    return {"trend": "NEUTRAL", "is_valid": False, "strength": 0.0}


STRUCTURAL_LOOKBACK = 60
PIVOT_ORDER = 3
CLUSTER_ATR_BAND = 0.5


def _find_swing_highs(highs: pd.Series, order: int = PIVOT_ORDER) -> list[float]:
    """Return prices at local maxima where the bar is the highest within ±order bars."""
    pivots: list[float] = []
    for i in range(order, len(highs) - order):
        window = highs.iloc[i - order: i + order + 1]
        if highs.iloc[i] == window.max():
            pivots.append(float(highs.iloc[i]))
    return pivots


def _find_swing_lows(lows: pd.Series, order: int = PIVOT_ORDER) -> list[float]:
    """Return prices at local minima where the bar is the lowest within ±order bars."""
    pivots: list[float] = []
    for i in range(order, len(lows) - order):
        window = lows.iloc[i - order: i + order + 1]
        if lows.iloc[i] == window.min():
            pivots.append(float(lows.iloc[i]))
    return pivots


def _cluster_levels(pivots: list[float], atr: float, band: float = CLUSTER_ATR_BAND) -> list[tuple[float, int]]:
    """Cluster pivot points within ``band * atr`` of each other.

    Returns a list of (level, touch_count) sorted by touch count descending.
    The cluster level is the mean of the grouped pivots.
    """
    if not pivots or atr <= 0:
        return []
    sorted_pivots = sorted(pivots)
    clusters: list[tuple[float, int]] = []
    current_group: list[float] = [sorted_pivots[0]]
    threshold = band * atr

    for p in sorted_pivots[1:]:
        if p - current_group[-1] <= threshold:
            current_group.append(p)
        else:
            clusters.append((sum(current_group) / len(current_group), len(current_group)))
            current_group = [p]
    clusters.append((sum(current_group) / len(current_group), len(current_group)))
    clusters.sort(key=lambda x: x[1], reverse=True)
    return clusters


def _best_level_below(clusters: list[tuple[float, int]], price: float) -> tuple[float, int] | None:
    """Strongest cluster below *price*, or None.  Returns (level, touch_count)."""
    below = [(lvl, cnt) for lvl, cnt in clusters if lvl < price]
    return below[0] if below else None


def _best_level_above(clusters: list[tuple[float, int]], price: float) -> tuple[float, int] | None:
    """Strongest cluster above *price*, or None.  Returns (level, touch_count)."""
    above = [(lvl, cnt) for lvl, cnt in clusters if lvl > price]
    return above[0] if above else None


def find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Find tactical and structural S/R levels using pivot-point clustering.

    Tactical levels are the strongest clusters within the ``lookback``-bar
    window.  Structural levels use a 60-bar window.  Falls back to simple
    min/max when clustering yields no valid result.
    """
    if len(df) < lookback + 1:
        raise ValueError(
            f"Need at least {lookback + 1} rows to compute support/resistance"
        )

    last = df.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr14"]) if not pd.isna(last.get("atr14")) else 0.0

    recent = df.iloc[-(lookback + 1): -1]

    # Tactical S/R via clustering
    swing_lows = _find_swing_lows(recent["low"])
    swing_highs = _find_swing_highs(recent["high"])
    low_clusters = _cluster_levels(swing_lows, atr)
    high_clusters = _cluster_levels(swing_highs, atr)

    sup_result = _best_level_below(low_clusters, close)
    res_result = _best_level_above(high_clusters, close)

    support = sup_result[0] if sup_result else float(recent["low"].min())
    resistance = res_result[0] if res_result else float(recent["high"].max())

    # Structural S/R (longer window)
    struct_bars = min(STRUCTURAL_LOOKBACK, len(df) - 1)
    structural = df.iloc[-(struct_bars + 1): -1]

    struct_lows = _find_swing_lows(structural["low"])
    struct_highs = _find_swing_highs(structural["high"])
    struct_low_clusters = _cluster_levels(struct_lows, atr)
    struct_high_clusters = _cluster_levels(struct_highs, atr)

    struct_sup_result = _best_level_below(struct_low_clusters, close)
    struct_res_result = _best_level_above(struct_high_clusters, close)

    structural_support = struct_sup_result[0] if struct_sup_result else float(structural["low"].min())
    structural_resistance = struct_res_result[0] if struct_res_result else float(structural["high"].max())
    structural_support_touches = struct_sup_result[1] if struct_sup_result else 0
    structural_resistance_touches = struct_res_result[1] if struct_res_result else 0

    return {
        "support": float(support),
        "resistance": float(resistance),
        "structural_support": float(structural_support),
        "structural_resistance": float(structural_resistance),
        "structural_support_touches": structural_support_touches,
        "structural_resistance_touches": structural_resistance_touches,
        "structural_high": float(structural["high"].max()),
        "structural_low": float(structural["low"].min()),
    }


def _signal_bar_near_ema(df: pd.DataFrame, max_atr_dist: float = 1.0) -> bool:
    """True when the signal bar's close is within *max_atr_dist* ATR of EMA20
    or EMA50.  Used to invalidate stale pullback/rally patterns where price
    has since moved far away from the EMA."""
    last = df.iloc[-1]
    atr = last.get("atr14")
    close = last.get("close")
    if pd.isna(atr) or atr <= 0 or pd.isna(close):
        return False
    for col in ("ema20", "ema50"):
        ema = last.get(col)
        if pd.notna(ema) and abs(close - ema) <= max_atr_dist * atr:
            return True
    return False


def is_pullback_to_ema(
    df: pd.DataFrame,
    atr_multiple: float = 0.3,
    lookback: int = 5,
) -> bool:
    """Check whether any of the last ``lookback`` bars pulled back to the
    EMA20 or EMA50.

    Each bar is evaluated against its own EMA values (not today's), since the
    EMA shifts daily.  Passes if the bar's low actually touched or nearly
    touched the EMA (within ``atr_multiple * ATR``) while closing above it,
    and the signal bar hasn't since moved far away from the EMA.
    """
    if len(df) < lookback:
        return False
    if not _signal_bar_near_ema(df):
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
            if low <= ema + band and close >= ema:
                return True

    return False


def is_rally_to_ema(
    df: pd.DataFrame,
    atr_multiple: float = 0.3,
    lookback: int = 5,
) -> bool:
    """Short-side equivalent of ``is_pullback_to_ema``.

    Passes if any of the last ``lookback`` bars rallied up to touch or nearly
    touch the EMA20 or EMA50 (within ``atr_multiple * ATR``) and was rejected
    (closed back below it), and the signal bar hasn't since moved far away.
    """
    if len(df) < lookback:
        return False
    if not _signal_bar_near_ema(df):
        return False

    window = df.iloc[-lookback:]

    for _, bar in window.iterrows():
        atr = bar.get("atr14")
        ema20 = bar.get("ema20")
        ema50 = bar.get("ema50")
        high = bar.get("high")
        close = bar.get("close")

        if any(pd.isna(v) for v in (atr, ema20, ema50, high, close)) or atr <= 0:
            continue

        band = atr_multiple * atr
        for ema in (ema20, ema50):
            if high >= ema - band and close <= ema:
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


def is_retest_and_confirm_long(
    df: pd.DataFrame,
    structural_support: float | None = None,
    struct_touches: int = 0,
) -> bool:
    """Two-day retest-and-confirm for longs.

    Day N-1 (penultimate bar) tests a level — low dips into it and close
    stays near it.  Day N (last bar) prints a strong conviction green candle.
    Valid test levels: EMA20, EMA50, and structural support (if >= 2 touches).
    """
    if len(df) < 3:
        return False

    prev = df.iloc[-2]
    last = df.iloc[-1]

    if last["close"] <= last["open"]:
        return False
    atr = last.get("atr14")
    if pd.isna(atr) or atr <= 0:
        return False
    candle_range = last["high"] - last["low"]
    if candle_range < 0.5 * atr:
        return False
    close_pos = (last["close"] - last["low"]) / candle_range if candle_range > 0 else 0
    if close_pos < 0.65:
        return False

    prev_atr = prev.get("atr14")
    if pd.isna(prev_atr) or prev_atr <= 0:
        return False
    prev_low = prev["low"]
    prev_close = prev["close"]
    touch_band = 0.3 * prev_atr
    close_band = 0.75 * prev_atr

    for col in ("ema20", "ema50"):
        ema = prev.get(col)
        if pd.isna(ema):
            continue
        if prev_low <= ema + touch_band and abs(prev_close - ema) <= close_band:
            return True

    if structural_support is not None and struct_touches >= 2:
        if prev_low <= structural_support + touch_band and abs(prev_close - structural_support) <= close_band:
            return True

    return False


def is_retest_and_confirm_short(
    df: pd.DataFrame,
    structural_resistance: float | None = None,
    struct_touches: int = 0,
) -> bool:
    """Two-day retest-and-confirm for shorts.

    Day N-1 rallies into resistance/EMA and closes near it.  Day N prints
    a strong conviction red candle.
    """
    if len(df) < 3:
        return False

    prev = df.iloc[-2]
    last = df.iloc[-1]

    if last["close"] >= last["open"]:
        return False
    atr = last.get("atr14")
    if pd.isna(atr) or atr <= 0:
        return False
    candle_range = last["high"] - last["low"]
    if candle_range < 0.5 * atr:
        return False
    close_pos = (last["high"] - last["close"]) / candle_range if candle_range > 0 else 0
    if close_pos < 0.65:
        return False

    prev_atr = prev.get("atr14")
    if pd.isna(prev_atr) or prev_atr <= 0:
        return False
    prev_high = prev["high"]
    prev_close = prev["close"]
    touch_band = 0.3 * prev_atr
    close_band = 0.75 * prev_atr

    for col in ("ema20", "ema50"):
        ema = prev.get(col)
        if pd.isna(ema):
            continue
        if prev_high >= ema - touch_band and abs(prev_close - ema) <= close_band:
            return True

    if structural_resistance is not None and struct_touches >= 2:
        if prev_high >= structural_resistance - touch_band and abs(prev_close - structural_resistance) <= close_band:
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
    structural_high: float,
    lookback: int = 2,
) -> bool:
    """True when price breaks above the 60-bar high on volume.

    The signal bar must still hold above the level (failed breakouts are
    rejected), and at least one bar in the lookback window must have closed
    above the level on above-average volume.
    """
    if len(df) < lookback:
        return False
    if df.iloc[-1]["close"] <= structural_high:
        return False

    window = df.iloc[-lookback:]
    for _, bar in window.iterrows():
        rel_vol = bar.get("rel_volume")
        if pd.isna(rel_vol):
            continue
        if bar["close"] > structural_high and rel_vol >= 1.0:
            return True
    return False


def is_structural_breakdown(
    df: pd.DataFrame,
    structural_low: float,
    lookback: int = 2,
) -> bool:
    """True when price breaks below the 60-bar low on volume.

    The signal bar must still hold below the level (failed breakdowns are
    rejected), and at least one bar in the lookback window must have closed
    below the level on above-average volume.
    """
    if len(df) < lookback:
        return False
    if df.iloc[-1]["close"] >= structural_low:
        return False

    window = df.iloc[-lookback:]
    for _, bar in window.iterrows():
        rel_vol = bar.get("rel_volume")
        if pd.isna(rel_vol):
            continue
        if bar["close"] < structural_low and rel_vol >= 1.0:
            return True
    return False


def measured_move_flag_long(df: pd.DataFrame, flag_bars: int = 4) -> float | None:
    """Compute the measured move target for a bullish flag breakout.

    The target is the flag high plus the height of the pole (impulse move
    preceding the flag).  Returns None when the pattern geometry is unclear.
    """
    needed = flag_bars + 6
    if len(df) < needed:
        return None
    flag = df.iloc[-(flag_bars + 1):-1]
    pole = df.iloc[-(flag_bars + 6):-(flag_bars + 1)]
    flag_high = float(flag["high"].max())
    pole_low = float(pole["low"].min())
    pole_high = float(pole["high"].max())
    pole_height = pole_high - pole_low
    if pole_height <= 0:
        return None
    return flag_high + pole_height


def measured_move_flag_short(df: pd.DataFrame, flag_bars: int = 4) -> float | None:
    """Compute the measured move target for a bearish flag breakdown."""
    needed = flag_bars + 6
    if len(df) < needed:
        return None
    flag = df.iloc[-(flag_bars + 1):-1]
    pole = df.iloc[-(flag_bars + 6):-(flag_bars + 1)]
    flag_low = float(flag["low"].min())
    pole_low = float(pole["low"].min())
    pole_high = float(pole["high"].max())
    pole_height = pole_high - pole_low
    if pole_height <= 0:
        return None
    return flag_low - pole_height


def measured_move_breakout(
    df: pd.DataFrame,
    structural_resistance: float,
    consolidation_bars: int = 20,
) -> float | None:
    """Measured move target for a structural breakout.

    Projects the consolidation range above the breakout level.
    """
    if len(df) < consolidation_bars + 1:
        return None
    consol = df.iloc[-(consolidation_bars + 1):-1]
    consol_low = float(consol["low"].min())
    range_height = structural_resistance - consol_low
    if range_height <= 0:
        return None
    return structural_resistance + range_height


def measured_move_breakdown(
    df: pd.DataFrame,
    structural_support: float,
    consolidation_bars: int = 20,
) -> float | None:
    """Measured move target for a structural breakdown."""
    if len(df) < consolidation_bars + 1:
        return None
    consol = df.iloc[-(consolidation_bars + 1):-1]
    consol_high = float(consol["high"].max())
    range_height = consol_high - structural_support
    if range_height <= 0:
        return None
    return structural_support - range_height


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
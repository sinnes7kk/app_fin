"""Conviction scoring for continuation setups — continuous 0-10 scale."""

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
    is_pullback_to_ema,
    is_rally_to_ema,
    is_retest_and_confirm_long,
    is_retest_and_confirm_short,
    is_structural_breakout,
    is_structural_breakdown,
    has_volume_confirmation,
    MIN_ROOM_ATR,
)

MAX_DISTANCE_FROM_EMA20_ATR = 2.0
BREAKOUT_MAX_DISTANCE_ATR = 4.0



# ---------------------------------------------------------------------------
# Continuous component helpers
# ---------------------------------------------------------------------------

def _clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _trend_score(trend_ok: bool, strength: float) -> float:
    """0-3.0: direction must match, then scaled by strength."""
    if not trend_ok:
        return 0.0
    return 1.5 + 1.5 * _clip(strength, 0.0, 1.0)


def _extension_score(df: pd.DataFrame, max_distance_atr: float) -> tuple[bool, float]:
    """Returns (hard_gate_passed, continuous 0-1 score)."""
    if df.empty:
        return False, 0.0
    last = df.iloc[-1]
    atr = last.get("atr14")
    ema20 = last.get("ema20")
    close = last.get("close")
    if pd.isna(atr) or pd.isna(ema20) or pd.isna(close) or atr <= 0:
        return False, 0.0
    distance = abs(close - ema20) / atr
    passed = distance <= max_distance_atr
    score = _clip(1.0 - distance / max_distance_atr)
    return passed, score


def _room_score(df: pd.DataFrame, level: float, is_long: bool) -> float:
    """0-1: fractional credit based on room in ATR units (0.5 floor, 3.0 ceiling)."""
    if df.empty:
        return 0.0
    last = df.iloc[-1]
    atr = last.get("atr14")
    close = last.get("close")
    if pd.isna(atr) or pd.isna(close) or atr <= 0:
        return 0.0
    room_atr = ((level - close) if is_long else (close - level)) / atr
    return _clip((room_atr - 0.5) / 2.5)


def _is_clean_trend(df: pd.DataFrame, is_long: bool, lookback: int = 10) -> bool:
    """True when the stock is grinding directionally without a specific pattern.

    Requires 6+ of the last 10 bars to print higher lows (longs) or
    lower highs (shorts).
    """
    if len(df) < lookback + 1:
        return False
    window = df.iloc[-(lookback + 1):]
    if is_long:
        count = sum(
            1 for i in range(1, len(window))
            if float(window.iloc[i]["low"]) > float(window.iloc[i - 1]["low"])
        )
    else:
        count = sum(
            1 for i in range(1, len(window))
            if float(window.iloc[i]["high"]) < float(window.iloc[i - 1]["high"])
        )
    return count >= 6


def _pattern_score(
    structural: bool, flag: bool, pullback: bool, ema: bool, bounce: bool,
    confluence: bool = False, retest: bool = False, clean_trend: bool = False,
) -> tuple[float, str]:
    """0-2: differentiated credit by pattern quality. Returns (score, label)."""
    if structural:
        return 2.0, "structural"
    if flag:
        return 1.6, "flag"
    if confluence:
        return 1.6, "confluence"
    if pullback:
        return 1.4, "pullback"
    if retest:
        return 1.4, "retest"
    if ema:
        return 1.2, "ema"
    if bounce:
        return 1.0, "bounce"
    if clean_trend:
        return 1.4, "trend_cont"
    return 0.0, ""


def _confirmation_volume_score(df: pd.DataFrame) -> float:
    """0-1: high relative volume on the signal bar."""
    if df.empty or "rel_volume" not in df.columns:
        return 0.0
    rel_vol = df.iloc[-1].get("rel_volume", 0)
    if pd.isna(rel_vol):
        return 0.0
    return _clip((float(rel_vol) - 0.5) / 1.5)


def _momentum_score(df: pd.DataFrame, is_long: bool, lookback: int = 5) -> float:
    """0-2.0: signal bar's close position in the recent high-low range.

    Close near the top of the 5-bar range rewards longs (directional energy
    is upward). Close near the bottom rewards shorts.
    """
    if len(df) < lookback:
        return 0.0
    window = df.iloc[-lookback:]
    high_n = float(window["high"].max())
    low_n = float(window["low"].min())
    rng = high_n - low_n
    if rng <= 0:
        return 0.0
    pos = (float(df.iloc[-1]["close"]) - low_n) / rng
    raw = _clip(pos) if is_long else _clip(1.0 - pos)
    return raw * 2.0


# ---------------------------------------------------------------------------
# Legacy boolean gate wrappers (used by state machine and checks_passed/failed)
# ---------------------------------------------------------------------------

def is_not_extended(df: pd.DataFrame, max_distance_atr: float = MAX_DISTANCE_FROM_EMA20_ATR) -> bool:
    passed, _ = _extension_score(df, max_distance_atr)
    return passed


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(
    *,
    direction: str,
    score: float,
    support: float,
    resistance: float,
    structural_support: float,
    structural_resistance: float,
    state: str,
    reasons: list[str],
    checks_passed: list[str],
    checks_failed: list[str],
    components: dict | None = None,
    broken_level: float | None = None,
) -> dict:
    result = {
        "direction": direction,
        "score": round(score, 2),
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
    if components:
        result["score_components"] = components
    if broken_level is not None:
        result["broken_level"] = broken_level
    return result


# ---------------------------------------------------------------------------
# Early rejection pre-check — avoids expensive API calls
# ---------------------------------------------------------------------------

def quick_reject_check(
    df: pd.DataFrame,
    direction: str,
) -> tuple[bool, str | None, dict, bool]:
    """Pre-filter using only price data.

    Returns (should_reject, reject_reason, price_signal_stub, counter_trend).
    Only price extension is a hard gate; trend misalignment is a soft flag.
    """
    trend = detect_trend(df)
    trend_dir = trend["trend"]

    if direction == "LONG":
        trend_opposite = trend_dir == "SHORT"
    else:
        trend_opposite = trend_dir == "LONG"

    _, ext_sc = _extension_score(df, MAX_DISTANCE_FROM_EMA20_ATR)
    not_extended_ok = ext_sc > 0

    if not_extended_ok:
        return False, None, {}, trend_opposite

    stub: dict = {
        "score": 0,
        "is_valid": False,
        "state": "REJECT",
        "reasons": [],
        "checks_passed": [],
        "checks_failed": ["not_extended"],
        "score_components": {
            "trend": 0.0,
            "extension": round(ext_sc, 2),
            "room": 0.0,
            "pattern": 0.0,
            "momentum": 0.0,
            "confirm_vol": 0.0,
        },
    }
    return True, "price_over_extended", stub, trend_opposite


# ---------------------------------------------------------------------------
# Main scoring functions — continuous 0-10
# ---------------------------------------------------------------------------

_LONG_PATTERN_LABELS = {
    "structural": "structural_breakout",
    "flag": "flag_breakout",
    "confluence": "support_ema_confluence",
    "pullback": "pullback_to_support",
    "retest": "retest_and_confirm",
    "ema": "ema_pullback",
    "bounce": "bounce_and_fail",
    "trend_cont": "trend_continuation",
}

_SHORT_PATTERN_LABELS = {
    "structural": "structural_breakdown",
    "flag": "flag_breakdown",
    "confluence": "resistance_ema_confluence",
    "pullback": "pullback_to_resistance",
    "retest": "retest_and_confirm",
    "ema": "ema_rally",
    "bounce": "bounce_and_fail",
    "trend_cont": "trend_continuation",
}


def score_long_setup(
    df: pd.DataFrame,
    signal_bar_offset: int = 0,
) -> dict:
    """Score a long continuation setup on a continuous 0-10 scale."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    trend_ok = trend["trend"] == "LONG"
    trend_neutral = trend["trend"] == "NEUTRAL"
    trend_opposite = trend["trend"] == "SHORT"
    trend_strength = trend.get("strength", 0.0)

    range_ceiling = levels.get("range_ceiling")
    range_ceiling_touches = levels.get("range_ceiling_touches", 0)

    structural_bo = (
        (range_ceiling_touches >= 2
         and range_ceiling is not None
         and is_structural_breakout(df, range_ceiling))
        or (levels["structural_resistance_touches"] >= 2
            and is_structural_breakout(df, levels["structural_resistance"]))
    )
    _bo_broken_level: float | None = None
    if structural_bo:
        if (range_ceiling is not None and range_ceiling_touches >= 2
                and is_structural_breakout(df, range_ceiling)):
            _bo_broken_level = range_ceiling
        else:
            _bo_broken_level = levels["structural_resistance"]

    max_dist = BREAKOUT_MAX_DISTANCE_ATR if structural_bo else MAX_DISTANCE_FROM_EMA20_ATR
    not_extended_ok, ext_sc = _extension_score(df, max_dist)

    room_ok = has_room_to_target_long(df, levels["structural_resistance"])
    room_sc = _room_score(df, levels["structural_resistance"], is_long=True)
    if structural_bo:
        room_sc = 1.0

    bounce_fail = is_bounce_and_fail_long(df)
    flag = is_flag_breakout(df)
    if levels["structural_support_touches"] >= 2:
        pullback_to_level = is_pullback_to_support(df, levels["structural_support"], atr_multiple=0.5)
    else:
        pullback_to_level = False
    ema_pullback = is_pullback_to_ema(df)
    retest_confirm = is_retest_and_confirm_long(
        df, levels["structural_support"], levels["structural_support_touches"],
    )
    pullback_ema_confluence = pullback_to_level and ema_pullback
    clean_trend = _is_clean_trend(df, is_long=True)
    continuation_ok = (
        bounce_fail or flag or pullback_to_level or structural_bo
        or ema_pullback or retest_confirm or clean_trend
    )
    pattern_sc, pattern_key = _pattern_score(
        structural_bo, flag, pullback_to_level, ema_pullback, bounce_fail,
        confluence=pullback_ema_confluence, retest=retest_confirm,
        clean_trend=clean_trend,
    )
    intraday = signal_bar_offset and len(df) > 1 and df.iloc[-1].get("is_intraday", False)
    conf_vol_df = df if intraday else (df.iloc[:-1] if signal_bar_offset and len(df) > 1 else df)
    confirmation_volume_ok = has_volume_confirmation(conf_vol_df)
    vol_conf_sc = _confirmation_volume_score(conf_vol_df)

    momentum_sc = _momentum_score(df, is_long=True)
    trend_sc = _trend_score(trend_ok, trend_strength)

    score = trend_sc + ext_sc + room_sc + pattern_sc + momentum_sc + vol_conf_sc
    score = min(10.0, score)

    reasons: list[str] = []
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    if trend_ok:
        if trend_strength >= 0.5:
            reasons.append("strong_trend")
        reasons.append("trend_aligned")
        checks_passed.append("trend_aligned")
    else:
        checks_failed.append("trend_aligned")

    if not_extended_ok:
        reasons.append("not_extended")
        checks_passed.append("not_extended")
    else:
        checks_failed.append("not_extended")

    if room_ok or structural_bo:
        reasons.append("room_to_target")
        checks_passed.append("room_to_target")
    else:
        checks_failed.append("room_to_target")

    if continuation_ok:
        label = _LONG_PATTERN_LABELS.get(pattern_key, "unknown")
        reasons.append(label)
        checks_passed.append(label)
    else:
        checks_failed.append("continuation_pattern")

    if confirmation_volume_ok:
        reasons.append("confirmation_volume")
        checks_passed.append("confirmation_volume")
    else:
        checks_failed.append("confirmation_volume")

    if momentum_sc >= 0.6:
        reasons.append("momentum_aligned")
        checks_passed.append("momentum_aligned")
    else:
        checks_failed.append("momentum_aligned")

    components = {
        "trend": round(trend_sc, 2),
        "extension": round(ext_sc, 2),
        "room": round(room_sc, 2),
        "pattern": round(pattern_sc, 2),
        "momentum": round(momentum_sc, 2),
        "confirm_vol": round(vol_conf_sc, 2),
    }

    if trend_opposite or not not_extended_ok:
        state = "REJECT"
    elif score >= 7.0:
        state = "SIGNAL"
    elif score >= 4.0:
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
        components=components,
        broken_level=_bo_broken_level,
    )


def score_short_setup(
    df: pd.DataFrame,
    signal_bar_offset: int = 0,
) -> dict:
    """Score a short continuation setup on a continuous 0-10 scale."""
    trend = detect_trend(df)
    levels = find_support_resistance(df)

    trend_ok = trend["trend"] == "SHORT"
    trend_neutral = trend["trend"] == "NEUTRAL"
    trend_opposite = trend["trend"] == "LONG"
    trend_strength = trend.get("strength", 0.0)

    range_floor = levels.get("range_floor")
    range_floor_touches = levels.get("range_floor_touches", 0)

    structural_bd = (
        (range_floor_touches >= 2
         and range_floor is not None
         and is_structural_breakdown(df, range_floor))
        or (levels["structural_support_touches"] >= 2
            and is_structural_breakdown(df, levels["structural_support"]))
    )
    _bd_broken_level: float | None = None
    if structural_bd:
        if (range_floor is not None and range_floor_touches >= 2
                and is_structural_breakdown(df, range_floor)):
            _bd_broken_level = range_floor
        else:
            _bd_broken_level = levels["structural_support"]

    max_dist = BREAKOUT_MAX_DISTANCE_ATR if structural_bd else MAX_DISTANCE_FROM_EMA20_ATR
    not_extended_ok, ext_sc = _extension_score(df, max_dist)

    room_ok = has_room_to_target_short(df, levels["structural_support"])
    room_sc = _room_score(df, levels["structural_support"], is_long=False)
    if structural_bd:
        room_sc = 1.0

    bounce_fail = is_bounce_and_fail_short(df)
    flag = is_flag_breakdown(df)
    if levels["structural_resistance_touches"] >= 2:
        pullback_to_level = is_pullback_to_resistance(df, levels["structural_resistance"], atr_multiple=0.5)
    else:
        pullback_to_level = False
    ema_rally = is_rally_to_ema(df)
    retest_confirm = is_retest_and_confirm_short(
        df, levels["structural_resistance"], levels["structural_resistance_touches"],
    )
    pullback_ema_confluence = pullback_to_level and ema_rally
    clean_trend = _is_clean_trend(df, is_long=False)
    continuation_ok = (
        bounce_fail or flag or pullback_to_level or structural_bd
        or ema_rally or retest_confirm or clean_trend
    )
    pattern_sc, pattern_key = _pattern_score(
        structural_bd, flag, pullback_to_level, ema_rally, bounce_fail,
        confluence=pullback_ema_confluence, retest=retest_confirm,
        clean_trend=clean_trend,
    )
    intraday = signal_bar_offset and len(df) > 1 and df.iloc[-1].get("is_intraday", False)
    conf_vol_df = df if intraday else (df.iloc[:-1] if signal_bar_offset and len(df) > 1 else df)
    confirmation_volume_ok = has_volume_confirmation(conf_vol_df)
    vol_conf_sc = _confirmation_volume_score(conf_vol_df)

    momentum_sc = _momentum_score(df, is_long=False)
    trend_sc = _trend_score(trend_ok, trend_strength)

    score = trend_sc + ext_sc + room_sc + pattern_sc + momentum_sc + vol_conf_sc
    score = min(10.0, score)

    reasons: list[str] = []
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    if trend_ok:
        if trend_strength >= 0.5:
            reasons.append("strong_trend")
        reasons.append("trend_aligned")
        checks_passed.append("trend_aligned")
    else:
        checks_failed.append("trend_aligned")

    if not_extended_ok:
        reasons.append("not_extended")
        checks_passed.append("not_extended")
    else:
        checks_failed.append("not_extended")

    if room_ok or structural_bd:
        reasons.append("room_to_target")
        checks_passed.append("room_to_target")
    else:
        checks_failed.append("room_to_target")

    if continuation_ok:
        label = _SHORT_PATTERN_LABELS.get(pattern_key, "unknown")
        reasons.append(label)
        checks_passed.append(label)
    else:
        checks_failed.append("continuation_pattern")

    if confirmation_volume_ok:
        reasons.append("confirmation_volume")
        checks_passed.append("confirmation_volume")
    else:
        checks_failed.append("confirmation_volume")

    if momentum_sc >= 0.6:
        reasons.append("momentum_aligned")
        checks_passed.append("momentum_aligned")
    else:
        checks_failed.append("momentum_aligned")

    components = {
        "trend": round(trend_sc, 2),
        "extension": round(ext_sc, 2),
        "room": round(room_sc, 2),
        "pattern": round(pattern_sc, 2),
        "momentum": round(momentum_sc, 2),
        "confirm_vol": round(vol_conf_sc, 2),
    }

    if trend_opposite or not not_extended_ok:
        state = "REJECT"
    elif score >= 7.0:
        state = "SIGNAL"
    elif score >= 4.0:
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
        components=components,
        broken_level=_bd_broken_level,
    )

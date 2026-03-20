"""Trade plan generation for continuation setups."""

from __future__ import annotations

import pandas as pd


def build_long_trade_plan(df: pd.DataFrame, scored_signal: dict) -> dict:
    """Build a simple long trade plan from a scored long setup."""
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

    stop_price = min(latest_low, support) - 0.25 * atr
    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        raise ValueError("Invalid long trade plan: non-positive risk per share")

    target_1 = entry_price + 1.5 * risk_per_share
    target_2 = entry_price + 3.0 * risk_per_share

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
        "time_stop_days": 5,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "reasons": scored_signal["reasons"],
    }


def build_short_trade_plan(df: pd.DataFrame, scored_signal: dict) -> dict:
    """Build a simple short trade plan from a scored short setup."""
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

    stop_price = max(latest_high, resistance) + 0.25 * atr
    risk_per_share = stop_price - entry_price

    if risk_per_share <= 0:
        raise ValueError("Invalid short trade plan: non-positive risk per share")

    target_1 = entry_price - 1.5 * risk_per_share
    target_2 = entry_price - 3.0 * risk_per_share

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
        "time_stop_days": 5,
        "support": round(support, 2),
        "resistance": round(resistance, 2),
        "reasons": scored_signal["reasons"],
    }

"""Auto-promote Flow Tracker Grade A entries into actionable signals.

Why this exists
---------------
Today, the main signal pipeline rejects setups whose price/structure
gating fails — even when Flow Tracker shows extreme multi-day
conviction. The "Saw it, couldn't trade it" panel surfaces these as
read-only watchlist items. This module is the bridge: when a Flow
Tracker row is Grade A AND meets a sanity gate, we synthesize an
ATR-derived trade plan and promote it back into ``final_results`` so
it appears as a signal in the dashboard.

What "synthesize a trade plan" means
------------------------------------
The Flow Tracker grade is built from options-flow features only. It
doesn't give us entry / stop / T1 / T2 levels. To promote a row into
a tradeable signal we need those levels. We derive them here from
ATR-14:

    direction = BULLISH (most flow tracker rows; bearish handled symmetrically)
    entry_price  = last close
    risk_per_share = 1.0 * ATR14
    stop_price   = entry - risk_per_share        (long)
    target_1     = entry + 1.5 * ATR14           (1.5R)
    target_2     = entry + 3.0 * ATR14           (3R)

These match the production trade plan's R-multiple ratios, so the
``replay_trade_plan`` engine and ``_check_exits`` code applies cleanly
to promoted signals without special-casing.

Sanity gates
------------
Promoting every Grade A blindly is unsafe. We require:
    1. Conviction grade is "A" or "A+".
    2. ``conviction_score`` >= ``FLOW_PROMOTE_MIN_SCORE`` (default 8.0).
    3. Latest close is on the right side of EMA20 (long: above; short:
       below) — same trend confirmation production breakout signals get.
    4. ATR14 is positive and the resulting risk_per_share > 0.
    5. The ticker is NOT already in ``existing_signals`` (no duplicates).

Rows failing any gate are skipped and noted in the return dict.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import (
    FLOW_PROMOTE_ENABLED,
    FLOW_PROMOTE_MIN_SCORE,
    FLOW_PROMOTE_REQUIRE_EMA20_TREND_CONFIRM,
    MIN_FINAL_SCORE,
)
from app.features.price_features import compute_features, fetch_ohlcv
from app.signals.hold_config import resolve_hold_config


def _atr_plan_long(close: float, atr: float) -> dict[str, float]:
    risk = atr  # 1*ATR risk
    return {
        "entry_price": round(close, 2),
        "stop_price": round(close - risk, 2),
        "risk_per_share": round(risk, 2),
        "target_1": round(close + 1.5 * atr, 2),
        "target_2": round(close + 3.0 * atr, 2),
        "rr_ratio": 1.5,
    }


def _atr_plan_short(close: float, atr: float) -> dict[str, float]:
    risk = atr
    return {
        "entry_price": round(close, 2),
        "stop_price": round(close + risk, 2),
        "risk_per_share": round(risk, 2),
        "target_1": round(close - 1.5 * atr, 2),
        "target_2": round(close - 3.0 * atr, 2),
        "rr_ratio": 1.5,
    }


def _passes_sanity_gate(grade_row: dict, df: pd.DataFrame) -> tuple[bool, str]:
    """Return (passed, reason). ``reason`` is empty when passed=True."""
    grade = str(grade_row.get("conviction_grade") or "").upper()
    if grade not in ("A", "A+"):
        return False, f"grade_{grade}_not_promotable"
    score = float(grade_row.get("conviction_score") or 0.0)
    if score < FLOW_PROMOTE_MIN_SCORE:
        return False, f"score_{score:.1f}_below_floor_{FLOW_PROMOTE_MIN_SCORE}"
    if df is None or df.empty:
        return False, "no_ohlcv"
    last = df.iloc[-1]
    try:
        close = float(last["close"])
        atr = float(last.get("atr14", 0))
        ema20 = float(last.get("ema20", 0))
    except Exception:
        return False, "ohlcv_features_missing"
    if atr <= 0 or close <= 0:
        return False, "atr_or_close_invalid"
    direction = str(grade_row.get("direction") or "BULLISH").upper()
    if FLOW_PROMOTE_REQUIRE_EMA20_TREND_CONFIRM and ema20 > 0:
        if direction == "BULLISH" and close < ema20:
            return False, "long_below_ema20_trend_failed"
        if direction == "BEARISH" and close > ema20:
            return False, "short_above_ema20_trend_failed"
    return True, ""


def promote_flow_tracker_grade_a(
    flow_tracker_qualified: list[dict],
    existing_signals: list[dict],
) -> tuple[list[dict], dict]:
    """Promote Flow Tracker Grade A entries that are not already signals.

    Returns ``(promoted_signals, summary)``. ``summary`` contains
    ``promoted``, ``skipped_already_signal``, ``skipped_sanity_gate`` and
    ``skipped_no_ohlcv`` counts plus a ``rejections`` list of (ticker,
    reason) tuples for diagnostic output.
    """
    if not FLOW_PROMOTE_ENABLED:
        return [], {
            "promoted": 0,
            "skipped_already_signal": 0,
            "skipped_sanity_gate": 0,
            "skipped_no_ohlcv": 0,
            "rejections": [],
            "feature_disabled": True,
        }

    existing_tickers = {
        str(s.get("ticker") or "").upper() for s in (existing_signals or [])
    }
    promoted: list[dict] = []
    rejections: list[tuple[str, str]] = []
    skipped_already = 0
    skipped_gate = 0
    skipped_no_ohlcv = 0

    for g in flow_tracker_qualified or []:
        ticker = str(g.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        if ticker in existing_tickers:
            skipped_already += 1
            continue

        # Fetch OHLCV with technical features.
        try:
            df_raw = fetch_ohlcv(ticker, lookback_days=120, include_partial=False)
            df = compute_features(df_raw) if df_raw is not None else None
        except Exception:
            df = None
        if df is None or df.empty:
            skipped_no_ohlcv += 1
            rejections.append((ticker, "ohlcv_fetch_failed"))
            continue

        passed, reason = _passes_sanity_gate(g, df)
        if not passed:
            skipped_gate += 1
            rejections.append((ticker, reason))
            continue

        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(last["atr14"])
        direction = str(g.get("direction") or "BULLISH").upper()
        plan = _atr_plan_long(close, atr) if direction == "BULLISH" else _atr_plan_short(close, atr)
        max_hold, _ = resolve_hold_config(g.get("dominant_dte_bucket"))

        # Build a signal record matching the shape of pipeline final_results.
        # The conviction_score (0-10) is mapped to MIN_FINAL_SCORE+ so the
        # promoted entry survives the regime threshold without further work.
        sig: dict[str, Any] = {
            "ticker": ticker,
            "direction": "LONG" if direction == "BULLISH" else "SHORT",
            "sector": g.get("sector"),
            "final_score": max(MIN_FINAL_SCORE, float(g.get("conviction_score") or 7.0)),
            "score": max(MIN_FINAL_SCORE, float(g.get("conviction_score") or 7.0)),
            "is_valid": True,
            "state": "SIGNAL",
            "reasons": ["flow_tracker_grade_a_promoted"],
            "pattern": "flow_promoted",
            "source": "flow_promoted",
            "promoted_from_flow_tracker": True,
            "conviction_grade": g.get("conviction_grade"),
            "conviction_score": g.get("conviction_score"),
            "flow_score_scaled": float(g.get("conviction_score") or 0.0),
            "flow_score_raw": float(g.get("conviction_score") or 0.0) / 10.0,
            "dominant_dte_bucket": g.get("dominant_dte_bucket"),
            **plan,
            "time_stop_days": max_hold,
            "support": round(close - 2.0 * atr, 2),
            "resistance": round(close + 2.0 * atr, 2),
            "atr14": round(atr, 2),
            "close": round(close, 2),
        }
        promoted.append(sig)
        existing_tickers.add(ticker)

    return promoted, {
        "promoted": len(promoted),
        "skipped_already_signal": skipped_already,
        "skipped_sanity_gate": skipped_gate,
        "skipped_no_ohlcv": skipped_no_ohlcv,
        "rejections": rejections[:25],  # cap for log readability
        "feature_disabled": False,
    }

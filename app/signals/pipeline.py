"""End-to-end pipeline: flow candidates -> price validation -> final ranked setups."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.features.decision_context import (
    clear_decision_cache,
    enrich_signals as enrich_decision_context,
)
from app.features.flow_features import build_flow_feature_table, rank_flow_candidates
from app.features.flow_persistence import apply_persistence_bonus, compute_persistence
from app.features.flow_trajectory import apply_trajectory_bonus, compute_intraday_trajectory
from app.features.market_regime import fetch_market_regime
from app.features.options_context import clear_context_cache, fetch_options_context
from app.features.price_features import clean_ohlcv, clear_price_cache, compute_features, fetch_ohlcv
from app.signals.scoring import quick_reject_check, score_long_setup, score_short_setup
from app.config import (
    COUNTER_TREND_PREMIUM,
    MIN_FINAL_SCORE,
    REGIME_THRESHOLD_BOOST,
    RS_LONG_MIN,
    RS_LOOKBACK_DAYS,
    RS_SHORT_MAX,
    WALL_PROXIMITY_REJECT_ATR,
    WALL_PROXIMITY_REJECT_PCT,
)
from app.signals.trade_plan import (
    MIN_RR,
    build_long_trade_plan,
    build_short_trade_plan,
)
from app.vendors.unusual_whales import (
    fetch_dark_pool,
    fetch_dark_pool_recent,
    fetch_earnings,
    fetch_flow_for_tickers,
    fetch_flow_raw,
    fetch_hottest_chains,
    fetch_insider_transactions,
    fetch_net_prem_ticks,
    fetch_stock_screener,
    fetch_uw_alerts,
    normalize_flow_response,
    print_api_summary,
    reset_api_stats,
)


NON_EQUITY_TICKERS = {"SPXW", "SPXE", "NDXP", "RUTW", "VIX", "VIXW"}

_ET = ZoneInfo("America/New_York")


def _is_market_hours() -> bool:
    """Return True if the current time is within US equity market hours."""
    now_et = datetime.now(_ET)
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    from datetime import time as _time
    return _time(9, 30) <= t < _time(16, 0)


_SPY_RETURN_CACHE: dict[str, float | None] = {}


def _spy_return(lookback: int = RS_LOOKBACK_DAYS) -> float | None:
    """Cached SPY return over the lookback window."""
    cache_key = f"spy_{lookback}"
    if cache_key in _SPY_RETURN_CACHE:
        return _SPY_RETURN_CACHE[cache_key]
    try:
        spy_df = clean_ohlcv(fetch_ohlcv("SPY", lookback_days=lookback + 30))
        if len(spy_df) < lookback:
            _SPY_RETURN_CACHE[cache_key] = None
            return None
        ret = (float(spy_df.iloc[-1]["close"]) / float(spy_df.iloc[-lookback]["close"])) - 1.0
        _SPY_RETURN_CACHE[cache_key] = ret
        return ret
    except Exception:
        _SPY_RETURN_CACHE[cache_key] = None
        return None


def _relative_strength(df: pd.DataFrame, lookback: int = RS_LOOKBACK_DAYS) -> float | None:
    """Compute ticker return minus SPY return over the lookback window."""
    spy_ret = _spy_return(lookback)
    if spy_ret is None or len(df) < lookback:
        return None
    ticker_ret = (float(df.iloc[-1]["close"]) / float(df.iloc[-lookback]["close"])) - 1.0
    return ticker_ret - spy_ret

CONTINUATION_PATTERNS = {
    "structural_breakout", "structural_breakdown",
    "bounce_and_fail", "flag_breakout", "flag_breakdown",
    "pullback_to_support", "pullback_to_resistance",
    "support_ema_confluence", "resistance_ema_confluence",
    "retest_and_confirm", "ema_pullback", "ema_rally",
    "trend_continuation",
}


def _extract_pattern(reasons: list[str]) -> str:
    """Extract the continuation pattern name from a scoring reasons list."""
    for r in reasons:
        if r in CONTINUATION_PATTERNS:
            return r
    return "unknown"


def minmax_scale(series: pd.Series, target_max: float = 10.0) -> pd.Series:
    """Scale a Series to 0–target_max via min-max normalization."""
    min_v, max_v = series.min(), series.max()
    if max_v == min_v:
        return series.apply(lambda _: target_max / 2)
    return target_max * (series - min_v) / (max_v - min_v)


def combine_scores(
    flow_score: float,
    price_score: float,
    options_score: float | None = None,
) -> float:
    """Combine conviction layers into one final score (0-10).

    When options context is available, uses 50/30/20 (flow/price/options).
    Falls back to 60/40 (flow/price) when options data is missing.
    """
    if options_score is not None:
        return round(0.50 * flow_score + 0.30 * price_score + 0.20 * options_score, 4)
    return round(0.60 * flow_score + 0.40 * price_score, 4)


def _iv_rank_score(iv_rank: float | None) -> float:
    """Piecewise linear IV rank → score (0-2.5).

    Peak at the 15-45 "value zone" where options are cheap but active,
    tapering smoothly to zero at both extremes (dead market / IV crush).
    """
    if iv_rank is None:
        return 0.0
    r = float(iv_rank)
    if r <= 5:
        return 0.0
    if r <= 15:
        return (r - 5) / 10 * 1.0
    if r <= 45:
        return 1.0 + (r - 15) / 30 * 1.5
    if r <= 60:
        return 2.5 - (r - 45) / 15 * 1.0
    if r <= 75:
        return 1.5 - (r - 60) / 15 * 1.0
    if r <= 100:
        return 0.5 - (r - 75) / 25 * 0.5
    return 0.0


def compute_options_context_score(direction: str, opts_ctx: dict | None) -> float | None:
    """Derive a 0-10 composite score from options context fields.

    Components (direction-aware):
      Gamma alignment  0-2.5  regime/flip supports thesis
      Wall proximity   0-2.5  supportive wall far, opposing wall near
      OI structure     0-1.5  swing DTE dominant, favorable P/C ratio
      Premium bias     0-1.0  daily premium aligns with direction
      IV rank          0-2.5  smooth piecewise curve; peak in 15-45 value zone

    Returns None when options data is unavailable so callers can
    distinguish "no data" from a genuine neutral score.
    """
    if not opts_ctx or not opts_ctx.get("options_context_available"):
        return None

    score = 0.0
    regime = opts_ctx.get("gamma_regime", "NEUTRAL")
    dist_call = opts_ctx.get("distance_to_call_wall_pct")
    dist_put = opts_ctx.get("distance_to_put_wall_pct")
    near_oi = opts_ctx.get("near_term_oi") or 0
    swing_oi = opts_ctx.get("swing_dte_oi") or 0
    pcr = opts_ctx.get("ticker_put_call_ratio")
    bull_prem = opts_ctx.get("daily_bullish_premium")
    bear_prem = opts_ctx.get("daily_bearish_premium")

    # -- Gamma alignment (0-2.5) --
    if direction == "LONG":
        if regime == "NEGATIVE":
            score += 2.5
        elif regime == "NEUTRAL":
            score += 1.25
    else:
        if regime == "NEGATIVE":
            score += 2.5
        elif regime == "NEUTRAL":
            score += 1.25

    # -- Wall proximity (0-2.5) --
    if direction == "LONG":
        if dist_call is not None:
            if dist_call > 5.0:
                score += 2.5
            elif dist_call > 2.0:
                score += 1.25
    else:
        if dist_put is not None:
            if dist_put > 5.0:
                score += 2.5
            elif dist_put > 2.0:
                score += 1.25

    # -- OI structure (0-1.5) --
    if swing_oi > near_oi and near_oi > 0:
        score += 0.75
    if pcr is not None:
        if direction == "LONG" and pcr < 0.7:
            score += 0.75
        elif direction == "SHORT" and pcr > 1.3:
            score += 0.75
        elif direction == "LONG" and pcr < 1.0:
            score += 0.375
        elif direction == "SHORT" and pcr > 1.0:
            score += 0.375

    # -- Premium bias (0-1.0) --
    if bull_prem is not None and bear_prem is not None:
        total = bull_prem + bear_prem
        if total > 0:
            aligned = bull_prem / total if direction == "LONG" else bear_prem / total
            score += min(1.0, aligned * 1.5)

    # -- IV rank (0-2.5, smooth piecewise linear) --
    score += _iv_rank_score(opts_ctx.get("iv_rank"))

    return round(min(10.0, score), 1)


def has_strong_bullish_flow(row, min_ratio: float = 1.5) -> bool:
    bull = row.get("bullish_premium_raw", row["bullish_premium"])
    bear = row.get("bearish_premium_raw", row["bearish_premium"])
    return bull > bear * min_ratio


def has_strong_bearish_flow(row, min_ratio: float = 1.5) -> bool:
    bear = row.get("bearish_premium_raw", row["bearish_premium"])
    bull = row.get("bullish_premium_raw", row["bullish_premium"])
    return bear > bull * min_ratio


MAX_SAME_DIRECTION = 5
DIRECTION_ESCALATION_SCORE = 1.5


def apply_directional_balance(results: list[dict]) -> list[dict]:
    """Limit directional concentration by requiring progressively higher
    conviction for additional same-direction signals beyond MAX_SAME_DIRECTION."""
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    counts: dict[str, int] = {"LONG": 0, "SHORT": 0}
    balanced: list[dict] = []

    for r in results:
        d = r["direction"]
        counts[d] = counts.get(d, 0) + 1
        excess = counts[d] - MAX_SAME_DIRECTION
        if excess > 0:
            min_required = 5.0 + excess * DIRECTION_ESCALATION_SCORE
            if r["final_score"] < min_required:
                continue
        balanced.append(r)

    return balanced


def dedupe_final_results(results: list[dict]) -> list[dict]:
    """Keep only the highest-scoring direction per ticker."""
    best_by_ticker: dict[str, dict] = {}
    for result in results:
        ticker = result["ticker"]
        if ticker not in best_by_ticker or result["final_score"] > best_by_ticker[ticker]["final_score"]:
            best_by_ticker[ticker] = result
    return list(best_by_ticker.values())


def _attach_price_checks_to_ranked(
    df: pd.DataFrame, accepted: list[dict], rejected: list[dict]
) -> pd.DataFrame:
    """Add checks_passed / checks_failed from validation outcomes to ranked candidate rows."""
    if df.empty:
        return df
    meta: dict[str, tuple[str, str]] = {}
    for r in accepted:
        meta[r["ticker"]] = (r.get("checks_passed", "none"), r.get("checks_failed", "none"))
    for r in rejected:
        t = r["ticker"]
        if t not in meta:
            meta[t] = (r.get("checks_passed", "none"), r.get("checks_failed", "none"))
    out = df.copy()
    out["checks_passed"] = out["ticker"].map(lambda t: meta.get(t, ("", ""))[0])
    out["checks_failed"] = out["ticker"].map(lambda t: meta.get(t, ("", ""))[1])
    return out


_FLOW_COMPONENT_KEYS = [
    "flow_velocity", "repeat_flow_count", "dte_score", "flow_imbalance_ratio",
    "bullish_flow_intensity", "bearish_flow_intensity",
    "bullish_premium_per_trade", "bearish_premium_per_trade",
    "bullish_vol_oi", "bearish_vol_oi",
    "bullish_sweep_count", "bearish_sweep_count",
    "bullish_breadth", "bearish_breadth",
]

def _attach_flow_components(target: dict, row) -> None:
    """Copy flow component fields from a ranked DataFrame row onto a dict."""
    for fk in _FLOW_COMPONENT_KEYS:
        val = row.get(fk) if hasattr(row, "get") else None
        if val is not None and not (isinstance(val, float) and val != val):
            target[fk] = float(val) if hasattr(val, "item") else val


LONG_ALL_REASONS = {
    "trend_aligned", "not_extended", "room_to_target",
    "confirmation_volume", "momentum_aligned",
}
SHORT_ALL_REASONS = {
    "trend_aligned", "not_extended", "room_to_target",
    "confirmation_volume", "momentum_aligned",
}


def _build_rejection_row(
    ticker: str,
    direction: str,
    flow_score_raw: float,
    price_signal: dict,
    all_reasons: set[str],
    reject_reason: str = "price_validation_failed",
    flow_score_scaled: float | None = None,
    opts_ctx: dict | None = None,
    trade_plan: dict | None = None,
) -> dict:
    passed_list = price_signal.get("checks_passed", [])
    failed_list = price_signal.get("checks_failed", [])
    _fs = flow_score_scaled if flow_score_scaled is not None else flow_score_raw
    _ps = price_signal.get("score", 0)
    _opts_score = compute_options_context_score(direction, opts_ctx)
    row: dict = {
        "ticker": ticker,
        "direction": direction,
        "flow_score_raw": flow_score_raw,
        "flow_score_scaled": _fs,
        "price_score": _ps,
        "final_score": combine_scores(_fs, float(_ps), _opts_score),
        "reject_reason": reject_reason,
        "checks_passed": ", ".join(sorted(passed_list)) or "none",
        "checks_failed": ", ".join(sorted(failed_list)) or "none",
        "options_context_score": _opts_score,
        "gamma_regime": opts_ctx.get("gamma_regime") if opts_ctx else None,
        "pattern": _extract_pattern(price_signal.get("reasons", [])),
        "net_gex": opts_ctx.get("net_gex") if opts_ctx else None,
        "gamma_flip_level_estimate": opts_ctx.get("gamma_flip_level_estimate") if opts_ctx else None,
        "nearest_call_wall": opts_ctx.get("nearest_call_wall") if opts_ctx else None,
        "nearest_put_wall": opts_ctx.get("nearest_put_wall") if opts_ctx else None,
        "distance_to_call_wall_pct": opts_ctx.get("distance_to_call_wall_pct") if opts_ctx else None,
        "distance_to_put_wall_pct": opts_ctx.get("distance_to_put_wall_pct") if opts_ctx else None,
        "ticker_call_oi": opts_ctx.get("ticker_call_oi") if opts_ctx else None,
        "ticker_put_oi": opts_ctx.get("ticker_put_oi") if opts_ctx else None,
        "ticker_put_call_ratio": opts_ctx.get("ticker_put_call_ratio") if opts_ctx else None,
        "near_term_oi": opts_ctx.get("near_term_oi") if opts_ctx else None,
        "swing_dte_oi": opts_ctx.get("swing_dte_oi") if opts_ctx else None,
        "long_dated_oi": opts_ctx.get("long_dated_oi") if opts_ctx else None,
        "iv_rank": opts_ctx.get("iv_rank") if opts_ctx else None,
        "iv_current": opts_ctx.get("iv_current") if opts_ctx else None,
    }
    sc = price_signal.get("score_components")
    if sc:
        for k, v in sc.items():
            row[f"price_{k}"] = v
    if trade_plan:
        row["entry_price"] = trade_plan.get("entry_price")
        row["stop_price"] = trade_plan.get("stop_price")
        row["target_1"] = trade_plan.get("target_1")
        row["rr_ratio"] = trade_plan.get("rr_ratio")
    return row


def run_price_validation_for_bullish_candidates(
    bullish_df,
    signal_bar_offset: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Run long-side price validation. Returns (accepted, rejected)."""
    if bullish_df.empty:
        return [], []

    accepted: list[dict] = []
    rejected: list[dict] = []

    for _, row in bullish_df.iterrows():
        ticker = row["ticker"]
        flow_scaled = float(row["bullish_score"])
        flow_raw = float(row.get("bullish_score_raw", flow_scaled))

        if not has_strong_bullish_flow(row):
            _rej = _build_rejection_row(
                ticker, "LONG", flow_raw, {}, LONG_ALL_REASONS,
                reject_reason="weak_bullish_flow",
                flow_score_scaled=flow_scaled,
            )
            _attach_flow_components(_rej, row)
            rejected.append(_rej)
            continue

        try:
            df = fetch_ohlcv(ticker)
            df = clean_ohlcv(df)
            df = compute_features(df)

            should_reject, rej_reason, stub, counter_trend = quick_reject_check(df, "LONG")
            if should_reject:
                _rej = _build_rejection_row(
                    ticker, "LONG", flow_raw, stub, LONG_ALL_REASONS,
                    reject_reason=rej_reason, flow_score_scaled=flow_scaled,
                )
                _attach_flow_components(_rej, row)
                rejected.append(_rej)
                continue

            spot = float(df.iloc[-1]["close"])
            opts_ctx = fetch_options_context(ticker, spot)

            price_signal = score_long_setup(df, signal_bar_offset=signal_bar_offset)
            price_signal["ticker"] = ticker

            if not price_signal["is_valid"]:
                _rej = _build_rejection_row(
                    ticker, "LONG", flow_raw, price_signal, LONG_ALL_REASONS,
                    flow_score_scaled=flow_scaled, opts_ctx=opts_ctx,
                )
                _attach_flow_components(_rej, row)
                rejected.append(_rej)
                continue

            rs = _relative_strength(df)
            if rs is not None and rs < RS_LONG_MIN:
                _rej = _build_rejection_row(
                    ticker, "LONG", flow_raw, price_signal, LONG_ALL_REASONS,
                    reject_reason=f"weak_relative_strength ({rs:+.1%} vs SPY)",
                    flow_score_scaled=flow_scaled, opts_ctx=opts_ctx,
                )
                _attach_flow_components(_rej, row)
                rejected.append(_rej)
                continue

            atr = float(df.iloc[-1]["atr14"])
            call_wall = opts_ctx.get("nearest_call_wall")
            if call_wall and call_wall > spot and atr > 0:
                wall_dist_pct = (call_wall - spot) / spot
                wall_dist_atr = (call_wall - spot) / atr
                if wall_dist_pct < WALL_PROXIMITY_REJECT_PCT and wall_dist_atr < WALL_PROXIMITY_REJECT_ATR:
                    _rej = _build_rejection_row(
                        ticker, "LONG", flow_raw, price_signal, LONG_ALL_REASONS,
                        reject_reason=f"wall_proximity ({wall_dist_pct:.1%}, {wall_dist_atr:.1f} ATR to call wall)",
                        flow_score_scaled=flow_scaled, opts_ctx=opts_ctx,
                    )
                    _attach_flow_components(_rej, row)
                    rejected.append(_rej)
                    continue

            trade_plan = build_long_trade_plan(df, price_signal, options_ctx=opts_ctx, signal_bar_offset=signal_bar_offset)

            if trade_plan["rr_ratio"] < MIN_RR:
                _rej = _build_rejection_row(
                    ticker, "LONG", flow_raw, price_signal, LONG_ALL_REASONS,
                    reject_reason=f"poor_rr ({trade_plan['rr_ratio']:.1f}:1)",
                    flow_score_scaled=flow_scaled, opts_ctx=opts_ctx,
                    trade_plan=trade_plan,
                )
                _attach_flow_components(_rej, row)
                rejected.append(_rej)
                continue

            _opts_score = compute_options_context_score("LONG", opts_ctx)
            _row = {
                "ticker": ticker,
                "direction": "LONG",
                "flow_score_raw": flow_raw,
                "flow_score_scaled": flow_scaled,
                "price_score": float(price_signal["score"]),
                "final_score": combine_scores(flow_scaled, float(price_signal["score"]), _opts_score),
                "options_context_score": _opts_score,
                "entry_price": trade_plan["entry_price"],
                "stop_price": trade_plan["stop_price"],
                "target_1": trade_plan["target_1"],
                "target_2": trade_plan["target_2"],
                "rr_ratio": trade_plan["rr_ratio"],
                "time_stop_days": trade_plan["time_stop_days"],
                "checks_passed": ", ".join(sorted(price_signal.get("checks_passed", []))) or "none",
                "checks_failed": ", ".join(sorted(price_signal.get("checks_failed", []))) or "none",
                "pattern": _extract_pattern(price_signal.get("reasons", [])),
                "gamma_regime": opts_ctx.get("gamma_regime"),
                "net_gex": opts_ctx.get("net_gex"),
                "gamma_flip_level_estimate": opts_ctx.get("gamma_flip_level_estimate"),
                "nearest_call_wall": opts_ctx.get("nearest_call_wall"),
                "nearest_put_wall": opts_ctx.get("nearest_put_wall"),
                "distance_to_call_wall_pct": opts_ctx.get("distance_to_call_wall_pct"),
                "distance_to_put_wall_pct": opts_ctx.get("distance_to_put_wall_pct"),
                "ticker_call_oi": opts_ctx.get("ticker_call_oi"),
                "ticker_put_oi": opts_ctx.get("ticker_put_oi"),
                "ticker_put_call_ratio": opts_ctx.get("ticker_put_call_ratio"),
                "near_term_oi": opts_ctx.get("near_term_oi"),
                "swing_dte_oi": opts_ctx.get("swing_dte_oi"),
                "long_dated_oi": opts_ctx.get("long_dated_oi"),
                "iv_rank": opts_ctx.get("iv_rank"),
                "iv_current": opts_ctx.get("iv_current"),
                "counter_trend": counter_trend,
                "source": "fresh",
                "trade_plan": trade_plan,
                "flow_snapshot": row.to_dict(),
                "price_snapshot": price_signal,
                "options_context": opts_ctx,
            }
            sc = price_signal.get("score_components")
            if sc:
                for k, v in sc.items():
                    _row[f"price_{k}"] = v
            _attach_flow_components(_row, row)
            accepted.append(_row)

        except Exception as e:
            _rej = _build_rejection_row(
                ticker, "LONG", flow_raw, {}, LONG_ALL_REASONS,
                reject_reason=f"error: {e}",
                flow_score_scaled=flow_scaled,
            )
            _attach_flow_components(_rej, row)
            rejected.append(_rej)

    return accepted, rejected


def run_price_validation_for_bearish_candidates(
    bearish_df,
    signal_bar_offset: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Run short-side price validation. Returns (accepted, rejected)."""
    if bearish_df.empty:
        return [], []

    accepted: list[dict] = []
    rejected: list[dict] = []

    for _, row in bearish_df.iterrows():
        ticker = row["ticker"]
        flow_scaled = float(row["bearish_score"])
        flow_raw = float(row.get("bearish_score_raw", flow_scaled))

        if not has_strong_bearish_flow(row):
            _rej = _build_rejection_row(
                ticker, "SHORT", flow_raw, {}, SHORT_ALL_REASONS,
                reject_reason="weak_bearish_flow",
                flow_score_scaled=flow_scaled,
            )
            _attach_flow_components(_rej, row)
            rejected.append(_rej)
            continue

        try:
            df = fetch_ohlcv(ticker)
            df = clean_ohlcv(df)
            df = compute_features(df)

            should_reject, rej_reason, stub, counter_trend = quick_reject_check(df, "SHORT")
            if should_reject:
                _rej = _build_rejection_row(
                    ticker, "SHORT", flow_raw, stub, SHORT_ALL_REASONS,
                    reject_reason=rej_reason, flow_score_scaled=flow_scaled,
                )
                _attach_flow_components(_rej, row)
                rejected.append(_rej)
                continue

            spot = float(df.iloc[-1]["close"])
            opts_ctx = fetch_options_context(ticker, spot)

            price_signal = score_short_setup(df, signal_bar_offset=signal_bar_offset)
            price_signal["ticker"] = ticker

            if not price_signal["is_valid"]:
                _rej = _build_rejection_row(
                    ticker, "SHORT", flow_raw, price_signal, SHORT_ALL_REASONS,
                    flow_score_scaled=flow_scaled, opts_ctx=opts_ctx,
                )
                _attach_flow_components(_rej, row)
                rejected.append(_rej)
                continue

            rs = _relative_strength(df)
            rs_demote = rs is not None and rs > RS_SHORT_MAX

            atr = float(df.iloc[-1]["atr14"])
            put_wall = opts_ctx.get("nearest_put_wall")
            if put_wall and put_wall < spot and atr > 0:
                wall_dist_pct = (spot - put_wall) / spot
                wall_dist_atr = (spot - put_wall) / atr
                if wall_dist_pct < WALL_PROXIMITY_REJECT_PCT and wall_dist_atr < WALL_PROXIMITY_REJECT_ATR:
                    _rej = _build_rejection_row(
                        ticker, "SHORT", flow_raw, price_signal, SHORT_ALL_REASONS,
                        reject_reason=f"wall_proximity ({wall_dist_pct:.1%}, {wall_dist_atr:.1f} ATR to put wall)",
                        flow_score_scaled=flow_scaled, opts_ctx=opts_ctx,
                    )
                    _attach_flow_components(_rej, row)
                    rejected.append(_rej)
                    continue

            trade_plan = build_short_trade_plan(df, price_signal, options_ctx=opts_ctx, signal_bar_offset=signal_bar_offset)

            if trade_plan["rr_ratio"] < MIN_RR:
                _rej = _build_rejection_row(
                    ticker, "SHORT", flow_raw, price_signal, SHORT_ALL_REASONS,
                    reject_reason=f"poor_rr ({trade_plan['rr_ratio']:.1f}:1)",
                    flow_score_scaled=flow_scaled, opts_ctx=opts_ctx,
                    trade_plan=trade_plan,
                )
                _attach_flow_components(_rej, row)
                rejected.append(_rej)
                continue

            _opts_score = compute_options_context_score("SHORT", opts_ctx)
            _row = {
                "ticker": ticker,
                "direction": "SHORT",
                "flow_score_raw": flow_raw,
                "flow_score_scaled": flow_scaled,
                "price_score": float(price_signal["score"]),
                "final_score": combine_scores(flow_scaled, float(price_signal["score"]), _opts_score),
                "options_context_score": _opts_score,
                "entry_price": trade_plan["entry_price"],
                "stop_price": trade_plan["stop_price"],
                "target_1": trade_plan["target_1"],
                "target_2": trade_plan["target_2"],
                "rr_ratio": trade_plan["rr_ratio"],
                "time_stop_days": trade_plan["time_stop_days"],
                "checks_passed": ", ".join(sorted(price_signal.get("checks_passed", []))) or "none",
                "checks_failed": ", ".join(sorted(price_signal.get("checks_failed", []))) or "none",
                "pattern": _extract_pattern(price_signal.get("reasons", [])),
                "gamma_regime": opts_ctx.get("gamma_regime"),
                "net_gex": opts_ctx.get("net_gex"),
                "gamma_flip_level_estimate": opts_ctx.get("gamma_flip_level_estimate"),
                "nearest_call_wall": opts_ctx.get("nearest_call_wall"),
                "nearest_put_wall": opts_ctx.get("nearest_put_wall"),
                "distance_to_call_wall_pct": opts_ctx.get("distance_to_call_wall_pct"),
                "distance_to_put_wall_pct": opts_ctx.get("distance_to_put_wall_pct"),
                "ticker_call_oi": opts_ctx.get("ticker_call_oi"),
                "ticker_put_oi": opts_ctx.get("ticker_put_oi"),
                "ticker_put_call_ratio": opts_ctx.get("ticker_put_call_ratio"),
                "near_term_oi": opts_ctx.get("near_term_oi"),
                "swing_dte_oi": opts_ctx.get("swing_dte_oi"),
                "long_dated_oi": opts_ctx.get("long_dated_oi"),
                "iv_rank": opts_ctx.get("iv_rank"),
                "iv_current": opts_ctx.get("iv_current"),
                "counter_trend": counter_trend,
                "source": "watchlist_rs_demote" if rs_demote else "fresh",
                "trade_plan": trade_plan,
                "flow_snapshot": row.to_dict(),
                "price_snapshot": price_signal,
                "options_context": opts_ctx,
            }
            sc = price_signal.get("score_components")
            if sc:
                for k, v in sc.items():
                    _row[f"price_{k}"] = v
            _attach_flow_components(_row, row)
            accepted.append(_row)

        except Exception as e:
            _rej = _build_rejection_row(
                ticker, "SHORT", flow_raw, {}, SHORT_ALL_REASONS,
                reject_reason=f"error: {e}",
                flow_score_scaled=flow_scaled,
            )
            _attach_flow_components(_rej, row)
            rejected.append(_rej)

    return accepted, rejected


AGG_PREMIUM_WEIGHT = 0.06
AGG_VOLUME_WEIGHT = 0.05


def _enrich_agg_options(results: list[dict]) -> list[dict]:
    """Add a small additive bonus from aggregated options context to flow scores.

    Only fetches for tickers already in results (cached per-run, so typically free).
    When data is unavailable, defaults to 0.5 (neutral contribution).
    """
    for r in results:
        ticker = r["ticker"]
        spot = r.get("entry_price", 0.0)
        try:
            ctx = fetch_options_context(ticker, spot)
        except Exception:
            ctx = {}

        bullish_p = ctx.get("daily_bullish_premium")
        bearish_p = ctx.get("daily_bearish_premium")
        if bullish_p is not None and bearish_p is not None:
            total = bullish_p + bearish_p
            if total > 0:
                raw_align = bullish_p / total if r["direction"] == "LONG" else bearish_p / total
            else:
                raw_align = 0.5
        else:
            raw_align = 0.5

        call_vs_avg = ctx.get("call_volume_vs_30d_avg")
        put_vs_avg = ctx.get("put_volume_vs_30d_avg")
        if r["direction"] == "LONG" and call_vs_avg is not None:
            raw_vol = min(call_vs_avg, 3.0) / 3.0
        elif r["direction"] == "SHORT" and put_vs_avg is not None:
            raw_vol = min(put_vs_avg, 3.0) / 3.0
        else:
            raw_vol = 0.5

        bonus = raw_align * AGG_PREMIUM_WEIGHT + raw_vol * AGG_VOLUME_WEIGHT
        r["agg_premium_alignment"] = round(raw_align, 4)
        r["agg_volume_unusualness"] = round(raw_vol, 4)
        r["final_score"] = round(r["final_score"] + bonus, 4)

    return results


NET_PREM_WEIGHT = 0.08
DARK_POOL_WEIGHT = 0.05


def _enrich_net_prem_ticks(results: list[dict]) -> list[dict]:
    """Enrich final signals with intraday net premium tick data.

    Adds a small bonus when intraday premium flow confirms the trade thesis.
    """
    for r in results:
        npt = fetch_net_prem_ticks(r["ticker"])
        if npt is None:
            r["intraday_premium_direction"] = None
            r["delta_momentum"] = None
            continue

        r["intraday_premium_direction"] = npt["intraday_premium_direction"]
        r["delta_momentum"] = npt["delta_momentum"]
        r["net_delta"] = npt["net_delta"]

        # Alignment: is intraday premium flowing in the direction of our thesis?
        prem_dir = npt["intraday_premium_direction"]
        if r["direction"] == "LONG":
            alignment = prem_dir
        else:
            alignment = 1.0 - prem_dir

        delta_factor = max(0.0, npt["delta_momentum"]) if r["direction"] == "LONG" else max(0.0, -npt["delta_momentum"])
        bonus = NET_PREM_WEIGHT * (0.7 * alignment + 0.3 * delta_factor)
        r["final_score"] = round(r["final_score"] + bonus, 4)

    return results


def _enrich_dark_pool(results: list[dict]) -> list[dict]:
    """Enrich final signals with dark pool institutional activity data."""
    for r in results:
        dp = fetch_dark_pool(r["ticker"])
        if dp is None:
            r["dark_pool_bias"] = None
            r["dark_pool_volume"] = None
            continue

        r["dark_pool_bias"] = dp["dark_pool_bias"]
        r["dark_pool_volume"] = dp["dark_pool_volume"]
        r["large_print_count"] = dp["large_print_count"]

        # Alignment: does institutional dark pool activity match our thesis?
        if r["direction"] == "LONG":
            alignment = dp["dark_pool_bias"]
        else:
            alignment = 1.0 - dp["dark_pool_bias"]

        bonus = DARK_POOL_WEIGHT * alignment
        r["final_score"] = round(r["final_score"] + bonus, 4)

    return results


EARNINGS_WEIGHT = 0.5  # matches EARNINGS_HOLD_PENALTY from config


def _enrich_earnings(results: list[dict]) -> list[dict]:
    """Enrich final signals with earnings proximity data.

    Attaches next earnings date and applies a score penalty when earnings
    fall within the hold window (binary event risk).
    """
    from app.config import MAX_HOLD_DAYS

    for r in results:
        er = fetch_earnings(r["ticker"])
        if er is None:
            r["earnings_date"] = None
            r["days_until_earnings"] = None
            continue

        r["earnings_date"] = er["next_earnings_date"]
        r["days_until_earnings"] = er["days_until_earnings"]
        r["last_eps_surprise"] = er["last_eps_surprise"]

        if er["days_until_earnings"] is not None and er["days_until_earnings"] <= MAX_HOLD_DAYS:
            r["final_score"] = round(r["final_score"] - EARNINGS_WEIGHT, 4)
            r["earnings_imminent"] = True
        else:
            r["earnings_imminent"] = False

    return results


def _enrich_insider(results: list[dict]) -> list[dict]:
    """Enrich final signals with insider transaction alignment.

    Applies a small score bonus when insider buying aligns with trade direction.
    """
    from app.config import INSIDER_BUY_BONUS
    from app.features.insider_tracker import classify_insider_activity

    insider_path = DATA_ROOT / "insider_recent.json"
    if not insider_path.is_file():
        for r in results:
            r["insider_direction"] = None
        return results

    import json
    try:
        raw = json.loads(insider_path.read_text())
    except Exception:
        for r in results:
            r["insider_direction"] = None
        return results

    insider_by_ticker = classify_insider_activity(raw)

    for r in results:
        ins = insider_by_ticker.get(r["ticker"])
        if ins is None:
            r["insider_direction"] = None
            continue

        r["insider_direction"] = ins["net_direction"]
        r["insider_buy_count"] = ins["buy_count"]
        r["insider_sell_count"] = ins["sell_count"]
        r["insider_buy_notional"] = ins["buy_notional"]

        aligned = (
            (r["direction"] == "LONG" and ins["net_direction"] == "buying")
            or (r["direction"] == "SHORT" and ins["net_direction"] == "selling")
        )
        if aligned and ins["buy_notional"] >= 50_000:
            r["final_score"] = round(r["final_score"] + INSIDER_BUY_BONUS, 4)
            r["insider_aligned"] = True
        else:
            r["insider_aligned"] = False

    return results


def _run_options_agent_shadow(results: list[dict]) -> None:
    """Run the Options Context Agent in shadow mode for each final result.

    Logs assessments to ``data/agent_shadow/options_context/`` for offline
    analysis.  Does NOT modify any scores or results.  Silently skips if
    the agent is unavailable (no OpenAI key).
    """
    from app.agents.options_context import is_agent_available, run_options_context_shadow

    if not is_agent_available():
        return

    n_run = 0
    for r in results:
        ticker = r["ticker"]
        direction = r["direction"]
        price = r.get("entry_price", 0.0)
        opts_ctx = r.get("options_context") or {}
        flow_snapshot = r.get("flow_snapshot")

        dp: dict | None = None
        if r.get("dark_pool_bias") is not None:
            dp = {
                "dark_pool_bias": r.get("dark_pool_bias"),
                "dark_pool_volume": r.get("dark_pool_volume"),
                "large_print_count": r.get("large_print_count"),
            }

        npt: dict | None = None
        if r.get("intraday_premium_direction") is not None:
            npt = {
                "intraday_premium_direction": r.get("intraday_premium_direction"),
                "delta_momentum": r.get("delta_momentum"),
                "net_delta": r.get("net_delta"),
            }

        atr_val = None
        tp = r.get("trade_plan")
        if tp and tp.get("risk_per_share"):
            atr_val = tp.get("risk_per_share")

        assessment = run_options_context_shadow(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr_val,
            flow_row=flow_snapshot,
            opts_ctx=opts_ctx,
            dark_pool=dp,
            net_prem_ticks=npt,
        )
        if assessment is not None:
            n_run += 1
            r["agent_options_conviction"] = assessment.directional_conviction
            r["agent_hedging_probability"] = assessment.hedging_probability
            r["agent_signal_consistency"] = assessment.signal_consistency

    if n_run > 0:
        print(f"  [agent:options_context] shadow assessed {n_run} candidates")


def _run_sr_quality_agent_shadow(results: list[dict]) -> None:
    """Run the S/R Quality Agent in shadow mode for each final result."""
    from app.agents.sr_quality import is_agent_available, run_sr_quality_shadow

    if not is_agent_available():
        return

    n_run = 0
    for r in results:
        ticker = r["ticker"]
        direction = r["direction"]
        price = r.get("entry_price", 0.0)
        price_snapshot = r.get("price_snapshot") or {}
        trade_plan = r.get("trade_plan") or {}

        atr_val = None
        if trade_plan.get("risk_per_share"):
            atr_val = trade_plan["risk_per_share"]

        assessment = run_sr_quality_shadow(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr_val,
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
        )
        if assessment is not None:
            n_run += 1
            r["agent_sr_quality"] = assessment.overall_sr_quality
            r["agent_sr_key_level_quality"] = assessment.key_level_quality

    if n_run > 0:
        print(f"  [agent:sr_quality] shadow assessed {n_run} candidates")


def _run_trade_plan_agent_shadow(results: list[dict]) -> None:
    """Run the Trade Plan Agent in shadow mode for each final result."""
    from app.agents.trade_plan import is_agent_available, run_trade_plan_shadow

    if not is_agent_available():
        return

    n_run = 0
    for r in results:
        ticker = r["ticker"]
        direction = r["direction"]
        price = r.get("entry_price", 0.0)
        price_snapshot = r.get("price_snapshot") or {}
        trade_plan = r.get("trade_plan") or {}
        opts_ctx = r.get("options_context") or {}

        atr_val = None
        if trade_plan.get("risk_per_share"):
            atr_val = trade_plan["risk_per_share"]

        assessment = run_trade_plan_shadow(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr_val,
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
            opts_ctx=opts_ctx,
        )
        if assessment is not None:
            n_run += 1
            r["agent_plan_score"] = assessment.plan_score
            r["agent_rr_assessment"] = assessment.rr_assessment
            r["agent_stop_quality"] = assessment.stop_quality

    if n_run > 0:
        print(f"  [agent:trade_plan] shadow assessed {n_run} candidates")


def _run_entry_timing_agent_shadow(results: list[dict]) -> None:
    """Run the Entry/Timing Agent in shadow mode for each final result."""
    from app.agents.entry_timing import is_agent_available, run_entry_timing_shadow

    if not is_agent_available():
        return

    n_run = 0
    for r in results:
        ticker = r["ticker"]
        direction = r["direction"]
        price = r.get("entry_price", 0.0)
        price_snapshot = r.get("price_snapshot") or {}
        trade_plan = r.get("trade_plan") or {}
        flow_sc = r.get("flow_score_scaled")

        atr_val = None
        if trade_plan.get("risk_per_share"):
            atr_val = trade_plan["risk_per_share"]

        assessment = run_entry_timing_shadow(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr_val,
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
            flow_score=flow_sc,
        )
        if assessment is not None:
            n_run += 1
            r["agent_entry_score"] = assessment.entry_score
            r["agent_chasing_risk"] = assessment.chasing_risk
            r["agent_entry_timing"] = assessment.entry_timing

    if n_run > 0:
        print(f"  [agent:entry_timing] shadow assessed {n_run} candidates")


def _run_devils_advocate_agent_shadow(results: list[dict]) -> None:
    """Run the Devil's Advocate Agent in shadow mode for each final result."""
    from app.agents.devils_advocate import is_agent_available, run_devils_advocate_shadow

    if not is_agent_available():
        return

    n_run = 0
    for r in results:
        ticker = r["ticker"]
        direction = r["direction"]
        price = r.get("entry_price", 0.0)
        trade_plan = r.get("trade_plan") or {}
        price_snapshot = r.get("price_snapshot") or {}
        flow_snapshot = r.get("flow_snapshot")
        opts_ctx = r.get("options_context") or {}

        atr_val = None
        if trade_plan.get("risk_per_share"):
            atr_val = trade_plan["risk_per_share"]

        assessment = run_devils_advocate_shadow(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr_val,
            final_score=r.get("final_score"),
            flow_score=r.get("flow_score_scaled"),
            price_score=r.get("price_score"),
            options_score=r.get("options_context_score"),
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
            flow_snapshot=flow_snapshot,
            opts_ctx=opts_ctx,
            counter_trend=bool(r.get("counter_trend")),
            sector=r.get("sector"),
        )
        if assessment is not None:
            n_run += 1
            r["agent_risk_score"] = assessment.risk_score
            r["agent_earnings_risk"] = assessment.earnings_risk
            r["agent_kill_reasons"] = assessment.kill_reasons

    if n_run > 0:
        print(f"  [agent:devils_advocate] shadow assessed {n_run} candidates")


def _run_orchestrator_shadow(results: list[dict]) -> None:
    """Run the V3 deterministic orchestrator in shadow mode.

    Reconstructs all 5 agent schema objects from shadow fields on each result
    dict and passes them to ``compute_agent_conviction()``.  Logs conviction
    alongside the deterministic ``final_score`` for offline comparison.
    Does NOT alter any scores used for actual decisions.
    """
    from app.agents.orchestrator import compute_agent_conviction

    shadow_dir = Path(__file__).resolve().parents[2] / "data" / "agent_shadow" / "orchestrator"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    _AGENT_KEYS = [
        "agent_options_conviction", "agent_sr_quality",
        "agent_plan_score", "agent_entry_score", "agent_risk_score",
    ]
    has_any_agent = any(
        any(r.get(k) is not None for k in _AGENT_KEYS) for r in results
    )
    if not has_any_agent:
        return

    import json
    from datetime import datetime as _dt
    from app.agents.schemas import (
        DevilsAdvocateAssessment,
        EntryTimingAssessment,
        LevelAssessment,
        OptionsContextAssessment,
        SRQualityOutput,
        TradePlanAssessment,
    )

    def _rebuild_oc(r: dict) -> OptionsContextAssessment | None:
        if r.get("agent_options_conviction") is None:
            return None
        return OptionsContextAssessment(
            directional_conviction=r["agent_options_conviction"],
            hedging_probability=r.get("agent_hedging_probability", 0.3),
            hedging_reasoning="from shadow",
            signal_consistency=r.get("agent_signal_consistency", "mixed"),
            consistency_detail="from shadow",
            gamma_significance="medium", gamma_reasoning="from shadow",
            wall_impact="neutral", wall_reasoning="from shadow",
            iv_assessment="fair", iv_reasoning="from shadow",
            institutional_confidence="medium", institutional_reasoning="from shadow",
            dark_pool_alignment="no_data", intraday_flow_alignment="no_data",
            key_concern="none", reasoning="reconstructed from shadow fields",
        )

    def _rebuild_sr(r: dict) -> SRQualityOutput | None:
        if r.get("agent_sr_quality") is None:
            return None
        _la = LevelAssessment(
            level_price=0, source="algo", touch_count=0, quality="tactical",
            confidence=0.5, reasoning="shadow stub", volume_confirmed=False,
            clean_rejections=False, recently_violated=False,
        )
        return SRQualityOutput(
            support_assessment=_la, resistance_assessment=_la,
            structural_support_assessment=_la, structural_resistance_assessment=_la,
            overall_sr_quality=r["agent_sr_quality"],
            key_level_for_trade=0, invalidation_level=0,
            key_level_quality=r.get("agent_sr_key_level_quality", "tactical"),
            key_level_source="algo",
            reasoning="reconstructed from shadow fields",
        )

    def _rebuild_tp(r: dict) -> TradePlanAssessment | None:
        if r.get("agent_plan_score") is None:
            return None
        return TradePlanAssessment(
            stop_quality=r.get("agent_stop_quality", "good"),
            stop_reasoning="from shadow",
            t1_quality="good", t1_reasoning="from shadow",
            t2_quality="good",
            rr_assessment=r.get("agent_rr_assessment", "favorable"),
            true_rr_estimate=2.5, hold_time_suggestion=10,
            hold_reasoning="from shadow", partial_at_t1_pct=0.5,
            plan_score=r["agent_plan_score"],
            reasoning="reconstructed from shadow fields",
        )

    def _rebuild_et(r: dict) -> EntryTimingAssessment | None:
        if r.get("agent_entry_score") is None:
            return None
        return EntryTimingAssessment(
            entry_timing=r.get("agent_entry_timing", "enter_now"),
            confidence=0.7,
            chasing_risk=r.get("agent_chasing_risk", "low"),
            chasing_reasoning="from shadow",
            gap_risk="low", gap_reasoning="from shadow",
            bar_quality="acceptable", bar_reasoning="from shadow",
            entry_score=r["agent_entry_score"],
            reasoning="reconstructed from shadow fields",
        )

    def _rebuild_da(r: dict) -> DevilsAdvocateAssessment | None:
        if r.get("agent_risk_score") is None:
            return None
        return DevilsAdvocateAssessment(
            risk_score=r["agent_risk_score"],
            earnings_risk=r.get("agent_earnings_risk", "unknown"),
            earnings_detail="from shadow",
            trap_probability=0.3, trap_reasoning="from shadow",
            liquidity_concern="none", liquidity_reasoning="from shadow",
            concentration_risk="none", concentration_detail="from shadow",
            catalyst_type="unknown", catalyst_reasoning="from shadow",
            crowded_trade_risk="low",
            kill_reasons=r.get("agent_kill_reasons", []),
            reasoning="reconstructed from shadow fields",
        )

    n_run = 0
    rows: list[dict] = []

    for r in results:
        orch_result = compute_agent_conviction(
            sr=_rebuild_sr(r),
            tp=_rebuild_tp(r),
            oc=_rebuild_oc(r),
            et=_rebuild_et(r),
            da=_rebuild_da(r),
        )

        r["agent_conviction"] = orch_result.conviction
        r["agent_vetoed"] = orch_result.vetoed
        r["agent_veto_reason"] = orch_result.veto_reason
        r["agent_penalties"] = orch_result.penalties_applied
        n_run += 1

        rows.append({
            "ticker": r["ticker"],
            "direction": r["direction"],
            "final_score": r.get("final_score"),
            **orch_result.to_dict(),
        })

    if n_run > 0:
        stamp = _dt.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = shadow_dir / f"orchestrator_{stamp}.json"
        with open(log_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"  [orchestrator] shadow scored {n_run} candidates → {log_path.name}")


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _run_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_run_outputs(
    feature_table: pd.DataFrame,
    bullish_ranked: pd.DataFrame,
    bearish_ranked: pd.DataFrame,
    signals_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    stamp: str | None = None,
) -> dict[str, Path]:
    """Save all intermediate, final, and rejected DataFrames to CSV under data/."""
    stamp = stamp or _run_stamp()
    paths: dict[str, Path] = {}

    for name, df, subdir in [
        ("flow_features", feature_table, "flow_features"),
        ("ranked_bullish", bullish_ranked, "ranked_candidates"),
        ("ranked_bearish", bearish_ranked, "ranked_candidates"),
        ("final_signals", signals_df, "final_signals"),
        ("rejected", rejected_df, "final_signals"),
    ]:
        out_dir = DATA_ROOT / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}_{stamp}.csv"
        df.to_csv(path, index=False)
        paths[name] = path

    return paths


_FLOW_STAT_COLS = [
    "bullish_flow_intensity", "bearish_flow_intensity",
    "bullish_ppt_bps", "bearish_ppt_bps",
]
_FLOW_STAT_PCTS = [0.25, 0.50, 0.75, 0.90, 0.99]


def _log_flow_stats(feature_table: pd.DataFrame) -> None:
    """Append per-scan distribution percentiles to data/flow_stats.csv."""
    stats_path = DATA_ROOT / "flow_stats.csv"
    row: dict = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "ticker_count": len(feature_table),
    }
    for col in _FLOW_STAT_COLS:
        if col not in feature_table.columns:
            continue
        series = feature_table[col].dropna()
        if series.empty:
            continue
        for pct in _FLOW_STAT_PCTS:
            label = f"{col}_p{int(pct * 100)}"
            row[label] = round(float(series.quantile(pct)), 6)

    header = not stats_path.exists()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(stats_path, mode="a", header=header, index=False)


def apply_agent_filter(results: list[dict]) -> list[dict]:
    """Filter pipeline results using agent orchestrator decisions.

    Returns a new list suitable for ``open_positions(filtered, portfolio="agent")``.
    Vetoed signals are removed; remaining signals use ``agent_conviction`` for sizing.
    """
    filtered: list[dict] = []
    for r in results:
        if r.get("agent_vetoed"):
            continue

        sig = dict(r)
        conviction = sig.get("agent_conviction")
        if conviction is not None and conviction > 0:
            sig["final_score"] = conviction
        filtered.append(sig)

    return filtered


def run_flow_to_price_pipeline(
    flow_limit: int = 2000,
    top_n: int = 50,
    min_premium: float = 500_000,
    save: bool = True,
    use_uw_alerts: bool = True,
    alert_hours_back: int = 48,
) -> dict:
    """
    Full V1 pipeline:
    1. Load + prune persistent watchlist
    2. Pull raw UW flow + UW curated alert tickers
    3. Normalize + build flow features
    4. Rank bullish/bearish candidates
    5. Scale flow scores, run price validation
    6. Re-evaluate watchlist entries (frozen flow, fresh price)
    7. Merge fresh + promoted signals, dedupe
    8. Update watchlist with newly rejected, save to disk
    """
    from app.signals.watchlist import (
        add_candidates,
        load_watchlist,
        prune_expired,
        reevaluate_watchlist,
        save_watchlist,
    )

    clear_context_cache()
    clear_price_cache()
    clear_decision_cache()
    reset_api_stats()
    _SPY_RETURN_CACHE.clear()
    market_regime = fetch_market_regime()
    rs = market_regime["regime_score"]
    print(f"  [regime] score={rs:.2f}  SPY={market_regime['spy_trend']}  VIX={market_regime.get('vix_close', '?')}  sizing_mult={market_regime['vix_sizing_mult']:.2f}")
    regime_path = DATA_ROOT / "market_regime.json"
    regime_path.parent.mkdir(parents=True, exist_ok=True)
    regime_path.write_text(json.dumps(market_regime, default=str))

    prev_watchlist = load_watchlist()
    active_watchlist, expired_watchlist = prune_expired(prev_watchlist)

    try:
        payload = fetch_flow_raw(limit=flow_limit)
    except Exception as api_err:
        print(f"\n  [FATAL] Flow API call failed: {api_err}")
        print("  Skipping scan — will retry next scheduled run.")
        print_api_summary()
        return {
            "results": [],
            "ranked": {"bullish": pd.DataFrame(), "bearish": pd.DataFrame()},
            "rejected": [],
            "market_regime": market_regime,
            "watchlist": active_watchlist,
            "alert_stats": {"alert_tickers": 0, "new_tickers": 0},
            "screener_stats": {"screener_tickers": 0, "screener_new": 0},
        }

    normalized = normalize_flow_response(payload)

    if not normalized.empty:
        normalized = normalized[~normalized["ticker"].isin(NON_EQUITY_TICKERS)]

    existing_tickers = set(normalized["ticker"].unique()) if not normalized.empty else set()

    # Stock screener discovery: one API call, broad net of unusual activity
    screener_stats = {"screener_tickers": 0, "screener_new": 0}
    screener_meta: dict[str, dict] = {}
    try:
        screener_rows = fetch_stock_screener()

        # Persist screener snapshot for multi-day flow tracker
        from app.features.flow_tracker import save_screener_snapshot
        save_screener_snapshot(screener_rows)

        # Fetch social sentiment for tickers that qualify on the multi-day tracker
        try:
            from app.features.flow_tracker import compute_multi_day_flow
            from app.features.sentiment_tracker import fetch_and_save_sentiment
            qualified = compute_multi_day_flow()
            if qualified:
                fetch_and_save_sentiment([t["ticker"] for t in qualified])
        except Exception as se:
            print(f"  [sentiment] skipped: {se}")

        for sr in screener_rows:
            sym = (sr.get("ticker") or "").upper().strip()
            if sym and sym not in NON_EQUITY_TICKERS:
                screener_meta[sym] = sr
        screener_stats["screener_tickers"] = len(screener_meta)
        screener_new = [t for t in screener_meta if t not in existing_tickers]
        screener_stats["screener_new"] = len(screener_new)
        if screener_new:
            screener_flow = fetch_flow_for_tickers(screener_new, limit_per_ticker=30)
            if not screener_flow.empty:
                normalized = pd.concat([normalized, screener_flow], ignore_index=True)
                existing_tickers.update(screener_flow["ticker"].unique())
        print(f"  [screener] {screener_stats['screener_tickers']} tickers, {screener_stats['screener_new']} new")
    except Exception as e:
        print(f"  [screener] failed: {e}")

    # Market-wide dark pool prints — single API call, saved for dashboard tab
    try:
        dp_raw = fetch_dark_pool_recent(min_premium=100_000, limit=200)
        if dp_raw:
            dp_path = DATA_ROOT / "dark_pool_recent.json"
            dp_path.write_text(json.dumps(dp_raw, default=str))
            print(f"  [dark-pool] saved {len(dp_raw)} market-wide prints")

            from app.features.dark_pool_tracker import (
                accumulate_daily_prints,
                aggregate_dark_pool_prints,
                save_dp_snapshot,
            )

            # Accumulate prints across all intra-day scans (deduped by tracking_id)
            accumulate_daily_prints(dp_raw)

            # Persist daily dark pool snapshot for multi-day tracker
            dp_agg = aggregate_dark_pool_prints(dp_raw, screener_meta=screener_meta)
            save_dp_snapshot(dp_agg.get("by_ticker", []), screener_meta=screener_meta)
    except Exception as e:
        print(f"  [dark-pool] skipped: {e}")

    # Market-wide hottest option chains — single API call
    try:
        hc_raw = fetch_hottest_chains(min_premium=250_000, limit=100, max_dte=90)
        if hc_raw:
            hc_path = DATA_ROOT / "hottest_chains.json"
            hc_path.write_text(json.dumps(hc_raw, default=str))
            print(f"  [hottest-chains] saved {len(hc_raw)} contracts")

            # Extract unique tickers from hottest chains for batch earnings fetch
            hc_tickers = {
                (row.get("ticker_symbol") or row.get("ticker") or "").upper().strip()
                for row in hc_raw
            }
            hc_tickers.discard("")
    except Exception as e:
        hc_tickers = set()
        print(f"  [hottest-chains] skipped: {e}")

    # Market-wide insider transactions — single API call
    try:
        insider_raw = fetch_insider_transactions(limit=200)
        if insider_raw:
            insider_path = DATA_ROOT / "insider_recent.json"
            insider_path.write_text(json.dumps(insider_raw, default=str))
            print(f"  [insider] saved {len(insider_raw)} transactions")
    except Exception as e:
        print(f"  [insider] skipped: {e}")

    # Batch earnings fetch — covers screener tickers + hottest chains tickers
    earnings_tickers: set[str] = set(existing_tickers)
    try:
        earnings_tickers |= hc_tickers
    except NameError:
        pass

    try:
        earnings_cache: dict[str, dict | None] = {}
        for ticker in sorted(earnings_tickers)[:80]:
            er = fetch_earnings(ticker)
            if er and er.get("next_earnings_date"):
                earnings_cache[ticker] = er
        if earnings_cache:
            er_path = DATA_ROOT / "earnings_cache.json"
            er_path.write_text(json.dumps(earnings_cache, default=str))
            print(f"  [earnings] cached {len(earnings_cache)} tickers with upcoming earnings")
    except Exception as e:
        print(f"  [earnings] batch fetch skipped: {e}")

    # UW alert discovery
    alert_stats = {"alert_tickers": 0, "new_tickers": 0}
    if use_uw_alerts:
        alert_tickers = [t for t in fetch_uw_alerts(hours_back=alert_hours_back)
                         if t not in NON_EQUITY_TICKERS]
        alert_stats["alert_tickers"] = len(alert_tickers)

        new_tickers = [t for t in alert_tickers if t not in existing_tickers]
        alert_stats["new_tickers"] = len(new_tickers)

        if new_tickers:
            alert_flow = fetch_flow_for_tickers(new_tickers)
            if not alert_flow.empty:
                normalized = pd.concat([normalized, alert_flow], ignore_index=True)

    if save and not normalized.empty:
        raw_flow_dir = DATA_ROOT / "raw_flow"
        raw_flow_dir.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(raw_flow_dir / f"raw_flow_{_run_stamp()}.csv", index=False)

    feature_table = build_flow_feature_table(normalized, min_premium=min_premium)

    # Merge flow-feature tickers into screener snapshots so the Flow Tracker
    # sees all tickers with unusual flow, not just those from the UW screener.
    try:
        from app.features.flow_tracker import save_flow_feature_snapshot
        save_flow_feature_snapshot(feature_table)
    except Exception as e:
        print(f"  [flow-tracker] flow-feature merge failed: {e}")

    ranked = rank_flow_candidates(feature_table, top_n=top_n)

    if not ranked["bullish"].empty:
        ranked["bullish"]["bullish_score_raw"] = ranked["bullish"]["bullish_score"]
        ranked["bullish"]["bullish_score"] = ranked["bullish"]["bullish_score_raw"] * 10
    if not ranked["bearish"].empty:
        ranked["bearish"]["bearish_score_raw"] = ranked["bearish"]["bearish_score"]
        ranked["bearish"]["bearish_score"] = ranked["bearish"]["bearish_score_raw"] * 10

    bar_offset = 1 if _is_market_hours() else 0
    if bar_offset:
        print("  [scoring] intraday mode: candle/volume checks use previous completed bar")

    bull_accepted, bull_rejected = run_price_validation_for_bullish_candidates(ranked["bullish"], signal_bar_offset=bar_offset)
    bear_accepted, bear_rejected = run_price_validation_for_bearish_candidates(ranked["bearish"], signal_bar_offset=bar_offset)

    ranked["bullish"] = _attach_price_checks_to_ranked(ranked["bullish"], bull_accepted, bull_rejected)
    ranked["bearish"] = _attach_price_checks_to_ranked(ranked["bearish"], bear_accepted, bear_rejected)

    fresh_results = bull_accepted + bear_accepted
    all_rejected = bull_rejected + bear_rejected

    fresh_tickers = {(r["ticker"], r["direction"]) for r in fresh_results + all_rejected}
    promoted, still_watching, watch_rejected = reevaluate_watchlist(
        active_watchlist, fresh_tickers=fresh_tickers,
        signal_bar_offset=bar_offset,
        flow_features=feature_table,
    )

    persistence = compute_persistence()
    trajectory = compute_intraday_trajectory()

    final_results = fresh_results + promoted
    final_results = dedupe_final_results(final_results)
    final_results = apply_persistence_bonus(final_results, persistence)
    final_results = apply_trajectory_bonus(final_results, trajectory)

    regime_score = market_regime["regime_score"]
    for r in final_results:
        r["regime_score"] = regime_score
        r["vix_sizing_mult"] = market_regime["vix_sizing_mult"]
        sm = screener_meta.get(r["ticker"])
        if sm:
            r["iv_rank"] = sm.get("iv_rank")
            r["sector"] = sm.get("sector")

    # Directional threshold boost: counter-regime trades need higher conviction
    long_min = MIN_FINAL_SCORE + (1.0 - regime_score) * REGIME_THRESHOLD_BOOST
    short_min = MIN_FINAL_SCORE + regime_score * REGIME_THRESHOLD_BOOST
    final_results = [
        r for r in final_results
        if r["final_score"] >= (
            (long_min if r["direction"] == "LONG" else short_min)
            + (COUNTER_TREND_PREMIUM if r.get("counter_trend") else 0)
        )
    ]

    # Post-ranking enrichment: add aggregated options premium/volume bonus
    final_results = _enrich_agg_options(final_results)
    final_results = _enrich_net_prem_ticks(final_results)
    final_results = _enrich_dark_pool(final_results)
    final_results = _enrich_earnings(final_results)
    final_results = _enrich_insider(final_results)

    # Decision-context enrichment (liquidity, session, RS, expression, R-at-market).
    # Runs last so it has access to every prior field (iv_rank, earnings, entry/stop).
    final_results = enrich_decision_context(final_results)

    # V3 agent shadow: all 5 agents + orchestrator (logged, does NOT modify scores)
    _run_options_agent_shadow(final_results)
    _run_sr_quality_agent_shadow(final_results)
    _run_trade_plan_agent_shadow(final_results)
    _run_entry_timing_agent_shadow(final_results)
    _run_devils_advocate_agent_shadow(final_results)
    _run_orchestrator_shadow(final_results)

    final_results = sorted(final_results, key=lambda x: x["final_score"], reverse=True)
    final_results = apply_directional_balance(final_results)

    signals_df = results_to_dataframe(final_results)

    all_rejected_combined = all_rejected + watch_rejected
    rejected_df = pd.DataFrame(all_rejected_combined) if all_rejected_combined else pd.DataFrame(
        columns=REJECTED_COLUMNS
    )

    updated_watchlist = add_candidates(still_watching, all_rejected)
    save_watchlist(updated_watchlist)

    saved_paths = {}
    if save:
        saved_paths = save_run_outputs(
            feature_table=feature_table,
            bullish_ranked=ranked["bullish"],
            bearish_ranked=ranked["bearish"],
            signals_df=signals_df,
            rejected_df=rejected_df,
        )
        for name, path in saved_paths.items():
            print(f"  saved {name} -> {path}")
        if not feature_table.empty:
            _log_flow_stats(feature_table)

        try:
            from app.analytics.grade_backtest import refresh_grade_stats
            refresh_grade_stats()
            print("  [grade-backtest] refreshed data/grade_stats.json")
        except Exception as e:
            print(f"  [grade-backtest] skipped: {e}")

    print_api_summary()

    return {
        "results": final_results,
        "signals_df": signals_df,
        "rejected_df": rejected_df,
        "feature_table": feature_table,
        "ranked_bullish": ranked["bullish"],
        "ranked_bearish": ranked["bearish"],
        "saved_paths": saved_paths,
        "watchlist": {
            "previous_count": len(prev_watchlist),
            "expired_count": len(expired_watchlist),
            "promoted_count": len(promoted),
            "still_watching_count": len(still_watching),
            "new_rejects_added": len(all_rejected),
            "current_count": len(updated_watchlist),
        },
        "alert_stats": alert_stats,
        "market_regime": market_regime,
    }


REJECTED_COLUMNS = [
    "ticker",
    "direction",
    "flow_score_raw",
    "flow_score_scaled",
    "price_score",
    "final_score",
    "options_context_score",
    "gamma_regime",
    "net_gex",
    "gamma_flip_level_estimate",
    "nearest_call_wall",
    "nearest_put_wall",
    "distance_to_call_wall_pct",
    "distance_to_put_wall_pct",
    "ticker_call_oi",
    "ticker_put_oi",
    "ticker_put_call_ratio",
    "near_term_oi",
    "swing_dte_oi",
    "long_dated_oi",
    "iv_rank",
    "iv_current",
    "pattern",
    "reject_reason",
    "checks_passed",
    "checks_failed",
    "entry_price",
    "stop_price",
    "target_1",
    "rr_ratio",
]

SIGNAL_COLUMNS = [
    "ticker",
    "direction",
    "pattern",
    "flow_score_raw",
    "flow_score_scaled",
    "price_score",
    "final_score",
    "options_context_score",
    "gamma_regime",
    "net_gex",
    "gamma_flip_level_estimate",
    "nearest_call_wall",
    "nearest_put_wall",
    "distance_to_call_wall_pct",
    "distance_to_put_wall_pct",
    "ticker_call_oi",
    "ticker_put_oi",
    "ticker_put_call_ratio",
    "near_term_oi",
    "swing_dte_oi",
    "long_dated_oi",
    "iv_rank",
    "iv_current",
    "entry_price",
    "stop_price",
    "target_1",
    "target_2",
    "rr_ratio",
    "time_stop_days",
    "regime_score",
    "agg_premium_alignment",
    "agg_volume_unusualness",
    "source",
    "checks_passed",
    "checks_failed",
]


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert pipeline results into a clean signals DataFrame."""
    if not results:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    df = pd.DataFrame(results)
    for col in SIGNAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[SIGNAL_COLUMNS]
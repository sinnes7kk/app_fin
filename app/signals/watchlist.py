"""Persistent watchlist for rejected pipeline candidates.

Candidates that pass flow validation but fail price validation are stored in
a JSON file.  Each subsequent pipeline run re-evaluates their price setup with
frozen flow scores and promotes any that now pass.  Entries expire after
WATCHLIST_TTL_DAYS calendar days.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from app.signals.pipeline import _FLOW_COMPONENT_KEYS
from app.config import (
    WALL_PROXIMITY_REJECT_ATR,
    WALL_PROXIMITY_REJECT_PCT,
    WATCHLIST_TTL_DAYS,
)
from app.features.options_context import fetch_options_context
from app.features.price_features import clean_ohlcv, compute_features, fetch_ohlcv
from app.signals.pipeline import _extract_pattern, combine_scores, compute_options_context_score
from app.signals.scoring import quick_reject_check, score_long_setup, score_short_setup
from app.signals.trade_plan import build_long_trade_plan, build_short_trade_plan

WATCHLIST_PATH = Path(__file__).resolve().parents[2] / "data" / "watchlist.json"

# Cap on how many historical observations we keep per (ticker, direction) entry.
# Long enough to compute a robust 5-day mean and flow trend, short enough to
# keep watchlist.json readable.
STREAK_HISTORY_MAX = 10
# Window used by ``mean_flow_score_5d``.
STREAK_MEAN_WINDOW = 5


def _compute_flow_trend(history: list[float]) -> str:
    """Classify a short flow-score series as rising/flat/falling.

    Uses last min(STREAK_MEAN_WINDOW, len(history)) points and a simple linear
    slope normalised by the series mean. Returns "n/a" for histories shorter
    than 3 observations — one or two data points don't define a trend.
    """
    if not history or len(history) < 3:
        return "n/a"
    window = history[-STREAK_MEAN_WINDOW:]
    n = len(window)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(window) / n
    num = sum((xs[i] - mean_x) * (window[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den <= 0:
        return "flat"
    slope = num / den
    # Normalise slope by mean magnitude so thresholds are scale-invariant.
    scale = max(abs(mean_y), 1e-6)
    norm = slope / scale
    if norm > 0.05:
        return "rising"
    if norm < -0.05:
        return "falling"
    return "flat"


def _streak_summary(entry: dict) -> dict:
    """Return derived streak fields for an entry.

    Pure function over the entry's persisted history. Safe to call before or
    after ``add_candidates`` so callers can read "today" or "before today"
    snapshots depending on when they invoke it.
    """
    history = list(entry.get("flow_score_history") or [])
    seen_dates = list(entry.get("seen_dates") or [])
    streak_days = len(seen_dates)
    if history:
        max_flow = max(history)
        window = history[-STREAK_MEAN_WINDOW:]
        mean_5d = sum(window) / len(window)
    else:
        max_flow = float(entry.get("flow_score_raw") or 0.0)
        mean_5d = float(entry.get("flow_score_raw") or 0.0)
    return {
        "watchlist_streak_days": int(streak_days),
        "watchlist_max_flow_score": float(max_flow),
        "watchlist_mean_flow_score_5d": float(mean_5d),
        "watchlist_flow_trend": _compute_flow_trend(history),
        "watchlist_first_seen": entry.get("first_seen"),
        "watchlist_last_seen": entry.get("last_seen") or entry.get("first_seen"),
    }


def build_streak_lookup(entries: list[dict]) -> dict[tuple[str, str], dict]:
    """Map (ticker, direction) -> streak summary fields for the given entries.

    Handy for enriching pipeline output rows after ``add_candidates`` has
    folded today's observation into the watchlist. Legacy entries are
    backfilled in-place via ``_backfill_streak``.
    """
    lookup: dict[tuple[str, str], dict] = {}
    for entry in entries:
        e = _backfill_streak(dict(entry))
        lookup[(e["ticker"], e["direction"])] = _streak_summary(e)
    return lookup


def apply_streak_lookup(rows: list[dict], lookup: dict[tuple[str, str], dict]) -> None:
    """Stamp watchlist streak fields onto rows in-place."""
    for row in rows:
        key = (row.get("ticker"), row.get("direction"))
        fields = lookup.get(key)
        if fields:
            for k, v in fields.items():
                row.setdefault(k, v)


def _backfill_streak(entry: dict) -> dict:
    """Populate streak bookkeeping on legacy entries that pre-date Layer 1.

    Mutates and returns the entry so downstream consumers can rely on the
    fields existing.  Idempotent.
    """
    if "seen_dates" not in entry or not entry.get("seen_dates"):
        first_seen = entry.get("first_seen")
        if first_seen:
            entry["seen_dates"] = [str(first_seen)]
        else:
            entry["seen_dates"] = []
    if "flow_score_history" not in entry or not entry.get("flow_score_history"):
        flow_raw = entry.get("flow_score_raw")
        if flow_raw is not None and entry["seen_dates"]:
            entry["flow_score_history"] = [float(flow_raw)] * len(entry["seen_dates"])
        else:
            entry["flow_score_history"] = []
    if "last_seen" not in entry and entry["seen_dates"]:
        entry["last_seen"] = entry["seen_dates"][-1]
    return entry


def load_watchlist() -> list[dict]:
    """Load the watchlist from disk, returning an empty list if missing."""
    if not WATCHLIST_PATH.exists():
        return []
    with open(WATCHLIST_PATH, "r") as f:
        return json.load(f)


def save_watchlist(entries: list[dict]) -> Path:
    """Write the watchlist to disk, creating the parent dir if needed."""
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)
    return WATCHLIST_PATH


def prune_expired(entries: list[dict], ttl_days: int = WATCHLIST_TTL_DAYS) -> tuple[list[dict], list[dict]]:
    """Remove entries older than ttl_days.  Returns (surviving, expired)."""
    cutoff = date.today() - timedelta(days=ttl_days)
    surviving, expired = [], []
    for entry in entries:
        first_seen = date.fromisoformat(entry["first_seen"])
        if first_seen >= cutoff:
            surviving.append(entry)
        else:
            expired.append(entry)
    return surviving, expired


def add_candidates(
    existing: list[dict],
    new_rejects: list[dict],
) -> list[dict]:
    """Merge newly rejected candidates into the watchlist.

    Layer 1 streak tracking: every refresh appends today's date and
    flow_score_raw to ``seen_dates`` / ``flow_score_history`` (capped at
    ``STREAK_HISTORY_MAX``) so we can surface "this name has been on the
    watchlist N days running with rising flow" on the dashboard.

    flow_score_raw / flow_score_scaled on the entry track *today's* values
    (so re-evaluation runs against the latest flow), while
    ``max_flow_score`` and ``flow_score_history`` retain the multi-day
    picture.
    """
    today = date.today().isoformat()

    keyed: dict[tuple[str, str], dict] = {}
    for entry in existing:
        key = (entry["ticker"], entry["direction"])
        keyed[key] = _backfill_streak(dict(entry))

    for rej in new_rejects:
        if rej.get("reject_reason", "").startswith("error:"):
            continue
        if rej.get("reject_reason") == "weak_bullish_flow":
            continue
        if rej.get("reject_reason") == "weak_bearish_flow":
            continue

        key = (rej["ticker"], rej["direction"])
        flow_raw = float(rej.get("flow_score_raw") or 0.0)
        flow_scaled = float(rej.get("flow_score_scaled", rej.get("flow_score_raw") or 0.0))

        if key in keyed:
            entry = keyed[key]
            seen = list(entry.get("seen_dates") or [])
            history = list(entry.get("flow_score_history") or [])
            if not seen or seen[-1] != today:
                seen.append(today)
                history.append(flow_raw)
            else:
                # Same trading day, multiple rejects — keep the strongest
                # observation so the day's streak data point reflects peak
                # flow rather than the last one we happened to process.
                if history:
                    history[-1] = max(history[-1], flow_raw)
                else:
                    history.append(flow_raw)
            seen = seen[-STREAK_HISTORY_MAX:]
            history = history[-STREAK_HISTORY_MAX:]

            entry["seen_dates"] = seen
            entry["flow_score_history"] = [float(x) for x in history]
            entry["last_seen"] = today
            entry["flow_score_raw"] = flow_raw
            entry["flow_score_scaled"] = flow_scaled
            entry["reject_reason"] = rej.get("reject_reason", entry.get("reject_reason", "price_validation_failed"))
            entry["checks_passed"] = rej.get("checks_passed", entry.get("checks_passed", ""))
            entry["checks_failed"] = rej.get("checks_failed", entry.get("checks_failed", ""))
            if rej.get("price_score") is not None:
                entry["price_score"] = rej.get("price_score")
            for fk in _FLOW_COMPONENT_KEYS:
                if fk in rej:
                    entry[fk] = rej[fk]
            keyed[key] = entry
        else:
            new_entry = {
                "ticker": rej["ticker"],
                "direction": rej["direction"],
                "flow_score_raw": flow_raw,
                "flow_score_scaled": flow_scaled,
                "first_seen": today,
                "last_seen": today,
                "seen_dates": [today],
                "flow_score_history": [flow_raw],
                "reject_reason": rej.get("reject_reason", "price_validation_failed"),
                "checks_passed": rej.get("checks_passed", ""),
                "checks_failed": rej.get("checks_failed", ""),
                "price_score": rej.get("price_score"),
            }
            for fk in _FLOW_COMPONENT_KEYS:
                if fk in rej:
                    new_entry[fk] = rej[fk]
            keyed[key] = new_entry

    return list(keyed.values())


def reevaluate_watchlist(
    entries: list[dict],
    fresh_tickers: set[tuple[str, str]] | None = None,
    signal_bar_offset: int = 0,
    flow_features=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Re-run price validation on watchlist entries using frozen flow scores.

    Parameters
    ----------
    entries : list[dict]
        Current watchlist entries (already pruned).
    fresh_tickers : set of (ticker, direction) tuples, optional
        Candidates that appeared in today's fresh pipeline run.  These are
        skipped here because the fresh run already evaluated them.
    signal_bar_offset : int
        Passed through to scoring functions (1 = intraday mode).
    flow_features : DataFrame, optional
        Current scan's flow feature table, used to backfill flow component
        data on entries that lack it.

    Returns
    -------
    promoted : list[dict]
        Entries that now pass price validation (final signal dicts).
    still_watching : list[dict]
        Entries that still fail — stay on the watchlist.
    watch_rejected : list[dict]
        Rejection detail rows for display/logging.
    """
    from app.signals.pipeline import LONG_ALL_REASONS, SHORT_ALL_REASONS, _build_rejection_row

    flow_lookup: dict[str, dict] = {}
    if flow_features is not None and not flow_features.empty and "ticker" in flow_features.columns:
        flow_lookup = flow_features.set_index("ticker").to_dict("index")

    fresh_tickers = fresh_tickers or set()
    promoted: list[dict] = []
    still_watching: list[dict] = []
    watch_rejected: list[dict] = []

    for entry in entries:
        entry = _backfill_streak(dict(entry))
        key = (entry["ticker"], entry["direction"])
        if key in fresh_tickers:
            continue

        ticker = entry["ticker"]
        direction = entry["direction"]
        flow_raw = entry["flow_score_raw"]
        flow_scaled = entry.get("flow_score_scaled", flow_raw)
        streak_fields = _streak_summary(entry)

        if ticker in flow_lookup:
            for fk in _FLOW_COMPONENT_KEYS:
                if fk not in entry or entry[fk] is None:
                    val = flow_lookup[ticker].get(fk)
                    if val is not None:
                        entry[fk] = val

        try:
            df = fetch_ohlcv(ticker)
            df = clean_ohlcv(df)
            df = compute_features(df)

            all_reasons = LONG_ALL_REASONS if direction == "LONG" else SHORT_ALL_REASONS

            should_reject, rej_reason, stub, _counter_trend = quick_reject_check(df, direction)
            if should_reject:
                rej = _build_rejection_row(
                    ticker, direction, flow_raw, stub, all_reasons,
                    reject_reason="watchlist_reeval_failed",
                    flow_score_scaled=flow_scaled,
                )
                for fk in _FLOW_COMPONENT_KEYS:
                    if fk in entry and entry[fk] is not None:
                        rej[fk] = entry[fk]
                updated = dict(entry)
                updated["checks_passed"] = rej["checks_passed"]
                updated["checks_failed"] = rej["checks_failed"]
                updated["reject_reason"] = rej["reject_reason"]
                if rej.get("price_score") is not None:
                    updated["price_score"] = rej["price_score"]
                rej.update(streak_fields)
                updated.update(streak_fields)
                still_watching.append(updated)
                watch_rejected.append(rej)
                continue

            spot = float(df.iloc[-1]["close"])
            atr = float(df.iloc[-1]["atr14"])

            try:
                opts_ctx = fetch_options_context(ticker, spot)
            except Exception:
                opts_ctx = None

            if direction == "LONG":
                price_signal = score_long_setup(df, signal_bar_offset=signal_bar_offset)
                price_signal["ticker"] = ticker
                build_plan = build_long_trade_plan
            else:
                price_signal = score_short_setup(df, signal_bar_offset=signal_bar_offset)
                price_signal["ticker"] = ticker
                build_plan = build_short_trade_plan

            if price_signal["is_valid"]:
                if opts_ctx and atr > 0:
                    wall = (opts_ctx.get("nearest_call_wall") if direction == "LONG"
                            else opts_ctx.get("nearest_put_wall"))
                    wall_ok = (wall is not None
                               and ((direction == "LONG" and wall > spot)
                                    or (direction == "SHORT" and wall < spot)))
                    if wall_ok:
                        wd_pct = abs(wall - spot) / spot
                        wd_atr = abs(wall - spot) / atr
                        if wd_pct < WALL_PROXIMITY_REJECT_PCT and wd_atr < WALL_PROXIMITY_REJECT_ATR:
                            rej = _build_rejection_row(
                                ticker, direction, flow_raw, price_signal, all_reasons,
                                reject_reason=f"wall_proximity ({wd_pct:.1%}, {wd_atr:.1f} ATR)",
                                flow_score_scaled=flow_scaled,
                                opts_ctx=opts_ctx,
                            )
                            for fk in _FLOW_COMPONENT_KEYS:
                                if fk in entry and entry[fk] is not None:
                                    rej[fk] = entry[fk]
                            updated = dict(entry)
                            updated["checks_passed"] = rej["checks_passed"]
                            updated["checks_failed"] = rej["checks_failed"]
                            updated["reject_reason"] = rej["reject_reason"]
                            if opts_ctx:
                                _opts_score = compute_options_context_score(direction, opts_ctx)
                                updated["options_context_score"] = _opts_score
                                updated["options_context"] = opts_ctx
                                rej["options_context_score"] = _opts_score
                            rej.update(streak_fields)
                            updated.update(streak_fields)
                            still_watching.append(updated)
                            watch_rejected.append(rej)
                            continue

                trade_plan = build_plan(df, price_signal, options_ctx=opts_ctx, signal_bar_offset=signal_bar_offset)
                _opts_score = compute_options_context_score(direction, opts_ctx)
                promoted.append({
                    "ticker": ticker,
                    "direction": direction,
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
                    "pattern": _extract_pattern(price_signal.get("reasons", [])),
                    "checks_passed": ", ".join(sorted(price_signal.get("checks_passed", []))) or "none",
                    "checks_failed": ", ".join(sorted(price_signal.get("checks_failed", []))) or "none",
                    "gamma_regime": opts_ctx.get("gamma_regime") if opts_ctx else None,
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
                    "source": "watchlist",
                    "first_seen": entry["first_seen"],
                    "trade_plan": trade_plan,
                    "price_snapshot": price_signal,
                    "options_context": opts_ctx,
                })
                for fk in _FLOW_COMPONENT_KEYS:
                    if fk in entry and entry[fk] is not None:
                        promoted[-1][fk] = entry[fk]
                promoted[-1].update(streak_fields)
            else:
                rej = _build_rejection_row(
                    ticker, direction, flow_raw, price_signal, all_reasons,
                    reject_reason="watchlist_reeval_failed",
                    flow_score_scaled=flow_scaled,
                    opts_ctx=opts_ctx,
                )
                for fk in _FLOW_COMPONENT_KEYS:
                    if fk in entry and entry[fk] is not None:
                        rej[fk] = entry[fk]
                updated = dict(entry)
                updated["checks_passed"] = rej["checks_passed"]
                updated["checks_failed"] = rej["checks_failed"]
                updated["reject_reason"] = rej["reject_reason"]
                if rej.get("price_score") is not None:
                    updated["price_score"] = rej["price_score"]
                sc = price_signal.get("score_components")
                if sc:
                    for k, v in sc.items():
                        updated[f"price_{k}"] = v
                        rej[f"price_{k}"] = v
                if opts_ctx:
                    _opts_score = compute_options_context_score(direction, opts_ctx)
                    updated["options_context_score"] = _opts_score
                    updated["options_context"] = opts_ctx
                    rej["options_context_score"] = _opts_score
                    for oc_key in ("gamma_regime", "net_gex", "gamma_flip_level_estimate",
                                   "nearest_call_wall", "nearest_put_wall",
                                   "distance_to_call_wall_pct", "distance_to_put_wall_pct",
                                   "ticker_call_oi", "ticker_put_oi", "ticker_put_call_ratio",
                                   "near_term_oi", "swing_dte_oi", "long_dated_oi",
                                   "iv_rank", "iv_current"):
                        rej[oc_key] = opts_ctx.get(oc_key)
                updated["pattern"] = _extract_pattern(price_signal.get("reasons", []))
                rej["pattern"] = updated["pattern"]
                rej.update(streak_fields)
                updated.update(streak_fields)
                still_watching.append(updated)
                watch_rejected.append(rej)

        except Exception as e:
            rej = _build_rejection_row(
                ticker, direction, flow_raw, {}, set(),
                reject_reason=f"watchlist_error: {e}",
            )
            for fk in _FLOW_COMPONENT_KEYS:
                if fk in entry and entry[fk] is not None:
                    rej[fk] = entry[fk]
            updated = dict(entry)
            updated["checks_passed"] = rej["checks_passed"]
            updated["checks_failed"] = rej["checks_failed"]
            updated["reject_reason"] = rej["reject_reason"]
            rej.update(streak_fields)
            updated.update(streak_fields)
            still_watching.append(updated)
            watch_rejected.append(rej)

    return promoted, still_watching, watch_rejected

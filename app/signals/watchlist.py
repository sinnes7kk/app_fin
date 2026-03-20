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

from app.config import WATCHLIST_TTL_DAYS
from app.features.price_features import clean_ohlcv, compute_features, fetch_ohlcv
from app.signals.pipeline import combine_scores
from app.signals.scoring import score_long_setup, score_short_setup
from app.signals.trade_plan import build_long_trade_plan, build_short_trade_plan

WATCHLIST_PATH = Path(__file__).resolve().parents[2] / "data" / "watchlist.json"


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

    If a ticker+direction already exists, keep the one with the higher
    flow_score_raw (the stronger original signal).  New entries get today's date
    as first_seen.
    """
    today = date.today().isoformat()

    keyed: dict[tuple[str, str], dict] = {}
    for entry in existing:
        key = (entry["ticker"], entry["direction"])
        keyed[key] = entry

    for rej in new_rejects:
        if rej.get("reject_reason", "").startswith("error:"):
            continue
        if rej.get("reject_reason") == "weak_bullish_flow":
            continue
        if rej.get("reject_reason") == "weak_bearish_flow":
            continue

        key = (rej["ticker"], rej["direction"])
        new_entry = {
            "ticker": rej["ticker"],
            "direction": rej["direction"],
            "flow_score_raw": rej["flow_score_raw"],
            "flow_score_scaled": rej.get("flow_score_scaled", rej["flow_score_raw"]),
            "first_seen": today,
            "reject_reason": rej.get("reject_reason", "price_validation_failed"),
            "checks_failed": rej.get("checks_failed", ""),
        }

        if key not in keyed or new_entry["flow_score_raw"] > keyed[key]["flow_score_raw"]:
            keyed[key] = new_entry

    return list(keyed.values())


def reevaluate_watchlist(
    entries: list[dict],
    fresh_tickers: set[tuple[str, str]] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Re-run price validation on watchlist entries using frozen flow scores.

    Parameters
    ----------
    entries : list[dict]
        Current watchlist entries (already pruned).
    fresh_tickers : set of (ticker, direction) tuples, optional
        Candidates that appeared in today's fresh pipeline run.  These are
        skipped here because the fresh run already evaluated them.

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

    fresh_tickers = fresh_tickers or set()
    promoted: list[dict] = []
    still_watching: list[dict] = []
    watch_rejected: list[dict] = []

    for entry in entries:
        key = (entry["ticker"], entry["direction"])
        if key in fresh_tickers:
            continue

        ticker = entry["ticker"]
        direction = entry["direction"]
        flow_raw = entry["flow_score_raw"]
        flow_scaled = entry.get("flow_score_scaled", flow_raw)

        try:
            df = fetch_ohlcv(ticker)
            df = clean_ohlcv(df)
            df = compute_features(df)

            if direction == "LONG":
                price_signal = score_long_setup(df)
                price_signal["ticker"] = ticker
                all_reasons = LONG_ALL_REASONS
                build_plan = build_long_trade_plan
            else:
                price_signal = score_short_setup(df)
                price_signal["ticker"] = ticker
                all_reasons = SHORT_ALL_REASONS
                build_plan = build_short_trade_plan

            if price_signal["is_valid"]:
                trade_plan = build_plan(df, price_signal)
                promoted.append({
                    "ticker": ticker,
                    "direction": direction,
                    "flow_score_raw": flow_raw,
                    "flow_score_scaled": flow_scaled,
                    "price_score": float(price_signal["score"]),
                    "final_score": combine_scores(flow_scaled, float(price_signal["score"])),
                    "entry_price": trade_plan["entry_price"],
                    "stop_price": trade_plan["stop_price"],
                    "target_1": trade_plan["target_1"],
                    "target_2": trade_plan["target_2"],
                    "time_stop_days": trade_plan["time_stop_days"],
                    "source": "watchlist",
                    "first_seen": entry["first_seen"],
                    "trade_plan": trade_plan,
                    "price_snapshot": price_signal,
                })
            else:
                still_watching.append(entry)
                watch_rejected.append(_build_rejection_row(
                    ticker, direction, flow_raw, price_signal, all_reasons,
                    reject_reason="watchlist_reeval_failed",
                ))

        except Exception as e:
            still_watching.append(entry)
            watch_rejected.append(_build_rejection_row(
                ticker, direction, flow_raw, {}, set(),
                reject_reason=f"watchlist_error: {e}",
            ))

    return promoted, still_watching, watch_rejected

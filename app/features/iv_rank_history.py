"""Per-ticker IV rank history store (Wave 2).

The Unusual Whales ``/interpolated-iv`` endpoint returns the *current*
30-day IV percentile but no history.  Swing traders need to know whether
IV is expanding or contracting — a ``45`` reading could be "bottoming
after a vol collapse" or "rolling over from 80".

This module persists the latest IV rank per ticker per day in
``data/iv_rank_history.json`` and computes a rolling delta used by the UI
and scoring.  One file, append-once-per-day, pruned to
``IV_RANK_HISTORY_RETENTION_DAYS`` (default 30d).

Design choices:
  * JSON (not CSV) keeps the per-ticker list contiguous — cheaper reads
    than filtering a big table.
  * Last-write-wins per (ticker, date) so re-running the pipeline the
    same day never double-counts.
  * The public ``compute_iv_rank_delta`` returns (current, delta, n_days)
    where ``delta`` is ``None`` when we lack enough history (<= 1 sample
    within the lookback window).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.utils.market_calendar import current_trading_day

# Keep three weeks of history.  This is enough for the default 5d delta
# to survive holiday weeks, and small enough that the JSON stays in the
# single-digit-KB range even with hundreds of tickers.
IV_RANK_HISTORY_RETENTION_DAYS = 30
IV_RANK_DEFAULT_LOOKBACK_DAYS = 5

_HISTORY_PATH = Path("data/iv_rank_history.json")


def _load() -> dict[str, list[dict[str, Any]]]:
    """Return the on-disk history, or an empty dict if absent/corrupt."""
    try:
        with open(_HISTORY_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        # Corrupt file — start fresh rather than crash the pipeline.
        return {}

    if not isinstance(raw, dict):
        return {}

    cleaned: dict[str, list[dict[str, Any]]] = {}
    for tk, samples in raw.items():
        if not isinstance(samples, list):
            continue
        rows: list[dict[str, Any]] = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            d = s.get("date")
            iv = s.get("iv_rank")
            if not isinstance(d, str) or iv is None:
                continue
            try:
                rows.append({"date": d, "iv_rank": float(iv)})
            except (TypeError, ValueError):
                continue
        if rows:
            cleaned[tk.upper()] = rows
    return cleaned


def _save(history: dict[str, list[dict[str, Any]]]) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)


def _prune(samples: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    cutoff = today - timedelta(days=IV_RANK_HISTORY_RETENTION_DAYS)
    out: list[dict[str, Any]] = []
    for s in samples:
        try:
            d = datetime.strptime(s["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError, TypeError):
            continue
        if d >= cutoff:
            out.append({"date": s["date"], "iv_rank": float(s["iv_rank"])})
    out.sort(key=lambda s: s["date"])
    return out


def record_iv_rank(ticker: str, iv_rank: float | None, *, on: date | None = None) -> None:
    """Append today's *iv_rank* for *ticker* to the history store.

    ``on`` is an override for testing.  ``iv_rank=None`` is a no-op so the
    caller can pipe the raw options-context value without a guard.  Writing
    the same (ticker, day) twice overwrites the previous sample.
    """
    if iv_rank is None:
        return
    try:
        iv_rank_f = float(iv_rank)
    except (TypeError, ValueError):
        return
    if iv_rank_f != iv_rank_f:  # NaN guard
        return

    tk = str(ticker or "").upper().strip()
    if not tk:
        return

    today = on or current_trading_day()
    today_str = today.isoformat()

    history = _load()
    existing = history.get(tk, [])
    existing = [s for s in existing if s.get("date") != today_str]
    existing.append({"date": today_str, "iv_rank": round(iv_rank_f, 2)})
    history[tk] = _prune(existing, today)
    _save(history)


def compute_iv_rank_delta(
    ticker: str,
    *,
    lookback_days: int = IV_RANK_DEFAULT_LOOKBACK_DAYS,
    on: date | None = None,
) -> tuple[float | None, float | None, int]:
    """Return ``(current_iv_rank, delta_iv_rank, n_samples_in_window)``.

    *delta* is ``current - baseline`` where baseline is the **oldest**
    sample within ``lookback_days``.  Returns all-``None``-ish when we
    don't have at least two samples in the window.
    """
    tk = str(ticker or "").upper().strip()
    if not tk:
        return None, None, 0

    today = on or current_trading_day()
    cutoff = today - timedelta(days=lookback_days)

    samples = _load().get(tk, [])
    window: list[tuple[date, float]] = []
    for s in samples:
        try:
            d = datetime.strptime(s["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError, TypeError):
            continue
        if d >= cutoff:
            window.append((d, float(s["iv_rank"])))

    if not window:
        return None, None, 0

    window.sort(key=lambda x: x[0])
    current = window[-1][1]

    if len(window) < 2:
        # Not enough history to compute a delta yet.
        return current, None, len(window)

    baseline = window[0][1]
    return current, round(current - baseline, 2), len(window)


# ---------------------------------------------------------------------------
# Test-only helpers (intentionally not prefixed __init__-private).
# ---------------------------------------------------------------------------

def _override_history_path(path: Path) -> None:
    """Pointed at a tmp-file for unit tests."""
    global _HISTORY_PATH
    _HISTORY_PATH = path


def _reset_history_path() -> None:
    """Restore the default data/iv_rank_history.json path."""
    global _HISTORY_PATH
    _HISTORY_PATH = Path("data/iv_rank_history.json")

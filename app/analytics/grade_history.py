"""Per-grade feature history for attribution analysis.

This module closes the quant feedback loop by persisting a **feature
vector** at grade-time (``persist_grade_history``), then back-filling
the **forward excess return** once enough time has elapsed
(``attach_forward_returns``).  Together they give
``grade_attribution.py`` the panel data it needs to measure which
Flow-Tracker inputs actually predict forward returns.

The file on disk is ``data/grade_history.csv`` — plain CSV, one row per
(ticker, direction, as_of).  Appends only; rewrites only when attaching
forward returns.

Design notes
------------
- Feature columns intentionally mix continuous scores
  (``conviction_score``, ``flow_intensity``), cohort-relative signals
  (``perc_3_day_total``), and categorical tags (``dominant_dte_bucket``,
  ``premium_source``).  The attribution report handles each type
  appropriately.
- ``forward_excess_return`` is left null until ``attach_forward_returns``
  back-fills it once the 5-day forward window has closed.  This is
  idempotent — re-running it is cheap once the OHLCV cache (see
  ``grade_backtest`` A4) is warm.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GRADE_HISTORY_PATH = DATA_DIR / "grade_history.csv"

# Snapshot-side metadata (first) + feature columns (middle) + outcome
# column (last).  Kept in a fixed order so future readers don't need a
# header-aware parser — but the writer uses DictWriter so schema
# extensions are still safe (new columns land at the end).
HISTORY_COLS = [
    # Metadata
    "as_of",
    "ticker",
    "direction",
    "sector",
    # Grade + primary composites
    "conviction_score",
    "conviction_grade",
    "conviction_stack",
    # Flow features (continuous)
    "flow_intensity",
    "persistence_ratio",
    "accumulation_score",
    "sweep_share",
    "multileg_share",
    "accel_ratio_today",
    "window_return_pct",
    "cumulative_premium",
    "prem_mcap_bps",
    # Latest snapshot context
    "latest_put_call_ratio",
    "latest_iv_rank",
    "latest_oi_change",
    "perc_3_day_total_latest",
    "perc_30_day_total_latest",
    # Categorical / tag
    "dominant_dte_bucket",
    "premium_source",
    # Outcome — filled in later by attach_forward_returns().
    "forward_excess_return",
    "forward_attached_at",
]


def _coerce(value: Any) -> Any:
    """Collapse nested dicts / lists to blank so CSV stays flat."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    return value


def _load_rows() -> list[dict[str, Any]]:
    if not GRADE_HISTORY_PATH.exists():
        return []
    try:
        with open(GRADE_HISTORY_PATH, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _write_rows(rows: list[dict[str, Any]]) -> None:
    GRADE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRADE_HISTORY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def persist_grade_history(grades: list[dict[str, Any]], as_of: str) -> int:
    """Append one row per graded ticker from today's scan.

    ``grades`` is the output of ``compute_multi_day_flow()``.  Only rows
    with a ``conviction_grade`` are persisted — un-graded rows would
    dilute the attribution signal.  Idempotent on ``(as_of, ticker,
    direction)``: if today's rows are already on disk they are
    replaced (keeps hourly-scan double-writes from doubling samples).

    Returns the number of rows written.
    """
    if not grades:
        return 0

    existing = _load_rows()
    # Drop any prior rows for today — hourly scans would otherwise
    # accumulate duplicates of the same (as_of, ticker, direction).
    existing = [
        r for r in existing
        if str(r.get("as_of", "")) != str(as_of)
    ]

    new_rows: list[dict[str, Any]] = []
    for g in grades:
        grade = g.get("conviction_grade")
        if not grade:
            continue
        row = {
            "as_of": as_of,
            "ticker": g.get("ticker"),
            "direction": g.get("direction") or "BULLISH",
            "sector": g.get("sector"),
            "conviction_score": g.get("conviction_score"),
            "conviction_grade": grade,
            "conviction_stack": g.get("conviction_stack"),
            "flow_intensity": g.get("flow_intensity") or g.get("prem_mcap_bps"),
            "persistence_ratio": g.get("persistence_ratio"),
            "accumulation_score": g.get("accumulation_score"),
            "sweep_share": g.get("sweep_share"),
            "multileg_share": g.get("multileg_share"),
            "accel_ratio_today": g.get("accel_ratio_today"),
            "window_return_pct": g.get("window_return_pct"),
            "cumulative_premium": g.get("cumulative_premium"),
            "prem_mcap_bps": g.get("prem_mcap_bps"),
            "latest_put_call_ratio": g.get("latest_put_call_ratio"),
            "latest_iv_rank": g.get("latest_iv_rank"),
            "latest_oi_change": g.get("latest_oi_change"),
            "perc_3_day_total_latest": g.get("perc_3_day_total_latest"),
            "perc_30_day_total_latest": g.get("perc_30_day_total_latest"),
            "dominant_dte_bucket": g.get("dominant_dte_bucket"),
            "premium_source": g.get("premium_source"),
            "forward_excess_return": "",
            "forward_attached_at": "",
        }
        new_rows.append({k: _coerce(v) for k, v in row.items()})

    combined = existing + new_rows
    _write_rows(combined)
    return len(new_rows)


def attach_forward_returns(window: int = 5) -> int:
    """Back-fill ``forward_excess_return`` on matured rows.

    A row is eligible when:
      1. ``as_of <= today - window - 1`` (forward window has closed), AND
      2. ``forward_excess_return`` is blank (not already attached).

    Uses the same cached OHLCV fetcher as ``grade_backtest`` so repeated
    calls are cheap after the first attach.  Returns the count of rows
    attached this call.
    """
    from app.analytics.grade_backtest import _forward_excess_return

    rows = _load_rows()
    if not rows:
        return 0

    today = date.today()
    cutoff = (today - timedelta(days=window + 1)).isoformat()

    attached = 0
    changed = False
    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for r in rows:
        if (r.get("forward_excess_return") or "").strip() != "":
            continue
        as_of = str(r.get("as_of") or "").strip()
        if not as_of or as_of > cutoff:
            continue
        ticker = (r.get("ticker") or "").strip()
        if not ticker:
            continue
        try:
            ret = _forward_excess_return(ticker, as_of, window=window)
        except Exception:
            ret = None
        if ret is None:
            continue
        r["forward_excess_return"] = f"{ret:.6f}"
        r["forward_attached_at"] = now_iso
        attached += 1
        changed = True

    if changed:
        _write_rows(rows)

    return attached


def load_history(with_returns_only: bool = False) -> list[dict[str, Any]]:
    """Read grade history.  Used by ``grade_attribution.py``."""
    rows = _load_rows()
    if not with_returns_only:
        return rows
    return [
        r for r in rows
        if (r.get("forward_excess_return") or "").strip() != ""
    ]

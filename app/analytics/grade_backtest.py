"""Historical Flow Tracker grade performance.

Replays the Flow Tracker algorithm over the append-only snapshot archive
(``data/snapshots_archive.csv.gz``) and measures the **forward 5-day return
vs SPY** of every (ticker, day, direction) that would have scored a Grade
A or B at that point in time.  Writes summary stats to
``data/grade_stats.json``.

The output powers the "Grade A: 58% hit, +1.2R avg over trailing 60" header
on the Flow Tracker tab — the quant feedback loop the system has been
missing.

Run periodically (it's idempotent — OHLCV lookups are disk-cached at 24h
TTL) from the end of each scan:

    from app.analytics.grade_backtest import refresh_grade_stats
    refresh_grade_stats()

Design notes
------------
- Uses the *current* ``compute_multi_day_flow`` logic via its new ``as_of``
  / ``snapshots_path`` kwargs — self-consistent posterior (not a pure
  out-of-sample test), which is what we want for calibrating UI confidence.
- Archive source lets the backtest regress over the full
  ``MAX_HISTORY_DAYS`` window (was previously capped at the hot CSV's
  21-day retention).
- Forward return uses close-to-close over 5 trading days, minus SPY's same
  return (pair-trade P&L proxy).
- "Hit rate" = fraction where direction-signed excess return > 0.
- "Avg R" converts excess return to an approximate R-multiple using a
  fixed 2% initial-stop assumption (matches our swing trade plan
  ballpark).
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.features.flow_tracker import (
    SNAPSHOTS_ARCHIVE_PATH,
    SNAPSHOTS_PATH,
    compute_multi_day_flow,
)
from app.features.price_features import fetch_ohlcv

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STATS_PATH = DATA_DIR / "grade_stats.json"
OHLCV_CACHE_DIR = DATA_DIR / "_ohlcv_cache"
OHLCV_CACHE_TTL_SECONDS = 24 * 3600  # 24h — backtest reads same bar many times per run

FORWARD_WINDOW = 5
ASSUMED_STOP_PCT = 0.02  # 2% — rough R-multiple normalizer for swing trades
MAX_HISTORY_DAYS = 90


def _archive_source() -> Path:
    """Prefer the append-only archive; fall back to the hot CSV.

    The hot CSV fallback keeps the backtest functional on fresh installs
    that haven't accumulated archive history yet.
    """
    if SNAPSHOTS_ARCHIVE_PATH.exists():
        return SNAPSHOTS_ARCHIVE_PATH
    return SNAPSHOTS_PATH


def _load_archive() -> pd.DataFrame | None:
    """Load the full snapshot archive (handles `.csv` and `.csv.gz`)."""
    src = _archive_source()
    if not src.exists():
        return None
    try:
        df = pd.read_csv(src)
    except Exception:
        return None
    if df.empty or "snapshot_date" not in df.columns:
        return None
    return df


def _historical_grades_as_of(as_of: str) -> list[dict[str, Any]]:
    """Reconstruct the Flow Tracker grades that would have been shown on ``as_of``.

    Uses ``compute_multi_day_flow``'s new ``as_of`` + ``snapshots_path``
    kwargs — no monkeypatching, no tmp files.
    """
    return compute_multi_day_flow(as_of=as_of, snapshots_path=_archive_source())


def _ohlcv_cache_path(ticker: str) -> Path:
    safe = "".join(ch for ch in str(ticker or "").upper() if ch.isalnum() or ch in "._-")
    return OHLCV_CACHE_DIR / f"{safe}.csv"


def _ohlcv_cached(ticker: str, lookback_days: int) -> pd.DataFrame | None:
    """Return cached OHLCV for ``ticker`` when present and fresh.

    Cache is keyed by ticker only (not lookback) because we always fetch
    the longest window required and sub-select via the date index — a
    single cache entry covers every backtest forward-return call for the
    same ticker.
    """
    p = _ohlcv_cache_path(ticker)
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > OHLCV_CACHE_TTL_SECONDS:
            return None
        df = pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def _ohlcv_cache_write(ticker: str, df: pd.DataFrame) -> None:
    try:
        OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_ohlcv_cache_path(ticker))
    except Exception:
        # Cache failures should never break the backtest.
        pass


def _fetch_ohlcv_cached(ticker: str, lookback_days: int) -> pd.DataFrame | None:
    """``fetch_ohlcv`` wrapped in a 24h disk cache.

    Silently returns ``None`` on yfinance errors / rate-limits so the
    backtest degrades gracefully instead of zeroing out.
    """
    cached = _ohlcv_cached(ticker, lookback_days)
    if cached is not None:
        return cached
    try:
        df = fetch_ohlcv(ticker, lookback_days=lookback_days, include_partial=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    _ohlcv_cache_write(ticker, df)
    return df


def _forward_excess_return(ticker: str, entry_date: str, window: int = FORWARD_WINDOW) -> float | None:
    """Return ticker excess return vs SPY over ``window`` trading days from entry.

    Returns ``None`` when there aren't enough forward bars yet.
    """
    lookback = window + 30
    df = _fetch_ohlcv_cached(ticker, lookback_days=lookback)
    spy = _fetch_ohlcv_cached("SPY", lookback_days=lookback)
    if df is None or spy is None or df.empty or spy.empty:
        return None

    t_entry = pd.to_datetime(entry_date)
    for d in (df, spy):
        d.index = pd.DatetimeIndex(d.index)

    t_future = df[df.index >= t_entry]
    s_future = spy[spy.index >= t_entry]
    if len(t_future) < 2 or len(s_future) < 2:
        return None

    start_t = float(t_future.iloc[0]["close"])
    start_s = float(s_future.iloc[0]["close"])
    end_idx = min(window, len(t_future) - 1, len(s_future) - 1)
    if end_idx < 1:
        return None
    end_t = float(t_future.iloc[end_idx]["close"])
    end_s = float(s_future.iloc[end_idx]["close"])
    if start_t <= 0 or start_s <= 0:
        return None

    return (end_t / start_t - 1.0) - (end_s / start_s - 1.0)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary stats for a list of {grade, direction, forward_excess_return} rows.

    Buckets 7-tier grades into their A/B/C family so trailing stats stay
    statistically stable even as UI tiers refine (A+ / A- still count as
    A for backtest purposes).
    """
    from app.features.grade_explainer import coarse_grade

    out: dict[str, Any] = {}
    for grade in ("A", "B", "C"):
        subset = [r for r in rows if coarse_grade(r["grade"]) == grade]
        if not subset:
            out[grade] = {
                "count": 0,
                "hit_rate": None,
                "avg_r": None,
                "best_r": None,
                "worst_r": None,
                "avg_excess_return_pct": None,
            }
            continue

        signed_rs: list[float] = []
        signed_excess: list[float] = []
        for r in subset:
            sign = 1.0 if r["direction"] == "BULLISH" else -1.0
            rr = (sign * r["excess_return"]) / ASSUMED_STOP_PCT
            signed_rs.append(rr)
            signed_excess.append(sign * r["excess_return"])

        wins = sum(1 for r in signed_rs if r > 0)
        out[grade] = {
            "count": len(subset),
            "hit_rate": round(wins / len(subset), 3),
            "avg_r": round(sum(signed_rs) / len(signed_rs), 2),
            "best_r": round(max(signed_rs), 2),
            "worst_r": round(min(signed_rs), 2),
            "avg_excess_return_pct": round(sum(signed_excess) / len(signed_excess) * 100, 2),
        }
    return out


def refresh_grade_stats(
    max_history_days: int = MAX_HISTORY_DAYS,
    forward_window: int = FORWARD_WINDOW,
    write: bool = True,
) -> dict[str, Any]:
    """Recompute historical grade statistics and optionally write them to disk.

    Only evaluates entries where we have at least ``forward_window``
    trading days of forward history (so we can measure the outcome).

    Evaluates both Grade A and Grade B families so the UI can surface
    either family when the other is too thin.
    """
    all_snaps = _load_archive()
    if all_snaps is None or all_snaps.empty:
        return {"error": "no snapshots"}

    dates = sorted(all_snaps["snapshot_date"].dropna().unique().tolist())
    today = date.today()
    today_str = today.isoformat()
    cutoff_past = (today - timedelta(days=max_history_days)).isoformat()
    cutoff_forward = (today - timedelta(days=forward_window + 2)).isoformat()
    candidate_dates = [d for d in dates if cutoff_past <= d <= cutoff_forward]

    evaluations: list[dict[str, Any]] = []

    from app.features.grade_explainer import coarse_grade as _cg

    for as_of in candidate_dates:
        grades = _historical_grades_as_of(as_of)
        for g in grades:
            if _cg(g.get("conviction_grade")) not in ("A", "B"):
                continue
            excess = _forward_excess_return(g["ticker"], as_of, window=forward_window)
            if excess is None:
                continue
            evaluations.append({
                "as_of": as_of,
                "ticker": g["ticker"],
                "direction": g.get("direction") or "BULLISH",
                "grade": g["conviction_grade"],
                "score": g.get("conviction_score"),
                "excess_return": excess,
            })

    stats = _aggregate(evaluations)
    result = {
        "window_days": forward_window,
        "history_days": max_history_days,
        "evaluations": len(evaluations),
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "as_of": today_str,
        "stats": stats,
        "assumed_stop_pct": ASSUMED_STOP_PCT,
        "source": str(_archive_source().name),
    }

    if write:
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_PATH, "w") as f:
            json.dump(result, f, indent=2)

    return result


def load_grade_stats() -> dict[str, Any] | None:
    """Read cached stats if present."""
    if not STATS_PATH.exists():
        return None
    try:
        with open(STATS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def format_header(stats: dict[str, Any] | None) -> str:
    """One-line header like: 'Grade A: 58% hit, +1.2R avg over 12 setups (5d window)'."""
    if not stats or "stats" not in stats:
        return "Grade stats unavailable — not enough history yet"
    a = stats["stats"].get("A", {})
    if not a.get("count"):
        b = stats["stats"].get("B", {})
        if b.get("count"):
            hr = int(round((b.get("hit_rate") or 0) * 100))
            return (
                f"Grade B: {hr}% hit · {b.get('avg_r'):+.1f}R avg "
                f"over {b['count']} setups ({stats.get('window_days')}d window)"
            )
        return "Grade stats unavailable — not enough history yet"
    hr = int(round((a.get("hit_rate") or 0) * 100))
    return (
        f"Grade A: {hr}% hit · {a.get('avg_r'):+.1f}R avg · best {a.get('best_r'):+.1f}R / worst {a.get('worst_r'):+.1f}R "
        f"over {a['count']} setups ({stats.get('window_days')}d window)"
    )

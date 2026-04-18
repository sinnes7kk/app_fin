"""Historical Flow Tracker grade performance.

Replays the Flow Tracker algorithm over the rolling screener snapshot history
(``data/screener_snapshots.csv``) and measures the **forward 5-day return vs
SPY** of every (ticker, day, direction) that would have scored a Grade A or B
at that point in time. Writes summary stats to ``data/grade_stats.json``.

The output powers the "Grade A: 58% hit, +1.2R avg over trailing 60" header
on the Flow Tracker tab — the quant feedback loop the system has been missing.

Run periodically (it's idempotent-ish — cheap once the ~60-day yfinance cache
is warm) from the end of each scan:

    from app.analytics.grade_backtest import refresh_grade_stats
    refresh_grade_stats()

Design notes
------------
- Uses the *current* `compute_multi_day_flow` logic, which makes this a
  self-consistent posterior (not a pure out-of-sample test — it reflects how
  today's algorithm would have graded historical flow, which is what we want
  for calibrating UI confidence).
- Forward return uses close-to-close over 5 trading days, minus SPY's same
  return (pair-trade P&L proxy).
- "Hit rate" = fraction where direction-signed excess return > 0.
- "Avg R" converts excess return to an approximate R-multiple using a fixed
  2% initial-stop assumption (matches our swing trade plan ballpark).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.features.flow_tracker import SNAPSHOTS_PATH, compute_multi_day_flow
from app.features.price_features import fetch_ohlcv

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STATS_PATH = DATA_DIR / "grade_stats.json"

FORWARD_WINDOW = 5
ASSUMED_STOP_PCT = 0.02  # 2% — rough R-multiple normalizer for swing trades
MAX_HISTORY_DAYS = 90
LOOKBACK_DAYS = 5  # must match FLOW_TRACKER_LOOKBACK_DAYS


def _load_snapshots(as_of: str) -> pd.DataFrame | None:
    """Return a snapshots DataFrame truncated to rows on/before ``as_of``."""
    if not SNAPSHOTS_PATH.exists():
        return None
    try:
        df = pd.read_csv(SNAPSHOTS_PATH)
    except Exception:
        return None
    if df.empty or "snapshot_date" not in df.columns:
        return None
    return df[df["snapshot_date"] <= as_of]


def _write_truncated_snapshots(df: pd.DataFrame, tmp_path: Path) -> None:
    """Write a point-in-time snapshots CSV to ``tmp_path`` for replay."""
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path, index=False)


def _historical_grades_as_of(as_of: str) -> list[dict[str, Any]]:
    """Reconstruct the Flow Tracker grades that would have been shown on ``as_of``.

    Uses a tmp CSV + monkeypatch trick: point ``compute_multi_day_flow`` at a
    truncated copy of the snapshots file.
    """
    from app.features import flow_tracker as ft_mod

    snaps = _load_snapshots(as_of)
    if snaps is None or snaps.empty:
        return []

    tmp = DATA_DIR / "_grade_backtest_tmp.csv"
    _write_truncated_snapshots(snaps, tmp)

    orig_path = ft_mod.SNAPSHOTS_PATH
    orig_today = ft_mod.date
    try:
        ft_mod.SNAPSHOTS_PATH = tmp

        class _FakeDate:
            @staticmethod
            def today():
                return datetime.strptime(as_of, "%Y-%m-%d").date()

            @staticmethod
            def fromisoformat(s):
                return date.fromisoformat(s)

        # compute_multi_day_flow uses date.today() via `from datetime import date`.
        # Patch the module-level name.
        ft_mod.date = _FakeDate  # type: ignore[attr-defined]
        grades = compute_multi_day_flow()
    finally:
        ft_mod.SNAPSHOTS_PATH = orig_path
        ft_mod.date = orig_today  # type: ignore[attr-defined]
        if tmp.exists():
            tmp.unlink()

    return grades


def _forward_excess_return(ticker: str, entry_date: str, window: int = FORWARD_WINDOW) -> float | None:
    """Return ticker excess return vs SPY over ``window`` trading days from entry.

    Returns ``None`` when there aren't enough forward bars yet.
    """
    try:
        df = fetch_ohlcv(ticker, lookback_days=window + 30, include_partial=False)
        spy = fetch_ohlcv("SPY", lookback_days=window + 30, include_partial=False)
    except Exception:
        return None
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
    """Summary stats for a list of {grade, direction, forward_excess_return} rows."""
    out: dict[str, Any] = {}
    for grade in ("A", "B", "C"):
        subset = [r for r in rows if r["grade"] == grade]
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

    Only evaluates entries where we have at least ``forward_window`` trading
    days of forward history (so we can measure the outcome).
    """
    if not SNAPSHOTS_PATH.exists():
        return {"error": "no snapshots"}

    try:
        all_snaps = pd.read_csv(SNAPSHOTS_PATH)
    except Exception:
        return {"error": "snapshots unreadable"}

    if all_snaps.empty or "snapshot_date" not in all_snaps.columns:
        return {"error": "snapshots empty"}

    dates = sorted(all_snaps["snapshot_date"].dropna().unique().tolist())
    today_str = str(date.today())
    cutoff_past = str(date.today() - timedelta(days=max_history_days))
    cutoff_forward = str(date.today() - timedelta(days=forward_window + 2))
    candidate_dates = [d for d in dates if cutoff_past <= d <= cutoff_forward]

    evaluations: list[dict[str, Any]] = []

    for as_of in candidate_dates:
        grades = _historical_grades_as_of(as_of)
        for g in grades:
            if g.get("conviction_grade") not in ("A", "B"):
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

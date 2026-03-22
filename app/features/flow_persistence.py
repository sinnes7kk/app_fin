"""Multi-day flow persistence scoring.

Scans archived ``raw_flow_*.csv`` snapshots (saved each pipeline run) to
measure how many distinct calendar days a ticker has appeared with qualifying
directional flow.  Persistent flow over 2+ days is a stronger conviction
signal than a single-day spike.

The result is a dict mapping ``(ticker, direction)`` to a persistence score
in ``[0.0, 1.0]`` that the pipeline can blend into the final flow score.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from app.config import FLOW_PERSISTENCE_DAYS, FLOW_PERSISTENCE_BONUS

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_flow"


def _recent_flow_files(lookback_days: int = FLOW_PERSISTENCE_DAYS) -> list[Path]:
    """Return raw_flow CSVs from the last ``lookback_days`` calendar days."""
    if not DATA_ROOT.exists():
        return []
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    files: list[tuple[datetime, Path]] = []
    for p in DATA_ROOT.glob("raw_flow_*.csv"):
        stem = p.stem.replace("raw_flow_", "")
        try:
            ts = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if ts >= cutoff:
            files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def compute_persistence(
    lookback_days: int = FLOW_PERSISTENCE_DAYS,
    min_premium: float = 100_000,
) -> dict[tuple[str, str], float]:
    """Compute a per-ticker per-direction persistence score in [0, 1].

    Score = min(distinct_days / lookback_days, 1.0).  Tickers that appeared
    every day within the window receive a score of 1.0.
    """
    files = _recent_flow_files(lookback_days)
    if not files:
        return {}

    # Collect (ticker, direction) presence per calendar date
    day_presence: dict[tuple[str, str], set[str]] = {}

    for path in files:
        try:
            df = pd.read_csv(path, usecols=["ticker", "direction", "premium"])
        except Exception:
            continue
        if df.empty:
            continue
        df["premium"] = pd.to_numeric(df["premium"], errors="coerce").fillna(0)
        df = df[df["premium"] >= min_premium]
        cal_date = path.stem.replace("raw_flow_", "")[:8]
        for _, row in df.iterrows():
            key = (str(row["ticker"]), str(row["direction"]))
            day_presence.setdefault(key, set()).add(cal_date)

    scores: dict[tuple[str, str], float] = {}
    for key, dates in day_presence.items():
        scores[key] = min(len(dates) / max(lookback_days, 1), 1.0)
    return scores


def apply_persistence_bonus(
    results: list[dict],
    persistence: dict[tuple[str, str], float],
    bonus: float = FLOW_PERSISTENCE_BONUS,
) -> list[dict]:
    """Add a persistence-scaled bonus to ``final_score`` for each result.

    Maximum bonus is ``bonus`` (default 1.0 score point) when persistence == 1.0.
    """
    for r in results:
        key = (r["ticker"], r["direction"])
        p = persistence.get(key, 0.0)
        if p > 0:
            r["flow_persistence"] = round(p, 3)
            r["final_score"] = round(r["final_score"] + bonus * p, 4)
        else:
            r["flow_persistence"] = 0.0
    return results

"""Intraday flow trajectory — acceleration, breadth, and hours-active metrics.

With 8 hourly pipeline runs per day, each saving ``raw_flow_<timestamp>.csv``,
we can detect intraday flow patterns that a single daily snapshot cannot:

  - **hours_active**: how many of today's snapshots included this ticker
  - **flow_acceleration**: is premium increasing vs the previous hourly snapshot
  - **participation_breadth**: are more distinct trades appearing over time

These metrics are computed from already-saved CSV files at zero additional API
cost and feed into the persistence/velocity scoring layer.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_flow"


def _todays_flow_files() -> list[tuple[datetime, Path]]:
    """Return today's raw_flow CSVs sorted by timestamp."""
    if not DATA_ROOT.exists():
        return []
    today_prefix = datetime.utcnow().strftime("%Y%m%d")
    files: list[tuple[datetime, Path]] = []
    for p in DATA_ROOT.glob(f"raw_flow_{today_prefix}_*.csv"):
        stem = p.stem.replace("raw_flow_", "")
        try:
            ts = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return files


def _load_snapshot_stats(path: Path, min_premium: float) -> dict[tuple[str, str], dict]:
    """Aggregate per (ticker, direction) stats from a single snapshot CSV."""
    try:
        df = pd.read_csv(path, usecols=["ticker", "direction", "premium", "contracts"])
    except Exception:
        return {}
    if df.empty:
        return {}
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce").fillna(0)
    df["contracts"] = pd.to_numeric(df["contracts"], errors="coerce").fillna(0)
    df = df[df["premium"] >= min_premium]

    stats: dict[tuple[str, str], dict] = {}
    for (ticker, direction), grp in df.groupby(["ticker", "direction"]):
        stats[(str(ticker), str(direction))] = {
            "premium": float(grp["premium"].sum()),
            "count": len(grp),
            "contracts": int(grp["contracts"].sum()),
        }
    return stats


def compute_intraday_trajectory(
    min_premium: float = 50_000,
) -> dict[tuple[str, str], dict]:
    """Compute intraday trajectory metrics for each (ticker, direction).

    Returns a dict mapping (ticker, direction) -> {
        hours_active: int (1-8),
        flow_acceleration: float (-1 to +1, positive = premium increasing),
        participation_breadth: float (0-1, higher = more distinct prints over time),
        total_premium_today: float,
    }
    """
    snapshots = _todays_flow_files()
    if not snapshots:
        return {}

    all_stats: list[dict[tuple[str, str], dict]] = []
    for _, path in snapshots:
        all_stats.append(_load_snapshot_stats(path, min_premium))

    # Gather all keys across all snapshots
    all_keys: set[tuple[str, str]] = set()
    for s in all_stats:
        all_keys.update(s.keys())

    result: dict[tuple[str, str], dict] = {}
    n_snapshots = len(all_stats)

    for key in all_keys:
        premiums = [s[key]["premium"] for s in all_stats if key in s]
        counts = [s[key]["count"] for s in all_stats if key in s]
        hours_active = sum(1 for s in all_stats if key in s)

        # Flow acceleration: compare last snapshot premium to first
        if len(premiums) >= 2:
            avg_first = premiums[0]
            avg_last = premiums[-1]
            denom = max(avg_first, avg_last, 1.0)
            acceleration = (avg_last - avg_first) / denom
            acceleration = max(-1.0, min(acceleration, 1.0))
        else:
            acceleration = 0.0

        # Participation breadth: are we seeing more distinct prints over time?
        if len(counts) >= 2 and counts[0] > 0:
            breadth = min(counts[-1] / counts[0], 2.0) / 2.0
        else:
            breadth = 0.5

        result[key] = {
            "hours_active": hours_active,
            "flow_acceleration": round(acceleration, 4),
            "participation_breadth": round(breadth, 4),
            "total_premium_today": round(sum(premiums), 2),
        }

    return result


def apply_trajectory_bonus(
    results: list[dict],
    trajectory: dict[tuple[str, str], dict],
    max_bonus: float = 0.3,
) -> list[dict]:
    """Add an intraday trajectory bonus to final_score.

    Bonus scales with hours_active and acceleration:
      - 1 snapshot = no bonus (noise)
      - 5+ snapshots with positive acceleration = full bonus
    """
    for r in results:
        key = (r["ticker"], r["direction"])
        t = trajectory.get(key)
        if t is None:
            r["intraday_hours_active"] = 0
            r["intraday_acceleration"] = 0.0
            continue

        r["intraday_hours_active"] = t["hours_active"]
        r["intraday_acceleration"] = t["flow_acceleration"]

        # No bonus for single-snapshot appearances
        if t["hours_active"] <= 1:
            continue

        # hours_active score: 2 = 0.25, 4 = 0.5, 6 = 0.75, 8 = 1.0
        hours_score = min(t["hours_active"] / 8.0, 1.0)
        accel_factor = max(0.0, t["flow_acceleration"])
        breadth_factor = t["participation_breadth"]

        bonus = max_bonus * hours_score * (0.5 + 0.3 * accel_factor + 0.2 * breadth_factor)
        r["final_score"] = round(r["final_score"] + bonus, 4)

    return results

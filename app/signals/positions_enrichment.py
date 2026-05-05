"""Daily flow-aware enrichment for open positions.

Stamps three live-data fields on each position record so the position
health components (Stage F.1) and the trailing-stop tightener (Stage F.3)
have live data to react to:

    pos['current_grade']             ← Flow Tracker latest grade for ticker
    pos['current_sector_heat']       ← signed sector-heat aggregate (-10..+10)
    pos['current_unusual_flow_dir']  ← "BULLISH" / "BEARISH" / None

Also stamps ``entry_*`` snapshots on the *first* enrichment call so we
always have a reference point for grade decay etc., regardless of when
the position was first opened.

Failures (data unavailable, file missing, etc.) leave the field at
``None`` — penalties downstream are 0 in that case, so the enrichment
fail-soft is safe.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _latest_grade_for(ticker: str) -> str | None:
    """Return the most recent ``conviction_grade`` Flow Tracker assigned to
    ``ticker``, or ``None`` if not present in the latest run.
    """
    try:
        import pandas as pd
        gh = DATA_DIR / "grade_history.csv"
        if not gh.exists():
            return None
        df = pd.read_csv(gh)
        if df.empty:
            return None
        sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
        if sub.empty:
            return None
        sub = sub.copy()
        sub["__as_of_dt"] = pd.to_datetime(sub["as_of"], errors="coerce")
        sub = sub.sort_values("__as_of_dt", ascending=False)
        latest = sub.iloc[0]
        g = str(latest.get("conviction_grade") or "").strip()
        return g or None
    except Exception:
        return None


def _latest_sector_heat_for(ticker: str, sector: str | None) -> float | None:
    """Return today's signed sector-heat aggregate (-10..+10) for the
    position's sector. Positive = bullish basket; negative = bearish.

    Reads ``data/sector_heat.csv`` (latest snapshot date) and matches by
    sector. Falls back to None when the file is missing / no row matches.
    """
    if not sector:
        return None
    try:
        import pandas as pd
        path = DATA_DIR / "sector_heat.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if df.empty or "sector" not in df.columns:
            return None
        if "snapshot_date" in df.columns:
            df = df.copy()
            df["__d"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
            df = df.sort_values("__d", ascending=False)
        sub = df[df["sector"].astype(str) == str(sector)]
        if sub.empty:
            return None
        row = sub.iloc[0]
        bull = float(row.get("bull_score", 0) or 0)
        bear = float(row.get("bear_score", 0) or 0)
        # Signed heat: bull positive, bear negative; max magnitude ~10.
        return float(bull - bear)
    except Exception:
        return None


def _latest_unusual_flow_dir_for(ticker: str) -> str | None:
    """Return the dominant direction of the latest unusual flow row for
    ``ticker``, or None when the ticker isn't in today's flow features.

    Uses the most recent ``data/flow_features/flow_features_*.csv`` file.
    """
    try:
        import pandas as pd
        flow_dir = DATA_DIR / "flow_features"
        if not flow_dir.exists():
            return None
        candidates = sorted(flow_dir.glob("flow_features_*.csv"), reverse=True)
        if not candidates:
            return None
        df = pd.read_csv(candidates[0])
        sub = df[df["ticker"].astype(str).str.upper() == ticker.upper()]
        if sub.empty:
            return None
        row = sub.iloc[0]
        bull = float(row.get("bullish_premium", 0) or 0)
        bear = float(row.get("bearish_premium", 0) or 0)
        if bull <= 0 and bear <= 0:
            return None
        # Need a clear lean (≥60% on dominant side) to call it directional.
        total = bull + bear
        if total <= 0:
            return None
        if bull / total >= 0.6:
            return "BULLISH"
        if bear / total >= 0.6:
            return "BEARISH"
        return None
    except Exception:
        return None


def enrich_position_with_live_flow(pos: dict[str, Any]) -> None:
    """Mutate ``pos`` in-place with today's flow / sector / grade snapshot.

    Also stamps ``entry_*`` reference fields on first call so we have a
    fixed comparison point for grade decay regardless of when the
    position was opened. Safe to call repeatedly (only the
    ``current_*`` fields are refreshed each call).
    """
    ticker = str(pos.get("ticker") or "").upper().strip()
    if not ticker:
        return

    sector = pos.get("sector")
    pos["current_grade"] = _latest_grade_for(ticker)
    pos["current_sector_heat"] = _latest_sector_heat_for(ticker, sector)
    pos["current_unusual_flow_dir"] = _latest_unusual_flow_dir_for(ticker)
    pos["flow_enrichment_updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # First-call: stamp entry references when not already set.
    if pos.get("entry_grade") is None:
        pos["entry_grade"] = pos.get("conviction_grade") or pos.get("current_grade")
    if pos.get("entry_sector_heat") is None:
        pos["entry_sector_heat"] = pos.get("current_sector_heat")
    if pos.get("entry_unusual_flow_dir") is None:
        pos["entry_unusual_flow_dir"] = pos.get("current_unusual_flow_dir")

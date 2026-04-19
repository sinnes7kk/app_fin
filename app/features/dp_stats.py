"""Rolling z-score statistics for dark-pool notional.

Sibling of :mod:`app.features.flow_stats`.  Computes a 30-day rolling
z-score of today's dark-pool ``total_notional`` against each ticker's own
history (with cross-sectional fallback for cold-start tickers) so the UI
can flag "today's DP is genuinely unusual for this ticker" vs "DP shows
up for this name every day".

Design choices:

* Robust statistics (median + MAD) — a single gamma-squeeze day would
  otherwise poison the 30-day stddev for a week.
* 4-tier fallback ladder (mirrors flow_stats):

      Tier 1 — per-ticker full history  (n >= ZSCORE_MIN_N_FULL)
      Tier 2 — per-ticker with shrinkage toward cross-section
      Tier 3 — cross-sectional peer z-score (cold start)
      Tier 4 — absolute fallback (no cohort available)

* Single headline column (``total_notional``).  We deliberately do NOT
  z-score ``bias`` — it's already a bounded 0-1 ratio and the UI uses
  it directly.

Source of truth is ``data/dp_snapshots.csv`` (one row per
(ticker, snapshot_date)).  The public API returns a cached lookup so
callers can iterate over tracker rows cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import (
    ZSCORE_CLIP,
    ZSCORE_LOOKBACK_DAYS,
    ZSCORE_MIN_COHORT_SIZE,
    ZSCORE_MIN_N_FULL,
    ZSCORE_MIN_N_SHRUNK,
    ZSCORE_SHRINKAGE_K,
)
from app.utils.market_calendar import current_trading_day

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DP_SNAPSHOTS_PATH = DATA_DIR / "dp_snapshots.csv"

TIER_FULL = 1       # per-ticker full history
TIER_SHRUNK = 2     # per-ticker with shrinkage toward cross-section
TIER_PEER = 3       # cross-sectional only (cold start)
TIER_ABS = 4        # absolute thresholds (no cohort available)

TIER_LABELS = {
    TIER_FULL: "dp·z30",
    TIER_SHRUNK: "dp·shrunk",
    TIER_PEER: "dp·peer",
    TIER_ABS: "dp·abs",
}

# MAD → stddev consistency factor under normality (1 / 0.6745 ≈ 1.4826).
# Used so ``0.6745 * (x - med) / MAD`` is directly comparable to a
# classical (x - μ) / σ.
_MAD_CONSISTENCY = 0.6745

# MAD floor relative to the series scale — protects against "all-same-
# value" degenerate tickers producing infinite z-scores on a modest day.
_MIN_MAD_FLOOR_REL = 1e-6


@dataclass
class TickerStat:
    median: float
    mad: float
    n: int


def _load_history(
    lookback_days: int = ZSCORE_LOOKBACK_DAYS,
    *,
    path: Path = DP_SNAPSHOTS_PATH,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Return the rolling-window slice of ``dp_snapshots.csv``."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty or "snapshot_date" not in df.columns or "ticker" not in df.columns:
        return pd.DataFrame()

    if as_of is None:
        as_of = current_trading_day()
    cutoff = (as_of - timedelta(days=lookback_days)).isoformat()
    df = df[df["snapshot_date"] >= cutoff].copy()
    if df.empty:
        return df

    df["total_notional"] = pd.to_numeric(df.get("total_notional"), errors="coerce")
    return df


def _per_ticker_stats(history: pd.DataFrame) -> dict[str, TickerStat]:
    out: dict[str, TickerStat] = {}
    if history.empty or "total_notional" not in history.columns:
        return out
    for ticker, grp in history.groupby("ticker"):
        vals = pd.to_numeric(grp["total_notional"], errors="coerce").dropna()
        if vals.empty:
            continue
        med = float(vals.median())
        mad = float((vals - med).abs().median())
        out[str(ticker).upper().strip()] = TickerStat(median=med, mad=mad, n=int(vals.size))
    return out


def _cross_sectional_stats(
    history: pd.DataFrame,
    *,
    as_of_date: str | None = None,
) -> tuple[float | None, float | None]:
    """Return (median, MAD) across today's cohort — the "peer" baseline
    used for Tier-3 cold-start tickers.  Falls back to the entire
    lookback window if today has too few tickers.
    """
    if history.empty or "total_notional" not in history.columns:
        return None, None
    today = history
    if as_of_date and "snapshot_date" in history.columns:
        today = history[history["snapshot_date"] == as_of_date]
    vals_today = pd.to_numeric(today.get("total_notional"), errors="coerce").dropna()
    if len(vals_today) >= ZSCORE_MIN_COHORT_SIZE:
        med = float(vals_today.median())
        return med, float((vals_today - med).abs().median())

    # Fall back to the whole window.
    vals = pd.to_numeric(history["total_notional"], errors="coerce").dropna()
    if len(vals) < ZSCORE_MIN_COHORT_SIZE:
        return None, None
    med = float(vals.median())
    return med, float((vals - med).abs().median())


def _safe_z(x: float, med: float, mad: float, scale_ref: float) -> float:
    floor = max(_MIN_MAD_FLOOR_REL, scale_ref * _MIN_MAD_FLOOR_REL)
    if not np.isfinite(mad) or mad <= floor:
        return 0.0
    z = _MAD_CONSISTENCY * (x - med) / mad
    if not np.isfinite(z):
        return 0.0
    return float(np.clip(z, -ZSCORE_CLIP, ZSCORE_CLIP))


def compute_dp_z_tier(
    ticker: str,
    total_notional: float | None,
    *,
    history: pd.DataFrame | None = None,
    as_of: date | None = None,
) -> dict:
    """Return ``{z, tier, tier_label, n}`` for a single DP tracker row.

    * ``z``          — clipped z-score in [-ZSCORE_CLIP, +ZSCORE_CLIP];
                       ``None`` when we fell all the way to Tier 4.
    * ``tier``       — 1/2/3/4 matching the ladder described above.
    * ``tier_label`` — short UI label (e.g. ``"dp·z30"``).
    * ``n``          — number of per-ticker samples that fed the stat.
    """
    if total_notional is None or (isinstance(total_notional, float) and total_notional != total_notional):
        return {"z": None, "tier": TIER_ABS, "tier_label": TIER_LABELS[TIER_ABS], "n": 0}

    if history is None:
        history = _load_history(as_of=as_of)

    if history.empty:
        return {"z": None, "tier": TIER_ABS, "tier_label": TIER_LABELS[TIER_ABS], "n": 0}

    per_ticker = _per_ticker_stats(history)
    today_str = (as_of or current_trading_day()).isoformat()
    cs_med, cs_mad = _cross_sectional_stats(history, as_of_date=today_str)

    key = str(ticker or "").upper().strip()
    stat = per_ticker.get(key)

    # Scale reference for MAD floor — median |notional| across the window.
    all_vals = pd.to_numeric(history["total_notional"], errors="coerce").dropna()
    scale_ref = float(all_vals.abs().median()) if not all_vals.empty else 0.0

    if stat is not None and stat.n >= ZSCORE_MIN_N_FULL:
        z = _safe_z(total_notional, stat.median, stat.mad, scale_ref)
        return {"z": z, "tier": TIER_FULL, "tier_label": TIER_LABELS[TIER_FULL], "n": stat.n}

    if stat is not None and stat.n >= ZSCORE_MIN_N_SHRUNK and cs_mad is not None:
        k = ZSCORE_SHRINKAGE_K
        mad_eff = (stat.n * stat.mad + k * cs_mad) / (stat.n + k)
        z = _safe_z(total_notional, stat.median, mad_eff, scale_ref)
        return {"z": z, "tier": TIER_SHRUNK, "tier_label": TIER_LABELS[TIER_SHRUNK], "n": stat.n}

    if cs_med is not None and cs_mad is not None:
        z = _safe_z(total_notional, cs_med, cs_mad, scale_ref)
        return {"z": z, "tier": TIER_PEER, "tier_label": TIER_LABELS[TIER_PEER], "n": int(all_vals.size)}

    return {"z": None, "tier": TIER_ABS, "tier_label": TIER_LABELS[TIER_ABS], "n": 0}


def attach_dp_z_tiers(
    dp_tracker: list[dict],
    *,
    history: pd.DataFrame | None = None,
    as_of: date | None = None,
) -> list[dict]:
    """Enrich a DP tracker list in-place with ``dp_z``/``dp_tier``/
    ``dp_tier_label`` keys.  Loads history once for the whole batch so
    the per-row call is cheap."""
    if not dp_tracker:
        return dp_tracker
    if history is None:
        history = _load_history(as_of=as_of)

    # Pre-compute stats once for efficiency.
    per_ticker = _per_ticker_stats(history) if not history.empty else {}
    today_str = (as_of or current_trading_day()).isoformat()
    cs_med, cs_mad = _cross_sectional_stats(history, as_of_date=today_str) if not history.empty else (None, None)
    all_vals = pd.to_numeric(history.get("total_notional"), errors="coerce").dropna() if not history.empty else pd.Series(dtype=float)
    scale_ref = float(all_vals.abs().median()) if not all_vals.empty else 0.0

    for row in dp_tracker:
        ticker = str(row.get("ticker") or "").upper().strip()
        # The dp-tracker row exposes `cumulative_notional` as its window
        # sum, but for the "unusualness today" read we want the *latest
        # day's* notional — the last entry of daily_snapshots.
        latest_notional = None
        snaps = row.get("daily_snapshots") or []
        if snaps:
            last = next((s for s in reversed(snaps) if s.get("active")), None)
            if last:
                try:
                    latest_notional = float(last.get("notional") or 0.0)
                except (TypeError, ValueError):
                    latest_notional = None
        if latest_notional is None:
            # Fallback — better-than-nothing is the window sum.
            try:
                latest_notional = float(row.get("cumulative_notional") or 0.0)
            except (TypeError, ValueError):
                latest_notional = None

        if latest_notional is None:
            row["dp_z"] = None
            row["dp_tier"] = TIER_ABS
            row["dp_tier_label"] = TIER_LABELS[TIER_ABS]
            continue

        stat = per_ticker.get(ticker)
        if stat is not None and stat.n >= ZSCORE_MIN_N_FULL:
            row["dp_z"] = _safe_z(latest_notional, stat.median, stat.mad, scale_ref)
            row["dp_tier"] = TIER_FULL
        elif stat is not None and stat.n >= ZSCORE_MIN_N_SHRUNK and cs_mad is not None:
            k = ZSCORE_SHRINKAGE_K
            mad_eff = (stat.n * stat.mad + k * cs_mad) / (stat.n + k)
            row["dp_z"] = _safe_z(latest_notional, stat.median, mad_eff, scale_ref)
            row["dp_tier"] = TIER_SHRUNK
        elif cs_med is not None and cs_mad is not None:
            row["dp_z"] = _safe_z(latest_notional, cs_med, cs_mad, scale_ref)
            row["dp_tier"] = TIER_PEER
        else:
            row["dp_z"] = None
            row["dp_tier"] = TIER_ABS
        row["dp_tier_label"] = TIER_LABELS[row["dp_tier"]]

    return dp_tracker

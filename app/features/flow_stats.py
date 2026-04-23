"""Rolling z-score statistics for directional flow components.

Replaces the hand-cut absolute thresholds in ``flow_features._FLOW_THRESHOLDS``
with a 30-day rolling z-score per ticker, with a 4-tier fallback ladder for
tickers missing full history:

    Tier 1 — per-ticker z-score (n >= ZSCORE_MIN_N_FULL valid days)
    Tier 2 — shrunk per-ticker z-score (ZSCORE_MIN_N_SHRUNK <= n < full)
    Tier 3 — cross-sectional z-score (n < shrunk threshold)
    Tier 4 — absolute thresholds (cohort too small, safety fallback)

Historical snapshots live in ``data/flow_features/flow_features_*.csv`` (one
file per pipeline run, filename timestamp ``YYYYMMDD_HHMMSS``). We collapse to
one observation per ticker per day (taking the last run of the day) before
computing stats.

Robust statistics (median + MAD) are used throughout — a single gamma-squeeze
day would otherwise poison a 30-day stddev for a week.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FLOW_FEATURES_DIR = DATA_DIR / "flow_features"
META_PATH = DATA_DIR / "flow_stats_meta.json"

TIER_FULL = 1       # per-ticker full history
TIER_SHRUNK = 2     # per-ticker with shrinkage toward cross-section
TIER_PEER = 3       # cross-sectional only (cold start)
TIER_ABS = 4        # absolute thresholds (no cohort available)

TIER_LABELS = {
    TIER_FULL: "z30",
    TIER_SHRUNK: "z·shrunk",
    TIER_PEER: "z·peer",
    TIER_ABS: "abs",
}

# MAD → stddev consistency factor under normality (1 / 0.6745 ≈ 1.4826). We
# multiply the raw deviation by 0.6745 so that z = 0.6745 * (x - med) / MAD
# is comparable in magnitude to a classical (x - μ) / σ.
_MAD_CONSISTENCY = 0.6745

# Floor MAD so a tight-but-not-degenerate ticker doesn't get infinite z-scores
# on a modest day. Relative to the series scale.
_MIN_MAD_FLOOR_REL = 1e-6


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def _parse_snapshot_date(path: Path) -> str | None:
    """Extract ``YYYY-MM-DD`` from ``flow_features_YYYYMMDD_HHMMSS.csv``."""
    stem = path.stem  # flow_features_20260417_215451
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    ymd = parts[-2]
    if len(ymd) != 8 or not ymd.isdigit():
        return None
    return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"


def load_history(
    lookback_days: int = ZSCORE_LOOKBACK_DAYS,
    *,
    flow_features_dir: Path = FLOW_FEATURES_DIR,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load flow-feature snapshots, collapsed to one row per (ticker, date).

    When multiple files exist for the same date (intraday re-runs), the **last**
    one wins (lexicographic sort on filename is equivalent to time-sort because
    the stamp is ``YYYYMMDD_HHMMSS``).

    The returned DataFrame includes a ``date`` column (ISO string).
    """
    if not flow_features_dir.exists():
        return pd.DataFrame()

    files = sorted(flow_features_dir.glob("flow_features_*.csv"))
    if not files:
        return pd.DataFrame()

    if as_of is None:
        as_of = pd.Timestamp.utcnow().normalize()
    cutoff = (as_of - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    by_date: dict[str, Path] = {}
    for p in files:
        d = _parse_snapshot_date(p)
        if d is None or d < cutoff:
            continue
        # Later filename (later HHMMSS) overrides earlier on same date
        by_date[d] = p

    if not by_date:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for d in sorted(by_date):
        try:
            df = pd.read_csv(by_date[d])
        except Exception:
            continue
        if df.empty or "ticker" not in df.columns:
            continue
        df = df.copy()
        df["date"] = d
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _mad(x: pd.Series) -> float:
    """Median absolute deviation (not rescaled). Returns NaN for empty input."""
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    med = s.median()
    return float((s - med).abs().median())


@dataclass
class TickerStat:
    median: float
    mad: float
    n: int


def per_ticker_stats(
    history: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, TickerStat]]:
    """Return ``{column: {ticker: TickerStat(median, mad, n)}}``.

    ``n`` counts valid (non-null) daily observations for that column.
    """
    out: dict[str, dict[str, TickerStat]] = {}
    if history.empty or "ticker" not in history.columns:
        return {c: {} for c in columns}

    for col in columns:
        out[col] = {}
        if col not in history.columns:
            continue
        series = pd.to_numeric(history[col], errors="coerce")
        mask = series.notna()
        grp = history.loc[mask].groupby("ticker")[col]
        for ticker, vals in grp:
            vals_num = pd.to_numeric(vals, errors="coerce").dropna()
            if vals_num.empty:
                continue
            med = float(vals_num.median())
            mad = float((vals_num - med).abs().median())
            out[col][str(ticker)] = TickerStat(median=med, mad=mad, n=int(vals_num.size))
    return out


def cross_sectional_stats(
    today_df: pd.DataFrame,
    columns: Iterable[str],
    *,
    sector_col: str | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return cross-sectional ``{column: {group_key: (median, mad)}}``.

    ``group_key`` is ``"__all__"`` plus per-sector keys when ``sector_col`` is
    provided and present in ``today_df``.
    """
    out: dict[str, dict[str, tuple[float, float]]] = {}
    if today_df.empty:
        return {c: {} for c in columns}

    have_sector = sector_col is not None and sector_col in today_df.columns

    for col in columns:
        out[col] = {}
        if col not in today_df.columns:
            continue
        series = pd.to_numeric(today_df[col], errors="coerce").dropna()
        if len(series) >= ZSCORE_MIN_COHORT_SIZE:
            med = float(series.median())
            mad = float((series - med).abs().median())
            out[col]["__all__"] = (med, mad)

        if have_sector:
            for sector, sub in today_df.groupby(sector_col):
                if not sector or str(sector).lower() in ("nan", "none", ""):
                    continue
                svals = pd.to_numeric(sub[col], errors="coerce").dropna()
                if len(svals) >= ZSCORE_MIN_COHORT_SIZE:
                    med = float(svals.median())
                    mad = float((svals - med).abs().median())
                    out[col][str(sector)] = (med, mad)

    return out


# ---------------------------------------------------------------------------
# Z-score with tier fallback
# ---------------------------------------------------------------------------

def _safe_z(x: float, med: float, mad: float, scale_ref: float) -> float:
    """Compute clipped z from median/MAD, flooring MAD relative to scale."""
    if not np.isfinite(mad) or mad <= max(_MIN_MAD_FLOOR_REL, scale_ref * _MIN_MAD_FLOOR_REL):
        # Degenerate distribution (all same value). Return 0 — no signal.
        return 0.0
    z = _MAD_CONSISTENCY * (x - med) / mad
    if not np.isfinite(z):
        return 0.0
    return float(np.clip(z, -ZSCORE_CLIP, ZSCORE_CLIP))


def compute_z_with_tier(
    today_df: pd.DataFrame,
    *,
    columns: Iterable[str],
    per_ticker: dict[str, dict[str, TickerStat]],
    cross: dict[str, dict[str, tuple[float, float]]],
    sector_col: str | None = None,
) -> pd.DataFrame:
    """Apply the 4-tier ladder per (ticker, component).

    Returns a DataFrame indexed identically to ``today_df`` with columns
    ``{col}_z`` (clipped z-score) and ``{col}_tier`` (int in {1,2,3,4}) for
    every column in ``columns``.

    Tier-4 rows (absolute fallback) produce ``z = NaN`` — the caller is
    expected to use the original ``_clip_scale`` path in that case.
    """
    columns = list(columns)
    out = pd.DataFrame(index=today_df.index)

    have_sector = sector_col is not None and sector_col in today_df.columns

    for col in columns:
        z_vals = np.full(len(today_df), np.nan, dtype=float)
        tiers = np.full(len(today_df), TIER_ABS, dtype=int)

        if col not in today_df.columns:
            out[f"{col}_z"] = z_vals
            out[f"{col}_tier"] = tiers
            continue

        vals = pd.to_numeric(today_df[col], errors="coerce").fillna(0.0).to_numpy()
        tickers = today_df["ticker"].astype(str).to_numpy() if "ticker" in today_df.columns else np.array([""] * len(today_df))
        sectors = (
            today_df[sector_col].astype(str).to_numpy()
            if have_sector
            else np.array([""] * len(today_df))
        )

        per_t = per_ticker.get(col, {})
        cs = cross.get(col, {})
        cs_all = cs.get("__all__")

        # Pooled MAD for Tier-2 shrinkage (fall back to all-cohort if present)
        pooled_mad = cs_all[1] if cs_all is not None else None

        series_vals = pd.to_numeric(today_df[col], errors="coerce").dropna()
        scale_ref = float(series_vals.abs().median()) if not series_vals.empty else 0.0

        for i, (val, ticker, sector) in enumerate(zip(vals, tickers, sectors)):
            stat = per_t.get(ticker)

            # Tier 1: full per-ticker history
            if stat is not None and stat.n >= ZSCORE_MIN_N_FULL:
                z_vals[i] = _safe_z(val, stat.median, stat.mad, scale_ref)
                tiers[i] = TIER_FULL
                continue

            # Tier 2: shrunk per-ticker z-score
            if (
                stat is not None
                and stat.n >= ZSCORE_MIN_N_SHRUNK
                and pooled_mad is not None
            ):
                n = stat.n
                k = ZSCORE_SHRINKAGE_K
                mad_eff = (n * stat.mad + k * pooled_mad) / (n + k)
                z_vals[i] = _safe_z(val, stat.median, mad_eff, scale_ref)
                tiers[i] = TIER_SHRUNK
                continue

            # Tier 3: cross-sectional (sector preferred, then all-cohort)
            grp_stats = None
            if have_sector and sector and sector != "nan":
                grp_stats = cs.get(sector)
            if grp_stats is None:
                grp_stats = cs_all
            if grp_stats is not None:
                med, mad = grp_stats
                z_vals[i] = _safe_z(val, med, mad, scale_ref)
                tiers[i] = TIER_PEER
                continue

            # Tier 4: absolute fallback (caller handles)
            z_vals[i] = np.nan
            tiers[i] = TIER_ABS

        out[f"{col}_z"] = z_vals
        out[f"{col}_tier"] = tiers

    return out


def logistic_to_unit(z):
    """Map z-score to a 0–1 score via the logistic function.

    ``z = 0`` → 0.5, ``z = 2`` → ~0.88, ``z = 3`` → ~0.95, ``z = -2`` → ~0.12.

    This preserves the contract that ``_weighted_flow_score`` expects each
    component on [0, 1]. The logistic keeps gradient above "unusual" values
    instead of flattening like a linear clip.
    """
    if isinstance(z, pd.Series):
        out = 1.0 / (1.0 + np.exp(-z))
        return out.where(z.notna(), other=np.nan)
    if np.ndim(z) == 0:
        zf = float(z)
        if np.isnan(zf):
            return float("nan")
        return float(1.0 / (1.0 + np.exp(-zf)))
    arr = np.asarray(z, dtype=float)
    mask = np.isnan(arr)
    out = 1.0 / (1.0 + np.exp(-arr))
    out = np.asarray(out, dtype=float).copy()
    out[mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Default components to z-score
# ---------------------------------------------------------------------------

# Pair (component, columns_by_side). Each component has a bullish and bearish
# variant (except dte_score which is shared). The z-scoring keys match the
# aggregate_flow_by_ticker output column names.
COMPONENT_COLUMNS: dict[str, dict[str, str]] = {
    "flow_intensity":        {"bullish": "bullish_flow_intensity",         "bearish": "bearish_flow_intensity"},
    "premium_per_trade":     {"bullish": "bullish_ppt_bps",                "bearish": "bearish_ppt_bps"},
    "vol_oi":                {"bullish": "bullish_vol_oi",                 "bearish": "bearish_vol_oi"},
    "unusual_premium_share": {"bullish": "bullish_unusual_premium_share",  "bearish": "bearish_unusual_premium_share"},
    "repeat":                {"bullish": "bullish_repeat_count",           "bearish": "bearish_repeat_count"},
    "sweep":                 {"bullish": "bullish_sweep_count",            "bearish": "bearish_sweep_count"},
    "breadth":               {"bullish": "bullish_breadth",                "bearish": "bearish_breadth"},
    "dte":                   {"bullish": "dte_score",                      "bearish": "dte_score"},
}


def all_scored_columns() -> list[str]:
    """Every aggregate column that needs a z-score baseline."""
    seen: set[str] = set()
    out: list[str] = []
    for cfg in COMPONENT_COLUMNS.values():
        for col in cfg.values():
            if col not in seen:
                seen.add(col)
                out.append(col)
    return out

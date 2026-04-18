"""Flow-derived ticker features from normalized Unusual Whales data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.features.flow_stats import TickerStat


@dataclass
class ZStatsBundle:
    """Container for pre-computed z-score statistics passed into
    ``aggregate_flow_by_ticker``. Produced by ``flow_stats.load_history`` +
    ``per_ticker_stats`` + ``cross_sectional_stats``.

    ``per_ticker`` maps column name → ticker → TickerStat(median, mad, n).
    ``cross`` maps column name → group key → (median, mad); ``__all__`` is
    the full-cohort bucket, others are sector buckets when available.
    ``sector_col`` is the name of the sector column on the ``agg`` DataFrame
    passed to aggregate_flow_by_ticker — typically None here because agg
    doesn't carry sector. Sector-aware peer groups live on the raw flow df.
    """

    per_ticker: dict[str, dict[str, "TickerStat"]]
    cross: dict[str, dict[str, tuple[float, float]]]
    sector_col: str | None = None


def filter_qualifying_flow(
    df: pd.DataFrame,
    min_premium: float = 500_000,
    min_dte: int = 30,
    max_dte: int = 120,
    require_ask_side: bool = False,
) -> pd.DataFrame:
    """
    Filter normalized flow events down to the ones relevant for the swing system.

    Rules:
    - premium >= min_premium
    - min_dte <= dte <= max_dte
    - option_type is CALL or PUT
    - execution_side is ASK or BID (MIXED excluded unless require_ask_side)
    """
    if df.empty:
        return df.copy()

    required_cols = ["premium", "dte", "option_type", "execution_side"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for flow filtering: {missing}")

    out = df.copy()

    out = out[
        (out["premium"] >= min_premium)
        & (out["dte"] >= min_dte)
        & (out["dte"] <= max_dte)
        & (out["option_type"].isin(["CALL", "PUT"]))
    ]

    if require_ask_side:
        out = out[out["execution_side"] == "ASK"]
    else:
        out = out[out["execution_side"].isin(["ASK", "BID"])]

    return out


def add_volume_oi_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume/open_interest ratio.

    If open_interest is missing or zero, ratio is NaN.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if "volume" not in out.columns or "open_interest" not in out.columns:
        out["volume_oi_ratio"] = pd.NA
        return out

    oi = pd.to_numeric(out["open_interest"], errors="coerce")
    vol = pd.to_numeric(out["volume"], errors="coerce")

    ratio = vol / oi.replace(0, float("nan"))
    out["volume_oi_ratio"] = ratio.astype("float64")

    return out


RECENCY_HALF_LIFE_HOURS = 6.0


def add_recency_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Apply exponential time-decay to premiums so recent flow matters more.

    Half-life is RECENCY_HALF_LIFE_HOURS — a trade 6 hours old contributes
    half the premium of an identical trade placed right now.
    """
    if df.empty or "event_ts" not in df.columns:
        out = df.copy()
        out["recency_weight"] = 1.0
        return out

    out = df.copy()
    now = pd.Timestamp(datetime.now(tz=timezone.utc))
    age_hours = (now - out["event_ts"]).dt.total_seconds() / 3600
    age_hours = age_hours.clip(lower=0)
    decay = np.exp(-np.log(2) * age_hours / RECENCY_HALF_LIFE_HOURS)
    out["recency_weight"] = decay
    out["premium"] = out["premium"] * decay
    return out


def add_repeat_flow_count(
    df: pd.DataFrame,
    group_col: str = "ticker",
) -> pd.DataFrame:
    """
    Add per-row repeat count within the filtered dataset.

    For V1 this is simply the count of qualifying events for that ticker.
    """
    if df.empty:
        return df.copy()

    if group_col not in df.columns:
        raise ValueError(f"Missing grouping column: {group_col}")

    out = df.copy()
    out["repeat_flow_count"] = out.groupby(group_col)[group_col].transform("count")
    return out


# ---------------------------------------------------------------------------
# Delta-weighted directional premium
# ---------------------------------------------------------------------------

# Proxy fallback when UW's /option-contract/{id}/greek-exposure is unreachable
# or the contract count for a scan exceeds the fetch budget. Black-Scholes
# delta with flat vol/zero rate is close enough for a weighting signal, and
# its bias (flat vol surface) is dwarfed by the accuracy gain over treating
# lottos and LEAPs as equal.
_PROXY_MIN_DTE_DAYS = 1


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _moneyness_delta_proxy(
    option_type: str,
    strike: float,
    underlying_price: float,
    dte: float,
    sigma: float = 0.35,
) -> float | None:
    """Signed Black-Scholes delta with r=0 and a flat vol assumption.

    Returns a value in [-1, 1] or ``None`` when inputs are invalid (so the
    caller can decide between dropping the row or using a conservative
    default).  Calls return positive delta, puts return negative.
    """
    if option_type is None or strike is None or underlying_price is None:
        return None
    try:
        K = float(strike)
        S = float(underlying_price)
        T_days = float(dte)
        vol = float(sigma)
    except (TypeError, ValueError):
        return None

    if K <= 0 or S <= 0 or vol <= 0:
        return None
    # Tiny floor on T to avoid log/sqrt explosions on same-day expiries.
    T = max(T_days, _PROXY_MIN_DTE_DAYS) / 365.0

    sqrt_t = math.sqrt(T)
    try:
        d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * sqrt_t)
    except (ValueError, ZeroDivisionError):
        return None

    ot = str(option_type).upper().strip()
    if ot.startswith("C"):
        return float(_norm_cdf(d1))
    if ot.startswith("P"):
        return float(_norm_cdf(d1) - 1.0)
    return None


def add_delta_weights(
    df: pd.DataFrame,
    *,
    max_unique_fetch: int | None = None,
    timeout: int | None = None,
    sigma: float | None = None,
) -> pd.DataFrame:
    """Attach per-event delta and delta-weighted premium.

    For every qualifying event we:
      1) Build the OCC contract id.
      2) Look up delta at (or near) ``event_ts`` via UW's greek-exposure
         endpoint (day-scoped file cache; see vendors.unusual_whales).
      3) Fall back to a Black-Scholes moneyness proxy on cache-miss or
         network failure.

    Output columns:
      - ``contract_id``        — OCC-style id or None
      - ``delta``              — signed (positive calls / negative puts)
      - ``delta_magnitude``    — ``|delta|``
      - ``delta_premium``      — ``premium * |delta|`` (uses recency-weighted premium)
      - ``delta_source``       — "uw" | "proxy" | "missing"

    Behaviour is a no-op (returns a copy) when ``df`` is empty.
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        if isinstance(out, pd.DataFrame) and not out.empty:
            out["delta"] = np.nan
            out["delta_magnitude"] = 0.0
            out["delta_premium"] = 0.0
            out["delta_source"] = "missing"
        return out

    # Lazy imports to avoid a hard dependency during tests / parity runs.
    from app import config
    from app.vendors.unusual_whales import build_occ_id, fetch_contract_greeks

    if max_unique_fetch is None:
        max_unique_fetch = getattr(config, "DELTA_MAX_UNIQUE_PER_SCAN", 250)
    if timeout is None:
        timeout = getattr(config, "DELTA_FETCH_TIMEOUT", 5)
    if sigma is None:
        sigma = getattr(config, "DELTA_PROXY_VOL", 0.35)

    out = df.copy()

    contract_ids: list[str | None] = []
    for _, row in out.iterrows():
        cid = build_occ_id(
            row.get("ticker"),
            row.get("expiration_date"),
            row.get("option_type"),
            row.get("strike"),
        )
        contract_ids.append(cid)
    out["contract_id"] = contract_ids

    # Pull unique (contract_id, event_day) pairs for the UW fetch plan.  Cap
    # unique contracts per scan: above the cap we skip the API entirely and
    # every row falls back to the proxy.  This preserves shadow behaviour
    # and protects the rate limiter when a single scan surfaces an outlier
    # number of contracts.
    unique_ids = [c for c in set(contract_ids) if c]
    allow_uw = 0 < len(unique_ids) <= max_unique_fetch

    uw_cache: dict[str, dict | None] = {}
    if allow_uw:
        for cid in unique_ids:
            # Use the most recent event_ts for that contract as the lookup ts
            # so we stay inside the cache day that matches the scan.
            sub = out[(out["contract_id"] == cid)]
            ts_val = None
            if "event_ts" in sub.columns and not sub.empty:
                try:
                    ts_val = sub["event_ts"].max()
                except Exception:
                    ts_val = None
            try:
                uw_cache[cid] = fetch_contract_greeks(cid, as_of_ts=ts_val, timeout=timeout)
            except Exception:
                uw_cache[cid] = None

    def _col_as_series(frame: pd.DataFrame, name: str, default_val=np.nan) -> pd.Series:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(default_val, index=frame.index, dtype="float64")

    premiums = _col_as_series(out, "premium", 0.0).fillna(0.0)
    underlyings = _col_as_series(out, "underlying_price")
    strikes = _col_as_series(out, "strike")
    dtes = _col_as_series(out, "dte")

    deltas: list[float | None] = []
    sources: list[str] = []
    for i, cid in enumerate(contract_ids):
        delta_val: float | None = None
        source = "missing"

        if cid and cid in uw_cache and uw_cache[cid] is not None:
            d_raw = uw_cache[cid].get("delta")
            try:
                delta_val = float(d_raw) if d_raw is not None else None
            except (TypeError, ValueError):
                delta_val = None
            if delta_val is not None:
                # UW returns unsigned delta for puts in some shapes; enforce sign from option_type.
                ot = str(out.iloc[i].get("option_type", "")).upper().strip()
                if ot.startswith("P") and delta_val > 0:
                    delta_val = -delta_val
                elif ot.startswith("C") and delta_val < 0:
                    delta_val = -delta_val
                source = "uw"

        if delta_val is None:
            proxy = _moneyness_delta_proxy(
                out.iloc[i].get("option_type"),
                strikes.iloc[i] if i < len(strikes) else None,
                underlyings.iloc[i] if i < len(underlyings) else None,
                dtes.iloc[i] if i < len(dtes) else None,
                sigma=sigma,
            )
            if proxy is not None:
                delta_val = proxy
                source = "proxy"

        deltas.append(delta_val)
        sources.append(source)

    delta_series = pd.Series(deltas, index=out.index, dtype="float64")
    out["delta"] = delta_series
    magnitude = delta_series.abs().clip(lower=0.0, upper=1.0).fillna(0.0)
    out["delta_magnitude"] = magnitude
    out["delta_premium"] = premiums * magnitude
    out["delta_source"] = sources
    return out


def _dte_score(dte: float) -> float:
    """Score DTE quality for swing continuation."""
    if pd.isna(dte):
        return 0.0
    if 30 <= dte <= 90:
        return 1.0
    if 90 < dte <= 150:
        return 0.7
    return 0.3


def _norm01(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to 0–1. Returns 0.5 for constant input."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _clip_scale(series: pd.Series, floor: float, ceiling: float) -> pd.Series:
    """Scale series to 0-1 using absolute thresholds.

    Values at or below *floor* map to 0, at or above *ceiling* map to 1,
    with linear interpolation between.
    """
    if ceiling == floor:
        return pd.Series(0.5, index=series.index)
    clamped = series.clip(lower=floor, upper=ceiling)
    return (clamped - floor) / (ceiling - floor)


# Weights sum to 1.0 — adjust here to re-prioritize.
# NOTE: "velocity" was retired. Its 0.13 weight was redistributed proportionally
# (× 1/0.87) across the remaining 7 components. Directional momentum is now
# applied as a post-score bonus on final_score inside
# app.signals.pipeline._enrich_net_prem_ticks.
_FLOW_WEIGHTS = {
    "flow_intensity":    0.29,
    "premium_per_trade": 0.17,
    "vol_oi":            0.17,
    "repeat":            0.14,
    "sweep":             0.12,
    "dte":               0.06,
    "breadth":           0.05,
}

# V2 thresholds — calibrated from domain knowledge. Tuning targets:
# - After 2-4 weeks of live data, replace with percentile-based thresholds
#   derived from the accumulated distribution of each metric.
# - V3+: switch to z-scores against a 30-day rolling distribution per
#   component, or percentile ranks against a historical lookback window.
#   This requires storing per-run flow statistics in data/flow_stats.csv.
_FLOW_THRESHOLDS = {
    "flow_intensity":    (np.log1p(0.01), np.log1p(1.0)),
    "premium_per_trade": (np.log1p(0.005), np.log1p(0.5)),
    "vol_oi":            (0.5,   3.0),
    "repeat":            (1.0,  10.0),
    "sweep":             (0.0,   8.0),
    "dte":               (0.3,   1.0),
    "breadth":           (0.1,   0.8),
}

# When USE_DELTA_WEIGHTED_FLOW is ON, the intensity distribution compresses to
# ~40-60% of the raw-premium version (since |delta| < 1 for every event), so
# the ceiling needs to come down or nothing ever saturates. These are the
# starting points; recalibrate from shadow-logged percentiles after ~1 week.
_FLOW_THRESHOLDS_DELTA = {
    **_FLOW_THRESHOLDS,
    "flow_intensity": (np.log1p(0.005), np.log1p(0.5)),
}


def _active_flow_thresholds() -> dict:
    """Return the threshold dict aligned with the current config flag.
    Called at scorer time so changing ``USE_DELTA_WEIGHTED_FLOW`` only affects
    scoring, not the shadow-logged columns."""
    try:
        from app import config as _cfg
        if getattr(_cfg, "USE_DELTA_WEIGHTED_FLOW", False):
            return _FLOW_THRESHOLDS_DELTA
    except Exception:
        pass
    return _FLOW_THRESHOLDS


def _active_flow_intensity_col(side: str) -> str:
    """Return the aggregate column that should feed the ``flow_intensity``
    component based on the current config flag.  ``side`` is ``"bullish"`` or
    ``"bearish"``.
    """
    try:
        from app import config as _cfg
        if getattr(_cfg, "USE_DELTA_WEIGHTED_FLOW", False):
            return f"{side}_delta_intensity"
    except Exception:
        pass
    return f"{side}_flow_intensity"


def _component_unit_scores(
    agg: pd.DataFrame,
    *,
    flow_intensity_col: str,
    ppt_col: str,
    vol_oi_col: str,
    repeat_col: str,
    sweep_col: str,
    breadth_col: str,
    dte_col: str,
) -> dict[str, pd.Series]:
    """Return the per-component 0-1 unit scores used by ``_weighted_flow_score``.

    Split out from ``_weighted_flow_score`` so we can attach per-component
    contributions to the aggregate DataFrame without recomputing the scorer.
    """
    t = _active_flow_thresholds()
    return {
        "flow_intensity":    _clip_scale(np.log1p(agg[flow_intensity_col].fillna(0) * 10_000), *t["flow_intensity"]),
        "premium_per_trade": _clip_scale(np.log1p(agg[ppt_col]), *t["premium_per_trade"]),
        "vol_oi":            _clip_scale(agg[vol_oi_col].fillna(0), *t["vol_oi"]),
        "repeat":            _clip_scale(agg[repeat_col], *t["repeat"]),
        "sweep":             _clip_scale(agg[sweep_col], *t["sweep"]),
        "dte":               _clip_scale(agg[dte_col], *t["dte"]),
        "breadth":           _clip_scale(agg[breadth_col], *t["breadth"]),
    }


def _weighted_flow_score(
    agg: pd.DataFrame,
    *,
    flow_intensity_col: str,
    ppt_col: str,
    vol_oi_col: str,
    repeat_col: str,
    sweep_col: str,
    breadth_col: str,
    dte_col: str,
) -> pd.Series:
    """Build a directional flow score from absolute-threshold-scaled 0-1 components."""
    w = _FLOW_WEIGHTS
    units = _component_unit_scores(
        agg,
        flow_intensity_col=flow_intensity_col,
        ppt_col=ppt_col,
        vol_oi_col=vol_oi_col,
        repeat_col=repeat_col,
        sweep_col=sweep_col,
        breadth_col=breadth_col,
        dte_col=dte_col,
    )
    score = pd.Series(0.0, index=agg.index)
    for comp, unit in units.items():
        score = score + w[comp] * unit
    return score


def _component_unit(
    agg: pd.DataFrame,
    abs_col: str,
    threshold_key: str,
    z_df: pd.DataFrame | None,
    z_col: str,
    *,
    log_transform: bool = False,
    log_scale: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Return (unit_score_0_1, tier) for one component.

    When ``z_df`` is provided and has a valid z-score for a row
    (``{z_col}_z`` non-null), the unit score is the logistic of that z.
    Otherwise (Tier-4 fallback, or z-scoring disabled) we use the legacy
    ``_clip_scale`` path so behaviour is identical to the absolute-threshold
    system.
    """
    from app.features.flow_stats import TIER_ABS, logistic_to_unit

    t = _active_flow_thresholds()[threshold_key]
    base = agg[abs_col].fillna(0)
    if log_transform:
        base = np.log1p(base * log_scale)
    abs_unit = _clip_scale(base, *t)

    if z_df is None or f"{z_col}_z" not in z_df.columns:
        tier = pd.Series(TIER_ABS, index=agg.index)
        return abs_unit, tier

    z = pd.to_numeric(z_df[f"{z_col}_z"], errors="coerce")
    tier = pd.to_numeric(z_df.get(f"{z_col}_tier", TIER_ABS), errors="coerce").fillna(TIER_ABS).astype(int)

    z_unit = logistic_to_unit(z)
    # Where z is NaN (Tier 4), fall back to absolute-threshold unit
    unit = pd.Series(z_unit, index=agg.index).where(z.notna(), other=abs_unit)
    return unit, tier


def _weighted_flow_score_mixed(
    agg: pd.DataFrame,
    *,
    flow_intensity_col: str,
    ppt_col: str,
    vol_oi_col: str,
    repeat_col: str,
    sweep_col: str,
    breadth_col: str,
    dte_col: str,
    z_df: pd.DataFrame | None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build a directional flow score with per-row z-score fallback ladder.

    Returns ``(score, tier_frame)`` where ``tier_frame`` has one column per
    component named ``{component}_tier`` carrying an int in {1,2,3,4}.
    """
    w = _FLOW_WEIGHTS

    parts: list[tuple[str, str, pd.Series, pd.Series]] = []

    unit, tier = _component_unit(agg, flow_intensity_col, "flow_intensity", z_df, flow_intensity_col, log_transform=True, log_scale=10_000)
    parts.append(("flow_intensity", flow_intensity_col, unit, tier))

    unit, tier = _component_unit(agg, ppt_col, "premium_per_trade", z_df, ppt_col, log_transform=True)
    parts.append(("premium_per_trade", ppt_col, unit, tier))

    unit, tier = _component_unit(agg, vol_oi_col, "vol_oi", z_df, vol_oi_col)
    parts.append(("vol_oi", vol_oi_col, unit, tier))

    unit, tier = _component_unit(agg, repeat_col, "repeat", z_df, repeat_col)
    parts.append(("repeat", repeat_col, unit, tier))

    unit, tier = _component_unit(agg, sweep_col, "sweep", z_df, sweep_col)
    parts.append(("sweep", sweep_col, unit, tier))

    unit, tier = _component_unit(agg, dte_col, "dte", z_df, dte_col)
    parts.append(("dte", dte_col, unit, tier))

    unit, tier = _component_unit(agg, breadth_col, "breadth", z_df, breadth_col)
    parts.append(("breadth", breadth_col, unit, tier))

    score = pd.Series(0.0, index=agg.index)
    tier_frame = pd.DataFrame(index=agg.index)
    for name, _col, unit, tier in parts:
        score = score + w[name] * unit
        tier_frame[f"{name}_tier"] = tier

    return score, tier_frame


def _contribution_frame(unit_scores: dict[str, pd.Series]) -> pd.DataFrame:
    """Given per-component 0-1 unit scores, return a DataFrame with:
      - ``{comp}_unit``   — the raw 0-1 unit score
      - ``{comp}_contrib`` — contribution in points on the 0-10 final scale
        (i.e. ``weight * unit * 10``). Summing ``_contrib`` columns across
        components recovers ``bullish_score * 10`` on each row.
    """
    w = _FLOW_WEIGHTS
    out = pd.DataFrame(index=next(iter(unit_scores.values())).index)
    for comp, unit in unit_scores.items():
        out[f"{comp}_unit"] = unit.astype(float)
        out[f"{comp}_contrib"] = (w[comp] * unit * 10.0).astype(float)
    return out


def _attach_component_breakdown(
    agg: pd.DataFrame,
    *,
    side: str,
    flow_intensity_col: str,
    ppt_col: str,
    vol_oi_col: str,
    repeat_col: str,
    sweep_col: str,
    breadth_col: str,
    dte_col: str,
    z_df: pd.DataFrame | None = None,
) -> None:
    """Attach ``{side}_{comp}_contrib`` columns to ``agg`` in place.

    Uses the z-score unit scores when ``z_df`` is provided (falling back to
    the absolute-threshold path per-row), otherwise uses the pure absolute
    path.  Mirrors the scorer logic so contributions match the score.
    """
    if z_df is None:
        units = _component_unit_scores(
            agg,
            flow_intensity_col=flow_intensity_col,
            ppt_col=ppt_col,
            vol_oi_col=vol_oi_col,
            repeat_col=repeat_col,
            sweep_col=sweep_col,
            breadth_col=breadth_col,
            dte_col=dte_col,
        )
    else:
        col_by_comp = {
            "flow_intensity":    (flow_intensity_col, True,  10_000.0),
            "premium_per_trade": (ppt_col,            True,  1.0),
            "vol_oi":            (vol_oi_col,         False, 1.0),
            "repeat":            (repeat_col,         False, 1.0),
            "sweep":             (sweep_col,          False, 1.0),
            "dte":               (dte_col,            False, 1.0),
            "breadth":           (breadth_col,        False, 1.0),
        }
        units = {}
        for comp, (col, log_transform, log_scale) in col_by_comp.items():
            unit, _tier = _component_unit(
                agg,
                col,
                comp,
                z_df,
                col,
                log_transform=log_transform,
                log_scale=log_scale,
            )
            units[comp] = unit

    breakdown = _contribution_frame(units)
    for comp in _FLOW_WEIGHTS:
        agg[f"{side}_{comp}_contrib"] = breakdown[f"{comp}_contrib"].round(4)


def _build_z_frame(
    agg: pd.DataFrame,
    z_stats: "ZStatsBundle",
    *,
    side: str,
) -> pd.DataFrame | None:
    """Run ``flow_stats.compute_z_with_tier`` on the directional columns for
    the requested side and return a DataFrame aligned to ``agg.index``."""
    from app.features.flow_stats import COMPONENT_COLUMNS, compute_z_with_tier

    cols = [cfg[side] for cfg in COMPONENT_COLUMNS.values()]
    # De-dupe (dte_score is shared across sides)
    cols = list(dict.fromkeys(cols))

    if agg.empty:
        return None

    today = agg[["ticker"] + [c for c in cols if c in agg.columns]].copy()
    if z_stats.sector_col and z_stats.sector_col in agg.columns:
        today[z_stats.sector_col] = agg[z_stats.sector_col]

    return compute_z_with_tier(
        today,
        columns=cols,
        per_ticker=z_stats.per_ticker,
        cross=z_stats.cross,
        sector_col=z_stats.sector_col,
    )


def build_z_stats_bundle(
    history: pd.DataFrame,
    today_agg: pd.DataFrame,
    *,
    sector_col: str | None = None,
) -> ZStatsBundle:
    """Convenience helper: build a full ZStatsBundle from loaded history and
    today's aggregated flow. Intended to be called once per pipeline run.
    """
    from app.features.flow_stats import (
        all_scored_columns,
        cross_sectional_stats,
        per_ticker_stats,
    )

    cols = all_scored_columns()
    per_t = per_ticker_stats(history, cols)
    cs = cross_sectional_stats(today_agg, cols, sector_col=sector_col)
    return ZStatsBundle(per_ticker=per_t, cross=cs, sector_col=sector_col)


def rescore_with_z(
    agg: pd.DataFrame,
    history: pd.DataFrame,
    *,
    sector_col: str | None = None,
) -> pd.DataFrame:
    """Rescore an existing aggregated flow table using z-score baselines.

    Replaces ``bullish_score`` and ``bearish_score`` in-place (returns the same
    DataFrame) and attaches per-component tier columns
    (``{side}_{component}_tier``) plus summary columns
    (``{side}_zscore_tier``). Behaviour is a no-op (legacy columns preserved,
    no tier columns added) when ``agg`` is empty.

    The legacy ``bullish_score`` / ``bearish_score`` values from the
    absolute-threshold path are preserved as
    ``bullish_score_abs`` / ``bearish_score_abs`` for shadow comparison.
    """
    if agg is None or agg.empty:
        return agg

    bundle = build_z_stats_bundle(history, agg, sector_col=sector_col)

    # Preserve legacy scores for shadow comparison
    if "bullish_score" in agg.columns:
        agg["bullish_score_abs"] = agg["bullish_score"].copy()
    if "bearish_score" in agg.columns:
        agg["bearish_score_abs"] = agg["bearish_score"].copy()

    bull_z = _build_z_frame(agg, bundle, side="bullish")
    bear_z = _build_z_frame(agg, bundle, side="bearish")

    bull_score, bull_tiers = _weighted_flow_score_mixed(
        agg,
        flow_intensity_col=_active_flow_intensity_col("bullish"),
        ppt_col="bullish_ppt_bps",
        vol_oi_col="bullish_vol_oi",
        repeat_col="bullish_repeat_count",
        sweep_col="bullish_sweep_count",
        breadth_col="bullish_breadth",
        dte_col="dte_score",
        z_df=bull_z,
    )
    bear_score, bear_tiers = _weighted_flow_score_mixed(
        agg,
        flow_intensity_col=_active_flow_intensity_col("bearish"),
        ppt_col="bearish_ppt_bps",
        vol_oi_col="bearish_vol_oi",
        repeat_col="bearish_repeat_count",
        sweep_col="bearish_sweep_count",
        breadth_col="bearish_breadth",
        dte_col="dte_score",
        z_df=bear_z,
    )
    agg["bullish_score"] = bull_score
    agg["bearish_score"] = bear_score

    for comp_col, series in bull_tiers.items():
        comp = comp_col.replace("_tier", "")
        agg[f"bullish_{comp}_tier"] = series.astype(int)
    for comp_col, series in bear_tiers.items():
        comp = comp_col.replace("_tier", "")
        agg[f"bearish_{comp}_tier"] = series.astype(int)

    agg["bullish_zscore_tier"] = bull_tiers.max(axis=1).astype(int)
    agg["bearish_zscore_tier"] = bear_tiers.max(axis=1).astype(int)

    _attach_component_breakdown(
        agg, side="bullish",
        flow_intensity_col=_active_flow_intensity_col("bullish"),
        ppt_col="bullish_ppt_bps", vol_oi_col="bullish_vol_oi",
        repeat_col="bullish_repeat_count", sweep_col="bullish_sweep_count",
        breadth_col="bullish_breadth", dte_col="dte_score", z_df=bull_z,
    )
    _attach_component_breakdown(
        agg, side="bearish",
        flow_intensity_col=_active_flow_intensity_col("bearish"),
        ppt_col="bearish_ppt_bps", vol_oi_col="bearish_vol_oi",
        repeat_col="bearish_repeat_count", sweep_col="bearish_sweep_count",
        breadth_col="bearish_breadth", dte_col="dte_score", z_df=bear_z,
    )

    return agg


def aggregate_flow_by_ticker(
    df: pd.DataFrame,
    *,
    z_stats: "ZStatsBundle | None" = None,
) -> pd.DataFrame:
    """
    Aggregate filtered normalized flow to one row per ticker.

    Output includes:
    - direction-specific premiums/counts
    - directional vol/OI
    - sweep counts/premium
    - premium-per-trade metrics
    - breadth metrics
    - DTE quality score
    - directional scores for ranking

    When ``z_stats`` is provided, component scoring uses the rolling z-score
    baseline (with the 4-tier fallback ladder in ``flow_stats.py``). Per-
    component tier columns ``{side}_{component}_tier`` are attached to the
    output, plus a summary ``{side}_zscore_tier`` (worst component tier on
    the scored side).
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "bullish_premium",
                "bearish_premium",
                "bullish_count",
                "bearish_count",
                "total_premium",
                "total_count",
                "avg_dte",
                "avg_volume_oi_ratio",
                "bullish_vol_oi",
                "bearish_vol_oi",
                "repeat_flow_count",
                "bullish_sweep_count",
                "bearish_sweep_count",
                "bullish_sweep_premium",
                "bearish_sweep_premium",
                "bullish_premium_per_trade",
                "bearish_premium_per_trade",
                "bullish_max_trade_premium",
                "bearish_max_trade_premium",
                "bullish_breadth",
                "bearish_breadth",
                "bullish_premium_raw",
                "bearish_premium_raw",
                "net_premium_bias",
                "flow_imbalance_ratio",
                "dominant_direction",
                "dte_score",
                "marketcap",
                "bullish_flow_intensity",
                "bearish_flow_intensity",
                "bullish_score",
                "bearish_score",
                # Wave 2 — repeat-flow acceleration columns.
                "bullish_repeat_2h",
                "bearish_repeat_2h",
                "bullish_accel_ratio",
                "bearish_accel_ratio",
                "bullish_accel_score",
                "bearish_accel_score",
            ]
        )

    required_cols = ["ticker", "premium", "dte", "direction"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for aggregation: {missing}")

    out = df.copy()

    if "volume_oi_ratio" not in out.columns:
        out = add_volume_oi_ratio(out)

    if "repeat_flow_count" not in out.columns:
        out = add_repeat_flow_count(out)

    if "is_sweep" not in out.columns:
        out["is_sweep"] = False

    # Ensure numeric / boolean stability
    out["premium"] = pd.to_numeric(out["premium"], errors="coerce").fillna(0.0)
    out["dte"] = pd.to_numeric(out["dte"], errors="coerce")
    out["is_sweep"] = out["is_sweep"].fillna(False).astype(bool)
    if "direction_confidence" not in out.columns:
        out["direction_confidence"] = 1.0
    out["direction_confidence"] = pd.to_numeric(out["direction_confidence"], errors="coerce").fillna(0.0)

    # Directional premium weighted by confidence (BID-side discounted)
    conf_premium = out["premium"] * out["direction_confidence"]

    # Direction-specific components
    out["bullish_premium_component"] = conf_premium.where(out["direction"] == "LONG", 0.0)
    out["bearish_premium_component"] = conf_premium.where(out["direction"] == "SHORT", 0.0)

    out["bullish_count_component"] = (out["direction"] == "LONG").astype(int)
    out["bearish_count_component"] = (out["direction"] == "SHORT").astype(int)

    out["bullish_sweep_component"] = (
        (out["direction"] == "LONG") & (out["is_sweep"])
    ).astype(int)
    out["bearish_sweep_component"] = (
        (out["direction"] == "SHORT") & (out["is_sweep"])
    ).astype(int)

    # Wave 2 — repeat-flow acceleration.  Flags trades within the last 2h
    # of the most-recent print in the frame (treats today's last observed
    # event as "now" so backtests are deterministic).  The downstream
    # aggregation converts this into `bullish_repeat_2h` / `bearish_repeat_2h`
    # and derives `*_accel_ratio = repeat_2h / count`.  Ratio centred on
    # 2/6.5≈0.308 (flat-distribution baseline for a 6.5h session).
    if "event_ts" in out.columns and not out["event_ts"].isna().all():
        _ts = pd.to_datetime(out["event_ts"], errors="coerce", utc=True)
        _now_ref = _ts.max()
        _within_2h = (_now_ref - _ts) <= pd.Timedelta(hours=2)
        out["is_within_2h"] = _within_2h.fillna(False)
    else:
        out["is_within_2h"] = False

    out["bullish_repeat_2h_component"] = (
        (out["direction"] == "LONG") & out["is_within_2h"]
    ).astype(int)
    out["bearish_repeat_2h_component"] = (
        (out["direction"] == "SHORT") & out["is_within_2h"]
    ).astype(int)

    out["bullish_sweep_premium_component"] = conf_premium.where(
        (out["direction"] == "LONG") & (out["is_sweep"]),
        0.0,
    )
    out["bearish_sweep_premium_component"] = conf_premium.where(
        (out["direction"] == "SHORT") & (out["is_sweep"]),
        0.0,
    )

    out["bullish_vol_oi_component"] = out["volume_oi_ratio"].where(out["direction"] == "LONG", np.nan)
    out["bearish_vol_oi_component"] = out["volume_oi_ratio"].where(out["direction"] == "SHORT", np.nan)

    out["bullish_trade_premium_component"] = conf_premium.where(out["direction"] == "LONG", np.nan)
    out["bearish_trade_premium_component"] = conf_premium.where(out["direction"] == "SHORT", np.nan)

    out["bullish_repeat_component"] = (out["direction"] == "LONG").astype(int)
    out["bearish_repeat_component"] = (out["direction"] == "SHORT").astype(int)

    # Raw (pre-decay) directional premiums for gate checks
    if "premium_raw" in out.columns:
        raw_conf = pd.to_numeric(out["premium_raw"], errors="coerce").fillna(0.0) * out["direction_confidence"]
        out["bullish_premium_raw_component"] = raw_conf.where(out["direction"] == "LONG", 0.0)
        out["bearish_premium_raw_component"] = raw_conf.where(out["direction"] == "SHORT", 0.0)
    else:
        out["bullish_premium_raw_component"] = out["bullish_premium_component"]
        out["bearish_premium_raw_component"] = out["bearish_premium_component"]

    # Delta-weighted premium components (populated when add_delta_weights ran).
    has_delta = "delta_premium" in out.columns
    if has_delta:
        delta_prem = pd.to_numeric(out["delta_premium"], errors="coerce").fillna(0.0)
        delta_mag = pd.to_numeric(out.get("delta_magnitude", 0.0), errors="coerce").fillna(0.0)
        out["delta_premium_component"] = delta_prem * out["direction_confidence"]
        out["bullish_delta_premium_component"] = out["delta_premium_component"].where(
            out["direction"] == "LONG", 0.0
        )
        out["bearish_delta_premium_component"] = out["delta_premium_component"].where(
            out["direction"] == "SHORT", 0.0
        )
        # Premium-weighted |delta| helpers — numerator is premium*|delta|, denom is premium
        bullish_mask = out["direction"] == "LONG"
        bearish_mask = out["direction"] == "SHORT"
        out["bullish_delta_weight_num"] = (pd.to_numeric(out["premium"], errors="coerce").fillna(0.0) * delta_mag).where(bullish_mask, 0.0)
        out["bearish_delta_weight_num"] = (pd.to_numeric(out["premium"], errors="coerce").fillna(0.0) * delta_mag).where(bearish_mask, 0.0)
        out["bullish_premium_side"] = pd.to_numeric(out["premium"], errors="coerce").fillna(0.0).where(bullish_mask, 0.0)
        out["bearish_premium_side"] = pd.to_numeric(out["premium"], errors="coerce").fillna(0.0).where(bearish_mask, 0.0)
        # Premium-weighted UW-source flag (for source mix): count premium whose delta came from UW.
        is_uw = (out.get("delta_source", "missing") == "uw")
        out["bullish_uw_delta_premium"] = out["bullish_premium_side"].where(is_uw, 0.0)
        out["bearish_uw_delta_premium"] = out["bearish_premium_side"].where(is_uw, 0.0)
    else:
        out["delta_premium_component"] = 0.0
        out["bullish_delta_premium_component"] = 0.0
        out["bearish_delta_premium_component"] = 0.0
        out["bullish_delta_weight_num"] = 0.0
        out["bearish_delta_weight_num"] = 0.0
        out["bullish_premium_side"] = 0.0
        out["bearish_premium_side"] = 0.0
        out["bullish_uw_delta_premium"] = 0.0
        out["bearish_uw_delta_premium"] = 0.0

    grouped = out.groupby("ticker", dropna=False)

    agg = grouped.agg(
        bullish_premium=("bullish_premium_component", "sum"),
        bearish_premium=("bearish_premium_component", "sum"),
        bullish_count=("bullish_count_component", "sum"),
        bearish_count=("bearish_count_component", "sum"),
        total_premium=("premium", "sum"),
        total_count=("ticker", "size"),
        avg_dte=("dte", "mean"),
        avg_volume_oi_ratio=("volume_oi_ratio", "mean"),
        bullish_vol_oi=("bullish_vol_oi_component", "mean"),
        bearish_vol_oi=("bearish_vol_oi_component", "mean"),
        repeat_flow_count=("repeat_flow_count", "max"),
        bullish_sweep_count=("bullish_sweep_component", "sum"),
        bearish_sweep_count=("bearish_sweep_component", "sum"),
        bullish_sweep_premium=("bullish_sweep_premium_component", "sum"),
        bearish_sweep_premium=("bearish_sweep_premium_component", "sum"),
        bullish_max_trade_premium=("bullish_trade_premium_component", "max"),
        bearish_max_trade_premium=("bearish_trade_premium_component", "max"),
        bullish_repeat_count=("bullish_repeat_component", "sum"),
        bearish_repeat_count=("bearish_repeat_component", "sum"),
        bullish_premium_raw=("bullish_premium_raw_component", "sum"),
        bearish_premium_raw=("bearish_premium_raw_component", "sum"),
        bullish_delta_premium_raw=("bullish_delta_premium_component", "sum"),
        bearish_delta_premium_raw=("bearish_delta_premium_component", "sum"),
        bullish_delta_weight_num=("bullish_delta_weight_num", "sum"),
        bearish_delta_weight_num=("bearish_delta_weight_num", "sum"),
        bullish_premium_side=("bullish_premium_side", "sum"),
        bearish_premium_side=("bearish_premium_side", "sum"),
        bullish_uw_delta_premium=("bullish_uw_delta_premium", "sum"),
        bearish_uw_delta_premium=("bearish_uw_delta_premium", "sum"),
        # Wave 2 — 2h repeat-flow counts per direction.
        bullish_repeat_2h=("bullish_repeat_2h_component", "sum"),
        bearish_repeat_2h=("bearish_repeat_2h_component", "sum"),
    ).reset_index()

    # Fill direction-specific nulls
    fill_zero_cols = [
        "bullish_premium",
        "bearish_premium",
        "bullish_count",
        "bearish_count",
        "total_premium",
        "total_count",
        "repeat_flow_count",
        "bullish_sweep_count",
        "bearish_sweep_count",
        "bullish_sweep_premium",
        "bearish_sweep_premium",
        "bullish_max_trade_premium",
        "bearish_max_trade_premium",
        "bullish_repeat_count",
        "bearish_repeat_count",
        "bullish_premium_raw",
        "bearish_premium_raw",
        "bullish_delta_premium_raw",
        "bearish_delta_premium_raw",
        "bullish_delta_weight_num",
        "bearish_delta_weight_num",
        "bullish_premium_side",
        "bearish_premium_side",
        "bullish_uw_delta_premium",
        "bearish_uw_delta_premium",
        "bullish_repeat_2h",
        "bearish_repeat_2h",
    ]
    agg[fill_zero_cols] = agg[fill_zero_cols].fillna(0)

    # Core directional metrics
    agg["net_premium_bias"] = agg["bullish_premium"] - agg["bearish_premium"]
    agg["dominant_direction"] = agg["net_premium_bias"].apply(
        lambda x: "LONG" if x > 0 else ("SHORT" if x < 0 else "NEUTRAL")
    )

    # Premium per trade
    agg["bullish_premium_per_trade"] = np.where(
        agg["bullish_count"] > 0,
        agg["bullish_premium"] / agg["bullish_count"],
        0.0,
    )
    agg["bearish_premium_per_trade"] = np.where(
        agg["bearish_count"] > 0,
        agg["bearish_premium"] / agg["bearish_count"],
        0.0,
    )

    # Breadth: 1 - (max single trade / total directional premium).
    # Higher breadth = many evenly-sized trades = stronger confirmation.
    agg["bullish_breadth"] = 1.0 - np.where(
        agg["bullish_premium"] > 0,
        agg["bullish_max_trade_premium"] / agg["bullish_premium"],
        1.0,
    )
    agg["bearish_breadth"] = 1.0 - np.where(
        agg["bearish_premium"] > 0,
        agg["bearish_max_trade_premium"] / agg["bearish_premium"],
        1.0,
    )

    # Flow imbalance ratio (capped at 99 to avoid inf when one side is zero)
    agg["flow_imbalance_ratio"] = np.where(
        agg["bearish_premium"] > 0,
        np.minimum(agg["bullish_premium"] / agg["bearish_premium"], 99.0),
        np.where(agg["bullish_premium"] > 0, 99.0, 1.0),
    )

    # DTE score
    agg["dte_score"] = agg["avg_dte"].apply(_dte_score)

    # ------------------------------------------------------------------
    # Wave 2 — repeat-flow acceleration.
    # `*_accel_ratio`   = fraction of today's directional prints that landed
    #                     in the last 2h.  0.308 is the flat-distribution
    #                     reference (last 2h of a 6.5h session).
    # `*_accel_score`   = ratio minus baseline, clipped to [-0.5, +0.7].
    #                     Positive → late-session ramp (accumulation cue).
    #                     Negative → flow died off earlier in the day.
    # ------------------------------------------------------------------
    _FLAT_BASELINE = 2.0 / 6.5  # ~0.308
    agg["bullish_accel_ratio"] = np.where(
        agg["bullish_count"] > 0,
        agg["bullish_repeat_2h"] / agg["bullish_count"],
        0.0,
    )
    agg["bearish_accel_ratio"] = np.where(
        agg["bearish_count"] > 0,
        agg["bearish_repeat_2h"] / agg["bearish_count"],
        0.0,
    )
    agg["bullish_accel_score"] = np.clip(
        agg["bullish_accel_ratio"] - _FLAT_BASELINE, -0.5, 0.7
    )
    agg["bearish_accel_score"] = np.clip(
        agg["bearish_accel_ratio"] - _FLAT_BASELINE, -0.5, 0.7
    )

    # ------------------------------------------------------------------
    # Wave 0.5 A1/A2 — structural enrichment per ticker.
    #   dominant_dte_bucket : premium-weighted dominant DTE bucket (str).
    #   sweep_share         : sweep-count / total-count on the dominant side.
    #   multileg_share      : multileg-count / total-count (all sides).
    # Persisted into screener_snapshots so `compute_multi_day_flow` can
    # consume window-averaged versions and the grade explainer can cite
    # them as reasons.
    # ------------------------------------------------------------------
    from app.config import FLOW_TRACKER_DTE_BUCKETS

    def _bucket_for(dte: float) -> str | None:
        if pd.isna(dte):
            return None
        for label, lo, hi, _mult in FLOW_TRACKER_DTE_BUCKETS:
            if lo <= dte <= hi:
                return label
        return None

    out["_dte_bucket"] = out["dte"].apply(_bucket_for)
    if "is_multileg" not in out.columns:
        out["is_multileg"] = False
    out["is_multileg"] = out["is_multileg"].fillna(False).astype(bool)

    # Dominant DTE bucket = bucket with the largest sum of premium per ticker.
    dte_bucket_premium = (
        out.dropna(subset=["_dte_bucket"])
        .groupby(["ticker", "_dte_bucket"])["premium"]
        .sum()
        .reset_index()
    )
    if not dte_bucket_premium.empty:
        idx = dte_bucket_premium.groupby("ticker")["premium"].idxmax()
        dominant = (
            dte_bucket_premium.loc[idx, ["ticker", "_dte_bucket"]]
            .rename(columns={"_dte_bucket": "dominant_dte_bucket"})
        )
        agg = agg.merge(dominant, on="ticker", how="left")
    else:
        agg["dominant_dte_bucket"] = None

    # Sweep & multileg shares.  Sweep share uses the side that owns the row's
    # `dominant_direction` so "bullish sweeps / bullish trades" is a clean
    # read of how aggressive the dominant-side flow is.
    total_count_series = agg["total_count"].replace(0, np.nan)
    bullish_count_safe = agg["bullish_count"].replace(0, np.nan)
    bearish_count_safe = agg["bearish_count"].replace(0, np.nan)
    bullish_sweep_ratio = (agg["bullish_sweep_count"] / bullish_count_safe).fillna(0.0)
    bearish_sweep_ratio = (agg["bearish_sweep_count"] / bearish_count_safe).fillna(0.0)
    agg["sweep_share"] = np.where(
        agg["dominant_direction"] == "SHORT",
        bearish_sweep_ratio,
        bullish_sweep_ratio,
    )

    multileg_per_ticker = out.groupby("ticker")["is_multileg"].sum()
    agg["multileg_count"] = agg["ticker"].map(multileg_per_ticker).fillna(0).astype(int)
    agg["multileg_share"] = (agg["multileg_count"] / total_count_series).fillna(0.0)
    agg[["sweep_share", "multileg_share"]] = agg[["sweep_share", "multileg_share"]].clip(0.0, 1.0).round(4)

    # Market-cap-based flow intensity — premium / marketcap gives a
    # size-normalised measure of how large the directional bet is relative
    # to the company (basis-point scale after * 10_000 in the scorer).
    if "marketcap" in out.columns:
        mcap_per_ticker = (
            pd.to_numeric(out["marketcap"], errors="coerce")
            .groupby(out["ticker"])
            .max()
        )
        agg["marketcap"] = agg["ticker"].map(mcap_per_ticker)
    else:
        agg["marketcap"] = np.nan

    mcap_safe = agg["marketcap"].copy()
    mcap_safe[mcap_safe < 1e8] = np.nan  # ignore ETFs / broken mcap < $100M
    agg["bullish_flow_intensity"] = agg["bullish_premium_raw"] / mcap_safe
    agg["bearish_flow_intensity"] = agg["bearish_premium_raw"] / mcap_safe
    agg[["bullish_flow_intensity", "bearish_flow_intensity"]] = (
        agg[["bullish_flow_intensity", "bearish_flow_intensity"]].fillna(0.0)
    )

    # Delta-weighted intensity: Σ(premium × |delta|) / marketcap. Same mcap
    # floor as flow_intensity so micro-caps / ETFs don't dominate the
    # distribution. Legacy `bullish_flow_intensity` is preserved above so we
    # can shadow-compare in snapshots regardless of USE_DELTA_WEIGHTED_FLOW.
    agg["bullish_delta_intensity"] = agg["bullish_delta_premium_raw"] / mcap_safe
    agg["bearish_delta_intensity"] = agg["bearish_delta_premium_raw"] / mcap_safe
    agg[["bullish_delta_intensity", "bearish_delta_intensity"]] = (
        agg[["bullish_delta_intensity", "bearish_delta_intensity"]].fillna(0.0)
    )

    # Premium-weighted average |delta| per side (0-1 — higher = flow is in
    # the money / higher directional conviction).
    agg["bullish_avg_delta"] = np.where(
        agg["bullish_premium_side"] > 0,
        agg["bullish_delta_weight_num"] / agg["bullish_premium_side"],
        0.0,
    )
    agg["bearish_avg_delta"] = np.where(
        agg["bearish_premium_side"] > 0,
        agg["bearish_delta_weight_num"] / agg["bearish_premium_side"],
        0.0,
    )

    # Share of premium whose delta came from UW (vs. BS proxy). Cheap QA
    # signal — if this collapses to ~0 for A-grades we know we're running
    # on proxy values and should treat the intensity with caution.
    agg["bullish_delta_source_mix"] = np.where(
        agg["bullish_premium_side"] > 0,
        agg["bullish_uw_delta_premium"] / agg["bullish_premium_side"],
        0.0,
    )
    agg["bearish_delta_source_mix"] = np.where(
        agg["bearish_premium_side"] > 0,
        agg["bearish_uw_delta_premium"] / agg["bearish_premium_side"],
        0.0,
    )

    # Premium-per-trade normalised to basis points of market cap
    agg["bullish_ppt_bps"] = agg["bullish_premium_per_trade"] / mcap_safe * 10_000
    agg["bearish_ppt_bps"] = agg["bearish_premium_per_trade"] / mcap_safe * 10_000
    agg[["bullish_ppt_bps", "bearish_ppt_bps"]] = (
        agg[["bullish_ppt_bps", "bearish_ppt_bps"]].fillna(0.0)
    )

    # Round selected columns
    round_cols = [
        "avg_dte",
        "avg_volume_oi_ratio",
        "bullish_vol_oi",
        "bearish_vol_oi",
        "bullish_breadth",
        "bearish_breadth",
        "dte_score",
        "bullish_flow_intensity",
        "bearish_flow_intensity",
        "bullish_delta_intensity",
        "bearish_delta_intensity",
        "bullish_avg_delta",
        "bearish_avg_delta",
        "bullish_delta_source_mix",
        "bearish_delta_source_mix",
    ]
    for col in round_cols:
        if col in agg.columns:
            agg[col] = agg[col].round(6)

    # Drop intermediate scratch columns we only needed for aggregation.
    for _scratch in (
        "bullish_delta_weight_num",
        "bearish_delta_weight_num",
        "bullish_premium_side",
        "bearish_premium_side",
        "bullish_uw_delta_premium",
        "bearish_uw_delta_premium",
    ):
        if _scratch in agg.columns:
            agg.drop(columns=[_scratch], inplace=True)

    # Directional scores — each component normalized to 0–1. When z_stats is
    # provided we use the 4-tier z-score fallback ladder; otherwise we use the
    # legacy absolute-threshold path.
    bull_intensity_col = _active_flow_intensity_col("bullish")
    bear_intensity_col = _active_flow_intensity_col("bearish")

    if z_stats is not None:
        bull_z = _build_z_frame(agg, z_stats, side="bullish")
        bear_z = _build_z_frame(agg, z_stats, side="bearish")

        bull_score, bull_tiers = _weighted_flow_score_mixed(
            agg,
            flow_intensity_col=bull_intensity_col,
            ppt_col="bullish_ppt_bps",
            vol_oi_col="bullish_vol_oi",
            repeat_col="bullish_repeat_count",
            sweep_col="bullish_sweep_count",
            breadth_col="bullish_breadth",
            dte_col="dte_score",
            z_df=bull_z,
        )
        bear_score, bear_tiers = _weighted_flow_score_mixed(
            agg,
            flow_intensity_col=bear_intensity_col,
            ppt_col="bearish_ppt_bps",
            vol_oi_col="bearish_vol_oi",
            repeat_col="bearish_repeat_count",
            sweep_col="bearish_sweep_count",
            breadth_col="bearish_breadth",
            dte_col="dte_score",
            z_df=bear_z,
        )
        agg["bullish_score"] = bull_score
        agg["bearish_score"] = bear_score

        # Per-component tier columns (for detail panels and backtest attribution)
        for comp, tier_col in bull_tiers.items():
            agg[f"bullish_{comp.replace('_tier','')}_tier"] = tier_col.astype(int)
        for comp, tier_col in bear_tiers.items():
            agg[f"bearish_{comp.replace('_tier','')}_tier"] = tier_col.astype(int)

        # Summary: worst tier across the scored side (higher tier int = weaker baseline)
        agg["bullish_zscore_tier"] = bull_tiers.max(axis=1).astype(int)
        agg["bearish_zscore_tier"] = bear_tiers.max(axis=1).astype(int)

        _attach_component_breakdown(
            agg, side="bullish",
            flow_intensity_col=bull_intensity_col,
            ppt_col="bullish_ppt_bps", vol_oi_col="bullish_vol_oi",
            repeat_col="bullish_repeat_count", sweep_col="bullish_sweep_count",
            breadth_col="bullish_breadth", dte_col="dte_score", z_df=bull_z,
        )
        _attach_component_breakdown(
            agg, side="bearish",
            flow_intensity_col=bear_intensity_col,
            ppt_col="bearish_ppt_bps", vol_oi_col="bearish_vol_oi",
            repeat_col="bearish_repeat_count", sweep_col="bearish_sweep_count",
            breadth_col="bearish_breadth", dte_col="dte_score", z_df=bear_z,
        )
    else:
        agg["bullish_score"] = _weighted_flow_score(
            agg,
            flow_intensity_col=bull_intensity_col,
            ppt_col="bullish_ppt_bps",
            vol_oi_col="bullish_vol_oi",
            repeat_col="bullish_repeat_count",
            sweep_col="bullish_sweep_count",
            breadth_col="bullish_breadth",
            dte_col="dte_score",
        )
        agg["bearish_score"] = _weighted_flow_score(
            agg,
            flow_intensity_col=bear_intensity_col,
            ppt_col="bearish_ppt_bps",
            vol_oi_col="bearish_vol_oi",
            repeat_col="bearish_repeat_count",
            sweep_col="bearish_sweep_count",
            breadth_col="bearish_breadth",
            dte_col="dte_score",
        )

        _attach_component_breakdown(
            agg, side="bullish",
            flow_intensity_col=bull_intensity_col,
            ppt_col="bullish_ppt_bps", vol_oi_col="bullish_vol_oi",
            repeat_col="bullish_repeat_count", sweep_col="bullish_sweep_count",
            breadth_col="bullish_breadth", dte_col="dte_score",
        )
        _attach_component_breakdown(
            agg, side="bearish",
            flow_intensity_col=bear_intensity_col,
            ppt_col="bearish_ppt_bps", vol_oi_col="bearish_vol_oi",
            repeat_col="bearish_repeat_count", sweep_col="bearish_sweep_count",
            breadth_col="bearish_breadth", dte_col="dte_score",
        )

    return agg


def rank_flow_candidates(
    agg_df: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Rank bullish and bearish candidates separately from aggregated ticker flow.

    Uses directional score first, then premium and sweep metrics as tie-breakers.
    """
    if agg_df.empty:
        empty = agg_df.copy()
        return {"bullish": empty, "bearish": empty}

    bullish = agg_df[agg_df["bullish_premium"] > 0].copy()
    bearish = agg_df[agg_df["bearish_premium"] > 0].copy()

    bullish = bullish.sort_values(
        by=[
            "bullish_score",
            "bullish_premium",
            "bullish_vol_oi",
            "bullish_sweep_premium",
        ],
        ascending=[False, False, False, False],
    ).head(top_n)

    bearish = bearish.sort_values(
        by=[
            "bearish_score",
            "bearish_premium",
            "bearish_vol_oi",
            "bearish_sweep_premium",
        ],
        ascending=[False, False, False, False],
    ).head(top_n)

    return {
        "bullish": bullish.reset_index(drop=True),
        "bearish": bearish.reset_index(drop=True),
    }


def build_flow_feature_table(
    normalized_df: pd.DataFrame,
    min_premium: float = 500_000,
    min_dte: int = 30,
    max_dte: int = 120,
    require_ask_side: bool = False,
    *,
    z_stats: ZStatsBundle | None = None,
) -> pd.DataFrame:
    """
    Full flow feature pipeline: filter qualifying flow (ASK + BID by default),
    add derived metrics, aggregate by ticker.

    When ``z_stats`` is supplied, directional scoring uses rolling z-scores
    with the 4-tier fallback ladder instead of absolute thresholds.
    """
    filtered = filter_qualifying_flow(
        normalized_df,
        min_premium=min_premium,
        min_dte=min_dte,
        max_dte=max_dte,
        require_ask_side=require_ask_side,
    )

    filtered["premium_raw"] = filtered["premium"]
    filtered = add_recency_weight(filtered)
    filtered = add_volume_oi_ratio(filtered)
    filtered = add_repeat_flow_count(filtered)
    filtered = add_delta_weights(filtered)

    agg = aggregate_flow_by_ticker(filtered, z_stats=z_stats)
    return agg
"""UW-backed 30-day rolling history for flow z-scoring.

The internal ``data/flow_features/`` store only accumulates as we run the
pipeline, so at cold-start most tickers have <5 days of history and fall into
Tier 3 (cross-sectional) or Tier 4 (absolute fallback) in
:mod:`app.features.flow_stats`. This module sidesteps that warm-up period by
hitting UW's ``/stock/{ticker}/options-volume?limit=30`` endpoint to pull each
ticker's own 30-day history on demand.

Supported components (derivable from the options-volume payload):

- ``flow_intensity``   — ``bullish_premium / marketcap`` (needs marketcap)
- ``vol_oi``           — ``call_volume / call_open_interest`` (bearish uses put side)
- ``unusual_premium_share`` — ``bullish_premium / call_premium`` (flagged-vs-total tape)

The remaining z-scored components (premium_per_trade, repeat, sweep, breadth,
dte) cannot be reconstructed from the options-volume payload and keep their
absolute-threshold scoring until the internal ``flow_features/`` history has
enough coverage or a dedicated UW endpoint is wired.

Cache: per-ticker JSON at ``data/uw_ticker_history/{TICKER}.json`` with 24h
TTL. Cache is forward-compatible — old rows missing newer columns simply
resolve to NaN for those components. Stale / missing files trigger a UW
fetch; failures are swallowed so the affected ticker simply falls to Tier 4
rather than breaking the pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from app.config import UW_HISTORY_CACHE_TTL_HOURS, ZSCORE_LOOKBACK_DAYS
from app.vendors.unusual_whales import fetch_ticker_options_history

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "uw_ticker_history"

# Marketcap floor matches ``app.features.flow_features`` — below $100M we drop
# the ticker so ETFs / micro-caps / broken mcap don't poison the baseline.
_MCAP_FLOOR = 1e8

# Minor politeness delay between cache misses. Existing ``_uw_request`` already
# handles 429 backoff; this just keeps us from machine-gunning the API on a
# cold-start run.
_SLEEP_BETWEEN_MISSES_SEC = 0.05


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.json"


def _cache_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    age_sec = time.time() - path.stat().st_mtime
    return age_sec < ttl_hours * 3600.0


def _load_cache(path: Path) -> pd.DataFrame | None:
    try:
        with path.open("r") as fh:
            payload = json.load(fh)
    except Exception:
        return None
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not rows:
        return None
    try:
        return pd.DataFrame(rows)
    except Exception:
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": pd.Timestamp.utcnow().isoformat(),
            "rows": df.to_dict(orient="records"),
        }
        with path.open("w") as fh:
            json.dump(payload, fh)
    except Exception as exc:
        logger.debug("uw_history: failed to write cache for %s: %s", path.name, exc)


def _fetch_with_cache(
    ticker: str,
    *,
    days: int,
    ttl_hours: float,
) -> pd.DataFrame | None:
    """Return cached history if fresh, else hit UW and refresh the cache."""
    path = _cache_path(ticker)

    if _cache_fresh(path, ttl_hours):
        cached = _load_cache(path)
        if cached is not None and not cached.empty:
            return cached

    df = fetch_ticker_options_history(ticker, days=days)
    time.sleep(_SLEEP_BETWEEN_MISSES_SEC)
    if df is None or df.empty:
        return None
    _write_cache(path, df)
    return df


# ---------------------------------------------------------------------------
# Component derivations
# ---------------------------------------------------------------------------

# Components that can be derived from the ``/stock/{t}/options-volume`` payload.
# Each entry maps a logical component name to the (bullish_col, bearish_col)
# aggregate column names that z-scoring consumes, and to a callable that turns
# one historical row (as a pandas DataFrame) + marketcap into those two
# columns.
_SUPPORTED_COMPONENTS: set[str] = {
    "flow_intensity",
    "vol_oi",
    "unusual_premium_share",
}


def _derive_flow_intensity(df: pd.DataFrame, mcap: float) -> pd.DataFrame:
    bullish = pd.to_numeric(df.get("bullish_premium"), errors="coerce") / mcap
    bearish = pd.to_numeric(df.get("bearish_premium"), errors="coerce") / mcap
    return pd.DataFrame(
        {
            "bullish_flow_intensity": bullish,
            "bearish_flow_intensity": bearish,
        }
    )


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    ratio = n / d.where(d > 0)
    return ratio


def _derive_vol_oi(df: pd.DataFrame, _mcap: float) -> pd.DataFrame:
    bullish = _safe_ratio(df.get("call_volume"), df.get("call_open_interest"))
    bearish = _safe_ratio(df.get("put_volume"), df.get("put_open_interest"))
    return pd.DataFrame(
        {
            "bullish_vol_oi": bullish,
            "bearish_vol_oi": bearish,
        }
    )


def _derive_unusual_premium_share(df: pd.DataFrame, _mcap: float) -> pd.DataFrame:
    bullish = _safe_ratio(df.get("bullish_premium"), df.get("call_premium"))
    bearish = _safe_ratio(df.get("bearish_premium"), df.get("put_premium"))
    return pd.DataFrame(
        {
            "bullish_unusual_premium_share": bullish,
            "bearish_unusual_premium_share": bearish,
        }
    )


_COMPONENT_DERIVERS = {
    "flow_intensity": _derive_flow_intensity,
    "vol_oi": _derive_vol_oi,
    "unusual_premium_share": _derive_unusual_premium_share,
}


def load_uw_baselines(
    tickers: list[str],
    marketcap_map: dict[str, float],
    *,
    components: list[str] | None = None,
    lookback_days: int = ZSCORE_LOOKBACK_DAYS,
    ttl_hours: float | None = None,
) -> pd.DataFrame:
    """Build a 30-day history frame with columns for each requested component.

    Returns a DataFrame shaped like :func:`app.features.flow_stats.load_history`
    output with ``ticker``, ``date``, and two columns per component
    (``bullish_{comp}`` and ``bearish_{comp}``).

    ``components`` defaults to ``["flow_intensity"]`` for backwards compat.
    Supported values: ``flow_intensity``, ``vol_oi``, ``unusual_premium_share``.

    ``flow_intensity`` is derived as ``bullish_premium / marketcap_today`` for
    each historical row (marketcap is not backfilled — today's cap is used as
    the denominator for all 30 days; drift is <5% typical over that window for
    tickers above the floor).

    Tickers with ``marketcap < 1e8`` or missing mcap are dropped **only when
    ``flow_intensity`` is among the requested components**; the other
    components do not depend on marketcap so the filter is skipped otherwise.
    Tickers that UW has no data for are skipped silently (they'll fall to
    Tier 4 in the caller's z-score ladder).
    """
    if ttl_hours is None:
        ttl_hours = float(UW_HISTORY_CACHE_TTL_HOURS)

    if components is None:
        components = ["flow_intensity"]

    unknown = [c for c in components if c not in _SUPPORTED_COMPONENTS]
    if unknown:
        raise ValueError(
            f"uw_history: unsupported components {unknown}. "
            f"Supported: {sorted(_SUPPORTED_COMPONENTS)}"
        )

    need_mcap = "flow_intensity" in components

    frames: list[pd.DataFrame] = []
    skipped_mcap = 0
    skipped_api = 0
    output_cols = ["ticker", "date"] + [
        col
        for comp in components
        for col in (f"bullish_{comp}", f"bearish_{comp}")
    ]

    for raw_ticker in tickers:
        ticker = str(raw_ticker).upper().strip()
        if not ticker:
            continue

        mcap = marketcap_map.get(ticker)
        mcap_valid = mcap is not None and pd.notna(mcap) and mcap >= _MCAP_FLOOR
        if need_mcap and not mcap_valid:
            skipped_mcap += 1
            continue
        # For non-mcap components use NaN — _derive_* that need mcap bail out
        # with NaN outputs naturally via division.
        mcap_for_deriv = float(mcap) if mcap_valid else float("nan")

        hist = _fetch_with_cache(ticker, days=lookback_days, ttl_hours=ttl_hours)
        if hist is None or hist.empty:
            skipped_api += 1
            continue

        df = hist.copy()
        df["ticker"] = ticker
        for comp in components:
            derived = _COMPONENT_DERIVERS[comp](df, mcap_for_deriv)
            for col in derived.columns:
                df[col] = derived[col].to_numpy()
        frames.append(df[output_cols])

    if not frames:
        logger.info(
            "uw_history: no history hydrated (components=%s, mcap_filtered=%d, "
            "api_missing=%d)",
            components,
            skipped_mcap,
            skipped_api,
        )
        return pd.DataFrame(columns=output_cols)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    logger.info(
        "uw_history: hydrated %d tickers (%d rows; components=%s); "
        "mcap_filtered=%d, api_missing=%d",
        out["ticker"].nunique(),
        len(out),
        components,
        skipped_mcap,
        skipped_api,
    )
    return out.reset_index(drop=True)


def load_uw_intensity_history(
    tickers: list[str],
    marketcap_map: dict[str, float],
    *,
    lookback_days: int = ZSCORE_LOOKBACK_DAYS,
    ttl_hours: float | None = None,
) -> pd.DataFrame:
    """Backwards-compatible wrapper around :func:`load_uw_baselines`.

    Returns a DataFrame with columns
    ``ticker, date, bullish_flow_intensity, bearish_flow_intensity`` — identical
    shape to the pre-refactor output. New call sites should prefer
    :func:`load_uw_baselines` directly.
    """
    return load_uw_baselines(
        tickers,
        marketcap_map,
        components=["flow_intensity"],
        lookback_days=lookback_days,
        ttl_hours=ttl_hours,
    )

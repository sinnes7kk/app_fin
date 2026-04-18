"""UW-backed 30-day rolling history for flow-intensity z-scoring.

The internal ``data/flow_features/`` store only accumulates as we run the
pipeline, so at cold-start most tickers have <5 days of history and fall into
Tier 3 (cross-sectional) or Tier 4 (absolute fallback) in
:mod:`app.features.flow_stats`. This module sidesteps that warm-up period by
hitting UW's ``/stock/{ticker}/options-volume?limit=30`` endpoint to pull each
ticker's own 30-day premium history on demand.

Scope: we only hydrate **flow_intensity** (``bullish_premium_raw / marketcap``
and its bearish twin). The other z-scored components (ppt, vol_oi, repeat,
sweep, breadth, dte) cannot be reconstructed from the options-volume payload
and keep their absolute-threshold scoring via the ``components=["flow_intensity"]``
gate in :func:`app.features.flow_features.rescore_with_z`.

Cache: per-ticker JSON at ``data/uw_ticker_history/{TICKER}.json`` with 24h
TTL. Stale / missing files trigger a UW fetch; failures are swallowed so the
affected ticker simply falls to Tier 4 rather than breaking the pipeline.
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


def load_uw_intensity_history(
    tickers: list[str],
    marketcap_map: dict[str, float],
    *,
    lookback_days: int = ZSCORE_LOOKBACK_DAYS,
    ttl_hours: float | None = None,
) -> pd.DataFrame:
    """Build a 30-day ``flow_intensity`` history frame for the given tickers.

    Returns a DataFrame matching the shape
    :func:`app.features.flow_stats.load_history` produces, with columns::

        ticker, date, bullish_flow_intensity, bearish_flow_intensity

    ``flow_intensity`` is derived as ``bullish_premium / marketcap_today`` for
    each historical row (marketcap is not backfilled — today's cap is used as
    the denominator for all 30 days; drift is <5% typical over that window for
    tickers above the floor).

    Tickers with ``marketcap < 1e8`` or missing mcap are dropped. Tickers that
    UW has no data for are skipped silently (they'll fall to Tier 4 in the
    caller's z-score ladder).
    """
    if ttl_hours is None:
        ttl_hours = float(UW_HISTORY_CACHE_TTL_HOURS)

    frames: list[pd.DataFrame] = []
    skipped_mcap = 0
    skipped_api = 0

    for raw_ticker in tickers:
        ticker = str(raw_ticker).upper().strip()
        if not ticker:
            continue

        mcap = marketcap_map.get(ticker)
        if mcap is None or not pd.notna(mcap) or mcap < _MCAP_FLOOR:
            skipped_mcap += 1
            continue

        hist = _fetch_with_cache(ticker, days=lookback_days, ttl_hours=ttl_hours)
        if hist is None or hist.empty:
            skipped_api += 1
            continue

        df = hist.copy()
        df["ticker"] = ticker
        df["bullish_flow_intensity"] = (
            pd.to_numeric(df.get("bullish_premium"), errors="coerce") / mcap
        )
        df["bearish_flow_intensity"] = (
            pd.to_numeric(df.get("bearish_premium"), errors="coerce") / mcap
        )
        frames.append(
            df[["ticker", "date", "bullish_flow_intensity", "bearish_flow_intensity"]]
        )

    if not frames:
        logger.info(
            "uw_history: no history hydrated (mcap_filtered=%d, api_missing=%d)",
            skipped_mcap,
            skipped_api,
        )
        return pd.DataFrame(
            columns=["ticker", "date", "bullish_flow_intensity", "bearish_flow_intensity"]
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    logger.info(
        "uw_history: hydrated %d tickers (%d rows); mcap_filtered=%d, api_missing=%d",
        out["ticker"].nunique(),
        len(out),
        skipped_mcap,
        skipped_api,
    )
    return out.reset_index(drop=True)

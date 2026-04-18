"""Wave 8 — daily-refreshed market indicators feeding the Risk Regime.

Exposes ``fetch_market_indicators`` which returns a small dict of the
signals the Risk Regime consumes:

  - ``vix``          — ^VIX close
  - ``vix3m``        — ^VIX3M close (VIX term-structure proxy)
  - ``spy_rsi``      — SPY 14-day RSI (Wilder)
  - ``spy_trend``    — BULLISH / BEARISH / NEUTRAL from EMAs (re-uses
                       ``market_regime`` internals)
  - ``spy_close``    — latest SPY close (informational)

Results are cached to ``data/market_indicators.json`` for
``MARKET_INDICATORS_CACHE_TTL_HOURS`` to avoid hammering Yahoo on every
page render.  The cache is tolerant: if it's stale or unreadable we
just re-fetch.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import (
    MARKET_INDICATORS_CACHE_PATH,
    MARKET_INDICATORS_CACHE_TTL_HOURS,
)


def _rsi_wilder(closes: pd.Series, period: int = 14) -> float | None:
    """Return the Wilder RSI for the last bar of ``closes``."""
    if closes is None or len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    if val is None or pd.isna(val):
        return None
    try:
        return round(float(val), 2)
    except (TypeError, ValueError):
        return None


def _cache_is_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return (datetime.now(tz=timezone.utc) - mtime) < timedelta(hours=ttl_hours)


def _read_cache(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return None
        return payload
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError:
        pass


def _fetch_close_series(ticker: str, lookback_days: int) -> pd.Series | None:
    try:
        from app.features.price_features import clean_ohlcv, fetch_ohlcv

        df = clean_ohlcv(fetch_ohlcv(ticker, lookback_days=lookback_days, include_partial=False))
        if df is None or df.empty:
            return None
        return df["close"]
    except Exception:
        return None


def fetch_market_indicators(
    *,
    force_refresh: bool = False,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the market-indicator payload used by the Risk Regime.

    Falls back gracefully on any per-ticker fetch failure — missing
    fields simply come back as ``None``.
    """
    cache = Path(cache_path) if cache_path is not None else Path(MARKET_INDICATORS_CACHE_PATH)
    if not force_refresh and _cache_is_fresh(cache, MARKET_INDICATORS_CACHE_TTL_HOURS):
        cached = _read_cache(cache)
        if cached is not None:
            cached.setdefault("cache_hit", True)
            return cached

    payload: dict[str, Any] = {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "vix": None,
        "vix3m": None,
        "spy_rsi": None,
        "spy_trend": None,
        "spy_close": None,
        "cache_hit": False,
    }

    # VIX + VIX3M
    vix = _fetch_close_series("^VIX", lookback_days=10)
    if vix is not None and not vix.empty:
        try:
            payload["vix"] = round(float(vix.iloc[-1]), 2)
        except (TypeError, ValueError):
            pass
    vix3m = _fetch_close_series("^VIX3M", lookback_days=10)
    if vix3m is not None and not vix3m.empty:
        try:
            payload["vix3m"] = round(float(vix3m.iloc[-1]), 2)
        except (TypeError, ValueError):
            pass

    # SPY RSI + trend
    spy_close_series = _fetch_close_series("SPY", lookback_days=90)
    if spy_close_series is not None and not spy_close_series.empty:
        try:
            payload["spy_close"] = round(float(spy_close_series.iloc[-1]), 2)
        except (TypeError, ValueError):
            pass
        payload["spy_rsi"] = _rsi_wilder(spy_close_series)

    # SPY trend — re-use market_regime's alignment computation when possible
    try:
        from app.features.market_regime import fetch_market_regime

        mr = fetch_market_regime()
        payload["spy_trend"] = mr.get("spy_trend")
        if payload["vix"] is None and mr.get("vix_close") is not None:
            payload["vix"] = mr.get("vix_close")
        if payload["spy_close"] is None and mr.get("spy_close") is not None:
            payload["spy_close"] = mr.get("spy_close")
    except Exception:
        # market_regime is optional — if it fails we just leave trend None.
        pass

    _write_cache(cache, payload)
    return payload


__all__ = ["fetch_market_indicators"]

"""Options gamma / OI context from the Unusual Whales API.

Produces a per-ticker context dict with gamma regime, OI walls, and DTE
bucketing that scoring uses to adjust conviction.

All API failures are handled gracefully — individual helpers return partial
results and the public function falls back to neutral defaults.
"""

from __future__ import annotations

from datetime import date, datetime

import requests

from app.config import UNUSUAL_WHALES_API_KEY

BASE_URL = "https://api.unusualwhales.com/api"

# Tunable: net_gex values inside this band are classified as NEUTRAL.
# Start conservatively; tighten after observing real magnitudes per ticker.
NEUTRAL_GEX_THRESHOLD = 0.0

_CONTEXT_CACHE: dict[str, dict] = {}


def clear_context_cache() -> None:
    """Reset the per-run cache.  Call at the start of each pipeline run."""
    _CONTEXT_CACHE.clear()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {UNUSUAL_WHALES_API_KEY}",
        "Accept": "application/json",
    }


def _empty_context() -> dict:
    """Baseline dict returned when all sources fail."""
    return {
        "options_context_available": False,
        "options_context_sources_used": [],
        "gamma_regime": "NEUTRAL",
        "gamma_flip_level_estimate": None,
        "net_gex": None,
        "nearest_call_wall": None,
        "nearest_put_wall": None,
        "distance_to_call_wall_pct": None,
        "distance_to_put_wall_pct": None,
        "ticker_call_oi": None,
        "ticker_put_oi": None,
        "ticker_put_call_ratio": None,
        "near_term_oi": None,
        "swing_dte_oi": None,
        "long_dated_oi": None,
    }


# ---------------------------------------------------------------------------
# Private helpers — one per UW endpoint
# ---------------------------------------------------------------------------

def _fetch_gex_context(ticker: str) -> dict | None:
    """Spot GEX by strike -> gamma regime, net GEX, flip level estimate."""
    url = f"{BASE_URL}/stock/{ticker}/spot-exposures/strike"
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception:
        return None

    if not rows:
        return None

    strikes: list[tuple[float, float]] = []
    for r in rows:
        try:
            strike = float(r.get("price") or r.get("strike", 0))
            call_g = float(r.get("call_gamma_oi", 0))
            put_g = float(r.get("put_gamma_oi", 0))
            net = call_g + put_g
            strikes.append((strike, net))
        except (TypeError, ValueError):
            continue

    if not strikes:
        return None

    strikes.sort(key=lambda x: x[0])

    net_gex = sum(g for _, g in strikes)

    if abs(net_gex) <= NEUTRAL_GEX_THRESHOLD:
        regime = "NEUTRAL"
    elif net_gex > 0:
        regime = "POSITIVE"
    else:
        regime = "NEGATIVE"

    # Gamma flip estimate: strike where cumulative GEX crosses zero
    flip_level: float | None = None
    cum = 0.0
    prev_strike = strikes[0][0]
    for strike, gamma in strikes:
        prev_cum = cum
        cum += gamma
        if prev_cum != 0.0 and ((prev_cum > 0 and cum <= 0) or (prev_cum < 0 and cum >= 0)):
            flip_level = strike
            break
        prev_strike = strike

    return {
        "net_gex": round(net_gex, 2),
        "gamma_regime": regime,
        "gamma_flip_level_estimate": round(flip_level, 2) if flip_level is not None else None,
    }


def _fetch_oi_walls(ticker: str, spot: float) -> dict | None:
    """OI per strike -> nearest call/put walls, distances, P/C ratio."""
    url = f"{BASE_URL}/stock/{ticker}/oi-per-strike"
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception:
        return None

    if not rows:
        return None

    total_call_oi = 0
    total_put_oi = 0
    best_call_wall: tuple[float, int] | None = None  # (strike, oi)
    best_put_wall: tuple[float, int] | None = None

    for r in rows:
        try:
            strike = float(r["strike"])
            call_oi = int(r.get("call_oi", 0))
            put_oi = int(r.get("put_oi", 0))
        except (TypeError, ValueError, KeyError):
            continue

        total_call_oi += call_oi
        total_put_oi += put_oi

        if strike >= spot:
            if best_call_wall is None or call_oi > best_call_wall[1]:
                best_call_wall = (strike, call_oi)

        if strike <= spot:
            if best_put_wall is None or put_oi > best_put_wall[1]:
                best_put_wall = (strike, put_oi)

    call_wall_strike = best_call_wall[0] if best_call_wall else None
    put_wall_strike = best_put_wall[0] if best_put_wall else None

    dist_call = round((call_wall_strike - spot) / spot * 100, 2) if call_wall_strike and spot > 0 else None
    dist_put = round((spot - put_wall_strike) / spot * 100, 2) if put_wall_strike and spot > 0 else None
    pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

    return {
        "nearest_call_wall": call_wall_strike,
        "nearest_put_wall": put_wall_strike,
        "distance_to_call_wall_pct": dist_call,
        "distance_to_put_wall_pct": dist_put,
        "ticker_call_oi": total_call_oi,
        "ticker_put_oi": total_put_oi,
        "ticker_put_call_ratio": pcr,
    }


def _fetch_expiry_context(ticker: str) -> dict | None:
    """Volume & OI per expiry -> DTE-bucketed OI totals."""
    url = f"{BASE_URL}/stock/{ticker}/option/volume-oi-expiry"
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception:
        return None

    if not rows:
        return None

    today = date.today()
    near_term = 0
    swing = 0
    long_dated = 0

    for r in rows:
        try:
            expiry_str = r.get("expires") or r.get("expiry", "")
            oi = int(r.get("oi", 0) or 0)
        except (TypeError, ValueError):
            continue

        try:
            expiry_date = datetime.strptime(expiry_str[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        dte = (expiry_date - today).days
        if dte < 0:
            continue

        if dte <= 30:
            near_term += oi
        elif dte <= 90:
            swing += oi
        else:
            long_dated += oi

    return {
        "near_term_oi": near_term,
        "swing_dte_oi": swing,
        "long_dated_oi": long_dated,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_options_context(ticker: str, spot: float) -> dict:
    """Return a complete options context dict for *ticker* at *spot* price.

    Results are cached per-run so each ticker triggers at most 3 API calls
    across an entire pipeline execution.
    """
    if ticker in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[ticker]

    ctx = _empty_context()
    sources: list[str] = []

    gex = _fetch_gex_context(ticker)
    if gex is not None:
        ctx.update(gex)
        sources.append("spot_gex")

    walls = _fetch_oi_walls(ticker, spot)
    if walls is not None:
        ctx.update(walls)
        sources.append("oi_strike")

    expiry = _fetch_expiry_context(ticker)
    if expiry is not None:
        ctx.update(expiry)
        sources.append("oi_expiry")

    ctx["options_context_sources_used"] = sources
    ctx["options_context_available"] = len(sources) > 0

    _CONTEXT_CACHE[ticker] = ctx
    return ctx

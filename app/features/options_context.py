"""Options gamma / OI context from the Unusual Whales API.

Produces a per-ticker context dict with gamma regime, OI walls, and DTE
bucketing that scoring uses to adjust conviction.

All API failures are handled gracefully — individual helpers return partial
results and the public function falls back to neutral defaults.
"""

from __future__ import annotations

from datetime import date, datetime

from app.vendors.unusual_whales import BASE_URL, _uw_request

# Tunable: net_gex values inside this band are classified as NEUTRAL.
# Start conservatively; tighten after observing real magnitudes per ticker.
NEUTRAL_GEX_THRESHOLD = 0.0

_CONTEXT_CACHE: dict[str, dict] = {}


def clear_context_cache() -> None:
    """Reset the per-run cache.  Call at the start of each pipeline run."""
    _CONTEXT_CACHE.clear()


def _empty_context() -> dict:
    """Baseline dict returned when all sources fail."""
    return {
        "options_context_available": False,
        "options_context_sources_used": [],
        "gamma_regime": "NEUTRAL",
        "gamma_flip_level_estimate": None,
        "net_gex": None,
        # Wave 2 — dealer hedge bias derived from net_gex sign/magnitude.
        # "suppress" = positive net gamma → dealers short vol / pin price;
        # "chase"    = negative net gamma → dealers long vol / amplify moves;
        # "neutral"  = |net_gex| too small to matter.
        "dealer_hedge_bias": "neutral",
        "dealer_hedge_label": None,
        # Wave 2 — pin-risk strike: highest-concentration gamma level within
        # ±5% of spot.  When the dominant strike carries a meaningful share
        # of the total gamma near spot, price tends to magnetise toward it
        # (especially into weekly expiry).
        "pin_risk_strike": None,
        "pin_risk_distance_pct": None,
        "pin_risk_concentration": None,  # share of near-spot gamma on pin strike (0-1)
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
        # Aggregated daily options volume/premium
        "daily_bullish_premium": None,
        "daily_bearish_premium": None,
        "daily_premium_bias": None,
        "call_volume_today": None,
        "put_volume_today": None,
        "call_volume_vs_30d_avg": None,
        "put_volume_vs_30d_avg": None,
        "call_ask_bid_ratio": None,
        "put_ask_bid_ratio": None,
        # Implied volatility context
        "iv_rank": None,
        "iv_current": None,
        # Wave 2 — rolling IV-rank delta (current minus oldest sample in the
        # 5d window) persisted via app.features.iv_rank_history.  Positive =
        # vol expanding (options getting richer); negative = vol contracting
        # (options getting cheaper / IV crush setup).
        "iv_rank_5d_delta": None,
        "iv_rank_5d_samples": 0,
    }


# ---------------------------------------------------------------------------
# Private helpers — one per UW endpoint
# ---------------------------------------------------------------------------

# Wave 2 — pin-risk tuning.
# Any strike within ±PIN_RISK_BAND_PCT of spot competes for "pin" status.
# A strike only qualifies as a pin-risk level when it carries at least
# PIN_RISK_MIN_CONCENTRATION of the absolute gamma in that band (otherwise
# the flow is diffuse and no single strike acts as a magnet).
PIN_RISK_BAND_PCT = 5.0
PIN_RISK_MIN_CONCENTRATION = 0.25


def _fetch_gex_context(ticker: str, spot: float | None = None) -> dict | None:
    """Spot GEX by strike -> gamma regime, net GEX, flip level estimate,
    dealer hedge bias, and (when *spot* is supplied) pin-risk strike."""
    url = f"{BASE_URL}/stock/{ticker}/spot-exposures/strike"
    try:
        resp = _uw_request(url)
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

    # Wave 2 — dealer hedge bias.  Translates the raw net_gex sign into a
    # trader-facing label:
    #   positive net_gex → dealers are NET LONG gamma → they SELL rallies
    #                      and BUY dips → dampens moves → "suppress".
    #   negative net_gex → dealers are NET SHORT gamma → they BUY rallies
    #                      and SELL dips → amplifies moves → "chase".
    if regime == "POSITIVE":
        dealer_bias = "suppress"
        dealer_label = "Dealers suppress (pinning)"
    elif regime == "NEGATIVE":
        dealer_bias = "chase"
        dealer_label = "Dealers chase (volatility amplified)"
    else:
        dealer_bias = "neutral"
        dealer_label = "Dealer gamma balanced"

    # Gamma flip estimate: strike where cumulative GEX crosses zero
    flip_level: float | None = None
    cum = 0.0
    for strike, gamma in strikes:
        prev_cum = cum
        cum += gamma
        if prev_cum != 0.0 and ((prev_cum > 0 and cum <= 0) or (prev_cum < 0 and cum >= 0)):
            flip_level = strike
            break

    # Wave 2 — pin-risk strike.  Look at strikes within ±PIN_RISK_BAND_PCT
    # of spot, find the one with the largest |gamma|, and only surface it
    # when its share of the near-spot gamma is high enough to actually
    # magnetise price (PIN_RISK_MIN_CONCENTRATION).
    pin_strike: float | None = None
    pin_distance_pct: float | None = None
    pin_concentration: float | None = None
    if spot is not None and spot > 0:
        lo = spot * (1 - PIN_RISK_BAND_PCT / 100.0)
        hi = spot * (1 + PIN_RISK_BAND_PCT / 100.0)
        band = [(s, g) for (s, g) in strikes if lo <= s <= hi]
        total_abs = sum(abs(g) for _, g in band)
        if band and total_abs > 0:
            dom_strike, dom_gamma = max(band, key=lambda pair: abs(pair[1]))
            concentration = abs(dom_gamma) / total_abs
            if concentration >= PIN_RISK_MIN_CONCENTRATION:
                pin_strike = round(dom_strike, 2)
                pin_distance_pct = round((dom_strike - spot) / spot * 100.0, 2)
                pin_concentration = round(concentration, 3)

    return {
        "net_gex": round(net_gex, 2),
        "gamma_regime": regime,
        "gamma_flip_level_estimate": round(flip_level, 2) if flip_level is not None else None,
        "dealer_hedge_bias": dealer_bias,
        "dealer_hedge_label": dealer_label,
        "pin_risk_strike": pin_strike,
        "pin_risk_distance_pct": pin_distance_pct,
        "pin_risk_concentration": pin_concentration,
    }


def _fetch_oi_walls(ticker: str, spot: float) -> dict | None:
    """OI per strike -> nearest call/put walls, distances, P/C ratio."""
    url = f"{BASE_URL}/stock/{ticker}/oi-per-strike"
    try:
        resp = _uw_request(url)
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
        resp = _uw_request(url)
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


def _fetch_options_volume(ticker: str) -> dict | None:
    """Aggregated daily options volume and premium from /options-volume.

    Returns derived metrics for premium bias, volume unusualness, and
    ask/bid aggressiveness ratios.  Requests the latest single day only.
    """
    url = f"{BASE_URL}/stock/{ticker}/options-volume"
    try:
        resp = _uw_request(url, params={"limit": 1})
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception:
        return None

    if not rows:
        return None

    r = rows[0]

    def _float(key: str) -> float:
        try:
            return float(r.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    bull_prem = _float("bullish_premium")
    bear_prem = _float("bearish_premium")
    call_vol = _float("call_volume")
    put_vol = _float("put_volume")
    call_vol_avg = _float("avg_30_day_call_volume")
    put_vol_avg = _float("avg_30_day_put_volume")
    call_ask = _float("call_open_ask_vol")
    call_bid = _float("call_open_bid_vol")
    put_ask = _float("put_open_ask_vol")
    put_bid = _float("put_open_bid_vol")

    return {
        "daily_bullish_premium": round(bull_prem, 2),
        "daily_bearish_premium": round(bear_prem, 2),
        "daily_premium_bias": round(bull_prem - bear_prem, 2),
        "call_volume_today": round(call_vol),
        "put_volume_today": round(put_vol),
        "call_volume_vs_30d_avg": round(call_vol / call_vol_avg, 2) if call_vol_avg > 0 else None,
        "put_volume_vs_30d_avg": round(put_vol / put_vol_avg, 2) if put_vol_avg > 0 else None,
        "call_ask_bid_ratio": round(call_ask / call_bid, 2) if call_bid > 0 else None,
        "put_ask_bid_ratio": round(put_ask / put_bid, 2) if put_bid > 0 else None,
    }


def _fetch_iv_context(ticker: str) -> dict | None:
    """Interpolated IV and rank from /interpolated-iv.

    The UW endpoint returns rows keyed by DTE horizon (1, 5, 7, 14, 30, 60,
    90, 180, 365 days).  We use the **30-day** row as the standard IV rank
    reference.  Fields returned by the API:
      - ``percentile``  (0-1 scale, we convert to 0-100)
      - ``volatility``  (annualised ATM implied vol)
    """
    url = f"{BASE_URL}/stock/{ticker}/interpolated-iv"
    try:
        resp = _uw_request(url)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return None

    if not data or not isinstance(data, list):
        return None

    # Prefer the 30-day horizon; fall back to closest available
    row = None
    for r in data:
        if r.get("days") == 30:
            row = r
            break
    if row is None:
        row = data[0]

    def _safe_float(key: str) -> float | None:
        val = row.get(key)
        if val is None:
            return None
        try:
            f = float(val)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    pct = _safe_float("percentile")
    iv_rank = round(pct * 100, 2) if pct is not None else None
    iv_current = _safe_float("volatility")

    if iv_rank is None and iv_current is None:
        return None

    return {
        "iv_rank": iv_rank,
        "iv_current": round(iv_current, 4) if iv_current is not None else None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_options_context(ticker: str, spot: float) -> dict:
    """Return a complete options context dict for *ticker* at *spot* price.

    Results are cached per-run so each ticker triggers at most 5 API calls
    (spot GEX, OI/strike, OI/expiry, options volume, interpolated IV)
    across an entire pipeline execution.
    """
    if ticker in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[ticker]

    ctx = _empty_context()
    sources: list[str] = []

    gex = _fetch_gex_context(ticker, spot)
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

    vol = _fetch_options_volume(ticker)
    if vol is not None:
        ctx.update(vol)
        sources.append("options_volume")

    iv = _fetch_iv_context(ticker)
    if iv is not None:
        ctx.update(iv)
        sources.append("interpolated_iv")

        # Wave 2 — record today's IV rank into the rolling history store
        # and attach the rolling 5d delta to the context.  Failures in the
        # history layer must never break the options-context pipeline.
        try:
            from app.features.iv_rank_history import (
                compute_iv_rank_delta,
                record_iv_rank,
            )
            record_iv_rank(ticker, ctx.get("iv_rank"))
            _cur, _delta, _n = compute_iv_rank_delta(ticker)
            ctx["iv_rank_5d_delta"] = _delta
            ctx["iv_rank_5d_samples"] = _n
        except Exception:
            pass

    ctx["options_context_sources_used"] = sources
    ctx["options_context_available"] = len(sources) > 0

    _CONTEXT_CACHE[ticker] = ctx
    return ctx

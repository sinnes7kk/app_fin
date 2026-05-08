"""UW endpoint orchestration for the feature lab.

Fans out the 6 research-backed UW endpoints (greek-exposure, iv-skew,
atm-iv-term, expiry-breakdown, max-pain, spot-exposures) per ticker and
caches the merged result per-day on disk so a re-run within the same
calendar day is free.  This module is the only place feature_lab.py
talks to UW directly.

Cache layout
------------
``data/feature_lab_cache/{YYYY-MM-DD}/{TICKER}.json`` — a flat dict
with the 12 UW feature columns. Negative results (i.e. UW returned no
data) are cached as ``{"_status": "no_data"}`` so we don't hammer
endpoints that consistently fail for a given ticker.
"""

from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "feature_lab_cache"

UW_FEATURE_COLS = (
    "gex_total",
    "vanna_total",
    "charm_total",
    "iv_skew_25d",
    "atm_iv_30d",
    "atm_iv_60d",
    "atm_iv_90d",
    "term_slope_30_90",
    "expiry_concentration_top1",
    "max_pain_dist_pct",
    "dealer_net_delta_at_spot",
    "dealer_net_gamma_at_spot",
)


def _cache_path(ticker: str, day: _date | None = None) -> Path:
    day = day or _date.today()
    safe = "".join(c for c in str(ticker or "").upper() if c.isalnum() or c in "._-")
    return CACHE_DIR / day.isoformat() / f"{safe}.json"


def _read_cache(ticker: str, day: _date | None = None) -> dict | None:
    p = _cache_path(ticker, day)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_cache(ticker: str, payload: dict, day: _date | None = None) -> None:
    p = _cache_path(ticker, day)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, default=str))
    except Exception:
        pass  # cache failures are non-fatal


def fetch_uw_features(ticker: str, spot: float | None = None) -> dict:
    """Fetch the 12 UW feature columns for a single ticker.

    Day-cached on disk. Returns a dict with all UW_FEATURE_COLS keys
    (values may be None where UW returned nothing).
    """
    cached = _read_cache(ticker)
    if cached is not None:
        if cached.get("_status") == "no_data":
            return {c: None for c in UW_FEATURE_COLS}
        # Trust cached values; missing keys default to None.
        return {c: cached.get(c) for c in UW_FEATURE_COLS}

    # Lazy imports so feature_lab.py can run unit tests without UW creds.
    from app.vendors.unusual_whales import (
        fetch_atm_iv_term,
        fetch_expiry_breakdown,
        fetch_greek_exposure,
        fetch_iv_skew,
        fetch_max_pain,
        fetch_spot_exposures,
    )

    out: dict = {c: None for c in UW_FEATURE_COLS}

    fetchers = (
        fetch_greek_exposure,
        fetch_iv_skew,
        fetch_atm_iv_term,
        fetch_expiry_breakdown,
        fetch_max_pain,
        fetch_spot_exposures,
    )
    saw_any = False
    for fn in fetchers:
        try:
            res = fn(ticker)
        except Exception:
            res = None
        if isinstance(res, dict):
            saw_any = True
            for k, v in res.items():
                if k in out:
                    out[k] = v

    # Stamp spot into max_pain calc if UW endpoint didn't return one and we
    # have a spot from OHLCV cache. The fetcher already requires UW to
    # return spot to compute the distance — if it didn't, we leave the
    # column None rather than guessing.
    _ = spot

    payload = {**out, "_status": "ok" if saw_any else "no_data"}
    _write_cache(ticker, payload)
    return out

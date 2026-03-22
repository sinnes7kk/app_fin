"""Broad market regime detection using SPY trend, VIX, and realized volatility.

Produces a continuous ``regime_score`` in [0.0, 1.0] where 0 = maximum
risk-off and 1 = maximum risk-on.  The pipeline uses this to:

- raise the effective score threshold for counter-regime trades,
- scale position sizing via a VIX-based multiplier, and
- attach regime metadata to every signal for later analysis.

The score blends three inputs:
  1. SPY alignment (close vs EMA20, EMA20 vs EMA50, EMA20 slope)
  2. VIX level (actual ^VIX close mapped inversely to 0-1)
  3. Realized volatility (SPY ATR14/close, mapped inversely)
"""

from __future__ import annotations

import pandas as pd

from app.config import (
    VIX_ELEVATED,
    VIX_HIGH,
    VIX_SIZING_ELEVATED,
    VIX_SIZING_HIGH,
)
from app.features.price_features import clean_ohlcv, compute_features, fetch_ohlcv
from app.vendors.unusual_whales import fetch_market_tide


def _spy_alignment_score(df: pd.DataFrame) -> tuple[float, dict]:
    """Score SPY trend alignment on a 0-1 scale.

    Components (each 0-1, blended equally):
      - close vs EMA20: above = 1, below = 0
      - EMA20 vs EMA50: above = 1, below = 0
      - EMA20 slope: positive and strong = 1, negative and strong = 0
    """
    last = df.iloc[-1]
    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    slope = float(last["ema20_slope"]) if "ema20_slope" in last.index and not pd.isna(last.get("ema20_slope")) else 0.0

    close_vs_20 = 1.0 if close > ema20 else 0.0
    ema20_vs_50 = 1.0 if ema20 > ema50 else 0.0
    # Map slope from [-2, 2] → [0, 1] with clipping
    slope_score = max(0.0, min((slope + 2.0) / 4.0, 1.0))

    alignment = (close_vs_20 + ema20_vs_50 + slope_score) / 3.0

    if close > ema20 and ema20 > ema50:
        trend_label = "BULLISH"
    elif close < ema20 and ema20 < ema50:
        trend_label = "BEARISH"
    else:
        trend_label = "NEUTRAL"

    info = {
        "spy_trend": trend_label,
        "spy_close": round(close, 2),
        "spy_ema20": round(ema20, 2),
        "spy_ema50": round(ema50, 2),
        "spy_ema20_slope": round(slope, 3),
    }
    return round(alignment, 4), info


def _vix_score(vix_close: float) -> float:
    """Map VIX level to 0-1 where low VIX = 1 (risk-on) and high VIX = 0.

    Linear mapping: VIX 12 → 1.0, VIX 35 → 0.0, clamped.
    """
    return round(max(0.0, min((35.0 - vix_close) / 23.0, 1.0)), 4)


def _realized_vol_score(df: pd.DataFrame) -> float:
    """Map SPY realized volatility (ATR14/close) inversely to 0-1.

    Low realized vol = 1.0 (calm market), high = 0.0.
    Typical range: 0.005 (calm) to 0.025 (volatile).
    """
    last = df.iloc[-1]
    atr = float(last["atr14"])
    close = float(last["close"])
    if close <= 0:
        return 0.5
    vol_ratio = atr / close
    # Map 0.005 → 1.0, 0.025 → 0.0
    return round(max(0.0, min((0.025 - vol_ratio) / 0.020, 1.0)), 4)


def _vix_sizing_mult(vix_close: float) -> float:
    """Return position sizing multiplier based on VIX level."""
    if vix_close >= VIX_HIGH:
        return VIX_SIZING_HIGH
    if vix_close >= VIX_ELEVATED:
        return VIX_SIZING_ELEVATED
    return 1.0


def fetch_market_regime() -> dict:
    """Return a snapshot of the broad market environment.

    Blends four components into a continuous regime_score [0, 1]:
      1. SPY alignment (price vs EMAs + slope)
      2. VIX level (inversely mapped)
      3. Realized volatility (SPY ATR14/close, inversely mapped)
      4. Market tide (UW options-market-wide sentiment)

    Any component that fails degrades gracefully to 0.5 (neutral).
    """
    result: dict = {
        "regime_score": 0.5,
        "spy_trend": "NEUTRAL",
        "spy_close": None,
        "spy_ema20": None,
        "spy_ema50": None,
        "spy_ema20_slope": None,
        "vix_close": None,
        "vix_sizing_mult": 1.0,
        "available": False,
        "_spy_alignment": 0.5,
        "_vix_component": 0.5,
        "_rvol_component": 0.5,
        "_tide_component": 0.5,
    }

    spy_df = None
    try:
        spy_df = compute_features(clean_ohlcv(fetch_ohlcv("SPY", lookback_days=120)))
        alignment, spy_info = _spy_alignment_score(spy_df)
        result.update(spy_info)
        result["_spy_alignment"] = alignment
    except Exception:
        pass

    vix_close = None
    try:
        vix_df = fetch_ohlcv("^VIX", lookback_days=10, include_partial=False)
        vix_df = clean_ohlcv(vix_df)
        vix_close = float(vix_df.iloc[-1]["close"])
        result["vix_close"] = round(vix_close, 2)
        result["vix_sizing_mult"] = _vix_sizing_mult(vix_close)
        result["_vix_component"] = _vix_score(vix_close)
    except Exception:
        pass

    if spy_df is not None:
        result["_rvol_component"] = _realized_vol_score(spy_df)

    # Market tide: UW options sentiment (1 API call)
    try:
        tide = fetch_market_tide()
        if tide is not None:
            result["_tide_component"] = tide["tide_score"]
            result["tide_net_call_premium"] = tide["net_call_premium"]
            result["tide_net_put_premium"] = tide["net_put_premium"]
    except Exception:
        pass

    # Blend: equal weight across four components
    score = (
        result["_spy_alignment"]
        + result["_vix_component"]
        + result["_rvol_component"]
        + result["_tide_component"]
    ) / 4.0
    result["regime_score"] = round(score, 4)
    result["available"] = result["spy_close"] is not None
    return result

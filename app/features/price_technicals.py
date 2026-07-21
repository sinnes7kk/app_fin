"""Tier-1 (price/volume/volatility) and Tier-2 (cross-sectional /
cross-asset) technical features.

These are computed off the daily OHLCV bars we *already* fetch for every
scanned ticker (see ``price_features.fetch_ohlcv``) plus a market proxy
(SPY) and the ticker's sector ETF. They add data axes that are orthogonal
to the options-flow features that dominate the rest of the pipeline —
classic momentum / mean-reversion / volume / volatility factors, and the
beta-residualised relative-strength signals a cross-sectional book leans
on.

Everything here is **shadow-only**: the outputs are logged into
``feature_lab.csv`` and evaluated by the weekly sweep, and feed no live
score until they clear the promotion gate.

Design rules
------------
- **Point-in-time safe.** Every function accepts an optional ``as_of`` and
  slices the frame to bars on-or-before that date, so a historical backfill
  can never see the future. Live scans pass ``as_of = today`` (a no-op
  slice) and the latest bar is the scan day.
- **Fail-soft.** Insufficient history / bad input returns ``None`` for that
  feature rather than raising, so one thin ticker never blocks the row.
- **Pure.** No I/O here — callers pass the frames in. That keeps the module
  fully unit-testable against synthetic candles.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# GICS sector (yfinance ``info.sector`` vocabulary) -> SPDR sector ETF, used
# as the benchmark for sector-relative strength. Falls back to SPY.
SECTOR_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}
MARKET_PROXY = "SPY"

# Feature columns produced by ``compute_price_technicals`` (stable order).
PRICE_TECHNICAL_COLS = (
    # Tier 1 — momentum
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "dist_52w_high",
    "px_vs_sma50",
    "px_vs_sma200",
    # Tier 1 — mean-reversion / oscillator
    "rsi_14",
    "bollinger_z",
    # Tier 1 — volume / volatility
    "rel_volume",
    "atr_pct",
    "gap_pct",
    # Tier 2 — cross-sectional / cross-asset
    "beta_63d",
    "resid_mom_21d",
    "rel_strength_spy_63d",
    "rel_strength_sector_63d",
)


def sector_etf(sector: str | None) -> str:
    """Map a GICS sector string to its benchmark ETF (SPY fallback)."""
    if not sector:
        return MARKET_PROXY
    return SECTOR_ETF.get(str(sector).strip(), MARKET_PROXY)


def _prep(df: pd.DataFrame | None, as_of: Any = None) -> pd.DataFrame | None:
    """Normalize to a sorted, datetime-indexed OHLCV frame up to ``as_of``."""
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        try:
            d.index = pd.to_datetime(d.index, errors="coerce")
        except Exception:
            return None
    d = d[~d.index.isna()].sort_index()
    if as_of is not None:
        cutoff = pd.to_datetime(as_of, errors="coerce")
        if not pd.isna(cutoff):
            d = d[d.index <= cutoff]
    if len(d) == 0 or "close" not in d.columns:
        return None
    return d


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _ret(close: pd.Series, n: int) -> float | None:
    if len(close) < n + 1:
        return None
    c0 = close.iloc[-1]
    cn = close.iloc[-1 - n]
    if cn is None or cn == 0 or pd.isna(cn) or pd.isna(c0):
        return None
    return _finite(c0 / cn - 1.0)


def _sma_dist(close: pd.Series, n: int) -> float | None:
    if len(close) < n:
        return None
    sma = close.iloc[-n:].mean()
    if sma is None or sma == 0 or pd.isna(sma):
        return None
    return _finite(close.iloc[-1] / sma - 1.0)


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    if len(delta) < period:
        return None
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing (EWMA with alpha = 1/period).
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return _finite(100.0 - 100.0 / (1.0 + rs))


def _bollinger_z(close: pd.Series, n: int = 20) -> float | None:
    if len(close) < n:
        return None
    window = close.iloc[-n:]
    mu = window.mean()
    sd = window.std(ddof=0)
    if sd is None or sd == 0 or pd.isna(sd):
        return None
    return _finite((close.iloc[-1] - mu) / sd)


def _rel_volume(volume: pd.Series | None, n: int = 20) -> float | None:
    if volume is None or len(volume) < n + 1:
        return None
    base = volume.iloc[-1 - n:-1].mean()
    if base is None or base == 0 or pd.isna(base):
        return None
    return _finite(volume.iloc[-1] / base)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float | None:
    if not {"high", "low", "close"}.issubset(df.columns) or len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    last_close = close.iloc[-1]
    if last_close is None or last_close == 0 or pd.isna(last_close):
        return None
    return _finite(atr / last_close)


def _gap_pct(df: pd.DataFrame) -> float | None:
    if not {"open", "close"}.issubset(df.columns) or len(df) < 2:
        return None
    today_open = df["open"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    if prev_close is None or prev_close == 0 or pd.isna(prev_close) or pd.isna(today_open):
        return None
    return _finite(today_open / prev_close - 1.0)


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def _beta(tkr_ret: pd.Series, mkt_ret: pd.Series, n: int = 63) -> float | None:
    """OLS slope of ticker returns on market returns over the last ``n`` days."""
    joined = pd.concat([tkr_ret, mkt_ret], axis=1, join="inner").dropna()
    if len(joined) < max(20, n // 2):
        return None
    joined = joined.iloc[-n:]
    x = joined.iloc[:, 1].to_numpy()
    y = joined.iloc[:, 0].to_numpy()
    var = np.var(x)
    if var == 0 or math.isnan(var):
        return None
    cov = np.cov(x, y, ddof=0)[0, 1]
    return _finite(cov / var)


def compute_price_technicals(
    df: pd.DataFrame | None,
    *,
    market_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
    as_of: Any = None,
) -> dict[str, float | None]:
    """Return the full technical feature dict for one ticker.

    ``df`` is the ticker OHLCV; ``market_df`` the SPY proxy; ``sector_df``
    the sector ETF (optional). All are sliced to ``as_of`` internally.
    Missing inputs degrade gracefully — the relevant features become
    ``None`` rather than raising.
    """
    out: dict[str, float | None] = {c: None for c in PRICE_TECHNICAL_COLS}

    d = _prep(df, as_of)
    if d is None:
        return out
    close = pd.to_numeric(d["close"], errors="coerce").dropna()
    if len(close) < 2:
        return out
    volume = pd.to_numeric(d["volume"], errors="coerce") if "volume" in d.columns else None

    # Tier 1 — momentum
    out["ret_5d"] = _ret(close, 5)
    out["ret_21d"] = _ret(close, 21)
    out["ret_63d"] = _ret(close, 63)
    out["ret_126d"] = _ret(close, 126)
    if len(close) >= 20:
        hi = close.iloc[-min(len(close), 252):].max()
        if hi and not pd.isna(hi) and hi != 0:
            out["dist_52w_high"] = _finite(close.iloc[-1] / hi - 1.0)
    out["px_vs_sma50"] = _sma_dist(close, 50)
    out["px_vs_sma200"] = _sma_dist(close, 200)

    # Tier 1 — mean-reversion / oscillator
    out["rsi_14"] = _rsi(close, 14)
    out["bollinger_z"] = _bollinger_z(close, 20)

    # Tier 1 — volume / volatility
    out["rel_volume"] = _rel_volume(volume, 20)
    out["atr_pct"] = _atr_pct(d, 14)
    out["gap_pct"] = _gap_pct(d)

    # Tier 2 — cross-sectional / cross-asset
    m = _prep(market_df, as_of)
    if m is not None:
        mkt_close = pd.to_numeric(m["close"], errors="coerce").dropna()
        beta = _beta(_daily_returns(close), _daily_returns(mkt_close), 63)
        out["beta_63d"] = beta
        t21 = _ret(close, 21)
        m21 = _ret(mkt_close, 21)
        if beta is not None and t21 is not None and m21 is not None:
            out["resid_mom_21d"] = _finite(t21 - beta * m21)
        t63 = _ret(close, 63)
        m63 = _ret(mkt_close, 63)
        if t63 is not None and m63 is not None:
            out["rel_strength_spy_63d"] = _finite(t63 - m63)

    s = _prep(sector_df, as_of)
    if s is not None:
        sec_close = pd.to_numeric(s["close"], errors="coerce").dropna()
        t63 = _ret(close, 63)
        s63 = _ret(sec_close, 63)
        if t63 is not None and s63 is not None:
            out["rel_strength_sector_63d"] = _finite(t63 - s63)

    return out

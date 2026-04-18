"""Decision context — enrich signals with the last-mile info a live trader needs.

Adds five decision-critical fields to each signal row:

- ``liquidity``: ADV$ tier + raw value so a trader can size without blowing size
- ``session``: today's price vs prior close (and vs open as a VWAP proxy)
- ``rs``: relative strength vs SPY over 5-day and 20-day windows
- ``trade_expression``: recommended vehicle (stock / long call / debit spread / ...)
- ``r_at_market``: live R multiple from entry, plus distance to stop in R

All enrichments degrade gracefully: if OHLCV or SPY data is unavailable the
corresponding field is omitted rather than raising.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.features.price_features import fetch_addv, fetch_ohlcv


def _liquidity_tier(adv_dollar: float | None) -> str:
    """Bucket ADV$ into trader-readable tiers."""
    if adv_dollar is None or adv_dollar <= 0:
        return "UNKNOWN"
    if adv_dollar >= 500_000_000:
        return "DEEP"
    if adv_dollar >= 100_000_000:
        return "HEALTHY"
    if adv_dollar >= 20_000_000:
        return "THIN"
    return "ILLIQUID"


def compute_liquidity(ticker: str, mcap: float | None = None) -> dict[str, Any]:
    """Return ADV$ + tier label + mcap for a ticker."""
    adv = fetch_addv(ticker)
    return {
        "adv_dollar": adv,
        "liquidity_tier": _liquidity_tier(adv),
        "marketcap": mcap,
    }


def compute_session_context(df: pd.DataFrame) -> dict[str, Any]:
    """Intra-day context derived from the latest OHLCV bar.

    ``px_vs_prior_close_pct``: today's close vs prior day close.
    ``px_vs_open_pct``: today's close vs today's open (VWAP proxy when
    intraday bar is partial).
    ``session_tone``: STRENGTH / WEAKNESS / FLAT based on these two together.
    """
    if df is None or len(df) < 2:
        return {"session_available": False}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    prev_close = float(prev.get("close") or 0.0)
    today_close = float(last.get("close") or 0.0)
    today_open = float(last.get("open") or 0.0)

    if prev_close <= 0 or today_close <= 0:
        return {"session_available": False}

    vs_prior = (today_close - prev_close) / prev_close
    vs_open = ((today_close - today_open) / today_open) if today_open > 0 else 0.0

    if vs_prior > 0.005 and vs_open > 0:
        tone = "STRENGTH"
    elif vs_prior < -0.005 and vs_open < 0:
        tone = "WEAKNESS"
    else:
        tone = "FLAT"

    return {
        "session_available": True,
        "px_vs_prior_close_pct": round(vs_prior * 100, 2),
        "px_vs_open_pct": round(vs_open * 100, 2),
        "session_tone": tone,
    }


def compute_rs(df: pd.DataFrame, spy_df: pd.DataFrame, windows: tuple[int, ...] = (5, 20)) -> dict[str, Any]:
    """Relative strength vs SPY over multiple windows.

    Returns a dict like ``{"rs_5d_pct": 3.1, "rs_20d_pct": -1.4}``; any window
    whose lookback isn't available is skipped.
    """
    out: dict[str, Any] = {}
    if df is None or spy_df is None or df.empty or spy_df.empty:
        return out

    t_close = pd.to_numeric(df["close"], errors="coerce").dropna()
    s_close = pd.to_numeric(spy_df["close"], errors="coerce").dropna()
    if t_close.empty or s_close.empty:
        return out

    for w in windows:
        if len(t_close) <= w or len(s_close) <= w:
            continue
        t_ret = (t_close.iloc[-1] / t_close.iloc[-w - 1]) - 1.0
        s_ret = (s_close.iloc[-1] / s_close.iloc[-w - 1]) - 1.0
        out[f"rs_{w}d_pct"] = round((t_ret - s_ret) * 100, 2)
    return out


def suggest_expression(
    direction: str,
    iv_rank: float | None,
    days_to_earnings: int | None,
    adv_dollar: float | None = None,
) -> dict[str, str]:
    """Pick a trade expression based on direction, IV context, and catalyst.

    Returns ``{"expression": <vehicle>, "rationale": <short why>}``.
    The vehicles are:
      - ``STOCK``: simplest, works for low IV or no strong IV signal
      - ``LONG_CALL`` / ``LONG_PUT``: cheap vol environment, directional
      - ``DEBIT_SPREAD``: elevated IV or near catalyst — kills vega
      - ``RISK_REVERSAL``: high IV + strong directional conviction (stock alt)
    """
    ivr = iv_rank if iv_rank is not None else -1.0
    dte = days_to_earnings if days_to_earnings is not None else 999
    d = (direction or "").upper()
    illiquid = adv_dollar is not None and adv_dollar < 20_000_000

    if illiquid:
        return {
            "expression": "STOCK",
            "rationale": "Option chain likely too illiquid",
        }

    near_catalyst = dte <= 5

    if d == "LONG":
        if near_catalyst and ivr >= 60:
            return {"expression": "DEBIT_SPREAD", "rationale": f"IVR {ivr:.0f} + earnings {dte}d — kill vega"}
        if ivr >= 70:
            return {"expression": "DEBIT_SPREAD", "rationale": f"IVR {ivr:.0f} — cap vega on overpriced vol"}
        if 0 <= ivr < 30:
            return {"expression": "LONG_CALL", "rationale": f"IVR {ivr:.0f} — cheap premium"}
        if ivr >= 30:
            return {"expression": "STOCK", "rationale": f"IVR {ivr:.0f} — mid, stock is cleanest"}
        return {"expression": "STOCK", "rationale": "No IV rank — default to stock"}

    if d == "SHORT":
        if near_catalyst and ivr >= 60:
            return {"expression": "DEBIT_SPREAD", "rationale": f"IVR {ivr:.0f} + earnings {dte}d — put debit spread"}
        if ivr >= 70:
            return {"expression": "DEBIT_SPREAD", "rationale": f"IVR {ivr:.0f} — put debit spread caps vega"}
        if 0 <= ivr < 30:
            return {"expression": "LONG_PUT", "rationale": f"IVR {ivr:.0f} — cheap put premium"}
        if ivr >= 30:
            return {"expression": "STOCK", "rationale": f"IVR {ivr:.0f} — short shares"}
        return {"expression": "STOCK", "rationale": "No IV rank — short shares"}

    return {"expression": "STOCK", "rationale": "Direction unknown"}


def compute_r_at_market(
    direction: str,
    entry_price: float | None,
    stop_price: float | None,
    spot: float | None,
) -> dict[str, Any]:
    """Current unrealized R and remaining R to stop, from the trader's POV.

    ``r_from_entry``: positive = in your favor since entry.
    ``r_to_stop``: remaining distance to stop in R (positive = still has room).
    """
    if not (entry_price and stop_price and spot):
        return {"r_available": False}
    if entry_price <= 0 or spot <= 0:
        return {"r_available": False}

    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return {"r_available": False}

    d = (direction or "").upper()
    if d == "LONG":
        r_from_entry = (spot - entry_price) / risk_per_share
        r_to_stop = (spot - stop_price) / risk_per_share
    else:
        r_from_entry = (entry_price - spot) / risk_per_share
        r_to_stop = (stop_price - spot) / risk_per_share

    return {
        "r_available": True,
        "r_from_entry": round(r_from_entry, 2),
        "r_to_stop": round(r_to_stop, 2),
    }


_SPY_CACHE: pd.DataFrame | None = None


def _get_spy_df(lookback_days: int = 60) -> pd.DataFrame | None:
    """Fetch + cache SPY OHLCV for relative-strength calcs."""
    global _SPY_CACHE
    if _SPY_CACHE is None or len(_SPY_CACHE) < 25:
        try:
            _SPY_CACHE = fetch_ohlcv("SPY", lookback_days=lookback_days)
        except Exception:
            _SPY_CACHE = None
    return _SPY_CACHE


def clear_decision_cache() -> None:
    """Reset SPY cache — call at the start of each pipeline run."""
    global _SPY_CACHE
    _SPY_CACHE = None


def enrich_signal(
    row: dict[str, Any],
    *,
    df: pd.DataFrame | None = None,
    spot: float | None = None,
) -> dict[str, Any]:
    """Mutate (and return) ``row`` with decision-context fields.

    ``df`` is the ticker's OHLCV DataFrame (with 'open'/'close' columns).
    ``spot`` falls back to ``row['entry_price']`` if not supplied.
    """
    ticker = row.get("ticker")
    if not ticker:
        return row

    mcap = row.get("marketcap") or (row.get("flow_snapshot") or {}).get("marketcap")
    row["liquidity"] = compute_liquidity(ticker, mcap=mcap)

    if df is not None and not df.empty:
        row["session"] = compute_session_context(df)
        spy_df = _get_spy_df()
        if spy_df is not None:
            rs = compute_rs(df, spy_df)
            if rs:
                row["rs"] = rs

    iv_rank = row.get("iv_rank")
    dte_earn = row.get("days_to_earnings")
    if dte_earn is None:
        earn = row.get("earnings") or {}
        dte_earn = earn.get("days_until") or earn.get("days_to_earnings")
    row["trade_expression"] = suggest_expression(
        row.get("direction", ""),
        iv_rank,
        dte_earn,
        adv_dollar=row["liquidity"].get("adv_dollar"),
    )

    effective_spot = spot
    if effective_spot is None and df is not None and not df.empty:
        try:
            effective_spot = float(df.iloc[-1]["close"])
        except Exception:
            effective_spot = None
    if effective_spot is None:
        effective_spot = row.get("entry_price")

    row["r_at_market"] = compute_r_at_market(
        row.get("direction", ""),
        row.get("entry_price"),
        row.get("stop_price"),
        effective_spot,
    )
    if effective_spot is not None:
        row["spot_price"] = round(float(effective_spot), 4)

    return row


def enrich_signals(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich every accepted signal row in-place."""
    for r in results:
        try:
            df = None
            try:
                df = fetch_ohlcv(r["ticker"])
            except Exception:
                df = None
            enrich_signal(r, df=df)
        except Exception:
            continue
    return results

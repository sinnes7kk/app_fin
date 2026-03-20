"""Unusual Whales vendor integration — options flow data and curated alerts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from app.config import UNUSUAL_WHALES_API_KEY

BASE_URL = "https://api.unusualwhales.com/api"
FLOW_ALERTS_ENDPOINT = f"{BASE_URL}/option-trades/flow-alerts"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {UNUSUAL_WHALES_API_KEY}",
        "Accept": "application/json",
    }


def fetch_flow_raw(
    ticker: str | None = None,
    min_premium: int | None = None,
    limit: int = 200,
    **extra_params,
) -> dict:
    """
    Fetch raw flow alerts from the Unusual Whales API.

    Parameters are passed straight through as query-string filters.
    """
    params: dict = {"limit": limit}
    if ticker:
        params["ticker_symbol"] = ticker
    if min_premium is not None:
        params["min_premium"] = min_premium
    params.update(extra_params)

    resp = requests.get(
        FLOW_ALERTS_ENDPOINT,
        headers=_headers(),
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


RENAME_MAP = {
    "ticker": "ticker",
    "created_at": "event_ts",
    "type": "option_type",
    "strike": "strike",
    "expiry": "expiration_date",
    "total_premium": "premium",
    "total_size": "contracts",
    "volume": "volume",
    "open_interest": "open_interest",
    "underlying_price": "underlying_price",
    "alert_rule": "alert_rule",
    "has_sweep": "is_sweep",
    "has_floor": "is_floor",
    "total_ask_side_prem": "ask_side_premium",
    "total_bid_side_prem": "bid_side_premium",
}


def normalize_flow_response(payload: dict) -> pd.DataFrame:
    """
    Turn raw API JSON into a clean DataFrame with standard column names.

    Adds derived columns:
      - dte
      - execution_side
      - direction
    """
    rows = payload.get("data", payload)
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    present = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=present)
    df = df[list(present.values())].copy()

    # Normalize text columns
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    if "option_type" in df.columns:
        df["option_type"] = df["option_type"].astype(str).str.upper().str.strip()

    if "alert_rule" in df.columns:
        df["alert_rule"] = df["alert_rule"].astype(str).str.upper().str.strip()

    # Datetime parsing
    if "event_ts" in df.columns:
        df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")

    if "expiration_date" in df.columns:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")

    # Numeric parsing
    numeric_cols = [
        "strike",
        "premium",
        "contracts",
        "volume",
        "open_interest",
        "underlying_price",
        "ask_side_premium",
        "bid_side_premium",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Nullable integer fields
    for col in ["contracts", "volume", "open_interest"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    # Derived DTE from event date, not today's date
    if "event_ts" in df.columns and "expiration_date" in df.columns:
        event_date = df["event_ts"].dt.tz_convert(None).dt.normalize()
        expiry_date = df["expiration_date"].dt.normalize()
        df["dte"] = (expiry_date - event_date).dt.days

    # Execution side
    df["execution_side"] = df.apply(_infer_side, axis=1)

    # Direction bias for V1
    df["direction"] = df.apply(_infer_direction, axis=1)

    return df


def _infer_side(row: pd.Series) -> str:
    ask = row.get("ask_side_premium", 0) or 0
    bid = row.get("bid_side_premium", 0) or 0

    if ask > bid:
        return "ASK"
    if bid > ask:
        return "BID"
    return "MIXED"


def _infer_direction(row: pd.Series) -> str | None:
    option_type = row.get("option_type")
    execution_side = row.get("execution_side")

    if execution_side != "ASK":
        return None

    if option_type == "CALL":
        return "LONG"
    if option_type == "PUT":
        return "SHORT"

    return None


def fetch_uw_alerts(limit: int = 200, hours_back: int = 24) -> list[str]:
    """Discover tickers with unusual options activity.

    Queries the flow alerts endpoint for sweeps where volume exceeds open
    interest — a strong signal of aggressive, unusual positioning.  Returns
    unique ticker symbols sorted alphabetically.
    """
    newer_than = (
        datetime.now(tz=timezone.utc) - timedelta(hours=hours_back)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    params: dict = {
        "limit": limit,
        "is_sweep": "true",
        "vol_greater_oi": "true",
        "newer_than": newer_than,
    }

    try:
        resp = requests.get(
            FLOW_ALERTS_ENDPOINT,
            headers=_headers(),
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return []

    tickers: set[str] = set()
    for row in data:
        symbol = (row.get("ticker") or "").upper().strip()
        if symbol:
            tickers.add(symbol)

    return sorted(tickers)


def fetch_flow_for_tickers(
    tickers: list[str],
    limit_per_ticker: int = 50,
) -> pd.DataFrame:
    """Fetch and normalize flow data for specific tickers.

    Makes one API call per ticker.  Returns a single concatenated and
    normalized DataFrame, or an empty DataFrame if nothing is found.
    """
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            payload = fetch_flow_raw(ticker=ticker, limit=limit_per_ticker)
            df = normalize_flow_response(payload)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
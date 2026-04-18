"""Unusual Whales vendor integration — options flow data and curated alerts."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import pandas as pd
import requests

from app.config import BID_SIDE_FLOW_CONFIDENCE, UNUSUAL_WHALES_API_KEY

BASE_URL = "https://api.unusualwhales.com/api"
FLOW_ALERTS_ENDPOINT = f"{BASE_URL}/option-trades/flow-alerts"

# ---------------------------------------------------------------------------
# Centralized API request tracking
# ---------------------------------------------------------------------------

_run_call_count: int = 0
_endpoint_counts: dict[str, int] = defaultdict(int)
_rate_limit_snapshot: dict[str, str | int] = {}


def reset_api_stats() -> None:
    """Reset per-run API counters.  Call at the start of each pipeline run."""
    global _run_call_count, _minute_throttle_until, _throttle_warned
    _run_call_count = 0
    _endpoint_counts.clear()
    _rate_limit_snapshot.clear()
    _minute_throttle_until = 0.0
    _throttle_warned = False


def get_api_stats() -> dict:
    """Return current run's API usage statistics."""
    return {
        "calls_this_run": _run_call_count,
        "by_endpoint": dict(_endpoint_counts),
        "rate_limit": dict(_rate_limit_snapshot),
    }


def print_api_summary() -> None:
    """Print a human-readable summary of API usage for this run."""
    stats = get_api_stats()
    print(f"\n  [UW API] {stats['calls_this_run']} calls this run")
    if stats["by_endpoint"]:
        for ep, count in sorted(stats["by_endpoint"].items(), key=lambda x: -x[1]):
            print(f"    {ep}: {count}")
    rl = stats["rate_limit"]
    if rl:
        daily = rl.get("daily_used", "?")
        limit = rl.get("daily_limit", "?")
        minute_rem = rl.get("minute_remaining", "?")
        print(f"  [UW API] daily {daily}/{limit}  minute remaining: {minute_rem}")


def _endpoint_key(url: str) -> str:
    """Normalize a full URL to a short endpoint label for tracking."""
    path = urlparse(url).path
    # Collapse ticker-specific paths: /api/stock/AAPL/foo -> /stock/{ticker}/foo
    path = re.sub(r"/stock/[A-Z0-9.]+/", "/stock/{ticker}/", path)
    path = re.sub(r"/darkpool/[A-Z][A-Z0-9.]+", "/darkpool/{ticker}", path)
    # Collapse OCC contract ids (e.g. AAPL270115C00200000) to {contract}
    path = re.sub(r"/option-contract/[A-Z0-9.]+/", "/option-contract/{contract}/", path)
    # Strip the /api prefix for brevity
    path = re.sub(r"^/api", "", path)
    return path or url


_minute_throttle_until: float = 0.0  # monotonic time we must wait until
_throttle_warned: bool = False


def _uw_request(url: str, *, params: dict | None = None, timeout: int = 30) -> requests.Response:
    """Centralized HTTP GET for all Unusual Whales API calls.

    Tracks per-endpoint counts, parses rate-limit headers, and auto-throttles
    when the per-minute quota is exhausted (sleeps until the window resets).
    """
    global _run_call_count, _minute_throttle_until, _throttle_warned

    # Respect any active throttle from a previous response
    wait = _minute_throttle_until - time.monotonic()
    if wait > 0:
        if not _throttle_warned:
            print(f"  [UW API] minute quota exhausted — pausing {wait:.0f}s")
            _throttle_warned = True
        time.sleep(wait)
        _throttle_warned = False

    headers = {
        "Authorization": f"Bearer {UNUSUAL_WHALES_API_KEY}",
        "Accept": "application/json",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)

    _run_call_count += 1
    _endpoint_counts[_endpoint_key(url)] += 1

    # Parse rate-limit headers when present
    daily_used = resp.headers.get("x-uw-daily-req-count")
    daily_limit = resp.headers.get("x-uw-token-req-limit")
    minute_rem = resp.headers.get("x-uw-req-per-minute-remaining")
    if daily_used is not None:
        _rate_limit_snapshot["daily_used"] = int(daily_used)
    if daily_limit is not None:
        _rate_limit_snapshot["daily_limit"] = int(daily_limit)
    if minute_rem is not None:
        remaining = int(minute_rem)
        _rate_limit_snapshot["minute_remaining"] = remaining
        if remaining <= 0:
            # Back off for the rest of the minute window (+ 2s safety margin)
            _minute_throttle_until = time.monotonic() + 62
        elif remaining < 5:
            print(f"  [UW API] {remaining} requests remaining this minute")

    return resp


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

    resp = _uw_request(FLOW_ALERTS_ENDPOINT, params=params)
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
    "marketcap": "marketcap",
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
        "marketcap",
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

    df["direction"] = df.apply(_infer_direction, axis=1)
    df["direction_confidence"] = df.apply(_infer_direction_confidence, axis=1)

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

    if option_type == "CALL":
        if execution_side == "ASK":
            return "LONG"
        if execution_side == "BID":
            return "SHORT"
    elif option_type == "PUT":
        if execution_side == "ASK":
            return "SHORT"
        if execution_side == "BID":
            return "LONG"
    return None


def _infer_direction_confidence(row: pd.Series) -> float:
    side = row.get("execution_side")
    if side == "ASK":
        return 1.0
    if side == "BID":
        return BID_SIDE_FLOW_CONFIDENCE
    return 0.0


def fetch_net_prem_ticks(ticker: str) -> dict | None:
    """Fetch intraday net premium flow for a ticker.

    Returns derived metrics:
      - intraday_premium_direction: 0-1 (1 = call premium strongly dominant)
      - delta_momentum: positive = smart money still accumulating
      - net_delta: latest accumulated directional exposure
    """
    url = f"{BASE_URL}/stock/{ticker}/net-prem-ticks"
    try:
        resp = _uw_request(url)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return None

    if not data:
        return None

    def _f(row: dict, key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    # Use last few ticks for momentum, latest tick for current state
    latest = data[-1]
    net_call = _f(latest, "net_call_premium")
    net_put = _f(latest, "net_put_premium")
    net_delta = _f(latest, "net_delta")

    total = abs(net_call) + abs(net_put)
    if total > 0:
        premium_direction = (net_call + total) / (2 * total)
    else:
        premium_direction = 0.5

    # Delta momentum: compare latest vs a point ~25% back in the series
    delta_momentum = 0.0
    if len(data) >= 4:
        earlier_idx = max(0, len(data) // 4)
        earlier_delta = _f(data[earlier_idx], "net_delta")
        if abs(earlier_delta) > 0:
            delta_momentum = (net_delta - earlier_delta) / max(abs(earlier_delta), abs(net_delta), 1.0)
        else:
            delta_momentum = 1.0 if net_delta > 0 else (-1.0 if net_delta < 0 else 0.0)

    return {
        "intraday_premium_direction": round(max(0.0, min(premium_direction, 1.0)), 4),
        "delta_momentum": round(max(-1.0, min(delta_momentum, 1.0)), 4),
        "net_delta": round(net_delta, 2),
        "net_call_premium": round(net_call, 2),
        "net_put_premium": round(net_put, 2),
    }


def fetch_dark_pool(ticker: str) -> dict | None:
    """Fetch recent dark pool prints for a ticker.

    Returns derived metrics:
      - dark_pool_bias: 0-1 (1 = aggressive buying dominant)
      - dark_pool_volume: total volume in recent prints
      - large_print_count: number of prints > $1M
    """
    url = f"{BASE_URL}/darkpool/{ticker}"
    try:
        resp = _uw_request(url)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return None

    if not data:
        return None

    buy_vol = 0.0
    sell_vol = 0.0
    total_vol = 0.0
    large_prints = 0

    for row in data:
        try:
            price = float(row.get("price", 0) or 0)
            size = float(row.get("size", 0) or 0)
            nbbo_ask = float(row.get("nbbo_ask", 0) or 0)
            nbbo_bid = float(row.get("nbbo_bid", 0) or 0)
        except (TypeError, ValueError):
            continue

        notional = price * size
        total_vol += size

        if notional >= 1_000_000:
            large_prints += 1

        if nbbo_ask > 0 and price >= nbbo_ask:
            buy_vol += size
        elif nbbo_bid > 0 and price <= nbbo_bid:
            sell_vol += size

    if total_vol > 0:
        bias = buy_vol / total_vol
    else:
        bias = 0.5

    return {
        "dark_pool_bias": round(bias, 4),
        "dark_pool_volume": round(total_vol),
        "large_print_count": large_prints,
    }


def fetch_dark_pool_recent(
    *,
    min_premium: int = 100_000,
    limit: int = 200,
) -> list[dict]:
    """Fetch market-wide recent dark pool prints.

    Returns the raw list of prints (up to *limit*) with at least
    *min_premium* notional.  Each dict includes ticker, price, size,
    premium, nbbo_bid, nbbo_ask, executed_at, and volume.
    """
    url = f"{BASE_URL}/darkpool/recent"
    params = {"min_premium": str(min_premium), "limit": str(limit)}
    try:
        resp = _uw_request(url, params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


def fetch_flow_recent(ticker: str, limit: int = 50) -> dict | None:
    """Fetch the most recent options flow for a ticker.

    Returns a summary indicating whether new qualifying flow has appeared.
    """
    url = f"{BASE_URL}/stock/{ticker}/flow-recent"
    try:
        resp = _uw_request(url, params={"limit": limit})
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return None

    if not data:
        return None

    total_premium = 0.0
    bullish_premium = 0.0
    bearish_premium = 0.0
    count = len(data)

    for row in data:
        try:
            prem = float(row.get("total_premium", 0) or 0)
        except (TypeError, ValueError):
            prem = 0.0
        total_premium += prem

        ask_p = float(row.get("total_ask_side_prem", 0) or 0)
        bid_p = float(row.get("total_bid_side_prem", 0) or 0)
        opt_type = (row.get("type") or "").upper()

        if opt_type == "CALL":
            if ask_p > bid_p:
                bullish_premium += prem
            else:
                bearish_premium += prem
        elif opt_type == "PUT":
            if ask_p > bid_p:
                bearish_premium += prem
            else:
                bullish_premium += prem

    if total_premium > 0:
        bullish_ratio = bullish_premium / total_premium
    else:
        bullish_ratio = 0.5

    return {
        "recent_flow_count": count,
        "recent_total_premium": round(total_premium, 2),
        "recent_bullish_ratio": round(bullish_ratio, 4),
    }


def fetch_market_tide() -> dict | None:
    """Fetch market-wide options sentiment from the Market Tide endpoint.

    Returns a dict with net_call_premium, net_put_premium, and a derived
    tide_score (0 = bearish, 1 = bullish).  Returns None on failure.
    """
    url = f"{BASE_URL}/market/market-tide"
    try:
        resp = _uw_request(url)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return None

    if not data:
        return None

    # Use the most recent data point
    latest = data[-1] if isinstance(data, list) else data

    def _f(key: str) -> float:
        try:
            return float(latest.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    net_call = _f("net_call_premium")
    net_put = _f("net_put_premium")
    total = abs(net_call) + abs(net_put)

    if total > 0:
        tide_score = (net_call + total) / (2 * total)
    else:
        tide_score = 0.5

    return {
        "net_call_premium": round(net_call, 2),
        "net_put_premium": round(net_put, 2),
        "tide_score": round(max(0.0, min(tide_score, 1.0)), 4),
    }


def fetch_stock_screener(
    min_premium: int = 100_000,
    min_price: float = 5.0,
    min_marketcap: int = 1_000_000_000,
    min_pct_30d_total: float = 1.5,
) -> list[dict]:
    """Fetch the UW stock screener for tickers with unusual aggregate activity.

    Returns a list of dicts with per-ticker metrics (premium, volume ratios,
    iv_rank, sector, etc.).  A single API call covers the entire optionable
    universe, making this the most efficient discovery endpoint.
    """
    url = f"{BASE_URL}/screener/stocks"
    params: dict = {
        "min_premium": min_premium,
        "issue_types[]": "Common Stock",
        "min_underlying_price": min_price,
        "min_marketcap": min_marketcap,
        "min_perc_30_day_total": min_pct_30d_total,
    }
    try:
        resp = _uw_request(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return []
    return data


def fetch_uw_alerts(limit: int = 500, hours_back: int = 48) -> list[str]:
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
        resp = _uw_request(FLOW_ALERTS_ENDPOINT, params=params)
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


def fetch_recent_alert_flow(limit: int = 150, hours_back: int = 24) -> pd.DataFrame:
    """Return normalized flow rows for recent UW sweep + vol>Oi alerts (for UI / research)."""
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
        resp = _uw_request(FLOW_ALERTS_ENDPOINT, params=params)
        resp.raise_for_status()
        return normalize_flow_response(resp.json())
    except Exception:
        return pd.DataFrame()


def fetch_earnings(ticker: str) -> dict | None:
    """Fetch earnings history for a ticker.

    Returns next earnings date (if available), days until earnings,
    and last reported EPS surprise.
    """
    url = f"{BASE_URL}/stock/{ticker}/earnings"
    try:
        resp = _uw_request(url)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return None

    if not data:
        return None

    from datetime import date as _date

    today = _date.today()
    next_date = None
    days_until = None
    last_reported = None
    last_eps_surprise = None

    for row in data:
        report_date_str = row.get("date") or row.get("report_date") or ""
        if not report_date_str:
            continue
        try:
            rd = _date.fromisoformat(str(report_date_str)[:10])
        except (ValueError, TypeError):
            continue

        if rd >= today:
            if next_date is None or rd < next_date:
                next_date = rd
        else:
            if last_reported is None or rd > last_reported:
                last_reported = rd
                try:
                    actual = float(row.get("actual_eps") or row.get("eps") or 0)
                    estimate = float(row.get("estimated_eps") or row.get("eps_estimate") or 0)
                    last_eps_surprise = round(actual - estimate, 4) if estimate else None
                except (TypeError, ValueError):
                    last_eps_surprise = None

    if next_date:
        days_until = (next_date - today).days

    return {
        "next_earnings_date": str(next_date) if next_date else None,
        "days_until_earnings": days_until,
        "last_reported_date": str(last_reported) if last_reported else None,
        "last_eps_surprise": last_eps_surprise,
    }


def fetch_hottest_chains(
    min_premium: int = 250_000,
    limit: int = 100,
    max_dte: int = 90,
) -> list[dict]:
    """Fetch top unusual option contracts from the contract screener.

    Returns a list of dicts with per-contract metrics: ticker, strike, expiry,
    type (Call/Put), premium, volume, OI, vol/OI ratio, ask-side percentage.
    Single API call covers the entire market.
    """
    url = f"{BASE_URL}/screener/option-contracts"
    params: dict = {
        "limit": limit,
        "min_premium": min_premium,
        "max_dte": max_dte,
        "issue_types[]": "Common Stock",
        "min_volume_oi_ratio": "1.0",
        "max_multileg_volume_ratio": "0.3",
    }
    try:
        resp = _uw_request(url, params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


def fetch_insider_transactions(limit: int = 200) -> list[dict]:
    """Fetch recent market-wide insider transactions.

    Returns a list of dicts with: ticker, insider name, title, transaction type
    (Buy/Sell/Grant), shares, value, filing date.
    Single API call covers the entire market.
    """
    url = f"{BASE_URL}/insider/transactions"
    params: dict = {"limit": limit}
    try:
        resp = _uw_request(url, params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


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


# ---------------------------------------------------------------------------
# Option-contract greeks (delta-weighted flow)
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402
import json as _json  # noqa: E402
import tempfile as _tempfile  # noqa: E402
import os as _os  # noqa: E402
from datetime import date as _date  # noqa: E402

_GREEK_CACHE_DIR = _Path(__file__).resolve().parents[2] / "data" / "greek_cache"
_greek_memo: dict[str, dict | None] = {}  # in-process memo keyed by contract_id
_greek_day_loaded: str | None = None      # which day is currently in _greek_memo


def build_occ_id(
    ticker: str,
    expiration_date,
    option_type: str,
    strike,
) -> str | None:
    """Construct an OCC-style option contract id that UW accepts on the
    /option-contract/{id}/... endpoints.

    Format: ``{TICKER}{YYMMDD}{C|P}{strike*1000:08d}`` — e.g. ``AAPL270115C00200000``.
    Returns ``None`` if any component is missing or unparseable so callers can
    skip the row cleanly.
    """
    if not ticker:
        return None
    t = str(ticker).upper().strip()
    if not t:
        return None

    # Normalize option type → single char
    ot = str(option_type or "").upper().strip()
    if ot.startswith("C"):
        type_char = "C"
    elif ot.startswith("P"):
        type_char = "P"
    else:
        return None

    # Parse expiration into YYMMDD
    exp = expiration_date
    exp_str: str | None = None
    try:
        if isinstance(exp, str):
            exp_str = pd.to_datetime(exp).strftime("%y%m%d")
        elif hasattr(exp, "strftime"):
            exp_str = exp.strftime("%y%m%d")
    except Exception:
        return None
    if not exp_str or len(exp_str) != 6:
        return None

    try:
        strike_int = int(round(float(strike) * 1000))
    except (TypeError, ValueError):
        return None
    if strike_int <= 0:
        return None

    return f"{t}{exp_str}{type_char}{strike_int:08d}"


def _greek_cache_path(cache_date: _date) -> _Path:
    return _GREEK_CACHE_DIR / f"{cache_date.isoformat()}.json"


def _load_greek_cache(cache_date: _date) -> dict[str, dict | None]:
    global _greek_memo, _greek_day_loaded
    day_key = cache_date.isoformat()
    if _greek_day_loaded == day_key:
        return _greek_memo
    path = _greek_cache_path(cache_date)
    _greek_memo = {}
    _greek_day_loaded = day_key
    if path.exists():
        try:
            with path.open("r") as f:
                data = _json.load(f)
            if isinstance(data, dict):
                _greek_memo = {str(k): v for k, v in data.items()}
        except Exception:
            _greek_memo = {}
    return _greek_memo


def _persist_greek_cache(cache_date: _date) -> None:
    """Atomically write the in-memory cache to disk."""
    try:
        _GREEK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _greek_cache_path(cache_date)
        # Write to a temp file in the same directory, then rename → atomic on POSIX.
        fd, tmp_path = _tempfile.mkstemp(
            prefix=f".{cache_date.isoformat()}.",
            suffix=".tmp.json",
            dir=str(_GREEK_CACHE_DIR),
        )
        try:
            with _os.fdopen(fd, "w") as f:
                _json.dump(_greek_memo, f)
            _os.replace(tmp_path, path)
        except Exception:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        pass


def fetch_contract_greeks(
    contract_id: str,
    *,
    as_of_ts=None,
    timeout: int = 5,
) -> dict | None:
    """Fetch delta/IV for a single option contract from UW's greek-exposure
    endpoint. Results are cached per-day in ``data/greek_cache/{YYYY-MM-DD}.json``
    keyed by ``contract_id``; the cache also stores negative results (``None``)
    so we don't hammer the endpoint for contracts UW doesn't return.

    Returns ``{"delta": float, "iv": float|None, "ts": iso8601}`` or ``None``.
    """
    if not contract_id:
        return None

    # Cache key day — either the event day or today.
    cache_date: _date
    if as_of_ts is None:
        cache_date = _date.today()
    else:
        try:
            cache_date = pd.to_datetime(as_of_ts).date()
        except Exception:
            cache_date = _date.today()

    cache = _load_greek_cache(cache_date)
    if contract_id in cache:
        return cache[contract_id]

    url = f"{BASE_URL}/option-contract/{contract_id}/greek-exposure"
    result: dict | None = None
    try:
        resp = _uw_request(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        payload = None

    if payload:
        data = payload.get("data") if isinstance(payload, dict) else None
        # UW typically returns a list of daily snapshots; some endpoints return
        # a single dict. Handle both shapes.
        rows: list[dict] = []
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            rows = [data]

        best: dict | None = None
        if rows:
            if as_of_ts is not None:
                try:
                    target = pd.to_datetime(as_of_ts)
                    def _row_ts(r: dict):
                        for k in ("timestamp", "time", "date", "as_of"):
                            v = r.get(k)
                            if v:
                                try:
                                    return pd.to_datetime(v)
                                except Exception:
                                    continue
                        return None
                    # Prefer the row with timestamp closest to target.
                    ranked = [
                        (abs((t - target).total_seconds()), r)
                        for r in rows
                        for t in [_row_ts(r)]
                        if t is not None
                    ]
                    if ranked:
                        ranked.sort(key=lambda x: x[0])
                        best = ranked[0][1]
                except Exception:
                    best = None
            if best is None:
                best = rows[-1]

        if best is not None:
            def _maybe_float(val) -> float | None:
                try:
                    if val is None or val == "":
                        return None
                    return float(val)
                except (TypeError, ValueError):
                    return None

            delta = _maybe_float(best.get("delta"))
            if delta is None:
                # Some shapes expose a nested "greeks" block.
                greeks = best.get("greeks") or {}
                if isinstance(greeks, dict):
                    delta = _maybe_float(greeks.get("delta"))
            iv = _maybe_float(best.get("iv") or best.get("implied_volatility"))
            ts_val = best.get("timestamp") or best.get("time") or best.get("date")
            ts_iso: str | None = None
            if ts_val:
                try:
                    ts_iso = pd.to_datetime(ts_val).isoformat()
                except Exception:
                    ts_iso = str(ts_val)

            if delta is not None:
                result = {"delta": delta, "iv": iv, "ts": ts_iso}

    cache[contract_id] = result
    _persist_greek_cache(cache_date)
    return result
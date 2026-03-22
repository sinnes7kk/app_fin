"""Price-derived features: OHLCV fetch, cleaning, and technical indicators."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from app.config import ATR_PERIOD, EMA_LONG, EMA_SHORT, OHLCV_LOOKBACK_DAYS

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

_ET = ZoneInfo("America/New_York")
_SESSION_SECONDS = 6.5 * 3600  # 9:30 – 16:00 ET

# Per-run OHLCV and ADDV caches — cleared at the start of each pipeline run.
_OHLCV_CACHE: dict[str, pd.DataFrame] = {}
_ADDV_CACHE: dict[str, float | None] = {}
STALE_DATA_MAX_DAYS = 3


def clear_price_cache() -> None:
    """Reset per-run caches."""
    _OHLCV_CACHE.clear()
    _ADDV_CACHE.clear()

RAW_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
RENAME_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _fetch_intraday_bar(ticker: str) -> pd.DataFrame | None:
    """Fetch today's partial daily bar if the market session is active."""
    try:
        df = yf.download(
            ticker,
            period="1d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")
        missing = [c for c in RAW_OHLCV_COLS if c not in df.columns]
        if missing:
            return None
        return df[RAW_OHLCV_COLS].rename(columns=RENAME_MAP)
    except Exception:
        return None


def fetch_ohlcv(
    ticker: str,
    lookback_days: int = OHLCV_LOOKBACK_DAYS,
    include_partial: bool = True,
) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance for a ticker.

    Results are cached per-run so multiple calls for the same ticker (e.g.
    during scoring and trade plan building) do not duplicate API calls.

    If ``include_partial`` is True and the market session is active, today's
    incomplete bar is appended so volume-based checks can use live data.
    """
    cache_key = f"{ticker}_{lookback_days}_{include_partial}"
    if cache_key in _OHLCV_CACHE:
        return _OHLCV_CACHE[cache_key].copy()

    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No OHLCV data returned for ticker={ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")

    missing = [col for col in RAW_OHLCV_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected OHLCV columns for {ticker}: {missing}")

    df = df[RAW_OHLCV_COLS].rename(columns=RENAME_MAP)

    if include_partial:
        partial = _fetch_intraday_bar(ticker)
        if partial is not None:
            partial.index = pd.DatetimeIndex(partial.index)
            df.index = pd.DatetimeIndex(df.index)
            today = partial.index[0].normalize()
            if today not in df.index.normalize():
                df = pd.concat([df, partial])

    # Stale data detection
    df.index = pd.DatetimeIndex(df.index)
    last_date = df.index[-1]
    days_since = (pd.Timestamp(end) - last_date).days
    if days_since > STALE_DATA_MAX_DAYS:
        raise ValueError(
            f"Stale OHLCV for {ticker}: last bar is {days_since} days old"
        )

    _OHLCV_CACHE[cache_key] = df
    return df.copy()


def fetch_addv(ticker: str, lookback_days: int = 40) -> float | None:
    """Average daily dollar volume (close * 20-day volume MA).

    Uses a lightweight 40-day download so the 20-day rolling mean is stable.
    Returns None on any failure so callers can fall back gracefully.
    Results are cached per-run.
    """
    if ticker in _ADDV_CACHE:
        return _ADDV_CACHE[ticker]
    try:
        end = datetime.today()
        start = end - timedelta(days=lookback_days)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            _ADDV_CACHE[ticker] = None
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")
        close = pd.to_numeric(df["Close"], errors="coerce")
        vol = pd.to_numeric(df["Volume"], errors="coerce")
        vol_ma = vol.rolling(20).mean()
        last_close = close.iloc[-1]
        last_vol_ma = vol_ma.iloc[-1]
        if pd.isna(last_close) or pd.isna(last_vol_ma) or last_vol_ma <= 0:
            _ADDV_CACHE[ticker] = None
            return None
        result = float(last_close * last_vol_ma)
        _ADDV_CACHE[ticker] = result
        return result
    except Exception:
        _ADDV_CACHE[ticker] = None
        return None


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN rows, enforce types, and sort by date ascending."""
    df = df.copy()

    if df.empty:
        raise ValueError("Input OHLCV DataFrame is empty")

    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    for col in OHLCV_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=OHLCV_COLS)

    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype("float64")
    df["volume"] = df["volume"].astype("int64")

    return df


def _session_elapsed_frac() -> float | None:
    """Fraction of the regular trading session elapsed (0.05–1.0).

    Returns None if the market is closed (weekends / outside 9:30–16:00 ET).
    """
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return None
    from datetime import time as _time
    open_t = _time(9, 30)
    close_t = _time(16, 0)
    t = now.time()
    if t < open_t or t >= close_t:
        return None
    elapsed = (now.hour * 3600 + now.minute * 60 + now.second) - (9 * 3600 + 30 * 60)
    return max(0.05, min(1.0, elapsed / _SESSION_SECONDS))


def _intraday_rel_volume(volume_today: float, vol_ma20: float) -> float:
    """Time-of-day normalized relative volume for a partial intraday bar.

    Compares cumulative volume so far against the expected volume at this
    point in the session: ``expected = vol_ma20 * elapsed_fraction``.
    """
    frac = _session_elapsed_frac()
    if frac is None or vol_ma20 <= 0:
        return volume_today / vol_ma20 if vol_ma20 > 0 else 0.0
    expected = vol_ma20 * frac
    if expected <= 0:
        return 0.0
    return volume_today / expected


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA and ATR columns to a cleaned OHLCV DataFrame."""
    df = df.copy()

    if df.empty:
        raise ValueError("Cannot compute features on an empty DataFrame")

    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    df["ema20"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

    df["vol_ma20"] = df["volume"].rolling(EMA_SHORT).mean()
    df["rel_volume"] = df["volume"] / df["vol_ma20"]

    # ADX (Average Directional Index) — trend strength indicator
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    atr_smooth = df["atr14"]
    plus_di = 100 * (plus_dm.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean() / atr_smooth)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9) * 100
    df["adx"] = dx.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    # EMA slopes (5-bar rate of change, normalized by ATR)
    ema20_roc = df["ema20"].diff(5)
    df["ema20_slope"] = ema20_roc / atr_smooth
    ema50_roc = df["ema50"].diff(5)
    df["ema50_slope"] = ema50_roc / atr_smooth

    # Tag the last bar as intraday and normalize its rel_volume if
    # the market session is currently active and the bar is today's partial.
    df["is_intraday"] = False
    frac = _session_elapsed_frac()
    if frac is not None and len(df) >= 2:
        last_date = df.index[-1]
        today = pd.Timestamp(datetime.now(_ET).date())
        if pd.Timestamp(last_date).normalize() == today:
            vol_ma = df["vol_ma20"].iloc[-1]
            if pd.notna(vol_ma) and vol_ma > 0:
                df.iloc[-1, df.columns.get_loc("rel_volume")] = _intraday_rel_volume(
                    float(df["volume"].iloc[-1]), float(vol_ma)
                )
            df.iloc[-1, df.columns.get_loc("is_intraday")] = True

    return df
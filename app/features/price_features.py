"""Price-derived features: OHLCV fetch, cleaning, and technical indicators."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from app.config import ATR_PERIOD, EMA_LONG, EMA_SHORT, OHLCV_LOOKBACK_DAYS

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

    If ``include_partial`` is True and the market session is active, today's
    incomplete bar is appended so volume-based checks can use live data.
    """
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

    return df


def fetch_addv(ticker: str, lookback_days: int = 40) -> float | None:
    """Average daily dollar volume (close * 20-day volume MA).

    Uses a lightweight 40-day download so the 20-day rolling mean is stable.
    Returns None on any failure so callers can fall back gracefully.
    """
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
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")
        close = pd.to_numeric(df["Close"], errors="coerce")
        vol = pd.to_numeric(df["Volume"], errors="coerce")
        vol_ma = vol.rolling(20).mean()
        last_close = close.iloc[-1]
        last_vol_ma = vol_ma.iloc[-1]
        if pd.isna(last_close) or pd.isna(last_vol_ma) or last_vol_ma <= 0:
            return None
        return float(last_close * last_vol_ma)
    except Exception:
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

    return df
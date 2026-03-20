"""As-of OHLCV loader for backtesting — no lookahead bias."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from app.config import ATR_PERIOD, EMA_LONG, EMA_SHORT, OHLCV_LOOKBACK_DAYS
from app.features.price_features import OHLCV_COLS, RAW_OHLCV_COLS, RENAME_MAP


class PriceCache:
    """Pre-fetches full OHLCV history per ticker and serves as-of slices."""

    def __init__(self, lookback_days: int = OHLCV_LOOKBACK_DAYS):
        self._cache: dict[str, pd.DataFrame] = {}
        self._lookback_days = lookback_days

    def prefetch(self, tickers: list[str]) -> None:
        """Bulk-download history for all tickers that aren't cached yet."""
        missing = [t for t in tickers if t not in self._cache]
        if not missing:
            return

        end = datetime.today()
        start = end - timedelta(days=self._lookback_days)

        for ticker in missing:
            try:
                df = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    auto_adjust=False,
                    progress=False,
                )
                if df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel("Ticker")

                present = [c for c in RAW_OHLCV_COLS if c in df.columns]
                if len(present) != len(RAW_OHLCV_COLS):
                    continue

                df = df[RAW_OHLCV_COLS].rename(columns=RENAME_MAP)
                df.index = pd.DatetimeIndex(df.index)
                df = df.sort_index()

                for col in OHLCV_COLS:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=OHLCV_COLS)
                df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype("float64")
                df["volume"] = df["volume"].astype("int64")

                self._cache[ticker] = df
            except Exception:
                continue

        print(f"  [price_cache] prefetched {len(missing)} tickers, {len(self._cache)} in cache")

    def get_as_of(self, ticker: str, as_of_date: str) -> pd.DataFrame | None:
        """Return OHLCV with indicators computed using only data up to as_of_date."""
        raw = self._cache.get(ticker)
        if raw is None:
            return None

        cutoff = pd.Timestamp(as_of_date)
        sliced = raw[raw.index <= cutoff].copy()
        if len(sliced) < EMA_LONG + 10:
            return None

        return _compute_features(sliced)


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA and ATR columns — mirrors app.features.price_features.compute_features."""
    df = df.copy()
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

#!/usr/bin/env python3
"""Refresh `data/_ohlcv_cache` so the weekly replay has bars to work with.

Yahoo refuses the GitHub runners outright — a fetch of 300+ tickers there
fails in under two seconds — so CI can only ever read this directory, never
refill it. Left alone the cache falls behind `grade_history` and the replay
starts shedding its most recent rows to `not_enough_forward_bars`.

This runs on a host Yahoo will actually serve (see
`ops/com.appfin.warm-ohlcv.plist` for the weekly schedule) and fetches the
same window `replay_panel` asks for: from `ATR_WARMUP_DAYS` before the oldest
`as_of` through today, so every row has both its warm-up and its forward bars.

Frames are merged, never replaced. `grade_backtest` shares this directory and
writes short windows into it; overwriting would let the two callers keep
truncating each other.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_replay_backtest import (  # noqa: E402
    ATR_WARMUP_DAYS,
    GRADE_HISTORY,
    OHLCV_CACHE_DIR,
    _merge_cached,
    _ohlcv_cache_path,
)

RAW_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
BATCH = 40
# Slack on the near edge so a run is still covering after a week of new signals.
EDGE_BUFFER_DAYS = 21


def _universe() -> tuple[list[str], pd.Timestamp]:
    """Tickers to warm, plus the earliest bar the replay will ask for."""
    hist = pd.read_csv(GRADE_HISTORY)
    tickers = sorted(
        {
            str(t).strip().upper()
            for t in hist.get("ticker", pd.Series(dtype=str)).dropna()
            if str(t).strip()
        }
    )
    as_of = pd.to_datetime(hist.get("as_of"), errors="coerce").dropna()
    oldest = as_of.min() if not as_of.empty else pd.Timestamp.today()
    start = oldest - pd.Timedelta(days=ATR_WARMUP_DAYS + EDGE_BUFFER_DAYS)
    # SPY backs the relative-strength leg and is not always in grade_history.
    if "SPY" not in tickers:
        tickers.append("SPY")
    return tickers, start


def main() -> int:
    tickers, start = _universe()
    end = date.today() + timedelta(days=1)
    OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Warming {len(tickers)} tickers: {start.date()} → {end}")

    ok, failed = 0, []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        try:
            data = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            print(f"  batch {i // BATCH + 1}: {type(exc).__name__}")
            failed.extend(batch)
            continue

        for ticker in batch:
            try:
                multi = isinstance(data.columns, pd.MultiIndex)
                df = (data[ticker] if multi else data).dropna(how="all")
                if df.empty:
                    failed.append(ticker)
                    continue
                df = df[[c for c in RAW_COLS if c in df.columns]].rename(columns=RENAME)
                df.index = pd.DatetimeIndex(df.index)
                df.index.name = "Date"
                path = _ohlcv_cache_path(ticker)
                _merge_cached(path, df.sort_index()).to_csv(path)
                ok += 1
            except Exception:
                failed.append(ticker)
        print(f"  {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)

    print(f"\nWarmed {ok}, failed {len(failed)}")
    if failed:
        print("  failed: " + ", ".join(sorted(failed)[:25]))
    # Delistings are routine and must not fail the job; a wholesale outage must.
    return 1 if ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

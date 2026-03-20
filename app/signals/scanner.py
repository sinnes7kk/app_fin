"""Scan a watchlist and produce standardized signal dicts."""

from __future__ import annotations

import pandas as pd

from app.config import WATCHLIST
from app.features.price_features import clean_ohlcv, compute_features, fetch_ohlcv
from app.signals.scoring import score_long_setup, score_short_setup


def scan_ticker(ticker: str) -> dict:
    """Fetch data, compute features, score setups, and return a signal dict."""
    df = fetch_ohlcv(ticker)
    df = clean_ohlcv(df)
    df = compute_features(df)

    long_score = score_long_setup(df)
    short_score = score_short_setup(df)

    best = long_score if long_score["score"] >= short_score["score"] else short_score

    return {
        "ticker": ticker,
        "direction": best["direction"],
        "score": best["score"],
        "max_score": best["max_score"],
        "is_valid": best["is_valid"],
        "reasons": best["reasons"],
        "support": best["support"],
        "resistance": best["resistance"],
        "long": long_score,
        "short": short_score,
    }


def scan_watchlist(tickers: list[str] | None = None) -> list[dict]:
    """Run the scanner over every ticker and return a list of signal dicts."""
    tickers = tickers or WATCHLIST
    results: list[dict] = []

    for ticker in tickers:
        try:
            signal = scan_ticker(ticker)
            results.append(signal)
        except Exception as exc:
            print(f"[scanner] skipping {ticker}: {exc}")

    return results


def print_scan_results(signals: list[dict]) -> None:
    """Print a one-line summary per ticker, sorted by score."""
    from app.signals.ranking import rank_signals

    ranked = rank_signals(signals)
    for s in ranked:
        valid_flag = " <<< VALID" if s["is_valid"] else ""
        reasons = ", ".join(s["reasons"]) if s["reasons"] else "none"
        print(
            f"{s['ticker']:6s}  {s['direction']:5s}  "
            f"score={s['score']:2d}/{s['max_score']}  "
            f"S={s['support']:.2f}  R={s['resistance']:.2f}"
            f"{valid_flag}\n"
            f"        reasons: {reasons}"
        )

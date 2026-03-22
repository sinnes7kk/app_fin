"""Sector classification for portfolio concentration guards.

Uses yfinance's ``info.sector`` with a fast in-memory cache.  Falls back to
"Unknown" when the lookup fails so the guard never blocks a trade by accident.
"""

from __future__ import annotations

import yfinance as yf

_SECTOR_CACHE: dict[str, str] = {}

UNKNOWN = "Unknown"


def clear_sector_cache() -> None:
    _SECTOR_CACHE.clear()


def get_sector(ticker: str) -> str:
    """Return the GICS sector for *ticker*, cached per session."""
    if ticker in _SECTOR_CACHE:
        return _SECTOR_CACHE[ticker]
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or UNKNOWN
    except Exception:
        sector = UNKNOWN
    _SECTOR_CACHE[ticker] = sector
    return sector

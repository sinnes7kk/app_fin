"""Tests for :mod:`app.utils.market_calendar`.

The helper maps wall-clock time to the NYSE trading day whose closing
data the downstream trackers should attribute a snapshot to.  These
tests pin the rules so a regression (e.g. someone removing the 16:15 ET
cutoff) is caught immediately.

Run with::

    python -m pytest tests/test_market_calendar.py -v
    python -m tests.test_market_calendar     # standalone
"""
from __future__ import annotations

from datetime import date, datetime

from app.utils.market_calendar import (
    current_trading_day,
    _is_session_day,
    _previous_session,
    _EASTERN,
)


def _et(y, m, d, hh=12, mm=0):
    """Helper: build an ET-aware datetime."""
    return datetime(y, m, d, hh, mm, tzinfo=_EASTERN)


def test_weekday_post_close_returns_today():
    assert current_trading_day(_et(2026, 4, 20, 16, 15)) == date(2026, 4, 20)  # Mon 16:15
    assert current_trading_day(_et(2026, 4, 20, 20, 0)) == date(2026, 4, 20)   # Mon 20:00


def test_weekday_pre_close_returns_previous_session():
    assert current_trading_day(_et(2026, 4, 20, 7, 0)) == date(2026, 4, 17)   # Mon 07:00 -> Fri
    assert current_trading_day(_et(2026, 4, 20, 15, 59)) == date(2026, 4, 17) # Mon 15:59 -> Fri


def test_weekend_returns_previous_friday():
    assert current_trading_day(_et(2026, 4, 18, 10, 0)) == date(2026, 4, 17)  # Sat 10:00
    assert current_trading_day(_et(2026, 4, 19, 22, 0)) == date(2026, 4, 17)  # Sun 22:00


def test_holiday_rolls_back_when_pmc_available():
    """If pandas_market_calendars is installed, Mon holiday -> prior Fri.
    If not, the fallback is weekday-only; we skip the strict assertion."""
    try:
        import pandas_market_calendars  # noqa: F401
    except ImportError:
        return
    # 2026-01-19: MLK Day (NYSE closed).
    assert current_trading_day(_et(2026, 1, 19, 10, 0)) == date(2026, 1, 16)


def test_naive_datetime_is_interpreted_as_eastern():
    """Passing a naive datetime should be treated as ET, not UTC."""
    naive = datetime(2026, 4, 20, 7, 0)  # Mon 07:00 ET
    assert current_trading_day(naive) == date(2026, 4, 17)


def test_previous_session_skips_weekend():
    assert _previous_session(date(2026, 4, 20)) == date(2026, 4, 17)  # Mon -> Fri
    assert _previous_session(date(2026, 4, 18)) == date(2026, 4, 17)  # Sat -> Fri


def test_is_session_day_weekends():
    assert _is_session_day(date(2026, 4, 17)) is True   # Fri
    assert _is_session_day(date(2026, 4, 18)) is False  # Sat
    assert _is_session_day(date(2026, 4, 19)) is False  # Sun
    assert _is_session_day(date(2026, 4, 20)) is True   # Mon


if __name__ == "__main__":
    import traceback
    tests = [
        test_weekday_post_close_returns_today,
        test_weekday_pre_close_returns_previous_session,
        test_weekend_returns_previous_friday,
        test_holiday_rolls_back_when_pmc_available,
        test_naive_datetime_is_interpreted_as_eastern,
        test_previous_session_skips_weekend,
        test_is_session_day_weekends,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)

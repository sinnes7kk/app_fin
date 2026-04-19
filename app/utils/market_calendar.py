"""NYSE trading-day helpers.

Scans can run at any wall-clock time — pre-market, weekend, holiday — but
UW's ingest only advances after a session closes.  Stamping rows with
``date.today()`` in those windows produces a brand-new ``snapshot_date``
that really belongs to the previous closed session, which then duplicates
rows in ``screener_snapshots.csv`` and inflates Active Days / persistence
in the multi-day regression.

This module resolves the ambiguity by mapping wall-clock time to the NYSE
trading day whose close the data belongs to:

- NYSE open today AND time >= 16:15 ET -> today (post-close, figures final).
- NYSE open today AND time <  16:15 ET -> previous NYSE session
  (pre-market / intraday running totals still settle against the prior
  session's close until UW finishes ingesting).
- NYSE closed (weekend / holiday)      -> previous NYSE session.

All four daily-snapshot trackers (Flow, Dark Pool, Sentiment, Insider)
call ``current_trading_day()`` so their writes and retention cutoffs stay
aligned across the codebase.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

POST_CLOSE_CUTOFF = time(16, 15)

try:
    from zoneinfo import ZoneInfo
    _EASTERN = ZoneInfo("America/New_York")
except Exception:
    _EASTERN = timezone(timedelta(hours=-5))

try:
    import pandas_market_calendars as _mcal
    _NYSE = _mcal.get_calendar("NYSE")
except Exception:
    _NYSE = None


def _to_eastern(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(tz=_EASTERN)
    if now.tzinfo is None:
        return now.replace(tzinfo=_EASTERN)
    return now.astimezone(_EASTERN)


def _is_session_day(d: date) -> bool:
    """True if ``d`` is an NYSE regular-session day."""
    if _NYSE is not None:
        try:
            sched = _NYSE.schedule(start_date=d, end_date=d)
            return not sched.empty
        except Exception:
            pass
    return d.weekday() < 5


def _previous_session(d: date) -> date:
    """Most recent NYSE session strictly before ``d``."""
    if _NYSE is not None:
        try:
            start = d - timedelta(days=14)
            sched = _NYSE.schedule(start_date=start, end_date=d - timedelta(days=1))
            if not sched.empty:
                return sched.index[-1].date()
        except Exception:
            pass
    probe = d - timedelta(days=1)
    while not _is_session_day(probe):
        probe -= timedelta(days=1)
    return probe


def current_trading_day(now: Optional[datetime] = None) -> date:
    """Return the NYSE trading day that the data at ``now`` belongs to.

    Parameters
    ----------
    now
        Reference instant.  Defaults to the current wall-clock time in
        Eastern time.  Naive datetimes are interpreted as Eastern.

    Returns
    -------
    date
        The NYSE session date.  Never a weekend or exchange holiday.
    """
    et = _to_eastern(now)
    today = et.date()
    if _is_session_day(today) and et.time() >= POST_CLOSE_CUTOFF:
        return today
    return _previous_session(today)


def current_trading_day_str(now: Optional[datetime] = None) -> str:
    """Convenience wrapper returning ``current_trading_day(...).isoformat()``."""
    return current_trading_day(now).isoformat()

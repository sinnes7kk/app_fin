"""Layer 1 watchlist streak bookkeeping.

Covers ``add_candidates`` per-day observation tracking, ``_streak_summary``
derived fields, ``_compute_flow_trend`` slope classification, and the
``build_streak_lookup`` / ``apply_streak_lookup`` helpers used by the
pipeline to stamp streak metadata onto rejection and signal rows.

Run with either:
    python -m pytest tests/test_watchlist_streak.py -v
    python -m tests.test_watchlist_streak           # standalone (no pytest required)
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from unittest.mock import patch

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    class _PytestShim:
        def parametrize(self, name, values):
            def decorator(fn):
                fn.__parametrize__ = (name, values)
                return fn
            return decorator
        mark = None  # type: ignore

        def __getattr__(self, attr):
            return self

    pytest = _PytestShim()  # type: ignore
    pytest.mark = pytest  # parametrize lives under .mark in real pytest

from app.signals import watchlist as wl


def _approx(actual: float, expected: float, tol: float = 1e-6) -> None:
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (
        f"expected {expected}, got {actual}"
    )


def _reject(
    ticker: str = "AAA",
    direction: str = "LONG",
    flow_score_raw: float = 0.7,
    flow_score_scaled: float | None = None,
    reject_reason: str = "price_validation_failed",
) -> dict:
    return {
        "ticker": ticker,
        "direction": direction,
        "flow_score_raw": flow_score_raw,
        "flow_score_scaled": flow_score_scaled if flow_score_scaled is not None else flow_score_raw * 10,
        "reject_reason": reject_reason,
        "checks_passed": "trend_aligned",
        "checks_failed": "not_extended",
        "price_score": 4.5,
    }


def _frozen_today(iso: str):
    """Patch ``date.today`` inside ``app.signals.watchlist`` to a fixed date."""

    class _FixedDate(date):
        @classmethod
        def today(cls) -> date:  # type: ignore[override]
            return date.fromisoformat(iso)

    return patch.object(wl, "date", _FixedDate)


# ---------------------------------------------------------------------------
# _compute_flow_trend
# ---------------------------------------------------------------------------


def test_compute_flow_trend_too_short_returns_na() -> None:
    assert wl._compute_flow_trend([]) == "n/a"
    assert wl._compute_flow_trend([0.5]) == "n/a"
    assert wl._compute_flow_trend([0.5, 0.6]) == "n/a"


def test_compute_flow_trend_rising() -> None:
    assert wl._compute_flow_trend([0.4, 0.5, 0.6, 0.75]) == "rising"


def test_compute_flow_trend_falling() -> None:
    assert wl._compute_flow_trend([0.8, 0.7, 0.6, 0.5]) == "falling"


def test_compute_flow_trend_flat_within_threshold() -> None:
    assert wl._compute_flow_trend([0.50, 0.51, 0.49, 0.50]) == "flat"


def test_compute_flow_trend_uses_only_last_window() -> None:
    history = [0.9] * 10 + [0.4, 0.45, 0.5, 0.55, 0.6]
    assert wl._compute_flow_trend(history) == "rising"


# ---------------------------------------------------------------------------
# _backfill_streak (legacy entries)
# ---------------------------------------------------------------------------


def test_backfill_streak_initialises_legacy_entry() -> None:
    legacy = {
        "ticker": "OLD",
        "direction": "LONG",
        "flow_score_raw": 0.6,
        "first_seen": "2026-04-10",
    }
    out = wl._backfill_streak(dict(legacy))
    assert out["seen_dates"] == ["2026-04-10"]
    assert out["flow_score_history"] == [0.6]
    assert out["last_seen"] == "2026-04-10"


def test_backfill_streak_is_idempotent() -> None:
    entry = {
        "ticker": "OK",
        "direction": "LONG",
        "flow_score_raw": 0.7,
        "first_seen": "2026-04-10",
        "seen_dates": ["2026-04-10", "2026-04-11"],
        "flow_score_history": [0.6, 0.7],
        "last_seen": "2026-04-11",
    }
    out = wl._backfill_streak(dict(entry))
    assert out["seen_dates"] == ["2026-04-10", "2026-04-11"]
    assert out["flow_score_history"] == [0.6, 0.7]


# ---------------------------------------------------------------------------
# add_candidates — first observation
# ---------------------------------------------------------------------------


def test_add_candidates_creates_new_entry_with_streak_one() -> None:
    with _frozen_today("2026-04-22"):
        result = wl.add_candidates(existing=[], new_rejects=[_reject("AAA", flow_score_raw=0.65)])
    assert len(result) == 1
    e = result[0]
    assert e["ticker"] == "AAA"
    assert e["seen_dates"] == ["2026-04-22"]
    assert e["flow_score_history"] == [0.65]
    assert e["first_seen"] == "2026-04-22"
    assert e["last_seen"] == "2026-04-22"


# ---------------------------------------------------------------------------
# add_candidates — multi-day refresh appends streak data
# ---------------------------------------------------------------------------


def test_add_candidates_subsequent_day_appends_to_streak() -> None:
    with _frozen_today("2026-04-21"):
        day1 = wl.add_candidates(existing=[], new_rejects=[_reject("AAA", flow_score_raw=0.50)])
    with _frozen_today("2026-04-22"):
        day2 = wl.add_candidates(existing=day1, new_rejects=[_reject("AAA", flow_score_raw=0.65)])
    with _frozen_today("2026-04-23"):
        day3 = wl.add_candidates(existing=day2, new_rejects=[_reject("AAA", flow_score_raw=0.80)])
    e = day3[0]
    assert e["seen_dates"] == ["2026-04-21", "2026-04-22", "2026-04-23"]
    assert e["flow_score_history"] == [0.50, 0.65, 0.80]
    assert e["first_seen"] == "2026-04-21"
    assert e["last_seen"] == "2026-04-23"
    # flow_score_raw on the entry tracks today's value, not the historical max.
    _approx(e["flow_score_raw"], 0.80)


def test_add_candidates_same_day_keeps_strongest_observation() -> None:
    with _frozen_today("2026-04-22"):
        first = wl.add_candidates(existing=[], new_rejects=[_reject("AAA", flow_score_raw=0.40)])
        second = wl.add_candidates(existing=first, new_rejects=[_reject("AAA", flow_score_raw=0.70)])
        third = wl.add_candidates(existing=second, new_rejects=[_reject("AAA", flow_score_raw=0.55)])
    e = third[0]
    assert e["seen_dates"] == ["2026-04-22"]
    # Same-day refresh keeps the peak observation rather than the latest.
    assert e["flow_score_history"] == [0.70]


def test_add_candidates_caps_history_at_max() -> None:
    history: list[dict] = []
    base = date(2026, 3, 1)
    for i in range(15):
        d = (base + timedelta(days=i)).isoformat()
        with _frozen_today(d):
            history = wl.add_candidates(
                existing=history,
                new_rejects=[_reject("AAA", flow_score_raw=0.5 + i * 0.01)],
            )
    e = history[0]
    assert len(e["seen_dates"]) == wl.STREAK_HISTORY_MAX
    assert len(e["flow_score_history"]) == wl.STREAK_HISTORY_MAX
    # Oldest entries fall off the front; most recent observation is preserved.
    assert e["seen_dates"][-1] == (base + timedelta(days=14)).isoformat()


def test_add_candidates_skips_weak_flow_rejects() -> None:
    with _frozen_today("2026-04-22"):
        result = wl.add_candidates(
            existing=[],
            new_rejects=[
                _reject("AAA", reject_reason="weak_bullish_flow"),
                _reject("BBB", reject_reason="weak_bearish_flow"),
                _reject("CCC", reject_reason="error: boom"),
                _reject("DDD", reject_reason="price_validation_failed"),
            ],
        )
    assert {e["ticker"] for e in result} == {"DDD"}


def test_add_candidates_upgrades_legacy_entry_in_place() -> None:
    legacy = {
        "ticker": "OLD",
        "direction": "LONG",
        "flow_score_raw": 0.5,
        "flow_score_scaled": 5.0,
        "first_seen": "2026-04-18",
        "reject_reason": "price_validation_failed",
        "checks_passed": "",
        "checks_failed": "",
        "price_score": 3.0,
    }
    with _frozen_today("2026-04-22"):
        result = wl.add_candidates(existing=[legacy], new_rejects=[_reject("OLD", flow_score_raw=0.6)])
    e = result[0]
    assert e["seen_dates"] == ["2026-04-18", "2026-04-22"]
    assert len(e["flow_score_history"]) == 2
    assert e["first_seen"] == "2026-04-18"
    assert e["last_seen"] == "2026-04-22"


# ---------------------------------------------------------------------------
# _streak_summary
# ---------------------------------------------------------------------------


def test_streak_summary_basic_metrics() -> None:
    entry = {
        "ticker": "AAA",
        "direction": "LONG",
        "flow_score_raw": 0.85,
        "seen_dates": ["2026-04-19", "2026-04-20", "2026-04-21", "2026-04-22"],
        "flow_score_history": [0.4, 0.55, 0.7, 0.85],
        "first_seen": "2026-04-19",
        "last_seen": "2026-04-22",
    }
    s = wl._streak_summary(entry)
    assert s["watchlist_streak_days"] == 4
    _approx(s["watchlist_max_flow_score"], 0.85)
    _approx(s["watchlist_mean_flow_score_5d"], (0.4 + 0.55 + 0.7 + 0.85) / 4)
    assert s["watchlist_flow_trend"] == "rising"
    assert s["watchlist_first_seen"] == "2026-04-19"
    assert s["watchlist_last_seen"] == "2026-04-22"


def test_streak_summary_empty_history_falls_back_to_flow_score_raw() -> None:
    entry = {
        "ticker": "X",
        "direction": "LONG",
        "flow_score_raw": 0.42,
        "seen_dates": [],
        "flow_score_history": [],
    }
    s = wl._streak_summary(entry)
    assert s["watchlist_streak_days"] == 0
    _approx(s["watchlist_max_flow_score"], 0.42)
    _approx(s["watchlist_mean_flow_score_5d"], 0.42)
    assert s["watchlist_flow_trend"] == "n/a"


# ---------------------------------------------------------------------------
# build_streak_lookup / apply_streak_lookup
# ---------------------------------------------------------------------------


def test_build_and_apply_streak_lookup_stamps_rows() -> None:
    entries = [
        {
            "ticker": "AAA",
            "direction": "LONG",
            "flow_score_raw": 0.8,
            "seen_dates": ["2026-04-19", "2026-04-20", "2026-04-22"],
            "flow_score_history": [0.5, 0.6, 0.8],
            "first_seen": "2026-04-19",
            "last_seen": "2026-04-22",
        },
    ]
    lookup = wl.build_streak_lookup(entries)
    rows = [
        {"ticker": "AAA", "direction": "LONG", "flow_score_raw": 0.8},
        {"ticker": "AAA", "direction": "SHORT"},  # different direction, no match
        {"ticker": "BBB", "direction": "LONG"},   # not on watchlist
    ]
    wl.apply_streak_lookup(rows, lookup)
    assert rows[0]["watchlist_streak_days"] == 3
    assert rows[0]["watchlist_flow_trend"] == "rising"
    assert "watchlist_streak_days" not in rows[1]
    assert "watchlist_streak_days" not in rows[2]


def test_apply_streak_lookup_does_not_clobber_existing_fields() -> None:
    lookup = {("AAA", "LONG"): {"watchlist_streak_days": 99, "watchlist_flow_trend": "rising"}}
    rows = [{"ticker": "AAA", "direction": "LONG", "watchlist_streak_days": 5}]
    wl.apply_streak_lookup(rows, lookup)
    assert rows[0]["watchlist_streak_days"] == 5
    assert rows[0]["watchlist_flow_trend"] == "rising"


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------


def _run_standalone() -> int:
    import sys
    import traceback

    mod = sys.modules[__name__]
    tests = [
        (name, getattr(mod, name))
        for name in dir(mod)
        if name.startswith("test_") and callable(getattr(mod, name))
    ]
    failed: list[tuple[str, str]] = []
    passed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}: {e}")

    print()
    print(f"{passed} passed, {len(failed)} failed")
    if failed:
        print()
        for name, tb in failed:
            print(f"--- {name} ---")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_standalone())

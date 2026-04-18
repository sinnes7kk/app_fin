"""Tests for Wave 2 ``app.features.dp_stats`` — DP notional z-tier.

Validates:

* Tier 1 full-history path on a ticker with enough samples.
* Tier 2 shrunk path when per-ticker n ∈ [MIN_SHRUNK, MIN_FULL).
* Tier 3 peer path for a cold-start ticker with usable cohort.
* Tier 4 absolute-fallback when everything is missing.
* ``attach_dp_z_tiers`` decorates rows with the latest-day notional
  from ``daily_snapshots`` (not the window sum).
* Graceful handling of NaN / None inputs.

Run with:
    python -m pytest tests/test_dp_stats.py -v
    python -m tests.test_dp_stats        # standalone
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.features.dp_stats import (
    TIER_ABS,
    TIER_FULL,
    TIER_PEER,
    TIER_SHRUNK,
    attach_dp_z_tiers,
    compute_dp_z_tier,
)


def _history_for_ticker(ticker: str, n: int, base_notional: float, as_of: date) -> pd.DataFrame:
    """Build a synthetic ``dp_snapshots.csv``-style frame with ``n``
    back-dated rows of stable notional for one ticker."""
    rows = []
    for i in range(n):
        d = as_of - timedelta(days=(i + 1))
        rows.append(
            {
                "snapshot_date": d.isoformat(),
                "ticker": ticker,
                "total_notional": base_notional,
            }
        )
    return pd.DataFrame(rows)


def _cohort_history(as_of: date, n_tickers: int = 15, per_ticker_days: int = 3) -> pd.DataFrame:
    """Synthetic cohort with many tickers and modest history — used for
    the Tier-3 peer baseline."""
    rows = []
    for t_i in range(n_tickers):
        ticker = f"PEER{t_i:02d}"
        for d_i in range(per_ticker_days):
            rows.append(
                {
                    "snapshot_date": (as_of - timedelta(days=d_i)).isoformat(),
                    "ticker": ticker,
                    "total_notional": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_tier_full_when_enough_history():
    as_of = date(2025, 1, 15)
    hist = _history_for_ticker("AAPL", n=25, base_notional=1_000_000, as_of=as_of)
    # Today's notional = 5x the median → should fire hot.
    res = compute_dp_z_tier("AAPL", 5_000_000.0, history=hist, as_of=as_of)
    assert res["tier"] == TIER_FULL
    assert res["n"] == 25
    # MAD is 0 for a perfectly-flat series; dp_stats floors it so z stays
    # finite.  With our floor that will collapse to 0, not infinite.
    assert res["z"] is not None


def test_tier_shrunk_with_modest_history():
    as_of = date(2025, 1, 15)
    ticker_hist = _history_for_ticker("TSLA", n=8, base_notional=2_000_000, as_of=as_of)
    cohort = _cohort_history(as_of)
    hist = pd.concat([ticker_hist, cohort], ignore_index=True)
    res = compute_dp_z_tier("TSLA", 6_000_000.0, history=hist, as_of=as_of)
    assert res["tier"] == TIER_SHRUNK
    assert res["n"] == 8


def test_tier_peer_for_cold_start():
    as_of = date(2025, 1, 15)
    cohort = _cohort_history(as_of, n_tickers=20, per_ticker_days=4)
    # "NEWIPO" has no history at all → peer baseline takes over.
    res = compute_dp_z_tier("NEWIPO", 1_000_000.0, history=cohort, as_of=as_of)
    assert res["tier"] == TIER_PEER
    # z should be tiny — we're basically at the peer median.
    assert res["z"] is not None
    assert abs(res["z"]) < 0.5


def test_tier_absolute_when_no_history():
    res = compute_dp_z_tier("XYZ", 1_000_000.0, history=pd.DataFrame(), as_of=date(2025, 1, 15))
    assert res["tier"] == TIER_ABS
    assert res["z"] is None


def test_none_notional_returns_absolute_fallback():
    as_of = date(2025, 1, 15)
    hist = _history_for_ticker("AAPL", n=25, base_notional=1_000_000, as_of=as_of)
    res = compute_dp_z_tier("AAPL", None, history=hist, as_of=as_of)
    assert res["tier"] == TIER_ABS
    assert res["z"] is None


def test_attach_dp_z_tiers_prefers_latest_snapshot():
    """``attach_dp_z_tiers`` should use the LAST active ``daily_snapshots``
    entry, not ``cumulative_notional``.  This keeps the z-score
    interpretable as "today's flow" rather than "window sum"."""
    as_of = date(2025, 1, 15)
    # Give AAPL a very stable 25d baseline with strict variance so that
    # a 5x day shows up as a clearly-positive z (even post-MAD floor).
    hist_rows = []
    # Use varied-but-stable daily notionals so median and MAD are both
    # well-defined (alternating two-value oscillation makes MAD collapse
    # to 0 which is a legitimate edge case but not what we want here).
    pattern = [900_000, 950_000, 1_000_000, 1_050_000, 1_100_000]
    for i in range(25):
        d = as_of - timedelta(days=(i + 1))
        n = pattern[i % len(pattern)]
        hist_rows.append({"snapshot_date": d.isoformat(), "ticker": "AAPL", "total_notional": n})
    hist = pd.DataFrame(hist_rows)

    tracker = [
        {
            "ticker": "AAPL",
            "cumulative_notional": 25_000_000.0,  # window sum — should be ignored
            "daily_snapshots": [
                {"date": "2025-01-14", "active": True, "notional": 950_000},
                {"date": "2025-01-15", "active": True, "notional": 5_000_000},
            ],
        }
    ]
    attach_dp_z_tiers(tracker, history=hist, as_of=as_of)
    row = tracker[0]
    assert row["dp_tier"] == TIER_FULL
    assert row["dp_z"] is not None
    # 5x a ~1M median is a very positive z.
    assert row["dp_z"] > 2.0


def test_attach_dp_z_tiers_handles_empty_snapshots_gracefully():
    tracker = [{"ticker": "ABC", "cumulative_notional": None, "daily_snapshots": []}]
    attach_dp_z_tiers(tracker, history=pd.DataFrame(), as_of=date(2025, 1, 15))
    assert tracker[0]["dp_tier"] == TIER_ABS
    assert tracker[0]["dp_z"] is None


if __name__ == "__main__":
    import traceback

    tests = [
        test_tier_full_when_enough_history,
        test_tier_shrunk_with_modest_history,
        test_tier_peer_for_cold_start,
        test_tier_absolute_when_no_history,
        test_none_notional_returns_absolute_fallback,
        test_attach_dp_z_tiers_prefers_latest_snapshot,
        test_attach_dp_z_tiers_handles_empty_snapshots_gracefully,
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

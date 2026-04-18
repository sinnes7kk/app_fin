"""Tests for the Premium-Taxonomy plan.

Covers four invariants that together guarantee the Flow Tracker's
multi-day regression sees consistent inputs:

1.  ``aggregate_premium_by_dte_bucket`` slices unusual flow into the
    configured DTE buckets and never leaks premium between them.
2.  ``SNAPSHOT_COLS`` carries every bucketed + total + unusual +
    ``premium_source`` column, and ``_build_snapshot_row`` populates
    them for both ``screener`` and ``per_ticker_api`` sources.
3.  ``save_flow_feature_snapshot`` drops tickers whose per-ticker API
    fetch failed (no silent narrow-premium fallback).
4.  ``compute_multi_day_flow`` regresses on ``total_bullish_premium``
    / ``total_bearish_premium`` — not the legacy narrow columns.

Run with either:

    python -m pytest tests/test_premium_taxonomy.py -v
    python -m tests.test_premium_taxonomy           # standalone (no pytest)
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# 1. DTE bucket aggregation.
# ─────────────────────────────────────────────────────────────────────────────

def test_aggregate_premium_by_dte_bucket_splits_flow_into_configured_buckets():
    """Each trade lands in exactly one of {lottery, swing, leap} given
    the default bucket boundaries (0-14 / 30-120 / 180+) and bearish
    prints stay on the bearish side."""
    from app.config import FLOW_TRACKER_PREMIUM_BUCKETS
    from app.features.flow_features import aggregate_premium_by_dte_bucket

    df = pd.DataFrame([
        {"ticker": "NVDA", "premium": 1_000_000, "dte":   7, "direction": "LONG"},
        {"ticker": "NVDA", "premium":   500_000, "dte":  45, "direction": "LONG"},
        {"ticker": "NVDA", "premium":   400_000, "dte":  45, "direction": "SHORT"},
        {"ticker": "NVDA", "premium": 2_000_000, "dte": 300, "direction": "LONG"},
        {"ticker": "NFLX", "premium":   750_000, "dte":  60, "direction": "SHORT"},
        {"ticker": "NFLX", "premium":   100_000, "dte":  20, "direction": "LONG"},
    ])

    out = aggregate_premium_by_dte_bucket(df, buckets=FLOW_TRACKER_PREMIUM_BUCKETS)
    out = out.set_index("ticker")

    nvda = out.loc["NVDA"]
    assert _approx(nvda["lottery_bullish_premium"], 1_000_000)
    assert _approx(nvda["lottery_bearish_premium"], 0)
    assert _approx(nvda["swing_bullish_premium"],   500_000)
    assert _approx(nvda["swing_bearish_premium"],   400_000)
    assert _approx(nvda["leap_bullish_premium"],    2_000_000)
    assert _approx(nvda["leap_bearish_premium"],    0)

    nflx = out.loc["NFLX"]
    assert _approx(nflx["swing_bearish_premium"], 750_000)
    # DTE 20 falls in none of the configured buckets (lottery=0-14,
    # swing=30-120, leap=180+).  The aggregator must not silently roll
    # it into an adjacent bucket.
    assert _approx(nflx["lottery_bullish_premium"], 0)
    assert _approx(nflx["swing_bullish_premium"],   0)
    assert _approx(nflx["leap_bullish_premium"],    0)


def test_aggregate_premium_by_dte_bucket_handles_empty_input():
    from app.features.flow_features import aggregate_premium_by_dte_bucket

    out = aggregate_premium_by_dte_bucket(pd.DataFrame())
    assert out.empty
    for col in ("ticker", "swing_bullish_premium", "lottery_bearish_premium", "leap_bullish_premium"):
        assert col in out.columns


# ─────────────────────────────────────────────────────────────────────────────
# 2. Snapshot column set + _build_snapshot_row.
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_TAXONOMY_COLS = {
    "total_bullish_premium",
    "total_bearish_premium",
    "unusual_bullish_premium",
    "unusual_bearish_premium",
    "lottery_bullish_premium",
    "lottery_bearish_premium",
    "swing_bullish_premium",
    "swing_bearish_premium",
    "leap_bullish_premium",
    "leap_bearish_premium",
    "premium_source",
}


def test_snapshot_cols_contains_full_premium_taxonomy():
    """Every column the multi-day regression + Premium Mix UI rely on
    must be present in ``SNAPSHOT_COLS`` so the CSV writer never emits
    ragged rows after a schema bump."""
    from app.features.flow_tracker import SNAPSHOT_COLS

    missing = REQUIRED_TAXONOMY_COLS - set(SNAPSHOT_COLS)
    assert not missing, f"SNAPSHOT_COLS missing taxonomy columns: {missing}"


def test_build_snapshot_row_tags_screener_source_and_fills_taxonomy():
    from app.features.flow_tracker import SNAPSHOT_COLS, _build_snapshot_row

    row = _build_snapshot_row(
        snapshot_date="2026-04-18",
        ticker="NVDA",
        source="screener",
        total_bullish_premium=10_000_000,
        total_bearish_premium=2_000_000,
        buckets={
            "lottery_bullish_premium": 500_000,
            "swing_bullish_premium":   8_000_000,
            "leap_bullish_premium":    1_500_000,
            "unusual_bullish_premium": 10_000_000,
            "unusual_bearish_premium": 1_000_000,
        },
        base={"sector": "Tech", "close": 910.0, "marketcap": 2_200_000_000_000},
    )

    assert set(row.keys()) == set(SNAPSHOT_COLS)
    assert row["premium_source"] == "screener"
    assert _approx(row["total_bullish_premium"], 10_000_000)
    assert _approx(row["total_bearish_premium"], 2_000_000)
    assert _approx(row["bullish_premium"], 10_000_000), "legacy alias must match total"
    assert _approx(row["bearish_premium"], 2_000_000)
    assert _approx(row["swing_bullish_premium"], 8_000_000)
    assert _approx(row["leap_bullish_premium"],  1_500_000)
    assert _approx(row["unusual_bullish_premium"], 10_000_000)
    assert _approx(row["unusual_bearish_premium"], 1_000_000)


def test_build_snapshot_row_zero_fills_missing_buckets():
    """Screener rows for tickers with no qualifying flow alerts get
    zeroed bucket columns — never ``None`` / ``NaN`` — so pandas
    coercion in the regression stays a no-op."""
    from app.features.flow_tracker import _build_snapshot_row

    row = _build_snapshot_row(
        snapshot_date="2026-04-18",
        ticker="AAPL",
        source="screener",
        total_bullish_premium=1_000_000,
        total_bearish_premium=500_000,
        buckets=None,
    )
    for col in (
        "lottery_bullish_premium", "lottery_bearish_premium",
        "swing_bullish_premium",   "swing_bearish_premium",
        "leap_bullish_premium",    "leap_bearish_premium",
    ):
        assert row[col] == 0.0, f"{col} should default to 0.0, got {row[col]!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. save_flow_feature_snapshot drops tickers with no API data (no silent
#    narrow-premium fallback).
# ─────────────────────────────────────────────────────────────────────────────

def test_save_flow_feature_snapshot_skips_tickers_with_failed_fetch():
    """The Premium-Taxonomy plan bans silent narrow-premium fallback.
    When ``fetch_ticker_options_snapshot`` returns ``None`` the ticker
    must be skipped entirely rather than written with its flow-alert
    premium."""
    from app.features import flow_tracker as ft_mod
    from app.vendors import unusual_whales as uw_mod

    tmp = Path(tempfile.mkdtemp(prefix="ft_taxo_"))
    original_path = ft_mod.SNAPSHOTS_PATH
    original_fetch = uw_mod.fetch_ticker_options_snapshot
    try:
        target = tmp / "screener_snapshots.csv"
        ft_mod.SNAPSHOTS_PATH = target

        def _fake_fetch(_ticker: str):
            return None

        uw_mod.fetch_ticker_options_snapshot = _fake_fetch

        feature_table = pd.DataFrame([
            {"ticker": "FAKE", "marketcap": 10_000_000_000, "total_count": 42},
        ])
        ft_mod.save_flow_feature_snapshot(feature_table)

        if target.exists():
            with open(target, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            assert all(r.get("ticker") != "FAKE" for r in rows), (
                "failed-fetch ticker must not be written — no silent narrow fallback"
            )
        # Either no file, or file without our ticker; both are fine.
    finally:
        ft_mod.SNAPSHOTS_PATH = original_path
        uw_mod.fetch_ticker_options_snapshot = original_fetch
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_multi_day_flow regresses on total_*_premium.
# ─────────────────────────────────────────────────────────────────────────────

def _days_back(n: int) -> list[str]:
    today = date.today()
    return [str(today - timedelta(days=i)) for i in range(n - 1, -1, -1)]


def _row(ticker: str, d: str, total_bull: float, total_bear: float = 0.0) -> dict:
    """Synthetic row with *only* total_*_premium populated (and the
    legacy alias set to zero) so we can prove the regression reads the
    total columns."""
    return {
        "snapshot_date": d,
        "ticker": ticker,
        "sector": "Tech",
        "close": 100.0,
        "marketcap": 80_000_000_000,
        "bullish_premium": 0.0,
        "bearish_premium": 0.0,
        "total_bullish_premium": total_bull,
        "total_bearish_premium": total_bear,
        "net_premium": total_bull - total_bear,
        "put_call_ratio": 0.3,
        "iv_rank": 50,
        "iv30d": 0.3,
        "perc_3_day_total": 0.8,
        "perc_30_day_total": 0.8,
        "premium_source": "screener",
    }


def _run_with_rows(rows: list[dict], lookback_days: int = 5, min_active_days: int = 2):
    from app.features import flow_tracker as ft_mod
    from app.features.flow_tracker import SNAPSHOT_COLS

    tmp = Path(tempfile.mkdtemp(prefix="ft_taxo_"))
    try:
        target = tmp / "screener_snapshots.csv"
        with open(target, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c) for c in SNAPSHOT_COLS})
        original = ft_mod.SNAPSHOTS_PATH
        ft_mod.SNAPSHOTS_PATH = target
        try:
            return ft_mod.compute_multi_day_flow(
                lookback_days=lookback_days, min_active_days=min_active_days
            )
        finally:
            ft_mod.SNAPSHOTS_PATH = original
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compute_multi_day_flow_regresses_on_total_premium():
    """Only ``total_*_premium`` is populated (legacy columns zeroed).
    If the regression is reading the new columns the ticker shows up
    with non-zero cumulative premium; if it's still reading the legacy
    narrow columns the ticker would be dropped at the premium floor."""
    days = _days_back(5)
    rows = [
        _row("TAXO", d, total_bull=3_000_000 + i * 1_500_000)
        for i, d in enumerate(days)
    ]
    out = _run_with_rows(rows)
    assert len(out) == 1, "regression should surface the ticker using total_*_premium"
    row = out[0]
    assert row["ticker"] == "TAXO"
    assert row["cumulative_premium"] >= 20_000_000, (
        f"expected cumulative ≥ 20M from total_*_premium, got {row['cumulative_premium']}"
    )
    assert row["direction"] == "BULLISH"
    assert "premium_mix" in row
    mix = row["premium_mix"]
    assert mix.get("total_bullish") >= 20_000_000
    assert mix.get("source") == "screener"


def test_compute_multi_day_flow_falls_back_to_legacy_columns():
    """Legacy rows (pre-taxonomy) only have ``bullish_premium`` /
    ``bearish_premium`` populated.  The regression should still work
    during the one-release migration window, pulling the legacy values
    in as ``total_*_premium`` via ``.fillna``."""
    days = _days_back(5)

    def _legacy_row(d: str, bull: float) -> dict:
        r = _row("LEGAC", d, total_bull=0.0)
        r["bullish_premium"] = bull
        r["total_bullish_premium"] = None
        return r

    rows = [_legacy_row(d, 3_000_000 + i * 1_500_000) for i, d in enumerate(days)]
    out = _run_with_rows(rows)
    assert len(out) == 1, "legacy-only rows must still flow through the regression"
    assert out[0]["cumulative_premium"] >= 20_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest required).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_aggregate_premium_by_dte_bucket_splits_flow_into_configured_buckets,
        test_aggregate_premium_by_dte_bucket_handles_empty_input,
        test_snapshot_cols_contains_full_premium_taxonomy,
        test_build_snapshot_row_tags_screener_source_and_fills_taxonomy,
        test_build_snapshot_row_zero_fills_missing_buckets,
        test_save_flow_feature_snapshot_skips_tickers_with_failed_fetch,
        test_compute_multi_day_flow_regresses_on_total_premium,
        test_compute_multi_day_flow_falls_back_to_legacy_columns,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(0 if failed == 0 else 1)

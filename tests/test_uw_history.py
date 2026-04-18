"""Tests for the UW-backed flow-intensity z-score hydrator.

Covers the four invariants that guarantee the Flow Tracker can z-score
``flow_intensity`` from day one, regardless of how thin the internal
``data/flow_features/`` store is:

1.  ``load_uw_intensity_history`` builds a ``ticker/date/bullish_flow_intensity/
    bearish_flow_intensity`` frame from a mocked UW payload and derives
    intensity as ``bullish_premium / marketcap_today``.
2.  The 24-hour on-disk cache is respected: fresh files short-circuit the
    UW fetch; stale files trigger a refresh.
3.  The marketcap floor (``1e8``) drops ETFs / micro-caps so they don't
    poison the baseline.
4.  End-to-end: feeding the UW-hydrated history into
    ``rescore_with_z(components=['flow_intensity'])`` produces Tier-1
    (per-ticker) tier tags for ``bullish_flow_intensity`` while every
    non-hydrated component (ppt, vol_oi, repeat, sweep, breadth, dte)
    stays on Tier 4 (absolute fallback).

Run with either:

    python -m pytest tests/test_uw_history.py -v
    python -m tests.test_uw_history           # standalone (no pytest)
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore


def _synthetic_history(ticker: str, days: int = 30) -> pd.DataFrame:
    """Newest-first frame mirroring ``fetch_ticker_options_history``."""
    base = pd.Timestamp("2026-04-18")
    rows = []
    for i in range(days):
        d = (base - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append(
            {
                "date": d,
                "bullish_premium": 1_000_000.0 + i * 25_000.0,
                "bearish_premium": 500_000.0 + i * 10_000.0,
                "call_volume": 100_000 + i * 1000,
                "put_volume": 50_000 + i * 500,
                "call_open_interest": 2_000_000 + i * 10_000,
                "put_open_interest": 1_000_000 + i * 5_000,
            }
        )
    return pd.DataFrame(rows)


def _with_tmp_cache_dir(fn):
    """Redirect uw_history.CACHE_DIR to a temp dir for the duration of ``fn``."""
    from app.features import uw_history

    tmp = Path(tempfile.mkdtemp(prefix="uw_history_test_"))
    original = uw_history.CACHE_DIR
    uw_history.CACHE_DIR = tmp
    try:
        return fn(tmp)
    finally:
        uw_history.CACHE_DIR = original
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic hydration: UW payload → flow_intensity frame.
# ─────────────────────────────────────────────────────────────────────────────

def test_load_uw_intensity_history_derives_flow_intensity_from_premium_and_mcap():
    """A synthetic 30-day payload for two tickers yields 60 rows with
    ``bullish_flow_intensity = bullish_premium / mcap`` and matching
    bearish values; no other component columns are populated."""
    from app.features import uw_history

    def _run(_tmp: Path) -> pd.DataFrame:
        with patch.object(
            uw_history,
            "fetch_ticker_options_history",
            side_effect=lambda t, days=30: _synthetic_history(t, days=days),
        ):
            return uw_history.load_uw_intensity_history(
                tickers=["NVDA", "AAPL"],
                marketcap_map={"NVDA": 3e12, "AAPL": 3.5e12},
            )

    out = _with_tmp_cache_dir(_run)

    assert not out.empty, "loader must produce rows for tickers with valid mcap"
    assert set(out.columns) == {
        "ticker",
        "date",
        "bullish_flow_intensity",
        "bearish_flow_intensity",
    }, f"unexpected columns: {out.columns.tolist()}"
    assert out["ticker"].nunique() == 2
    assert (out.groupby("ticker").size() == 30).all(), (
        "each ticker should have 30 daily rows"
    )

    # Spot-check the derivation for the newest NVDA row.
    nvda = out[out["ticker"] == "NVDA"].sort_values("date", ascending=False).iloc[0]
    expected_bull = 1_000_000.0 / 3e12
    expected_bear = 500_000.0 / 3e12
    assert abs(nvda["bullish_flow_intensity"] - expected_bull) < 1e-15
    assert abs(nvda["bearish_flow_intensity"] - expected_bear) < 1e-15


# ─────────────────────────────────────────────────────────────────────────────
# 2. 24h cache TTL.
# ─────────────────────────────────────────────────────────────────────────────

def test_fresh_cache_short_circuits_uw_fetch():
    """When a ticker's cache file is <24h old, the loader must not call
    ``fetch_ticker_options_history`` at all."""
    from app.features import uw_history

    def _run(tmp: Path) -> int:
        uw_history._write_cache(
            uw_history._cache_path("NVDA"), _synthetic_history("NVDA")
        )
        call_count = {"n": 0}

        def _tracking_fetch(ticker, days=30):
            call_count["n"] += 1
            return _synthetic_history(ticker, days=days)

        with patch.object(uw_history, "fetch_ticker_options_history", _tracking_fetch):
            out = uw_history.load_uw_intensity_history(
                tickers=["NVDA"], marketcap_map={"NVDA": 3e12}
            )
            assert not out.empty, "cached rows should still produce a frame"
        return call_count["n"]

    n = _with_tmp_cache_dir(_run)
    assert n == 0, f"fresh cache should skip UW fetch; got {n} calls"


def test_stale_cache_triggers_uw_refresh():
    """When a cache file is older than TTL, the loader refreshes from UW."""
    from app.features import uw_history

    def _run(tmp: Path) -> int:
        path = uw_history._cache_path("NVDA")
        uw_history._write_cache(path, _synthetic_history("NVDA"))
        stale = time.time() - 25 * 3600
        os.utime(path, (stale, stale))

        call_count = {"n": 0}

        def _tracking_fetch(ticker, days=30):
            call_count["n"] += 1
            return _synthetic_history(ticker, days=days)

        with patch.object(uw_history, "fetch_ticker_options_history", _tracking_fetch):
            uw_history.load_uw_intensity_history(
                tickers=["NVDA"], marketcap_map={"NVDA": 3e12}
            )
        return call_count["n"]

    n = _with_tmp_cache_dir(_run)
    assert n == 1, f"stale cache must trigger exactly 1 refresh; got {n}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Marketcap floor.
# ─────────────────────────────────────────────────────────────────────────────

def test_micro_cap_ticker_is_filtered_out():
    """Tickers below the $100M mcap floor (ETFs, broken mcap) are dropped
    before any UW fetch — mirrors the floor in ``flow_features.py``."""
    from app.features import uw_history

    def _run(_tmp: Path) -> pd.DataFrame:
        call_count = {"n": 0}

        def _tracking_fetch(ticker, days=30):
            call_count["n"] += 1
            return _synthetic_history(ticker, days=days)

        with patch.object(uw_history, "fetch_ticker_options_history", _tracking_fetch):
            out = uw_history.load_uw_intensity_history(
                tickers=["TINY", "NVDA"],
                marketcap_map={"TINY": 5e7, "NVDA": 3e12},
            )
        assert call_count["n"] == 1, (
            f"mcap-filtered ticker must NOT hit UW; got {call_count['n']} calls"
        )
        return out

    out = _with_tmp_cache_dir(_run)
    assert set(out["ticker"].unique()) == {"NVDA"}, (
        f"TINY should be dropped at the mcap floor; got {sorted(out['ticker'].unique())}"
    )


def test_missing_mcap_ticker_is_skipped():
    """Tickers with missing marketcap are skipped (no divide-by-NaN)."""
    from app.features import uw_history

    def _run(_tmp: Path) -> pd.DataFrame:
        with patch.object(
            uw_history,
            "fetch_ticker_options_history",
            side_effect=lambda t, days=30: _synthetic_history(t, days=days),
        ):
            return uw_history.load_uw_intensity_history(
                tickers=["GHOST", "NVDA"],
                marketcap_map={"NVDA": 3e12},
            )

    out = _with_tmp_cache_dir(_run)
    assert "GHOST" not in out["ticker"].unique()
    assert "NVDA" in out["ticker"].unique()


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-end: rescore_with_z(components=['flow_intensity']).
# ─────────────────────────────────────────────────────────────────────────────

def _build_agg_today(tickers: list[str]) -> pd.DataFrame:
    """Minimal aggregated flow frame with the columns
    ``_weighted_flow_score_mixed`` expects. Intensity today is ``10x`` the
    historical median so the per-ticker z-score clearly goes positive."""
    rows = []
    for t in tickers:
        rows.append(
            {
                "ticker": t,
                "marketcap": 3e12,
                # flow_intensity: today is 10x the historical level
                "bullish_flow_intensity": (1_000_000.0 / 3e12) * 10,
                "bearish_flow_intensity": (500_000.0 / 3e12) * 10,
                # Non-hydrated components — valued so abs path has something
                # to clip-scale but the z path should NOT touch them.
                "bullish_ppt_bps": 100.0,
                "bearish_ppt_bps": 50.0,
                "bullish_vol_oi": 0.5,
                "bearish_vol_oi": 0.2,
                "bullish_repeat_count": 3,
                "bearish_repeat_count": 1,
                "bullish_sweep_count": 2,
                "bearish_sweep_count": 1,
                "bullish_breadth": 0.5,
                "bearish_breadth": 0.2,
                "dte_score": 0.5,
                "bullish_score": 0.0,
                "bearish_score": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_rescore_with_z_components_hydrates_intensity_only():
    """With UW-backed history for ``bullish_flow_intensity`` /
    ``bearish_flow_intensity`` only, ``rescore_with_z(components=['flow_intensity'])``
    must tag the intensity component as Tier 1 (n=30 per ticker) while
    every other component stays on Tier 4 (absolute fallback)."""
    from app.features import uw_history
    from app.features.flow_features import rescore_with_z
    from app.features.flow_stats import TIER_ABS, TIER_FULL

    tickers = ["NVDA", "AAPL"]

    def _run(_tmp: Path) -> pd.DataFrame:
        with patch.object(
            uw_history,
            "fetch_ticker_options_history",
            side_effect=lambda t, days=30: _synthetic_history(t, days=days),
        ):
            history = uw_history.load_uw_intensity_history(
                tickers=tickers,
                marketcap_map={t: 3e12 for t in tickers},
            )
        agg = _build_agg_today(tickers)
        return rescore_with_z(agg, history, components=["flow_intensity"])

    out = _with_tmp_cache_dir(_run)

    # Intensity: Tier 1 (full per-ticker history, n >= 20)
    assert (out["bullish_flow_intensity_tier"] == TIER_FULL).all(), (
        f"expected all Tier 1 for bullish_flow_intensity; got "
        f"{out['bullish_flow_intensity_tier'].tolist()}"
    )
    assert (out["bearish_flow_intensity_tier"] == TIER_FULL).all(), (
        f"expected all Tier 1 for bearish_flow_intensity; got "
        f"{out['bearish_flow_intensity_tier'].tolist()}"
    )

    # Non-hydrated components must stay on Tier 4 (absolute).
    for abs_col in (
        "bullish_premium_per_trade_tier",
        "bearish_premium_per_trade_tier",
        "bullish_vol_oi_tier",
        "bearish_vol_oi_tier",
        "bullish_repeat_tier",
        "bearish_repeat_tier",
        "bullish_sweep_tier",
        "bearish_sweep_tier",
        "bullish_breadth_tier",
        "bearish_breadth_tier",
        "bullish_dte_tier",
        "bearish_dte_tier",
    ):
        assert abs_col in out.columns, f"missing tier column: {abs_col}"
        assert (out[abs_col] == TIER_ABS).all(), (
            f"{abs_col} should stay on Tier 4 (absolute) when not in components="
            f"['flow_intensity']; got {out[abs_col].tolist()}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest required).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_load_uw_intensity_history_derives_flow_intensity_from_premium_and_mcap,
        test_fresh_cache_short_circuits_uw_fetch,
        test_stale_cache_triggers_uw_refresh,
        test_micro_cap_ticker_is_filtered_out,
        test_missing_mcap_ticker_is_skipped,
        test_rescore_with_z_components_hydrates_intensity_only,
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

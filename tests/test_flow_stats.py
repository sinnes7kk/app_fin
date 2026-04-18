"""Tests for app.features.flow_stats and the z-score integration path.

Run with either:
    python -m pytest tests/test_flow_stats.py -v
    python -m tests.test_flow_stats           # standalone (no pytest required)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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

from app.features.flow_features import (
    ZStatsBundle,
    aggregate_flow_by_ticker,
    build_flow_feature_table,
    build_z_stats_bundle,
    rescore_with_z,
)
from app.features.flow_stats import (
    TIER_ABS,
    TIER_FULL,
    TIER_PEER,
    TIER_SHRUNK,
    all_scored_columns,
    compute_z_with_tier,
    cross_sectional_stats,
    logistic_to_unit,
    per_ticker_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_history(days: int, tickers: list[str], *, value: float = 1.0, noise: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic history with `days` daily rows per ticker.

    Each component column is populated with `value` ± noise so median ≈ value,
    MAD ≈ noise / 2.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-01-01")
    for i in range(days):
        d = (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for t in tickers:
            row = {"ticker": t, "date": d}
            for col in all_scored_columns():
                row[col] = value + rng.uniform(-noise, noise)
            rows.append(row)
    return pd.DataFrame(rows)


def _synth_today(tickers: list[str], *, value: float = 1.0) -> pd.DataFrame:
    """Build a minimal `today` agg DataFrame with all scored columns."""
    rows = []
    for t in tickers:
        row = {"ticker": t}
        for col in all_scored_columns():
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tier assignment tests
# ---------------------------------------------------------------------------

def test_tier_full_with_30d_history():
    """Ticker with 30 days → Tier 1 (full per-ticker z-score)."""
    hist = _synth_history(days=30, tickers=["AAPL"], value=1.0, noise=0.05)
    today = _synth_today(["AAPL"], value=1.5)
    bundle = build_z_stats_bundle(hist, today)

    z_df = compute_z_with_tier(
        today,
        columns=all_scored_columns(),
        per_ticker=bundle.per_ticker,
        cross=bundle.cross,
    )
    for col in all_scored_columns():
        tier = int(z_df[f"{col}_tier"].iloc[0])
        assert tier == TIER_FULL, f"Expected Tier 1 for {col}, got {tier}"


def test_tier_shrunk_with_10d_history():
    """Ticker with 10 days of history → Tier 2 (shrunk), needs cohort for pooled MAD."""
    hist = _synth_history(days=10, tickers=["AAPL"], value=1.0, noise=0.05)
    # Need a cohort of at least ZSCORE_MIN_COHORT_SIZE (10) tickers today so cross-sectional
    # stats are available to provide the pooled MAD for shrinkage.
    cohort = ["AAPL"] + [f"T{i}" for i in range(10)]
    today = _synth_today(cohort, value=1.5)
    bundle = build_z_stats_bundle(hist, today)

    z_df = compute_z_with_tier(
        today,
        columns=all_scored_columns(),
        per_ticker=bundle.per_ticker,
        cross=bundle.cross,
    )
    aapl_idx = today.index[today["ticker"] == "AAPL"][0]
    for col in all_scored_columns():
        tier = int(z_df[f"{col}_tier"].iloc[aapl_idx])
        assert tier == TIER_SHRUNK, f"Expected Tier 2 for {col}, got {tier}"


def test_tier_peer_with_2d_history():
    """Ticker with 2 days (< ZSCORE_MIN_N_SHRUNK=5) falls to Tier 3 peer baseline."""
    hist = _synth_history(days=2, tickers=["AAPL"], value=1.0, noise=0.05)
    cohort = ["AAPL"] + [f"T{i}" for i in range(10)]
    today = _synth_today(cohort, value=1.5)
    bundle = build_z_stats_bundle(hist, today)

    z_df = compute_z_with_tier(
        today,
        columns=all_scored_columns(),
        per_ticker=bundle.per_ticker,
        cross=bundle.cross,
    )
    aapl_idx = today.index[today["ticker"] == "AAPL"][0]
    for col in all_scored_columns():
        tier = int(z_df[f"{col}_tier"].iloc[aapl_idx])
        assert tier == TIER_PEER, f"Expected Tier 3 for {col}, got {tier}"


def test_tier_abs_when_cohort_too_small():
    """No history + cohort < ZSCORE_MIN_COHORT_SIZE → Tier 4 absolute fallback."""
    today = _synth_today(["AAPL", "NVDA"], value=1.0)  # only 2 tickers
    # No history at all
    empty_hist = pd.DataFrame(columns=["ticker", "date"] + all_scored_columns())
    bundle = build_z_stats_bundle(empty_hist, today)

    z_df = compute_z_with_tier(
        today,
        columns=all_scored_columns(),
        per_ticker=bundle.per_ticker,
        cross=bundle.cross,
    )
    for col in all_scored_columns():
        for i in range(len(today)):
            tier = int(z_df[f"{col}_tier"].iloc[i])
            assert tier == TIER_ABS
            assert pd.isna(z_df[f"{col}_z"].iloc[i])


def test_per_ticker_stats_robust_to_outlier():
    """One extreme day shouldn't poison median/MAD (that's why we use robust stats)."""
    hist = _synth_history(days=30, tickers=["AAPL"], value=1.0, noise=0.05, seed=1)
    # Inject one gamma-squeeze outlier
    col = "bullish_flow_intensity"
    outlier_idx = hist.index[hist["ticker"] == "AAPL"][0]
    hist.loc[outlier_idx, col] = 100.0

    stats = per_ticker_stats(hist, [col])
    aapl = stats[col]["AAPL"]
    # Median/MAD should be largely unaffected (still ~1.0 / ~0.03)
    assert 0.9 < aapl.median < 1.1, f"median was {aapl.median}"
    assert aapl.mad < 0.2, f"MAD was {aapl.mad} (outlier leaked)"


# ---------------------------------------------------------------------------
# Logistic mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "z,expected",
    [
        (0.0, 0.50),
        (2.0, 0.88),
        (3.0, 0.95),
        (-2.0, 0.12),
        (-3.0, 0.047),
    ],
)
def test_logistic_to_unit(z, expected):
    result = logistic_to_unit(z)
    assert abs(result - expected) < 0.01, f"z={z} → {result}, expected {expected}"


def test_logistic_preserves_nan():
    s = pd.Series([1.0, np.nan, -1.0])
    out = logistic_to_unit(s)
    assert pd.isna(out.iloc[1])
    assert 0.7 < out.iloc[0] < 0.8
    assert 0.2 < out.iloc[2] < 0.3


# ---------------------------------------------------------------------------
# Parity / integration tests
# ---------------------------------------------------------------------------

def _base_normalized_flow() -> pd.DataFrame:
    """Synthetic normalized flow DataFrame that exercises the full pipeline."""
    n = 20
    np.random.seed(42)
    return pd.DataFrame({
        "ticker": ["AAPL"] * 10 + ["NVDA"] * 10,
        "premium": np.random.uniform(600_000, 3_000_000, n),
        "premium_raw": np.random.uniform(600_000, 3_000_000, n),
        "dte": np.random.uniform(35, 90, n),
        "option_type": ["CALL"] * n,
        "execution_side": ["ASK"] * n,
        "direction": ["LONG"] * n,
        "volume": np.random.randint(100, 2000, n),
        "open_interest": np.random.randint(100, 2000, n),
        "is_sweep": np.random.choice([True, False], n),
        "event_ts": pd.Timestamp.utcnow(),
        "marketcap": 2e12,
        "direction_confidence": 1.0,
    })


def test_z_path_parity_when_no_history():
    """build_flow_feature_table(..., z_stats=bundle_with_empty_hist) must match
    the legacy absolute-threshold path. Everything should fall through to
    Tier 4 and use `_clip_scale`."""
    norm = _base_normalized_flow()
    legacy = build_flow_feature_table(norm, min_premium=500_000)

    # Build bundle with empty history + degenerate today cohort (2 tickers, below
    # ZSCORE_MIN_COHORT_SIZE=10). Every row should be Tier 4.
    empty_hist = pd.DataFrame(columns=["ticker", "date"] + all_scored_columns())
    bundle = ZStatsBundle(per_ticker={c: {} for c in all_scored_columns()}, cross={c: {} for c in all_scored_columns()})

    z_path = build_flow_feature_table(norm, min_premium=500_000, z_stats=bundle)

    # Bullish/bearish scores must match exactly
    pd.testing.assert_series_equal(
        legacy["bullish_score"].reset_index(drop=True),
        z_path["bullish_score"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        legacy["bearish_score"].reset_index(drop=True),
        z_path["bearish_score"].reset_index(drop=True),
        check_names=False,
    )

    # All summary tiers must be TIER_ABS
    assert (z_path["bullish_zscore_tier"] == TIER_ABS).all()
    assert (z_path["bearish_zscore_tier"] == TIER_ABS).all()


def test_rescore_with_z_attaches_tier_columns():
    """Public helper: run against a real agg, confirm it adds tier columns
    and preserves legacy scores under bullish_score_abs / bearish_score_abs."""
    norm = _base_normalized_flow()
    legacy = build_flow_feature_table(norm, min_premium=500_000)
    legacy_bull = legacy["bullish_score"].copy()

    # Fake some history so at least some tickers get Tier 1
    hist = _synth_history(days=30, tickers=["AAPL", "NVDA"], value=0.5)

    out = rescore_with_z(legacy.copy(), hist)

    assert "bullish_score_abs" in out.columns
    assert "bearish_score_abs" in out.columns
    assert "bullish_zscore_tier" in out.columns
    assert "bearish_zscore_tier" in out.columns

    # All per-component tier columns
    for comp in ("flow_intensity", "premium_per_trade", "vol_oi", "repeat", "sweep", "dte", "breadth"):
        assert f"bullish_{comp}_tier" in out.columns
        assert f"bearish_{comp}_tier" in out.columns

    # Legacy preserved
    pd.testing.assert_series_equal(
        out["bullish_score_abs"].reset_index(drop=True),
        legacy_bull.reset_index(drop=True),
        check_names=False,
    )


def test_cross_sectional_stats_respects_min_cohort():
    """Cohort below ZSCORE_MIN_COHORT_SIZE → no __all__ bucket populated."""
    small = _synth_today(["AAPL", "NVDA"])  # only 2 tickers
    cs = cross_sectional_stats(small, all_scored_columns())
    for col in all_scored_columns():
        assert "__all__" not in cs[col], f"{col} should not have __all__ with only 2 tickers"


def test_cross_sectional_stats_with_healthy_cohort():
    """Cohort ≥ ZSCORE_MIN_COHORT_SIZE → __all__ bucket populated with (median, mad)."""
    big = _synth_today([f"T{i}" for i in range(15)])
    cs = cross_sectional_stats(big, all_scored_columns())
    for col in all_scored_columns():
        assert "__all__" in cs[col]
        med, mad = cs[col]["__all__"]
        assert np.isfinite(med)
        assert mad >= 0


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

def _run_standalone() -> int:
    """Run all `test_*` functions in this module and report pass/fail."""
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
        param_info = getattr(fn, "__parametrize__", None)
        if param_info is not None:
            pname, values = param_info
            for val in values:
                label = f"{name}[{val!r}]"
                args = val if isinstance(val, tuple) else (val,)
                try:
                    fn(*args)
                    passed += 1
                    print(f"  PASS  {label}")
                except Exception as e:
                    failed.append((label, traceback.format_exc()))
                    print(f"  FAIL  {label}: {e}")
            continue
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        for name, tb in failed:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())

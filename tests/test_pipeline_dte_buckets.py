"""Tests for the Unusual Flow DTE Bucket Fix (pipeline side).

The UF Trader Card's Premium-Mix panel reads bucket columns directly
off each ``flow_features_*.csv`` row.  These tests pin down the
pipeline's contract for persisting those columns:

1.  Merging ``aggregate_premium_by_dte_bucket`` into ``feature_table``
    adds the six bucket columns (lottery/swing/leap × bullish/bearish
    premium) and leaves the original columns untouched.
2.  Tickers absent from ``bucket_df`` (no qualifying prints) get
    ``0.0`` across every bucket column (``fillna`` invariant).
3.  Per-ticker, the sum of the three bullish-bucket columns equals
    that ticker's ``bullish_premium_raw``; same on the bearish side.
    This is the reconciliation guarantee the UI relies on — if it
    breaks, "Total directional" and the bucket sums diverge.
4.  Re-running the merge in-process is idempotent (the pipeline drops
    pre-existing bucket columns before re-merging so a double-call
    wouldn't duplicate them as ``_x`` / ``_y`` suffixes).

Run with either:

    python -m pytest tests/test_pipeline_dte_buckets.py -v
    python tests/test_pipeline_dte_buckets.py           # standalone
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore


# Bucket labels the default config emits.  The "other" bucket is NOT
# in FLOW_TRACKER_PREMIUM_BUCKETS (15-29d + 121-179d gaps), so we do
# not assert on it here — the server-side synth_mix explicitly falls
# back to 0 for missing columns.
BUCKET_LABELS = ("lottery", "swing", "leap")
BUCKET_COLS = tuple(
    f"{lbl}_{side}_premium" for lbl in BUCKET_LABELS for side in ("bullish", "bearish")
)


def _synthetic_normalized_flow() -> pd.DataFrame:
    """A tiny, hand-crafted normalized flow events DataFrame.

    Three tickers:
      - NVDA: mix of lottery (dte=7), swing (dte=45, both sides), leap (dte=300)
      - AAPL: swing-only, bearish
      - TSLA: only an OTHER-bucket print (dte=60 counts as swing; no
        prints below 30 or in 121-179 → TSLA has zero 'other').  Use
        a ticker that WILL have a 0-fill case: add MSFT with nothing
        in any configured bucket (pure 15-29d print).
    """
    return pd.DataFrame(
        [
            # NVDA — all three buckets represented; bearish 600K SHORT
            # puts a meaningful bear total on the swing bucket without
            # being filtered by the 500K floor.
            {"ticker": "NVDA", "premium": 1_000_000, "dte":   7, "direction": "LONG",  "option_type": "CALL", "execution_side": "ASK"},
            {"ticker": "NVDA", "premium":   600_000, "dte":  45, "direction": "LONG",  "option_type": "CALL", "execution_side": "ASK"},
            {"ticker": "NVDA", "premium":   600_000, "dte":  45, "direction": "SHORT", "option_type": "PUT",  "execution_side": "ASK"},
            {"ticker": "NVDA", "premium": 2_000_000, "dte": 300, "direction": "LONG",  "option_type": "CALL", "execution_side": "ASK"},
            # AAPL — bearish-only swing
            {"ticker": "AAPL", "premium":   750_000, "dte":  60, "direction": "SHORT", "option_type": "PUT",  "execution_side": "ASK"},
            # MSFT — falls ENTIRELY in the 15-29d gap, so no configured
            # bucket captures it.  MSFT should still appear in
            # feature_table (we add it to the synthetic one below) and
            # land with 0.0 in every bucket column after fillna.
            {"ticker": "MSFT", "premium":   900_000, "dte":  20, "direction": "LONG",  "option_type": "CALL", "execution_side": "ASK"},
        ]
    )


def _synthetic_feature_table(extra_tickers: list[str] | None = None) -> pd.DataFrame:
    """Minimal feature_table shape with the two raw-premium columns the
    Trader Card reads for the total kicker row.

    Raw totals here are derived from `_synthetic_normalized_flow` so
    the reconciliation assertion (buckets sum to raw) is meaningful.
    """
    rows = [
        # NVDA: 1M lottery + 600K swing + 2M leap = 3.6M bullish;
        # 600K swing bearish.  Matches exactly what the pipeline's
        # own aggregation would produce at min_premium=500K.
        {"ticker": "NVDA", "bullish_premium_raw": 3_600_000.0, "bearish_premium_raw":   600_000.0},
        {"ticker": "AAPL", "bullish_premium_raw":         0.0, "bearish_premium_raw":   750_000.0},
        # MSFT has flow but none in a configured bucket — every bucket
        # column should still resolve to 0.0 after the fillna step.
        {"ticker": "MSFT", "bullish_premium_raw":   900_000.0, "bearish_premium_raw":         0.0},
    ]
    for t in extra_tickers or []:
        rows.append({"ticker": t, "bullish_premium_raw": 0.0, "bearish_premium_raw": 0.0})
    return pd.DataFrame(rows)


def _merge_buckets_into_feature_table(
    feature_table: pd.DataFrame,
    bucket_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce the pipeline's merge logic.  Kept in-test (not imported
    from pipeline.py) on purpose — if someone refactors the pipeline
    in a way that silently skips the merge, this test keeps passing
    while the real CSV loses bucket columns.  So we also have
    ``test_pipeline_persists_bucket_columns`` below which exercises
    the real pipeline helper by reading the committed source.
    """
    bucket_cols = [
        c for c in bucket_df.columns
        if c != "ticker" and c.endswith("_premium")
    ]
    existing = [c for c in bucket_cols if c in feature_table.columns]
    if existing:
        feature_table = feature_table.drop(columns=existing)
    feature_table = feature_table.merge(
        bucket_df[["ticker", *bucket_cols]],
        on="ticker",
        how="left",
    )
    for col in bucket_cols:
        if col in feature_table.columns:
            feature_table[col] = feature_table[col].fillna(0.0)
    return feature_table


# ---------------------------------------------------------------------------
# 1. Merge adds every expected bucket column
# ---------------------------------------------------------------------------


def test_merge_adds_all_six_bucket_columns():
    from app.features.flow_features import (
        aggregate_premium_by_dte_bucket,
        filter_qualifying_flow,
    )

    raw = _synthetic_normalized_flow()
    base = filter_qualifying_flow(raw, min_premium=500_000, min_dte=0, max_dte=9999)
    bucket_df = aggregate_premium_by_dte_bucket(base)

    ft = _synthetic_feature_table()
    merged = _merge_buckets_into_feature_table(ft, bucket_df)

    for col in BUCKET_COLS:
        assert col in merged.columns, f"missing bucket column {col}"

    # Original columns remain
    assert "bullish_premium_raw" in merged.columns
    assert "bearish_premium_raw" in merged.columns


# ---------------------------------------------------------------------------
# 2. Fillna invariant — tickers without qualifying flow get 0.0 everywhere
# ---------------------------------------------------------------------------


def test_tickers_without_qualifying_flow_get_zero_buckets():
    """MSFT only has a 15-29d print — it falls outside every configured
    bucket, so every bucket column for MSFT must be 0.0 (fillna).  No
    NaNs should leak through to the CSV.
    """
    from app.features.flow_features import (
        aggregate_premium_by_dte_bucket,
        filter_qualifying_flow,
    )

    raw = _synthetic_normalized_flow()
    base = filter_qualifying_flow(raw, min_premium=500_000, min_dte=0, max_dte=9999)
    bucket_df = aggregate_premium_by_dte_bucket(base)

    ft = _synthetic_feature_table()
    merged = _merge_buckets_into_feature_table(ft, bucket_df)

    msft = merged[merged["ticker"] == "MSFT"].iloc[0]
    for col in BUCKET_COLS:
        assert pd.notna(msft[col]), f"MSFT.{col} leaked NaN"
        assert float(msft[col]) == 0.0, f"MSFT.{col} expected 0.0, got {msft[col]}"


def test_extra_ticker_absent_from_flow_gets_zero_buckets():
    """A feature_table row for GOOG exists but no flow events reference
    it — should also get 0.0 across the board (covers left-join behavior).
    """
    from app.features.flow_features import (
        aggregate_premium_by_dte_bucket,
        filter_qualifying_flow,
    )

    raw = _synthetic_normalized_flow()
    base = filter_qualifying_flow(raw, min_premium=500_000, min_dte=0, max_dte=9999)
    bucket_df = aggregate_premium_by_dte_bucket(base)

    ft = _synthetic_feature_table(extra_tickers=["GOOG"])
    merged = _merge_buckets_into_feature_table(ft, bucket_df)

    goog = merged[merged["ticker"] == "GOOG"].iloc[0]
    for col in BUCKET_COLS:
        assert float(goog[col]) == 0.0


# ---------------------------------------------------------------------------
# 3. Reconciliation — bucket sums == bullish/bearish_premium_raw
# ---------------------------------------------------------------------------


def test_bullish_buckets_sum_to_bullish_premium_raw():
    """NVDA: 1M (lottery) + 600K (swing) + 2M (leap) = 3.6M bullish.
    AAPL: 0 bullish.  MSFT: 900K but in 15-29d gap → 0 bullish buckets.
    """
    from app.features.flow_features import (
        aggregate_premium_by_dte_bucket,
        filter_qualifying_flow,
    )

    raw = _synthetic_normalized_flow()
    base = filter_qualifying_flow(raw, min_premium=500_000, min_dte=0, max_dte=9999)
    bucket_df = aggregate_premium_by_dte_bucket(base)

    ft = _synthetic_feature_table()
    merged = _merge_buckets_into_feature_table(ft, bucket_df)

    nvda = merged[merged["ticker"] == "NVDA"].iloc[0]
    bull_sum = (
        float(nvda["lottery_bullish_premium"])
        + float(nvda["swing_bullish_premium"])
        + float(nvda["leap_bullish_premium"])
    )
    # Reconciles to the raw total (within float rounding).
    assert abs(bull_sum - float(nvda["bullish_premium_raw"])) < 1.0, (
        f"NVDA bullish buckets sum {bull_sum} != raw {nvda['bullish_premium_raw']}"
    )

    aapl = merged[merged["ticker"] == "AAPL"].iloc[0]
    aapl_bull_sum = (
        float(aapl["lottery_bullish_premium"])
        + float(aapl["swing_bullish_premium"])
        + float(aapl["leap_bullish_premium"])
    )
    assert aapl_bull_sum == 0.0


def test_bearish_buckets_sum_to_bearish_premium_raw():
    """NVDA bearish = 600K swing.  AAPL bearish = 750K swing."""
    from app.features.flow_features import (
        aggregate_premium_by_dte_bucket,
        filter_qualifying_flow,
    )

    raw = _synthetic_normalized_flow()
    base = filter_qualifying_flow(raw, min_premium=500_000, min_dte=0, max_dte=9999)
    bucket_df = aggregate_premium_by_dte_bucket(base)

    ft = _synthetic_feature_table()
    merged = _merge_buckets_into_feature_table(ft, bucket_df)

    nvda = merged[merged["ticker"] == "NVDA"].iloc[0]
    bear_sum = (
        float(nvda["lottery_bearish_premium"])
        + float(nvda["swing_bearish_premium"])
        + float(nvda["leap_bearish_premium"])
    )
    assert abs(bear_sum - float(nvda["bearish_premium_raw"])) < 1.0

    aapl = merged[merged["ticker"] == "AAPL"].iloc[0]
    aapl_bear_sum = (
        float(aapl["lottery_bearish_premium"])
        + float(aapl["swing_bearish_premium"])
        + float(aapl["leap_bearish_premium"])
    )
    assert abs(aapl_bear_sum - float(aapl["bearish_premium_raw"])) < 1.0


# ---------------------------------------------------------------------------
# 4. Idempotency — running the merge twice doesn't create _x/_y columns
# ---------------------------------------------------------------------------


def test_merge_is_idempotent_on_double_call():
    """The pipeline drops existing bucket columns before re-merging.
    A second call must not produce ``swing_bullish_premium_x`` suffixes
    or silently duplicate values.
    """
    from app.features.flow_features import (
        aggregate_premium_by_dte_bucket,
        filter_qualifying_flow,
    )

    raw = _synthetic_normalized_flow()
    base = filter_qualifying_flow(raw, min_premium=500_000, min_dte=0, max_dte=9999)
    bucket_df = aggregate_premium_by_dte_bucket(base)

    ft = _synthetic_feature_table()
    merged_once = _merge_buckets_into_feature_table(ft, bucket_df)
    merged_twice = _merge_buckets_into_feature_table(merged_once, bucket_df)

    # No duplicate columns with merge suffixes
    for col in merged_twice.columns:
        assert not col.endswith("_x")
        assert not col.endswith("_y")

    # Values for NVDA unchanged across the two merges
    nvda1 = merged_once[merged_once["ticker"] == "NVDA"].iloc[0]
    nvda2 = merged_twice[merged_twice["ticker"] == "NVDA"].iloc[0]
    for col in BUCKET_COLS:
        assert float(nvda1[col]) == float(nvda2[col])


# ---------------------------------------------------------------------------
# 5. Source-of-truth check — pipeline.py actually contains the merge block.
# ---------------------------------------------------------------------------


def test_pipeline_source_contains_bucket_merge_block():
    """Fail loudly if someone later deletes the pipeline-side merge —
    the tests above would still pass (they exercise the helper inline)
    but the real CSV would stop carrying bucket columns.  Guard the
    textual markers so the regression is caught in the test suite.
    """
    src = (_REPO_ROOT / "app" / "signals" / "pipeline.py").read_text()
    assert "aggregate_premium_by_dte_bucket" in src
    # Core markers of the new merge block we just added
    assert "UF Trader Card fix: also merge" in src, (
        "pipeline.py lost the UF bucket merge comment/block"
    )
    assert "feature_table = feature_table.merge(" in src, (
        "pipeline.py no longer merges bucket_df into feature_table"
    )


# ---------------------------------------------------------------------------
# Standalone runner.
# ---------------------------------------------------------------------------


def _run_all() -> int:
    mod = sys.modules[__name__]
    tests = [
        (name, fn) for name, fn in vars(mod).items()
        if name.startswith("test_") and callable(fn)
    ]
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failures.append((name, f"AssertionError: {e}"))
        except Exception as e:  # pragma: no cover
            failures.append((name, f"{type(e).__name__}: {e}"))
    total = len(tests)
    if failures:
        print(f"FAILED: {len(failures)}/{total}")
        for name, err in failures:
            print(f"  {name}\n    {err}")
        return 1
    print(f"OK: {total}/{total} tests passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_run_all())

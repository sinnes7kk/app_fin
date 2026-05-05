"""Unit tests for app/signals/hold_config.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.signals.hold_config import (
    CANONICAL_BUCKETS,
    normalize_bucket,
    resolve_earnings_window_days,
    resolve_hold_config,
    resolve_trail_config,
)


def test_canonical_buckets_complete():
    assert set(CANONICAL_BUCKETS) == {"lottery", "swing", "position", "leap", "unknown"}
    print("  PASS: test_canonical_buckets_complete")


def test_normalize_aliases():
    assert normalize_bucket("0-7") == "lottery"
    assert normalize_bucket("8-30") == "swing"
    assert normalize_bucket("31-90") == "position"
    assert normalize_bucket("91+") == "leap"
    assert normalize_bucket("LEAPS") == "leap"
    assert normalize_bucket("LOTTERY") == "lottery"
    print("  PASS: test_normalize_aliases")


def test_normalize_unknown_fallbacks():
    assert normalize_bucket(None) == "unknown"
    assert normalize_bucket("") == "unknown"
    assert normalize_bucket("nan") == "unknown"
    assert normalize_bucket("XYZ") == "unknown"
    assert normalize_bucket("none") == "unknown"
    print("  PASS: test_normalize_unknown_fallbacks")


def test_resolve_hold_config_returns_tuple_of_int_and_float():
    for b in ("lottery", "swing", "position", "leap", "unknown"):
        mh, tsr = resolve_hold_config(b)
        assert isinstance(mh, int) and mh > 0, f"max_hold for {b}: {mh}"
        assert isinstance(tsr, float) and tsr >= 0, f"time_stop for {b}: {tsr}"
    # Unknown alias falls back without error
    mh, tsr = resolve_hold_config("foobar")
    assert mh > 0 and tsr >= 0
    print("  PASS: test_resolve_hold_config_returns_tuple_of_int_and_float")


def test_resolve_hold_config_per_bucket_distinct():
    # Lottery should have shortest hold; leap longest.
    lott_mh, _ = resolve_hold_config("lottery")
    leap_mh, _ = resolve_hold_config("leap")
    pos_mh, _ = resolve_hold_config("position")
    assert lott_mh < pos_mh, f"lottery {lott_mh} should be < position {pos_mh}"
    assert pos_mh <= leap_mh, f"position {pos_mh} should be ≤ leap {leap_mh}"
    print("  PASS: test_resolve_hold_config_per_bucket_distinct")


def test_resolve_trail_config_per_bucket_distinct():
    # All > 1.0 (no negative or zero trail).
    for b in CANONICAL_BUCKETS:
        t = resolve_trail_config(b)
        assert isinstance(t, float) and t > 1.0, f"{b}: {t}"
    # Lottery should have tightest trail.
    lott = resolve_trail_config("lottery")
    pos = resolve_trail_config("position")
    assert lott <= pos, f"lottery {lott} should be tighter than position {pos}"
    print("  PASS: test_resolve_trail_config_per_bucket_distinct")


def test_resolve_earnings_window_returns_int():
    n = resolve_earnings_window_days()
    assert isinstance(n, int) and n >= 1
    print("  PASS: test_resolve_earnings_window_returns_int")


def test_aliased_bucket_resolves_same_as_canonical():
    # Pass alias and canonical, expect the same config.
    assert resolve_hold_config("0-7") == resolve_hold_config("lottery")
    assert resolve_hold_config("8-30") == resolve_hold_config("swing")
    assert resolve_hold_config("31-90") == resolve_hold_config("position")
    assert resolve_hold_config("91+") == resolve_hold_config("leap")
    print("  PASS: test_aliased_bucket_resolves_same_as_canonical")


def main():
    tests = [
        test_canonical_buckets_complete,
        test_normalize_aliases,
        test_normalize_unknown_fallbacks,
        test_resolve_hold_config_returns_tuple_of_int_and_float,
        test_resolve_hold_config_per_bucket_distinct,
        test_resolve_trail_config_per_bucket_distinct,
        test_resolve_earnings_window_returns_int,
        test_aliased_bucket_resolves_same_as_canonical,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

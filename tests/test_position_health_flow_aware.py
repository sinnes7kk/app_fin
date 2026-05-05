"""Unit tests for the Stage F.1 flow-aware position-health components and
the Stage F.3 ``flow_decay_factor`` propagation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.signals.position_health import (
    _flow_decay_factor,
    _grade_decay_penalty,
    _sector_heat_reversal_penalty,
    _unusual_flow_flip_penalty,
)


def test_grade_decay_no_change_returns_zero():
    pos = {"conviction_grade": "A", "current_grade": "A", "direction": "LONG"}
    assert _grade_decay_penalty(pos) == 0.0
    print("  PASS: test_grade_decay_no_change_returns_zero")


def test_grade_decay_one_tier_drop_returns_half():
    pos = {"conviction_grade": "A", "current_grade": "A-", "direction": "LONG"}
    assert _grade_decay_penalty(pos) == 0.5
    print("  PASS: test_grade_decay_one_tier_drop_returns_half")


def test_grade_decay_three_tier_drop_caps_at_1_5():
    pos = {"conviction_grade": "A+", "current_grade": "B", "direction": "LONG"}
    p = _grade_decay_penalty(pos)
    # A+(6) → B(2) = 4 tiers; 0.5 * 4 = 2.0 capped at 1.5
    assert p == 1.5, f"got {p}"
    print("  PASS: test_grade_decay_three_tier_drop_caps_at_1_5")


def test_grade_decay_missing_data_returns_zero():
    assert _grade_decay_penalty({"current_grade": "A"}) == 0.0
    assert _grade_decay_penalty({"conviction_grade": "A"}) == 0.0
    assert _grade_decay_penalty({}) == 0.0
    print("  PASS: test_grade_decay_missing_data_returns_zero")


def test_sector_heat_reversal_long_against_bearish_basket():
    pos = {"direction": "LONG", "current_sector_heat": -5.0}
    p = _sector_heat_reversal_penalty(pos)
    assert p == 1.0, f"got {p}"  # 5/5 = 1.0
    print("  PASS: test_sector_heat_reversal_long_against_bearish_basket")


def test_sector_heat_reversal_short_against_bullish_basket():
    pos = {"direction": "SHORT", "current_sector_heat": 4.0}
    p = _sector_heat_reversal_penalty(pos)
    assert p == 0.8, f"got {p}"  # 4/5 = 0.8
    print("  PASS: test_sector_heat_reversal_short_against_bullish_basket")


def test_sector_heat_reversal_aligned_returns_zero():
    pos = {"direction": "LONG", "current_sector_heat": 5.0}  # bullish basket, long → no penalty
    assert _sector_heat_reversal_penalty(pos) == 0.0
    pos = {"direction": "SHORT", "current_sector_heat": -5.0}
    assert _sector_heat_reversal_penalty(pos) == 0.0
    print("  PASS: test_sector_heat_reversal_aligned_returns_zero")


def test_sector_heat_caps_at_1_5():
    pos = {"direction": "LONG", "current_sector_heat": -10.0}
    assert _sector_heat_reversal_penalty(pos) == 1.5
    print("  PASS: test_sector_heat_caps_at_1_5")


def test_unusual_flow_flip_long_now_bearish():
    pos = {"direction": "LONG", "current_unusual_flow_dir": "BEARISH"}
    assert _unusual_flow_flip_penalty(pos) == 1.0
    print("  PASS: test_unusual_flow_flip_long_now_bearish")


def test_unusual_flow_flip_short_now_bullish():
    pos = {"direction": "SHORT", "current_unusual_flow_dir": "BULLISH"}
    assert _unusual_flow_flip_penalty(pos) == 1.0
    print("  PASS: test_unusual_flow_flip_short_now_bullish")


def test_unusual_flow_flip_aligned_returns_zero():
    pos = {"direction": "LONG", "current_unusual_flow_dir": "BULLISH"}
    assert _unusual_flow_flip_penalty(pos) == 0.0
    print("  PASS: test_unusual_flow_flip_aligned_returns_zero")


def test_flow_decay_factor_no_decay_returns_one():
    pos = {
        "conviction_grade": "A", "current_grade": "A",
        "current_sector_heat": 5.0,
        "current_unusual_flow_dir": "BULLISH",
        "direction": "LONG",
    }
    f = _flow_decay_factor(pos)
    assert f == 1.0, f"got {f}"
    print("  PASS: test_flow_decay_factor_no_decay_returns_one")


def test_flow_decay_factor_max_decay_returns_low():
    pos = {
        "conviction_grade": "A+", "current_grade": "B",  # 4 tier drop → cap 1.5
        "current_sector_heat": -10.0,                    # → cap 1.5
        "current_unusual_flow_dir": "BEARISH",           # → 1.0
        "direction": "LONG",
    }
    f = _flow_decay_factor(pos)
    # total_pen = 1.5 + 1.5 + 1.0 = 4.0 → factor = 1 - 4/4 = 0.0
    assert f == 0.0, f"got {f}"
    print("  PASS: test_flow_decay_factor_max_decay_returns_low")


def test_flow_decay_factor_partial_decay():
    pos = {
        "conviction_grade": "A", "current_grade": "A-",  # 0.5
        "current_sector_heat": 0.0,
        "current_unusual_flow_dir": None,
        "direction": "LONG",
    }
    f = _flow_decay_factor(pos)
    # total_pen = 0.5 → factor = 1 - 0.5/4 = 0.875
    assert abs(f - 0.875) < 1e-6, f"got {f}"
    print("  PASS: test_flow_decay_factor_partial_decay")


def test_compute_trailing_stops_signature_accepts_flow_decay_factor():
    # Smoke check that the signature accepts the new kwarg.
    from inspect import signature
    from app.signals.trade_plan import compute_trailing_stops
    sig = signature(compute_trailing_stops)
    assert "flow_decay_factor" in sig.parameters, "kwarg not exposed"
    assert sig.parameters["flow_decay_factor"].default == 1.0
    print("  PASS: test_compute_trailing_stops_signature_accepts_flow_decay_factor")


def main():
    tests = [
        test_grade_decay_no_change_returns_zero,
        test_grade_decay_one_tier_drop_returns_half,
        test_grade_decay_three_tier_drop_caps_at_1_5,
        test_grade_decay_missing_data_returns_zero,
        test_sector_heat_reversal_long_against_bearish_basket,
        test_sector_heat_reversal_short_against_bullish_basket,
        test_sector_heat_reversal_aligned_returns_zero,
        test_sector_heat_caps_at_1_5,
        test_unusual_flow_flip_long_now_bearish,
        test_unusual_flow_flip_short_now_bullish,
        test_unusual_flow_flip_aligned_returns_zero,
        test_flow_decay_factor_no_decay_returns_one,
        test_flow_decay_factor_max_decay_returns_low,
        test_flow_decay_factor_partial_decay,
        test_compute_trailing_stops_signature_accepts_flow_decay_factor,
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

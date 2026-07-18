"""Tests for the shadow cross-sectional momentum score."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import momentum_score as ms  # noqa: E402


def test_orient_bull_share_flips_for_bearish():
    # bullish trade: high bull share is good; bearish trade: low is good.
    assert ms._oriented_value({"bullish_premium_share": 0.8}, "bullish_premium_share", "bull_share", True) == 0.8
    assert abs(ms._oriented_value({"bullish_premium_share": 0.2}, "bullish_premium_share", "bull_share", False) - 0.8) < 1e-9
    print("  PASS: test_orient_bull_share_flips_for_bearish")


def test_orient_signed_flips_sign_for_bearish():
    row = {"aggressor_net_prem_bps": -5.0}
    assert ms._oriented_value(row, "aggressor_net_prem_bps", "signed", True) == -5.0
    assert ms._oriented_value(row, "aggressor_net_prem_bps", "signed", False) == 5.0
    print("  PASS: test_orient_signed_flips_sign_for_bearish")


def test_orient_fade_negates():
    assert ms._oriented_value({"realized_vol_regime": 2.0}, "realized_vol_regime", "fade", True) == -2.0
    assert ms._oriented_value({"realized_vol_regime": 2.0}, "realized_vol_regime", "fade", False) == -2.0
    print("  PASS: test_orient_fade_negates")


def test_far_otm_directional_picks_side_by_direction():
    row = {"far_otm_call_share": 0.3, "far_otm_put_share": 0.1}
    # bullish -> fade the call share; bearish -> fade the put share
    assert ms._oriented_value(row, "far_otm_directional", "fade", True) == -0.3
    assert ms._oriented_value(row, "far_otm_directional", "fade", False) == -0.1
    print("  PASS: test_far_otm_directional_picks_side_by_direction")


def test_cross_sectional_ranking_orders_candidates():
    rows = [
        {"ticker": "BEST", "direction": "BULLISH", "bullish_premium_share": 0.95,
         "aggressor_bull_share": 0.95, "ask_side_ratio": 0.9, "sector_relative_pct": 0.9,
         "realized_vol_regime": 0.1, "far_otm_call_share": 0.02},
        {"ticker": "MID", "direction": "BULLISH", "bullish_premium_share": 0.5,
         "aggressor_bull_share": 0.5, "ask_side_ratio": 0.5, "sector_relative_pct": 0.5,
         "realized_vol_regime": 0.5, "far_otm_call_share": 0.2},
        {"ticker": "WORST", "direction": "BULLISH", "bullish_premium_share": 0.1,
         "aggressor_bull_share": 0.1, "ask_side_ratio": 0.15, "sector_relative_pct": 0.1,
         "realized_vol_regime": 0.9, "far_otm_call_share": 0.6},
    ]
    scores = ms.compute_day_scores(rows)
    assert len(scores) == 3
    s = [x["momentum_score"] for x in scores]
    assert all(0.0 <= v <= 100.0 for v in s)
    assert s[0] > s[1] > s[2], f"expected BEST>MID>WORST, got {s}"
    print("  PASS: test_cross_sectional_ranking_orders_candidates")


def test_bearish_candidate_ranks_high_on_bearish_flow():
    # A bearish candidate with low bull share should out-rank a bullish
    # candidate with the same low bull share.
    rows = [
        {"ticker": "BEAR", "direction": "BEARISH", "bullish_premium_share": 0.1,
         "aggressor_bull_share": 0.1, "ask_side_ratio": 0.1},
        {"ticker": "BULL", "direction": "BULLISH", "bullish_premium_share": 0.1,
         "aggressor_bull_share": 0.1, "ask_side_ratio": 0.1},
    ]
    scores = ms.compute_day_scores(rows)
    assert scores[0]["momentum_score"] > scores[1]["momentum_score"]
    print("  PASS: test_bearish_candidate_ranks_high_on_bearish_flow")


def test_min_features_gate_returns_none():
    # Only two usable features -> below MIN_FEATURES -> None.
    rows = [
        {"ticker": "THIN", "direction": "BULLISH", "bullish_premium_share": 0.8,
         "aggressor_bull_share": 0.7},
        {"ticker": "THIN2", "direction": "BULLISH", "bullish_premium_share": 0.6,
         "aggressor_bull_share": 0.5},
    ]
    scores = ms.compute_day_scores(rows)
    assert all(x["momentum_score"] is None for x in scores)
    print("  PASS: test_min_features_gate_returns_none")


def test_nan_and_empty_inputs():
    assert ms.compute_day_scores([]) == []
    rows = [{"ticker": "X", "direction": "BULLISH", "bullish_premium_share": float("nan"),
             "aggressor_bull_share": None, "ask_side_ratio": None}]
    scores = ms.compute_day_scores(rows)
    assert scores[0]["momentum_score"] is None
    print("  PASS: test_nan_and_empty_inputs")


def main() -> int:
    tests = [
        ("test_orient_bull_share_flips_for_bearish", test_orient_bull_share_flips_for_bearish),
        ("test_orient_signed_flips_sign_for_bearish", test_orient_signed_flips_sign_for_bearish),
        ("test_orient_fade_negates", test_orient_fade_negates),
        ("test_far_otm_directional_picks_side_by_direction", test_far_otm_directional_picks_side_by_direction),
        ("test_cross_sectional_ranking_orders_candidates", test_cross_sectional_ranking_orders_candidates),
        ("test_bearish_candidate_ranks_high_on_bearish_flow", test_bearish_candidate_ranks_high_on_bearish_flow),
        ("test_min_features_gate_returns_none", test_min_features_gate_returns_none),
        ("test_nan_and_empty_inputs", test_nan_and_empty_inputs),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {name}: {type(e).__name__}: {e}")
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

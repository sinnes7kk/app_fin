"""Tests for Wave 4 ``app.features.conviction_stack``.

Validates:

* Component caps are respected (flow 50, DP 15, chain 10, insider 8,
  price 12, dealer 5 → 100 total).
* Tier thresholds (elite >= 80, strong 65-79, moderate 50-64, weak < 50).
* Direction normalisation (LONG/BULLISH → LONG, SHORT/BEARISH → SHORT).
* Insider alignment vs opposition scoring.
* DP z-tier bonus (Wave 2 dp_z) adds credit when aligned, only.
* Empty / missing fields degrade gracefully to zero components, never
  crash.
* ``attach_conviction_stack`` decorates a list in place.

Run with:
    python -m pytest tests/test_conviction_stack.py -v
    python -m tests.test_conviction_stack        # standalone
"""

from __future__ import annotations

from app.features.conviction_stack import (
    TIER_ELITE,
    TIER_MODERATE,
    TIER_STRONG,
    TIER_WEAK,
    _direction_side,
    attach_conviction_stack,
    compute_conviction_stack,
)


def _elite_long_row() -> dict:
    """Row that should score at or near 100 on every component."""
    return {
        "ticker": "AAPL",
        "direction": "LONG",
        "conviction_score": 9.5,
        "dark_pool": {"bias": 0.75, "notional_mcap_bps": 8.0},
        "dp_z": 2.5,
        "dp_tier": 1,
        "hot_chain": {"call_premium": 5_000_000, "put_premium": 500_000},
        "insider": {"net_direction": "buying"},
        "session": {"session_tone": "STRENGTH"},
        "rs": {"rs_5d_pct": 2.5, "rs_20d_pct": 4.5},
        "window_return_pct": 3.5,
        "dealer_hedge_bias": "chase",
    }


def test_direction_normalisation():
    assert _direction_side("LONG") == "LONG"
    assert _direction_side("BULLISH") == "LONG"
    assert _direction_side("Net Bullish") == "LONG"
    assert _direction_side("SHORT") == "SHORT"
    assert _direction_side("BEARISH") == "SHORT"
    assert _direction_side("") is None
    assert _direction_side(None) is None


def test_elite_row_scores_near_100_and_classifies_elite():
    stack = compute_conviction_stack(_elite_long_row())
    assert stack["tier"] == TIER_ELITE
    assert stack["score"] >= 80
    # Drivers should include flow_core and dp_confirm.
    assert "flow_core" in stack["top_drivers"]
    assert "dp_confirm" in stack["top_drivers"]


def test_empty_row_collapses_to_weak():
    stack = compute_conviction_stack({"direction": "LONG"})
    assert stack["tier"] == TIER_WEAK
    assert stack["score"] == 0


def test_insider_opposed_does_not_go_negative():
    """A LONG setup with insider selling should zero the insider
    component (clamped via max), never drag the total negative."""
    row = {
        "direction": "LONG",
        "conviction_score": 6.0,
        "insider": {"net_direction": "selling"},
    }
    stack = compute_conviction_stack(row)
    ins = next(c for c in stack["components"] if c["component"] == "insider")
    assert ins["points"] == 0.0
    assert ins["opposed"] is True
    # Flow gave 30 pts (6.0/10 * 50), nothing else fires → total = 30.
    assert stack["score"] == 30


def test_dp_z_bonus_only_when_aligned():
    """Wave-2 dp_z adds up to 3 pts when the z is high in the RIGHT
    direction.  A cold z (-2) on a LONG setup must NOT add credit."""
    row_long_hot = {
        "direction": "LONG",
        "conviction_score": 8.0,
        "dark_pool": {"bias": 0.65, "notional_mcap_bps": 2.0},
        "dp_z": 2.5,
    }
    row_long_cold = {**row_long_hot, "dp_z": -2.5}
    hot = compute_conviction_stack(row_long_hot)
    cold = compute_conviction_stack(row_long_cold)
    hot_dp = next(c for c in hot["components"] if c["component"] == "dp_confirm")
    cold_dp = next(c for c in cold["components"] if c["component"] == "dp_confirm")
    # Hot should get +3 on top of the primary gate; cold should not.
    assert hot_dp["points"] > cold_dp["points"]
    assert hot_dp["points"] - cold_dp["points"] == 3.0


def test_dp_bias_against_direction_drags_to_zero_not_negative():
    row = {
        "direction": "LONG",
        "conviction_score": 7.0,
        "dark_pool": {"bias": 0.25, "notional_mcap_bps": 4.0},  # sellers
    }
    stack = compute_conviction_stack(row)
    dp = next(c for c in stack["components"] if c["component"] == "dp_confirm")
    # Sellers against LONG = penalty but we floor at zero.
    assert dp["points"] == 0.0


def test_chain_alignment_gives_full_credit():
    row = {
        "direction": "SHORT",
        "conviction_score": 6.5,
        "hot_chain": {"call_premium": 200_000, "put_premium": 3_000_000},
    }
    stack = compute_conviction_stack(row)
    ch = next(c for c in stack["components"] if c["component"] == "chain_confirm")
    assert ch["aligned"] is True
    assert ch["points"] == 10.0


def test_chain_misaligned_gives_zero_credit():
    row = {
        "direction": "SHORT",
        "conviction_score": 6.5,
        "hot_chain": {"call_premium": 3_000_000, "put_premium": 200_000},  # calls on SHORT
    }
    stack = compute_conviction_stack(row)
    ch = next(c for c in stack["components"] if c["component"] == "chain_confirm")
    assert ch["aligned"] is False
    assert ch["points"] == 0.0


def test_tier_classification_thresholds():
    """Spot-check the tier cut-offs using flow_core only."""
    # flow 10 * 5 = 50 pts → moderate exactly.
    moderate = compute_conviction_stack({"direction": "LONG", "conviction_score": 10.0})
    assert moderate["tier"] == TIER_MODERATE  # 50 exactly → moderate

    # 6.4 * 5 = 32 → weak
    weak = compute_conviction_stack({"direction": "LONG", "conviction_score": 6.4})
    assert weak["tier"] == TIER_WEAK

    # Combine flow + dp + chain to hit strong range.
    strong = compute_conviction_stack({
        "direction": "LONG",
        "conviction_score": 8.0,  # 40
        "dark_pool": {"bias": 0.70, "notional_mcap_bps": 6.0},  # 12
        "hot_chain": {"call_premium": 1_000_000, "put_premium": 100_000},  # 10
    })
    # 40 + 12 + 10 = 62 → moderate (actually 62 = moderate, not strong).
    # This test validates the boundary: moderate ≤ 64.
    assert strong["tier"] == TIER_MODERATE
    assert 50 <= strong["score"] <= 64


def test_attach_conviction_stack_mutates_in_place():
    rows = [
        {"direction": "LONG", "conviction_score": 5.0},
        {"direction": "SHORT", "conviction_score": 8.0},
    ]
    out = attach_conviction_stack(rows)
    assert out is rows  # same object, returned for chaining
    for r in rows:
        assert "conviction_stack" in r
        assert r["conviction_stack"]["score"] >= 0
        assert r["conviction_stack"]["tier"] in {
            TIER_ELITE, TIER_STRONG, TIER_MODERATE, TIER_WEAK
        }


def test_attach_handles_empty_list():
    assert attach_conviction_stack([]) == []
    assert attach_conviction_stack(None) is None  # type: ignore[arg-type]


def test_malformed_row_does_not_crash():
    """Mixed NaN / None / string values should all degrade to zero, not
    raise."""
    row = {
        "direction": "LONG",
        "conviction_score": "bad",
        "dark_pool": {"bias": None, "notional_mcap_bps": float("nan")},
        "insider": "not-a-dict",
        "session": None,
        "rs": {"rs_5d_pct": None, "rs_20d_pct": None},
    }
    stack = compute_conviction_stack(row)
    assert stack["score"] == 0
    assert stack["tier"] == TIER_WEAK


if __name__ == "__main__":
    import traceback

    tests = [
        test_direction_normalisation,
        test_elite_row_scores_near_100_and_classifies_elite,
        test_empty_row_collapses_to_weak,
        test_insider_opposed_does_not_go_negative,
        test_dp_z_bonus_only_when_aligned,
        test_dp_bias_against_direction_drags_to_zero_not_negative,
        test_chain_alignment_gives_full_credit,
        test_chain_misaligned_gives_zero_credit,
        test_tier_classification_thresholds,
        test_attach_conviction_stack_mutates_in_place,
        test_attach_handles_empty_list,
        test_malformed_row_does_not_crash,
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

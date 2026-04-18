"""Tests for Wave 7 ``app.features.trade_structure``.

Validates the IV / earnings / conviction / direction / liquidity
matrix that picks the primary trade structure + alternatives + avoid
list + caveats.

We don't assert on exact prose — just on the structural contract:

* ``primary.structure`` is the right enum for each archetype.
* ``avoid`` / ``caveats`` surface the right risks.
* Malformed / empty rows return ``STAND_ASIDE`` without crashing.

Run with:
    python -m pytest tests/test_trade_structure.py -v
    python tests/test_trade_structure.py    # standalone
"""

from __future__ import annotations

import traceback

from app.features.trade_structure import (
    STRUCTURE_LABELS,
    TONE_WARNING,
    recommend_structure,
    attach_trade_structure,
)


# ---------------------------------------------------------------------------
# Archetype row factories
# ---------------------------------------------------------------------------
def _elite_long_cheap_iv_row() -> dict:
    return {
        "ticker": "AAPL",
        "direction": "LONG",
        "conviction_stack": {"score": 85, "tier": "elite", "tier_label": "Stack A"},
        "latest_iv_rank": 18,
        "liquidity": {"adv_dollar": 2_500_000_000, "liquidity_tier": "DEEP"},
        "earnings": {"days_until_earnings": 30},
        "dealer_hedge_bias": "chase",
        "window_return_pct": 1.8,
    }


def _strong_long_rich_iv_row() -> dict:
    return {
        "ticker": "NVDA",
        "direction": "LONG",
        "conviction_stack": {"score": 72, "tier": "strong", "tier_label": "Stack B"},
        "latest_iv_rank": 78,
        "liquidity": {"adv_dollar": 5_000_000_000, "liquidity_tier": "DEEP"},
        "earnings": {"days_until_earnings": 25},
        "dealer_hedge_bias": "neutral",
    }


def _moderate_long_mid_iv_row() -> dict:
    return {
        "ticker": "MSFT",
        "direction": "LONG",
        "conviction_stack": {"score": 58, "tier": "moderate", "tier_label": "Stack C"},
        "latest_iv_rank": 45,
        "liquidity": {"adv_dollar": 1_500_000_000, "liquidity_tier": "DEEP"},
        "earnings": {"days_until_earnings": 45},
    }


def _long_near_earnings_row() -> dict:
    return {
        "ticker": "META",
        "direction": "LONG",
        "conviction_stack": {"score": 70, "tier": "strong", "tier_label": "Stack B"},
        "latest_iv_rank": 55,
        "liquidity": {"adv_dollar": 3_000_000_000, "liquidity_tier": "DEEP"},
        "earnings": {"days_until_earnings": 3},
    }


def _short_rich_iv_row() -> dict:
    return {
        "ticker": "XYZ",
        "direction": "SHORT",
        "conviction_stack": {"score": 82, "tier": "elite", "tier_label": "Stack A"},
        "latest_iv_rank": 74,
        "liquidity": {"adv_dollar": 500_000_000, "liquidity_tier": "HEALTHY"},
        "earnings": {"days_until_earnings": 40},
        "dealer_hedge_bias": "chase",
    }


def _short_cheap_iv_row() -> dict:
    return {
        "ticker": "ABC",
        "direction": "SHORT",
        "conviction_stack": {"score": 72, "tier": "strong"},
        "latest_iv_rank": 14,
        "liquidity": {"adv_dollar": 250_000_000, "liquidity_tier": "HEALTHY"},
    }


def _weak_conviction_row() -> dict:
    return {
        "ticker": "WEAK",
        "direction": "LONG",
        "conviction_stack": {"score": 38, "tier": "weak"},
        "latest_iv_rank": 40,
        "liquidity": {"adv_dollar": 200_000_000, "liquidity_tier": "HEALTHY"},
    }


def _illiquid_long_row() -> dict:
    return {
        "ticker": "TINY",
        "direction": "LONG",
        "conviction_stack": {"score": 72, "tier": "strong"},
        "latest_iv_rank": 25,
        "liquidity": {"adv_dollar": 5_000_000, "liquidity_tier": "ILLIQUID"},
    }


def _pinning_dealer_row() -> dict:
    return {
        "ticker": "PIN",
        "direction": "LONG",
        "conviction_stack": {"score": 70, "tier": "strong"},
        "latest_iv_rank": 72,
        "liquidity": {"adv_dollar": 1_000_000_000, "liquidity_tier": "DEEP"},
        "dealer_hedge_bias": "suppress",
    }


def _late_chase_long_row() -> dict:
    return {
        "ticker": "LATE",
        "direction": "LONG",
        "conviction_stack": {"score": 68, "tier": "strong"},
        "latest_iv_rank": 35,
        "window_return_pct": 9.0,
        "liquidity": {"adv_dollar": 800_000_000, "liquidity_tier": "DEEP"},
    }


def _neutral_direction_row() -> dict:
    return {
        "ticker": "MIX",
        "direction": "MIXED",
        "conviction_stack": {"score": 55, "tier": "moderate"},
        "latest_iv_rank": 70,
        "liquidity": {"adv_dollar": 1_000_000_000, "liquidity_tier": "DEEP"},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _alt_structures(s: dict) -> list[str]:
    return [a.get("structure") for a in (s.get("alternatives") or [])]


def _avoid_structures(s: dict) -> list[str]:
    return [a.get("structure") for a in (s.get("avoid") or [])]


def _caveats_text(s: dict) -> str:
    return " | ".join(s.get("caveats") or []).lower()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_empty_row_returns_stand_aside():
    s = recommend_structure(None)
    assert s["primary"]["structure"] == "STAND_ASIDE"
    assert s["side"] == "NEUTRAL"
    assert s["alternatives"] == []


def test_elite_long_cheap_iv_recommends_long_call():
    s = recommend_structure(_elite_long_cheap_iv_row())
    assert s["side"] == "LONG"
    assert s["primary"]["structure"] == "LONG_CALL"
    # Elite conviction unlocks risk reversal as an alternative
    assert "RISK_REVERSAL_LONG" in _alt_structures(s)


def test_strong_long_rich_iv_uses_debit_spread():
    s = recommend_structure(_strong_long_rich_iv_row())
    assert s["primary"]["structure"] == "CALL_DEBIT_SPREAD"
    assert "LONG_CALL" in _avoid_structures(s)
    caveats = _caveats_text(s)
    # No earnings caveats for 25d
    assert "earnings in" not in caveats or "25" not in caveats


def test_moderate_long_mid_iv_uses_stock():
    s = recommend_structure(_moderate_long_mid_iv_row())
    assert s["primary"]["structure"] == "STOCK_LONG"
    # Moderate conviction — does NOT unlock risk reversal
    assert "RISK_REVERSAL_LONG" not in _alt_structures(s)


def test_long_near_earnings_forces_defined_risk():
    s = recommend_structure(_long_near_earnings_row())
    assert s["primary"]["structure"] == "CALL_DEBIT_SPREAD"
    avoid = _avoid_structures(s)
    assert "LONG_CALL" in avoid, f"expected LONG_CALL in avoid list near earnings: {avoid}"
    # Earnings caveat should appear
    assert "earnings" in _caveats_text(s)


def test_short_rich_iv_uses_put_debit_spread():
    s = recommend_structure(_short_rich_iv_row())
    assert s["primary"]["structure"] == "PUT_DEBIT_SPREAD"
    # Elite short conviction unlocks the risk reversal alt
    assert "RISK_REVERSAL_SHORT" in _alt_structures(s)


def test_short_cheap_iv_recommends_long_put():
    s = recommend_structure(_short_cheap_iv_row())
    assert s["primary"]["structure"] == "LONG_PUT"


def test_weak_conviction_stands_aside():
    s = recommend_structure(_weak_conviction_row())
    assert s["primary"]["structure"] == "STAND_ASIDE"
    assert s["primary"]["tone"] == TONE_WARNING
    assert any("conviction" in c.lower() or "signal" in c.lower() for c in s["caveats"])


def test_illiquid_long_row_collapses_to_stock():
    s = recommend_structure(_illiquid_long_row())
    assert s["primary"]["structure"] == "STOCK_LONG"
    # Spread structures should be explicitly avoided
    avoid = _avoid_structures(s)
    assert "CALL_DEBIT_SPREAD" in avoid
    assert "PUT_DEBIT_SPREAD" in avoid


def test_pinning_dealer_adds_pinning_caveat():
    s = recommend_structure(_pinning_dealer_row())
    caveats = _caveats_text(s)
    assert "gamma" in caveats or "grind" in caveats or "pin" in caveats, \
        f"expected pinning-regime caveat: {caveats}"


def test_late_chase_surfaces_chase_caveat():
    s = recommend_structure(_late_chase_long_row())
    caveats = _caveats_text(s)
    assert "late chase" in caveats or "size smaller" in caveats, \
        f"expected late-chase caveat: {caveats}"


def test_neutral_direction_stands_aside():
    s = recommend_structure(_neutral_direction_row())
    assert s["side"] == "NEUTRAL"
    assert s["primary"]["structure"] == "STAND_ASIDE"


def test_override_side_forces_direction():
    row = _strong_long_rich_iv_row()
    # Force NEUTRAL even though row says LONG
    s = recommend_structure(row, side="NEUTRAL")
    assert s["side"] == "NEUTRAL"
    assert s["primary"]["structure"] == "STAND_ASIDE"


def test_payload_shape_is_stable():
    s = recommend_structure(_strong_long_rich_iv_row())
    assert set(s.keys()) >= {"side", "primary", "alternatives", "avoid", "caveats"}
    assert isinstance(s["alternatives"], list)
    assert isinstance(s["avoid"], list)
    assert isinstance(s["caveats"], list)
    # Every structure ref maps to a display label
    all_structs = [s["primary"]["structure"]] + _alt_structures(s) + _avoid_structures(s)
    for st in all_structs:
        assert st in STRUCTURE_LABELS, f"missing label for {st}"


def test_attach_trade_structure_mutates_in_place():
    rows = [_elite_long_cheap_iv_row(), _weak_conviction_row()]
    out = attach_trade_structure(rows)
    assert out is rows
    assert rows[0]["trade_structure"]["primary"]["structure"] == "LONG_CALL"
    assert rows[1]["trade_structure"]["primary"]["structure"] == "STAND_ASIDE"


def test_malformed_row_does_not_crash():
    weird = {
        "direction": 42,
        "conviction_stack": "not a dict",
        "latest_iv_rank": "rich",
        "liquidity": [1, 2, 3],
        "earnings": "none",
    }
    s = recommend_structure(weird)
    # Must return at minimum a stand-aside payload
    assert "primary" in s
    assert isinstance(s.get("alternatives"), list)


def test_alternatives_cap():
    s = recommend_structure(_elite_long_cheap_iv_row())
    # Even when many alts apply, list is capped to 4
    assert len(s["alternatives"]) <= 4


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
_TESTS = [
    ("test_empty_row_returns_stand_aside", test_empty_row_returns_stand_aside),
    ("test_elite_long_cheap_iv_recommends_long_call", test_elite_long_cheap_iv_recommends_long_call),
    ("test_strong_long_rich_iv_uses_debit_spread", test_strong_long_rich_iv_uses_debit_spread),
    ("test_moderate_long_mid_iv_uses_stock", test_moderate_long_mid_iv_uses_stock),
    ("test_long_near_earnings_forces_defined_risk", test_long_near_earnings_forces_defined_risk),
    ("test_short_rich_iv_uses_put_debit_spread", test_short_rich_iv_uses_put_debit_spread),
    ("test_short_cheap_iv_recommends_long_put", test_short_cheap_iv_recommends_long_put),
    ("test_weak_conviction_stands_aside", test_weak_conviction_stands_aside),
    ("test_illiquid_long_row_collapses_to_stock", test_illiquid_long_row_collapses_to_stock),
    ("test_pinning_dealer_adds_pinning_caveat", test_pinning_dealer_adds_pinning_caveat),
    ("test_late_chase_surfaces_chase_caveat", test_late_chase_surfaces_chase_caveat),
    ("test_neutral_direction_stands_aside", test_neutral_direction_stands_aside),
    ("test_override_side_forces_direction", test_override_side_forces_direction),
    ("test_payload_shape_is_stable", test_payload_shape_is_stable),
    ("test_attach_trade_structure_mutates_in_place", test_attach_trade_structure_mutates_in_place),
    ("test_malformed_row_does_not_crash", test_malformed_row_does_not_crash),
    ("test_alternatives_cap", test_alternatives_cap),
]


def _main() -> int:
    passed = 0
    failed = 0
    for name, fn in _TESTS:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {name}: {exc}")
            failed += 1
        except Exception:
            print(f"ERROR {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())

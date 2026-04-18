"""Tests for Wave 2 dealer-hedge bias + pin-risk-strike derivations.

Validates that ``_fetch_gex_context`` (the spot-exposures consumer inside
``app.features.options_context``):

  * Classifies ``dealer_hedge_bias`` correctly (suppress / chase / neutral)
    from the sign of ``net_gex``.
  * Computes ``pin_risk_strike`` only when the dominant near-spot strike
    carries ≥ ``PIN_RISK_MIN_CONCENTRATION`` of the absolute gamma in the
    ±``PIN_RISK_BAND_PCT`` band around spot.
  * Returns ``pin_risk_strike=None`` when the gamma is diffuse across the
    near-spot band (no single magnet).
  * Gracefully skips the pin-risk calc when spot is ``None``.

Run with either:
    python -m pytest tests/test_options_context_gamma.py -v
    python -m tests.test_options_context_gamma      # standalone (no pytest)
"""

from __future__ import annotations

import unittest.mock as mock


class _FakeResp:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"data": self._rows}


def _with_rows(rows: list[dict]):
    return mock.patch(
        "app.features.options_context._uw_request",
        return_value=_FakeResp(rows),
    )


def test_positive_net_gex_labels_dealer_as_suppress():
    from app.features.options_context import _fetch_gex_context
    rows = [
        {"price": 100.0, "call_gamma_oi": 500.0, "put_gamma_oi": -100.0},
        {"price": 105.0, "call_gamma_oi": 300.0, "put_gamma_oi": -50.0},
    ]
    with _with_rows(rows):
        ctx = _fetch_gex_context("TEST", spot=100.0)
    assert ctx["gamma_regime"] == "POSITIVE"
    assert ctx["dealer_hedge_bias"] == "suppress"
    assert "pinning" in (ctx["dealer_hedge_label"] or "").lower()


def test_negative_net_gex_labels_dealer_as_chase():
    from app.features.options_context import _fetch_gex_context
    rows = [
        {"price": 100.0, "call_gamma_oi": 100.0, "put_gamma_oi": -600.0},
        {"price": 95.0, "call_gamma_oi": 50.0, "put_gamma_oi": -400.0},
    ]
    with _with_rows(rows):
        ctx = _fetch_gex_context("TEST", spot=100.0)
    assert ctx["gamma_regime"] == "NEGATIVE"
    assert ctx["dealer_hedge_bias"] == "chase"
    assert "amplified" in (ctx["dealer_hedge_label"] or "").lower()


def test_pin_risk_flags_concentrated_near_spot_strike():
    """A single strike carrying >= PIN_RISK_MIN_CONCENTRATION of the
    near-spot gamma should be surfaced as the pin-risk level."""
    from app.features.options_context import (
        _fetch_gex_context,
        PIN_RISK_MIN_CONCENTRATION,
    )
    rows = [
        # 92% of near-spot |gamma| on the 100 strike.
        {"price": 100.0, "call_gamma_oi": 5000.0, "put_gamma_oi": -2000.0},
        {"price": 98.0, "call_gamma_oi": 150.0, "put_gamma_oi": -80.0},
        {"price": 102.0, "call_gamma_oi": 150.0, "put_gamma_oi": -80.0},
    ]
    with _with_rows(rows):
        ctx = _fetch_gex_context("TEST", spot=100.0)
    assert ctx["pin_risk_strike"] == 100.0
    assert ctx["pin_risk_distance_pct"] == 0.0
    assert ctx["pin_risk_concentration"] >= PIN_RISK_MIN_CONCENTRATION


def test_pin_risk_none_when_diffuse_across_near_spot_band():
    """Evenly distributed gamma in the ±5% band should NOT be flagged as
    pin risk — no single strike acts as a magnet."""
    from app.features.options_context import _fetch_gex_context
    rows = [
        {"price": 96.0, "call_gamma_oi": 200.0, "put_gamma_oi": -100.0},
        {"price": 98.0, "call_gamma_oi": 200.0, "put_gamma_oi": -100.0},
        {"price": 100.0, "call_gamma_oi": 200.0, "put_gamma_oi": -100.0},
        {"price": 102.0, "call_gamma_oi": 200.0, "put_gamma_oi": -100.0},
        {"price": 104.0, "call_gamma_oi": 200.0, "put_gamma_oi": -100.0},
    ]
    with _with_rows(rows):
        ctx = _fetch_gex_context("TEST", spot=100.0)
    assert ctx["pin_risk_strike"] is None
    assert ctx["pin_risk_distance_pct"] is None
    assert ctx["pin_risk_concentration"] is None


def test_pin_risk_skipped_without_spot():
    from app.features.options_context import _fetch_gex_context
    rows = [
        {"price": 100.0, "call_gamma_oi": 5000.0, "put_gamma_oi": -2000.0},
        {"price": 98.0, "call_gamma_oi": 150.0, "put_gamma_oi": -80.0},
    ]
    with _with_rows(rows):
        ctx = _fetch_gex_context("TEST", spot=None)
    # Gamma-regime + dealer bias should still resolve.
    assert ctx["gamma_regime"] in ("POSITIVE", "NEGATIVE", "NEUTRAL")
    assert ctx["dealer_hedge_bias"] in ("suppress", "chase", "neutral")
    # Pin-risk fields must be untouched.
    assert ctx["pin_risk_strike"] is None
    assert ctx["pin_risk_distance_pct"] is None


def test_pin_risk_ignores_strikes_outside_band():
    """Strikes >5% away from spot must not compete for the pin-risk slot,
    even if they have colossal gamma."""
    from app.features.options_context import _fetch_gex_context
    rows = [
        # 100 strike: 50% of near-spot gamma (below threshold? ≥25% yes, pin wins).
        {"price": 100.0, "call_gamma_oi": 400.0, "put_gamma_oi": -200.0},
        # 102 strike: 50% of near-spot gamma.
        {"price": 102.0, "call_gamma_oi": 400.0, "put_gamma_oi": -200.0},
        # Way-out-of-band strike w/ huge gamma — MUST NOT be picked.
        {"price": 140.0, "call_gamma_oi": 100000.0, "put_gamma_oi": -50000.0},
    ]
    with _with_rows(rows):
        ctx = _fetch_gex_context("TEST", spot=100.0)
    # 50% concentration exceeds the 25% minimum, and the far-OTM strike is excluded.
    assert ctx["pin_risk_strike"] in (100.0, 102.0)


if __name__ == "__main__":
    import traceback
    tests = [
        test_positive_net_gex_labels_dealer_as_suppress,
        test_negative_net_gex_labels_dealer_as_chase,
        test_pin_risk_flags_concentrated_near_spot_strike,
        test_pin_risk_none_when_diffuse_across_near_spot_band,
        test_pin_risk_skipped_without_spot,
        test_pin_risk_ignores_strikes_outside_band,
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

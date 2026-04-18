"""Tests for Wave 2 OLS-based delta_momentum in ``fetch_net_prem_ticks``.

The prior implementation compared ``latest vs data[len/4]`` — a single
outlier tick at the 25% mark could flip the momentum sign.  The Wave 2
upgrade fits an OLS slope to the full net_delta series and returns both
the normalised ``delta_momentum`` and a ``delta_momentum_tstat`` for
confidence.

What we validate:
  * Monotone increasing net_delta → positive momentum + positive t-stat.
  * Monotone decreasing → negative momentum + negative t-stat.
  * Flat series → near-zero momentum/t-stat.
  * A single outlier no longer flips the sign (robustness vs old impl).
  * Short series (< 4 points) falls back to 0 momentum without crashing.
  * Output keys match the documented contract.

Run with:
    python -m pytest tests/test_net_prem_ticks_regression.py -v
    python -m tests.test_net_prem_ticks_regression    # standalone
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


def _series(deltas: list[float]) -> list[dict]:
    """Build a fake net-prem-ticks payload with controlled net_delta values."""
    return [
        {
            "net_call_premium": 1000.0,
            "net_put_premium": 500.0,
            "net_delta": d,
        }
        for d in deltas
    ]


def _with_rows(rows: list[dict]):
    return mock.patch(
        "app.vendors.unusual_whales._uw_request",
        return_value=_FakeResp(rows),
    )


def test_monotone_rising_delta_gives_positive_momentum_and_tstat():
    from app.vendors.unusual_whales import fetch_net_prem_ticks
    deltas = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    with _with_rows(_series(deltas)):
        npt = fetch_net_prem_ticks("AAPL")
    assert npt is not None
    assert npt["delta_momentum"] > 0.0
    # Clean linear ramp → zero residuals → effectively infinite |t|.
    # The function caps the magnitude at 5.0.
    assert npt["delta_momentum_tstat"] == 5.0


def test_monotone_falling_delta_gives_negative_momentum_and_tstat():
    from app.vendors.unusual_whales import fetch_net_prem_ticks
    deltas = [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0]
    with _with_rows(_series(deltas)):
        npt = fetch_net_prem_ticks("AAPL")
    assert npt is not None
    assert npt["delta_momentum"] < 0.0
    assert npt["delta_momentum_tstat"] == -5.0


def test_flat_delta_returns_near_zero_momentum():
    from app.vendors.unusual_whales import fetch_net_prem_ticks
    deltas = [42.0] * 12
    with _with_rows(_series(deltas)):
        npt = fetch_net_prem_ticks("AAPL")
    assert npt is not None
    # Flat series → slope 0 → momentum 0.
    assert abs(npt["delta_momentum"]) < 1e-6
    assert abs(npt["delta_momentum_tstat"]) < 1e-6


def test_single_outlier_does_not_flip_momentum_sign():
    """Smoke-test the robustness claim: a single wild tick at the 25%
    mark (which would have dominated the legacy calc) should not flip
    the sign when the overall trend is clearly up."""
    from app.vendors.unusual_whales import fetch_net_prem_ticks
    # Clear uptrend with a rogue negative spike near the 25% index.
    deltas = [10, 20, -500, 35, 45, 55, 65, 75, 85, 95, 105, 115]
    with _with_rows(_series(deltas)):
        npt = fetch_net_prem_ticks("AAPL")
    assert npt is not None
    # OLS keeps the slope positive despite the outlier.
    assert npt["delta_momentum"] > 0.0


def test_short_series_returns_zero_momentum_without_crashing():
    from app.vendors.unusual_whales import fetch_net_prem_ticks
    deltas = [5.0, 10.0, 15.0]  # n < 4, regression skipped
    with _with_rows(_series(deltas)):
        npt = fetch_net_prem_ticks("AAPL")
    assert npt is not None
    assert npt["delta_momentum"] == 0.0
    assert npt["delta_momentum_tstat"] == 0.0


def test_output_contract_includes_tstat_and_direction():
    from app.vendors.unusual_whales import fetch_net_prem_ticks
    deltas = [0.0, 10.0, 20.0, 30.0]
    with _with_rows(_series(deltas)):
        npt = fetch_net_prem_ticks("AAPL")
    assert npt is not None
    for key in (
        "intraday_premium_direction",
        "delta_momentum",
        "delta_momentum_tstat",
        "net_delta",
        "net_call_premium",
        "net_put_premium",
    ):
        assert key in npt, f"missing key {key}"


if __name__ == "__main__":
    import traceback
    tests = [
        test_monotone_rising_delta_gives_positive_momentum_and_tstat,
        test_monotone_falling_delta_gives_negative_momentum_and_tstat,
        test_flat_delta_returns_near_zero_momentum,
        test_single_outlier_does_not_flip_momentum_sign,
        test_short_series_returns_zero_momentum_without_crashing,
        test_output_contract_includes_tstat_and_direction,
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

"""Regression tests for ``fetch_stock_screener`` derivations.

UW's ``/screener/stocks`` response stopped returning (or never returned)
the ratio fields the pipeline keys on — ``total_oi_change_perc``,
``call_oi_change_perc``, ``put_oi_change_perc``, ``perc_3_day_total``,
``perc_30_day_total`` and the total options ``volume``.  The vendor
module now derives them from the raw inputs; these tests pin the math
so a UI regression ("OI change +0.0% on every card") doesn't silently
return.

Runnable standalone (``python tests/test_screener_enrichment.py``) or
under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vendors.unusual_whales import _enrich_screener_derivations


def _sample_row() -> dict:
    """Realistic UW screener row (values drawn from a live NFLX response)."""
    return {
        "ticker": "NFLX",
        "call_open_interest": 3_241_474,
        "put_open_interest": 2_999_776,
        "total_open_interest": 6_241_250,
        "prev_call_oi": 3_296_300,
        "prev_put_oi": 2_861_805,
        "call_volume": 1_295_372,
        "put_volume": 586_740,
        "avg_30_day_call_volume": 267_917.6,
        "avg_30_day_put_volume": 133_273.5,
        "avg_3_day_call_volume": 773_994,
        "avg_3_day_put_volume": 193_014,
        "bullish_premium": 12_000_000,
        "bearish_premium": 4_000_000,
    }


def test_derives_oi_change_perc() -> None:
    row = _sample_row()
    _enrich_screener_derivations(row)

    assert abs(row["call_oi_change_perc"] - ((3_241_474 - 3_296_300) / 3_296_300 * 100)) < 1e-6
    assert abs(row["put_oi_change_perc"] - ((2_999_776 - 2_861_805) / 2_861_805 * 100)) < 1e-6
    prev_total = 3_296_300 + 2_861_805
    assert abs(row["total_oi_change_perc"] - ((6_241_250 - prev_total) / prev_total * 100)) < 1e-6


def test_derives_volume_and_ratios() -> None:
    row = _sample_row()
    _enrich_screener_derivations(row)

    total_opt_vol = 1_295_372 + 586_740
    assert row["volume"] == total_opt_vol

    denom_30 = 267_917.6 + 133_273.5
    assert abs(row["perc_30_day_total"] - total_opt_vol / denom_30) < 1e-6

    denom_3 = 773_994 + 193_014
    assert abs(row["perc_3_day_total"] - total_opt_vol / denom_3) < 1e-6


def test_derives_net_premium_when_missing() -> None:
    row = _sample_row()
    _enrich_screener_derivations(row)
    assert row["net_premium"] == 12_000_000 - 4_000_000


def test_preserves_existing_values() -> None:
    """If UW ever starts returning a ratio itself, don't clobber it."""
    row = _sample_row()
    row["total_oi_change_perc"] = 99.0
    row["perc_30_day_total"] = 7.0
    row["volume"] = 1
    _enrich_screener_derivations(row)
    assert row["total_oi_change_perc"] == 99.0
    assert row["perc_30_day_total"] == 7.0
    assert row["volume"] == 1


def test_fail_soft_on_missing_inputs() -> None:
    """Blank row should stay blank, not raise or insert bogus values."""
    row: dict = {"ticker": "XYZ"}
    _enrich_screener_derivations(row)
    for k in (
        "total_oi_change_perc",
        "call_oi_change_perc",
        "put_oi_change_perc",
        "perc_30_day_total",
        "perc_3_day_total",
        "volume",
        "net_premium",
    ):
        assert row.get(k) in (None, "", 0, 0.0)  # i.e. not fabricated


def test_fail_soft_on_zero_prev_oi() -> None:
    """Prev OI = 0 must not cause ZeroDivisionError — ratio just stays None."""
    row = {
        "ticker": "NEW",
        "call_open_interest": 100,
        "put_open_interest": 50,
        "prev_call_oi": 0,
        "prev_put_oi": 0,
        "call_volume": 10,
        "put_volume": 5,
    }
    _enrich_screener_derivations(row)
    assert row.get("call_oi_change_perc") in (None, "")
    assert row.get("put_oi_change_perc") in (None, "")
    assert row.get("total_oi_change_perc") in (None, "")


def _run_all() -> None:
    tests = [
        test_derives_oi_change_perc,
        test_derives_volume_and_ratios,
        test_derives_net_premium_when_missing,
        test_preserves_existing_values,
        test_fail_soft_on_missing_inputs,
        test_fail_soft_on_zero_prev_oi,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {t.__name__}: {e}")
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()

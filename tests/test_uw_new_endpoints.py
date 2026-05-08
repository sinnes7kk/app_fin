"""Tests for the 6 new UW fetchers added for the feature lab.

All UW HTTP calls are mocked — these tests verify the response-shape
parsing and "fail soft" behaviour, not the live API.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.vendors.unusual_whales as uw  # noqa: E402
import app.features.feature_lab_uw as flu  # noqa: E402


def _mock_resp(payload: object, *, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = {}
    r.json.return_value = payload
    if status >= 400:
        r.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        r.raise_for_status.return_value = None
    return r


def test_fetch_greek_exposure_parses_aliases():
    payload = {"data": [{
        "gex": 1.23e9,
        "vanna": -2.5e7,
        "charm_exposure": 4.4e6,  # alias variant
    }]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_greek_exposure("AAPL")
    assert out is not None
    assert out["gex_total"] == 1.23e9
    assert out["vanna_total"] == -2.5e7
    assert out["charm_total"] == 4.4e6
    print("  PASS: test_fetch_greek_exposure_parses_aliases")


def test_fetch_greek_exposure_returns_none_on_empty():
    with patch.object(uw, "_uw_request", return_value=_mock_resp({"data": []})):
        out = uw.fetch_greek_exposure("AAPL")
    assert out is None
    print("  PASS: test_fetch_greek_exposure_returns_none_on_empty")


def test_fetch_iv_skew_direct_field():
    with patch.object(uw, "_uw_request",
                      return_value=_mock_resp({"data": {"skew": 0.18}})):
        out = uw.fetch_iv_skew("AAPL")
    assert out == {"iv_skew_25d": 0.18}
    print("  PASS: test_fetch_iv_skew_direct_field")


def test_fetch_iv_skew_derived_from_components():
    payload = {"data": {"put_iv_25d": 0.40, "call_iv_25d": 0.30, "atm_iv": 0.35}}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_iv_skew("AAPL")
    assert out is not None
    # (0.40 - 0.30) / 0.35 ≈ 0.2857
    assert abs(out["iv_skew_25d"] - (0.10 / 0.35)) < 1e-6
    print("  PASS: test_fetch_iv_skew_derived_from_components")


def test_fetch_atm_iv_term_picks_closest_dte():
    rows = [
        {"dte": 28, "iv": 0.30},
        {"dte": 60, "iv": 0.32},
        {"dte": 90, "iv": 0.35},
        {"dte": 365, "iv": 0.40},
    ]
    with patch.object(uw, "_uw_request",
                      return_value=_mock_resp({"data": rows})):
        out = uw.fetch_atm_iv_term("AAPL")
    assert out is not None
    assert out["atm_iv_30d"] == 0.30
    assert out["atm_iv_60d"] == 0.32
    assert out["atm_iv_90d"] == 0.35
    # term_slope_30_90 = (0.35 - 0.30) / 0.30 ≈ 0.1667
    assert abs(out["term_slope_30_90"] - (0.05 / 0.30)) < 1e-6
    print("  PASS: test_fetch_atm_iv_term_picks_closest_dte")


def test_fetch_expiry_breakdown_concentration():
    payload = {"data": [
        {"expiry": "2026-05-15", "premium": 800},
        {"expiry": "2026-06-19", "premium": 100},
        {"expiry": "2026-07-17", "premium": 100},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_expiry_breakdown("AAPL")
    assert out is not None
    assert abs(out["expiry_concentration_top1"] - 0.8) < 1e-9
    print("  PASS: test_fetch_expiry_breakdown_concentration")


def test_fetch_max_pain_distance():
    payload = {"data": {"max_pain": 100, "spot": 105}}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_max_pain("AAPL")
    assert out is not None
    # (105 - 100) / 105 ≈ 0.0476
    assert abs(out["max_pain_dist_pct"] - 5/105) < 1e-9
    print("  PASS: test_fetch_max_pain_distance")


def test_fetch_spot_exposures():
    payload = {"data": {"dealer_delta": -1e6, "dealer_gamma": 5e4}}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_spot_exposures("AAPL")
    assert out is not None
    assert out["dealer_net_delta_at_spot"] == -1e6
    assert out["dealer_net_gamma_at_spot"] == 5e4
    print("  PASS: test_fetch_spot_exposures")


def test_all_fetchers_fail_soft_on_http_error():
    bad = _mock_resp({}, status=500)
    fetchers = [
        uw.fetch_greek_exposure,
        uw.fetch_iv_skew,
        uw.fetch_atm_iv_term,
        uw.fetch_expiry_breakdown,
        uw.fetch_max_pain,
        uw.fetch_spot_exposures,
    ]
    for fn in fetchers:
        with patch.object(uw, "_uw_request", return_value=bad):
            assert fn("AAPL") is None, f"{fn.__name__} should return None on 500"
    print("  PASS: test_all_fetchers_fail_soft_on_http_error")


def test_all_fetchers_handle_empty_ticker():
    fetchers = [
        uw.fetch_greek_exposure,
        uw.fetch_iv_skew,
        uw.fetch_atm_iv_term,
        uw.fetch_expiry_breakdown,
        uw.fetch_max_pain,
        uw.fetch_spot_exposures,
    ]
    for fn in fetchers:
        assert fn("") is None, f"{fn.__name__} should return None on empty ticker"
    print("  PASS: test_all_fetchers_handle_empty_ticker")


def test_feature_lab_uw_caches_per_day():
    """fetch_uw_features should cache per-ticker per-day on disk and not
    re-call UW on the second invocation for the same ticker."""
    import datetime
    import json
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "feature_lab_cache"
        # Patch the module-level cache dir
        with patch.object(flu, "CACHE_DIR", cache_dir):
            # Pre-populate disk cache to simulate prior fetch
            today = datetime.date.today().isoformat()
            d = cache_dir / today
            d.mkdir(parents=True, exist_ok=True)
            (d / "AAPL.json").write_text(json.dumps({
                "_status": "ok",
                "gex_total": 1.0,
                "iv_skew_25d": 0.2,
            }))
            # Even with all fetchers patched to error out, the cached
            # values should be returned.
            with patch("app.vendors.unusual_whales.fetch_greek_exposure",
                       side_effect=Exception("should not be called")):
                out = flu.fetch_uw_features("AAPL")
            assert out["gex_total"] == 1.0
            assert out["iv_skew_25d"] == 0.2
    print("  PASS: test_feature_lab_uw_caches_per_day")


def main():
    tests = [
        test_fetch_greek_exposure_parses_aliases,
        test_fetch_greek_exposure_returns_none_on_empty,
        test_fetch_iv_skew_direct_field,
        test_fetch_iv_skew_derived_from_components,
        test_fetch_atm_iv_term_picks_closest_dte,
        test_fetch_expiry_breakdown_concentration,
        test_fetch_max_pain_distance,
        test_fetch_spot_exposures,
        test_all_fetchers_fail_soft_on_http_error,
        test_all_fetchers_handle_empty_ticker,
        test_feature_lab_uw_caches_per_day,
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

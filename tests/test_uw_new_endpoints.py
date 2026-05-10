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
    """Legacy single-row aliases — preserved for backwards compatibility."""
    payload = {"data": [{
        "gex": 1.23e9,
        "vanna": -2.5e7,
        "charm_exposure": 4.4e6,
    }]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_greek_exposure("AAPL")
    assert out is not None
    assert out["gex_total"] == 1.23e9
    assert out["vanna_total"] == -2.5e7
    assert out["charm_total"] == 4.4e6
    print("  PASS: test_fetch_greek_exposure_parses_aliases")


def test_fetch_greek_exposure_call_plus_put_schema():
    """Live UW schema (probed 2026-05-10): list of daily rows where
    each row carries call_<X> and put_<X> fields. Net = call + put
    (put values are already signed)."""
    payload = {"data": [
        # Older row — should be ignored, only the latest is used.
        {"date": "2026-05-07", "call_gamma": "1", "put_gamma": "1",
         "call_vanna": "1", "put_vanna": "1",
         "call_charm": "1", "put_charm": "1"},
        # Latest row.
        {"date": "2026-05-08",
         "call_gamma": "7675148.7155", "put_gamma": "-2050225.2204",
         "call_vanna": "120226635.0754", "put_vanna": "-26607932.1126",
         "call_charm": "117601585.0144", "put_charm": "341644753.4743"},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_greek_exposure("NVDA")
    assert out is not None
    # Net = call + put (signs already encoded).
    assert abs(out["gex_total"] - (7675148.7155 - 2050225.2204)) < 1e-3
    assert abs(out["vanna_total"] - (120226635.0754 - 26607932.1126)) < 1e-3
    assert abs(out["charm_total"] - (117601585.0144 + 341644753.4743)) < 1e-3
    print("  PASS: test_fetch_greek_exposure_call_plus_put_schema")


def test_fetch_greek_exposure_returns_none_on_empty():
    with patch.object(uw, "_uw_request", return_value=_mock_resp({"data": []})):
        out = uw.fetch_greek_exposure("AAPL")
    assert out is None
    print("  PASS: test_fetch_greek_exposure_returns_none_on_empty")


def test_fetch_iv_skew_picks_latest_delta25_row():
    """Live UW shape (probed 2026-05-10): list of daily rows from
    /historical-risk-reversal-skew with {date, ticker, delta,
    risk_reversal}. Take the latest date at delta=25."""
    payload = {"data": [
        {"date": "2026-04-25", "ticker": "NVDA", "delta": 25,
         "risk_reversal": "0.5"},
        {"date": "2026-05-08", "ticker": "NVDA", "delta": 25,
         "risk_reversal": "-13.89"},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_iv_skew("NVDA")
    assert out is not None
    assert abs(out["iv_skew_25d"] - (-13.89)) < 1e-6
    print("  PASS: test_fetch_iv_skew_picks_latest_delta25_row")


def test_fetch_iv_skew_legacy_dict_shape():
    """Legacy single-dict ``{skew}`` shape — backwards compatibility."""
    with patch.object(uw, "_uw_request",
                      return_value=_mock_resp({"data": {"skew": 0.18}})):
        out = uw.fetch_iv_skew("AAPL")
    assert out == {"iv_skew_25d": 0.18}
    print("  PASS: test_fetch_iv_skew_legacy_dict_shape")


def test_fetch_iv_skew_derived_from_components():
    """Component fallback (put_iv − call_iv) / atm_iv."""
    payload = {"data": {"put_iv_25d": 0.40, "call_iv_25d": 0.30, "atm_iv": 0.35}}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_iv_skew("AAPL")
    assert out is not None
    assert abs(out["iv_skew_25d"] - (0.10 / 0.35)) < 1e-6
    print("  PASS: test_fetch_iv_skew_derived_from_components")


def test_fetch_atm_iv_term_interpolated_iv_schema():
    """Live UW schema (probed 2026-05-10): /interpolated-iv returns
    {date, days, percentile, volatility, implied_move_perc}. Lookup
    by exact ``days`` value."""
    payload = {"data": [
        {"date": "2026-05-08", "days": 1, "volatility": "0.45"},
        {"date": "2026-05-08", "days": 30, "volatility": "0.30"},
        {"date": "2026-05-08", "days": 60, "volatility": "0.32"},
        {"date": "2026-05-08", "days": 90, "volatility": "0.35"},
        {"date": "2026-05-08", "days": 365, "volatility": "0.40"},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_atm_iv_term("NVDA")
    assert out is not None
    assert out["atm_iv_30d"] == 0.30
    assert out["atm_iv_60d"] == 0.32
    assert out["atm_iv_90d"] == 0.35
    assert abs(out["term_slope_30_90"] - (0.05 / 0.30)) < 1e-6
    print("  PASS: test_fetch_atm_iv_term_interpolated_iv_schema")


def test_fetch_atm_iv_term_picks_closest_dte():
    """Closest-DTE fallback when exact 30/60/90 aren't available."""
    rows = [
        {"dte": 28, "iv": 0.30},
        {"dte": 62, "iv": 0.32},
        {"dte": 88, "iv": 0.35},
        {"dte": 365, "iv": 0.40},
    ]
    with patch.object(uw, "_uw_request",
                      return_value=_mock_resp({"data": rows})):
        out = uw.fetch_atm_iv_term("AAPL")
    assert out is not None
    assert out["atm_iv_30d"] == 0.30
    assert out["atm_iv_60d"] == 0.32
    assert out["atm_iv_90d"] == 0.35
    assert abs(out["term_slope_30_90"] - (0.05 / 0.30)) < 1e-6
    print("  PASS: test_fetch_atm_iv_term_picks_closest_dte")


def test_fetch_expiry_breakdown_volume_schema():
    """Live UW schema (probed 2026-05-10): {expires, volume,
    open_interest, chains}. No premium field — concentration is
    computed from volume."""
    payload = {"data": [
        {"expires": "2026-05-08", "volume": 2_000_000, "open_interest": 500_000, "chains": 100},
        {"expires": "2026-05-15", "volume": 500_000, "open_interest": 300_000, "chains": 80},
        {"expires": "2026-06-19", "volume": 500_000, "open_interest": 700_000, "chains": 90},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_expiry_breakdown("NVDA")
    assert out is not None
    # 2M / 3M = 0.667
    assert abs(out["expiry_concentration_top1"] - (2_000_000 / 3_000_000)) < 1e-6
    print("  PASS: test_fetch_expiry_breakdown_volume_schema")


def test_fetch_expiry_breakdown_legacy_premium():
    """Legacy premium-keyed shape — backwards compatibility."""
    payload = {"data": [
        {"expiry": "2026-05-15", "premium": 800},
        {"expiry": "2026-06-19", "premium": 100},
        {"expiry": "2026-07-17", "premium": 100},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_expiry_breakdown("AAPL")
    assert out is not None
    assert abs(out["expiry_concentration_top1"] - 0.8) < 1e-9
    print("  PASS: test_fetch_expiry_breakdown_legacy_premium")


def test_fetch_max_pain_picks_nearest_expiry():
    """Live UW shape: list of per-expiry rows. Pick the nearest
    expiry (first when sorted ascending), not the farthest."""
    payload = {"data": [
        # Farthest expiry — must NOT be picked.
        {"expiry": "2028-12-15", "max_pain": "190", "close": "215.2"},
        # Nearest expiry — must be picked.
        {"expiry": "2026-05-08", "max_pain": "200", "close": "215.2"},
        {"expiry": "2026-06-19", "max_pain": "205", "close": "215.2"},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_max_pain("NVDA")
    assert out is not None
    # Nearest = max_pain=200, close=215.2 → (215.2 - 200) / 215.2
    expected = (215.2 - 200) / 215.2
    assert abs(out["max_pain_dist_pct"] - expected) < 1e-6
    print("  PASS: test_fetch_max_pain_picks_nearest_expiry")


def test_fetch_max_pain_legacy_dict_shape():
    """Legacy single-dict shape — backwards compatibility."""
    payload = {"data": {"max_pain": 100, "spot": 105}}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_max_pain("AAPL")
    assert out is not None
    assert abs(out["max_pain_dist_pct"] - 5/105) < 1e-9
    print("  PASS: test_fetch_max_pain_legacy_dict_shape")


def test_fetch_spot_exposures_minute_schema():
    """Live UW schema (probed 2026-05-10): minute-by-minute list
    with gamma_per_one_percent_move_oi (gamma) and
    vanna_per_one_percent_move_oi (used as delta proxy). Pick the
    latest minute by ``time``."""
    payload = {"data": [
        # Older minute — should be ignored.
        {"time": "2026-05-08T10:48:51Z",
         "gamma_per_one_percent_move_oi": "1.0",
         "vanna_per_one_percent_move_oi": "2.0"},
        # Latest minute.
        {"time": "2026-05-08T19:59:38Z",
         "gamma_per_one_percent_move_oi": "2437135032.96",
         "vanna_per_one_percent_move_oi": "160795653.38"},
    ]}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_spot_exposures("NVDA")
    assert out is not None
    assert abs(out["dealer_net_gamma_at_spot"] - 2437135032.96) < 1e-3
    assert abs(out["dealer_net_delta_at_spot"] - 160795653.38) < 1e-3
    print("  PASS: test_fetch_spot_exposures_minute_schema")


def test_fetch_spot_exposures_legacy_aliases():
    """Legacy single-dict shape with dealer_delta/dealer_gamma."""
    payload = {"data": {"dealer_delta": -1e6, "dealer_gamma": 5e4}}
    with patch.object(uw, "_uw_request", return_value=_mock_resp(payload)):
        out = uw.fetch_spot_exposures("AAPL")
    assert out is not None
    assert out["dealer_net_delta_at_spot"] == -1e6
    assert out["dealer_net_gamma_at_spot"] == 5e4
    print("  PASS: test_fetch_spot_exposures_legacy_aliases")


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
        test_fetch_greek_exposure_call_plus_put_schema,
        test_fetch_greek_exposure_returns_none_on_empty,
        test_fetch_iv_skew_picks_latest_delta25_row,
        test_fetch_iv_skew_legacy_dict_shape,
        test_fetch_iv_skew_derived_from_components,
        test_fetch_atm_iv_term_interpolated_iv_schema,
        test_fetch_atm_iv_term_picks_closest_dte,
        test_fetch_expiry_breakdown_volume_schema,
        test_fetch_expiry_breakdown_legacy_premium,
        test_fetch_max_pain_picks_nearest_expiry,
        test_fetch_max_pain_legacy_dict_shape,
        test_fetch_spot_exposures_minute_schema,
        test_fetch_spot_exposures_legacy_aliases,
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

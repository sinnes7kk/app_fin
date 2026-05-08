"""Unit tests for ``app/features/feature_lab.py``."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.features.feature_lab as fl  # noqa: E402


def _ohlcv(seed: int = 0, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.02, n)
    closes = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": rng.uniform(1e6, 5e6, n),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="B"),
    )


def test_bullish_premium_share():
    g = {"bullish_premium": 800.0, "bearish_premium": 200.0}
    assert fl._bullish_premium_share(g) == 0.8
    g = {"bullish_premium": 0, "bearish_premium": 0}
    assert fl._bullish_premium_share(g) is None
    print("  PASS: test_bullish_premium_share")


def test_unusual_premium_share():
    g = {"unusual_bullish_premium": 100, "unusual_bearish_premium": 300}
    assert abs(fl._unusual_premium_share(g) - 0.25) < 1e-9
    g = {}
    assert fl._unusual_premium_share(g) is None
    print("  PASS: test_unusual_premium_share")


def test_vrp_proxy_basic():
    g = {"latest_iv_rank": 80}
    out = fl._vrp_proxy(g, _ohlcv())
    assert out is not None
    assert -100 <= out <= 100
    print("  PASS: test_vrp_proxy_basic")


def test_vrp_proxy_missing_iv_rank_returns_none():
    g = {"latest_iv_rank": None}
    assert fl._vrp_proxy(g, _ohlcv()) is None
    print("  PASS: test_vrp_proxy_missing_iv_rank_returns_none")


def test_far_otm_shares():
    rows = pd.DataFrame([
        {"strike": 100, "underlying_price": 100, "premium": 100,
         "option_type": "CALL"},   # ATM call: not far
        {"strike": 120, "underlying_price": 100, "premium": 100,
         "option_type": "CALL"},   # 20% OTM: far
        {"strike": 80,  "underlying_price": 100, "premium": 100,
         "option_type": "PUT"},    # 20% OTM put: far
        {"strike": 102, "underlying_price": 100, "premium": 100,
         "option_type": "PUT"},    # ~2% ITM put: not far
    ])
    far_call, far_put = fl._far_otm_shares(rows)
    # Total premium = 400; far call = 100 (25%), far put = 100 (25%)
    assert abs(far_call - 0.25) < 1e-9
    assert abs(far_put - 0.25) < 1e-9
    print("  PASS: test_far_otm_shares")


def test_dollar_delta_weighted_flow_signed():
    """Bullish ITM call flow should produce a strongly positive value;
    bullish far-OTM put flow should produce a strongly negative value
    (dollar-delta of a put is negative, multiplied by direction +1)."""
    rows_calls = pd.DataFrame([
        {"strike": 90, "underlying_price": 100, "premium": 1000,
         "option_type": "CALL"},   # ITM, delta ~0.95
    ])
    val_calls = fl._dollar_delta_weighted_flow(rows_calls, "BULLISH")
    assert val_calls is not None and val_calls > 0

    rows_puts = pd.DataFrame([
        {"strike": 110, "underlying_price": 100, "premium": 1000,
         "option_type": "PUT"},    # ITM put, delta ~ -0.95
    ])
    val_puts = fl._dollar_delta_weighted_flow(rows_puts, "BULLISH")
    # ITM put bullish flow → premium × negative_delta × +1 = negative
    assert val_puts is not None and val_puts < 0
    print("  PASS: test_dollar_delta_weighted_flow_signed")


def test_realized_vol_regime_compression():
    rng = np.random.default_rng(42)
    # Build series with low variance recent / high variance older
    high = rng.normal(0, 0.04, 30)
    low = rng.normal(0, 0.005, 10)
    rets = np.concatenate([high, low])
    closes = 100 * np.exp(np.cumsum(rets))
    ohlcv = pd.DataFrame(
        {"close": closes},
        index=pd.date_range("2026-01-01", periods=len(closes), freq="B"),
    )
    val = fl._realized_vol_regime(ohlcv)
    assert val is not None and val < 1.0, \
        f"compressed-vol window should give regime < 1, got {val}"
    print("  PASS: test_realized_vol_regime_compression")


def test_sector_relative_pct_lookup():
    grades = [
        {"ticker": "A", "sector": "Tech", "prem_mcap_bps": 1.0},
        {"ticker": "B", "sector": "Tech", "prem_mcap_bps": 5.0},
        {"ticker": "C", "sector": "Tech", "prem_mcap_bps": 10.0},
        {"ticker": "D", "sector": "Tech", "prem_mcap_bps": 15.0},
        {"ticker": "E", "sector": "Tech", "prem_mcap_bps": 20.0},
        # Healthcare has only 2 rows → skipped (n < 3)
        {"ticker": "X", "sector": "Healthcare", "prem_mcap_bps": 1.0},
        {"ticker": "Y", "sector": "Healthcare", "prem_mcap_bps": 100.0},
    ]
    out = fl._sector_relative_pct_lookup(grades)
    # Tech median is 10. C is the median, so its pct ≈ 0.
    assert abs(out["C"]) < 0.01
    # E is well above the median → positive
    assert out["E"] > 0
    # Healthcare names should be missing.
    assert "X" not in out
    print("  PASS: test_sector_relative_pct_lookup")


def test_compute_lab_features_full_schema():
    grades = [
        {
            "ticker": "AAA", "direction": "BULLISH", "sector": "Tech",
            "conviction_score": 8, "conviction_grade": "A",
            "bullish_premium": 1000, "bearish_premium": 200,
            "unusual_bullish_premium": 500, "unusual_bearish_premium": 50,
            "prem_mcap_bps": 12.0, "cumulative_premium": 5e6,
            "latest_iv_rank": 60.0, "latest_oi_change": 0.2,
        },
        {
            "ticker": "BBB", "direction": "BULLISH", "sector": "Tech",
            "conviction_score": 5, "conviction_grade": "B",
            "bullish_premium": 100, "bearish_premium": 100,
            "unusual_bullish_premium": 0, "unusual_bearish_premium": 0,
            "prem_mcap_bps": 1.0, "cumulative_premium": 1e5,
            "latest_iv_rank": None, "latest_oi_change": 0.0,
        },
    ]
    rows = fl.compute_lab_features(
        grades,
        fetch_uw=False,
        ohlcv_loader=lambda t: _ohlcv(),
    )
    assert len(rows) == 2
    for r in rows:
        for col in fl.LAB_COLS:
            assert col in r, f"missing column {col}"
        assert r["bullish_premium_share"] is not None
    print("  PASS: test_compute_lab_features_full_schema")


def test_persist_feature_lab_idempotent_on_as_of(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "feature_lab.csv"
        monkeypatch.setattr(fl, "FEATURE_LAB_PATH", tmp)

        rows_a = [{"ticker": "X", "direction": "BULLISH"}]
        rows_b = [
            {"ticker": "Y", "direction": "BULLISH"},
            {"ticker": "Z", "direction": "BEARISH"},
        ]
        fl.persist_feature_lab(rows_a, as_of="2026-05-07")
        fl.persist_feature_lab(rows_b, as_of="2026-05-08")
        # Re-running today should REPLACE today's rows, not duplicate.
        fl.persist_feature_lab(rows_b, as_of="2026-05-08")

        on_disk = pd.read_csv(tmp)
        assert (on_disk["as_of"] == "2026-05-07").sum() == 1
        assert (on_disk["as_of"] == "2026-05-08").sum() == 2
    print("  PASS: test_persist_feature_lab_idempotent_on_as_of")


def test_compute_lab_features_topn_gates_uw():
    """Only the top-N candidates by conviction_score should get UW fetches."""
    grades = []
    for i in range(5):
        grades.append({
            "ticker": f"T{i}",
            "direction": "BULLISH",
            "conviction_score": float(i),
            "bullish_premium": 100, "bearish_premium": 50,
        })
    fetched: list[str] = []

    def fake_uw(ticker: str, spot: float | None) -> dict:
        fetched.append(ticker)
        return {c: 1.0 for c in fl.UW_FEATURE_COLS}

    rows = fl.compute_lab_features(
        grades,
        fetch_uw=True,
        topn_cutoff=2,
        ohlcv_loader=lambda t: None,
        uw_loader=fake_uw,
    )
    # Only the top 2 by conviction_score (T4, T3) should be fetched.
    assert set(fetched) == {"T4", "T3"}, f"fetched: {fetched}"
    # Their UW columns should be populated.
    by_t = {r["ticker"]: r for r in rows}
    assert by_t["T4"]["gex_total"] == 1.0
    assert by_t["T0"]["gex_total"] is None
    print("  PASS: test_compute_lab_features_topn_gates_uw")


# Minimal monkeypatch helper.
class _Monkeypatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        for t, n, v in reversed(self._undo):
            setattr(t, n, v)


def main():
    tests = [
        ("test_bullish_premium_share", test_bullish_premium_share, False),
        ("test_unusual_premium_share", test_unusual_premium_share, False),
        ("test_vrp_proxy_basic", test_vrp_proxy_basic, False),
        ("test_vrp_proxy_missing_iv_rank_returns_none", test_vrp_proxy_missing_iv_rank_returns_none, False),
        ("test_far_otm_shares", test_far_otm_shares, False),
        ("test_dollar_delta_weighted_flow_signed", test_dollar_delta_weighted_flow_signed, False),
        ("test_realized_vol_regime_compression", test_realized_vol_regime_compression, False),
        ("test_sector_relative_pct_lookup", test_sector_relative_pct_lookup, False),
        ("test_compute_lab_features_full_schema", test_compute_lab_features_full_schema, False),
        ("test_persist_feature_lab_idempotent_on_as_of", test_persist_feature_lab_idempotent_on_as_of, True),
        ("test_compute_lab_features_topn_gates_uw", test_compute_lab_features_topn_gates_uw, False),
    ]
    failures = 0
    for name, fn, needs_mp in tests:
        try:
            if needs_mp:
                mp = _Monkeypatch()
                try:
                    fn(mp)
                finally:
                    mp.undo()
            else:
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

"""Tests for the delta-weighted directional premium pipeline.

Covers:
    - OCC id construction
    - Black-Scholes moneyness proxy signs + magnitudes
    - add_delta_weights with UW cache hit / miss
    - aggregate_flow_by_ticker emits delta columns
    - parity: USE_DELTA_WEIGHTED_FLOW=False preserves legacy scoring

Run with either:
    python -m pytest tests/test_delta_weighting.py -v
    python -m tests.test_delta_weighting           # standalone (no pytest required)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    class _PytestShim:
        def parametrize(self, name, values):
            def decorator(fn):
                fn.__parametrize__ = (name, values)
                return fn
            return decorator
        mark = None  # type: ignore

        def __getattr__(self, attr):
            return self

    pytest = _PytestShim()  # type: ignore
    pytest.mark = pytest

from app import config
from app.features import flow_features as ff
from app.features.flow_features import (
    _moneyness_delta_proxy,
    add_delta_weights,
    aggregate_flow_by_ticker,
    build_flow_feature_table,
)
from app.vendors import unusual_whales as uw
from app.vendors.unusual_whales import build_occ_id


# ---------------------------------------------------------------------------
# OCC id construction
# ---------------------------------------------------------------------------

def test_build_occ_id_known_symbols():
    """AAPL 2027-01-15 200C → AAPL270115C00200000 (standard OCC convention)."""
    occ = build_occ_id("AAPL", "2027-01-15", "CALL", 200.0)
    assert occ == "AAPL270115C00200000", occ


def test_build_occ_id_put_and_fractional_strike():
    occ = build_occ_id("SPY", "2024-12-20", "PUT", 420.5)
    assert occ == "SPY241220P00420500", occ


def test_build_occ_id_accepts_datetime():
    occ = build_occ_id("NVDA", pd.Timestamp("2025-06-20"), "C", 1200)
    assert occ == "NVDA250620C01200000", occ


def test_build_occ_id_rejects_bad_input():
    assert build_occ_id("", "2027-01-15", "C", 100) is None
    assert build_occ_id("AAPL", None, "C", 100) is None
    assert build_occ_id("AAPL", "2027-01-15", "X", 100) is None
    assert build_occ_id("AAPL", "2027-01-15", "C", 0) is None
    assert build_occ_id("AAPL", "2027-01-15", "C", "n/a") is None


# ---------------------------------------------------------------------------
# Moneyness proxy
# ---------------------------------------------------------------------------

def test_moneyness_proxy_call_deep_itm_near_one():
    """Call with S >> K, mid-DTE, sigma=0.35 → delta close to 1."""
    d = _moneyness_delta_proxy("CALL", strike=100, underlying_price=200, dte=45, sigma=0.35)
    assert d is not None
    assert 0.95 <= d <= 1.0


def test_moneyness_proxy_call_otm_below_half():
    """Call with K far above S → delta < 0.5."""
    d = _moneyness_delta_proxy("CALL", strike=200, underlying_price=100, dte=45, sigma=0.35)
    assert d is not None
    assert 0.0 <= d < 0.5


def test_moneyness_proxy_atm_roughly_half_and_sign_split():
    """ATM call delta ≈ 0.5; ATM put delta ≈ -0.5 (small vol skew from r=0)."""
    d_c = _moneyness_delta_proxy("CALL", strike=100, underlying_price=100, dte=45, sigma=0.35)
    d_p = _moneyness_delta_proxy("PUT", strike=100, underlying_price=100, dte=45, sigma=0.35)
    assert d_c is not None and d_p is not None
    # BS with r=0: N(d1) where d1 = 0.5*sigma*sqrt(T) > 0 → slightly above 0.5
    assert 0.5 < d_c < 0.6
    # Put delta N(d1) - 1 is negative but > -0.5
    assert -0.5 < d_p < 0.0


def test_moneyness_proxy_put_delta_magnitude_in_range():
    d = _moneyness_delta_proxy("PUT", strike=120, underlying_price=100, dte=60, sigma=0.35)
    assert d is not None
    assert -1.0 <= d <= 0.0
    assert abs(d) > 0.5  # put is ITM → |delta| > 0.5


def test_moneyness_proxy_handles_bad_inputs():
    assert _moneyness_delta_proxy("CALL", 0, 100, 45) is None
    assert _moneyness_delta_proxy("CALL", 100, 0, 45) is None
    assert _moneyness_delta_proxy("CALL", 100, 100, 45, sigma=0) is None
    assert _moneyness_delta_proxy("X", 100, 100, 45) is None
    assert _moneyness_delta_proxy(None, 100, 100, 45) is None


# ---------------------------------------------------------------------------
# add_delta_weights
# ---------------------------------------------------------------------------

def _synth_flow_events(n_call: int = 2, n_put: int = 1) -> pd.DataFrame:
    base = {
        "ticker": "AAPL",
        "expiration_date": pd.Timestamp(date.today() + timedelta(days=45)),
        "underlying_price": 200.0,
        "dte": 45,
        "event_ts": pd.Timestamp.utcnow().tz_localize(None),
        "direction": "LONG",
        "direction_confidence": 1.0,
        "premium": 750_000.0,
        "premium_raw": 750_000.0,
        "is_sweep": False,
    }
    rows = []
    for i in range(n_call):
        rows.append({**base, "option_type": "CALL", "strike": 180.0 + i})
    for i in range(n_put):
        rows.append({**base, "option_type": "PUT", "strike": 220.0 + i, "direction": "SHORT"})
    return pd.DataFrame(rows)


def test_add_delta_weights_uses_uw_when_available(monkeypatch):
    """When fetch_contract_greeks returns a valid delta, delta_source='uw' and
    delta_premium == premium * |delta|."""
    df = _synth_flow_events(n_call=1, n_put=0)

    def _fake_greeks(contract_id, *, as_of_ts=None, timeout=5):
        return {"delta": 0.62, "iv": 0.30, "ts": "2024-01-01"}

    monkeypatch.setattr(ff, "add_delta_weights", ff.add_delta_weights)  # keep real impl
    monkeypatch.setattr("app.features.flow_features.fetch_contract_greeks", _fake_greeks, raising=False)
    # The feature module imports fetch_contract_greeks inside the function; patch the vendor
    # module too so the local import grabs our fake.
    monkeypatch.setattr(uw, "fetch_contract_greeks", _fake_greeks)

    out = add_delta_weights(df)
    assert not out.empty
    assert (out["delta_source"] == "uw").all(), out[["contract_id", "delta_source"]].to_dict()
    np.testing.assert_allclose(out["delta"].iloc[0], 0.62, atol=1e-6)
    np.testing.assert_allclose(out["delta_premium"].iloc[0], 750_000 * 0.62, rtol=1e-6)


def test_add_delta_weights_falls_back_to_proxy(monkeypatch):
    """When UW returns None, delta_source='proxy' and delta is finite & signed."""
    df = _synth_flow_events(n_call=1, n_put=1)

    def _fake_none(*a, **kw):
        return None

    monkeypatch.setattr(uw, "fetch_contract_greeks", _fake_none)

    out = add_delta_weights(df)
    assert (out["delta_source"] == "proxy").all()
    # Call row has strike 180 under S=200 → delta > 0.5
    call_row = out[out["option_type"] == "CALL"].iloc[0]
    assert 0.5 < float(call_row["delta"]) <= 1.0
    # Put row has strike 220 above S=200 → delta < -0.5 (ITM put)
    put_row = out[out["option_type"] == "PUT"].iloc[0]
    assert -1.0 <= float(put_row["delta"]) < -0.5
    # delta_premium is always non-negative (uses |delta|)
    assert (out["delta_premium"] >= 0).all()


def test_add_delta_weights_on_empty_df():
    out = add_delta_weights(pd.DataFrame())
    assert out.empty


def test_add_delta_weights_above_cap_skips_uw(monkeypatch):
    """When unique contracts exceed DELTA_MAX_UNIQUE_PER_SCAN, we skip the UW
    fetch entirely and everything falls back to the proxy."""
    df = _synth_flow_events(n_call=3, n_put=2)

    calls = {"n": 0}
    def _fake(*a, **kw):
        calls["n"] += 1
        return {"delta": 0.5, "iv": None, "ts": None}

    monkeypatch.setattr(uw, "fetch_contract_greeks", _fake)
    out = add_delta_weights(df, max_unique_fetch=1)  # cap below the 5 rows we have
    assert calls["n"] == 0, "UW should be skipped above the cap"
    assert (out["delta_source"] == "proxy").all()


# ---------------------------------------------------------------------------
# Aggregation emits delta columns
# ---------------------------------------------------------------------------

def _synth_aggregate_input(mcap: float = 5.0e9) -> pd.DataFrame:
    rows = [
        {
            "ticker": "AAPL",
            "premium": 1_000_000.0,
            "premium_raw": 1_000_000.0,
            "dte": 45,
            "direction": "LONG",
            "direction_confidence": 1.0,
            "is_sweep": False,
            "volume_oi_ratio": 1.5,
            "marketcap": mcap,
            "delta": 0.60,
            "delta_magnitude": 0.60,
            "delta_premium": 600_000.0,
            "delta_source": "uw",
        },
        {
            "ticker": "AAPL",
            "premium": 2_000_000.0,
            "premium_raw": 2_000_000.0,
            "dte": 45,
            "direction": "LONG",
            "direction_confidence": 1.0,
            "is_sweep": True,
            "volume_oi_ratio": 2.0,
            "marketcap": mcap,
            "delta": 0.20,
            "delta_magnitude": 0.20,
            "delta_premium": 400_000.0,
            "delta_source": "proxy",
        },
    ]
    return pd.DataFrame(rows)


def test_aggregate_includes_delta_columns():
    mcap = 5.0e9
    df = _synth_aggregate_input(mcap=mcap)
    agg = aggregate_flow_by_ticker(df)
    row = agg[agg["ticker"] == "AAPL"].iloc[0]

    # raw delta-premium = 600k + 400k = 1_000_000
    np.testing.assert_allclose(row["bullish_delta_premium_raw"], 1_000_000, atol=1.0)
    # delta intensity = 1_000_000 / mcap (rounded to 6 decimals by aggregate)
    np.testing.assert_allclose(row["bullish_delta_intensity"], round(1_000_000 / mcap, 6), atol=1e-9)
    # avg |delta| is premium-weighted: (1m*0.60 + 2m*0.20) / 3m = 1m/3m = 0.3333
    np.testing.assert_allclose(row["bullish_avg_delta"], (1_000_000 * 0.60 + 2_000_000 * 0.20) / 3_000_000, atol=1e-6)
    # source_mix: UW premium 1m / total 3m = 0.3333
    np.testing.assert_allclose(row["bullish_delta_source_mix"], 1_000_000 / 3_000_000, atol=1e-6)
    # bearish side has no activity
    assert row["bearish_delta_premium_raw"] == 0.0


def test_aggregate_delta_columns_zero_when_enrichment_skipped():
    df = _synth_aggregate_input().drop(columns=[
        "delta", "delta_magnitude", "delta_premium", "delta_source"
    ])
    agg = aggregate_flow_by_ticker(df)
    row = agg[agg["ticker"] == "AAPL"].iloc[0]
    assert row["bullish_delta_premium_raw"] == 0.0
    assert row["bullish_delta_intensity"] == 0.0
    assert row["bullish_avg_delta"] == 0.0


# ---------------------------------------------------------------------------
# Parity: flag OFF → score unchanged
# ---------------------------------------------------------------------------

def test_parity_when_flag_off(monkeypatch):
    """With USE_DELTA_WEIGHTED_FLOW=False, bullish_score is identical whether
    or not add_delta_weights ran — the scorer still reads bullish_flow_intensity.
    """
    monkeypatch.setattr(config, "USE_DELTA_WEIGHTED_FLOW", False, raising=False)

    df_no_delta = _synth_aggregate_input().drop(columns=[
        "delta", "delta_magnitude", "delta_premium", "delta_source",
    ])
    df_with_delta = _synth_aggregate_input()

    score_no = float(aggregate_flow_by_ticker(df_no_delta).iloc[0]["bullish_score"])
    score_with = float(aggregate_flow_by_ticker(df_with_delta).iloc[0]["bullish_score"])
    np.testing.assert_allclose(score_no, score_with, atol=1e-9)


def test_score_changes_when_flag_on(monkeypatch):
    """With USE_DELTA_WEIGHTED_FLOW=True, the scorer reads bullish_delta_intensity
    instead of bullish_flow_intensity — so the score should differ from the
    flag-OFF baseline when the two intensities differ. Uses a mid-cap so the
    raw intensity doesn't saturate the clip-scale ceiling."""
    df = _synth_aggregate_input(mcap=5.0e10)

    monkeypatch.setattr(config, "USE_DELTA_WEIGHTED_FLOW", False, raising=False)
    off = float(aggregate_flow_by_ticker(df).iloc[0]["bullish_score"])

    monkeypatch.setattr(config, "USE_DELTA_WEIGHTED_FLOW", True, raising=False)
    on = float(aggregate_flow_by_ticker(df).iloc[0]["bullish_score"])

    # Same dataset, different scoring input column → scores should differ
    assert abs(on - off) > 1e-6, f"expected scores to differ (on={on}, off={off})"


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

class _MonkeyPatch:
    """Tiny monkeypatch shim mimicking pytest's API (setattr with restore)."""

    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target, name, value=None, raising: bool = True):
        if isinstance(target, str):
            mod_path, _, attr = target.rpartition(".")
            module = __import__(mod_path, fromlist=[attr])
            obj = module
            final_name = attr
            final_value = name
        else:
            obj = target
            final_name = name
            final_value = value
        try:
            old = getattr(obj, final_name)
            had = True
        except AttributeError:
            if raising:
                raise
            old = None
            had = False
        setattr(obj, final_name, final_value)
        self._undo.append((obj, final_name, old, had))

    def undo(self) -> None:
        for obj, name, old, had in reversed(self._undo):
            if had:
                setattr(obj, name, old)
            else:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
        self._undo.clear()


def _run_standalone() -> int:
    import inspect
    import sys
    import traceback

    mod = sys.modules[__name__]
    tests = [
        (name, getattr(mod, name))
        for name in dir(mod)
        if name.startswith("test_") and callable(getattr(mod, name))
    ]
    failed: list[tuple[str, str]] = []
    passed = 0
    for name, fn in tests:
        mp = _MonkeyPatch()
        try:
            sig = inspect.signature(fn)
            kwargs = {"monkeypatch": mp} if "monkeypatch" in sig.parameters else {}
            fn(**kwargs)
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}: {e}")
        finally:
            mp.undo()

    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        for name, tb in failed:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())

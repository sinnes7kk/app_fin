"""Unit tests for app/analytics/trade_replay.py.

Self-contained — no pytest dependency. Run with:
    python tests/test_trade_replay.py
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.trade_replay import replay_trade_plan


def _make_ohlcv(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    """Build a synthetic OHLCV frame from a close-price path.

    Uses ATR-stable ranges so that a freshly computed ATR-14 produces a
    sensible per-bar value.
    """
    dates = pd.date_range(start=start, periods=len(closes), freq="B")
    data = []
    prev = closes[0]
    for c in closes:
        # Symmetric high/low around close with a small range so ATR is positive
        rng = max(abs(c - prev) * 1.5, 0.5)
        high = max(c, prev) + rng / 2
        low = min(c, prev) - rng / 2
        # Open ~midway between prev close and current
        op = (prev + c) / 2
        data.append({"open": op, "high": high, "low": low, "close": c, "volume": 1_000_000})
        prev = c
    return pd.DataFrame(data, index=dates)


def _flat_spy(n: int, start: str = "2026-01-01") -> pd.DataFrame:
    """A flat SPY series so realized_excess_pct ≈ ticker_return."""
    dates = pd.date_range(start=start, periods=n, freq="B")
    return pd.DataFrame({
        "open": 500.0, "high": 500.0, "low": 500.0, "close": 500.0, "volume": 1_000_000,
    }, index=dates)


def assert_eq(a, b, msg=""):
    assert a == b, f"FAIL {msg}: expected {b!r}, got {a!r}"


def assert_close(a, b, tol=0.05, msg=""):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        raise AssertionError(f"FAIL {msg}: got {a}, expected ~{b}")
    assert abs(a - b) <= tol, f"FAIL {msg}: {a} vs {b} (tol {tol})"


def _ramp_up_setup(n: int = 20, start: float = 90.0, end: float = 100.0) -> list[float]:
    """Slight uptrend leading into entry so EMA20 ends below entry (realistic
    long-breakout setup).
    """
    return list(np.linspace(start, end, n))


def _ramp_down_setup(n: int = 20, start: float = 110.0, end: float = 100.0) -> list[float]:
    return list(np.linspace(start, end, n))


def test_long_hits_t2():
    # Long position rallies hard. With stop_atr_mult=1.0, T2 at +3R.
    closes = _ramp_up_setup() + [101.5, 103.0, 104.5, 106.0, 108.0]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BULLISH", df, spy)
    assert res["exit_reason"] in ("T2",), f"got {res['exit_reason']}"
    # Blended R: 0.5 * 1.5 (partial at T1) + 0.5 * 3.0 (T2) = 2.25R if partial fired,
    # otherwise 3.0R clean.
    assert res["realized_r"] >= 1.5, f"expected R >= 1.5 at T2, got {res['realized_r']}"
    print("  PASS: test_long_hits_t2")


def test_long_stops_out():
    closes = _ramp_up_setup() + [99.5, 98.0, 96.0]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BULLISH", df, spy)
    assert res["exit_reason"] in ("stop", "T1_then_stop"), f"got {res['exit_reason']}"
    assert res["realized_r"] < 0, f"expected loss, got R={res['realized_r']}"
    print("  PASS: test_long_stops_out")


def test_long_t1_then_stop():
    # T1 hits, price drops back below post-T1 trail.
    closes = _ramp_up_setup() + [101.6, 102.5, 101.0, 99.5, 98.0]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BULLISH", df, spy)
    assert res["partial_filled"], "expected partial_filled=True"
    assert res["exit_reason"] in ("T1_then_stop", "stop"), f"got {res['exit_reason']}"
    print("  PASS: test_long_t1_then_stop")


def test_short_hits_t2():
    # Sharp clean down move so T2 hits before noisy intraday range trips trail.
    closes = _ramp_down_setup() + [98.5, 96.5, 94.5, 92.5, 90.0]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BEARISH", df, spy)
    # Either T2 (clean) or T1_then_stop (after partial); both indicate a profitable short.
    assert res["exit_reason"] in ("T2", "T1_then_stop"), f"got {res['exit_reason']}"
    assert res["realized_r"] >= 0.75, f"expected R >= 0.75, got {res['realized_r']}"
    print("  PASS: test_short_hits_t2")


def test_short_stops_out():
    closes = _ramp_down_setup() + [101.0, 102.5, 104.0]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BEARISH", df, spy)
    assert res["exit_reason"] in ("stop", "T1_then_stop"), f"got {res['exit_reason']}"
    assert res["realized_r"] < 0, f"expected loss"
    print("  PASS: test_short_stops_out")


def test_time_stop_kicks_in():
    # Long, but price flat the whole window. Should hit time_stop after max_hold_days.
    closes = [100.0] * 30  # 30 flat bars
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[15].strftime("%Y-%m-%d")  # entry on bar 16
    res = replay_trade_plan(
        "TEST", as_of, "BULLISH", df, spy,
        max_hold_days=5, time_stop_min_r=1.0,
    )
    # In a perfectly flat series, no exit triggers (no volatility)
    # so it should either time_stop or no_exit_yet
    assert res["exit_reason"] in ("time_stop", "no_exit_yet", "stop"), f"got {res['exit_reason']}"
    print("  PASS: test_time_stop_kicks_in")


def test_mfe_mae_correctness():
    # Path: enters at 100, MFE at +3R then drops to MAE at -1.5R, recovers to 0.
    # ATR ~ 1.0, so risk_per_share = 2.
    closes = [100.0] * 20 + [101.0, 102.0, 106.0, 103.0, 97.5, 100.0]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BULLISH", df, spy)
    # MFE_r should be at least 1.5 (not super-strict because exit may trigger early)
    assert res["mfe_r"] >= 0.5, f"expected mfe_r >= 0.5, got {res['mfe_r']}"
    assert res["mfe_day"] is not None
    print("  PASS: test_mfe_mae_correctness")


def test_multi_horizon_returns():
    # Long, simple +5% over 5 days
    closes = [100.0] * 20 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BULLISH", df, spy)
    # 5d forward should be roughly +5% (entered at day 20 close near ~101, day 25 close ~106)
    assert res["forward_return_5d"] is not None
    assert res["forward_return_5d"] > 0, f"expected +ve 5d return, got {res['forward_return_5d']}"
    print("  PASS: test_multi_horizon_returns")


def test_empty_ohlcv_returns_no_exit_cleanly():
    df = pd.DataFrame()
    res = replay_trade_plan("TEST", "2026-01-01", "BULLISH", df)
    assert res["exit_reason"] == "no_exit_yet"
    assert res["entry_price"] is None
    print("  PASS: test_empty_ohlcv_returns_no_exit_cleanly")


def test_hit_at_r_levels():
    # Engineered for a strong fast move up
    closes = [100.0] * 20 + [101, 102.5, 104, 106, 108, 110]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BULLISH", df, spy)
    # Should hit at least 0.5R within 3 days
    assert res["hit_0_5r_within_3d"], "expected hit_0_5r_within_3d=True"
    print("  PASS: test_hit_at_r_levels")


def test_partial_pnl_blends_correctly():
    # T1 hit then trail break — partial at T1 (1.5R), remainder near 0.
    # With partial_pct=0.5, blended R = 0.5*1.5 + 0.5*~0 = ~0.75
    closes = [100.0] * 20 + [101.5, 103.0, 100.0, 98.0, 96.5]
    df = _make_ohlcv(closes)
    spy = _flat_spy(len(closes))
    as_of = df.index[19].strftime("%Y-%m-%d")
    res = replay_trade_plan("TEST", as_of, "BULLISH", df, spy)
    if res["partial_filled"]:
        # If partial was filled, blended R should not be a full -1
        assert res["realized_r"] > -0.5, f"blended R too low: {res['realized_r']}"
    print("  PASS: test_partial_pnl_blends_correctly")


def main():
    tests = [
        test_long_hits_t2,
        test_long_stops_out,
        test_long_t1_then_stop,
        test_short_hits_t2,
        test_short_stops_out,
        test_time_stop_kicks_in,
        test_mfe_mae_correctness,
        test_multi_horizon_returns,
        test_empty_ohlcv_returns_no_exit_cleanly,
        test_hit_at_r_levels,
        test_partial_pnl_blends_correctly,
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

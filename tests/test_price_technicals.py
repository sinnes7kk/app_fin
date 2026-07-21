"""Tests for app/features/price_technicals.py.

Feeds synthetic candles with known properties and asserts the derived
features match hand-computed expectations: momentum sign, RSI extremes,
SMA distance, Bollinger z, relative volume, gap, and the Tier-2 beta /
residual-momentum / relative-strength signals.

Run:
    python -m pytest tests/test_price_technicals.py -v
    python tests/test_price_technicals.py            # standalone
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.features.price_technicals import (  # noqa: E402
    compute_price_technicals,
    sector_etf,
)


def _df_from_close(close: list[float], *, volume: list[float] | None = None,
                   opens: list[float] | None = None) -> pd.DataFrame:
    n = len(close)
    idx = pd.bdate_range(end="2026-06-30", periods=n)
    close = np.asarray(close, dtype=float)
    o = np.asarray(opens, dtype=float) if opens is not None else close
    return pd.DataFrame({
        "open": o,
        "high": np.maximum(o, close) * 1.001,
        "low": np.minimum(o, close) * 0.999,
        "close": close,
        "volume": (np.asarray(volume, dtype=float) if volume is not None
                   else np.full(n, 1_000_000.0)),
    }, index=idx)


def test_uptrend_positive_momentum_and_high_rsi():
    close = [100.0 * (1.005 ** i) for i in range(260)]
    out = compute_price_technicals(_df_from_close(close))
    assert out["ret_21d"] is not None and out["ret_21d"] > 0
    assert out["ret_63d"] > 0 and out["ret_126d"] > 0
    assert out["px_vs_sma50"] > 0 and out["px_vs_sma200"] > 0
    # Pure uptrend -> RSI pinned near 100.
    assert out["rsi_14"] is not None and out["rsi_14"] > 95
    # At/near the high -> dist_52w_high ~ 0 (slightly negative allowed).
    assert out["dist_52w_high"] is not None and out["dist_52w_high"] >= -1e-6


def test_downtrend_negative_momentum_low_rsi():
    close = [100.0 * (0.995 ** i) for i in range(260)]
    out = compute_price_technicals(_df_from_close(close))
    assert out["ret_21d"] < 0 and out["ret_63d"] < 0
    assert out["px_vs_sma50"] < 0
    assert out["rsi_14"] is not None and out["rsi_14"] < 5
    # Well below the trailing high.
    assert out["dist_52w_high"] < 0


def test_flat_series_neutral_rsi_and_no_bollinger():
    close = [100.0] * 60
    out = compute_price_technicals(_df_from_close(close))
    assert out["ret_21d"] == 0.0
    # No variance -> Bollinger z undefined.
    assert out["bollinger_z"] is None
    # Flat -> no gains or losses -> neutral RSI.
    assert out["rsi_14"] == 50.0


def test_relative_volume_spike():
    close = [100.0] * 40
    vol = [1_000_000.0] * 39 + [3_000_000.0]  # last bar 3x baseline
    out = compute_price_technicals(_df_from_close(close, volume=vol))
    assert out["rel_volume"] is not None
    assert abs(out["rel_volume"] - 3.0) < 1e-6


def test_gap_pct():
    close = [100.0] * 30
    opens = [100.0] * 29 + [105.0]  # gap up 5% on the last bar's open
    out = compute_price_technicals(_df_from_close(close, opens=opens))
    assert out["gap_pct"] is not None
    assert abs(out["gap_pct"] - 0.05) < 1e-6


def test_beta_and_residual_against_market():
    rng = np.random.default_rng(42)
    mkt_ret = rng.normal(0.0005, 0.01, 200)
    mkt_close = 100.0 * np.cumprod(1 + mkt_ret)
    # Ticker moves 2x the market, exactly -> beta ~ 2, residual ~ 0.
    tkr_close = 100.0 * np.cumprod(1 + 2 * mkt_ret)
    market_df = _df_from_close(list(mkt_close))
    tkr_df = _df_from_close(list(tkr_close))
    out = compute_price_technicals(tkr_df, market_df=market_df)
    assert out["beta_63d"] is not None
    assert abs(out["beta_63d"] - 2.0) < 0.15, out["beta_63d"]
    # rel_strength = ticker 63d return - market 63d return; ticker outruns.
    assert out["rel_strength_spy_63d"] is not None


def test_identical_to_market_zero_relative_strength():
    rng = np.random.default_rng(7)
    ret = rng.normal(0.0003, 0.008, 120)
    close = 100.0 * np.cumprod(1 + ret)
    df = _df_from_close(list(close))
    out = compute_price_technicals(df, market_df=_df_from_close(list(close)))
    assert out["rel_strength_spy_63d"] is not None
    assert abs(out["rel_strength_spy_63d"]) < 1e-6
    assert out["resid_mom_21d"] is not None
    assert abs(out["resid_mom_21d"]) < 1e-6
    assert abs(out["beta_63d"] - 1.0) < 1e-6


def test_as_of_slice_is_point_in_time():
    # 100 rising bars then a crash on the very last bar. If as_of excludes the
    # crash, momentum stays positive.
    close = [100.0 + i for i in range(100)] + [10.0]
    df = _df_from_close(close)
    as_of = df.index[-2]  # exclude the crash bar
    out = compute_price_technicals(df, as_of=as_of)
    assert out["ret_5d"] is not None and out["ret_5d"] > 0
    # Without the slice, the last bar is the crash -> negative 5d.
    out_full = compute_price_technicals(df)
    assert out_full["ret_5d"] < 0


def test_insufficient_history_returns_none():
    out = compute_price_technicals(_df_from_close([100.0, 101.0, 102.0]))
    assert out["ret_63d"] is None
    assert out["px_vs_sma200"] is None
    assert out["beta_63d"] is None


def test_none_input_returns_all_none():
    out = compute_price_technicals(None)
    assert all(v is None for v in out.values())


def test_sector_etf_mapping():
    assert sector_etf("Technology") == "XLK"
    assert sector_etf("Financial Services") == "XLF"
    assert sector_etf("Nonexistent") == "SPY"
    assert sector_etf(None) == "SPY"


if __name__ == "__main__":
    tests = [
        test_uptrend_positive_momentum_and_high_rsi,
        test_downtrend_negative_momentum_low_rsi,
        test_flat_series_neutral_rsi_and_no_bollinger,
        test_relative_volume_spike,
        test_gap_pct,
        test_beta_and_residual_against_market,
        test_identical_to_market_zero_relative_strength,
        test_as_of_slice_is_point_in_time,
        test_insufficient_history_returns_none,
        test_none_input_returns_all_none,
        test_sector_etf_mapping,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

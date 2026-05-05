"""Unit tests for app/signals/flow_promote.py.

Synthetic OHLCV is patched in via a monkey-patch on
``app.features.price_features`` so the tests don't hit yfinance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.signals.flow_promote import promote_flow_tracker_grade_a


def _make_ohlcv_with_features(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    """Build a synthetic OHLCV frame with atr14 / ema20 already computed.

    flow_promote.py reads atr14 and ema20 directly from the last row, so the
    test fixture pre-computes them rather than relying on
    ``compute_features``.
    """
    dates = pd.date_range(start=start, periods=len(closes), freq="B")
    rng = 1.0
    df = pd.DataFrame({
        "open": closes,
        "high": [c + rng for c in closes],
        "low": [c - rng for c in closes],
        "close": closes,
        "volume": 1_000_000,
    }, index=dates)
    df["atr14"] = rng * 1.5
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


def _patched_fetch_factory(closes: list[float] | None):
    def _fetch(ticker, lookback_days=120, include_partial=False):
        if closes is None:
            return None
        return _make_ohlcv_with_features(closes)
    return _fetch


def _patched_compute_features(df):
    """Pass-through for tests — fixture already has atr14/ema20."""
    return df


def _grade_a_long(ticker: str = "AAPL", score: float = 8.5) -> dict:
    return {
        "ticker": ticker,
        "direction": "BULLISH",
        "conviction_grade": "A",
        "conviction_score": score,
        "dominant_dte_bucket": "swing",
        "sector": "Technology",
    }


def test_disabled_flag_skips_promotion(monkeypatch):
    monkeypatch.setattr("app.signals.flow_promote.FLOW_PROMOTE_ENABLED", False)
    promoted, summary = promote_flow_tracker_grade_a([_grade_a_long()], [])
    assert promoted == []
    assert summary["feature_disabled"] is True
    print("  PASS: test_disabled_flag_skips_promotion")


def test_grade_a_promotes_when_above_ema20():
    # Trending up: EMA20 will be below the latest close.
    closes = list(np.linspace(80, 100, 40))
    fetch = _patched_fetch_factory(closes)

    with patch("app.signals.flow_promote.fetch_ohlcv", fetch), \
         patch("app.signals.flow_promote.compute_features", _patched_compute_features):
        promoted, summary = promote_flow_tracker_grade_a([_grade_a_long()], [])

    assert summary["promoted"] == 1, f"summary={summary}"
    assert len(promoted) == 1
    p = promoted[0]
    assert p["direction"] == "LONG"
    assert p["promoted_from_flow_tracker"] is True
    assert p["entry_price"] > 0 and p["stop_price"] < p["entry_price"]
    assert p["target_1"] > p["entry_price"]
    assert p["target_2"] > p["target_1"]
    assert p["source"] == "flow_promoted"
    print("  PASS: test_grade_a_promotes_when_above_ema20")


def test_grade_a_long_below_ema20_rejected_by_sanity():
    closes = list(np.linspace(120, 80, 40))
    fetch = _patched_fetch_factory(closes)

    with patch("app.signals.flow_promote.fetch_ohlcv", fetch), \
         patch("app.signals.flow_promote.compute_features", _patched_compute_features):
        promoted, summary = promote_flow_tracker_grade_a([_grade_a_long()], [])

    assert summary["promoted"] == 0
    assert summary["skipped_sanity_gate"] == 1
    assert any("ema20" in r[1] for r in summary["rejections"])
    print("  PASS: test_grade_a_long_below_ema20_rejected_by_sanity")


def test_score_below_floor_rejected():
    closes = list(np.linspace(80, 100, 40))
    fetch = _patched_fetch_factory(closes)
    g = _grade_a_long(score=7.5)  # below 8.0 floor

    with patch("app.signals.flow_promote.fetch_ohlcv", fetch), \
         patch("app.signals.flow_promote.compute_features", _patched_compute_features):
        promoted, summary = promote_flow_tracker_grade_a([g], [])

    assert summary["promoted"] == 0
    assert summary["skipped_sanity_gate"] == 1
    assert any("score_" in r[1] for r in summary["rejections"])
    print("  PASS: test_score_below_floor_rejected")


def test_existing_signal_not_double_promoted():
    closes = list(np.linspace(80, 100, 40))
    fetch = _patched_fetch_factory(closes)
    existing = [{"ticker": "AAPL", "direction": "LONG"}]

    with patch("app.signals.flow_promote.fetch_ohlcv", fetch), \
         patch("app.signals.flow_promote.compute_features", _patched_compute_features):
        promoted, summary = promote_flow_tracker_grade_a([_grade_a_long()], existing)

    assert summary["promoted"] == 0
    assert summary["skipped_already_signal"] == 1
    print("  PASS: test_existing_signal_not_double_promoted")


def test_grade_b_not_promoted():
    closes = list(np.linspace(80, 100, 40))
    fetch = _patched_fetch_factory(closes)
    g = _grade_a_long()
    g["conviction_grade"] = "B+"  # not promotable

    with patch("app.signals.flow_promote.fetch_ohlcv", fetch), \
         patch("app.signals.flow_promote.compute_features", _patched_compute_features):
        promoted, summary = promote_flow_tracker_grade_a([g], [])

    assert summary["promoted"] == 0
    assert summary["skipped_sanity_gate"] == 1
    print("  PASS: test_grade_b_not_promoted")


def test_no_ohlcv_skips_with_count():
    fetch = _patched_fetch_factory(None)  # always returns None

    with patch("app.signals.flow_promote.fetch_ohlcv", fetch), \
         patch("app.signals.flow_promote.compute_features", _patched_compute_features):
        promoted, summary = promote_flow_tracker_grade_a([_grade_a_long()], [])

    assert summary["promoted"] == 0
    assert summary["skipped_no_ohlcv"] == 1
    print("  PASS: test_no_ohlcv_skips_with_count")


def test_short_promotes_when_below_ema20():
    closes = list(np.linspace(120, 80, 40))
    fetch = _patched_fetch_factory(closes)
    g = _grade_a_long()
    g["direction"] = "BEARISH"

    with patch("app.signals.flow_promote.fetch_ohlcv", fetch), \
         patch("app.signals.flow_promote.compute_features", _patched_compute_features):
        promoted, summary = promote_flow_tracker_grade_a([g], [])

    assert summary["promoted"] == 1
    p = promoted[0]
    assert p["direction"] == "SHORT"
    assert p["stop_price"] > p["entry_price"]  # short stop is above entry
    assert p["target_1"] < p["entry_price"]
    print("  PASS: test_short_promotes_when_below_ema20")


def main():
    # Provide a no-op monkeypatch fixture for the standalone runner.
    class MP:
        def __init__(self): self._restores = []
        def setattr(self, target, value):
            mod_path, attr = target.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(mod_path)
            old = getattr(mod, attr)
            setattr(mod, attr, value)
            self._restores.append((mod, attr, old))
        def restore(self):
            for mod, attr, old in self._restores:
                setattr(mod, attr, old)
            self._restores = []

    tests = [
        (test_disabled_flag_skips_promotion, True),
        (test_grade_a_promotes_when_above_ema20, False),
        (test_grade_a_long_below_ema20_rejected_by_sanity, False),
        (test_score_below_floor_rejected, False),
        (test_existing_signal_not_double_promoted, False),
        (test_grade_b_not_promoted, False),
        (test_no_ohlcv_skips_with_count, False),
        (test_short_promotes_when_below_ema20, False),
    ]
    failures = 0
    for t, needs_mp in tests:
        try:
            if needs_mp:
                mp = MP()
                try:
                    t(mp)
                finally:
                    mp.restore()
            else:
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

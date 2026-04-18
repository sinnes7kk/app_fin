"""Tests for Wave 2 repeat-flow acceleration (2h-window counts + ratio).

Validates ``aggregate_flow_by_ticker`` correctly:

  * Counts prints inside the last 2h of the frame's most-recent event
    into ``bullish_repeat_2h`` / ``bearish_repeat_2h``.
  * Derives ``*_accel_ratio = repeat_2h / count`` per direction.
  * Clips ``*_accel_score = ratio - 0.308`` into [-0.5, +0.7].
  * Handles missing timestamps gracefully (no crash, 2h counts = 0).
  * Does not leak 2h counts across tickers.

Run with:
    python -m pytest tests/test_flow_accel.py -v
    python -m tests.test_flow_accel      # standalone
"""

from __future__ import annotations

import pandas as pd

from app.features.flow_features import aggregate_flow_by_ticker


def _row(
    ticker: str,
    *,
    ts: str,
    direction: str = "LONG",
    premium: float = 100_000.0,
    dte: int = 21,
    is_sweep: bool = False,
    direction_confidence: float = 1.0,
) -> dict:
    return {
        "ticker": ticker,
        "event_ts": ts,
        "direction": direction,
        "premium": premium,
        "dte": dte,
        "is_sweep": is_sweep,
        "direction_confidence": direction_confidence,
        "volume": 100,
        "open_interest": 50,
    }


def test_repeat_2h_counts_only_last_two_hours_of_prints():
    """Five prints across 6 hours: only the last two land inside the 2h
    window anchored on the most-recent event."""
    df = pd.DataFrame([
        _row("AAPL", ts="2025-01-10T14:00:00Z"),
        _row("AAPL", ts="2025-01-10T15:00:00Z"),
        _row("AAPL", ts="2025-01-10T17:00:00Z"),
        _row("AAPL", ts="2025-01-10T19:00:00Z"),  # T-1h
        _row("AAPL", ts="2025-01-10T20:00:00Z"),  # T
    ])
    out = aggregate_flow_by_ticker(df)
    row = out[out["ticker"] == "AAPL"].iloc[0]
    assert int(row["bullish_count"]) == 5
    assert int(row["bullish_repeat_2h"]) == 2
    # ratio = 2/5 = 0.4, score = 0.4 - 0.308 ≈ 0.092
    assert abs(float(row["bullish_accel_ratio"]) - 0.4) < 1e-6
    assert abs(float(row["bullish_accel_score"]) - 0.092) < 0.01


def test_accel_score_clipped_into_bounded_range():
    """10 prints all inside the 2h window → ratio = 1.0, score =
    1.0 - 0.308 = 0.692 (just under the 0.7 ceiling). Verifies the
    score is bounded and monotonically increasing with ratio."""
    rows = [
        _row("NVDA", ts=f"2025-01-10T19:{m:02d}:00Z")
        for m in range(5, 55, 5)  # 19:05 .. 19:50
    ]
    out = aggregate_flow_by_ticker(pd.DataFrame(rows))
    row = out[out["ticker"] == "NVDA"].iloc[0]
    assert int(row["bullish_count"]) == 10
    assert int(row["bullish_repeat_2h"]) == 10
    assert float(row["bullish_accel_ratio"]) == 1.0
    # score = ratio - 2/6.5 ≈ 0.692, within the [-0.5, +0.7] clip range.
    score = float(row["bullish_accel_score"])
    assert 0.68 <= score <= 0.7


def test_accel_ratio_directionally_separated():
    """Bullish 2h prints don't leak into bearish counts and vice versa."""
    df = pd.DataFrame([
        _row("SPY", ts="2025-01-10T10:00:00Z", direction="LONG"),
        _row("SPY", ts="2025-01-10T11:00:00Z", direction="LONG"),
        _row("SPY", ts="2025-01-10T19:30:00Z", direction="LONG"),   # in 2h window
        _row("SPY", ts="2025-01-10T20:00:00Z", direction="SHORT"),  # in 2h window
        _row("SPY", ts="2025-01-10T10:30:00Z", direction="SHORT"),  # out of window
    ])
    out = aggregate_flow_by_ticker(df)
    row = out[out["ticker"] == "SPY"].iloc[0]
    assert int(row["bullish_repeat_2h"]) == 1
    assert int(row["bearish_repeat_2h"]) == 1
    assert abs(float(row["bullish_accel_ratio"]) - 1 / 3) < 1e-6
    assert abs(float(row["bearish_accel_ratio"]) - 1 / 2) < 1e-6


def test_accel_ratio_zero_when_no_2h_window_coverage():
    """Six prints spread evenly across a 10-hour window → the last two
    hours only cover 2/10 ≈ 20%.  Score must be negative (fading)."""
    df = pd.DataFrame([
        _row("IWM", ts=f"2025-01-10T{10+h:02d}:00:00Z") for h in range(10)
    ])
    out = aggregate_flow_by_ticker(df)
    row = out[out["ticker"] == "IWM"].iloc[0]
    assert int(row["bullish_count"]) == 10
    # 20:00 anchor, 2h window = [18:00, 20:00].  Prints at 18, 19 are
    # inside (<=2h old); the 20:00 print is exactly at the anchor.
    assert int(row["bullish_repeat_2h"]) == 3
    assert abs(float(row["bullish_accel_ratio"]) - 0.3) < 1e-6
    # score = 0.3 - 0.308 ≈ -0.008 → steady-range
    assert float(row["bullish_accel_score"]) < 0


def test_missing_event_ts_does_not_crash():
    """When the flow payload lacks event_ts (shouldn't happen in prod but
    guards against vendor surprises), aggregation still runs and the
    2h counts all fall to zero."""
    df = pd.DataFrame([
        {**_row("TLT", ts="2025-01-10T14:00:00Z"), "event_ts": None},
        {**_row("TLT", ts="2025-01-10T15:00:00Z"), "event_ts": None},
    ])
    # Dropping the None event_ts column entirely:
    df = df.drop(columns=["event_ts"])
    out = aggregate_flow_by_ticker(df)
    row = out[out["ticker"] == "TLT"].iloc[0]
    assert int(row["bullish_count"]) == 2
    assert int(row["bullish_repeat_2h"]) == 0
    assert float(row["bullish_accel_ratio"]) == 0.0


def test_two_tickers_dont_cross_contaminate():
    """The 2h anchor is per-frame, but counts must still be grouped
    correctly per ticker without mixing signals."""
    df = pd.DataFrame([
        _row("AAPL", ts="2025-01-10T10:00:00Z"),
        _row("AAPL", ts="2025-01-10T19:30:00Z"),  # within 2h
        _row("MSFT", ts="2025-01-10T19:45:00Z"),  # within 2h; sets anchor
        _row("MSFT", ts="2025-01-10T20:00:00Z"),  # within 2h
    ])
    out = aggregate_flow_by_ticker(df)
    aapl = out[out["ticker"] == "AAPL"].iloc[0]
    msft = out[out["ticker"] == "MSFT"].iloc[0]
    assert int(aapl["bullish_repeat_2h"]) == 1
    assert int(msft["bullish_repeat_2h"]) == 2


if __name__ == "__main__":
    import traceback
    tests = [
        test_repeat_2h_counts_only_last_two_hours_of_prints,
        test_accel_score_clipped_into_bounded_range,
        test_accel_ratio_directionally_separated,
        test_accel_ratio_zero_when_no_2h_window_coverage,
        test_missing_event_ts_does_not_crash,
        test_two_tickers_dont_cross_contaminate,
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

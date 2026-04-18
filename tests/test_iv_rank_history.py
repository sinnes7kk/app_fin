"""Tests for app.features.iv_rank_history (Wave 2).

Validates:
  * ``record_iv_rank`` persists per (ticker, day) with last-write-wins.
  * ``compute_iv_rank_delta`` returns ``None`` delta when <2 samples in
    the lookback window, and a correct ``current - baseline`` otherwise.
  * Retention prunes samples older than ``IV_RANK_HISTORY_RETENTION_DAYS``.
  * NaN / ``None`` inputs are silently skipped so the pipeline never
    crashes on a bad IV fetch.

Run with:
    python -m pytest tests/test_iv_rank_history.py -v
    python -m tests.test_iv_rank_history     # standalone
"""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

from app.features import iv_rank_history as irh


def _fresh_store() -> Path:
    """Point the module at a throwaway JSON file for this test."""
    tmp = Path(tempfile.mkdtemp()) / "iv_rank_history.json"
    irh._override_history_path(tmp)
    return tmp


def test_record_and_load_single_sample():
    _fresh_store()
    irh.record_iv_rank("AAPL", 45.0, on=date(2025, 1, 10))

    current, delta, n = irh.compute_iv_rank_delta("AAPL", on=date(2025, 1, 10))
    assert current == 45.0
    assert delta is None   # need ≥2 samples for a delta.
    assert n == 1


def test_delta_uses_oldest_in_window_as_baseline():
    _fresh_store()
    irh.record_iv_rank("AAPL", 30.0, on=date(2025, 1, 6))
    irh.record_iv_rank("AAPL", 35.0, on=date(2025, 1, 8))
    irh.record_iv_rank("AAPL", 45.0, on=date(2025, 1, 10))

    # 5d window ending 2025-01-10 = cutoff 2025-01-05 → all three included.
    current, delta, n = irh.compute_iv_rank_delta(
        "AAPL", lookback_days=5, on=date(2025, 1, 10)
    )
    assert current == 45.0
    assert delta == 15.0   # 45 - 30
    assert n == 3


def test_delta_excludes_samples_outside_lookback_window():
    _fresh_store()
    # Ancient sample far outside the 5d lookback.
    irh.record_iv_rank("NVDA", 10.0, on=date(2025, 1, 1))
    # Fresh pair inside the window.
    irh.record_iv_rank("NVDA", 60.0, on=date(2025, 1, 9))
    irh.record_iv_rank("NVDA", 55.0, on=date(2025, 1, 10))

    current, delta, n = irh.compute_iv_rank_delta(
        "NVDA", lookback_days=5, on=date(2025, 1, 10)
    )
    assert current == 55.0
    # Baseline is the 60.0 sample from 01-09 (oldest inside window), NOT the
    # 10.0 from 01-01.
    assert delta == -5.0
    assert n == 2


def test_same_day_overwrite_keeps_latest_only():
    _fresh_store()
    irh.record_iv_rank("TSLA", 50.0, on=date(2025, 1, 10))
    irh.record_iv_rank("TSLA", 55.0, on=date(2025, 1, 10))  # overwrite
    irh.record_iv_rank("TSLA", 70.0, on=date(2025, 1, 12))

    current, delta, n = irh.compute_iv_rank_delta(
        "TSLA", lookback_days=5, on=date(2025, 1, 12)
    )
    assert current == 70.0
    assert delta == 15.0   # 70 - 55 (overwritten baseline), not 70 - 50
    assert n == 2


def test_none_and_nan_inputs_are_noop():
    _fresh_store()
    irh.record_iv_rank("IBM", None, on=date(2025, 1, 10))
    irh.record_iv_rank("IBM", float("nan"), on=date(2025, 1, 10))
    irh.record_iv_rank("IBM", "garbage", on=date(2025, 1, 10))  # type: ignore[arg-type]

    current, delta, n = irh.compute_iv_rank_delta("IBM", on=date(2025, 1, 10))
    assert current is None and delta is None and n == 0


def test_retention_prunes_samples_older_than_retention_days():
    _fresh_store()
    today = date(2025, 2, 15)
    # Old sample well outside retention (30d).
    irh.record_iv_rank("MSFT", 20.0, on=today - timedelta(days=60))
    # Fresh sample inside retention.
    irh.record_iv_rank("MSFT", 40.0, on=today)

    history = irh._load()  # internal but ok for a unit test
    samples = history.get("MSFT", [])
    # The 60-day-old one must have been pruned on write; only the fresh one remains.
    assert len(samples) == 1
    assert samples[0]["iv_rank"] == 40.0


def test_delta_returns_all_none_for_unknown_ticker():
    _fresh_store()
    current, delta, n = irh.compute_iv_rank_delta("ZZZZ")
    assert current is None and delta is None and n == 0


if __name__ == "__main__":
    import traceback
    tests = [
        test_record_and_load_single_sample,
        test_delta_uses_oldest_in_window_as_baseline,
        test_delta_excludes_samples_outside_lookback_window,
        test_same_day_overwrite_keeps_latest_only,
        test_none_and_nan_inputs_are_noop,
        test_retention_prunes_samples_older_than_retention_days,
        test_delta_returns_all_none_for_unknown_ticker,
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
        finally:
            irh._reset_history_path()
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)

"""Tests for ``compute_multi_day_flow`` point-in-time replay (Grade
Backtest A2).

The new ``as_of`` + ``snapshots_path`` kwargs let the backtest replay
grades against a fixed historical date without monkeypatching
``date.today()`` or copying the snapshots CSV to a tmp file.

This test verifies:
  1. ``snapshots_path`` reads from a custom path (incl. gzip).
  2. ``as_of`` drops rows strictly after the given date.
  3. Different ``as_of`` values produce different windows
     deterministically.
  4. A missing path returns an empty list (doesn't raise).

Run with:

    python -m pytest tests/test_multi_day_flow_as_of.py -v
    python tests/test_multi_day_flow_as_of.py           # standalone
"""

from __future__ import annotations

import csv
import gzip
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _write_fixture(path: Path, rows: list[dict], gzipped: bool = False) -> None:
    """Write ``rows`` to ``path`` with the full ``SNAPSHOT_COLS`` header."""
    from app.features.flow_tracker import SNAPSHOT_COLS

    path.parent.mkdir(parents=True, exist_ok=True)
    if gzipped:
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _ticker_day_row(ticker: str, iso_date: str, premium: float = 2_000_000) -> dict:
    """Minimal row with enough fields to pass the min-premium / min-mcap
    gates in ``compute_multi_day_flow``."""
    return {
        "snapshot_date": iso_date,
        "ticker": ticker,
        "sector": "Technology",
        "close": 100.0,
        "marketcap": 5_000_000_000.0,
        "bullish_premium": premium,
        "bearish_premium": 100_000.0,
        "total_bullish_premium": premium,
        "total_bearish_premium": 100_000.0,
        "net_premium": premium - 100_000.0,
        "iv_rank": 40.0,
        "iv30": 0.30,
        "total_oi_change_perc": 5.0,
        "call_volume": 10_000,
        "put_volume": 2_000,
        "volume": 12_000,
        "call_open_interest": 100_000,
        "put_open_interest": 20_000,
        "put_call_ratio": 0.2,
        "perc_3_day_total": 80.0,
        "perc_30_day_total": 85.0,
        "sweep_share": 0.5,
        "multileg_share": 0.1,
    }


def _days_before(iso_today: str, n: int) -> str:
    return (date.fromisoformat(iso_today) - timedelta(days=n)).isoformat()


def test_missing_snapshots_path_returns_empty():
    from app.features.flow_tracker import compute_multi_day_flow

    nowhere = Path(tempfile.mkdtemp()) / "does_not_exist.csv"
    result = compute_multi_day_flow(snapshots_path=nowhere)
    assert result == [], "missing path must return [] (not raise)"


def test_snapshots_path_kwarg_reads_custom_path_no_monkeypatch():
    """The kwarg replaces the old monkeypatch dance.  Passing a fixture
    path should read from it without touching the module-level
    ``SNAPSHOTS_PATH``."""
    from app.features.flow_tracker import SNAPSHOTS_PATH, compute_multi_day_flow

    tmp = Path(tempfile.mkdtemp()) / "fixture.csv"
    as_of = "2025-01-15"
    rows = [
        _ticker_day_row("AAAA", _days_before(as_of, i))
        for i in range(5)
    ]
    _write_fixture(tmp, rows)

    # Call with an explicit path + as_of so today's wall clock doesn't
    # push the cutoff window past the fixture data.
    result = compute_multi_day_flow(
        as_of=as_of,
        snapshots_path=tmp,
        min_active_days=3,
    )
    # Module-level SNAPSHOTS_PATH must not be touched by the call.
    assert SNAPSHOTS_PATH is not tmp, (
        "kwarg must not mutate module-level SNAPSHOTS_PATH"
    )
    # The fixture ticker should come through.
    tickers = {r["ticker"] for r in result}
    assert "AAAA" in tickers, f"expected AAAA in result; got {tickers}"


def test_as_of_drops_rows_strictly_in_the_future():
    """Point-in-time replay: rows stamped after ``as_of`` must be
    ignored, even if they sit in the same fixture file."""
    from app.features.flow_tracker import compute_multi_day_flow

    tmp = Path(tempfile.mkdtemp()) / "fixture.csv"
    as_of = "2025-01-15"

    past_rows = [
        _ticker_day_row("BBBB", _days_before(as_of, i))
        for i in range(5)
    ]
    future_rows = [
        _ticker_day_row("ZZZZ", (date.fromisoformat(as_of) + timedelta(days=i)).isoformat())
        for i in range(1, 4)
    ]
    _write_fixture(tmp, past_rows + future_rows)

    result = compute_multi_day_flow(
        as_of=as_of,
        snapshots_path=tmp,
        min_active_days=3,
    )
    tickers = {r["ticker"] for r in result}
    assert "BBBB" in tickers, "past ticker dropped unexpectedly"
    assert "ZZZZ" not in tickers, (
        "future-dated row leaked through as_of filter — point-in-time "
        "replay is broken"
    )


def test_gzipped_archive_path_is_transparently_readable():
    """The backtest points at ``snapshots_archive.csv.gz``; pandas
    handles the `.gz` extension natively, so the same kwarg should
    work with the gzipped archive."""
    from app.features.flow_tracker import compute_multi_day_flow

    tmp = Path(tempfile.mkdtemp()) / "fixture.csv.gz"
    as_of = "2025-01-15"
    rows = [
        _ticker_day_row("CCCC", _days_before(as_of, i))
        for i in range(5)
    ]
    _write_fixture(tmp, rows, gzipped=True)

    result = compute_multi_day_flow(
        as_of=as_of,
        snapshots_path=tmp,
        min_active_days=3,
    )
    tickers = {r["ticker"] for r in result}
    assert "CCCC" in tickers, (
        f"gzip archive read failed — got {tickers}"
    )


def test_as_of_shifts_window_deterministically():
    """Calling with the same fixture but different ``as_of`` values
    must produce different windows.  We seed a ticker only at the
    recent tail and verify it falls *out* of the window when ``as_of``
    is rolled back."""
    from app.features.flow_tracker import compute_multi_day_flow

    tmp = Path(tempfile.mkdtemp()) / "fixture.csv"
    late = "2025-01-15"
    early = "2025-01-01"

    rows = (
        # 5 days of LATE data around `late`
        [_ticker_day_row("LATE", _days_before(late, i)) for i in range(5)]
        # 5 days of EARLY data around `early`
        + [_ticker_day_row("EARLY", _days_before(early, i)) for i in range(5)]
    )
    _write_fixture(tmp, rows)

    late_result = compute_multi_day_flow(
        as_of=late, snapshots_path=tmp, min_active_days=3, lookback_days=5,
    )
    early_result = compute_multi_day_flow(
        as_of=early, snapshots_path=tmp, min_active_days=3, lookback_days=5,
    )
    late_tickers = {r["ticker"] for r in late_result}
    early_tickers = {r["ticker"] for r in early_result}

    assert "LATE" in late_tickers, f"LATE missing on as_of={late}: {late_tickers}"
    assert "EARLY" in early_tickers, f"EARLY missing on as_of={early}: {early_tickers}"
    assert "LATE" not in early_tickers, (
        f"LATE ticker leaked into early-as_of result — cutoff broken: {early_tickers}"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_missing_snapshots_path_returns_empty,
        test_snapshots_path_kwarg_reads_custom_path_no_monkeypatch,
        test_as_of_drops_rows_strictly_in_the_future,
        test_gzipped_archive_path_is_transparently_readable,
        test_as_of_shifts_window_deterministically,
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

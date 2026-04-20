"""Tests for the append-only snapshot archive (Grade Backtest A1).

The archive lives at ``data/snapshots_archive.csv.gz`` and is written
alongside the hot CSV on every ``save_screener_snapshot`` /
``save_flow_feature_snapshot`` call.  It must be:

  1. Gzipped.
  2. Strictly append-only — new rows never overwrite existing rows.
  3. Header written once on first create.
  4. Tolerant of the hot CSV being pruned (archive is independent).

Run with:

    python -m pytest tests/test_snapshots_archive.py -v
    python tests/test_snapshots_archive.py           # standalone
"""

from __future__ import annotations

import csv
import gzip
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _screener_row(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "bullish_premium": 1_000_000.0,
        "bearish_premium": 500_000.0,
        "marketcap": 1_000_000_000.0,
        "volume": 50_000,
        "close": 150.0,
        "call_volume": 30_000,
        "put_volume": 20_000,
        "call_open_interest": 100_000,
        "put_open_interest": 80_000,
        "iv30": 0.30,
        "iv_rank": 45.0,
        "net_premium": 500_000.0,
    }


def _read_archive(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def test_archive_writes_gzip_and_creates_header_once():
    """First write should create a gzip file with a header; second
    write should append rows without duplicating the header."""
    from app.features import flow_tracker as ft

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_hot = tmp_dir / "screener_snapshots.csv"
    tmp_archive = tmp_dir / "snapshots_archive.csv.gz"

    with patch.object(ft, "SNAPSHOTS_PATH", tmp_hot), \
         patch.object(ft, "SNAPSHOTS_ARCHIVE_PATH", tmp_archive):
        ft.save_screener_snapshot([_screener_row("AAPL")])
        # Second write on a later trading day simulates two scans.
        # We can't actually travel in time from here, so we just call
        # again with a different ticker and rely on the writer merging
        # same-day rows in the hot CSV while the archive is strictly
        # additive.
        ft.save_screener_snapshot([_screener_row("MSFT")])

    assert tmp_archive.exists(), "archive file never created"

    # Raw gzip sanity — readable as gzip, not plain CSV.
    with gzip.open(tmp_archive, "rt") as f:
        raw = f.read()
    assert raw.startswith("snapshot_date,"), "header missing or malformed"
    assert raw.count("snapshot_date,ticker") == 1, (
        "header written more than once — archive must only write header on "
        "first create"
    )

    rows = _read_archive(tmp_archive)
    tickers = {r.get("ticker") for r in rows}
    assert "AAPL" in tickers, "first-write ticker missing from archive"
    assert "MSFT" in tickers, "second-write ticker missing from archive"


def test_archive_is_append_only():
    """Later writes must never shrink or overwrite the archive.

    The hot CSV is self-pruning; the archive is not.  This is the
    core guarantee the backtest depends on.
    """
    from app.features import flow_tracker as ft

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_hot = tmp_dir / "screener_snapshots.csv"
    tmp_archive = tmp_dir / "snapshots_archive.csv.gz"

    with patch.object(ft, "SNAPSHOTS_PATH", tmp_hot), \
         patch.object(ft, "SNAPSHOTS_ARCHIVE_PATH", tmp_archive):
        ft.save_screener_snapshot([_screener_row("AAPL")])
        n1 = len(_read_archive(tmp_archive))

        ft.save_screener_snapshot([_screener_row("MSFT"), _screener_row("GOOG")])
        n2 = len(_read_archive(tmp_archive))

    assert n2 >= n1 + 2, (
        f"archive row count shrank or failed to grow: {n1} -> {n2}"
    )


def test_archive_gzip_roundtrip_via_pandas():
    """pandas.read_csv transparently handles .csv.gz — the backtest
    relies on this, so we verify the roundtrip explicitly."""
    from app.features import flow_tracker as ft

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_hot = tmp_dir / "screener_snapshots.csv"
    tmp_archive = tmp_dir / "snapshots_archive.csv.gz"

    with patch.object(ft, "SNAPSHOTS_PATH", tmp_hot), \
         patch.object(ft, "SNAPSHOTS_ARCHIVE_PATH", tmp_archive):
        ft.save_screener_snapshot([_screener_row("AAPL"), _screener_row("MSFT")])

    df = pd.read_csv(tmp_archive)
    assert not df.empty, "pandas could not read the gzipped archive"
    assert "ticker" in df.columns, "ticker column missing from archive"
    assert "snapshot_date" in df.columns, "snapshot_date missing"
    assert {"AAPL", "MSFT"}.issubset(set(df["ticker"].astype(str).unique()))


def test_append_helper_tolerates_empty_rows():
    """Calling the helper with an empty list is a no-op (no file
    created, no exception)."""
    from app.features import flow_tracker as ft

    tmp_archive = Path(tempfile.mkdtemp()) / "snapshots_archive.csv.gz"
    with patch.object(ft, "SNAPSHOTS_ARCHIVE_PATH", tmp_archive):
        ft._append_rows_to_archive([])
    assert not tmp_archive.exists(), "empty append should not create file"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_archive_writes_gzip_and_creates_header_once,
        test_archive_is_append_only,
        test_archive_gzip_roundtrip_via_pandas,
        test_append_helper_tolerates_empty_rows,
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

"""Tests for the Flow Tracker snapshot-writer call order.

Week-1 Signal Quality Batch bug: ``save_screener_snapshot`` rewrites the
entire today-slice of ``screener_snapshots.csv`` (drops all rows where
``snapshot_date == today`` then re-writes only screener rows).  If the
pipeline calls ``save_flow_feature_snapshot`` *before*
``save_screener_snapshot``, the latter silently wipes the flow-feature
gap-filler rows that fill in tickers missing from UW's screener.

The pipeline must call ``save_screener_snapshot`` first, then
``save_flow_feature_snapshot`` second — the gap-filler writer already
skips tickers already present for today, so the ordered pair yields
"screener rows + non-overlapping flow-feature gap-fillers" as intended.

Run with:

    python -m pytest tests/test_snapshot_writer_order.py -v
    python tests/test_snapshot_writer_order.py           # standalone
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore


def _read_today_tickers(csv_path: Path, today_str: str) -> set[str]:
    """Return the set of tickers present for today in the snapshot CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return {
            (r.get("ticker") or "").upper().strip()
            for r in reader
            if r.get("snapshot_date") == today_str
        }


def _screener_row(ticker: str) -> dict:
    """Minimal UW screener row shape consumed by save_screener_snapshot."""
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


def _feature_row(ticker: str) -> dict:
    """Minimal flow-feature row shape consumed by save_flow_feature_snapshot."""
    return {
        "ticker": ticker,
        "marketcap": 5_000_000_000.0,
        "dominant_dte_bucket": "31-90",
        "sweep_share": 0.55,
        "multileg_share": 0.10,
        "bullish_accel_ratio": 1.2,
        "bearish_accel_ratio": 0.8,
        "total_count": 42,
    }


def _fake_ticker_options_snapshot(ticker: str) -> dict:
    """Fake /stock/{t}/options-volume + friends response."""
    return {
        "bullish_premium": 800_000.0,
        "bearish_premium": 200_000.0,
        "marketcap": 5_000_000_000.0,
        "volume": 30_000,
        "close": 250.0,
        "call_volume": 20_000,
        "put_volume": 10_000,
        "call_open_interest": 60_000,
        "put_open_interest": 40_000,
        "iv30": 0.35,
        "iv_rank": 55.0,
        "net_premium": 600_000.0,
    }


def test_screener_then_flow_feature_preserves_gap_tickers(tmp_path=None):
    """Happy path: calling save_screener_snapshot FIRST, then
    save_flow_feature_snapshot, must yield both screener tickers AND
    flow-feature-only gap tickers in today's slice.

    This is the bug-fix contract — if the pipeline ever regresses to the
    reversed order, this test fails because save_screener_snapshot wipes
    the gap-filler rows written by save_flow_feature_snapshot.
    """
    from app.features import flow_tracker as ft

    tmp_csv = Path(tempfile.mkdtemp()) / "screener_snapshots.csv"

    screener_rows = [_screener_row("AAPL"), _screener_row("MSFT")]
    feature_table = pd.DataFrame(
        [
            _feature_row("AAPL"),       # overlap — should stay as screener row
            _feature_row("MSFT"),       # overlap — should stay as screener row
            _feature_row("SOFI"),       # gap — should be added by flow-feature writer
            _feature_row("PLTR"),       # gap — should be added by flow-feature writer
        ]
    )

    today_str = ft.current_trading_day().isoformat()

    with patch.object(ft, "SNAPSHOTS_PATH", tmp_csv):
        with patch(
            "app.vendors.unusual_whales.fetch_ticker_options_snapshot",
            side_effect=_fake_ticker_options_snapshot,
        ):
            ft.save_screener_snapshot(screener_rows)
            ft.save_flow_feature_snapshot(feature_table)

    tickers = _read_today_tickers(tmp_csv, today_str)
    assert "AAPL" in tickers, "screener ticker AAPL missing after correct-order writes"
    assert "MSFT" in tickers, "screener ticker MSFT missing after correct-order writes"
    assert "SOFI" in tickers, (
        "flow-feature gap-filler SOFI missing — save_screener_snapshot wiped it"
    )
    assert "PLTR" in tickers, (
        "flow-feature gap-filler PLTR missing — save_screener_snapshot wiped it"
    )


def test_reversed_order_wipes_gap_tickers(tmp_path=None):
    """Regression guard: demonstrate that the BROKEN order (feature
    first, screener second) wipes the gap-fillers.  This test exists so
    anyone re-introducing the bug can see exactly what it does."""
    from app.features import flow_tracker as ft

    tmp_csv = Path(tempfile.mkdtemp()) / "screener_snapshots.csv"

    screener_rows = [_screener_row("AAPL")]
    feature_table = pd.DataFrame([_feature_row("AAPL"), _feature_row("SOFI")])

    today_str = ft.current_trading_day().isoformat()

    with patch.object(ft, "SNAPSHOTS_PATH", tmp_csv):
        with patch(
            "app.vendors.unusual_whales.fetch_ticker_options_snapshot",
            side_effect=_fake_ticker_options_snapshot,
        ):
            # BROKEN ORDER — flow-feature gap-filler gets wiped
            ft.save_flow_feature_snapshot(feature_table)
            ft.save_screener_snapshot(screener_rows)

    tickers = _read_today_tickers(tmp_csv, today_str)
    assert "AAPL" in tickers
    assert "SOFI" not in tickers, (
        "Regression guard failed: the broken order should wipe SOFI — if "
        "SOFI is present, save_screener_snapshot's replace-today logic has "
        "been changed and this test is no longer meaningful."
    )


def test_correct_order_preserves_historical_rows():
    """Rows from prior dates must survive both writes untouched."""
    from app.features import flow_tracker as ft

    tmp_csv = Path(tempfile.mkdtemp()) / "screener_snapshots.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)

    # Seed with a historical row (yesterday)
    today = ft.current_trading_day()
    yesterday_str = (today.replace(day=max(today.day - 1, 1))).isoformat()
    # Use a safely-old date that survives the retention cutoff
    from datetime import timedelta
    yesterday_str = (today - timedelta(days=1)).isoformat()

    with open(tmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ft.SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(
            {
                "snapshot_date": yesterday_str,
                "ticker": "LEGACY",
                "source": "screener",
                "total_bullish_premium": 100.0,
                "total_bearish_premium": 50.0,
            }
        )

    screener_rows = [_screener_row("AAPL")]
    feature_table = pd.DataFrame([_feature_row("SOFI")])

    with patch.object(ft, "SNAPSHOTS_PATH", tmp_csv):
        with patch(
            "app.vendors.unusual_whales.fetch_ticker_options_snapshot",
            side_effect=_fake_ticker_options_snapshot,
        ):
            ft.save_screener_snapshot(screener_rows)
            ft.save_flow_feature_snapshot(feature_table)

    with open(tmp_csv, "r", newline="") as f:
        all_rows = list(csv.DictReader(f))

    dates = {r.get("snapshot_date") for r in all_rows}
    tickers_by_date = {
        d: {r.get("ticker") for r in all_rows if r.get("snapshot_date") == d}
        for d in dates
    }
    assert yesterday_str in dates, "historical date was wiped"
    assert "LEGACY" in tickers_by_date[yesterday_str], "historical LEGACY row wiped"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_screener_then_flow_feature_preserves_gap_tickers,
        test_reversed_order_wipes_gap_tickers,
        test_correct_order_preserves_historical_rows,
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

"""Tests for ``_flow_tracker_distinct_days`` helper in ``app.web.server``.

The helper drives the per-horizon empty-state copy in ``index.html``:
when the 15d view (``min_active_days=4``) is empty but the user only
has 2 closed-session days of screener history, we want to tell them
*why* rather than a generic "not enough data" blurb.

Covers:
  - Missing CSV                 → 0
  - Empty CSV (header only)     → 0
  - Malformed CSV (no column)   → 0 (graceful handling)
  - 1 distinct snapshot_date    → 1
  - 3 distinct snapshot_dates   → 3 (duplicates collapsed)

Run with either:
    python -m pytest tests/test_flow_tracker_distinct_days.py -v
    python -m tests.test_flow_tracker_distinct_days   # standalone
"""

from __future__ import annotations

import csv
import tempfile
import traceback
from pathlib import Path

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, dates: list[str]) -> None:
    """Write a minimal screener_snapshots.csv with one row per date."""
    from app.features.flow_tracker import SNAPSHOT_COLS

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        for i, d in enumerate(dates):
            writer.writerow({
                "snapshot_date": d,
                "ticker": f"TKR{i}",
                "sector": "Tech",
                "close": 100.0,
                "marketcap": 1_000_000_000,
            })


def _patch_snapshots_path(monkeypatch_path: Path):
    """Point ``app.features.flow_tracker.SNAPSHOTS_PATH`` at a temp file.

    The helper re-imports the symbol inside its body (for test isolation),
    so we have to mutate the module attribute itself rather than a local.
    """
    import app.features.flow_tracker as ft_mod

    original = ft_mod.SNAPSHOTS_PATH
    ft_mod.SNAPSHOTS_PATH = monkeypatch_path
    return original


def _restore_snapshots_path(original: Path) -> None:
    import app.features.flow_tracker as ft_mod

    ft_mod.SNAPSHOTS_PATH = original


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_csv_returns_zero() -> None:
    from app.web.server import _flow_tracker_distinct_days

    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does_not_exist.csv"
        original = _patch_snapshots_path(missing)
        try:
            assert _flow_tracker_distinct_days() == 0
        finally:
            _restore_snapshots_path(original)


def test_empty_csv_header_only_returns_zero() -> None:
    from app.web.server import _flow_tracker_distinct_days

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "screener_snapshots.csv"
        _write_csv(path, dates=[])
        original = _patch_snapshots_path(path)
        try:
            assert _flow_tracker_distinct_days() == 0
        finally:
            _restore_snapshots_path(original)


def test_malformed_csv_without_column_returns_zero() -> None:
    """CSV missing the ``snapshot_date`` column should not raise."""
    from app.web.server import _flow_tracker_distinct_days

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "screener_snapshots.csv"
        path.write_text("ticker,close\nAAPL,100.0\nMSFT,200.0\n")
        original = _patch_snapshots_path(path)
        try:
            assert _flow_tracker_distinct_days() == 0
        finally:
            _restore_snapshots_path(original)


def test_single_day_returns_one() -> None:
    from app.web.server import _flow_tracker_distinct_days

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "screener_snapshots.csv"
        _write_csv(path, dates=["2026-04-17"])
        original = _patch_snapshots_path(path)
        try:
            assert _flow_tracker_distinct_days() == 1
        finally:
            _restore_snapshots_path(original)


def test_three_distinct_days_returns_three_even_with_dupes() -> None:
    """Duplicate dates must collapse — we count *distinct* trading days."""
    from app.web.server import _flow_tracker_distinct_days

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "screener_snapshots.csv"
        _write_csv(
            path,
            dates=[
                "2026-04-15",
                "2026-04-15",  # dup
                "2026-04-16",
                "2026-04-17",
                "2026-04-17",  # dup
            ],
        )
        original = _patch_snapshots_path(path)
        try:
            assert _flow_tracker_distinct_days() == 3
        finally:
            _restore_snapshots_path(original)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest required).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_missing_csv_returns_zero,
        test_empty_csv_header_only_returns_zero,
        test_malformed_csv_without_column_returns_zero,
        test_single_day_returns_one,
        test_three_distinct_days_returns_three_even_with_dupes,
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
    raise SystemExit(0 if failed == 0 else 1)

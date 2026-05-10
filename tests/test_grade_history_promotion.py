"""Tests for ``stamp_promotion_outcomes`` in
``app/analytics/grade_history.py``.

Verifies that:
  - promoted rows get is_promoted='true' / blank reject_reason
  - rejected rows get is_promoted='false' / matching reject_reason
  - rows in neither list get is_promoted='false' / 'not_evaluated'
  - the helper is idempotent on the same as_of
  - the new HISTORY_COLS schema includes is_promoted + reject_reason
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.analytics.grade_history as gh  # noqa: E402


def _seed_history(tmp_path: Path, as_of: str) -> Path:
    """Write a synthetic grade_history.csv with three rows for a target
    as_of plus one row for an older date that should NOT be touched.
    """
    p = tmp_path / "grade_history.csv"
    rows = [
        # Older row — must be left alone by the stamp.
        {"as_of": "2026-04-01", "ticker": "OLD", "direction": "BULLISH",
         "is_promoted": "true", "reject_reason": ""},
        # Today's rows — to be stamped.
        {"as_of": as_of, "ticker": "AAPL", "direction": "BULLISH",
         "is_promoted": "", "reject_reason": ""},
        {"as_of": as_of, "ticker": "NVDA", "direction": "BULLISH",
         "is_promoted": "", "reject_reason": ""},
        {"as_of": as_of, "ticker": "TSLA", "direction": "BEARISH",
         "is_promoted": "", "reject_reason": ""},
    ]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gh.HISTORY_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return p


def _read(path: Path) -> list[dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def test_history_cols_include_promotion_fields():
    assert "is_promoted" in gh.HISTORY_COLS
    assert "reject_reason" in gh.HISTORY_COLS
    print("  PASS: test_history_cols_include_promotion_fields")


def test_stamp_promoted_and_rejected(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _seed_history(tmp, "2026-05-08")
        monkeypatch.setattr(gh, "GRADE_HISTORY_PATH", path)

        n = gh.stamp_promotion_outcomes(
            as_of="2026-05-08",
            promoted=[
                {"ticker": "aapl", "direction": "bullish"},  # case-insensitive
            ],
            rejected=[
                {"ticker": "TSLA", "direction": "BEARISH",
                 "reject_reason": "weak_relative_strength"},
            ],
        )
        assert n == 3, f"stamped {n}, expected 3"

        rows = _read(path)
        by_key = {(r["ticker"], r["direction"]): r for r in rows}
        assert by_key[("OLD", "BULLISH")]["is_promoted"] == "true", \
            "older row must not be re-stamped"
        assert by_key[("AAPL", "BULLISH")]["is_promoted"] == "true"
        assert by_key[("AAPL", "BULLISH")]["reject_reason"] == ""
        assert by_key[("TSLA", "BEARISH")]["is_promoted"] == "false"
        assert by_key[("TSLA", "BEARISH")]["reject_reason"] == "weak_relative_strength"
        # NVDA is in neither promoted nor rejected → not_evaluated
        assert by_key[("NVDA", "BULLISH")]["is_promoted"] == "false"
        assert by_key[("NVDA", "BULLISH")]["reject_reason"] == "not_evaluated"
    print("  PASS: test_stamp_promoted_and_rejected")


def test_stamp_idempotent(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _seed_history(tmp, "2026-05-08")
        monkeypatch.setattr(gh, "GRADE_HISTORY_PATH", path)

        promoted = [{"ticker": "AAPL", "direction": "BULLISH"}]
        rejected = [{"ticker": "TSLA", "direction": "BEARISH",
                     "reject_reason": "poor_rr"}]
        gh.stamp_promotion_outcomes("2026-05-08", promoted, rejected)
        first = _read(path)
        gh.stamp_promotion_outcomes("2026-05-08", promoted, rejected)
        second = _read(path)
        assert first == second, "second stamp produced different rows"
    print("  PASS: test_stamp_idempotent")


def test_stamp_overwrites_with_new_inputs(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _seed_history(tmp, "2026-05-08")
        monkeypatch.setattr(gh, "GRADE_HISTORY_PATH", path)

        # First pass — AAPL promoted.
        gh.stamp_promotion_outcomes(
            "2026-05-08",
            promoted=[{"ticker": "AAPL", "direction": "BULLISH"}],
            rejected=[],
        )
        # Second pass — AAPL now appears as rejected. The latest stamp
        # wins (matching the hourly-scan behaviour where re-runs of
        # the day re-evaluate).
        gh.stamp_promotion_outcomes(
            "2026-05-08",
            promoted=[],
            rejected=[{"ticker": "AAPL", "direction": "BULLISH",
                       "reject_reason": "wall_proximity"}],
        )
        rows = _read(path)
        aapl = next(r for r in rows if r["ticker"] == "AAPL")
        assert aapl["is_promoted"] == "false"
        assert aapl["reject_reason"] == "wall_proximity"
    print("  PASS: test_stamp_overwrites_with_new_inputs")


def test_stamp_normalizes_long_to_bullish(monkeypatch):
    """Production reality check: ``final_results`` rows from the signal
    pipeline carry ``direction='LONG'`` while grade_history rows carry
    ``direction='BULLISH'``. Without normalization at the join boundary
    every promoted row falls through to ``not_evaluated``. This test
    locks the LONG→BULLISH alias in so the bug cannot regress.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _seed_history(tmp, "2026-05-08")
        monkeypatch.setattr(gh, "GRADE_HISTORY_PATH", path)

        gh.stamp_promotion_outcomes(
            as_of="2026-05-08",
            promoted=[
                # Pipeline-native vocabulary — LONG must map to BULLISH.
                {"ticker": "AAPL", "direction": "LONG"},
            ],
            rejected=[],
        )
        rows = _read(path)
        aapl = next(r for r in rows if r["ticker"] == "AAPL")
        assert aapl["is_promoted"] == "true", \
            f"LONG must alias to BULLISH; got is_promoted={aapl['is_promoted']!r}"
        assert aapl["reject_reason"] == ""
        # Idempotence: NVDA is still not in any list → not_evaluated.
        nvda = next(r for r in rows if r["ticker"] == "NVDA")
        assert nvda["reject_reason"] == "not_evaluated"
    print("  PASS: test_stamp_normalizes_long_to_bullish")


def test_stamp_normalizes_short_to_bearish(monkeypatch):
    """Symmetric check for SHORT→BEARISH on the rejection path."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = _seed_history(tmp, "2026-05-08")
        monkeypatch.setattr(gh, "GRADE_HISTORY_PATH", path)

        gh.stamp_promotion_outcomes(
            as_of="2026-05-08",
            promoted=[],
            rejected=[
                # SHORT must alias to BEARISH on the rejection lookup.
                {"ticker": "TSLA", "direction": "SHORT",
                 "reject_reason": "weak_bearish_flow"},
            ],
        )
        rows = _read(path)
        tsla = next(r for r in rows if r["ticker"] == "TSLA")
        assert tsla["is_promoted"] == "false"
        assert tsla["reject_reason"] == "weak_bearish_flow", \
            f"SHORT must alias to BEARISH; got reject_reason={tsla['reject_reason']!r}"
    print("  PASS: test_stamp_normalizes_short_to_bearish")


# Minimal monkeypatch shim so we don't require pytest as a dependency.
class _Monkeypatch:
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        for t, n, v in reversed(self._undo):
            setattr(t, n, v)


def main():
    tests = [
        ("test_history_cols_include_promotion_fields", test_history_cols_include_promotion_fields, False),
        ("test_stamp_promoted_and_rejected", test_stamp_promoted_and_rejected, True),
        ("test_stamp_idempotent", test_stamp_idempotent, True),
        ("test_stamp_overwrites_with_new_inputs", test_stamp_overwrites_with_new_inputs, True),
        ("test_stamp_normalizes_long_to_bullish", test_stamp_normalizes_long_to_bullish, True),
        ("test_stamp_normalizes_short_to_bearish", test_stamp_normalizes_short_to_bearish, True),
    ]
    failures = 0
    for name, fn, needs_mp in tests:
        try:
            if needs_mp:
                mp = _Monkeypatch()
                try:
                    fn(mp)
                finally:
                    mp.undo()
            else:
                fn()
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {name}: {type(e).__name__}: {e}")
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

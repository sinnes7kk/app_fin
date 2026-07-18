"""Tests for the purged/embargoed walk-forward splitter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.purged_cv import (  # noqa: E402
    oos_spearman_by_fold,
    purged_walk_forward_splits,
    spearman,
)


def _panel(n_days=40, per_day=5):
    days = pd.bdate_range("2026-01-01", periods=n_days)
    rows = []
    rng = np.random.default_rng(0)
    for d in days:
        for _ in range(per_day):
            f = rng.normal()
            rows.append({"as_of": d.strftime("%Y-%m-%d"), "feat": f,
                         "replay_realized_r": 0.7 * f + rng.normal() * 0.5})
    return pd.DataFrame(rows)


def test_purge_prevents_train_test_overlap_and_leakage():
    p = _panel()
    folds = purged_walk_forward_splits(p["as_of"], n_splits=4, label_days=5,
                                       min_train_days=5)
    assert folds, "expected some folds"
    for fold in folds:
        train_days = set(pd.Timestamp(d).normalize() for d in fold.train_days)
        test_days = set(pd.Timestamp(d).normalize() for d in fold.test_days)
        # No day appears in both sides.
        assert not (train_days & test_days)
        # Purge: every train day is >= label_days before the test start.
        test_start = min(test_days)
        assert max(train_days) <= test_start - pd.Timedelta(days=5)
    print("  PASS: test_purge_prevents_train_test_overlap_and_leakage")


def test_expanding_window_grows():
    p = _panel()
    folds = purged_walk_forward_splits(p["as_of"], n_splits=4, label_days=2,
                                       min_train_days=3, expanding=True)
    sizes = [len(f.train_days) for f in folds]
    assert sizes == sorted(sizes), f"expanding train sets should grow: {sizes}"
    print("  PASS: test_expanding_window_grows")


def test_too_few_days_returns_empty():
    p = _panel(n_days=3, per_day=2)
    folds = purged_walk_forward_splits(p["as_of"], n_splits=5, label_days=5)
    assert folds == []
    print("  PASS: test_too_few_days_returns_empty")


def test_oos_spearman_recovers_positive_signal():
    p = _panel(n_days=60, per_day=6)
    res = oos_spearman_by_fold(p, "feat", n_splits=5, label_days=3,
                               min_train_days=5)
    assert res["n_folds"] >= 2
    # Signal is genuinely positive; pooled OOS should be clearly > 0.
    assert res["pooled"] > 0.2, res
    print("  PASS: test_oos_spearman_recovers_positive_signal")


def test_spearman_guards():
    v, n = spearman(pd.Series([1, 2, 3]), pd.Series([1, 1, 1]))
    assert np.isnan(v)
    v, n = spearman(pd.Series([1, 2, 3, 4, 5]), pd.Series([2, 4, 6, 8, 10]))
    assert abs(v - 1.0) < 1e-9 and n == 5
    print("  PASS: test_spearman_guards")


def main() -> int:
    tests = [
        test_purge_prevents_train_test_overlap_and_leakage,
        test_expanding_window_grows,
        test_too_few_days_returns_empty,
        test_oos_spearman_recovers_positive_signal,
        test_spearman_guards,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR: {fn.__name__}: {type(e).__name__}: {e}")
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

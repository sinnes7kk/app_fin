"""Tests for the sample-size-aware acceptance threshold in
``app/analytics/conviction_recalibration.py``.

Loose regime (n_train < MIN_N_FOR_TIGHT_THRESHOLD): accepts on
positive OOS Spearman that beats legacy.

Tight regime (n_train >= MIN_N_FOR_TIGHT_THRESHOLD): demands a
meaningful OOS Spearman *and* a meaningful lift over legacy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.conviction_recalibration import (  # noqa: E402
    LEGACY_WEIGHTS,
    MIN_N_FOR_TIGHT_THRESHOLD,
    OOS_LIFT_OVER_LEGACY,
    OOS_SPEARMAN_MIN_ACCEPT,
    fit_walk_forward,
)


def _make_panel(n: int, *, signal: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2026-01-01")
    for i in range(n):
        persistence = float(rng.uniform(0, 1))
        intensity = float(rng.uniform(0.5, 50))
        true_r = (
            0.4 * persistence
            + 0.3 * np.log1p(intensity) / np.log1p(50)
            + rng.normal(0, 0.25)
        )
        rows.append({
            "as_of": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "ticker": f"T{i % 30:03d}",
            "direction": "BULLISH",
            "dominant_dte_bucket": "position",
            "persistence_ratio": persistence,
            "prem_mcap_bps": intensity,
            "accumulation_score": float(rng.uniform(-1, 1)),
            "accel_ratio_today": float(rng.uniform(0, 1)),
            "cumulative_premium": float(rng.uniform(1e6, 1e8)),
            "latest_oi_change": float(rng.uniform(-1, 1)),
            "replay_realized_r": signal * true_r + (1 - signal) * rng.normal(0, 1),
        })
    return pd.DataFrame(rows)


def test_loose_regime_kicks_in_below_threshold():
    """Below MIN_N_FOR_TIGHT_THRESHOLD, threshold_regime == 'loose'."""
    n = MIN_N_FOR_TIGHT_THRESHOLD - 5
    df = _make_panel(n, signal=1.0)
    res = fit_walk_forward(df)
    assert res.threshold_regime == "loose", \
        f"expected loose regime, got {res.threshold_regime}"
    assert res.accept_threshold == 0.0
    assert res.accept_lift_threshold == 0.0
    print("  PASS: test_loose_regime_kicks_in_below_threshold")


def test_tight_regime_kicks_in_at_threshold():
    """At/above MIN_N_FOR_TIGHT_THRESHOLD, threshold_regime == 'tight'."""
    n_train_target = MIN_N_FOR_TIGHT_THRESHOLD
    # train_frac=0.6 → need ~n_train_target / 0.6 total rows
    n_total = int(n_train_target / 0.6) + 10
    df = _make_panel(n_total, signal=1.0)
    res = fit_walk_forward(df)
    assert res.n_train >= MIN_N_FOR_TIGHT_THRESHOLD
    assert res.threshold_regime == "tight", \
        f"expected tight regime at n_train={res.n_train}, got {res.threshold_regime}"
    assert res.accept_threshold == OOS_SPEARMAN_MIN_ACCEPT
    assert res.accept_lift_threshold == OOS_LIFT_OVER_LEGACY
    print("  PASS: test_tight_regime_kicks_in_at_threshold")


def test_tight_regime_rejects_marginal_positive_oos():
    """In tight regime, an OOS Spearman of e.g. 0.05 is below
    OOS_SPEARMAN_MIN_ACCEPT=0.10 and must be rejected even if positive.
    """
    n_total = int(MIN_N_FOR_TIGHT_THRESHOLD / 0.6) + 20
    # Weak signal: predictor barely correlated with target.
    df = _make_panel(n_total, signal=0.15, seed=7)
    res = fit_walk_forward(df)
    assert res.threshold_regime == "tight"
    if 0 < res.oos_spearman < OOS_SPEARMAN_MIN_ACCEPT:
        assert not res.accept, (
            f"tight regime should reject sp={res.oos_spearman:.3f} below threshold"
        )
        assert res.weights == LEGACY_WEIGHTS
    print("  PASS: test_tight_regime_rejects_marginal_positive_oos")


def test_tight_regime_accepts_strong_signal_with_lift():
    """A strong-signal synthetic panel produces large OOS Spearman; should
    accept under the tight regime.
    """
    n_total = int(MIN_N_FOR_TIGHT_THRESHOLD / 0.6) + 50
    df = _make_panel(n_total, signal=1.0, seed=1)
    res = fit_walk_forward(df)
    assert res.threshold_regime == "tight"
    if res.oos_spearman >= OOS_SPEARMAN_MIN_ACCEPT and res.oos_spearman >= (
        (res.oos_spearman_legacy + OOS_LIFT_OVER_LEGACY)
        if not (res.oos_spearman_legacy != res.oos_spearman_legacy) else 0
    ):
        assert res.accept, f"expected accept, got reason={res.reason}"
    print("  PASS: test_tight_regime_accepts_strong_signal_with_lift")


def test_constants_have_expected_values():
    """Guard against accidental config drift."""
    assert OOS_SPEARMAN_MIN_ACCEPT == 0.10
    assert OOS_LIFT_OVER_LEGACY == 0.05
    assert MIN_N_FOR_TIGHT_THRESHOLD == 60
    print("  PASS: test_constants_have_expected_values")


def main():
    tests = [
        test_loose_regime_kicks_in_below_threshold,
        test_tight_regime_kicks_in_at_threshold,
        test_tight_regime_rejects_marginal_positive_oos,
        test_tight_regime_accepts_strong_signal_with_lift,
        test_constants_have_expected_values,
    ]
    failures = 0
    for t in tests:
        try:
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

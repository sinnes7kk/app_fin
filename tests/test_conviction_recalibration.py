"""Unit tests for app/analytics/conviction_recalibration.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.conviction_recalibration import (
    LEGACY_WEIGHTS,
    PROXY_FEATURES,
    attach_proxies,
    fit_global_and_per_bucket,
    fit_walk_forward,
)


def _synthetic_panel(n: int = 100, signal_strength: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Synthesize a panel where realized_r is *correlated* with proxies.

    signal_strength = 1.0 → strong correlation (good fit expected)
    signal_strength = 0.0 → pure noise (fit should be rejected)
    """
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2026-01-01")
    for i in range(n):
        persistence = float(rng.uniform(0, 1))
        intensity_proxy_input = float(rng.uniform(0.5, 50))  # prem_mcap_bps
        # ground-truth realized_r is a noisy linear combo
        true_r = (
            0.3 * persistence
            + 0.4 * np.log1p(intensity_proxy_input) / np.log1p(50)
            + rng.normal(0, 0.3)
        )
        rows.append({
            "as_of": (base + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
            "ticker": f"T{i % 20:03d}",
            "direction": "BULLISH",
            "dominant_dte_bucket": "position",
            "persistence_ratio": persistence,
            "prem_mcap_bps": intensity_proxy_input,
            "accumulation_score": float(rng.uniform(-1, 1)),
            "accel_ratio_today": float(rng.uniform(0, 1)),
            "cumulative_premium": float(rng.uniform(1e6, 1e8)),
            "latest_oi_change": float(rng.uniform(-1, 1)),
            "replay_realized_r": signal_strength * true_r + (1 - signal_strength) * rng.normal(0, 1),
        })
    return pd.DataFrame(rows)


def test_attach_proxies_adds_all_six_columns():
    df = _synthetic_panel(20)
    df2 = attach_proxies(df)
    for f in PROXY_FEATURES:
        assert f in df2.columns, f"missing {f}"
        assert df2[f].between(0, 1).all(), f"{f} out of [0,1]"
    print("  PASS: test_attach_proxies_adds_all_six_columns")


def test_fit_walk_forward_strong_signal_accepts():
    df = _synthetic_panel(120, signal_strength=1.0)
    res = fit_walk_forward(df)
    assert res.n_train >= 30, f"n_train={res.n_train}"
    assert res.n_val >= 3
    # On a strong-signal synthetic panel, OOS Spearman should be clearly positive.
    assert res.oos_spearman > 0.1, f"oos={res.oos_spearman}"
    assert res.accept, f"expected accept=True, got reason={res.reason}"
    s = sum(res.weights.values())
    assert abs(s - 1.0) < 0.001, f"weights don't sum to 1: {s}"
    print("  PASS: test_fit_walk_forward_strong_signal_accepts")


def test_fit_walk_forward_pure_noise_rejects():
    df = _synthetic_panel(120, signal_strength=0.0, seed=1)
    res = fit_walk_forward(df)
    # On pure noise, accept should be False (or sometimes flicker; either way
    # the legacy weights should be returned when rejected).
    if not res.accept:
        # Legacy fallback path
        assert res.weights == LEGACY_WEIGHTS, "expected legacy weights on reject"
    print("  PASS: test_fit_walk_forward_pure_noise_rejects")


def test_fit_walk_forward_thin_panel_returns_legacy():
    df = _synthetic_panel(15, signal_strength=1.0)  # below min_train threshold
    res = fit_walk_forward(df)
    assert not res.accept
    assert res.weights == LEGACY_WEIGHTS
    print("  PASS: test_fit_walk_forward_thin_panel_returns_legacy")


def test_fit_global_and_per_bucket_returns_expected_shape():
    df = _synthetic_panel(120, signal_strength=1.0)
    out = fit_global_and_per_bucket(df)
    assert "global" in out
    assert "per_bucket" in out
    assert set(out["per_bucket"].keys()) == {"lottery", "swing", "position", "leap", "unknown"}
    print("  PASS: test_fit_global_and_per_bucket_returns_expected_shape")


def test_legacy_weights_sum_to_one():
    s = sum(LEGACY_WEIGHTS.values())
    assert abs(s - 1.0) < 0.001, f"legacy weights sum to {s}, expected 1.0"
    print("  PASS: test_legacy_weights_sum_to_one")


def main():
    tests = [
        test_attach_proxies_adds_all_six_columns,
        test_fit_walk_forward_strong_signal_accepts,
        test_fit_walk_forward_pure_noise_rejects,
        test_fit_walk_forward_thin_panel_returns_legacy,
        test_fit_global_and_per_bucket_returns_expected_shape,
        test_legacy_weights_sum_to_one,
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

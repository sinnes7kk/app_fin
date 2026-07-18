"""Tests for the meta-labeling walk-forward trainer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.meta_label import MetaResult, train_meta_walk_forward  # noqa: E402


def _panel(n_days=70, per_day=8, signal=1.2, seed=0):
    """Panel where bullish_premium_share genuinely predicts the win."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2026-01-01", periods=n_days)
    rows = []
    for d in days:
        for _ in range(per_day):
            f = rng.uniform(0, 1)
            f2 = rng.normal()
            # realized_r increases with f -> winners cluster at high f.
            r = signal * (f - 0.5) + rng.normal() * 0.4
            rows.append({
                "as_of": d.strftime("%Y-%m-%d"),
                "bullish_premium_share": f,
                "sector_relative_pct": rng.uniform(0, 1),
                "dollar_delta_weighted_flow": f2,
                "aggressor_bull_share": f * 0.8 + rng.uniform(0, 0.2),
                "aggressor_net_prem_bps": f2,
                "ask_side_ratio": rng.uniform(0, 1),
                "directional_sweep_share": rng.normal() * 0.1,
                "far_otm_call_share": rng.uniform(0, 0.3),
                "realized_vol_regime": rng.uniform(0, 2),
                "momentum_composite": f,
                "replay_realized_r": r,
            })
    return pd.DataFrame(rows)


def test_meta_recovers_predictive_signal():
    res = train_meta_walk_forward(_panel(), n_splits=5, label_days=3, min_train_days=5)
    assert isinstance(res, MetaResult)
    assert res.n_folds >= 2, res.note
    assert res.auc > 0.55, f"expected OOS AUC > 0.55, got {res.auc} ({res.note})"
    assert res.r_lift > 0, f"expected positive R lift, got {res.r_lift}"
    print(f"  PASS: test_meta_recovers_predictive_signal (auc={res.auc:.3f}, lift={res.r_lift:+.3f})")


def test_meta_pure_noise_auc_near_half():
    res = train_meta_walk_forward(_panel(signal=0.0, seed=7), n_splits=5,
                                  label_days=3, min_train_days=5)
    assert res.n_folds >= 2, res.note
    # Pure noise -> AUC should hover around 0.5 (allow slack for small OOS).
    assert 0.35 <= res.auc <= 0.65, f"noise AUC drifted: {res.auc}"
    print(f"  PASS: test_meta_pure_noise_auc_near_half (auc={res.auc:.3f})")


def test_meta_small_panel_returns_note():
    tiny = _panel(n_days=3, per_day=3)
    res = train_meta_walk_forward(tiny)
    assert res.auc != res.auc or res.n_folds == 0  # NaN auc or no folds
    assert res.note
    print(f"  PASS: test_meta_small_panel_returns_note ('{res.note}')")


def main() -> int:
    tests = [
        test_meta_recovers_predictive_signal,
        test_meta_pure_noise_auc_near_half,
        test_meta_small_panel_returns_note,
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

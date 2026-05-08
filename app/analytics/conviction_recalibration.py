"""Walk-forward refit of Flow Tracker conviction weights.

Why this exists
---------------
The diagnostic_grade_a report showed that the existing
``FLOW_TRACKER_WEIGHTS_ACCUM`` produce a ``conviction_score`` that is
**negatively correlated with forward excess returns** on the current
panel (~-0.10 Spearman). The 7-tier grade ladder was therefore
non-monotonic on the live panel: A grades had worse realized R than
some B grades.

This module re-fits the weights using non-negative least squares
(NNLS) against the **realized R from the replay backtest**
(``data/grade_history_with_replay.csv``), which is the trustworthy
ground truth. NNLS guarantees non-negative weights so the resulting
score is monotone in each component, and we re-normalize the fitted
vector to sum to 1.0 so the 0-10 scale is preserved.

Walk-forward validation
-----------------------
We split the panel chronologically into a 60% train / 40% validate
window. Weights are fit on the train slice; out-of-sample Spearman
rank-correlation against ``replay_realized_r`` is computed on the
validate slice. We accept the new weights only when the OOS
correlation is positive **and** at least as good as the legacy
weights' OOS correlation. Otherwise we keep the legacy weights and
flag the bucket as 'insufficient evidence'.

Per-bucket fits
---------------
Each DTE bucket gets its own fit. Buckets with n_train < 30 are
flagged ``confidence='low'`` and the recommendation falls back to the
global fit. ``unknown`` always gets a fit (it has the largest n).

Outputs
-------
``fit_global_and_per_bucket(panel)`` returns::

    {
      "global":   {"weights": {...}, "n_train": int, "n_val": int,
                   "oos_spearman": float, "oos_spearman_legacy": float,
                   "accept": bool, "confidence": "high|medium|low"},
      "per_bucket": {
         "lottery":  {... same fields ...},
         "swing":    ...,
         "position": ...,
         "leap":     ...,
         "unknown":  ...,
      },
    }

The numeric proxy features used in the fit
------------------------------------------
- ``persistence_proxy``  = ``persistence_ratio`` (already 0..1)
- ``intensity_proxy``    = log-normalized ``prem_mcap_bps`` (0..1)
- ``consistency_proxy``  = ``|accumulation_score|`` clipped to 0..1
- ``accel_proxy``        = ``accel_ratio_today`` clipped to 0..1
- ``mass_proxy``         = log-normalized ``cumulative_premium`` (0..1)
- ``oi_change_proxy``    = ``|latest_oi_change|`` clipped to 0..2 / 2

These proxies are NOT identical to the in-flight ``*_norm``
intermediates the production scorer computes — they are the
*persisted* approximations available in ``grade_history.csv``. The
fitted weights are interpreted as bucket-specific re-weightings of
these proxies, not literal substitutions into the production
formula. Stage D.3 applies the fitted weights only when the OOS
validation accepts them; otherwise we keep the legacy formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Keep these in sync with the proxy definitions in the docstring.
PROXY_FEATURES = (
    "persistence_proxy",
    "intensity_proxy",
    "consistency_proxy",
    "accel_proxy",
    "mass_proxy",
    "oi_change_proxy",
)

LEGACY_WEIGHTS = {
    "persistence_proxy": 0.25,
    "intensity_proxy":   0.20,
    "consistency_proxy": 0.25,
    "accel_proxy":       0.20,
    "mass_proxy":        0.05,
    "oi_change_proxy":   0.05,
}

# Acceptance thresholds for the walk-forward fit. Below MIN_N_FOR_TIGHT_THRESHOLD
# we keep the loose original rule (positive AND >= legacy) since any positive
# OOS lift on a small sample is worth taking. Once we cross the tight threshold
# the bar rises to a meaningful Spearman *and* a meaningful lift over legacy,
# so we don't churn weights on noise around 0.
OOS_SPEARMAN_MIN_ACCEPT: float = 0.10
OOS_LIFT_OVER_LEGACY: float = 0.05
MIN_N_FOR_TIGHT_THRESHOLD: int = 60

# Floors/ceils used to log-normalize prem_mcap_bps and cumulative_premium —
# match the production formula (`_INTENSITY_FLOOR/_CEIL`, `_MASS_FLOOR/_CEIL`)
# in app/features/flow_tracker.py.
_INTENSITY_FLOOR = math.log1p(0.5)
_INTENSITY_CEIL = math.log1p(50.0)
_MASS_FLOOR = math.log1p(5e5)
_MASS_CEIL = math.log1p(5e7)


@dataclass
class FitResult:
    n_train: int
    n_val: int
    weights: dict[str, float] = field(default_factory=dict)
    oos_spearman: float = float("nan")
    oos_spearman_legacy: float = float("nan")
    accept: bool = False
    confidence: str = "low"
    reason: str = ""
    threshold_regime: str = "loose"
    accept_threshold: float = 0.0
    accept_lift_threshold: float = 0.0


# ---- proxy construction ----------------------------------------------


def _clip01(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


def _intensity_norm(prem_mcap_bps: float | None) -> float:
    if prem_mcap_bps is None or (isinstance(prem_mcap_bps, float) and math.isnan(prem_mcap_bps)):
        return 0.0
    val = float(prem_mcap_bps)
    if val <= 0:
        return 0.0
    log_val = math.log1p(val)
    return _clip01((log_val - _INTENSITY_FLOOR) / (_INTENSITY_CEIL - _INTENSITY_FLOOR))


def _mass_norm(cum_premium: float | None) -> float:
    if cum_premium is None or (isinstance(cum_premium, float) and math.isnan(cum_premium)):
        return 0.0
    val = float(cum_premium)
    if val <= 0:
        return 0.0
    log_val = math.log1p(val)
    return _clip01((log_val - _MASS_FLOOR) / (_MASS_CEIL - _MASS_FLOOR))


def attach_proxies(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the six proxy feature columns in-place; return the same df."""
    df = panel.copy()
    df["persistence_proxy"] = df.get("persistence_ratio", 0).apply(_clip01)
    df["intensity_proxy"] = df.get("prem_mcap_bps", 0).apply(_intensity_norm)
    df["consistency_proxy"] = (
        df.get("accumulation_score", 0).apply(lambda v: _clip01(abs(v)) if v is not None else 0.0)
    )
    df["accel_proxy"] = df.get("accel_ratio_today", 0).apply(
        lambda v: _clip01(abs(v)) if v is not None else 0.0
    )
    df["mass_proxy"] = df.get("cumulative_premium", 0).apply(_mass_norm)
    df["oi_change_proxy"] = df.get("latest_oi_change", 0).apply(
        lambda v: _clip01(abs(v) / 2.0) if v is not None else 0.0
    )
    return df


# ---- NNLS fit + walk-forward -----------------------------------------


def _nnls_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain NNLS via scipy.optimize.nnls, falling back to OLS-with-clip."""
    try:
        from scipy.optimize import nnls
        coefs, _ = nnls(X, y)
        return coefs
    except Exception:
        # Fallback: ordinary least squares, then clip negatives to 0.
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return np.maximum(coefs, 0.0)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; returns NaN if degenerate."""
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    ar = pd.Series(a).rank()
    br = pd.Series(b).rank()
    if ar.std() == 0 or br.std() == 0:
        return float("nan")
    return float(ar.corr(br))


def _weights_to_dict(coefs: np.ndarray) -> dict[str, float]:
    s = float(coefs.sum())
    if s <= 0:
        return {f: 1.0 / len(PROXY_FEATURES) for f in PROXY_FEATURES}
    return {f: round(float(c) / s, 4) for f, c in zip(PROXY_FEATURES, coefs)}


def _score_with_weights(df_proxies: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return np.asarray([
        sum(weights[f] * row[f] for f in PROXY_FEATURES) * 10.0
        for _, row in df_proxies.iterrows()
    ])


def fit_walk_forward(
    panel: pd.DataFrame,
    *,
    target_col: str = "replay_realized_r",
    train_frac: float = 0.6,
    min_train: int = 30,
) -> FitResult:
    """Fit a single set of weights via NNLS on the chronological train
    slice, then evaluate Spearman on the held-out validate slice.

    Accepts the new weights only when their OOS Spearman is **positive
    AND** >= legacy OOS Spearman. Otherwise returns the legacy weights
    with ``accept=False``.
    """
    sub = panel.dropna(subset=[target_col]).copy()
    if len(sub) == 0:
        return FitResult(0, 0, weights=dict(LEGACY_WEIGHTS), reason="no_target_rows")

    sub = attach_proxies(sub)
    sub["__as_of_dt"] = pd.to_datetime(sub.get("as_of"), errors="coerce")
    sub = sub.sort_values("__as_of_dt").reset_index(drop=True)

    n = len(sub)
    n_train = max(min_train, int(n * train_frac))
    n_train = min(n_train, n - 5)  # need at least 5 in validate
    if n_train < min_train or (n - n_train) < 3:
        return FitResult(
            n_train=n,
            n_val=0,
            weights=dict(LEGACY_WEIGHTS),
            reason=f"insufficient_n (n={n}, need ≥{min_train + 3})",
            confidence="low",
            accept=False,
        )

    train = sub.iloc[:n_train].copy()
    val = sub.iloc[n_train:].copy()

    X_train = train[list(PROXY_FEATURES)].to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=float)
    coefs = _nnls_fit(X_train, y_train)
    new_weights = _weights_to_dict(coefs)

    # OOS evaluation
    val_proxies = val.copy()
    val_pred_new = np.asarray([
        sum(new_weights[f] * row[f] for f in PROXY_FEATURES) for _, row in val_proxies.iterrows()
    ])
    val_pred_legacy = np.asarray([
        sum(LEGACY_WEIGHTS[f] * row[f] for f in PROXY_FEATURES) for _, row in val_proxies.iterrows()
    ])
    y_val = val[target_col].to_numpy(dtype=float)

    sp_new = _spearman(val_pred_new, y_val)
    sp_legacy = _spearman(val_pred_legacy, y_val)

    # Sample-size-aware acceptance: tight regime kicks in once we have
    # enough training data that a small positive Spearman could be noise.
    if n_train >= MIN_N_FOR_TIGHT_THRESHOLD:
        threshold_regime = "tight"
        min_accept = OOS_SPEARMAN_MIN_ACCEPT
        min_lift = OOS_LIFT_OVER_LEGACY
        legacy_floor = (
            (sp_legacy + min_lift) if not math.isnan(sp_legacy) else min_accept
        )
        accept = (
            not math.isnan(sp_new)
            and sp_new >= min_accept
            and sp_new >= legacy_floor
        )
        reason_reject = (
            f"tight_threshold_unmet (need spearman>={min_accept:.2f} and "
            f"lift>={min_lift:.2f}, got new={sp_new:.3f}, legacy={sp_legacy:.3f})"
        )
    else:
        threshold_regime = "loose"
        min_accept = 0.0
        min_lift = 0.0
        accept = (
            not math.isnan(sp_new)
            and sp_new > 0
            and (math.isnan(sp_legacy) or sp_new >= sp_legacy)
        )
        reason_reject = "oos_spearman_not_better_than_legacy"

    confidence = (
        "high" if n_train >= 60
        else "medium" if n_train >= 30
        else "low"
    )

    if not accept:
        return FitResult(
            n_train=n_train,
            n_val=len(val),
            weights=dict(LEGACY_WEIGHTS),
            oos_spearman=sp_new,
            oos_spearman_legacy=sp_legacy,
            accept=False,
            confidence=confidence,
            reason=reason_reject,
            threshold_regime=threshold_regime,
            accept_threshold=min_accept,
            accept_lift_threshold=min_lift,
        )

    return FitResult(
        n_train=n_train,
        n_val=len(val),
        weights=new_weights,
        oos_spearman=sp_new,
        oos_spearman_legacy=sp_legacy,
        accept=True,
        confidence=confidence,
        reason="ok",
        threshold_regime=threshold_regime,
        accept_threshold=min_accept,
        accept_lift_threshold=min_lift,
    )


def fit_global_and_per_bucket(panel: pd.DataFrame) -> dict[str, Any]:
    """Run the global fit + a per-bucket fit. Buckets with too few rows are
    skipped (recommendation falls back to global).
    """
    out: dict[str, Any] = {"global": None, "per_bucket": {}}
    out["global"] = fit_walk_forward(panel).__dict__

    for b in ("lottery", "swing", "position", "leap", "unknown"):
        sub = panel[panel["dominant_dte_bucket"].astype(str).str.lower() == b]
        if sub.empty:
            # Try alias forms used in the panel
            alias = {"lottery": "0-7", "swing": "8-30", "position": "31-90", "leap": "91+"}.get(b)
            if alias:
                sub = panel[panel["dominant_dte_bucket"].astype(str) == alias]
        if sub.empty:
            out["per_bucket"][b] = {
                "weights": dict(LEGACY_WEIGHTS),
                "n_train": 0, "n_val": 0,
                "oos_spearman": float("nan"),
                "oos_spearman_legacy": float("nan"),
                "accept": False,
                "confidence": "low",
                "reason": "no_rows_in_bucket",
                "threshold_regime": "loose",
                "accept_threshold": 0.0,
                "accept_lift_threshold": 0.0,
            }
            continue
        out["per_bucket"][b] = fit_walk_forward(sub).__dict__

    return out

"""Meta-labeling layer (López de Prado, AFML ch. 3).

Meta-labeling does *not* pick the trade direction — the primary model
(flow direction / momentum score) already does that. Instead it answers a
second, easier question: **given that we have a candidate, how likely is
it to be a winner, and how much size does it deserve?**

Concretely:

1. The *primary* filter selects candidate rows (e.g. every graded flow
   signal). Direction is taken as given.
2. The *meta* label is binary: did the trade realise a positive
   ``realized_r`` (a win) or not.
3. A small, heavily-regularised classifier (L2 logistic regression on
   standardised features) predicts ``P(win)`` from the validated feature
   set. Its output is a *sizing multiplier*, not a new direction.

The point is precision, not recall: meta-labeling lets us stand down on
low-probability candidates and lean into high-probability ones, which
historically improves the Sharpe of an otherwise-noisy primary signal.

Everything here is evaluated with the purged/embargoed walk-forward
splitter so the reported OOS numbers are leakage-safe. This module is
shadow-only until it clears the same promotion discipline as the
momentum score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.purged_cv import purged_walk_forward_splits

# Default feature set: the validated free features + the aggressor family
# + the shadow momentum composite. Deliberately excludes the dead legacy
# conviction components.
DEFAULT_META_FEATURES: tuple[str, ...] = (
    "bullish_premium_share",
    "sector_relative_pct",
    "dollar_delta_weighted_flow",
    "aggressor_bull_share",
    "aggressor_net_prem_bps",
    "ask_side_ratio",
    "directional_sweep_share",
    "far_otm_call_share",
    "realized_vol_regime",
    "momentum_composite",
)


@dataclass
class MetaResult:
    n: int = 0
    n_folds: int = 0
    auc: float = float("nan")           # OOS ROC-AUC, pooled over folds
    brier: float = float("nan")         # OOS Brier score (calibration)
    base_win_rate: float = float("nan")
    # Lift: mean realized_r of high-P(win) tercile minus low tercile (OOS).
    top_tercile_r: float = float("nan")
    bottom_tercile_r: float = float("nan")
    r_lift: float = float("nan")
    # Sizing curve: mean realized_r by predicted-probability decile.
    decile_r: list[float] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    note: str = ""


def _standardize(train: np.ndarray, test: np.ndarray):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    tr = np.nan_to_num((train - mu) / sd, nan=0.0)
    te = np.nan_to_num((test - mu) / sd, nan=0.0)
    return tr, te


def train_meta_walk_forward(
    panel: pd.DataFrame,
    *,
    features: tuple[str, ...] = DEFAULT_META_FEATURES,
    target: str = "replay_realized_r",
    date_col: str = "as_of",
    n_splits: int = 5,
    label_days: int = 15,
    embargo_days: int = 5,
    min_train_days: int = 10,
    C: float = 0.25,
) -> MetaResult:
    """Fit an L2-logistic meta-model across purged walk-forward folds.

    Returns pooled OOS AUC / Brier plus a realized-R lift table so we can
    see whether P(win) actually sorts winners from losers out of sample.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
    except Exception:
        return MetaResult(note="scikit-learn unavailable")

    cols = [c for c in features if c in panel.columns]
    if not cols:
        return MetaResult(note="no meta features present in panel")

    df = panel[[date_col, target] + cols].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[target])
    # Require at least half the features present per row.
    keep = df[cols].notna().sum(axis=1) >= max(1, len(cols) // 2)
    df = df[keep].reset_index(drop=True)
    if len(df) < 40:
        return MetaResult(n=len(df), features=cols,
                          note="panel too small for meta-labeling (<40 rows)")

    y = (df[target].to_numpy() > 0).astype(int)
    X = df[cols].to_numpy(dtype=float)
    r = df[target].to_numpy(dtype=float)

    folds = purged_walk_forward_splits(
        df[date_col], n_splits=n_splits, label_days=label_days,
        embargo_days=embargo_days, min_train_days=min_train_days,
    )
    if not folds:
        return MetaResult(n=len(df), features=cols,
                          note="no valid walk-forward folds")

    oos_p: list[float] = []
    oos_y: list[int] = []
    oos_r: list[float] = []
    used_folds = 0
    for fold in folds:
        tr_i = fold.train_mask
        te_i = fold.test_mask
        ytr = y[tr_i]
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            continue  # degenerate (all wins or all losses) — skip fold
        Xtr, Xte = _standardize(X[tr_i], X[te_i])
        try:
            clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
        except Exception:
            continue
        oos_p.extend(p.tolist())
        oos_y.extend(y[te_i].tolist())
        oos_r.extend(r[te_i].tolist())
        used_folds += 1

    if used_folds == 0 or len(oos_p) < 10:
        return MetaResult(n=len(df), n_folds=used_folds, features=cols,
                          note="insufficient OOS predictions")

    p_arr = np.array(oos_p)
    y_arr = np.array(oos_y)
    r_arr = np.array(oos_r)

    res = MetaResult(n=len(df), n_folds=used_folds, features=cols)
    res.base_win_rate = float(y_arr.mean())
    try:
        res.auc = float(roc_auc_score(y_arr, p_arr)) if len(set(y_arr.tolist())) > 1 else float("nan")
    except Exception:
        res.auc = float("nan")
    try:
        res.brier = float(brier_score_loss(y_arr, p_arr))
    except Exception:
        res.brier = float("nan")

    # Tercile lift on realized R, sorted by P(win).
    if len(p_arr) >= 15 and len(np.unique(p_arr)) >= 3:
        q = np.quantile(p_arr, [1 / 3, 2 / 3])
        top = r_arr[p_arr >= q[1]]
        bot = r_arr[p_arr <= q[0]]
        if len(top) and len(bot):
            res.top_tercile_r = float(top.mean())
            res.bottom_tercile_r = float(bot.mean())
            res.r_lift = float(top.mean() - bot.mean())

    # Decile sizing curve.
    if len(p_arr) >= 20:
        try:
            deciles = pd.qcut(pd.Series(p_arr), 10, labels=False, duplicates="drop")
            res.decile_r = [
                float(r_arr[deciles.to_numpy() == d].mean())
                for d in sorted(pd.Series(deciles).dropna().unique())
            ]
        except Exception:
            res.decile_r = []

    return res

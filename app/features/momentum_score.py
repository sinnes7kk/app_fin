"""Standalone cross-sectional momentum score (shadow).

A direction-aware, cross-sectional momentum score built *only* from
features that survived out-of-sample validation, plus the new
aggressor-signed premium family. It deliberately excludes the six legacy
``conviction_score`` components, which the backtests showed to have
near-zero predictive power.

Design principles (RenTec-style cross-sectional ranking):

- **Direction-aware orientation.** Every feature is oriented so that a
  higher oriented value means "stronger in the trade's own direction".
  For a BEARISH candidate the bullish features are flipped.
- **Cross-sectional percentile ranking.** Within a single ``as_of`` day
  each oriented feature is ranked into a 0..1 percentile across that
  day's candidate universe. This neutralises day-to-day regime shifts in
  raw feature magnitudes (a strong day lifts everyone).
- **Equal weight.** We do not fit weights yet — with a single market
  regime in-sample, learned weights overfit. The composite is the mean
  of available percentile ranks. Weight-fitting is deferred until the
  shadow period accumulates enough purged walk-forward folds.

The output is shadow-logged next to ``conviction_score`` and never
affects live grades until it clears the promotion gate documented in the
plan (OOS Spearman >= +0.10 and >= conviction_score + 0.05 over 3-4
weeks of walk-forward folds).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# --- Feature specification --------------------------------------------
#
# kind semantics (how the raw value is oriented to "higher = better for
# this trade"):
#   "bull_share"   value in 0..1 that measures bullishness; bearish trades
#                  use (1 - value).
#   "signed"       signed value where positive = bullish; bearish trades
#                  multiply by -1.
#   "sector"       sector-strength percentile in 0..1; bull wants strong
#                  sector, bear wants weak -> (1 - value) for bearish.
#   "neutral"      already oriented to the trade direction upstream;
#                  higher is better regardless of direction.
#   "fade"         higher = worse for any direction; contribution negated.
#
# ``far_otm_directional`` is handled specially: it picks far_otm_call_share
# for BULLISH and far_otm_put_share for BEARISH, then fades it.

FEATURE_SPECS: tuple[tuple[str, str], ...] = (
    ("bullish_premium_share", "bull_share"),
    ("aggressor_bull_share", "bull_share"),
    ("ask_side_ratio", "bull_share"),
    ("sector_relative_pct", "sector"),
    ("aggressor_net_prem_bps", "signed"),
    ("directional_sweep_share", "signed"),
    ("dollar_delta_weighted_flow", "neutral"),
    ("realized_vol_regime", "fade"),
    ("far_otm_directional", "fade"),
)

# Minimum number of non-null oriented features required to emit a score.
MIN_FEATURES = 3


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _is_bullish(direction: Any) -> bool:
    d = str(direction or "BULLISH").upper().strip()
    return d not in {"BEARISH", "SHORT", "PUT", "DOWN"}


def _oriented_value(row: dict, col: str, kind: str, bullish: bool) -> float | None:
    """Return the direction-oriented raw value for one feature (pre-rank)."""
    if col == "far_otm_directional":
        raw = _f(row.get("far_otm_call_share")) if bullish else _f(row.get("far_otm_put_share"))
        if raw is None:
            return None
        return -raw  # fade: more lottery = worse

    v = _f(row.get(col))
    if v is None:
        return None

    if kind == "bull_share":
        return v if bullish else (1.0 - v)
    if kind == "sector":
        return v if bullish else (1.0 - v)
    if kind == "signed":
        return v if bullish else -v
    if kind == "neutral":
        return v
    if kind == "fade":
        return -v
    return v


def compute_day_scores(rows: list[dict]) -> list[dict]:
    """Compute cross-sectional momentum scores for one day's candidates.

    Parameters
    ----------
    rows:
        Feature-lab rows for a single ``as_of`` day. Each dict must carry
        ``ticker``, ``direction`` and the feature columns in
        ``FEATURE_SPECS``.

    Returns
    -------
    A list aligned to ``rows``; each element is a dict with
    ``momentum_composite`` (mean percentile in 0..1) and
    ``momentum_score`` (0..100), or ``None`` values when fewer than
    ``MIN_FEATURES`` features are available for that row.
    """
    n = len(rows)
    if n == 0:
        return []

    # Build the oriented-value matrix: one column per feature.
    oriented: dict[str, list[float | None]] = {}
    for col, kind in FEATURE_SPECS:
        bulls = [_is_bullish(r.get("direction")) for r in rows]
        oriented[col] = [
            _oriented_value(r, col, kind, bulls[i]) for i, r in enumerate(rows)
        ]

    df = pd.DataFrame(oriented)
    # Percentile rank each feature across the day's universe (0..1).
    # A lone non-null value ranks at 0.5 (neutral) rather than 1.0 so a
    # thin universe doesn't manufacture extreme scores.
    ranks = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        nn = s.notna().sum()
        if nn <= 1:
            ranks[col] = s.notna().map({True: 0.5, False: float("nan")})
        else:
            ranks[col] = s.rank(pct=True, na_option="keep")

    out: list[dict] = []
    for i in range(n):
        row_ranks = ranks.iloc[i].dropna()
        if len(row_ranks) < MIN_FEATURES:
            out.append({"momentum_composite": None, "momentum_score": None})
            continue
        composite = float(row_ranks.mean())
        out.append({
            "momentum_composite": round(composite, 6),
            "momentum_score": round(composite * 100.0, 4),
        })
    return out

"""Single source of truth for resolving DTE-bucket-specific hold / trail config.

All call sites that previously read ``MAX_HOLD_DAYS`` or ``ATR_TRAIL_MULT``
from ``app.config`` should migrate to the helpers in this module. This
ensures the production trade-management logic and the replay backtest in
``app.analytics.trade_replay`` stay in lockstep — change the bucket
config in one place (``app.config``) and both production and replay
update.

DTE-bucket labels accepted (canonical):
    lottery   — 0-7 DTE
    swing     — 8-30 DTE
    position  — 31-90 DTE
    leap      — 91+ DTE
    unknown   — DTE enrichment unavailable (fallback)

Aliases such as ``"0-7"``, ``"8-30"``, ``"31-90"``, ``"91+"``, ``"leaps"``
are normalized to the canonical labels.
"""

from __future__ import annotations

from app.config import (
    ATR_TRAIL_MULT_BY_BUCKET,
    EARNINGS_RISK_WINDOW_DAYS,
    MAX_HOLD_DAYS_BY_BUCKET,
    TIME_STOP_MIN_R_BY_BUCKET,
)

# Canonical labels — kept here so that callers can reference the set
# without needing to know the alias mapping.
CANONICAL_BUCKETS = ("lottery", "swing", "position", "leap", "unknown")


def normalize_bucket(bucket: str | None) -> str:
    """Normalize a raw ``dominant_dte_bucket`` value to a canonical label.

    Returns ``"unknown"`` for empty / null / unrecognized values.
    """
    if bucket is None:
        return "unknown"
    s = str(bucket).strip().lower()
    if not s or s in ("nan", "none", "null"):
        return "unknown"
    if s in ("0-7", "0-7d"):
        return "lottery"
    if s in ("8-30", "8-30d"):
        return "swing"
    if s in ("31-90", "31-90d"):
        return "position"
    if s in ("91+", "91+d", "leap", "leaps"):
        return "leap"
    if s in CANONICAL_BUCKETS:
        return s
    return "unknown"


def resolve_hold_config(dominant_dte_bucket: str | None) -> tuple[int, float]:
    """Return ``(max_hold_days, time_stop_min_r)`` for the bucket.

    Falls back to the ``unknown`` bucket if the value isn't recognized.
    """
    canon = normalize_bucket(dominant_dte_bucket)
    max_hold = MAX_HOLD_DAYS_BY_BUCKET.get(canon, MAX_HOLD_DAYS_BY_BUCKET["unknown"])
    time_stop_min_r = TIME_STOP_MIN_R_BY_BUCKET.get(
        canon, TIME_STOP_MIN_R_BY_BUCKET["unknown"]
    )
    return int(max_hold), float(time_stop_min_r)


def resolve_trail_config(dominant_dte_bucket: str | None) -> float:
    """Return the ATR trail multiplier for the bucket."""
    canon = normalize_bucket(dominant_dte_bucket)
    return float(
        ATR_TRAIL_MULT_BY_BUCKET.get(canon, ATR_TRAIL_MULT_BY_BUCKET["unknown"])
    )


def resolve_earnings_window_days() -> int:
    """Return the earnings-risk window in trading days (decoupled from hold)."""
    return int(EARNINGS_RISK_WINDOW_DAYS)

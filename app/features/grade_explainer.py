"""Grade tier mapping + component-driven "why this grade" bullet builder.

Two audiences live here:

- The **flow features** scorer (candidates, /api/scan-ticker, Top Flow
  Intensity). Each row carries ``{side}_{component}_contrib`` columns with
  each component's 0-10-point contribution to ``bullish_score``/``bearish_score``.
- The **Flow Tracker** multi-day composite. Each row carries raw
  ``persistence_ratio`` / ``prem_mcap_bps`` / acceleration / mass ratios
  that roll up into ``conviction_score``.

Both produce a ranked ``list[dict]`` with ``{label, points, kind}`` entries:
top-3 ``driver`` bullets and up to 1 ``drag`` bullet when one component is
materially below its potential. The UI renders these as inline lists on the
trader card and as tooltips on Flow Tracker / Top Flow Intensity tables.
"""

from __future__ import annotations

from typing import Any, Mapping

# ---------------------------------------------------------------------------
# 7-tier conviction grade
# ---------------------------------------------------------------------------
#
# Thresholds are monotone in score so the original A / B / C boundaries are
# preserved; A+/A-/B+/B- only refine within each bucket. See plan notes for
# recalibration guidance once live data accumulates.

GRADE_TIERS: list[tuple[float, str]] = [
    (8.5,  "A+"),
    (7.5,  "A"),
    (6.75, "A-"),
    (6.0,  "B+"),
    (5.0,  "B"),
    (4.0,  "B-"),
]


def conviction_grade(score: float | None) -> str:
    """Map a 0-10 composite conviction score to a 7-tier letter grade."""
    try:
        s = float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        return "C"
    if s != s:  # NaN
        return "C"
    for threshold, label in GRADE_TIERS:
        if s >= threshold:
            return label
    return "C"


def coarse_grade(grade: str | None) -> str:
    """Collapse a 7-tier grade to its A/B/C family (used by backtest stats)."""
    if not grade:
        return "C"
    return grade[0].upper() if grade[0].upper() in ("A", "B", "C") else "C"


# ---------------------------------------------------------------------------
# Flow-features component labels and weights
# ---------------------------------------------------------------------------

# Mirrors app.features.flow_features._FLOW_WEIGHTS.  Kept here as a frozen
# map so this module stays importable without the heavy flow_features
# dependency (the values only matter for computing each component's max
# possible contribution).
_FLOW_WEIGHTS_SNAPSHOT: dict[str, float] = {
    "flow_intensity":    0.29,
    "premium_per_trade": 0.17,
    "vol_oi":            0.17,
    "repeat":            0.14,
    "sweep":             0.12,
    "dte":               0.06,
    "breadth":           0.05,
}

_COMPONENT_LABEL_RAW: dict[str, str] = {
    "flow_intensity":    "Flow intensity",
    "premium_per_trade": "Premium / trade",
    "vol_oi":            "Vol / OI",
    "repeat":            "Repeat flow",
    "sweep":             "Sweep activity",
    "dte":               "DTE quality",
    "breadth":           "Breadth",
}

_COMPONENT_LABEL_DELTA: dict[str, str] = {
    **_COMPONENT_LABEL_RAW,
    "flow_intensity": "Δ-weighted intensity",
}

# Drag threshold: a component is flagged as a "drag" when its contribution
# is below this fraction of its max possible points (weight * 10). 0.20
# means below 20% of potential — low enough to single out genuinely weak
# components without flagging normally-idle ones (DTE, breadth).
_DRAG_RATIO = 0.20


def _active_intensity_label() -> str:
    """Return the label for the intensity component, honouring
    USE_DELTA_WEIGHTED_FLOW at render time (so flipping the flag changes the
    user-visible reason without a code edit).
    """
    try:
        from app import config as _cfg
        if getattr(_cfg, "USE_DELTA_WEIGHTED_FLOW", False):
            return _COMPONENT_LABEL_DELTA["flow_intensity"]
    except Exception:
        pass
    return _COMPONENT_LABEL_RAW["flow_intensity"]


def _resolve_side(row: Mapping[str, Any], side: str | None) -> str | None:
    """Pick the side to explain. Explicit ``side`` wins; otherwise fall back
    to ``direction`` (LONG/SHORT) or ``dominant_direction``. Returns None
    when no directional context is available — caller should skip reasons.
    """
    if side in ("bullish", "bearish"):
        return side
    direction = str(row.get("direction") or row.get("dominant_direction") or "").upper()
    if direction in ("LONG", "BULLISH"):
        return "bullish"
    if direction in ("SHORT", "BEARISH"):
        return "bearish"
    return None


def build_flow_grade_reasons(
    row: Mapping[str, Any],
    *,
    side: str | None = None,
    max_drivers: int = 3,
    max_drags: int = 1,
) -> list[dict[str, Any]]:
    """Build ranked reasons from flow-feature component contributions.

    Expects the row to carry ``{side}_{component}_contrib`` keys from
    ``flow_features.aggregate_flow_by_ticker`` (points on a 0-10 scale).

    Returns a list of ``{label, points, kind, component}`` entries, ranked
    with drivers first (highest contribution) then drags (biggest shortfall
    vs. potential).  Empty list when no component data is present.
    """
    resolved = _resolve_side(row, side)
    if resolved is None:
        return []

    label_map = {**_COMPONENT_LABEL_RAW, "flow_intensity": _active_intensity_label()}

    contribs: list[tuple[str, float]] = []
    for comp in _FLOW_WEIGHTS_SNAPSHOT:
        key = f"{resolved}_{comp}_contrib"
        val = row.get(key)
        if val is None:
            continue
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        contribs.append((comp, fv))

    if not contribs:
        return []

    contribs_sorted = sorted(contribs, key=lambda kv: kv[1], reverse=True)

    drivers: list[dict[str, Any]] = []
    for comp, pts in contribs_sorted:
        if len(drivers) >= max_drivers:
            break
        if pts <= 0:
            break
        drivers.append({
            "label": label_map.get(comp, comp.replace("_", " ").title()),
            "points": round(pts, 2),
            "kind": "driver",
            "component": comp,
        })

    drags: list[dict[str, Any]] = []
    if max_drags > 0:
        driver_comps = {d["component"] for d in drivers}
        # Sort ascending by (deficit vs potential, contribution) to surface the
        # biggest missed opportunity first.
        shortfall_ranked = sorted(
            ((c, p) for c, p in contribs if c not in driver_comps),
            key=lambda kv: kv[1] / (_FLOW_WEIGHTS_SNAPSHOT[kv[0]] * 10.0),
        )
        for comp, pts in shortfall_ranked:
            if len(drags) >= max_drags:
                break
            max_pts = _FLOW_WEIGHTS_SNAPSHOT[comp] * 10.0
            if max_pts <= 0:
                continue
            if pts / max_pts < _DRAG_RATIO:
                drags.append({
                    "label": label_map.get(comp, comp.replace("_", " ").title()),
                    "points": round(pts, 2),
                    "kind": "drag",
                    "component": comp,
                })

    return drivers + drags


# ---------------------------------------------------------------------------
# Flow Tracker (multi-day composite) reasons
# ---------------------------------------------------------------------------

# Mirrors the composite in app.features.flow_tracker.compute_multi_day_flow.
# Wave 0.5 A7 — weights mirror FLOW_TRACKER_WEIGHTS_ACCUM so reason
# contributions match the actual scoring path.  `oi_change` is the new 0.05
# component reclaimed from mass; other weights unchanged from Wave 0.
_TRACKER_WEIGHTS: dict[str, float] = {
    "persistence":  0.25,
    "intensity":    0.20,
    "consistency":  0.25,
    "acceleration": 0.20,
    "mass":         0.05,
    "oi_change":    0.05,
}

_TRACKER_LABELS: dict[str, str] = {
    "persistence":  "Persistence",
    "intensity":    "Intensity (prem / mcap)",
    "consistency":  "Directional consistency",
    "acceleration": "Acceleration",
    "mass":         "Premium mass",
    "oi_change":    "Open-interest build",
}


def build_tracker_grade_reasons(
    row: Mapping[str, Any],
    *,
    max_drivers: int = 3,
    max_drags: int = 1,
) -> list[dict[str, Any]]:
    """Build ranked reasons from Flow Tracker multi-day components.

    Expects the row to carry the intermediate normalized inputs used by
    ``compute_multi_day_flow`` (``persistence_ratio``, ``_intensity_norm``,
    ``_consistency_norm``, ``_accel_norm``, ``_mass_norm``,
    ``_oi_change_norm``).  Also consumes optional structural inputs
    (``sweep_share``, ``multileg_share``, ``dominant_dte_bucket``,
    ``window_return_pct``, ``direction``) to surface Wave 0.5 context
    reasons when they materially alter conviction.
    """
    components: dict[str, float | None] = {
        "persistence":  _coerce_ratio(row.get("persistence_ratio")),
        "intensity":    _coerce_ratio(row.get("_intensity_norm")),
        "consistency":  _coerce_ratio(row.get("_consistency_norm")),
        "acceleration": _coerce_ratio(row.get("_accel_norm")),
        "mass":         _coerce_ratio(row.get("_mass_norm")),
        "oi_change":    _coerce_ratio(row.get("_oi_change_norm")),
    }

    contribs: list[tuple[str, float]] = []
    for comp, norm in components.items():
        if norm is None:
            continue
        pts = _TRACKER_WEIGHTS[comp] * norm * 10.0  # 0-10 scale
        contribs.append((comp, pts))

    if not contribs:
        return []

    contribs_sorted = sorted(contribs, key=lambda kv: kv[1], reverse=True)

    drivers: list[dict[str, Any]] = []
    for comp, pts in contribs_sorted:
        if len(drivers) >= max_drivers:
            break
        if pts <= 0:
            break
        drivers.append({
            "label": _TRACKER_LABELS[comp],
            "points": round(pts, 2),
            "kind": "driver",
            "component": comp,
        })

    drags: list[dict[str, Any]] = []
    if max_drags > 0:
        driver_comps = {d["component"] for d in drivers}
        shortfall_ranked = sorted(
            ((c, p) for c, p in contribs if c not in driver_comps),
            key=lambda kv: kv[1] / (_TRACKER_WEIGHTS[kv[0]] * 10.0),
        )
        for comp, pts in shortfall_ranked:
            if len(drags) >= max_drags:
                break
            max_pts = _TRACKER_WEIGHTS[comp] * 10.0
            if max_pts <= 0:
                continue
            if pts / max_pts < _DRAG_RATIO:
                drags.append({
                    "label": _TRACKER_LABELS[comp],
                    "points": round(pts, 2),
                    "kind": "drag",
                    "component": comp,
                })

    # ------------------------------------------------------------------
    # Wave 0.5 — structural context reasons (non-component).  These don't
    # affect the 0-10 score but colour the narrative: structure quality
    # (sweep/multileg), DTE bucket, window-return alignment.  Only added
    # when meaningfully above/below the noise floor.
    # ------------------------------------------------------------------
    context: list[dict[str, Any]] = []

    # Context reasons share the {label, points, kind, component} shape so
    # templates iterating over grade_reasons stay uniform.  points=0.0 means
    # "no direct score contribution — context only".
    def _ctx(label: str, kind: str, component: str) -> dict[str, Any]:
        return {"label": label, "points": 0.0, "kind": kind, "component": component}

    sweep_share = _coerce_ratio(row.get("sweep_share"))
    if sweep_share is not None:
        if sweep_share >= 0.50:
            context.append(_ctx(
                f"Sweep-heavy ({sweep_share*100:.0f}% of trades)",
                "context-driver", "sweep_share",
            ))
        elif sweep_share <= 0.05 and sweep_share > 0:
            context.append(_ctx(
                "Few sweeps (flow may be positioning, not urgent)",
                "context-drag", "sweep_share",
            ))

    multileg = _coerce_ratio(row.get("multileg_share"))
    if multileg is not None and multileg >= 0.30:
        context.append(_ctx(
            f"Spread-heavy ({multileg*100:.0f}% multileg) — structured bets",
            "context-drag", "multileg_share",
        ))

    dte = str(row.get("dominant_dte_bucket") or "")
    if dte == "0-7":
        context.append(_ctx(
            "Weekly (0-7d) flow — lottery / hedging-prone",
            "context-drag", "dte_bucket",
        ))
    elif dte in ("31-90", "91+"):
        context.append(_ctx(
            f"{dte}d flow — structural commitment",
            "context-driver", "dte_bucket",
        ))

    try:
        ret = float(row.get("window_return_pct") or 0.0)
    except (TypeError, ValueError):
        ret = 0.0
    direction = str(row.get("direction") or "").upper()
    if abs(ret) >= 1.0 and direction in ("BULLISH", "BEARISH"):
        aligned = (direction == "BULLISH" and ret > 0) or (direction == "BEARISH" and ret < 0)
        if aligned:
            context.append(_ctx(
                f"Price up {ret:+.1f}% confirms flow" if direction == "BULLISH" else f"Price down {ret:+.1f}% confirms flow",
                "context-driver", "window_return",
            ))
        else:
            context.append(_ctx(
                f"Price {ret:+.1f}% fighting flow — chase risk",
                "context-drag", "window_return",
            ))

    return drivers + drags + context


def _coerce_ratio(val: Any) -> float | None:
    """Safe 0-1 coercion. Returns None for missing / non-numeric inputs."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# Compact rendering helpers (used by templates / HTML fragments)
# ---------------------------------------------------------------------------

def format_reasons_inline(reasons: list[dict[str, Any]], *, max_items: int = 3) -> str:
    """Return a compact single-line summary suitable for table cells.

    Example: ``"Repeat flow · Vol / OI · Sweep activity"``. Drags are
    suffixed with ``"(drag)"`` so they read correctly when combined.
    """
    if not reasons:
        return ""
    parts: list[str] = []
    for r in reasons[:max_items]:
        label = str(r.get("label", "")).strip()
        if not label:
            continue
        if r.get("kind") == "drag":
            parts.append(f"{label} (drag)")
        else:
            parts.append(label)
    return " · ".join(parts)


def format_reasons_tooltip(reasons: list[dict[str, Any]]) -> str:
    """Return a multi-line tooltip with ``label: +X.YY pts`` / ``-X.YY pts``."""
    if not reasons:
        return ""
    lines: list[str] = []
    for r in reasons:
        label = str(r.get("label", "")).strip()
        try:
            pts = float(r.get("points", 0.0))
        except (TypeError, ValueError):
            pts = 0.0
        kind = r.get("kind", "driver")
        sign = "+" if kind == "driver" else "↓"
        lines.append(f"{sign} {label}: {pts:.2f} pts")
    return "\n".join(lines)

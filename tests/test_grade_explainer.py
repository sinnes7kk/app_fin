"""Tests for the component-driven grade explainer.

Covers:
    - 7-tier conviction_grade ladder
    - Flow-features component ranking (drivers + drag)
    - Flag-aware intensity label (USE_DELTA_WEIGHTED_FLOW)
    - Flow Tracker multi-day component ranking
    - Graceful handling of empty / missing data

Run with either:
    python3 -m pytest tests/test_grade_explainer.py -v
    python3 -m tests.test_grade_explainer           # standalone (no pytest required)
"""

from __future__ import annotations

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    class _PytestShim:
        mark = None

        def __getattr__(self, attr):
            return self

    pytest = _PytestShim()  # type: ignore
    pytest.mark = pytest

from app import config
from app.features.grade_explainer import (
    build_flow_grade_reasons,
    build_tracker_grade_reasons,
    coarse_grade,
    conviction_grade,
    format_reasons_inline,
    format_reasons_tooltip,
)


# ---------------------------------------------------------------------------
# 7-tier grade ladder
# ---------------------------------------------------------------------------

def test_conviction_grade_7_tiers():
    assert conviction_grade(9.0) == "A+"
    assert conviction_grade(8.5) == "A+"
    assert conviction_grade(8.49) == "A"
    assert conviction_grade(7.5) == "A"
    assert conviction_grade(7.49) == "A-"
    assert conviction_grade(6.75) == "A-"
    assert conviction_grade(6.74) == "B+"
    assert conviction_grade(6.0) == "B+"
    assert conviction_grade(5.99) == "B"
    assert conviction_grade(5.0) == "B"
    assert conviction_grade(4.99) == "B-"
    assert conviction_grade(4.0) == "B-"
    assert conviction_grade(3.99) == "C"
    assert conviction_grade(0.0) == "C"


def test_conviction_grade_handles_bad_input():
    assert conviction_grade(None) == "C"
    assert conviction_grade(float("nan")) == "C"
    assert conviction_grade("not-a-number") == "C"


def test_coarse_grade_buckets_to_abc():
    assert coarse_grade("A+") == "A"
    assert coarse_grade("A") == "A"
    assert coarse_grade("A-") == "A"
    assert coarse_grade("B+") == "B"
    assert coarse_grade("B-") == "B"
    assert coarse_grade("C") == "C"
    assert coarse_grade(None) == "C"
    assert coarse_grade("") == "C"


# ---------------------------------------------------------------------------
# Flow-feature reasons
# ---------------------------------------------------------------------------

def _synth_row(side="bullish", **overrides) -> dict:
    """Build a synthetic component-breakdown row. Defaults: repeat-dominant."""
    base = {
        "direction": "LONG" if side == "bullish" else "SHORT",
        f"{side}_flow_intensity_contrib":    0.8,
        f"{side}_premium_per_trade_contrib": 0.5,
        f"{side}_vol_oi_contrib":            1.3,
        f"{side}_repeat_contrib":            1.4,
        f"{side}_sweep_contrib":             1.1,
        f"{side}_dte_contrib":               0.6,
        f"{side}_breadth_contrib":           0.02,  # drag (below 20% of max 0.5)
    }
    base.update(overrides)
    return base


def test_flow_reasons_returns_top_3_drivers_and_one_drag():
    row = _synth_row()
    reasons = build_flow_grade_reasons(row, side="bullish")

    assert len(reasons) == 4, reasons
    assert [r["component"] for r in reasons[:3]] == ["repeat", "vol_oi", "sweep"]
    assert [r["kind"] for r in reasons[:3]] == ["driver", "driver", "driver"]
    assert reasons[0]["points"] == 1.4
    # breadth at 0.02 / 0.5 = 4% — well under the 20% drag threshold
    assert reasons[3]["component"] == "breadth"
    assert reasons[3]["kind"] == "drag"


def test_flow_reasons_auto_picks_side_from_direction():
    row = _synth_row(side="bearish")
    reasons = build_flow_grade_reasons(row)
    assert reasons
    assert all(r["component"] in {"repeat", "vol_oi", "sweep", "breadth", "flow_intensity",
                                   "premium_per_trade", "dte"} for r in reasons)


def test_flow_reasons_empty_when_no_component_columns():
    reasons = build_flow_grade_reasons({"direction": "LONG"}, side="bullish")
    assert reasons == []


def test_flow_reasons_empty_when_side_unresolvable():
    # No direction + no side arg → empty list (can't know which side to score)
    reasons = build_flow_grade_reasons(_synth_row())
    # The synth row still has direction="LONG" so this resolves; verify with no direction
    row_no_dir = {k: v for k, v in _synth_row().items() if k != "direction"}
    assert build_flow_grade_reasons(row_no_dir) == []


def test_flow_reasons_skips_drag_when_all_components_are_strong():
    row = _synth_row(bullish_breadth_contrib=0.4)  # 0.4 / 0.5 = 80% → no drag
    reasons = build_flow_grade_reasons(row, side="bullish")
    kinds = [r["kind"] for r in reasons]
    assert "drag" not in kinds, reasons


def test_flow_reasons_respects_delta_flag_for_intensity_label(monkeypatch):
    """When USE_DELTA_WEIGHTED_FLOW is ON, the intensity reason is
    rendered as 'Δ-weighted intensity' rather than 'Flow intensity'.
    """
    row = _synth_row(bullish_flow_intensity_contrib=2.5)  # force into top 3

    monkeypatch.setattr(config, "USE_DELTA_WEIGHTED_FLOW", False, raising=False)
    off = build_flow_grade_reasons(row, side="bullish")
    off_labels = {r["label"] for r in off if r["component"] == "flow_intensity"}

    monkeypatch.setattr(config, "USE_DELTA_WEIGHTED_FLOW", True, raising=False)
    on = build_flow_grade_reasons(row, side="bullish")
    on_labels = {r["label"] for r in on if r["component"] == "flow_intensity"}

    assert "Flow intensity" in off_labels
    assert any("Δ" in lbl for lbl in on_labels)


# ---------------------------------------------------------------------------
# Flow Tracker reasons
# ---------------------------------------------------------------------------

def test_tracker_reasons_ranks_persistence_first_when_dominant():
    row = {
        "persistence_ratio": 1.0,
        "_intensity_norm":   0.3,
        "_consistency_norm": 0.5,
        "_accel_norm":       0.2,
        "_mass_norm":        0.6,
    }
    reasons = build_tracker_grade_reasons(row)
    assert reasons
    # persistence (1.0 * 0.30 = 3.0 pts) beats intensity (0.3 * 0.30 = 0.9 pts)
    # and consistency (0.5 * 0.20 = 1.0 pts); mass (0.6 * 0.10 = 0.6 pts).
    # Expected driver order: persistence, consistency, intensity
    assert reasons[0]["component"] == "persistence"
    assert reasons[0]["points"] == 3.0


def test_tracker_reasons_empty_row_returns_empty():
    assert build_tracker_grade_reasons({}) == []


def test_tracker_reasons_flags_drag_when_component_weak():
    row = {
        "persistence_ratio": 1.0,
        "_intensity_norm":   0.9,
        "_consistency_norm": 0.01,  # well below 20% → drag
        "_accel_norm":       0.9,
        "_mass_norm":        0.5,
    }
    reasons = build_tracker_grade_reasons(row)
    drag_comps = [r["component"] for r in reasons if r["kind"] == "drag"]
    assert "consistency" in drag_comps


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def test_format_reasons_inline_joins_top_n():
    reasons = [
        {"label": "Repeat flow", "points": 1.4, "kind": "driver"},
        {"label": "Vol / OI",    "points": 1.3, "kind": "driver"},
        {"label": "Sweep",       "points": 1.1, "kind": "driver"},
        {"label": "Breadth",     "points": 0.02, "kind": "drag"},
    ]
    out = format_reasons_inline(reasons, max_items=3)
    assert "Repeat flow" in out
    assert "Vol / OI" in out
    assert "Sweep" in out
    assert "Breadth" not in out  # capped at 3


def test_format_reasons_tooltip_shows_signs_and_points():
    reasons = [
        {"label": "Repeat flow", "points": 1.4, "kind": "driver"},
        {"label": "Breadth",     "points": 0.02, "kind": "drag"},
    ]
    out = format_reasons_tooltip(reasons)
    assert "+ Repeat flow" in out
    assert "1.40" in out
    assert "↓ Breadth" in out


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

class _MonkeyPatch:
    """Tiny monkeypatch shim mimicking pytest's API."""

    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target, name, value=None, raising: bool = True):
        if isinstance(target, str):
            mod_path, _, attr = target.rpartition(".")
            module = __import__(mod_path, fromlist=[attr])
            obj = module
            final_name = attr
            final_value = name
        else:
            obj = target
            final_name = name
            final_value = value
        try:
            old = getattr(obj, final_name)
            had = True
        except AttributeError:
            if raising:
                raise
            old = None
            had = False
        setattr(obj, final_name, final_value)
        self._undo.append((obj, final_name, old, had))

    def undo(self) -> None:
        for obj, name, old, had in reversed(self._undo):
            if had:
                setattr(obj, name, old)
            else:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
        self._undo.clear()


def _run_standalone() -> int:
    import inspect
    import sys
    import traceback

    mod = sys.modules[__name__]
    tests = [
        (name, getattr(mod, name))
        for name in dir(mod)
        if name.startswith("test_") and callable(getattr(mod, name))
    ]
    failed: list[tuple[str, str]] = []
    passed = 0
    for name, fn in tests:
        mp = _MonkeyPatch()
        try:
            sig = inspect.signature(fn)
            kwargs = {"monkeypatch": mp} if "monkeypatch" in sig.parameters else {}
            fn(**kwargs)
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}: {e}")
        finally:
            mp.undo()

    print(f"\n{passed} passed, {len(failed)} failed")
    if failed:
        for name, tb in failed:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())

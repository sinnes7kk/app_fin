"""Day-level directional persistence + Strong gate (2026-05-05 redesign).

Validates:
  - ``_compute_day_persistence`` classifies each day correctly and
    returns ``(dominant_direction, persistence_ratio, has_flips)``.
  - The new ``min_day_persistence`` and ``require_no_flips`` keys in
    ``FLOW_TRACKER_MODES`` are honoured by ``_mode_passes`` so a row
    with strong aggregate consistency but flat day-level breakdown
    still fails Strong (and a row with one opposite-direction day
    fails on the flip check).

Run with either:
    python -m pytest tests/test_flow_tracker_day_persistence.py -v
    python -m tests.test_flow_tracker_day_persistence    # standalone
"""

from __future__ import annotations

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore

from app.config import FLOW_TRACKER_MODES
from app.features.flow_tracker import _compute_day_persistence, _mode_passes


def _row(**overrides) -> dict:
    """Minimal scored-row dict that clears every Strong gate by default.

    Tests then override one field at a time to assert the targeted gate
    is the *only* thing that fails. Keeps assertions narrow.
    """
    base = {
        "active_days": 5,
        "_cum_total": 50_000_000,
        "cumulative_premium": 50_000_000,
        "prem_mcap_bps": 8.0,
        "_consistency_raw": 0.4,
        "_accel_t_stat": 1.5,
        "hedging_risk": False,
        "conviction_grade": "A",
        "perc_3_day_total_max": 0.85,
        "day_persistence": 0.8,
        "has_flips": False,
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# _compute_day_persistence — direct unit tests.
# ─────────────────────────────────────────────────────────────────────────────

def test_day_persistence_all_clearly_bullish():
    daily = [(0, 100, 30, 130), (1, 200, 60, 260), (2, 150, 50, 200)]
    direction, persistence, flips = _compute_day_persistence(daily)
    assert direction == "BULLISH"
    assert persistence == 1.0
    assert flips is False


def test_day_persistence_all_clearly_bearish():
    daily = [(0, 30, 100, 130), (1, 60, 200, 260)]
    direction, persistence, flips = _compute_day_persistence(daily)
    assert direction == "BEARISH"
    assert persistence == 1.0
    assert flips is False


def test_day_persistence_flip_window_marks_has_flips():
    """One bull day, one bear day → has_flips=True regardless of skew."""
    daily = [(0, 100, 30, 130), (1, 30, 100, 130)]
    direction, persistence, flips = _compute_day_persistence(daily)
    assert flips is True
    # Tied counts → no dominant direction.
    assert direction == "NONE"
    assert persistence == 0.0


def test_day_persistence_mixed_with_dominant_direction():
    """3 bull, 1 bear, 1 flat → BULLISH, persistence=3/5, flips=True."""
    daily = [
        (0, 100, 30, 130),   # bull
        (1, 100, 30, 130),   # bull
        (2, 100, 30, 130),   # bull
        (3, 30, 100, 130),   # bear
        (4, 50, 50, 100),    # flat (skew = 0.0)
    ]
    direction, persistence, flips = _compute_day_persistence(daily)
    assert direction == "BULLISH"
    assert persistence == 0.6  # 3 / 5
    assert flips is True


def test_day_persistence_all_flat_returns_none():
    """Every day inside the skew floor → no dominant direction."""
    daily = [(0, 50, 50, 100), (1, 51, 49, 100), (2, 52, 48, 100)]
    direction, persistence, flips = _compute_day_persistence(daily)
    assert direction == "NONE"
    assert persistence == 0.0
    assert flips is False


def test_day_persistence_zero_premium_day_skipped():
    """Days with no premium are dropped before persistence math."""
    daily = [(0, 0, 0, 0), (1, 100, 30, 130), (2, 0, 0, 0)]
    direction, persistence, flips = _compute_day_persistence(daily)
    # Only one active day, clearly bullish.
    assert direction == "BULLISH"
    assert persistence == 1.0
    assert flips is False


def test_day_persistence_empty_input():
    direction, persistence, flips = _compute_day_persistence([])
    assert direction == "NONE"
    assert persistence == 0.0
    assert flips is False


def test_day_persistence_three_tuple_fallback():
    """Helper accepts (bull, bear, total) tuples too, defensively."""
    daily = [(100, 30, 130), (90, 25, 115)]
    direction, persistence, flips = _compute_day_persistence(daily)
    assert direction == "BULLISH"
    assert persistence == 1.0
    assert flips is False


def test_day_persistence_skew_floor_is_inclusive():
    """A day at exactly the skew floor counts as directional."""
    # bull/total = 0.6, bear/total = 0.4 → skew = +0.2 (floor exactly).
    daily = [(0, 60, 40, 100)]
    direction, persistence, _ = _compute_day_persistence(daily, skew_floor=0.20)
    assert direction == "BULLISH"
    assert persistence == 1.0


def test_day_persistence_below_skew_floor_is_flat():
    """Just below the floor → flat, not directional."""
    # skew = 0.198 — below 0.20.
    daily = [(0, 599, 401, 1000)]
    direction, persistence, _ = _compute_day_persistence(daily, skew_floor=0.20)
    assert direction == "NONE"
    assert persistence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration with _mode_passes — Strong's new day-level gates.
# ─────────────────────────────────────────────────────────────────────────────

def test_strong_passes_when_day_persistence_meets_floor():
    strong_cfg = FLOW_TRACKER_MODES["5d"]["strong_accumulation"]
    assert _mode_passes(_row(day_persistence=0.6, has_flips=False), strong_cfg)


def test_strong_fails_when_day_persistence_below_floor():
    strong_cfg = FLOW_TRACKER_MODES["5d"]["strong_accumulation"]
    # 0.5 < 0.6 → fail.
    assert not _mode_passes(_row(day_persistence=0.5), strong_cfg)


def test_strong_fails_when_window_has_flips_even_with_high_persistence():
    """A window with both bull and bear days fails the flip gate even
    if the dominant direction commands a clear majority."""
    strong_cfg = FLOW_TRACKER_MODES["5d"]["strong_accumulation"]
    assert not _mode_passes(
        _row(day_persistence=0.8, has_flips=True), strong_cfg
    )


def test_activity_ignores_day_persistence_and_flips():
    """Activity has no day-level gate, so a row that fails Strong's
    persistence / flip checks must still pass Activity (assuming the
    other gates are satisfied)."""
    activity_cfg = FLOW_TRACKER_MODES["5d"]["activity"]
    assert _mode_passes(
        _row(day_persistence=0.0, has_flips=True, _accel_t_stat=0.2), activity_cfg
    )


def test_all_mode_ignores_day_persistence_and_flips():
    all_cfg = FLOW_TRACKER_MODES["5d"]["all"]
    assert _mode_passes(
        _row(day_persistence=0.0, has_flips=True), all_cfg
    )


def test_strong_consistency_floor_still_enforced():
    """The aggregate-consistency floor on Strong (0.10 after the
    2026-05-09 calibration sweep) still rejects a too-mixed row even
    when day-level persistence and the flip gate pass."""
    strong_cfg = FLOW_TRACKER_MODES["5d"]["strong_accumulation"]
    # Below the new floor — should fail.
    assert not _mode_passes(_row(_consistency_raw=0.05), strong_cfg)
    # At the floor — should pass.
    assert _mode_passes(_row(_consistency_raw=0.10), strong_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Strong @ 2d horizon — the 2-day same-direction radar.
#
# Renamed 2026-05-09 from the standalone "Early" mode after we
# refactored ``FLOW_TRACKER_MODES`` into a per-horizon dict. The 2-day
# pattern is now expressed as Strong on the 2d horizon — same quality
# bar (full day-persistence, no flips, B+ grade) but a smaller window.
# ─────────────────────────────────────────────────────────────────────────────

def _strong_2d_row(**overrides) -> dict:
    """Minimal scored-row dict that clears every Strong @ 2d gate by default."""
    base = {
        "active_days": 2,
        "_cum_total": 8_000_000,
        "cumulative_premium": 8_000_000,
        "prem_mcap_bps": 3.0,
        "_consistency_raw": 0.5,
        "_accel_t_stat": 0.0,
        "hedging_risk": False,
        "conviction_grade": "B+",
        "perc_3_day_total_max": 0.85,
        "day_persistence": 1.0,
        "has_flips": False,
    }
    base.update(overrides)
    return base


def test_strong_2d_passes_two_day_same_direction():
    cfg = FLOW_TRACKER_MODES["2d"]["strong_accumulation"]
    assert _mode_passes(_strong_2d_row(), cfg)


def test_strong_2d_fails_when_one_active_day_only():
    cfg = FLOW_TRACKER_MODES["2d"]["strong_accumulation"]
    assert not _mode_passes(_strong_2d_row(active_days=1), cfg)


def test_strong_2d_requires_full_day_persistence():
    cfg = FLOW_TRACKER_MODES["2d"]["strong_accumulation"]
    # 0.5 = 1 of 2 days same direction — one is flat → fails Strong @ 2d's 1.0 floor.
    assert not _mode_passes(_strong_2d_row(day_persistence=0.5), cfg)


def test_strong_2d_rejects_any_flip():
    cfg = FLOW_TRACKER_MODES["2d"]["strong_accumulation"]
    # Even a strong directional read on the dominant side is rejected
    # if there's any opposite-direction day in the window.
    assert not _mode_passes(_strong_2d_row(has_flips=True), cfg)


def test_strong_2d_excludes_hedging():
    cfg = FLOW_TRACKER_MODES["2d"]["strong_accumulation"]
    assert not _mode_passes(_strong_2d_row(hedging_risk=True), cfg)


def test_strong_2d_grade_floor_b_plus():
    cfg = FLOW_TRACKER_MODES["2d"]["strong_accumulation"]
    # Grade B (rank 2) should fail Strong @ 2d's B+ (rank 3) floor.
    assert not _mode_passes(_strong_2d_row(conviction_grade="B"), cfg)
    assert _mode_passes(_strong_2d_row(conviction_grade="B+"), cfg)
    assert _mode_passes(_strong_2d_row(conviction_grade="A"), cfg)


def test_strong_2d_size_floors_smaller_than_strong_5d():
    """Strong @ 2d permits a $5M cumulative floor (vs Strong @ 5d's
    $25M) because the window covers fewer trading days. This catches
    emerging same-direction flow before it scales up to the longer-
    horizon thresholds."""
    cfg_2d = FLOW_TRACKER_MODES["2d"]["strong_accumulation"]
    cfg_5d = FLOW_TRACKER_MODES["5d"]["strong_accumulation"]
    # $6M cumulative — clears Strong @ 2d ($5M) but not Strong @ 5d ($25M).
    row = _strong_2d_row(_cum_total=6_000_000, cumulative_premium=6_000_000)
    assert _mode_passes(row, cfg_2d)
    assert not _mode_passes(row, cfg_5d)


def test_horizon_resolver_returns_correct_dict():
    """The per-horizon resolver should map ``2d`` / ``5d`` / ``15d`` to
    distinct mode-config dicts and fall back to the default for
    unknown keys."""
    from app.config import resolve_modes_for_horizon, FLOW_TRACKER_HORIZON_DEFAULT

    assert resolve_modes_for_horizon("2d") is FLOW_TRACKER_MODES["2d"]
    assert resolve_modes_for_horizon("5d") is FLOW_TRACKER_MODES["5d"]
    assert resolve_modes_for_horizon("15d") is FLOW_TRACKER_MODES["15d"]
    assert resolve_modes_for_horizon(None) is FLOW_TRACKER_MODES[FLOW_TRACKER_HORIZON_DEFAULT]
    assert resolve_modes_for_horizon("nonsense") is FLOW_TRACKER_MODES[FLOW_TRACKER_HORIZON_DEFAULT]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import traceback

    mod = sys.modules[__name__]
    tests = [
        (n, getattr(mod, n))
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

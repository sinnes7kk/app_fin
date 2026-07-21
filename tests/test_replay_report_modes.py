"""Tests for the flow-tracker mode/streak section of the replay report.

Covers ``section_8_flow_tracker_modes`` and its ``_parse_bool_col`` helper
in ``scripts/build_replay_backtest.py``. The section must:

  - degrade to an explanatory note when the mode columns are absent or
    entirely blank (every pre-2026-07-21 grade_history row), and
  - produce a mode-tier + streak breakdown once rows carry the flags.

Run with:

    python -m pytest tests/test_replay_report_modes.py -v
    python tests/test_replay_report_modes.py           # standalone
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_replay_backtest as brb  # noqa: E402


def test_parse_bool_col_absent_returns_none():
    panel = pd.DataFrame({"replay_realized_r": [0.1, -0.2]})
    assert brb._parse_bool_col(panel, "passes_strong") is None


def test_parse_bool_col_all_blank_returns_none():
    panel = pd.DataFrame({"passes_strong": ["", "", ""]})
    assert brb._parse_bool_col(panel, "passes_strong") is None


def test_parse_bool_col_maps_true_false():
    panel = pd.DataFrame({"passes_strong": ["true", "false", ""]})
    parsed = brb._parse_bool_col(panel, "passes_strong")
    assert parsed is not None
    assert bool(parsed.iloc[0]) is True
    assert bool(parsed.iloc[1]) is False
    assert pd.isna(parsed.iloc[2])


def test_section_8_empty_when_no_flags():
    panel = pd.DataFrame({
        "replay_realized_r": [0.5, -0.3, 1.2],
        "conviction_grade": ["A", "B", "A"],
    })
    md = brb.section_8_flow_tracker_modes(panel)
    assert "No rows carry mode flags yet" in md


def test_section_8_builds_mode_and_streak_tables():
    # Strong rows should be a subset of activity; give Strong the best R so
    # the monotonic-improvement narrative is exercised.
    panel = pd.DataFrame({
        "replay_realized_r": [1.5, 1.0, 0.2, -0.4, 0.6, -0.1],
        "passes_strong":     ["true", "true", "false", "false", "false", "false"],
        "passes_activity":   ["true", "true", "true", "true", "false", "false"],
        "passes_all":        ["true", "true", "true", "true", "true", "true"],
        "active_days":       [5, 4, 3, 2, 2, 3],
        "day_persistence":   [1.0, 1.0, 0.6, 0.4, 0.5, 0.8],
    })
    md = brb.section_8_flow_tracker_modes(panel)
    assert "Mode tier" in md
    assert "Strong" in md
    assert "Activity-only" in md
    assert "All-only" in md
    # Streak + persistence tables render.
    assert "Active-day streak vs realized R" in md
    assert "Day-persistence vs realized R" in md


if __name__ == "__main__":
    tests = [
        test_parse_bool_col_absent_returns_none,
        test_parse_bool_col_all_blank_returns_none,
        test_parse_bool_col_maps_true_false,
        test_section_8_empty_when_no_flags,
        test_section_8_builds_mode_and_streak_tables,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

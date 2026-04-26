"""Layer 2 (streak bonus) + Layer 3 (freight-train flag).

Covers ``compute_streak_bonus`` + ``apply_watchlist_streak_bonus`` for
the score-bonus path and ``build_sector_heat_lookups`` /
``is_freight_train_candidate`` / ``apply_freight_train_flag`` for the
freight-train labelling path.

Run with either:
    python -m pytest tests/test_streak_bonus_freight_train.py -v
    python -m tests.test_streak_bonus_freight_train          # standalone
"""

from __future__ import annotations

import math

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    class _PytestShim:
        def parametrize(self, name, values):
            def decorator(fn):
                fn.__parametrize__ = (name, values)
                return fn
            return decorator
        mark = None  # type: ignore

        def __getattr__(self, attr):
            return self

    pytest = _PytestShim()  # type: ignore
    pytest.mark = pytest  # parametrize lives under .mark in real pytest

import pandas as pd

from app import config as cfg
from app.signals import watchlist as wl


def _approx(actual: float, expected: float, tol: float = 1e-6) -> None:
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=tol), (
        f"expected {expected}, got {actual}"
    )


# ---------------------------------------------------------------------------
# compute_streak_bonus
# ---------------------------------------------------------------------------


def test_streak_bonus_below_min_days_returns_zero() -> None:
    bonus = wl.compute_streak_bonus(
        2, "rising", 0.7,
        min_days=3, step=0.25, max_bonus=1.0, mean_flow_floor=0.4,
    )
    _approx(bonus, 0.0)


def test_streak_bonus_below_mean_floor_returns_zero() -> None:
    bonus = wl.compute_streak_bonus(
        5, "rising", 0.30,
        min_days=3, step=0.25, max_bonus=1.0, mean_flow_floor=0.4,
    )
    _approx(bonus, 0.0)


def test_streak_bonus_rising_trend_full_credit() -> None:
    bonus = wl.compute_streak_bonus(
        3, "rising", 0.7,
        min_days=3, step=0.25, max_bonus=1.0, mean_flow_floor=0.4,
    )
    _approx(bonus, 0.25)


def test_streak_bonus_caps_at_max_bonus() -> None:
    bonus = wl.compute_streak_bonus(
        10, "rising", 0.9,
        min_days=3, step=0.25, max_bonus=1.0, mean_flow_floor=0.4,
    )
    _approx(bonus, 1.0)


def test_streak_bonus_flat_trend_half_credit() -> None:
    bonus = wl.compute_streak_bonus(
        4, "flat", 0.7,
        min_days=3, step=0.25, max_bonus=1.0, mean_flow_floor=0.4,
    )
    _approx(bonus, 0.5 * 0.5)


def test_streak_bonus_falling_trend_zero() -> None:
    bonus = wl.compute_streak_bonus(
        5, "falling", 0.7,
        min_days=3, step=0.25, max_bonus=1.0, mean_flow_floor=0.4,
    )
    _approx(bonus, 0.0)


def test_streak_bonus_unknown_trend_treated_as_flat() -> None:
    bonus = wl.compute_streak_bonus(
        4, "n/a", 0.7,
        min_days=3, step=0.25, max_bonus=1.0, mean_flow_floor=0.4,
    )
    _approx(bonus, 0.5 * 0.5)


# ---------------------------------------------------------------------------
# apply_watchlist_streak_bonus
# ---------------------------------------------------------------------------


def test_apply_streak_bonus_adds_to_final_score(monkeypatch=None) -> None:
    rows = [
        {
            "ticker": "AAA", "direction": "LONG", "final_score": 6.0,
            "watchlist_streak_days": 4,
            "watchlist_flow_trend": "rising",
            "watchlist_mean_flow_score_5d": 0.65,
        },
        {
            "ticker": "BBB", "direction": "LONG", "final_score": 6.0,
            "watchlist_streak_days": 1,
            "watchlist_flow_trend": "rising",
            "watchlist_mean_flow_score_5d": 0.65,
        },
        {"ticker": "CCC", "direction": "LONG", "final_score": 6.0},
    ]
    wl.apply_watchlist_streak_bonus(rows)
    _approx(rows[0]["streak_bonus"], 0.5)
    _approx(rows[0]["final_score"], 6.5)
    _approx(rows[1]["streak_bonus"], 0.0)
    _approx(rows[1]["final_score"], 6.0)
    _approx(rows[2]["streak_bonus"], 0.0)
    _approx(rows[2]["final_score"], 6.0)


def test_apply_streak_bonus_disabled_via_config_keeps_score() -> None:
    saved = getattr(cfg, "USE_WATCHLIST_STREAK_BONUS", True)
    try:
        cfg.USE_WATCHLIST_STREAK_BONUS = False
        rows = [{
            "ticker": "AAA", "direction": "LONG", "final_score": 7.0,
            "watchlist_streak_days": 6,
            "watchlist_flow_trend": "rising",
            "watchlist_mean_flow_score_5d": 0.7,
        }]
        wl.apply_watchlist_streak_bonus(rows)
        _approx(rows[0]["final_score"], 7.0)
        _approx(rows[0]["streak_bonus"], 0.0)
    finally:
        cfg.USE_WATCHLIST_STREAK_BONUS = saved


# ---------------------------------------------------------------------------
# build_sector_heat_lookups
# ---------------------------------------------------------------------------


def _fake_heat_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "snapshot_date", "sector", "direction", "n_tickers", "n_above_thresh",
        "share_above_thresh", "mean_score_topk", "max_score",
        "total_directional_premium", "top_tickers", "sector_heat_score",
    ])


def test_build_sector_heat_lookups_handles_empty_frame() -> None:
    score_by, tops_by = wl.build_sector_heat_lookups(_fake_heat_df([]))
    assert score_by == {}
    assert tops_by == {}


def test_build_sector_heat_lookups_handles_none() -> None:
    score_by, tops_by = wl.build_sector_heat_lookups(None)
    assert score_by == {}
    assert tops_by == {}


def test_build_sector_heat_lookups_normalises_directions() -> None:
    df = _fake_heat_df([
        {
            "snapshot_date": "2026-04-22", "sector": "Semiconductors",
            "direction": "bullish", "n_tickers": 8, "n_above_thresh": 4,
            "share_above_thresh": 0.5, "mean_score_topk": 0.7, "max_score": 0.85,
            "total_directional_premium": 1_000_000, "top_tickers": "NVDA,AMD,AVGO,LRCX,MRVL",
            "sector_heat_score": 6.2,
        },
        {
            "snapshot_date": "2026-04-22", "sector": "Biotech",
            "direction": "bearish", "n_tickers": 6, "n_above_thresh": 2,
            "share_above_thresh": 0.33, "mean_score_topk": 0.55, "max_score": 0.7,
            "total_directional_premium": 500_000, "top_tickers": "MRNA,BNTX",
            "sector_heat_score": 4.6,
        },
    ])
    score_by, tops_by = wl.build_sector_heat_lookups(df)
    assert score_by[("Semiconductors", "LONG")] == 6.2
    assert "AMD" in tops_by[("Semiconductors", "LONG")]
    assert score_by[("Biotech", "SHORT")] == 4.6
    assert tops_by[("Biotech", "SHORT")] == {"MRNA", "BNTX"}


# ---------------------------------------------------------------------------
# is_freight_train_candidate
# ---------------------------------------------------------------------------


def _row(**overrides) -> dict:
    base = {
        "ticker": "AVGO",
        "direction": "LONG",
        "watchlist_streak_days": 4,
        "watchlist_flow_trend": "rising",
        "watchlist_mean_flow_score_5d": 0.6,
    }
    base.update(overrides)
    return base


_HEAT_HOT_SEMIS = (
    {("Semiconductors", "LONG"): 6.5},
    {("Semiconductors", "LONG"): {"AVGO", "NVDA", "MRVL"}},
)
_HEAT_COLD = ({}, {})
_HEAT_BARELY = (
    {("Semiconductors", "LONG"): 4.0},
    {("Semiconductors", "LONG"): {"AVGO"}},  # ticker in top_tickers rescues low score
)


def test_freight_train_qualifies_with_hot_sector() -> None:
    score_by, tops_by = _HEAT_HOT_SEMIS
    ok, reason = wl.is_freight_train_candidate(
        _row(),
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta=None,
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is True
    assert reason is not None
    assert "Semiconductors" in reason
    assert "streak=4d" in reason


def test_freight_train_qualifies_when_in_top_tickers_even_below_heat_floor() -> None:
    score_by, tops_by = _HEAT_BARELY
    ok, _ = wl.is_freight_train_candidate(
        _row(),
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta=None,
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is True


def test_freight_train_rejected_when_streak_too_short() -> None:
    score_by, tops_by = _HEAT_HOT_SEMIS
    ok, reason = wl.is_freight_train_candidate(
        _row(watchlist_streak_days=3),
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta=None,
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is False
    assert reason is None


def test_freight_train_rejected_when_trend_not_rising() -> None:
    score_by, tops_by = _HEAT_HOT_SEMIS
    ok, _ = wl.is_freight_train_candidate(
        _row(watchlist_flow_trend="flat"),
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta=None,
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is False


def test_freight_train_rejected_when_mean_flow_too_low() -> None:
    score_by, tops_by = _HEAT_HOT_SEMIS
    ok, _ = wl.is_freight_train_candidate(
        _row(watchlist_mean_flow_score_5d=0.3),
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta=None,
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is False


def test_freight_train_rejected_when_sector_cold() -> None:
    score_by, tops_by = _HEAT_COLD
    ok, _ = wl.is_freight_train_candidate(
        _row(),
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta=None,
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is False


def test_freight_train_uses_screener_meta_for_unknown_subsector() -> None:
    score_by = {("Healthcare", "LONG"): 5.5}
    tops_by = {("Healthcare", "LONG"): {"PFE"}}
    ok, _ = wl.is_freight_train_candidate(
        _row(ticker="PFE"),
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta={"PFE": {"sector": "Healthcare"}},
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is True


def test_freight_train_unknown_direction_short_circuits() -> None:
    ok, _ = wl.is_freight_train_candidate(
        _row(direction="LONG_BAD"),
        sector_heat_score={("Semiconductors", "LONG"): 7.0},
        sector_top_tickers={("Semiconductors", "LONG"): {"AVGO"}},
        screener_meta=None,
        min_streak=4, min_mean_flow=0.5, require_rising=True, sector_heat_floor=5.0,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# apply_freight_train_flag
# ---------------------------------------------------------------------------


def test_apply_freight_train_flag_stamps_qualifiers() -> None:
    rows = [
        _row(),  # AVGO LONG — qualifies
        _row(ticker="JNJ", watchlist_streak_days=2),  # short streak
        {"ticker": "MSFT", "direction": "LONG"},      # no streak data
    ]
    score_by, tops_by = _HEAT_HOT_SEMIS
    n = wl.apply_freight_train_flag(
        rows,
        sector_heat_score=score_by, sector_top_tickers=tops_by,
        screener_meta=None,
    )
    assert n == 1
    assert rows[0]["freight_train"] is True
    assert "freight_train_reason" in rows[0]
    assert rows[1]["freight_train"] is False
    assert rows[2]["freight_train"] is False


def test_apply_freight_train_flag_disabled_via_config() -> None:
    saved = getattr(cfg, "USE_FREIGHT_TRAIN_FLAG", True)
    try:
        cfg.USE_FREIGHT_TRAIN_FLAG = False
        rows = [_row()]
        score_by, tops_by = _HEAT_HOT_SEMIS
        n = wl.apply_freight_train_flag(
            rows,
            sector_heat_score=score_by, sector_top_tickers=tops_by,
            screener_meta=None,
        )
        assert n == 0
        assert rows[0]["freight_train"] is False
    finally:
        cfg.USE_FREIGHT_TRAIN_FLAG = saved


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------


def _run_standalone() -> int:
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
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}: {e}")

    print()
    print(f"{passed} passed, {len(failed)} failed")
    if failed:
        print()
        for name, tb in failed:
            print(f"--- {name} ---")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_standalone())

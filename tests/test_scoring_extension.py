"""Tests for the extension-cap selector and the soft-extension gate.

Covers the Wave 9 extension-policy refactor in ``app/signals/scoring.py``:

1.  ``_select_extension_cap`` ladder — sustained-trend > breakout > consolidation
    > default; flag-gated.
2.  ``_resolve_extension_state`` — extension failure surfaces as WATCHLIST
    (not REJECT) when the soft gate is enabled and the score earns visibility.
3.  ``_is_sustained_trend`` — clean trend + EMA stack + close on trend side.
4.  ``score_long_setup`` end-to-end — a 2.5-ATR-extended sustained-trend long
    survives as WATCHLIST with ``extended=True`` instead of REJECT.
5.  Backward compat — toggling the flags off restores hard-gate REJECT.

Run with either:

    python -m pytest tests/test_scoring_extension.py -v
    python -m tests.test_scoring_extension     # standalone (no pytest)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config as cfg
from app.signals import scoring


# ---------------------------------------------------------------------------
# Helpers — synthetic OHLCV generator
# ---------------------------------------------------------------------------

def _make_ohlcv(
    *,
    days: int = 60,
    direction: str = "trending_up",
    final_close: float | None = None,
    final_atr: float = 1.0,
    final_ema20: float | None = None,
    final_ema50: float | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV frame the scorer's helpers can consume.

    Generates plausible higher-low / lower-high series for ``direction`` and
    overrides the last bar's ATR / EMA / close so callers can pin the
    extension scenario precisely. ``rel_volume`` and ``ema20`` / ``ema50``
    series get filled in to satisfy ``_is_sustained_trend``,
    ``_extension_score`` and ``_confirmation_volume_score`` defaults.
    """
    idx = pd.date_range("2026-01-01", periods=days, freq="B")
    if direction == "trending_up":
        base = np.linspace(100, 130, days)
    elif direction == "trending_down":
        base = np.linspace(130, 100, days)
    elif direction == "choppy":
        base = 100 + np.sin(np.linspace(0, 6 * np.pi, days)) * 2
    else:
        base = np.full(days, 100.0)

    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.15, days)
    close = base + noise

    high = close + 0.6
    low = close - 0.6
    open_ = close - 0.1

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(days, 1_000_000.0),
            "rel_volume": np.full(days, 1.0),
            "atr14": np.full(days, max(0.5, final_atr)),
            "ema20": pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy(),
            "ema50": pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy(),
        },
        index=idx,
    )

    last = df.index[-1]
    if final_close is not None:
        df.at[last, "close"] = final_close
        df.at[last, "high"] = final_close + 0.5
        df.at[last, "low"] = final_close - 0.5
        df.at[last, "open"] = final_close - 0.2
    df.at[last, "atr14"] = final_atr
    if final_ema20 is not None:
        df.at[last, "ema20"] = final_ema20
    if final_ema50 is not None:
        df.at[last, "ema50"] = final_ema50

    return df


# ---------------------------------------------------------------------------
# Cap-selector ladder
# ---------------------------------------------------------------------------

def test_select_extension_cap_structural_break_wins() -> None:
    """structural_break (4.0) is the most permissive non-sustained cap."""
    df = _make_ohlcv()
    cap, label = scoring._select_extension_cap(
        df, is_long=True, structural_break=True, consolidation_break=False,
    )
    assert label == "structural_break"
    assert cap == scoring.BREAKOUT_MAX_DISTANCE_ATR


def test_select_extension_cap_sustained_trend_widens_default() -> None:
    """A clean multi-day uptrend earns the SUSTAINED_TREND cap (5.0)."""
    df = _make_ohlcv(direction="trending_up")
    last = df.iloc[-1]
    assert last["close"] >= last["ema20"] >= last["ema50"], "fixture must stack"

    cap, label = scoring._select_extension_cap(
        df, is_long=True, structural_break=False, consolidation_break=False,
    )
    assert label == "sustained_trend"
    assert cap == scoring.SUSTAINED_TREND_MAX_DISTANCE_ATR


def test_select_extension_cap_consolidation_then_default() -> None:
    """Without sustained trend, consolidation (3.0) and default (2.5) ladder."""
    df = _make_ohlcv(direction="choppy")  # not a clean trend

    cap_cons, label_cons = scoring._select_extension_cap(
        df, is_long=True, structural_break=False, consolidation_break=True,
    )
    assert label_cons == "consolidation_break"
    assert cap_cons == scoring.CONSOLIDATION_MAX_DISTANCE_ATR

    cap_def, label_def = scoring._select_extension_cap(
        df, is_long=True, structural_break=False, consolidation_break=False,
    )
    assert label_def == "default"
    assert cap_def == scoring.MAX_DISTANCE_FROM_EMA20_ATR


def test_select_extension_cap_sustained_flag_off(monkeypatch) -> None:
    """USE_SUSTAINED_TREND_EXTENSION=False reverts to default cap."""
    monkeypatch.setattr(cfg, "USE_SUSTAINED_TREND_EXTENSION", False)
    df = _make_ohlcv(direction="trending_up")

    cap, label = scoring._select_extension_cap(
        df, is_long=True, structural_break=False, consolidation_break=False,
    )
    assert label == "default"
    assert cap == scoring.MAX_DISTANCE_FROM_EMA20_ATR


# ---------------------------------------------------------------------------
# Soft-gate state resolver
# ---------------------------------------------------------------------------

def test_resolve_extension_state_passing_signal() -> None:
    state, promoted = scoring._resolve_extension_state(
        not_extended_ok=True, score=8.0, extension_cap_label="default",
    )
    assert state == "SIGNAL"
    assert promoted is False


def test_resolve_extension_state_passing_watchlist() -> None:
    state, promoted = scoring._resolve_extension_state(
        not_extended_ok=True, score=5.0, extension_cap_label="default",
    )
    assert state == "WATCHLIST"
    assert promoted is False


def test_resolve_extension_state_soft_promotes_extended_to_watchlist(monkeypatch) -> None:
    """Soft gate: extension failure with score≥4 → WATCHLIST + promoted flag."""
    monkeypatch.setattr(cfg, "USE_SOFT_EXTENSION_GATE", True)
    state, promoted = scoring._resolve_extension_state(
        not_extended_ok=False, score=5.5, extension_cap_label="default",
    )
    assert state == "WATCHLIST"
    assert promoted is True


def test_resolve_extension_state_soft_low_score_still_visible_for_sustained(
    monkeypatch,
) -> None:
    """Sustained-trend setups get visibility even at low score (key SOXL case)."""
    monkeypatch.setattr(cfg, "USE_SOFT_EXTENSION_GATE", True)
    state, promoted = scoring._resolve_extension_state(
        not_extended_ok=False, score=2.5, extension_cap_label="sustained_trend",
    )
    assert state == "WATCHLIST"
    assert promoted is True


def test_resolve_extension_state_soft_low_score_default_cap_still_rejects(
    monkeypatch,
) -> None:
    """Soft gate isn't a free pass — non-sustained low-score setups still REJECT."""
    monkeypatch.setattr(cfg, "USE_SOFT_EXTENSION_GATE", True)
    state, promoted = scoring._resolve_extension_state(
        not_extended_ok=False, score=2.5, extension_cap_label="default",
    )
    assert state == "REJECT"
    assert promoted is False


def test_resolve_extension_state_hard_gate_when_disabled(monkeypatch) -> None:
    """USE_SOFT_EXTENSION_GATE=False restores the legacy hard REJECT."""
    monkeypatch.setattr(cfg, "USE_SOFT_EXTENSION_GATE", False)
    state, promoted = scoring._resolve_extension_state(
        not_extended_ok=False, score=8.5, extension_cap_label="sustained_trend",
    )
    assert state == "REJECT"
    assert promoted is False


# ---------------------------------------------------------------------------
# _is_sustained_trend predicate
# ---------------------------------------------------------------------------

def test_is_sustained_trend_textbook_uptrend_true() -> None:
    df = _make_ohlcv(direction="trending_up")
    assert scoring._is_sustained_trend(df, is_long=True) is True
    assert scoring._is_sustained_trend(df, is_long=False) is False


def test_is_sustained_trend_choppy_false() -> None:
    df = _make_ohlcv(direction="choppy")
    assert scoring._is_sustained_trend(df, is_long=True) is False


def test_is_sustained_trend_close_below_ema_breaks_predicate() -> None:
    """A clean trend that just pulled back below EMA20 is *not* sustained.

    That's correct — the wider cap should fire when we're chasing strength,
    not when we're at the EMA where the default 2.5 ATR cap is plenty.
    """
    df = _make_ohlcv(direction="trending_up", final_close=110.0, final_ema20=115.0, final_ema50=112.0)
    assert scoring._is_sustained_trend(df, is_long=True) is False


# ---------------------------------------------------------------------------
# End-to-end score_long_setup behaviour
# ---------------------------------------------------------------------------

def test_score_long_setup_sustained_cap_rescues_otherwise_rejected_setup(
    monkeypatch,
) -> None:
    """The semis-on-Apr-16 scenario: extended past 2.5 ATR but inside 5.0.

    With a clean trend + EMA stack and close ~3.5 ATR above EMA20, the
    default cap (2.5) would mark the setup ``extended`` and force a hard
    REJECT under the legacy logic. The sustained-trend cap (5.0) lets it
    pass the gate, so ``extended=False`` and the state is determined by
    the score (WATCHLIST or SIGNAL).  ``extension_cap`` should attribute
    the rescue to ``sustained_trend``.
    """
    monkeypatch.setattr(cfg, "USE_SOFT_EXTENSION_GATE", True)
    monkeypatch.setattr(cfg, "USE_SUSTAINED_TREND_EXTENSION", True)
    df = _make_ohlcv(
        direction="trending_up",
        final_close=140.0,
        final_atr=1.0,
        final_ema20=136.5,   # 3.5 ATR away — past default(2.5), inside sustained(5.0)
        final_ema50=125.0,
    )
    res = scoring.score_long_setup(df)
    assert res["extension_cap"] == "sustained_trend"
    assert res["extended"] is False
    assert res["state"] in {"SIGNAL", "WATCHLIST"}, res


def test_score_long_setup_far_extended_soft_gate_promotes_to_watchlist(
    monkeypatch,
) -> None:
    """When the setup is extended past *every* cap, soft gate keeps it visible.

    Close 6 ATR above EMA20 — outside even the 5.0-ATR sustained cap. Soft
    gate must promote to WATCHLIST with ``extended=True`` instead of REJECT.
    """
    monkeypatch.setattr(cfg, "USE_SOFT_EXTENSION_GATE", True)
    monkeypatch.setattr(cfg, "USE_SUSTAINED_TREND_EXTENSION", True)
    df = _make_ohlcv(
        direction="trending_up",
        final_close=143.0,
        final_atr=1.0,
        final_ema20=137.0,   # 6 ATR away — past every cap
        final_ema50=125.0,
    )
    res = scoring.score_long_setup(df)
    assert res["extended"] is True
    assert res["extension_soft_promoted"] is True
    assert res["state"] == "WATCHLIST"
    assert res["extension_cap"] == "sustained_trend"


def test_score_long_setup_extended_hard_gate_rejects_when_flags_off(
    monkeypatch,
) -> None:
    """With both flags off the legacy hard REJECT is restored."""
    monkeypatch.setattr(cfg, "USE_SOFT_EXTENSION_GATE", False)
    monkeypatch.setattr(cfg, "USE_SUSTAINED_TREND_EXTENSION", False)
    df = _make_ohlcv(
        direction="trending_up",
        final_close=140.0,
        final_atr=1.0,
        final_ema20=136.5,    # 3.5 ATR — extended under default(2.5) cap
        final_ema50=125.0,
    )
    res = scoring.score_long_setup(df)
    assert res["state"] == "REJECT"
    assert res["extended"] is True
    # Cap selection here can be ``default`` or ``consolidation_break``
    # depending on level detection; either way the cap is below the
    # 3.5-ATR distance, so REJECT must fire under the hard gate.
    assert res["extension_cap"] in {"default", "consolidation_break"}


def test_score_long_setup_not_extended_signal_keeps_extended_false() -> None:
    """A close *inside* the cap leaves ``extended`` False and lets SIGNAL fire."""
    df = _make_ohlcv(
        direction="trending_up",
        final_close=131.0,
        final_atr=1.0,
        final_ema20=130.0,   # 1 ATR away → well within any cap
        final_ema50=120.0,
    )
    res = scoring.score_long_setup(df)
    assert res["extended"] is False


# ---------------------------------------------------------------------------
# Standalone runner — ``python -m tests.test_scoring_extension``
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover

    class _MP:
        """Minimal monkeypatch shim so the suite runs without pytest."""

        def __init__(self) -> None:
            self._undo: list[tuple[object, str, object]] = []

        def setattr(self, target, name: str, value):
            self._undo.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo_all(self) -> None:
            for target, name, value in reversed(self._undo):
                setattr(target, name, value)
            self._undo.clear()

    funcs = [
        test_select_extension_cap_structural_break_wins,
        test_select_extension_cap_sustained_trend_widens_default,
        test_select_extension_cap_consolidation_then_default,
        test_select_extension_cap_sustained_flag_off,
        test_resolve_extension_state_passing_signal,
        test_resolve_extension_state_passing_watchlist,
        test_resolve_extension_state_soft_promotes_extended_to_watchlist,
        test_resolve_extension_state_soft_low_score_still_visible_for_sustained,
        test_resolve_extension_state_soft_low_score_default_cap_still_rejects,
        test_resolve_extension_state_hard_gate_when_disabled,
        test_is_sustained_trend_textbook_uptrend_true,
        test_is_sustained_trend_choppy_false,
        test_is_sustained_trend_close_below_ema_breaks_predicate,
        test_score_long_setup_sustained_cap_rescues_otherwise_rejected_setup,
        test_score_long_setup_far_extended_soft_gate_promotes_to_watchlist,
        test_score_long_setup_extended_hard_gate_rejects_when_flags_off,
        test_score_long_setup_not_extended_signal_keeps_extended_false,
    ]
    failures: list[tuple[str, BaseException]] = []
    for f in funcs:
        mp = _MP()
        try:
            if "monkeypatch" in f.__code__.co_varnames:
                f(mp)
            else:
                f()
            print(f"  PASS  {f.__name__}")
        except BaseException as e:  # pragma: no cover
            print(f"  FAIL  {f.__name__}: {e}")
            failures.append((f.__name__, e))
        finally:
            mp.undo_all()

    print()
    if failures:
        print(f"{len(failures)} failed of {len(funcs)}")
        raise SystemExit(1)
    print(f"{len(funcs)} passed")

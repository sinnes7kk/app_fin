"""Wave 8 — tests for ``app.features.risk_regime``.

The regime aggregator has four hard invariants worth locking in:

1. **Calm passthrough** — a quiet tape (VIX 15, SPY RSI 50, no heat) must
   return ``tier=calm``, ``multiplier ≈ 1.0``, and a majority-``positive``
   check list.  Regressions here silently throttle normal-market sizing.
2. **VIX ladder** — bumping VIX into each configured tier must move the
   tier label and drop the multiplier monotonically.  This guards the
   sizing curve that Structure + Trader Cards consume.
3. **Hard halts** — portfolio heat at the freeze threshold, or a macro
   catalyst inside its halt window, must set ``tier=halt`` /
   ``multiplier=0``.  Losing this would let us put on trades the day of
   FOMC which is the whole reason the module exists.
4. **Fail-soft** — empty / malformed inputs must still return a
   well-shaped payload so the Action Bar keeps rendering when data
   fetches fail.

Run with:
    python -m pytest tests/test_risk_regime.py -v
    python tests/test_risk_regime.py     # standalone, no pytest required
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from datetime import date, timedelta
from pathlib import Path

# Allow standalone execution from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features.risk_regime import (
    TIER_CALM,
    TIER_ELEVATED,
    TIER_HALT,
    TIER_PANIC,
    compute_risk_regime,
    summarise_for_ui,
)


def _tmp_calendar(payload: dict) -> Path:
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, fh)
    fh.close()
    return Path(fh.name)


def _empty_calendar() -> Path:
    return _tmp_calendar({"fomc": [], "cpi": [], "nfp": []})


def _calm_mi() -> dict:
    return {"vix": 15.0, "vix3m": 17.0, "spy_rsi": 50.0, "spy_trend": "BULLISH"}


def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_calm_market_returns_unity_multiplier() -> None:
    cal = _empty_calendar()
    try:
        out = compute_risk_regime(
            market_indicators=_calm_mi(),
            positions=[],
            heat_pct=0.5,
            calendar_path=cal,
        )
        assert out["tier"] == TIER_CALM
        assert _approx(out["multiplier"], 1.0), out["multiplier"]
        assert out["score"] == 100
        assert out["halt_reason"] is None
        labels = [c["label"] for c in out["checks"]]
        assert any("VIX" in lbl for lbl in labels)
        assert any("SPY RSI" in lbl for lbl in labels)
    finally:
        os.unlink(cal)


def test_vix_ladder_monotonic() -> None:
    """Higher VIX must never reward the trader with a larger multiplier."""
    cal = _empty_calendar()
    try:
        levels = [10.0, 20.0, 28.0, 42.0]
        mults: list[float] = []
        tiers: list[str] = []
        for v in levels:
            mi = {**_calm_mi(), "vix": v, "vix3m": v * 1.1}
            out = compute_risk_regime(
                market_indicators=mi,
                positions=[],
                heat_pct=0.0,
                calendar_path=cal,
            )
            mults.append(out["multiplier"])
            tiers.append(out["tier"])
        assert mults == sorted(mults, reverse=True), mults
        assert tiers[0] == TIER_CALM
        assert tiers[-1] == TIER_HALT
    finally:
        os.unlink(cal)


def test_heat_freeze_halts_regime() -> None:
    cal = _empty_calendar()
    try:
        out = compute_risk_regime(
            market_indicators=_calm_mi(),
            positions=[],
            heat_pct=6.0,  # above HEAT_FREEZE_PCT=5.0
            calendar_path=cal,
        )
        assert out["tier"] == TIER_HALT
        assert out["multiplier"] == 0.0
        assert out["score"] == 0
        heat_check = next(c for c in out["checks"] if "heat" in c["label"].lower())
        assert heat_check["tone"] == "negative"
    finally:
        os.unlink(cal)


def test_heat_clamp_drops_size_in_half() -> None:
    cal = _empty_calendar()
    try:
        out = compute_risk_regime(
            market_indicators=_calm_mi(),
            positions=[],
            heat_pct=3.5,  # between clamp (3%) and freeze (5%)
            calendar_path=cal,
        )
        assert out["tier"] != TIER_HALT
        # Base VIX mult is 1.0, heat clamp halves it.
        assert _approx(out["multiplier"], 0.5), out["multiplier"]
        heat_check = next(c for c in out["checks"] if "heat" in c["label"].lower())
        assert heat_check["tone"] == "warning"
    finally:
        os.unlink(cal)


def test_fomc_within_window_forces_halt() -> None:
    today = date(2026, 6, 10)
    cal = _tmp_calendar({"fomc": [today.isoformat()], "cpi": [], "nfp": []})
    try:
        out = compute_risk_regime(
            market_indicators=_calm_mi(),
            positions=[],
            heat_pct=0.0,
            today=today,
            calendar_path=cal,
        )
        assert out["tier"] == TIER_HALT
        assert out["halt_reason"]
        assert "FOMC" in (out["halt_reason"] or "")
        assert out["multiplier"] == 0.0
    finally:
        os.unlink(cal)


def test_macro_event_outside_window_is_informational() -> None:
    today = date(2026, 6, 10)
    cpi_day = (today + timedelta(days=10)).isoformat()
    cal = _tmp_calendar({"fomc": [], "cpi": [cpi_day], "nfp": []})
    try:
        out = compute_risk_regime(
            market_indicators=_calm_mi(),
            positions=[],
            heat_pct=0.0,
            today=today,
            calendar_path=cal,
        )
        assert out["tier"] == TIER_CALM
        assert out["halt_reason"] is None
    finally:
        os.unlink(cal)


def test_backwardated_vix_term_bumps_to_elevated() -> None:
    cal = _empty_calendar()
    try:
        # VIX3M < VIX => backwardation.
        out = compute_risk_regime(
            market_indicators={
                "vix": 20.0, "vix3m": 18.0, "spy_rsi": 55.0, "spy_trend": "BULLISH"
            },
            positions=[],
            heat_pct=0.0,
            calendar_path=cal,
        )
        assert out["tier"] in (TIER_ELEVATED, TIER_PANIC)
        term_check = next(c for c in out["checks"] if "term" in c["label"].lower())
        assert term_check["tone"] == "warning"
    finally:
        os.unlink(cal)


def test_sector_concentration_reduces_multiplier() -> None:
    cal = _empty_calendar()
    try:
        positions = [
            {"ticker": f"T{i}", "sector": "tech", "direction": "LONG"} for i in range(3)
        ]
        base = compute_risk_regime(
            market_indicators=_calm_mi(),
            positions=[],
            heat_pct=0.0,
            sector="tech",
            direction="LONG",
            calendar_path=cal,
        )
        concentrated = compute_risk_regime(
            market_indicators=_calm_mi(),
            positions=positions,
            heat_pct=0.0,
            sector="tech",
            direction="LONG",
            calendar_path=cal,
        )
        assert concentrated["multiplier"] < base["multiplier"]
        conc_check = next(
            c for c in concentrated["checks"] if "concentration" in c["label"].lower()
        )
        assert conc_check["tone"] == "warning"
    finally:
        os.unlink(cal)


def test_malformed_inputs_fail_soft() -> None:
    cal = _empty_calendar()
    try:
        out = compute_risk_regime(
            market_indicators={"vix": "not-a-number", "spy_trend": None},
            positions=None,
            heat_pct=None,
            calendar_path=cal,
        )
        assert out["tier"] == TIER_CALM
        assert out["multiplier"] == 1.0
        assert out["checks"]
    finally:
        os.unlink(cal)


def test_summarise_for_ui_shape() -> None:
    fake_regime = {
        "tier": "elevated",
        "tier_label": "Elevated vol",
        "score": 75,
        "multiplier": 0.75,
        "halt_reason": None,
    }
    s = summarise_for_ui(fake_regime)
    assert set(s.keys()) == {"tier", "label", "score", "multiplier", "halt_reason"}
    assert s["tier"] == "elevated"
    assert _approx(s["multiplier"], 0.75)

    fallback = summarise_for_ui({})
    assert fallback["tier"] == TIER_CALM
    assert fallback["multiplier"] == 1.0


# ---------------------------------------------------------------------------
# Wave 8 integration — attach_sizing_context from view_models applies the
# regime to a trade_structure payload so Structure tabs everywhere show the
# same pill / HALT caveat.  This guards the server-side plumbing.
# ---------------------------------------------------------------------------
def test_attach_sizing_context_adds_block_and_caveat() -> None:
    from app.web.view_models import attach_sizing_context

    structure = {
        "primary": {"name": "LONG_CALL"},
        "alternatives": [],
        "avoid": [],
        "caveats": [],
    }
    regime = {
        "tier": "panic",
        "tier_label": "Panic regime",
        "multiplier": 0.5,
        "checks": [{"label": "VIX 28", "tone": "negative", "detail": ""}],
        "halt_reason": None,
    }
    out = attach_sizing_context(dict(structure), regime)
    assert isinstance(out, dict)
    assert out["sizing_context"]["tier"] == "panic"
    assert _approx(out["sizing_context"]["multiplier"], 0.5)
    assert any("regime" in c.lower() for c in out["caveats"])


def test_attach_sizing_context_halt_inserts_first_caveat() -> None:
    from app.web.view_models import attach_sizing_context

    structure = {
        "primary": {"name": "STOCK"},
        "alternatives": [],
        "avoid": [],
        "caveats": ["Watch overnight gap risk"],
    }
    regime = {
        "tier": "halt",
        "tier_label": "Halt",
        "multiplier": 0.0,
        "checks": [],
        "halt_reason": "FOMC today",
    }
    out = attach_sizing_context(dict(structure), regime)
    assert out is not None
    assert out["sizing_context"]["tier"] == "halt"
    # Halt caveat must be the first one so traders see it immediately.
    assert out["caveats"][0].startswith("Regime HALT")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_calm_market_returns_unity_multiplier,
        test_vix_ladder_monotonic,
        test_heat_freeze_halts_regime,
        test_heat_clamp_drops_size_in_half,
        test_fomc_within_window_forces_halt,
        test_macro_event_outside_window_is_informational,
        test_backwardated_vix_term_bumps_to_elevated,
        test_sector_concentration_reduces_multiplier,
        test_malformed_inputs_fail_soft,
        test_summarise_for_ui_shape,
        test_attach_sizing_context_adds_block_and_caveat,
        test_attach_sizing_context_halt_inserts_first_caveat,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)

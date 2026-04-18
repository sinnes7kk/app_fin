"""Wave 8 — Market-aware risk regime aggregator.

Condenses a dozen scattered sizing inputs (VIX level, VIX term structure,
SPY trend / RSI, realised vol, portfolio heat, sector / direction
concentration, macro event proximity) into a single 0-100 regime score,
tier label, sizing multiplier, and a flat list of human-readable checks
that the Action Bar pill + the Structure tab's *Sizing Context* block
both render.

Design goals
------------

1. **One payload, many surfaces**.  ``compute_risk_regime`` is the only
   function the UI should call; everything else in this module is
   private helpers.  The payload is a pure dict — no Flask, no pandas
   coupling — so ``view_models.TraderCardView``, ``server._build_action_bar``,
   and ``_enrich_flow_tracker_decision`` can all attach it.
2. **Deterministic from inputs**.  Every check reads explicit fields
   from ``regime`` / ``positions`` / ``market_indicators`` / calendar
   so tests can stub inputs and assert on the ``checks`` list directly.
3. **Fail-soft**.  A missing field degrades the regime score toward
   neutral rather than raising.  The Action Bar stays rendered even if
   SPY fetch fails.
4. **Composable tiers**.  Base multiplier comes from VIX.  Each negative
   check multiplies.  Hard halts (HALT_ON_FOMC_WINDOW_DAYS etc.) skip
   the math and jump straight to ``halt``.

Output::

    {
        "tier":        "calm" | "elevated" | "panic" | "halt",
        "tier_label":  "Calm market" | "Elevated vol" | ...,
        "score":       int 0..100,                   # 100 = fully risk-on
        "multiplier":  float 0..1,                   # sizing multiplier
        "checks":      [                              # evidence shown in UI
            {"label": "VIX 23.4", "detail": "Elevated tier",
             "tone":  "warning"},
            ...
        ],
        "halt_reason": "FOMC tomorrow" | None,
        "inputs":      { raw snapshot we used, for debugging },
    }

Tone vocabulary matches ``trade_structure`` / ``flow_narrative``:
``positive`` / ``neutral`` / ``warning`` / ``negative`` / ``info``.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from app.config import (
    ECONOMIC_CALENDAR_PATH,
    HALT_ON_CPI_WINDOW_DAYS,
    HALT_ON_FOMC_WINDOW_DAYS,
    HALT_ON_NFP_WINDOW_DAYS,
    HEAT_CLAMP_PCT,
    HEAT_FREEZE_PCT,
    MAX_SAME_DIRECTION,
    MAX_SAME_SECTOR,
    SPY_RSI_OVERBOUGHT,
    SPY_RSI_OVERSOLD,
    VIX_BACKWARDATION_THRESHOLD,
    VIX_TIERS,
)

TONE_POSITIVE = "positive"
TONE_NEUTRAL = "neutral"
TONE_WARNING = "warning"
TONE_NEGATIVE = "negative"
TONE_INFO = "info"

TIER_CALM = "calm"
TIER_ELEVATED = "elevated"
TIER_PANIC = "panic"
TIER_HALT = "halt"

TIER_LABELS = {
    TIER_CALM: "Calm market",
    TIER_ELEVATED: "Elevated vol",
    TIER_PANIC: "Panic regime",
    TIER_HALT: "Halt",
}

TIER_RANK = {TIER_CALM: 0, TIER_ELEVATED: 1, TIER_PANIC: 2, TIER_HALT: 3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _f(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    try:
        x = float(v)
        if x != x:
            return default
        return x
    except (TypeError, ValueError):
        return default


def _check(label: str, tone: str, detail: str | None = None) -> dict[str, str]:
    return {"label": label, "tone": tone, "detail": detail or ""}


def _vix_tier(vix: float | None) -> tuple[str, float]:
    """Return ``(tier, multiplier)`` for the VIX level."""
    if vix is None:
        return TIER_CALM, 1.0
    for name, lo, hi, mult in VIX_TIERS:
        if lo <= vix < hi:
            return name, mult
    return VIX_TIERS[-1][0], VIX_TIERS[-1][3]


# ---------------------------------------------------------------------------
# Economic calendar
# ---------------------------------------------------------------------------
def _load_calendar(path: str | Path | None = None) -> dict:
    """Return the economic calendar payload, or an empty dict on failure."""
    p = Path(path) if path is not None else Path(ECONOMIC_CALENDAR_PATH)
    try:
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return {}
        return payload
    except (OSError, json.JSONDecodeError):
        return {}


def _nearest_event(today: date, dates: Iterable[str]) -> int | None:
    """Minimum absolute day-delta between ``today`` and any date in ``dates``.

    Returns ``None`` if there are no parseable entries or the closest
    future event is > 30 days away (we don't need to render that).
    """
    best: int | None = None
    for s in dates or []:
        try:
            d = datetime.strptime(str(s), "%Y-%m-%d").date()
        except ValueError:
            continue
        delta = (d - today).days
        if abs(delta) > 30:
            continue
        if best is None or abs(delta) < abs(best):
            best = delta
    return best


# ---------------------------------------------------------------------------
# Sub-checks
# ---------------------------------------------------------------------------
def _check_vix_level(vix: float | None) -> tuple[str, float, dict]:
    tier, mult = _vix_tier(vix)
    if vix is None:
        return TIER_CALM, 1.0, _check("VIX n/a", TONE_NEUTRAL, "No VIX reading — assume calm.")
    detail = f"{TIER_LABELS[tier]} — ×{mult:.2f} sizing"
    if tier == TIER_CALM:
        tone = TONE_POSITIVE
    elif tier == TIER_ELEVATED:
        tone = TONE_WARNING
    elif tier == TIER_PANIC:
        tone = TONE_NEGATIVE
    else:
        tone = TONE_NEGATIVE
    return tier, mult, _check(f"VIX {vix:.1f}", tone, detail)


def _check_vix_term(vix: float | None, vix3m: float | None) -> dict | None:
    if vix is None or vix3m is None or vix <= 0 or vix3m <= 0:
        return None
    ratio = vix3m / vix
    if VIX_BACKWARDATION_THRESHOLD is None:
        return None
    if ratio < VIX_BACKWARDATION_THRESHOLD:
        return _check(
            f"VIX term backwardated ({ratio:.2f})",
            TONE_WARNING,
            "VIX3M below VIX — market pricing fear-now, not normal contango.",
        )
    return _check(
        f"VIX term contango ({ratio:.2f})",
        TONE_INFO,
        "Normal vol term structure.",
    )


def _check_spy_rsi(rsi: float | None) -> dict | None:
    if rsi is None:
        return None
    if rsi >= SPY_RSI_OVERBOUGHT:
        return _check(
            f"SPY RSI {rsi:.0f}",
            TONE_WARNING,
            "Overbought — late longs face reversal risk, fade-the-rip more likely.",
        )
    if rsi <= SPY_RSI_OVERSOLD:
        return _check(
            f"SPY RSI {rsi:.0f}",
            TONE_WARNING,
            "Oversold — shorts face squeeze risk, bounces more violent than usual.",
        )
    return _check(
        f"SPY RSI {rsi:.0f}",
        TONE_POSITIVE,
        "SPY in range — neither overbought nor oversold.",
    )


def _check_spy_trend(trend: str | None) -> dict | None:
    if not trend:
        return None
    t = str(trend).upper()
    if t == "BULLISH":
        return _check("SPY trend bullish", TONE_POSITIVE,
                       "Price above both EMAs and slope positive.")
    if t == "BEARISH":
        return _check("SPY trend bearish", TONE_WARNING,
                       "Price below both EMAs and slope negative — counter-trend longs need premium.")
    return _check("SPY trend mixed", TONE_INFO, "EMAs crossed / unconfirmed.")


def _check_heat(heat_pct: float | None) -> tuple[float, dict | None]:
    if heat_pct is None:
        return 1.0, None
    if heat_pct >= HEAT_FREEZE_PCT:
        return 0.0, _check(
            f"Portfolio heat {heat_pct:.1f}%",
            TONE_NEGATIVE,
            f"Above freeze threshold ({HEAT_FREEZE_PCT:.1f}%) — no new risk.",
        )
    if heat_pct >= HEAT_CLAMP_PCT:
        return 0.5, _check(
            f"Portfolio heat {heat_pct:.1f}%",
            TONE_WARNING,
            f"Above clamp threshold ({HEAT_CLAMP_PCT:.1f}%) — size at half.",
        )
    return 1.0, _check(
        f"Portfolio heat {heat_pct:.1f}%",
        TONE_POSITIVE,
        "Below clamp threshold — room for new risk.",
    )


def _check_concentration(positions: list[dict] | None, sector: str | None, direction: str | None) -> list[dict]:
    out: list[dict] = []
    if not positions:
        return out
    pos_list = positions if isinstance(positions, list) else []
    if sector and MAX_SAME_SECTOR is not None:
        same_sec = sum(
            1 for p in pos_list
            if str(p.get("sector", "")).lower() == str(sector).lower()
        )
        if same_sec >= MAX_SAME_SECTOR:
            out.append(_check(
                f"Sector concentration {same_sec}/{MAX_SAME_SECTOR}",
                TONE_WARNING,
                f"Already at or above the {MAX_SAME_SECTOR}-name cap for {sector}.",
            ))
    if direction and MAX_SAME_DIRECTION is not None:
        side = str(direction).upper()
        same_dir = sum(
            1 for p in pos_list
            if str(p.get("direction", "")).upper() == side
        )
        if same_dir >= MAX_SAME_DIRECTION:
            out.append(_check(
                f"Direction concentration {same_dir}/{MAX_SAME_DIRECTION} {side.lower()}",
                TONE_WARNING,
                "Book is already heavily one-sided.",
            ))
    return out


def _check_macro_events(today: date, calendar: dict) -> tuple[dict | None, str | None]:
    """Return ``(check, halt_reason_or_None)``.

    Halt kicks in when *any* event is within its configured window.
    """
    def _near(name: str, window: int, tone_near: str = TONE_NEGATIVE) -> tuple[int | None, dict | None]:
        delta = _nearest_event(today, calendar.get(name) or [])
        if delta is None:
            return None, None
        if 0 <= delta <= window:
            label_prefix = name.upper()
            if delta == 0:
                msg = f"{label_prefix} today"
            elif delta == 1:
                msg = f"{label_prefix} tomorrow"
            else:
                msg = f"{label_prefix} in {delta}d"
            return delta, _check(msg, tone_near, "Binary macro catalyst — halt new risk through the print.")
        if 0 <= delta <= window + 2:
            label_prefix = name.upper()
            return delta, _check(
                f"{label_prefix} in {delta}d",
                TONE_WARNING,
                "Catalyst on the horizon — size defensively.",
            )
        return delta, None

    for event, window in [
        ("fomc", HALT_ON_FOMC_WINDOW_DAYS),
        ("cpi", HALT_ON_CPI_WINDOW_DAYS),
        ("nfp", HALT_ON_NFP_WINDOW_DAYS),
    ]:
        delta, check = _near(event, window)
        if delta is not None and 0 <= delta <= window:
            return check, check["label"] if check else event.upper()
    # Soft (non-halting) catalyst checks
    for event, window in [
        ("fomc", HALT_ON_FOMC_WINDOW_DAYS),
        ("cpi", HALT_ON_CPI_WINDOW_DAYS),
        ("nfp", HALT_ON_NFP_WINDOW_DAYS),
    ]:
        delta, check = _near(event, window)
        if check is not None:
            return check, None
    return None, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_risk_regime(
    *,
    market_indicators: dict | None = None,
    positions: list[dict] | None = None,
    heat_pct: float | None = None,
    sector: str | None = None,
    direction: str | None = None,
    today: date | None = None,
    calendar_path: str | Path | None = None,
) -> dict:
    """Aggregate all sizing checks into a single regime payload.

    All arguments are optional — the function degrades to ``calm`` with
    no-op checks when inputs are missing.  This is intentional so the
    Action Bar keeps rendering even when data fetches fail upstream.

    Parameters
    ----------
    market_indicators
        Expected keys: ``vix`` (float), ``vix3m`` (float), ``spy_rsi``
        (float), ``spy_trend`` (``BULLISH``/``BEARISH``/``NEUTRAL``).
    positions
        List of open-position dicts used for concentration checks.
    heat_pct
        Current portfolio heat as a %.  ``None`` skips the heat check.
    sector / direction
        Candidate row's sector / direction to check against
        concentration caps.
    today
        Override for testing; defaults to ``date.today()``.
    calendar_path
        Override for the economic-calendar JSON; defaults to
        ``ECONOMIC_CALENDAR_PATH``.
    """
    mi = market_indicators or {}
    today = today or date.today()

    vix = _f(mi.get("vix") or mi.get("vix_close"))
    vix3m = _f(mi.get("vix3m") or mi.get("vix3m_close"))
    spy_rsi = _f(mi.get("spy_rsi"))
    spy_trend = mi.get("spy_trend")

    # Base VIX tier drives the starting multiplier.
    base_tier, base_mult, vix_check = _check_vix_level(vix)
    checks: list[dict] = [vix_check]
    multiplier = base_mult
    tier = base_tier

    # VIX term structure.
    term_check = _check_vix_term(vix, vix3m)
    if term_check is not None:
        checks.append(term_check)
        if term_check["tone"] == TONE_WARNING:
            multiplier *= 0.85
            tier = _bump_tier(tier, TIER_ELEVATED)

    # SPY trend + RSI.
    trend_check = _check_spy_trend(spy_trend)
    if trend_check is not None:
        checks.append(trend_check)
        if trend_check["tone"] == TONE_WARNING:
            multiplier *= 0.90

    rsi_check = _check_spy_rsi(spy_rsi)
    if rsi_check is not None:
        checks.append(rsi_check)
        if rsi_check["tone"] == TONE_WARNING:
            multiplier *= 0.90

    # Portfolio heat.
    heat_mult, heat_check = _check_heat(heat_pct)
    if heat_check is not None:
        checks.append(heat_check)
    if heat_mult <= 0:
        tier = TIER_HALT
        multiplier = 0.0
    else:
        multiplier *= heat_mult
        if heat_mult < 1.0:
            tier = _bump_tier(tier, TIER_ELEVATED)

    # Sector / direction concentration.
    conc_checks = _check_concentration(positions, sector, direction)
    checks.extend(conc_checks)
    for c in conc_checks:
        multiplier *= 0.75

    # Macro events.
    calendar = _load_calendar(calendar_path)
    event_check, halt_reason = _check_macro_events(today, calendar)
    if event_check is not None:
        checks.append(event_check)
    if halt_reason is not None:
        tier = TIER_HALT
        multiplier = 0.0

    # Clamp multiplier to the base VIX ceiling — tiers can't be more
    # permissive than the raw VIX tier suggests, but they can be more
    # restrictive.
    multiplier = max(0.0, min(multiplier, base_mult))

    # Final tier adjustment based on multiplier (for mult < 0.4 force panic).
    if tier != TIER_HALT:
        if multiplier <= 0.0:
            tier = TIER_HALT
        elif multiplier < 0.4:
            tier = _bump_tier(tier, TIER_PANIC)
        elif multiplier < 0.7:
            tier = _bump_tier(tier, TIER_ELEVATED)

    # Translate tier + multiplier into a 0-100 score (calm=100).
    score = int(round(multiplier * 100))
    if tier == TIER_HALT:
        score = 0

    # Assemble inputs dict for debugging / UI inspection.
    inputs = {
        "vix": vix,
        "vix3m": vix3m,
        "spy_rsi": spy_rsi,
        "spy_trend": spy_trend,
        "heat_pct": heat_pct,
        "sector": sector,
        "direction": direction,
        "positions_count": len(positions) if positions else 0,
    }

    return {
        "tier": tier,
        "tier_label": TIER_LABELS.get(tier, tier.title()),
        "score": score,
        "multiplier": round(multiplier, 3),
        "checks": checks,
        "halt_reason": halt_reason,
        "inputs": inputs,
    }


def _bump_tier(current: str, candidate: str) -> str:
    """Move tier up to whichever of ``current`` / ``candidate`` is worse."""
    return current if TIER_RANK.get(current, 0) >= TIER_RANK.get(candidate, 0) else candidate


def summarise_for_ui(regime: dict) -> dict:
    """Return a small payload optimised for the Action Bar pill.

    The full ``compute_risk_regime`` payload may be large; UI surfaces
    that only need "pill text + multiplier" should read this instead.
    """
    if not regime or not isinstance(regime, dict):
        return {"tier": TIER_CALM, "label": TIER_LABELS[TIER_CALM], "score": 100, "multiplier": 1.0}
    return {
        "tier": regime.get("tier") or TIER_CALM,
        "label": regime.get("tier_label") or TIER_LABELS.get(regime.get("tier") or TIER_CALM, "Calm"),
        "score": int(regime.get("score") or 0),
        "multiplier": float(regime.get("multiplier") or 0.0),
        "halt_reason": regime.get("halt_reason"),
    }


__all__ = [
    "compute_risk_regime",
    "summarise_for_ui",
    "TIER_CALM",
    "TIER_ELEVATED",
    "TIER_PANIC",
    "TIER_HALT",
    "TIER_LABELS",
    "TONE_POSITIVE",
    "TONE_NEUTRAL",
    "TONE_WARNING",
    "TONE_NEGATIVE",
    "TONE_INFO",
]

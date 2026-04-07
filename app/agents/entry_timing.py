"""Entry/Timing Agent.

Assesses whether the current bar represents a good entry point: detects
chasing, gap risk, bar quality, and whether the entry window has passed.

Shadow mode: logs assessment but does NOT modify scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.base import BaseAgent
from app.agents.schemas import EntryTimingAssessment

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "entry_timing.md"

_agent: BaseAgent | None = None


def _get_agent() -> BaseAgent:
    global _agent
    if _agent is None:
        _agent = BaseAgent(
            name="entry_timing",
            model="gpt-4o-mini",
            system_prompt_path=_PROMPT_PATH,
            schema=EntryTimingAssessment,
        )
    return _agent


def is_agent_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _fmt(val, fmt_str: str = ".2f", fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    try:
        return f"{val:{fmt_str}}"
    except (TypeError, ValueError):
        return str(val)


def build_entry_timing_summary(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    price_snapshot: dict,
    trade_plan: dict,
    flow_score: float | None = None,
) -> dict:
    """Build structured input for the Entry/Timing Agent."""
    atr_val = atr or 0.0
    lines: list[str] = []
    lines.append(f"Ticker: {ticker}  Direction: {direction}  Price: ${price:.2f}  ATR: {_fmt(atr)}")
    lines.append("")

    # --- Entry zone ---
    lines.append("=== ENTRY ZONE ===")
    entry = trade_plan.get("entry_price")
    zone_lo = trade_plan.get("entry_zone_low")
    zone_hi = trade_plan.get("entry_zone_high")
    lines.append(f"Entry price: ${_fmt(entry)}")
    if zone_lo and zone_hi:
        lines.append(f"Entry zone: ${_fmt(zone_lo)} – ${_fmt(zone_hi)}")
    in_zone = "yes"
    if entry and zone_lo and zone_hi:
        if price < zone_lo or price > zone_hi:
            in_zone = "no"
    lines.append(f"Price currently in entry zone: {in_zone}")
    lines.append("")

    # --- Extension from EMA ---
    lines.append("=== EXTENSION / EMA DISTANCE ===")
    sc = price_snapshot.get("score_components", {})
    ext_score = sc.get("extension")
    lines.append(f"Extension score (0-1, higher=closer to EMA): {_fmt(ext_score)}")

    checks = price_snapshot.get("checks_passed", "")
    failed = price_snapshot.get("checks_failed", "")
    if checks:
        lines.append(f"Checks passed: {checks}")
    if failed:
        lines.append(f"Checks failed: {failed}")
    lines.append("")

    # --- Pattern type ---
    lines.append("=== PATTERN ===")
    reasons = price_snapshot.get("reasons", [])
    if isinstance(reasons, list):
        lines.append(f"Pattern: {', '.join(reasons)}")
    else:
        lines.append(f"Pattern: {reasons}")

    broken = price_snapshot.get("broken_level")
    if broken:
        lines.append(f"Broken level: ${_fmt(broken)}")
    lines.append("")

    # --- Bar quality context ---
    lines.append("=== BAR QUALITY ===")
    mom_score = sc.get("momentum")
    vol_score = sc.get("confirm_vol")
    lines.append(f"Momentum score (0-2, close position in range): {_fmt(mom_score)}")
    lines.append(f"Volume confirmation (0-1): {_fmt(vol_score)}")
    lines.append("")

    # --- Targets for chasing reference ---
    lines.append("=== TARGETS (chasing reference) ===")
    t1 = trade_plan.get("target_1")
    stop = trade_plan.get("stop_price")
    lines.append(f"T1: ${_fmt(t1)}")
    lines.append(f"Stop: ${_fmt(stop)}")
    if t1 and entry and direction == "LONG" and price > (t1 or 0):
        lines.append("WARNING: Current price is ABOVE T1 — extreme chasing risk")
    elif t1 and entry and direction == "SHORT" and price < (t1 or float("inf")):
        lines.append("WARNING: Current price is BELOW T1 — extreme chasing risk")
    lines.append("")

    # --- Flow conviction (context) ---
    if flow_score is not None:
        lines.append(f"Flow score: {_fmt(flow_score)}/10")

    return {
        "ticker": ticker,
        "direction": direction,
        "price": round(price, 2),
        "summary": "\n".join(lines),
    }


def run_entry_timing_shadow(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    price_snapshot: dict,
    trade_plan: dict,
    flow_score: float | None = None,
) -> EntryTimingAssessment | None:
    """Run the Entry/Timing Agent in shadow mode.  Never raises."""
    if not is_agent_available():
        return None
    try:
        agent = _get_agent()
        input_dict = build_entry_timing_summary(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr,
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
            flow_score=flow_score,
        )
        assessment = agent.call(input_dict)
        agent.log_shadow(ticker, direction, assessment.model_dump())
        return assessment
    except Exception as e:
        print(f"  [agent:entry_timing] {ticker} failed: {e}")
        return None

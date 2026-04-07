"""S/R Quality Agent.

Assesses the quality of algorithmically identified support/resistance levels
and proposes additional levels the algorithm may have missed (gaps, round
numbers, volume zones, prior ATH/ATL).

Shadow mode: logs assessment but does NOT modify scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.base import BaseAgent
from app.agents.schemas import SRQualityOutput

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "sr_quality.md"

_agent: BaseAgent | None = None


def _get_agent() -> BaseAgent:
    global _agent
    if _agent is None:
        _agent = BaseAgent(
            name="sr_quality",
            model="gpt-4o-mini",
            system_prompt_path=_PROMPT_PATH,
            schema=SRQualityOutput,
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


def build_sr_summary(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    price_snapshot: dict,
    trade_plan: dict,
) -> dict:
    """Build structured input for the S/R Quality Agent."""
    atr_val = atr or 0.0
    lines: list[str] = []
    lines.append(f"Ticker: {ticker}  Direction: {direction}  Price: ${price:.2f}  ATR: {_fmt(atr)}")
    lines.append("")

    # --- Algo-identified levels ---
    lines.append("=== ALGO-IDENTIFIED LEVELS ===")
    support = price_snapshot.get("support")
    resistance = price_snapshot.get("resistance")
    struct_s = price_snapshot.get("structural_support")
    struct_r = price_snapshot.get("structural_resistance")

    def _dist(level):
        if level is None or atr_val <= 0:
            return ""
        return f" ({abs(price - level) / atr_val:.1f} ATR from price)"

    lines.append(f"Tactical support:      {_fmt(support)}{_dist(support)}")
    lines.append(f"Tactical resistance:   {_fmt(resistance)}{_dist(resistance)}")
    lines.append(f"Structural support:    {_fmt(struct_s)}{_dist(struct_s)}")
    lines.append(f"Structural resistance: {_fmt(struct_r)}{_dist(struct_r)}")

    broken = price_snapshot.get("broken_level")
    if broken:
        lines.append(f"Broken level (breakout): {_fmt(broken)}")
    lines.append("")

    # --- Pattern context ---
    lines.append("=== PATTERN CONTEXT ===")
    reasons = price_snapshot.get("reasons", [])
    pattern_str = ", ".join(reasons) if isinstance(reasons, list) else str(reasons)
    lines.append(f"Pattern: {pattern_str}")

    sc = price_snapshot.get("score_components", {})
    lines.append(f"Trend score: {_fmt(sc.get('trend'))}")
    lines.append(f"Extension score: {_fmt(sc.get('extension'))}")
    lines.append(f"Room score: {_fmt(sc.get('room'))}")
    lines.append(f"Pattern score: {_fmt(sc.get('pattern'))}")
    lines.append(f"Momentum score: {_fmt(sc.get('momentum'))}")
    lines.append(f"Volume confirmation: {_fmt(sc.get('confirm_vol'))}")
    lines.append("")

    # --- Trade plan levels ---
    lines.append("=== TRADE PLAN LEVELS ===")
    lines.append(f"Entry: {_fmt(trade_plan.get('entry_price'))}")
    lines.append(f"Stop:  {_fmt(trade_plan.get('stop_price'))}{_dist(trade_plan.get('stop_price'))}")
    lines.append(f"T1:    {_fmt(trade_plan.get('target_1'))}{_dist(trade_plan.get('target_1'))}")
    lines.append(f"T2:    {_fmt(trade_plan.get('target_2'))}{_dist(trade_plan.get('target_2'))}")
    lines.append(f"R:R:   {_fmt(trade_plan.get('rr_ratio'))}")
    lines.append("")

    # --- Options walls (if available) ---
    opts_meta = trade_plan.get("options_context") or {}
    call_wall = opts_meta.get("call_wall")
    put_wall = opts_meta.get("put_wall")
    if call_wall or put_wall:
        lines.append("=== OPTIONS WALLS ===")
        lines.append(f"Call wall: {_fmt(call_wall)}{_dist(call_wall)}")
        lines.append(f"Put wall:  {_fmt(put_wall)}{_dist(put_wall)}")

    return {
        "ticker": ticker,
        "direction": direction,
        "price": round(price, 2),
        "summary": "\n".join(lines),
    }


def run_sr_quality_shadow(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    price_snapshot: dict,
    trade_plan: dict,
) -> SRQualityOutput | None:
    """Run the S/R Quality Agent in shadow mode.

    Returns the assessment or None if unavailable.  Never raises.
    """
    if not is_agent_available():
        return None
    try:
        agent = _get_agent()
        input_dict = build_sr_summary(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr,
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
        )
        assessment = agent.call(input_dict)
        agent.log_shadow(ticker, direction, assessment.model_dump())
        return assessment
    except Exception as e:
        print(f"  [agent:sr_quality] {ticker} failed: {e}")
        return None

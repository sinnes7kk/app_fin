"""Trade Plan Agent.

Evaluates the algorithmically generated trade plan (stop, targets, R:R) against
the S/R landscape and options walls.  Suggests improvements when the plan has
structural flaws.

Shadow mode: logs assessment but does NOT modify scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.base import BaseAgent
from app.agents.schemas import TradePlanAssessment

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "trade_plan.md"

_agent: BaseAgent | None = None


def _get_agent() -> BaseAgent:
    global _agent
    if _agent is None:
        _agent = BaseAgent(
            name="trade_plan",
            model="gpt-4o-mini",
            system_prompt_path=_PROMPT_PATH,
            schema=TradePlanAssessment,
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


def build_trade_plan_summary(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    price_snapshot: dict,
    trade_plan: dict,
    opts_ctx: dict | None = None,
) -> dict:
    """Build structured input for the Trade Plan Agent."""
    atr_val = atr or 0.0
    lines: list[str] = []
    lines.append(f"Ticker: {ticker}  Direction: {direction}  Price: ${price:.2f}  ATR: {_fmt(atr)}")
    lines.append("")

    def _dist(level):
        if level is None or atr_val <= 0:
            return ""
        return f" ({abs(price - level) / atr_val:.1f} ATR)"

    # --- Algo trade plan ---
    lines.append("=== ALGO TRADE PLAN ===")
    entry = trade_plan.get("entry_price")
    stop = trade_plan.get("stop_price")
    t1 = trade_plan.get("target_1")
    t2 = trade_plan.get("target_2")
    rr = trade_plan.get("rr_ratio")
    risk = trade_plan.get("risk_per_share")
    time_stop = trade_plan.get("time_stop_days")

    lines.append(f"Entry: ${_fmt(entry)}")
    lines.append(f"Stop:  ${_fmt(stop)}{_dist(stop)}")
    lines.append(f"T1:    ${_fmt(t1)}{_dist(t1)}")
    lines.append(f"T2:    ${_fmt(t2)}{_dist(t2)}")
    lines.append(f"R:R:   {_fmt(rr)}")
    lines.append(f"Risk/share: ${_fmt(risk)}")
    lines.append(f"Time stop: {time_stop} days")
    lines.append("")

    # --- S/R landscape ---
    lines.append("=== S/R LANDSCAPE ===")
    support = price_snapshot.get("support")
    resistance = price_snapshot.get("resistance")
    struct_s = price_snapshot.get("structural_support")
    struct_r = price_snapshot.get("structural_resistance")
    lines.append(f"Tactical support:      ${_fmt(support)}{_dist(support)}")
    lines.append(f"Tactical resistance:   ${_fmt(resistance)}{_dist(resistance)}")
    lines.append(f"Structural support:    ${_fmt(struct_s)}{_dist(struct_s)}")
    lines.append(f"Structural resistance: ${_fmt(struct_r)}{_dist(struct_r)}")

    broken = price_snapshot.get("broken_level")
    if broken:
        lines.append(f"Broken level: ${_fmt(broken)}")
    lines.append("")

    # --- Pattern ---
    lines.append("=== PATTERN ===")
    reasons = price_snapshot.get("reasons", [])
    if isinstance(reasons, list):
        lines.append(f"Pattern: {', '.join(reasons)}")
    else:
        lines.append(f"Pattern: {reasons}")
    lines.append("")

    # --- Options walls ---
    plan_opts = trade_plan.get("options_context") or {}
    call_wall = plan_opts.get("call_wall")
    put_wall = plan_opts.get("put_wall")
    gamma = plan_opts.get("gamma_regime")
    if call_wall or put_wall:
        lines.append("=== OPTIONS WALLS ===")
        lines.append(f"Call wall: ${_fmt(call_wall)}{_dist(call_wall)}")
        lines.append(f"Put wall:  ${_fmt(put_wall)}{_dist(put_wall)}")
        if gamma:
            lines.append(f"Gamma regime: {gamma}")
        wall_warn = plan_opts.get("wall_proximity_warning")
        if wall_warn:
            lines.append(f"Wall warning: {wall_warn}")
        lines.append("")

    # --- IV context (affects hold time) ---
    if opts_ctx:
        iv_rank = opts_ctx.get("iv_rank")
        if iv_rank is not None:
            lines.append(f"IV rank: {_fmt(iv_rank, '.0f')}")

    return {
        "ticker": ticker,
        "direction": direction,
        "price": round(price, 2),
        "summary": "\n".join(lines),
    }


def run_trade_plan_shadow(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    price_snapshot: dict,
    trade_plan: dict,
    opts_ctx: dict | None = None,
) -> TradePlanAssessment | None:
    """Run the Trade Plan Agent in shadow mode.  Never raises."""
    if not is_agent_available():
        return None
    try:
        agent = _get_agent()
        input_dict = build_trade_plan_summary(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr,
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
            opts_ctx=opts_ctx,
        )
        assessment = agent.call(input_dict)
        agent.log_shadow(ticker, direction, assessment.model_dump())
        return assessment
    except Exception as e:
        print(f"  [agent:trade_plan] {ticker} failed: {e}")
        return None

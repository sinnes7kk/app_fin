"""Devil's Advocate Agent.

Adversarial risk analyst that finds reasons NOT to take a trade: earnings
risk, trap patterns, liquidity concerns, concentration risk, post-catalyst
setups, and crowded trades.

Shadow mode: logs assessment but does NOT modify scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.base import BaseAgent
from app.agents.schemas import DevilsAdvocateAssessment

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "devils_advocate.md"

_agent: BaseAgent | None = None


def _get_agent() -> BaseAgent:
    global _agent
    if _agent is None:
        _agent = BaseAgent(
            name="devils_advocate",
            model="gpt-4o-mini",
            system_prompt_path=_PROMPT_PATH,
            schema=DevilsAdvocateAssessment,
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


def _dollar(val, fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    if abs(val) >= 1_000_000_000:
        return f"${val / 1_000_000_000:.1f}B"
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:.0f}K"
    return f"${val:.0f}"


def build_devils_advocate_summary(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    final_score: float | None,
    flow_score: float | None,
    price_score: float | None,
    options_score: float | None,
    price_snapshot: dict,
    trade_plan: dict,
    flow_snapshot: dict | None,
    opts_ctx: dict | None,
    counter_trend: bool = False,
    sector: str | None = None,
) -> dict:
    """Build structured input for the Devil's Advocate Agent."""
    lines: list[str] = []
    lines.append(f"Ticker: {ticker}  Direction: {direction}  Price: ${price:.2f}  ATR: {_fmt(atr)}")
    lines.append(f"Counter-trend: {'YES' if counter_trend else 'no'}")
    if sector:
        lines.append(f"Sector: {sector}")
    lines.append("")

    # --- Scores ---
    lines.append("=== SCORES (the bull case) ===")
    lines.append(f"Final score: {_fmt(final_score)}/10")
    lines.append(f"Flow score:  {_fmt(flow_score)}/10")
    lines.append(f"Price score: {_fmt(price_score)}/10")
    lines.append(f"Options score: {_fmt(options_score)}/10")
    lines.append("")

    # --- Pattern ---
    lines.append("=== PATTERN ===")
    reasons = price_snapshot.get("reasons", [])
    if isinstance(reasons, list):
        lines.append(f"Pattern: {', '.join(reasons)}")
    else:
        lines.append(f"Pattern: {reasons}")

    sc = price_snapshot.get("score_components", {})
    lines.append(f"Trend: {_fmt(sc.get('trend'))}  Extension: {_fmt(sc.get('extension'))}  "
                 f"Room: {_fmt(sc.get('room'))}  Pattern: {_fmt(sc.get('pattern'))}")
    lines.append(f"Momentum: {_fmt(sc.get('momentum'))}  Volume: {_fmt(sc.get('confirm_vol'))}")

    broken = price_snapshot.get("broken_level")
    if broken:
        lines.append(f"Broken level: ${_fmt(broken)}")
    lines.append("")

    # --- Trade plan ---
    lines.append("=== TRADE PLAN ===")
    lines.append(f"Entry: ${_fmt(trade_plan.get('entry_price'))}")
    lines.append(f"Stop: ${_fmt(trade_plan.get('stop_price'))}")
    lines.append(f"T1: ${_fmt(trade_plan.get('target_1'))}  T2: ${_fmt(trade_plan.get('target_2'))}")
    lines.append(f"R:R: {_fmt(trade_plan.get('rr_ratio'))}")
    lines.append("")

    # --- Flow data ---
    lines.append("=== FLOW DATA ===")
    if flow_snapshot:
        for key in [
            "bullish_flow_intensity", "bearish_flow_intensity",
            "bullish_sweep_count", "bearish_sweep_count",
            "bullish_repeat_count", "bearish_repeat_count",
            "bullish_ppt_bps", "bearish_ppt_bps",
        ]:
            val = flow_snapshot.get(key)
            if val is not None:
                lines.append(f"  {key}: {_fmt(val)}")
    else:
        lines.append("No flow data available")
    lines.append("")

    # --- Options context ---
    lines.append("=== OPTIONS CONTEXT ===")
    if opts_ctx:
        iv_rank = opts_ctx.get("iv_rank")
        iv_current = opts_ctx.get("iv_current")
        gamma = opts_ctx.get("gamma_regime")
        pcr = opts_ctx.get("ticker_put_call_ratio")
        lines.append(f"IV rank: {_fmt(iv_rank, '.0f')}  IV current: {_fmt(iv_current, '.1f')}%")
        lines.append(f"Gamma regime: {gamma or 'N/A'}  P/C ratio: {_fmt(pcr)}")

        call_wall = opts_ctx.get("nearest_call_wall")
        put_wall = opts_ctx.get("nearest_put_wall")
        lines.append(f"Call wall: ${_fmt(call_wall)}  Put wall: ${_fmt(put_wall)}")
    else:
        lines.append("No options context available")

    return {
        "ticker": ticker,
        "direction": direction,
        "price": round(price, 2),
        "summary": "\n".join(lines),
    }


def run_devils_advocate_shadow(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    final_score: float | None,
    flow_score: float | None,
    price_score: float | None,
    options_score: float | None,
    price_snapshot: dict,
    trade_plan: dict,
    flow_snapshot: dict | None,
    opts_ctx: dict | None,
    counter_trend: bool = False,
    sector: str | None = None,
) -> DevilsAdvocateAssessment | None:
    """Run the Devil's Advocate Agent in shadow mode.  Never raises."""
    if not is_agent_available():
        return None
    try:
        agent = _get_agent()
        input_dict = build_devils_advocate_summary(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr,
            final_score=final_score,
            flow_score=flow_score,
            price_score=price_score,
            options_score=options_score,
            price_snapshot=price_snapshot,
            trade_plan=trade_plan,
            flow_snapshot=flow_snapshot,
            opts_ctx=opts_ctx,
            counter_trend=counter_trend,
            sector=sector,
        )
        assessment = agent.call(input_dict)
        agent.log_shadow(ticker, direction, assessment.model_dump())
        return assessment
    except Exception as e:
        print(f"  [agent:devils_advocate] {ticker} failed: {e}")
        return None

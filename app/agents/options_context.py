"""Options Context Confirmation Agent.

Replaces (in shadow mode) the deterministic ``compute_options_context_score()``
and enrichment bonuses with LLM-based judgment.  Detects hedging ambiguity,
signal conflicts across gamma/walls/OI/dark-pool/net-premium, and context-
dependent IV interpretation.

Shadow mode: the agent logs its assessment but does NOT modify scoring.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.agents.base import BaseAgent
from app.agents.schemas import OptionsContextAssessment

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "options_context.md"

_agent: BaseAgent | None = None


def _get_agent() -> BaseAgent:
    global _agent
    if _agent is None:
        _agent = BaseAgent(
            name="options_context",
            model="gpt-4o-mini",
            system_prompt_path=_PROMPT_PATH,
            schema=OptionsContextAssessment,
        )
    return _agent


def is_agent_available() -> bool:
    """Return True if an OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Summary builder: converts raw data dicts into the structured text the
# agent receives.  All computation happens here — the agent only judges.
# ---------------------------------------------------------------------------

def _fmt(val, fmt_str: str = ".2f", fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    try:
        return f"{val:{fmt_str}}"
    except (TypeError, ValueError):
        return str(val)


def _pct(val, fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    return f"{val:.1f}%"


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


def _int(val, fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    return f"{int(val):,}"


def build_options_summary(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    flow_summary: dict,
    opts_ctx: dict,
    dark_pool: dict | None,
    net_prem_ticks: dict | None,
) -> dict:
    """Build the structured input dict for the Options Context Agent.

    Returns a dict ready to be passed to ``agent.call()``.  The dict is also
    serialisable for cache hashing.
    """
    atr_val = atr or 0.0

    flip = opts_ctx.get("gamma_flip_level_estimate")
    flip_dist_atr = ""
    if flip is not None and atr_val > 0:
        d = abs(price - flip) / atr_val
        flip_dist_atr = f" ({d:.2f} ATR from price)"

    lines: list[str] = []
    lines.append(f"Ticker: {ticker}  Direction: {direction}  Price: ${price:.2f}")
    lines.append("")

    # --- Flow summary (read-only) ---
    lines.append("=== FLOW SCORE SUMMARY (read-only, pre-computed) ===")
    bull_sc = flow_summary.get("bullish_score", "N/A")
    bear_sc = flow_summary.get("bearish_score", "N/A")
    if direction == "LONG":
        lines.append(f"Bullish flow score: {_fmt(bull_sc)}/10")
        lines.append(f"Bearish flow score: {_fmt(bear_sc)}/10")
    else:
        lines.append(f"Bearish flow score: {_fmt(bear_sc)}/10")
        lines.append(f"Bullish flow score: {_fmt(bull_sc)}/10")
    dominant = flow_summary.get("dominant_components", "")
    if dominant:
        lines.append(f"Dominant components: {dominant}")
    lines.append("")

    # --- Gamma / GEX ---
    lines.append("=== GAMMA / GEX ===")
    regime = opts_ctx.get("gamma_regime", "N/A")
    net_gex = opts_ctx.get("net_gex")
    lines.append(f"Regime: {regime}  Net GEX: {_dollar(net_gex)}")
    lines.append(f"Flip estimate: {_fmt(flip, '.2f')}{flip_dist_atr}")
    lines.append("")

    # --- Walls ---
    lines.append("=== OPTIONS WALLS ===")
    call_wall = opts_ctx.get("nearest_call_wall")
    put_wall = opts_ctx.get("nearest_put_wall")
    dist_call = opts_ctx.get("distance_to_call_wall_pct")
    dist_put = opts_ctx.get("distance_to_put_wall_pct")
    cw_atr = f" ({abs(call_wall - price) / atr_val:.1f} ATR)" if call_wall and atr_val > 0 else ""
    pw_atr = f" ({abs(price - put_wall) / atr_val:.1f} ATR)" if put_wall and atr_val > 0 else ""
    lines.append(f"Nearest call wall: {_fmt(call_wall, '.2f')} ({_pct(dist_call)} above){cw_atr}")
    lines.append(f"Nearest put wall:  {_fmt(put_wall, '.2f')} ({_pct(dist_put)} below){pw_atr}")
    lines.append("")

    # --- OI structure ---
    lines.append("=== OPEN INTEREST STRUCTURE ===")
    call_oi = opts_ctx.get("ticker_call_oi")
    put_oi = opts_ctx.get("ticker_put_oi")
    pcr = opts_ctx.get("ticker_put_call_ratio")
    near_oi = opts_ctx.get("near_term_oi")
    swing_oi = opts_ctx.get("swing_dte_oi")
    long_oi = opts_ctx.get("long_dated_oi")
    lines.append(f"Call OI: {_int(call_oi)}  Put OI: {_int(put_oi)}  P/C ratio: {_fmt(pcr)}")
    lines.append(f"Near-term OI (0-7 DTE): {_int(near_oi)}  Swing OI (30-90 DTE): {_int(swing_oi)}")
    lines.append(f"Long-dated OI (90+ DTE): {_int(long_oi)}")
    swing_dominant = "yes" if (swing_oi or 0) > (near_oi or 0) and (near_oi or 0) > 0 else "no"
    lines.append(f"Swing DTE dominant: {swing_dominant}")
    lines.append("")

    # --- Premium / volume ---
    lines.append("=== DAILY PREMIUM / VOLUME ===")
    bull_p = opts_ctx.get("daily_bullish_premium")
    bear_p = opts_ctx.get("daily_bearish_premium")
    total_p = (bull_p or 0) + (bear_p or 0)
    bias_pct = ""
    if total_p > 0 and bull_p is not None and bear_p is not None:
        aligned = bull_p / total_p if direction == "LONG" else bear_p / total_p
        bias_pct = f"  Aligned share: {aligned:.1%}"
    lines.append(f"Daily bullish premium: {_dollar(bull_p)}  Daily bearish premium: {_dollar(bear_p)}{bias_pct}")
    cv = opts_ctx.get("call_volume_today")
    pv = opts_ctx.get("put_volume_today")
    cva = opts_ctx.get("call_volume_vs_30d_avg")
    pva = opts_ctx.get("put_volume_vs_30d_avg")
    lines.append(f"Call volume: {_int(cv)} ({_fmt(cva, '.1f')}x 30d avg)  Put volume: {_int(pv)} ({_fmt(pva, '.1f')}x 30d avg)")
    cab = opts_ctx.get("call_ask_bid_ratio")
    pab = opts_ctx.get("put_ask_bid_ratio")
    lines.append(f"Call ask/bid ratio: {_fmt(cab)}  Put ask/bid ratio: {_fmt(pab)}")
    lines.append("")

    # --- IV ---
    lines.append("=== IMPLIED VOLATILITY ===")
    iv_rank = opts_ctx.get("iv_rank")
    iv_current = opts_ctx.get("iv_current")
    lines.append(f"IV rank (30d): {_fmt(iv_rank, '.0f')}  IV current: {_pct(iv_current)}")
    lines.append("")

    # --- Dark pool ---
    lines.append("=== DARK POOL ===")
    if dark_pool is not None:
        dp_bias = dark_pool.get("dark_pool_bias")
        dp_vol = dark_pool.get("dark_pool_volume")
        dp_prints = dark_pool.get("large_print_count")
        bias_label = "neutral"
        if dp_bias is not None:
            if dp_bias > 0.6:
                bias_label = "bullish-leaning"
            elif dp_bias < 0.4:
                bias_label = "bearish-leaning"
        lines.append(f"Dark pool bias: {_fmt(dp_bias)} ({bias_label})")
        lines.append(f"Dark pool volume: {_dollar(dp_vol)}  Large prints: {_int(dp_prints)}")
    else:
        lines.append("No dark pool data available")
    lines.append("")

    # --- Net premium ticks ---
    lines.append("=== INTRADAY NET PREMIUM ===")
    if net_prem_ticks is not None:
        prem_dir = net_prem_ticks.get("intraday_premium_direction")
        dir_label = "neutral"
        if prem_dir is not None:
            if prem_dir > 0.6:
                dir_label = "bullish-leaning"
            elif prem_dir < 0.4:
                dir_label = "bearish-leaning"
        delta_mom = net_prem_ticks.get("delta_momentum")
        net_delta = net_prem_ticks.get("net_delta")
        lines.append(f"Intraday premium direction: {_fmt(prem_dir)} ({dir_label})")
        lines.append(f"Delta momentum: {_fmt(delta_mom)}  Net delta: {_int(net_delta)}")
    else:
        lines.append("No intraday net premium data available")

    summary_text = "\n".join(lines)

    return {
        "ticker": ticker,
        "direction": direction,
        "price": round(price, 2),
        "summary": summary_text,
    }


def build_flow_summary(row) -> dict:
    """Extract a concise flow summary from a ranked flow row.

    Returns a dict with keys used by ``build_options_summary``.
    """
    summary: dict = {}

    bull = row.get("bullish_score")
    if bull is not None:
        try:
            summary["bullish_score"] = round(float(bull), 2)
        except (TypeError, ValueError):
            pass

    bear = row.get("bearish_score")
    if bear is not None:
        try:
            summary["bearish_score"] = round(float(bear), 2)
        except (TypeError, ValueError):
            pass

    components: list[str] = []
    for col, label in [
        ("bullish_flow_intensity", "flow_intensity"),
        ("bearish_flow_intensity", "flow_intensity"),
        ("bullish_repeat_count", "repeat"),
        ("bearish_repeat_count", "repeat"),
        ("bullish_sweep_count", "sweeps"),
        ("bearish_sweep_count", "sweeps"),
    ]:
        val = row.get(col)
        if val is not None:
            try:
                v = float(val)
                if v > 0:
                    components.append(f"{label}={v:.2f}")
            except (TypeError, ValueError):
                pass

    if components:
        summary["dominant_components"] = ", ".join(dict.fromkeys(components))

    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_options_context_shadow(
    ticker: str,
    direction: str,
    price: float,
    atr: float | None,
    flow_row,
    opts_ctx: dict,
    dark_pool: dict | None,
    net_prem_ticks: dict | None,
) -> OptionsContextAssessment | None:
    """Run the Options Context Agent in shadow mode.

    Returns the assessment (for logging) or None if the agent is unavailable
    (no API key, import error, etc.).  Never raises — failures are caught and
    logged.
    """
    if not is_agent_available():
        return None

    try:
        agent = _get_agent()

        flow_summary = build_flow_summary(flow_row) if flow_row is not None else {}
        input_dict = build_options_summary(
            ticker=ticker,
            direction=direction,
            price=price,
            atr=atr,
            flow_summary=flow_summary,
            opts_ctx=opts_ctx,
            dark_pool=dark_pool,
            net_prem_ticks=net_prem_ticks,
        )

        assessment = agent.call(input_dict)

        agent.log_shadow(ticker, direction, assessment.model_dump())

        return assessment

    except Exception as e:
        print(f"  [agent:options_context] {ticker} failed: {e}")
        return None

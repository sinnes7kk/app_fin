"""Pydantic output schemas for all V3 agents.

Each schema defines the structured JSON response an agent must return.
OpenAI's ``response_format`` enforces the schema at generation time so
there is no free-text parsing.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# S/R Quality Agent (placeholder for future implementation)
# ---------------------------------------------------------------------------

class LevelAssessment(BaseModel):
    level_price: float
    source: Literal["algo", "agent_proposed"]
    touch_count: int
    quality: Literal["institutional", "structural", "tactical", "noise"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=200)
    volume_confirmed: bool
    clean_rejections: bool
    recently_violated: bool


class ProposedLevel(BaseModel):
    level_price: float
    level_type: Literal["gap", "round_number", "volume_zone", "prior_ath_atl", "earnings_reaction"]
    role: Literal["support", "resistance", "both"]
    significance: Literal["high", "medium", "low"]
    reasoning: str = Field(max_length=150)


class SRQualityOutput(BaseModel):
    support_assessment: LevelAssessment
    resistance_assessment: LevelAssessment
    structural_support_assessment: LevelAssessment
    structural_resistance_assessment: LevelAssessment
    proposed_levels: List[ProposedLevel] = Field(
        default_factory=list, max_length=3,
        description="Additional levels the algo missed. Only propose if genuinely significant.",
    )
    overall_sr_quality: float = Field(ge=0.0, le=10.0)
    key_level_for_trade: float
    key_level_quality: Literal["institutional", "structural", "tactical", "noise"]
    key_level_source: Literal["algo", "agent_proposed"]
    invalidation_level: float
    better_stop_level: Optional[float] = Field(default=None)
    reasoning: str = Field(max_length=300)


# ---------------------------------------------------------------------------
# Options Context Confirmation Agent
# ---------------------------------------------------------------------------

class OptionsContextAssessment(BaseModel):
    """Structured output for the Options Context Confirmation Agent."""

    directional_conviction: float = Field(
        ge=0.0, le=10.0,
        description="How strongly the options landscape supports the thesis direction",
    )

    hedging_probability: float = Field(
        ge=0.0, le=1.0,
        description="Likelihood that observed flow/OI is hedging, not directional",
    )
    hedging_reasoning: str = Field(max_length=150)

    signal_consistency: Literal["consistent", "mixed", "conflicting"]
    consistency_detail: str = Field(
        max_length=150,
        description="Which signals agree/disagree and why",
    )

    gamma_significance: Literal["high", "medium", "low", "irrelevant"]
    gamma_reasoning: str = Field(max_length=100)

    wall_impact: Literal["supportive", "neutral", "blocking"]
    wall_reasoning: str = Field(max_length=100)

    iv_assessment: Literal["cheap_and_active", "fair", "expensive", "crushed", "no_data"]
    iv_reasoning: str = Field(max_length=100)

    institutional_confidence: Literal["high", "medium", "low"]
    institutional_reasoning: str = Field(max_length=100)

    dark_pool_alignment: Literal["confirming", "neutral", "contradicting", "no_data"]
    intraday_flow_alignment: Literal["confirming", "neutral", "contradicting", "no_data"]

    key_concern: str = Field(
        max_length=200,
        description="Primary risk the options structure reveals, or 'none'",
    )
    reasoning: str = Field(max_length=300)


# ---------------------------------------------------------------------------
# Trade Plan Agent (placeholder for future implementation)
# ---------------------------------------------------------------------------

class TradePlanAssessment(BaseModel):
    """Structured output for the Trade Plan Agent."""

    stop_quality: Literal["good", "too_tight", "too_wide", "wrong_level"]
    stop_reasoning: str = Field(max_length=150)
    suggested_stop: Optional[float] = Field(default=None)

    t1_quality: Literal["good", "too_ambitious", "too_conservative", "wrong_level"]
    t1_reasoning: str = Field(max_length=150)
    suggested_t1: Optional[float] = Field(default=None)

    t2_quality: Literal["good", "too_ambitious", "too_conservative"]
    suggested_t2: Optional[float] = Field(default=None)

    rr_assessment: Literal["favorable", "marginal", "unfavorable"]
    true_rr_estimate: float = Field(ge=0.0, le=20.0)

    hold_time_suggestion: int = Field(ge=3, le=25)
    hold_reasoning: str = Field(max_length=100)

    partial_at_t1_pct: float = Field(ge=0.0, le=1.0)

    plan_score: float = Field(ge=0.0, le=10.0)
    reasoning: str = Field(max_length=300)


# ---------------------------------------------------------------------------
# Entry/Timing Agent (placeholder for future implementation)
# ---------------------------------------------------------------------------

class EntryTimingAssessment(BaseModel):
    """Structured output for the Entry/Timing Agent."""

    entry_timing: Literal[
        "enter_now", "wait_for_retest", "wait_for_pullback", "skip_window",
    ]
    confidence: float = Field(ge=0.0, le=1.0)

    chasing_risk: Literal["low", "medium", "high"]
    chasing_reasoning: str = Field(max_length=100)

    gap_risk: Literal["low", "medium", "high"]
    gap_reasoning: str = Field(max_length=100)

    bar_quality: Literal["strong", "acceptable", "weak"]
    bar_reasoning: str = Field(max_length=100)

    if_waiting_target: Optional[float] = Field(default=None)
    if_waiting_max_bars: Optional[int] = Field(default=None, ge=1, le=10)

    entry_score: float = Field(ge=0.0, le=10.0)
    reasoning: str = Field(max_length=300)


# ---------------------------------------------------------------------------
# Devil's Advocate Agent (placeholder for future implementation)
# ---------------------------------------------------------------------------

class DevilsAdvocateAssessment(BaseModel):
    """Structured output for the Devil's Advocate Agent."""

    risk_score: float = Field(
        ge=0.0, le=10.0,
        description="Overall risk. 10 = do NOT take this trade. 0 = no concerns.",
    )

    earnings_risk: Literal["none", "imminent", "unknown"]
    earnings_detail: str = Field(max_length=100)

    trap_probability: float = Field(ge=0.0, le=1.0)
    trap_reasoning: str = Field(max_length=150)

    liquidity_concern: Literal["none", "low", "high"]
    liquidity_reasoning: str = Field(max_length=100)

    concentration_risk: Literal["none", "moderate", "high"]
    concentration_detail: str = Field(max_length=100)

    catalyst_type: Literal["organic", "post_catalyst", "unknown"]
    catalyst_reasoning: str = Field(max_length=100)

    crowded_trade_risk: Literal["low", "moderate", "high"]

    kill_reasons: List[str] = Field(
        default_factory=list, max_length=5,
        description="Specific reasons to NOT take this trade.",
    )
    reasoning: str = Field(max_length=300)

"""V3 Deterministic Orchestrator.

Combines the 5 agent outputs into a single conviction score using:
  1. Hard veto gates (any one kills the trade)
  2. Weighted blend with dynamic Devil's Advocate scaling
  3. Soft multiplier penalties for cross-cutting concerns

No LLM — pure Python, fully auditable, backtestable, $0 cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.agents.schemas import (
    DevilsAdvocateAssessment,
    EntryTimingAssessment,
    OptionsContextAssessment,
    SRQualityOutput,
    TradePlanAssessment,
)

# ---------------------------------------------------------------------------
# Base weights (sum to 1.0)
# ---------------------------------------------------------------------------
_W_SR = 0.30
_W_TP = 0.20
_W_OC = 0.20
_W_ET = 0.15
_W_DA = 0.15

_DA_ELEVATED_WEIGHT = 0.25
_DA_ELEVATED_THRESHOLD = 5.0


@dataclass
class OrchestratorResult:
    """Full audit trail of the orchestrator decision."""

    conviction: float
    vetoed: bool = False
    veto_reason: str = ""
    raw_blend: float = 0.0
    penalties_applied: list[str] = field(default_factory=list)
    penalty_multiplier: float = 1.0
    weights_used: dict[str, float] = field(default_factory=dict)
    component_contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "conviction": round(self.conviction, 4),
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
            "raw_blend": round(self.raw_blend, 4),
            "penalties_applied": self.penalties_applied,
            "penalty_multiplier": round(self.penalty_multiplier, 4),
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "component_contributions": {
                k: round(v, 4) for k, v in self.component_contributions.items()
            },
        }


def _veto(reason: str) -> OrchestratorResult:
    return OrchestratorResult(conviction=0.0, vetoed=True, veto_reason=reason)


def compute_agent_conviction(
    sr: Optional[SRQualityOutput] = None,
    tp: Optional[TradePlanAssessment] = None,
    oc: Optional[OptionsContextAssessment] = None,
    et: Optional[EntryTimingAssessment] = None,
    da: Optional[DevilsAdvocateAssessment] = None,
) -> OrchestratorResult:
    """Combine agent outputs into a single conviction score.

    Handles partial agent availability gracefully — missing agents get a
    neutral score (5.0) so the blend still works when only some agents ran.
    """

    sr_score = sr.overall_sr_quality if sr else 5.0
    tp_score = tp.plan_score if tp else 5.0
    oc_score = oc.directional_conviction if oc else 5.0
    et_score = et.entry_score if et else 5.0
    da_risk = da.risk_score if da else 5.0

    # -----------------------------------------------------------------------
    # Layer 1: Hard Veto Gates
    # -----------------------------------------------------------------------

    if da is not None:
        if da.earnings_risk == "imminent":
            return _veto(f"earnings imminent: {da.earnings_detail}")

        if da.risk_score >= 8 and len(da.kill_reasons) >= 2:
            reasons = "; ".join(da.kill_reasons[:3])
            return _veto(f"multiple critical risks (DA={da.risk_score:.1f}): {reasons}")

        if da.liquidity_concern == "high":
            return _veto(f"high liquidity concern: {da.liquidity_reasoning}")

    if et is not None:
        if et.entry_timing == "skip_window":
            return _veto("entry window missed (skip_window)")

    # -----------------------------------------------------------------------
    # Layer 2: Weighted Blend with Dynamic DA Weight
    # -----------------------------------------------------------------------

    if da_risk >= _DA_ELEVATED_THRESHOLD:
        w_da = _DA_ELEVATED_WEIGHT
    else:
        w_da = _W_DA

    non_da_total = _W_SR + _W_TP + _W_OC + _W_ET
    scale = (1.0 - w_da) / non_da_total

    w_sr = _W_SR * scale
    w_tp = _W_TP * scale
    w_oc = _W_OC * scale
    w_et = _W_ET * scale

    weights = {
        "sr_quality": w_sr,
        "trade_plan": w_tp,
        "options_context": w_oc,
        "entry_timing": w_et,
        "devils_advocate": w_da,
    }

    da_contribution = 10.0 - da_risk

    contributions = {
        "sr_quality": w_sr * sr_score,
        "trade_plan": w_tp * tp_score,
        "options_context": w_oc * oc_score,
        "entry_timing": w_et * et_score,
        "devils_advocate": w_da * da_contribution,
    }

    raw_blend = sum(contributions.values())

    # -----------------------------------------------------------------------
    # Layer 3: Soft Multiplier Penalties
    # -----------------------------------------------------------------------

    multiplier = 1.0
    penalties: list[str] = []

    if et is not None and et.chasing_risk == "high":
        multiplier *= 0.85
        penalties.append("chasing_risk_high (x0.85)")

    if et is not None and et.gap_risk == "high":
        multiplier *= 0.90
        penalties.append("gap_risk_high (x0.90)")

    if oc is not None and oc.hedging_probability > 0.5:
        multiplier *= 0.90
        penalties.append(f"hedging_probability_{oc.hedging_probability:.2f} (x0.90)")

    if oc is not None and oc.signal_consistency == "conflicting":
        multiplier *= 0.90
        penalties.append("signals_conflicting (x0.90)")

    if sr is not None and sr.key_level_quality == "noise":
        multiplier *= 0.85
        penalties.append("sr_key_level_noise (x0.85)")

    final = min(10.0, raw_blend * multiplier)

    return OrchestratorResult(
        conviction=final,
        vetoed=False,
        raw_blend=raw_blend,
        penalties_applied=penalties,
        penalty_multiplier=multiplier,
        weights_used=weights,
        component_contributions=contributions,
    )

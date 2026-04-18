"""View models for the trader dashboard.

Thin adapters that flatten + enrich raw signal/row dicts into the canonical
shape the Jinja2 `trader_card` macro expects. Keeps template logic minimal
and gives us a single place to compute size / heat / projected risk from
the existing sizing tiers + VIX multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from app.config import PORTFOLIO_CAPITAL, SIZING_TIERS
from app.features.flow_stats import TIER_ABS, TIER_LABELS
from app.features.grade_explainer import build_flow_grade_reasons


def _derive_flow_confidence_tier(row: dict[str, Any]) -> tuple[int | None, str | None]:
    """Return ``(tier_int, label)`` for the worst component tier on the row.

    Looks at both directional summary columns (``bullish_zscore_tier`` and
    ``bearish_zscore_tier``) and picks the worst (numerically highest) for
    the traded side if direction is known, otherwise across both.
    """
    direction = (row.get("direction") or "").upper()

    candidates: list[int] = []
    if direction == "LONG" and row.get("bullish_zscore_tier") is not None:
        candidates.append(int(row["bullish_zscore_tier"]))
    elif direction == "SHORT" and row.get("bearish_zscore_tier") is not None:
        candidates.append(int(row["bearish_zscore_tier"]))
    else:
        for k in ("bullish_zscore_tier", "bearish_zscore_tier"):
            if row.get(k) is not None:
                try:
                    candidates.append(int(row[k]))
                except (TypeError, ValueError):
                    continue

    if not candidates:
        return None, None

    worst = max(candidates)
    return worst, TIER_LABELS.get(worst)


def _risk_pct_for_score(score: float | None) -> float:
    """Mirror of positions._risk_pct_for_score — kept here to avoid importing
    the heavier positions module for a single constant lookup.
    """
    if score is None:
        return 0.0
    try:
        s = float(score)
    except (TypeError, ValueError):
        return 0.0
    for threshold, pct in SIZING_TIERS:
        if s >= threshold:
            return pct
    return 0.0


@dataclass
class TraderCardView:
    """Shape expected by `_trader_card.html::trader_card`.

    This intentionally preserves the raw row's keys (so the macro can still
    read enriched sub-dicts like `liquidity`, `session`, `rs`, etc.) and
    layers projected-sizing fields on top:

      - ``size_shares``     — projected share count at entry
      - ``notional_dollar`` — shares × entry
      - ``risk_dollar``     — dollars at risk (entry → stop)
      - ``heat_pct``        — risk_dollar / PORTFOLIO_CAPITAL × 100
    """

    row: dict[str, Any]
    size_shares: int | None = None
    notional_dollar: float | None = None
    risk_dollar: float | None = None
    heat_pct: float | None = None
    vix_sizing_mult: float | None = None
    capital: float = PORTFOLIO_CAPITAL
    flow_confidence_tier: int | None = None
    flow_confidence_label: str | None = None
    avg_delta: float | None = None
    delta_source_mix: float | None = None
    grade_reasons: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        vix_sizing_mult: float | None = None,
        capital: float = PORTFOLIO_CAPITAL,
    ) -> "TraderCardView":
        """Build a view from an enriched signal row.

        Sizing is a *projection* based on SIZING_TIERS × VIX multiplier — it
        is deliberately identical to what `open_positions` would pick at
        portfolio-open time so the card's numbers match the executed trade.
        """
        entry = row.get("entry_price")
        stop = row.get("stop_price")
        direction = (row.get("direction") or "").upper()
        score = row.get("final_score")

        size_shares = None
        notional_dollar = None
        risk_dollar = None
        heat_pct = None

        try:
            ep = float(entry) if entry is not None else None
            sp = float(stop) if stop is not None else None
            if ep and sp and ep > 0 and sp > 0:
                risk_per_share = ep - sp if direction == "LONG" else sp - ep
                if risk_per_share > 0:
                    risk_pct = _risk_pct_for_score(score)
                    vm = vix_sizing_mult or 1.0
                    risk_dollar_budget = capital * risk_pct * vm
                    if risk_dollar_budget > 0:
                        shares = int(risk_dollar_budget / risk_per_share)
                        if shares > 0:
                            size_shares = shares
                            notional_dollar = round(shares * ep, 2)
                            risk_dollar = round(shares * risk_per_share, 2)
                            heat_pct = round(risk_dollar / capital * 100, 2)
        except (TypeError, ValueError):
            pass

        tier_int, tier_label = _derive_flow_confidence_tier(row)

        side = "bullish" if direction == "LONG" else ("bearish" if direction == "SHORT" else None)
        avg_delta: float | None = None
        delta_src_mix: float | None = None
        if side is not None:
            try:
                ad = row.get(f"{side}_avg_delta")
                if ad is not None and float(ad) > 0:
                    avg_delta = float(ad)
                sm = row.get(f"{side}_delta_source_mix")
                if sm is not None:
                    delta_src_mix = float(sm)
            except (TypeError, ValueError):
                pass

        reasons = row.get("grade_reasons")
        if not reasons and side is not None:
            try:
                reasons = build_flow_grade_reasons(row, side=side)
            except Exception:
                reasons = []

        return cls(
            row=dict(row),
            size_shares=size_shares,
            notional_dollar=notional_dollar,
            risk_dollar=risk_dollar,
            heat_pct=heat_pct,
            vix_sizing_mult=vix_sizing_mult,
            capital=capital,
            flow_confidence_tier=tier_int,
            flow_confidence_label=tier_label,
            avg_delta=avg_delta,
            delta_source_mix=delta_src_mix,
            grade_reasons=reasons or [],
        )

    def to_template(self) -> dict[str, Any]:
        """Return a flat dict suitable for passing to `trader_card(row)`.

        We merge projected sizing onto the original row so templates can read
        `row.size_shares` / `row.heat_pct` without any extra indirection.
        """
        out = dict(self.row)
        out["size_shares"] = self.size_shares
        out["notional_dollar"] = self.notional_dollar
        out["risk_dollar"] = self.risk_dollar
        out["heat_pct"] = self.heat_pct
        out["vix_sizing_mult"] = self.vix_sizing_mult
        out["flow_confidence_tier"] = self.flow_confidence_tier
        out["flow_confidence_label"] = self.flow_confidence_label
        out["avg_delta"] = self.avg_delta
        out["delta_source_mix"] = self.delta_source_mix
        out["grade_reasons"] = self.grade_reasons
        return out


def build_trader_card_rows(
    rows: list[dict[str, Any]],
    *,
    vix_sizing_mult: float | None = None,
    capital: float = PORTFOLIO_CAPITAL,
) -> list[dict[str, Any]]:
    """Convenience wrapper for list-of-rows."""
    return [
        TraderCardView.from_row(
            r,
            vix_sizing_mult=vix_sizing_mult,
            capital=capital,
        ).to_template()
        for r in rows
    ]

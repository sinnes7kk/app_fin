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
from app.features.conviction_stack import compute_conviction_stack
from app.features.flow_narrative import build_flow_feature_narrative
from app.features.flow_stats import TIER_ABS, TIER_LABELS
from app.features.grade_explainer import build_flow_grade_reasons
from app.features.trade_structure import recommend_structure


def attach_sizing_context(
    trade_structure: dict[str, Any] | None,
    risk_regime: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Attach a ``sizing_context`` block to a ``trade_structure`` payload.

    Wave 8 — the Structure tab renders ``trade_structure.sizing_context``
    as a coloured strip beneath the primary recommendation, and surfaces
    any regime HALT/elevated warning as an extra caveat so size-down is
    obvious at a glance.  Returns the (possibly mutated) structure dict
    or ``None`` if either input is missing.
    """
    if not isinstance(trade_structure, dict) or not isinstance(risk_regime, dict):
        return trade_structure

    tier = risk_regime.get("tier") or "calm"
    trade_structure["sizing_context"] = {
        "tier": tier,
        "label": risk_regime.get("tier_label") or tier.title(),
        "multiplier": risk_regime.get("multiplier"),
        "checks": risk_regime.get("checks") or [],
        "halt_reason": risk_regime.get("halt_reason"),
    }

    halt = risk_regime.get("halt_reason")
    caveats = trade_structure.setdefault("caveats", [])
    if halt:
        caveats.insert(0, f"Regime HALT — {halt}. Stand aside until the catalyst clears.")
    elif tier in {"panic", "elevated"}:
        mult = risk_regime.get("multiplier")
        caveats.append(
            f"Risk regime is {risk_regime.get('tier_label') or tier}"
            + (f" (×{float(mult):.2f} sizing)" if isinstance(mult, (int, float)) else "")
            + " — trim size accordingly."
        )
    return trade_structure


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


# Premium-Mix UI copy — single source of truth so templates + narrative
# modules pull from the same blurbs.  Each bucket has a short label, a
# plain-English tooltip, and a "signal" hint (what a trader should take
# away when that bucket dominates).
PREMIUM_BUCKET_COPY: dict[str, dict[str, str]] = {
    "total": {
        "label": "Total directional",
        "tip": "UW's aggregate of all directional (ASK calls + BID puts) premium for the day, across every DTE. Authoritative for Flow Tracker's multi-day accumulation regression.",
        "signal": "The honest total bid from the tape.",
    },
    "lottery": {
        "label": "Lottery (0-14d)",
        "tip": "Unusual flow (>=$500K/trade) expiring within 2 weeks. Cheap, high-gamma, fast decay. Common on earnings, squeeze setups, and binary catalysts.",
        "signal": "Event bet — size accordingly, not a positional conviction read.",
    },
    "swing": {
        "label": "Swing (30-120d)",
        "tip": "Unusual flow in the 30-120 day sweet spot. This is the institutional swing window — most of your signal pipeline's edge lives here.",
        "signal": "Classic institutional positioning. The bread-and-butter signal.",
    },
    "leap": {
        "label": "LEAP (180d+)",
        "tip": "Unusual flow expiring 6+ months out. Slow, deep-pocket structural bets. Less common but high conviction when present.",
        "signal": "Deep positional commitment — worth taking seriously.",
    },
    "other": {
        "label": "Other DTE",
        "tip": "Unusual flow in 15-29d or 121-179d — the gaps between lottery / swing / LEAP buckets. Usually a mix of earnings-cycle and mid-term bets.",
        "signal": "Mixed horizons. Look at the side (bull/bear) more than the bucket.",
    },
    "unusual": {
        "label": "Unusual flow",
        "tip": "Sum of lottery + swing + LEAP + other — all unusual (>=$500K/trade) directional flow, regardless of DTE. Subset of Total directional.",
        "signal": "How much of today's aggregate came from institutional-sized prints.",
    },
}


def _pct(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    try:
        return round(float(part) / float(whole) * 100.0, 1)
    except (TypeError, ValueError):
        return 0.0


def _build_premium_mix_ui(row: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize ``row.premium_mix`` into a UI-friendly payload.

    Combines bullish + bearish into a single "dominant side" number per
    bucket (falls back to totals when direction is unclear) plus per-
    bucket percent-of-total for easy sparkline / stacked-bar rendering.
    Returns ``None`` when the row has no taxonomy data (legacy / ranked
    signal rows — those fall back to the existing Flow-Feature panel).
    """
    mix = row.get("premium_mix")
    if not isinstance(mix, dict):
        return None

    def _f(key: str) -> float:
        try:
            return float(mix.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    total_bull = _f("total_bullish")
    total_bear = _f("total_bearish")
    total_gross = total_bull + total_bear
    if total_gross <= 0:
        return None

    bull_share_pct = _pct(total_bull, total_gross)

    def _bucket(key: str) -> dict[str, Any]:
        bull = _f(f"{key}_bullish")
        bear = _f(f"{key}_bearish")
        gross = bull + bear
        copy = PREMIUM_BUCKET_COPY.get(key, {})
        return {
            "key": key,
            "label": copy.get("label", key.title()),
            "tip": copy.get("tip"),
            "signal": copy.get("signal"),
            "bullish": round(bull, 2),
            "bearish": round(bear, 2),
            "gross": round(gross, 2),
            "pct_of_total": _pct(gross, total_gross),
            "pct_bullish": _pct(bull, gross) if gross > 0 else 0.0,
        }

    buckets = [_bucket(k) for k in ("swing", "lottery", "leap", "other")]
    dominant = max(buckets, key=lambda b: b["gross"]) if any(b["gross"] > 0 for b in buckets) else None

    total_copy = PREMIUM_BUCKET_COPY["total"]
    unusual_copy = PREMIUM_BUCKET_COPY["unusual"]

    return {
        "source": mix.get("source"),
        "total": {
            "label": total_copy["label"],
            "tip": total_copy["tip"],
            "signal": total_copy["signal"],
            "bullish": round(total_bull, 2),
            "bearish": round(total_bear, 2),
            "gross": round(total_gross, 2),
            "pct_bullish": bull_share_pct,
        },
        "unusual": {
            "label": unusual_copy["label"],
            "tip": unusual_copy["tip"],
            "signal": unusual_copy["signal"],
            "bullish": round(_f("unusual_bullish"), 2),
            "bearish": round(_f("unusual_bearish"), 2),
            "gross": round(_f("unusual_bullish") + _f("unusual_bearish"), 2),
            "pct_of_total": _pct(
                _f("unusual_bullish") + _f("unusual_bearish"),
                total_gross,
            ),
        },
        "buckets": buckets,
        "dominant_bucket": dominant["key"] if dominant else None,
        "dominant_label": dominant["label"] if dominant else None,
    }


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
    # Wave 4 — composite 0-100 Conviction Stack.  Condenses the old F/D/C/I
    # dot row plus dealer / price confirmation into a single chip.
    conviction_stack: dict[str, Any] | None = None
    # Wave 6 — plain-English narrative bullets for the Trader Card's
    # "Why?" tab.  A short ordered list of {tone, icon, label, detail}.
    narrative: list[dict[str, Any]] = field(default_factory=list)
    # Wave 7 — structure recommendation payload (primary vehicle +
    # alternatives + avoid + caveats) consumed by the Trader Card
    # "Structure" tab.
    trade_structure: dict[str, Any] | None = None
    # Premium-Taxonomy plan — normalized mix payload for the Trader Card
    # "Premium Mix" panel (total directional + lottery / swing / LEAP /
    # other, each with tooltip copy).  ``None`` when the row has no
    # taxonomy data (legacy rows / ranked signal rows).
    premium_mix_ui: dict[str, Any] | None = None

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        vix_sizing_mult: float | None = None,
        capital: float = PORTFOLIO_CAPITAL,
        risk_regime: dict[str, Any] | None = None,
    ) -> "TraderCardView":
        """Build a view from an enriched signal row.

        Sizing is a *projection* based on SIZING_TIERS × VIX multiplier × the
        Wave 8 ``risk_regime.multiplier`` (if provided) — it is deliberately
        identical to what `open_positions` would pick at portfolio-open
        time so the card's numbers match the executed trade.
        """
        entry = row.get("entry_price")
        stop = row.get("stop_price")
        direction = (row.get("direction") or "").upper()
        score = row.get("final_score")

        size_shares = None
        notional_dollar = None
        risk_dollar = None
        heat_pct = None

        # Wave 8 — layered regime multiplier on top of the legacy VIX one.
        regime_mult = 1.0
        if isinstance(risk_regime, dict):
            try:
                rm = float(risk_regime.get("multiplier"))
                if rm >= 0:
                    regime_mult = rm
            except (TypeError, ValueError):
                pass

        try:
            ep = float(entry) if entry is not None else None
            sp = float(stop) if stop is not None else None
            if ep and sp and ep > 0 and sp > 0:
                risk_per_share = ep - sp if direction == "LONG" else sp - ep
                if risk_per_share > 0:
                    risk_pct = _risk_pct_for_score(score)
                    vm = vix_sizing_mult or 1.0
                    risk_dollar_budget = capital * risk_pct * vm * regime_mult
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

        try:
            stack = compute_conviction_stack(dict(row))
        except Exception:
            stack = None

        # Wave 6 — build narrative AFTER stack so the narrative can
        # consume the stack tier as its headline bullet.
        row_for_narrative = dict(row)
        if stack is not None:
            row_for_narrative["conviction_stack"] = stack
        try:
            narrative = build_flow_feature_narrative(row_for_narrative)
        except Exception:
            narrative = []

        # Wave 7 — structure recommendation uses the same enriched row
        # (including the freshly-attached conviction_stack) so the ladder
        # tier-gating stays consistent with the Stack chip.
        try:
            structure = recommend_structure(row_for_narrative)
        except Exception:
            structure = None

        # Wave 8 — attach the regime-aware sizing context to the structure
        # payload so the Structure tab can render it inline.
        structure = attach_sizing_context(structure, risk_regime)

        premium_mix_ui = _build_premium_mix_ui(row)

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
            conviction_stack=stack,
            narrative=narrative,
            trade_structure=structure,
            premium_mix_ui=premium_mix_ui,
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
        out["conviction_stack"] = self.conviction_stack
        out["narrative"] = self.narrative
        out["trade_structure"] = self.trade_structure
        out["premium_mix_ui"] = self.premium_mix_ui
        return out


def build_trader_card_rows(
    rows: list[dict[str, Any]],
    *,
    vix_sizing_mult: float | None = None,
    capital: float = PORTFOLIO_CAPITAL,
    risk_regime: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper for list-of-rows.

    Wave 8 — pass ``risk_regime`` through so trader cards carry the
    regime-aware sizing multiplier and attach the sizing context to
    their Structure tab.
    """
    return [
        TraderCardView.from_row(
            r,
            vix_sizing_mult=vix_sizing_mult,
            capital=capital,
            risk_regime=risk_regime,
        ).to_template()
        for r in rows
    ]

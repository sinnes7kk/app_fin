"""Composite 0-100 Conviction Stack (Wave 4).

Condenses the six signals that used to live as binary F/D/C/I dots (plus
the unwieldy spread of chips we accumulated in Waves 0-2) into a single
0-100 score and four-tier classification:

    elite     >= 80   — stack up, confirmed across the board
    strong    65-79   — clean setup; act on it
    moderate  50-64   — tradable but watch for confirmation
    weak      <50     — don't force it

The six weighted components (total 100 pts):

  1) ``flow_core``      (50 pts) — the options flow backbone.  Scales
     ``conviction_score`` (0-10) linearly.  Falls back to
     ``flow_score_scaled`` when conviction isn't attached yet.
  2) ``dp_confirm``     (15 pts) — institutional dark-pool footprint
     aligned with trade direction.  Size gate on
     ``notional_mcap_bps``, direction gate on ``bias``.  Bonus for
     Wave-2 ``dp_z`` z-tier flags.
  3) ``chain_confirm``  (10 pts) — hot-chain / top-chain confirmation
     with direction alignment (call-dominant for LONG, put-dominant for
     SHORT).
  4) ``insider``        (8 pts)  — slow-moving directional signal.
     Aligned = full credit.  Opposing = -4 drag (floored at 0).
  5) ``price_confirm``  (12 pts) — the underlying's own story.
     Combines session tone, relative strength (5d/20d vs SPY), and
     window-return (flow tracker only).  Best sub-signal wins.
  6) ``dealer_regime``  (5 pts)  — Wave-2 dealer hedge bias.  Negative
     gamma ("chase") amplifies follow-through; positive gamma
     ("suppress") is a drag when we're trying to break out.

The module is deliberately framework-free — takes a plain dict, returns
a plain dict — so both ``view_models.TraderCardView`` (candidates) and
``server._enrich_flow_tracker_decision`` (flow tracker rows) can attach
it without caring about upstream shape differences.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Tier thresholds.  Tuned so that a "typical passing" flow tracker row
# lands around 60-70; elite requires multi-signal confirmation.
# ---------------------------------------------------------------------------
TIER_ELITE = "elite"
TIER_STRONG = "strong"
TIER_MODERATE = "moderate"
TIER_WEAK = "weak"

_TIER_CUTOFFS: list[tuple[float, str]] = [
    (80.0, TIER_ELITE),
    (65.0, TIER_STRONG),
    (50.0, TIER_MODERATE),
    (0.0, TIER_WEAK),
]

TIER_LABELS = {
    TIER_ELITE: "Stack A",
    TIER_STRONG: "Stack B",
    TIER_MODERATE: "Stack C",
    TIER_WEAK: "Stack D",
}

# Component weight caps (sum = 100).
_W_FLOW = 50.0
_W_DP = 15.0
_W_CHAIN = 10.0
_W_INSIDER = 8.0
_W_PRICE = 12.0
_W_DEALER = 5.0


def _direction_side(direction: str | None) -> str | None:
    """Normalise assorted direction labels into ``"LONG"``/``"SHORT"``."""
    if not direction:
        return None
    d = str(direction).upper().strip()
    if d in {"LONG", "BULLISH", "BUY", "NET BULLISH"}:
        return "LONG"
    if d in {"SHORT", "BEARISH", "SELL", "NET BEARISH"}:
        return "SHORT"
    return None


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if f != f:  # NaN guard
            return default
        return f
    except (TypeError, ValueError):
        return default


def _score_flow_core(row: dict) -> tuple[float, dict]:
    """Flow backbone — 50 pts at conviction_score = 10."""
    score = row.get("conviction_score")
    src = "conviction"
    if score is None:
        # Flow Tracker may or may not have conviction; Unusual Flow rows carry
        # ``flow_score_scaled`` instead.
        score = row.get("flow_score_scaled")
        src = "flow_scaled"
    if score is None:
        # Last-resort: final_score (candidates) also runs 0-10.
        score = row.get("final_score")
        src = "final_score"
    score_f = _safe_float(score)
    pts = max(0.0, min(score_f, 10.0)) / 10.0 * _W_FLOW
    return pts, {
        "component": "flow_core",
        "source": src,
        "raw": round(score_f, 2),
        "points": round(pts, 2),
        "max": _W_FLOW,
    }


def _score_dp(row: dict, side: str | None) -> tuple[float, dict]:
    """Dark-pool alignment + size — up to 15 pts."""
    dp = row.get("dark_pool") or {}
    if not isinstance(dp, dict):
        dp = {}
    bias = dp.get("bias")
    bps = _safe_float(dp.get("notional_mcap_bps"))
    dp_z = row.get("dp_z")
    dp_z_tier = row.get("dp_tier")
    pts = 0.0
    notes: list[str] = []

    if bias is None and bps == 0 and dp_z is None:
        return 0.0, {"component": "dp_confirm", "points": 0.0, "max": _W_DP, "notes": "no_dp_data"}

    bias_f = _safe_float(bias, default=0.5)

    # 1) Primary bias × notional gate (worth up to 12 pts).
    if side == "LONG":
        if bias_f >= 0.60 and bps >= 5.0:
            pts += 12.0; notes.append("strong_buyers_outsized")
        elif bias_f >= 0.60 and bps >= 1.0:
            pts += 9.0; notes.append("buyers_notable")
        elif bias_f >= 0.55 and bps >= 1.0:
            pts += 5.0; notes.append("leaning_buyers")
        elif bias_f <= 0.40 and bps >= 1.0:
            pts -= 4.0; notes.append("sellers_against_long")
    elif side == "SHORT":
        if bias_f <= 0.40 and bps >= 5.0:
            pts += 12.0; notes.append("strong_sellers_outsized")
        elif bias_f <= 0.40 and bps >= 1.0:
            pts += 9.0; notes.append("sellers_notable")
        elif bias_f <= 0.45 and bps >= 1.0:
            pts += 5.0; notes.append("leaning_sellers")
        elif bias_f >= 0.60 and bps >= 1.0:
            pts -= 4.0; notes.append("buyers_against_short")

    # 2) Wave-2 dp_z bonus (up to +3 when genuinely unusual for this ticker).
    if dp_z is not None:
        z = _safe_float(dp_z)
        if side == "LONG" and z >= 2.0:
            pts += 3.0; notes.append(f"dp_z_hot_{z:.1f}")
        elif side == "SHORT" and z <= -2.0:
            pts += 3.0; notes.append(f"dp_z_cold_{z:.1f}")

    pts = max(0.0, min(pts, _W_DP))
    return pts, {
        "component": "dp_confirm",
        "bias": round(bias_f, 2),
        "notional_mcap_bps": round(bps, 2),
        "dp_z": dp_z,
        "dp_tier": dp_z_tier,
        "points": round(pts, 2),
        "max": _W_DP,
        "notes": notes or None,
    }


def _score_chain(row: dict, side: str | None) -> tuple[float, dict]:
    """Hottest-chain confirmation — up to 10 pts."""
    hc = row.get("hot_chain") or row.get("top_chain")
    if not hc or not isinstance(hc, dict):
        return 0.0, {"component": "chain_confirm", "points": 0.0, "max": _W_CHAIN}

    # Chain call-dominance: callers use a variety of keys.  We want one of
    # ``call_premium`` / ``put_premium`` (preferred) or a pre-computed
    # ``dominant_side`` label.
    call_prem = _safe_float(hc.get("call_premium") or hc.get("total_call_premium"))
    put_prem = _safe_float(hc.get("put_premium") or hc.get("total_put_premium"))
    dominant = (hc.get("dominant_side") or "").upper()

    is_hot = bool(row.get("hot_chain"))
    cap = _W_CHAIN if is_hot else 5.0  # top_chain-only = half credit

    pts = 0.0
    aligned: bool | None = None
    if call_prem or put_prem:
        # Align: LONG wants calls dominant, SHORT wants puts.
        if side == "LONG":
            aligned = call_prem >= put_prem
        elif side == "SHORT":
            aligned = put_prem >= call_prem
    elif dominant:
        if side == "LONG":
            aligned = dominant in {"CALL", "CALLS", "BULLISH"}
        elif side == "SHORT":
            aligned = dominant in {"PUT", "PUTS", "BEARISH"}

    if aligned is True:
        pts = cap
    elif aligned is False:
        pts = 0.0
    else:
        # No side info but chain is hot — credit at half (presence signal).
        pts = cap * 0.5

    return pts, {
        "component": "chain_confirm",
        "is_hot_chain": is_hot,
        "aligned": aligned,
        "points": round(pts, 2),
        "max": _W_CHAIN,
    }


def _score_insider(row: dict, side: str | None) -> tuple[float, dict]:
    """Insider direction — up to 8 pts, with a -4 drag when opposing."""
    ins = row.get("insider") or {}
    if not isinstance(ins, dict):
        ins = {}
    nd = (ins.get("net_direction") or "").lower()
    if not nd or nd == "neutral":
        return 0.0, {"component": "insider", "points": 0.0, "max": _W_INSIDER}

    aligned = (nd == "buying" and side == "LONG") or (nd == "selling" and side == "SHORT")
    opposed = (nd == "buying" and side == "SHORT") or (nd == "selling" and side == "LONG")

    if aligned:
        pts = _W_INSIDER
    elif opposed:
        pts = -4.0  # drag; floored later
    else:
        pts = 0.0

    pts = max(0.0, pts)
    return pts, {
        "component": "insider",
        "net_direction": nd,
        "aligned": aligned,
        "opposed": opposed,
        "points": round(pts, 2),
        "max": _W_INSIDER,
    }


def _score_price(row: dict, side: str | None) -> tuple[float, dict]:
    """Underlying price confirmation — up to 12 pts.

    Combines three sub-signals; we take the best available rather than
    summing them (they're correlated — double-counting is unfair).
    """
    best = 0.0
    notes: list[str] = []

    # (a) Session tone (intraday)
    sess = row.get("session") or {}
    if not isinstance(sess, dict):
        sess = {}
    tone = (sess.get("session_tone") or "").upper()
    if side == "LONG" and tone == "STRENGTH":
        best = max(best, 8.0); notes.append("session_strong")
    elif side == "SHORT" and tone == "WEAKNESS":
        best = max(best, 8.0); notes.append("session_weak")
    elif tone in {"STRENGTH", "WEAKNESS"} and side and tone != (side == "LONG" and "STRENGTH" or "WEAKNESS"):
        # Fighting intraday tone — mild drag, don't award.
        notes.append("session_fight")

    # (b) Relative strength (5d / 20d).
    rs = row.get("rs") or {}
    if not isinstance(rs, dict):
        rs = {}
    rs5 = rs.get("rs_5d_pct")
    rs20 = rs.get("rs_20d_pct")
    if side == "LONG":
        if rs5 is not None and rs5 > 0 and rs20 is not None and rs20 > 0:
            best = max(best, 10.0); notes.append("rs_both_positive")
        elif rs20 is not None and rs20 > 0:
            best = max(best, 6.0); notes.append("rs_20d_positive")
    elif side == "SHORT":
        if rs5 is not None and rs5 < 0 and rs20 is not None and rs20 < 0:
            best = max(best, 10.0); notes.append("rs_both_negative")
        elif rs20 is not None and rs20 < 0:
            best = max(best, 6.0); notes.append("rs_20d_negative")

    # (c) Window return (flow tracker multi-day view).
    wr = row.get("window_return_pct")
    if wr is not None:
        wrf = _safe_float(wr)
        if side == "LONG" and wrf > 0:
            best = max(best, min(_W_PRICE, 6.0 + min(wrf, 6.0)))  # +6 base, +1/%
            notes.append(f"window_up_{wrf:+.1f}")
        elif side == "SHORT" and wrf < 0:
            best = max(best, min(_W_PRICE, 6.0 + min(-wrf, 6.0)))
            notes.append(f"window_dn_{wrf:+.1f}")

    best = min(best, _W_PRICE)
    return best, {
        "component": "price_confirm",
        "session_tone": tone or None,
        "rs_5d": rs5,
        "rs_20d": rs20,
        "window_return_pct": wr,
        "points": round(best, 2),
        "max": _W_PRICE,
        "notes": notes or None,
    }


def _score_dealer(row: dict, side: str | None) -> tuple[float, dict]:
    """Dealer hedge bias — up to 5 pts."""
    bias = (row.get("dealer_hedge_bias") or "").lower()
    if not bias or bias == "neutral" or side is None:
        return 0.0, {"component": "dealer_regime", "points": 0.0, "max": _W_DEALER}

    # Short-gamma dealers amplify whichever direction we're trading.
    # Long-gamma dealers suppress moves — tiny credit when we're counting
    # on mean-reversion, zero when we want breakout.
    if bias == "chase":
        pts = _W_DEALER
        note = "dealers_short_gamma"
    elif bias == "suppress":
        pts = 1.5
        note = "dealers_long_gamma_pinning"
    else:
        pts = 0.0
        note = bias

    return pts, {
        "component": "dealer_regime",
        "bias": bias,
        "points": round(pts, 2),
        "max": _W_DEALER,
        "note": note,
    }


def _classify_tier(score: float) -> tuple[str, str]:
    for cutoff, tier in _TIER_CUTOFFS:
        if score >= cutoff:
            return tier, TIER_LABELS[tier]
    return TIER_WEAK, TIER_LABELS[TIER_WEAK]


def compute_conviction_stack(row: dict) -> dict:
    """Return the full Conviction Stack payload for ``row``.

    Output shape::

        {
            "score": 72,                   # 0-100 int
            "tier": "strong",              # elite/strong/moderate/weak
            "tier_label": "Stack B",       # short UI label
            "components": [ {...}, ... ],  # list of component dicts
            "top_drivers": ["flow_core", "dp_confirm"],  # highest-scoring
            "top_drags":   ["insider"],                  # opposed / missing
        }

    All keys are template-safe (no NaN, no None where a number is
    expected).  Callers attach the returned dict to the row under the
    key ``conviction_stack``.
    """
    side = _direction_side(row.get("direction"))

    flow_pts, flow_c = _score_flow_core(row)
    dp_pts, dp_c = _score_dp(row, side)
    chain_pts, chain_c = _score_chain(row, side)
    ins_pts, ins_c = _score_insider(row, side)
    price_pts, price_c = _score_price(row, side)
    dealer_pts, dealer_c = _score_dealer(row, side)

    total = flow_pts + dp_pts + chain_pts + ins_pts + price_pts + dealer_pts
    total = max(0.0, min(total, 100.0))

    tier, tier_label = _classify_tier(total)

    components = [flow_c, dp_c, chain_c, ins_c, price_c, dealer_c]
    # Drivers = components with utilisation ≥ 60% of their cap.
    drivers = [c["component"] for c in components if c["max"] > 0 and c["points"] / c["max"] >= 0.60]
    # Drags = scored components with utilisation ≤ 20% of cap that
    # nonetheless had data available.  These are the "next-step to fix".
    drags = [
        c["component"]
        for c in components
        if c["max"] > 0
        and c["points"] / c["max"] <= 0.20
        and c["component"] != "flow_core"  # flow core weakness → score itself is weak; redundant
    ]

    return {
        "score": int(round(total)),
        "raw_score": round(total, 2),
        "tier": tier,
        "tier_label": tier_label,
        "components": components,
        "top_drivers": drivers,
        "top_drags": drags,
        "direction": side,
    }


def attach_conviction_stack(rows: list[dict]) -> list[dict]:
    """Decorate ``rows`` in-place with ``conviction_stack``.  Returns
    the same list for chaining."""
    for r in rows or []:
        try:
            r["conviction_stack"] = compute_conviction_stack(r)
        except Exception:  # pragma: no cover — never break the page
            r["conviction_stack"] = {
                "score": 0,
                "tier": TIER_WEAK,
                "tier_label": TIER_LABELS[TIER_WEAK],
                "components": [],
                "top_drivers": [],
                "top_drags": [],
                "direction": None,
            }
    return rows

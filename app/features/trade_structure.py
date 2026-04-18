"""Wave 7 — Trade structure / options-vehicle recommendation engine.

Given a flow tracker or trader-card row, returns a ranked list of
options structures (primary + alternatives) with plain-English
rationale, plus explicit ``avoid`` vetoes and ``caveats`` the trader
should price in before putting the order in.

This module is deliberately framework-free — pure dict in, pure dict
out — so it can be attached server-side by both
``server._enrich_flow_tracker_decision`` (multi-day rows) and
``view_models.TraderCardView`` (single-day intraday rows) without
coupling to Flask or pandas.

Design principles
-----------------

1. **Direction first**.  Long flow = call-side vehicles; short flow =
   put-side.  Neutral / mixed rows get defined-risk fallbacks.
2. **IV context chooses vega side**.  Cheap IV (≤30) → long premium
   rewarded.  Middle IV (30-60) → stock / simple debit.  Rich IV
   (≥70) → debit spread / risk reversal / short premium.
3. **Catalyst windows veto naked long premium**.  Earnings ≤ 5 days
   force defined-risk structures.
4. **Conviction gates position exposure**.  Elite conviction unlocks
   risk reversals / leaps; weak conviction restricts to pure stock.
5. **Window return drives alternates**.  When price is confirming
   direction, add trend-continuation alternates (ratio, diagonal);
   when price is fighting, keep defensive alternates (spread, small
   debit).
6. **Liquidity is a hard floor**.  Illiquid names (ADV $< 20M) collapse
   the whole ladder to stock / small debit — wider bid/ask kills every
   spread.

Output shape::

    {
        "side": "LONG" | "SHORT" | "NEUTRAL",
        "primary": {"structure": "DEBIT_SPREAD", "label": "Call debit",
                    "tone": "positive", "rationale": "..."},
        "alternatives": [
            {"structure": "LONG_CALL", "label": "Long call", ...},
            ...
        ],
        "avoid": [
            {"structure": "NAKED_PUT", "reason": "..."},
            ...
        ],
        "caveats": ["Earnings in 3 days — size small.", ...],
    }

The whole payload is template-safe (no NaN, no unescaped HTML, every
list is initialised).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Tone + helpers
# ---------------------------------------------------------------------------
TONE_POSITIVE = "positive"
TONE_NEUTRAL = "neutral"
TONE_WARNING = "warning"
TONE_NEGATIVE = "negative"
TONE_INFO = "info"

# Fixed labels so UI doesn't have to split_case the structure enum.
STRUCTURE_LABELS: dict[str, str] = {
    "STOCK_LONG": "Long stock",
    "STOCK_SHORT": "Short stock",
    "LONG_CALL": "Long call",
    "LONG_PUT": "Long put",
    "CALL_DEBIT_SPREAD": "Call debit spread",
    "PUT_DEBIT_SPREAD": "Put debit spread",
    "CALL_DIAGONAL": "Call diagonal",
    "PUT_DIAGONAL": "Put diagonal",
    "CALL_RATIO": "Call ratio (1x2)",
    "PUT_RATIO": "Put ratio (1x2)",
    "RISK_REVERSAL_LONG": "Risk reversal (short put / long call)",
    "RISK_REVERSAL_SHORT": "Risk reversal (short call / long put)",
    "CALL_CREDIT_SPREAD": "Short call credit spread (bearish)",
    "PUT_CREDIT_SPREAD": "Short put credit spread (bullish)",
    "LONG_CALL_LEAP": "Long call LEAP",
    "LONG_PUT_LEAP": "Long put LEAP",
    "COVERED_CALL": "Covered call / buy-write",
    "IRON_CONDOR": "Iron condor (mean-revert)",
    "STRADDLE_LONG": "Long straddle (vol expansion)",
    "STRANGLE_LONG": "Long strangle (vol expansion)",
    "STOCK_SHORT_WITH_HEDGE": "Short stock + long call hedge",
    "STOCK_LONG_WITH_HEDGE": "Long stock + protective put",
    "STAND_ASIDE": "Stand aside",
}


def _direction_side(direction: Any) -> str:
    """Normalise assorted direction labels into ``LONG``/``SHORT``/``NEUTRAL``."""
    if not direction:
        return "NEUTRAL"
    d = str(direction).upper().strip()
    if d in {"LONG", "BULLISH", "BUY", "NET BULLISH"}:
        return "LONG"
    if d in {"SHORT", "BEARISH", "SELL", "NET BEARISH"}:
        return "SHORT"
    return "NEUTRAL"


def _f(v: Any, default: float | None = None) -> float | None:
    """Safe float coerce — returns ``default`` on NaN / junk."""
    if v is None:
        return default
    try:
        x = float(v)
        if x != x:
            return default
        return x
    except (TypeError, ValueError):
        return default


def _struct(
    structure: str,
    tone: str = TONE_NEUTRAL,
    rationale: str | None = None,
) -> dict[str, str]:
    return {
        "structure": structure,
        "label": STRUCTURE_LABELS.get(structure, structure.replace("_", " ").title()),
        "tone": tone,
        "rationale": rationale or "",
    }


def _avoid(structure: str, reason: str) -> dict[str, str]:
    return {
        "structure": structure,
        "label": STRUCTURE_LABELS.get(structure, structure.replace("_", " ").title()),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------
def _iv_bucket(iv_rank: float | None) -> str:
    """Cheap (<=30) / mid (30-60) / rich (60-70) / expensive (>=70)."""
    if iv_rank is None:
        return "UNKNOWN"
    if iv_rank >= 70:
        return "EXPENSIVE"
    if iv_rank >= 60:
        return "RICH"
    if iv_rank >= 30:
        return "MID"
    return "CHEAP"


def _liquidity_floor(liquidity: dict | None) -> bool:
    """True when liquidity forbids spread structures."""
    if not isinstance(liquidity, dict):
        return False
    tier = str(liquidity.get("liquidity_tier") or "").upper()
    adv = _f(liquidity.get("adv_dollar"))
    if tier == "ILLIQUID":
        return True
    if tier == "THIN":
        return False  # thin is usable but we flag it
    if adv is not None and adv > 0 and adv < 20_000_000:
        return True
    return False


def _thin_liquidity(liquidity: dict | None) -> bool:
    if not isinstance(liquidity, dict):
        return False
    return str(liquidity.get("liquidity_tier") or "").upper() == "THIN"


def _conviction_tier(row: dict) -> tuple[int, str]:
    """Return ``(score, tier)`` where ``tier`` in {elite, strong, moderate, weak}.

    Prefers ``conviction_stack.score`` (Wave 4 composite) when available;
    falls back to ``conviction_score`` on the 0-10 scale.
    """
    stack = row.get("conviction_stack") or {}
    if isinstance(stack, dict):
        s = _f(stack.get("score"))
        tier = str(stack.get("tier") or "").lower()
        if s is not None and tier:
            return int(round(s)), tier
    cs = _f(row.get("conviction_score"))
    if cs is None:
        return 0, "weak"
    # Map 0-10 → 0-100 tier buckets
    s100 = int(round(cs * 10))
    if s100 >= 80:
        return s100, "elite"
    if s100 >= 65:
        return s100, "strong"
    if s100 >= 50:
        return s100, "moderate"
    return s100, "weak"


def _days_until_earnings(row: dict) -> int | None:
    earn = row.get("earnings") or {}
    if not isinstance(earn, dict):
        return None
    due = _f(earn.get("days_until_earnings"))
    if due is None:
        return None
    try:
        return int(due)
    except (TypeError, ValueError):
        return None


def _dealer_bias(row: dict) -> str:
    return str(row.get("dealer_hedge_bias") or "").lower()


def _window_return(row: dict) -> float | None:
    """Multi-day row → ``window_return_pct``; trader card → session px vs prev."""
    wr = _f(row.get("window_return_pct"))
    if wr is not None:
        return wr
    sess = row.get("session") or {}
    if isinstance(sess, dict):
        return _f(sess.get("px_vs_prior_close_pct"))
    return None


def _pin_risk_close(row: dict) -> bool:
    pin = _f(row.get("pin_risk_distance_pct"))
    return pin is not None and pin <= 1.5


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def recommend_structure(row: dict | None, side: str | None = None) -> dict:
    """Return a full structure recommendation payload for ``row``.

    ``side`` can be passed explicitly (``"LONG"``, ``"SHORT"``,
    ``"NEUTRAL"``) to force a direction; if omitted the row's
    ``direction`` field is used.
    """
    if not row or not isinstance(row, dict):
        return {
            "side": "NEUTRAL",
            "primary": _struct("STAND_ASIDE", TONE_WARNING, "Missing row context."),
            "alternatives": [],
            "avoid": [],
            "caveats": [],
        }

    if side is None:
        side = _direction_side(row.get("direction"))
    else:
        side = _direction_side(side)

    iv_rank = _f(row.get("latest_iv_rank"))
    if iv_rank is None:
        iv_rank = _f(row.get("iv_rank"))
    iv_bucket = _iv_bucket(iv_rank)

    iv_delta = _f(row.get("iv_rank_5d_delta"))
    dte_earn = _days_until_earnings(row)
    near_catalyst = dte_earn is not None and dte_earn <= 5
    mid_catalyst = dte_earn is not None and 5 < dte_earn <= 14

    score, tier = _conviction_tier(row)
    dealer = _dealer_bias(row)
    wr = _window_return(row)
    pin_close = _pin_risk_close(row)
    liq = row.get("liquidity")
    illiquid = _liquidity_floor(liq)
    thin = _thin_liquidity(liq)
    dte_bucket = str(row.get("dominant_dte_bucket") or "").lower()

    caveats: list[str] = []
    avoid: list[dict] = []

    # --- Hard vetoes that override structure choice ------------------------
    if tier == "weak":
        primary = _struct(
            "STAND_ASIDE",
            TONE_WARNING,
            f"Conviction stack only {score}/100 — signal too thin to size confidently.",
        )
        avoid.append(
            _avoid("LONG_CALL" if side == "LONG" else "LONG_PUT",
                   "Weak conviction doesn't justify burning theta.")
        )
        avoid.append(
            _avoid("RISK_REVERSAL_LONG" if side == "LONG" else "RISK_REVERSAL_SHORT",
                   "Unlimited-risk structures require elite conviction.")
        )
        caveats.append("Wait for either flow confirmation or a second signal to align.")
        return _finalise(side, primary, [], avoid, caveats, row, iv_bucket, iv_delta,
                          dealer, wr, pin_close, dte_earn)

    if illiquid:
        if side == "LONG":
            primary = _struct(
                "STOCK_LONG",
                TONE_NEUTRAL,
                "Option chain too illiquid for spreads — size via shares.",
            )
        elif side == "SHORT":
            primary = _struct(
                "STOCK_SHORT",
                TONE_NEUTRAL,
                "Option chain too illiquid for spreads — short shares directly.",
            )
        else:
            primary = _struct(
                "STAND_ASIDE", TONE_WARNING,
                "Illiquid chain + mixed direction — no clean expression.",
            )
        avoid.append(_avoid("CALL_DEBIT_SPREAD", "Wide bid/ask will eat edge."))
        avoid.append(_avoid("PUT_DEBIT_SPREAD", "Wide bid/ask will eat edge."))
        caveats.append("Illiquid underlying (ADV < $20M) — widen slippage assumptions.")
        return _finalise(side, primary, [], avoid, caveats, row, iv_bucket, iv_delta,
                          dealer, wr, pin_close, dte_earn)

    if side == "NEUTRAL":
        primary = _struct(
            "STAND_ASIDE",
            TONE_INFO,
            "Flow is mixed — no directional read strong enough to commit.",
        )
        if iv_bucket in {"RICH", "EXPENSIVE"}:
            alts = [_struct("IRON_CONDOR", TONE_NEUTRAL,
                            f"IV rank {iv_rank:.0f} rich — collect premium around the range." if iv_rank is not None else "Rich IV — collect premium.")]
        else:
            alts = []
        caveats.append("Direction still undefined — revisit after the next session.")
        return _finalise(side, primary, alts, avoid, caveats, row, iv_bucket, iv_delta,
                          dealer, wr, pin_close, dte_earn)

    # --- Directional recommendation -----------------------------------------
    if side == "LONG":
        primary, alts, extra_avoid, extra_caveats = _long_ladder(
            row, score, tier, iv_bucket, iv_rank, dte_earn, near_catalyst,
            mid_catalyst, dealer, wr, pin_close, thin, dte_bucket,
        )
    else:
        primary, alts, extra_avoid, extra_caveats = _short_ladder(
            row, score, tier, iv_bucket, iv_rank, dte_earn, near_catalyst,
            mid_catalyst, dealer, wr, pin_close, thin, dte_bucket,
        )

    avoid.extend(extra_avoid)
    caveats.extend(extra_caveats)

    return _finalise(side, primary, alts, avoid, caveats, row, iv_bucket, iv_delta,
                      dealer, wr, pin_close, dte_earn)


# ---------------------------------------------------------------------------
# LONG ladder
# ---------------------------------------------------------------------------
def _long_ladder(
    row: dict,
    score: int,
    tier: str,
    iv_bucket: str,
    iv_rank: float | None,
    dte_earn: int | None,
    near_catalyst: bool,
    mid_catalyst: bool,
    dealer: str,
    wr: float | None,
    pin_close: bool,
    thin: bool,
    dte_bucket: str,
) -> tuple[dict, list[dict], list[dict], list[str]]:
    alts: list[dict] = []
    avoid: list[dict] = []
    caveats: list[str] = []

    ivr_txt = f"IVR {iv_rank:.0f}" if iv_rank is not None else "IV context unknown"

    if near_catalyst:
        # Binary event dominates — force defined-risk.
        primary = _struct(
            "CALL_DEBIT_SPREAD",
            TONE_WARNING,
            f"Earnings in {dte_earn}d — defined-risk call debit spread caps vega crush.",
        )
        alts.append(_struct(
            "PUT_CREDIT_SPREAD",
            TONE_NEUTRAL,
            "If IV is screaming, selling a put spread below support monetises theta.",
        ))
        avoid.append(_avoid("LONG_CALL", f"Earnings in {dte_earn}d — single-leg long premium is an IV crush trap."))
        avoid.append(_avoid("RISK_REVERSAL_LONG", f"Earnings in {dte_earn}d — unlimited-downside before binary print."))
        caveats.append(f"Earnings in {dte_earn} day(s) — size small, don't roll through the event naked.")
        return primary, alts, avoid, caveats

    if iv_bucket == "EXPENSIVE":
        primary = _struct(
            "CALL_DEBIT_SPREAD",
            TONE_POSITIVE if tier in {"elite", "strong"} else TONE_NEUTRAL,
            f"{ivr_txt} — spread kills the vega tax a long call would pay.",
        )
        if tier in {"elite", "strong"}:
            alts.append(_struct(
                "RISK_REVERSAL_LONG",
                TONE_POSITIVE,
                f"Elite stack ({score}/100) + rich IV — short a put, buy a call, net-credit or near-zero debit.",
            ))
        alts.append(_struct(
            "PUT_CREDIT_SPREAD",
            TONE_NEUTRAL,
            "Short put spread under support monetises theta while keeping long-delta exposure.",
        ))
        avoid.append(_avoid("LONG_CALL", f"{ivr_txt} — paying extreme vega; better expressed as spread."))
        avoid.append(_avoid("STRADDLE_LONG", "Rich IV — buying vol here is paying retail."))
        return primary, alts, avoid, caveats

    if iv_bucket == "RICH":
        primary = _struct(
            "CALL_DEBIT_SPREAD",
            TONE_POSITIVE,
            f"{ivr_txt} — spread caps vega and keeps cost controlled.",
        )
        if tier == "elite":
            alts.append(_struct(
                "RISK_REVERSAL_LONG",
                TONE_POSITIVE,
                f"Elite conviction unlocks the risk reversal — finance the call with a short put.",
            ))
        alts.append(_struct("STOCK_LONG", TONE_NEUTRAL,
                            "Stock is the cleanest expression when IV is elevated but not blown out."))
        avoid.append(_avoid("LONG_CALL", f"{ivr_txt} — premium isn't cheap, spread is kinder."))
        return primary, alts, avoid, caveats

    if iv_bucket == "CHEAP":
        primary = _struct(
            "LONG_CALL",
            TONE_POSITIVE,
            f"{ivr_txt} — long premium is cheap and asymmetric.",
        )
        alts.append(_struct(
            "CALL_DEBIT_SPREAD",
            TONE_NEUTRAL,
            "Defined-risk alternative if you want to lower capital outlay.",
        ))
        if tier in {"elite", "strong"} and dte_bucket in {"leap", "leaps"}:
            alts.append(_struct(
                "LONG_CALL_LEAP",
                TONE_POSITIVE,
                "Dominant LEAP DTE + cheap IV — buy duration instead of theta.",
            ))
        if tier == "elite":
            alts.append(_struct(
                "RISK_REVERSAL_LONG",
                TONE_NEUTRAL,
                "Elite stack lets you finance the call by selling a put — larger exposure.",
            ))
        return primary, alts, avoid, caveats

    # MID IV or unknown — stock is the cleanest carrier.
    primary = _struct(
        "STOCK_LONG",
        TONE_NEUTRAL,
        f"{ivr_txt} — mid IV, stock is the cleanest expression.",
    )
    alts.append(_struct(
        "CALL_DEBIT_SPREAD",
        TONE_NEUTRAL,
        "Defined-risk alternative for traders allergic to overnight gaps.",
    ))
    if tier == "elite":
        alts.append(_struct(
            "RISK_REVERSAL_LONG",
            TONE_POSITIVE,
            "Elite conviction — short put / long call builds leverage without paying vega.",
        ))
    if thin:
        caveats.append("Option chain is thin — prefer stock over spreads unless spreads fill at mid.")
    return primary, alts, avoid, caveats


# ---------------------------------------------------------------------------
# SHORT ladder
# ---------------------------------------------------------------------------
def _short_ladder(
    row: dict,
    score: int,
    tier: str,
    iv_bucket: str,
    iv_rank: float | None,
    dte_earn: int | None,
    near_catalyst: bool,
    mid_catalyst: bool,
    dealer: str,
    wr: float | None,
    pin_close: bool,
    thin: bool,
    dte_bucket: str,
) -> tuple[dict, list[dict], list[dict], list[str]]:
    alts: list[dict] = []
    avoid: list[dict] = []
    caveats: list[str] = []

    ivr_txt = f"IVR {iv_rank:.0f}" if iv_rank is not None else "IV context unknown"

    if near_catalyst:
        primary = _struct(
            "PUT_DEBIT_SPREAD",
            TONE_WARNING,
            f"Earnings in {dte_earn}d — put debit spread caps vega crush and defines risk.",
        )
        alts.append(_struct(
            "CALL_CREDIT_SPREAD",
            TONE_NEUTRAL,
            "Short call spread above resistance monetises rich pre-ER IV if you don't want naked long puts.",
        ))
        avoid.append(_avoid("LONG_PUT", f"Earnings in {dte_earn}d — single-leg long premium is an IV crush trap."))
        avoid.append(_avoid("STOCK_SHORT", f"Earnings in {dte_earn}d — gap risk through the print uncontrolled."))
        caveats.append(f"Earnings in {dte_earn} day(s) — no naked shorts through the event.")
        return primary, alts, avoid, caveats

    if iv_bucket == "EXPENSIVE":
        primary = _struct(
            "PUT_DEBIT_SPREAD",
            TONE_POSITIVE if tier in {"elite", "strong"} else TONE_NEUTRAL,
            f"{ivr_txt} — spread kills the vega tax a long put would pay.",
        )
        if tier in {"elite", "strong"}:
            alts.append(_struct(
                "RISK_REVERSAL_SHORT",
                TONE_POSITIVE,
                f"Elite stack ({score}/100) + rich IV — short a call, buy a put, net-credit or near-zero debit.",
            ))
        alts.append(_struct(
            "CALL_CREDIT_SPREAD",
            TONE_NEUTRAL,
            "Short call spread above resistance is a theta-harvesting bearish play.",
        ))
        avoid.append(_avoid("LONG_PUT", f"{ivr_txt} — paying extreme vega; better expressed as spread."))
        avoid.append(_avoid("STRADDLE_LONG", "Rich IV — buying vol here is paying retail."))
        return primary, alts, avoid, caveats

    if iv_bucket == "RICH":
        primary = _struct(
            "PUT_DEBIT_SPREAD",
            TONE_POSITIVE,
            f"{ivr_txt} — spread caps vega and keeps cost controlled.",
        )
        if tier == "elite":
            alts.append(_struct(
                "RISK_REVERSAL_SHORT",
                TONE_POSITIVE,
                "Elite conviction unlocks the risk reversal — finance the put with a short call.",
            ))
        alts.append(_struct("STOCK_SHORT", TONE_NEUTRAL,
                            "Short shares keep the directional expression clean when IV is elevated but not blown out."))
        avoid.append(_avoid("LONG_PUT", f"{ivr_txt} — premium isn't cheap, spread is kinder."))
        return primary, alts, avoid, caveats

    if iv_bucket == "CHEAP":
        primary = _struct(
            "LONG_PUT",
            TONE_POSITIVE,
            f"{ivr_txt} — long put premium is cheap and asymmetric.",
        )
        alts.append(_struct(
            "PUT_DEBIT_SPREAD",
            TONE_NEUTRAL,
            "Defined-risk alternative if you want to lower capital outlay.",
        ))
        if tier in {"elite", "strong"} and dte_bucket in {"leap", "leaps"}:
            alts.append(_struct(
                "LONG_PUT_LEAP",
                TONE_POSITIVE,
                "Dominant LEAP DTE + cheap IV — buy duration instead of theta.",
            ))
        return primary, alts, avoid, caveats

    # MID IV or unknown — short shares with optional hedge.
    primary = _struct(
        "STOCK_SHORT",
        TONE_NEUTRAL,
        f"{ivr_txt} — mid IV, short shares is the cleanest expression.",
    )
    alts.append(_struct(
        "PUT_DEBIT_SPREAD",
        TONE_NEUTRAL,
        "Defined-risk alternative that sidesteps squeeze risk.",
    ))
    if tier == "elite":
        alts.append(_struct(
            "STOCK_SHORT_WITH_HEDGE",
            TONE_POSITIVE,
            "Elite conviction warrants a live short; a cheap OTM call caps squeeze risk.",
        ))
    if thin:
        caveats.append("Option chain is thin — prefer short shares over spreads unless spreads fill at mid.")
    return primary, alts, avoid, caveats


# ---------------------------------------------------------------------------
# Finalisation: attach cross-cutting caveats (dealer / window-return / pin / IV delta)
# ---------------------------------------------------------------------------
def _finalise(
    side: str,
    primary: dict,
    alternatives: list[dict],
    avoid: list[dict],
    caveats: list[str],
    row: dict,
    iv_bucket: str,
    iv_delta: float | None,
    dealer: str,
    wr: float | None,
    pin_close: bool,
    dte_earn: int | None,
) -> dict:
    """Attach cross-cutting caveats / alternates that apply regardless of bucket."""
    # Dealer hedge regime caveat.
    if dealer == "suppress" and side != "NEUTRAL":
        caveats.append(
            "Dealers long gamma — expect slow grind rather than a rip; size patience over urgency."
        )
        if not any(a.get("structure", "").startswith(("CALL_RATIO", "PUT_RATIO")) for a in alternatives):
            # Long-gamma / pinning regimes favour risk-reversal-light
            # structures that earn theta inside the pin.
            if iv_bucket in {"RICH", "EXPENSIVE"}:
                if side == "LONG":
                    alternatives.append(_struct(
                        "PUT_CREDIT_SPREAD",
                        TONE_NEUTRAL,
                        "Pinning regime + rich IV — a put credit spread earns theta without needing a rip.",
                    ))
                elif side == "SHORT":
                    alternatives.append(_struct(
                        "CALL_CREDIT_SPREAD",
                        TONE_NEUTRAL,
                        "Pinning regime + rich IV — a call credit spread earns theta without needing a drop.",
                    ))
    elif dealer == "chase":
        caveats.append(
            "Dealers short gamma — hedging flow amplifies moves, breakouts tend to extend."
        )

    # Window-return confirmation / fighting.
    if wr is not None and side == "LONG":
        if wr <= -2.0:
            caveats.append(
                f"Underlying {wr:.1f}% in window — price is still fighting the call flow. Scale in, don't chase."
            )
        elif wr >= 5.0:
            caveats.append(
                f"Underlying already +{wr:.1f}% in window — late chase risk; size smaller on fresh entries."
            )
    elif wr is not None and side == "SHORT":
        if wr >= 2.0:
            caveats.append(
                f"Underlying +{wr:.1f}% in window — price is still fighting the put flow. Scale in, don't chase."
            )
        elif wr <= -5.0:
            caveats.append(
                f"Underlying already {wr:.1f}% in window — late chase risk; size smaller on fresh entries."
            )

    # Pin risk proximity.
    if pin_close:
        pin_k = row.get("pin_risk_strike")
        pin_pct = _f(row.get("pin_risk_distance_pct"))
        if pin_pct is not None:
            caveats.append(
                f"Heavy OI pin {pin_pct:.1f}% away"
                + (f" (strike {pin_k})" if pin_k is not None else "")
                + " — expiry magnet may cap near-term move."
            )

    # IV rank delta.
    if iv_delta is not None and abs(iv_delta) >= 10:
        if iv_delta > 0:
            caveats.append(
                f"IV rank expanded {iv_delta:+.0f} pts in 5d — someone is pricing event risk; prefer debit over naked long premium."
            )
        else:
            caveats.append(
                f"IV rank dropped {iv_delta:+.0f} pts in 5d — premium is getting cheaper, long-premium structures improving."
            )

    # Earnings mid-catalyst (5 < dte <= 14) — include gentle reminder when not already near-cat.
    if dte_earn is not None and 5 < dte_earn <= 14:
        caveats.append(
            f"Earnings in {dte_earn} day(s) — catalyst on the horizon; prefer structures whose risk is defined before the print."
        )

    # Deduplicate caveats preserving order.
    seen = set()
    deduped_caveats: list[str] = []
    for c in caveats:
        if c not in seen:
            deduped_caveats.append(c)
            seen.add(c)

    # Cap alternatives at 4 so UI stays scannable.
    alternatives = alternatives[:4]

    return {
        "side": side,
        "primary": primary,
        "alternatives": alternatives,
        "avoid": avoid,
        "caveats": deduped_caveats,
    }


def attach_trade_structure(rows: list[dict]) -> list[dict]:
    """Decorate ``rows`` in-place with ``trade_structure``.  Returns rows for chaining."""
    for r in rows or []:
        try:
            r["trade_structure"] = recommend_structure(r)
        except Exception:  # pragma: no cover — never break the page
            r["trade_structure"] = {
                "side": "NEUTRAL",
                "primary": _struct("STAND_ASIDE", TONE_WARNING, "Error computing structure."),
                "alternatives": [],
                "avoid": [],
                "caveats": [],
            }
    return rows


__all__ = [
    "recommend_structure",
    "attach_trade_structure",
    "STRUCTURE_LABELS",
    "TONE_POSITIVE",
    "TONE_NEUTRAL",
    "TONE_WARNING",
    "TONE_NEGATIVE",
    "TONE_INFO",
]

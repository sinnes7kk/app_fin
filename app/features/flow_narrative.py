"""Wave 6 — plain-English narrative bullets for a Flow Tracker / Trader Card row.

The dashboard already exposes the raw components (chips, reasons, stack
breakdown) but interpreting them still requires translating a grid of
numbers into "what's actually happening".  This module produces a short,
ordered list of natural-language bullets so the ``Why?`` tab can read
like a concise analyst note rather than a metric dump.

Two entry points, same output shape::

    build_flow_tracker_narrative(row)  -> list[Bullet]
    build_flow_feature_narrative(row)  -> list[Bullet]

Each bullet is a dict::

    {
        "tone": "positive" | "negative" | "warning" | "info" | "neutral",
        "icon": "↑" | "↓" | "⚠" | "ℹ",
        "label": "Accumulation on 5 of 5 days",
        "detail": "Persistence 100% — classic slow-walk setup.",
    }

The module is deliberately framework-free: takes a plain dict, returns
a plain list, no pandas / Flask / UW coupling.  This keeps it testable
and lets both the Flow Tracker card modal and the Trader Card modal
reuse the exact same prose.

Design rules:

* **Be directional.** Always pick a side first; "bullish" vs "bearish"
  narratives differ in phrasing and sign of window return.
* **Be selective.** Cap the list at ~8 bullets so the panel doesn't
  become another metric wall.  Rank by actionable salience: stack tier
  first, persistence/accumulation next, confirmations after, caveats
  last.
* **Be specific.** Always include at least one concrete number so the
  reader can sanity-check ("Persistence 100%" not "very persistent").
* **Never crash.** Every field is looked up with ``_safe_get`` / ``_f``
  helpers; missing fields simply drop that bullet.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Tone constants.  Templates map these to CSS classes (``.tc-narrative-*``).
# ---------------------------------------------------------------------------
TONE_POSITIVE = "positive"
TONE_NEGATIVE = "negative"
TONE_WARNING = "warning"
TONE_INFO = "info"
TONE_NEUTRAL = "neutral"

_ICONS = {
    TONE_POSITIVE: "↑",
    TONE_NEGATIVE: "↓",
    TONE_WARNING: "⚠",
    TONE_INFO: "ℹ",
    TONE_NEUTRAL: "•",
}

# Hard cap so the panel doesn't turn into a metric dump.
_MAX_BULLETS = 8


def _f(v: Any, default: float | None = None) -> float | None:
    """Safe float coerce — returns ``default`` on NaN / junk."""
    if v is None:
        return default
    try:
        x = float(v)
        if x != x:  # NaN guard
            return default
        return x
    except (TypeError, ValueError):
        return default


def _direction_side(direction: Any) -> str | None:
    """Normalise LONG/SHORT/BULLISH/BEARISH → ``"LONG"``/``"SHORT"``."""
    if not direction:
        return None
    d = str(direction).upper().strip()
    if d in {"LONG", "BULLISH", "BUY", "NET BULLISH"}:
        return "LONG"
    if d in {"SHORT", "BEARISH", "SELL", "NET BEARISH"}:
        return "SHORT"
    return None


def _bullet(tone: str, label: str, detail: str | None = None) -> dict:
    return {
        "tone": tone,
        "icon": _ICONS.get(tone, "•"),
        "label": label,
        "detail": detail,
    }


def _fmt_money(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


# ===========================================================================
# Flow Tracker narrative (multi-day aggregated row)
# ===========================================================================
def build_flow_tracker_narrative(row: dict | None) -> list[dict]:
    """Return ordered natural-language bullets for a Flow Tracker card.

    Input is the dict shape produced by ``app.features.flow_tracker`` and
    further decorated by ``server._enrich_flow_tracker_decision`` — so
    expect keys like ``persistence_ratio``, ``trend``,
    ``cumulative_premium``, ``window_return_pct``, ``dealer_hedge_bias``,
    ``conviction_stack`` etc.  All lookups are tolerant of missing keys.
    """
    if not row or not isinstance(row, dict):
        return []

    bullets: list[dict] = []
    side = _direction_side(row.get("direction"))
    side_word = "bullish" if side == "LONG" else ("bearish" if side == "SHORT" else "mixed")

    # -- 1. Conviction stack (headline) -------------------------------------
    stack = row.get("conviction_stack") or {}
    stack_score = stack.get("score") if isinstance(stack, dict) else None
    stack_tier = stack.get("tier_label") if isinstance(stack, dict) else None
    if stack_score is not None and stack_tier:
        if stack_score >= 80:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"{stack_tier}: elite {side_word} setup ({stack_score}/100)",
                "Signals stack across flow, dark pool, chain and price — high-conviction trade.",
            ))
        elif stack_score >= 65:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"{stack_tier}: strong {side_word} setup ({stack_score}/100)",
                "Multiple confirming signals — clean setup, act on it.",
            ))
        elif stack_score >= 50:
            bullets.append(_bullet(
                TONE_NEUTRAL,
                f"{stack_tier}: moderate {side_word} setup ({stack_score}/100)",
                "Tradable but watch for additional confirmation.",
            ))
        else:
            bullets.append(_bullet(
                TONE_WARNING,
                f"{stack_tier}: weak setup ({stack_score}/100)",
                "Flow without broad confirmation — don't force it.",
            ))

    # -- 2. Persistence / accumulation structure ----------------------------
    pr = _f(row.get("persistence_ratio"))
    active_days = row.get("active_days")
    total_days = row.get("total_days")
    if pr is not None and active_days is not None and total_days:
        pr_pct = pr * 100
        if pr >= 0.99 and total_days >= 3:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"Active every day ({int(active_days)}/{int(total_days)})",
                "Unusual activity on every session in the window — classic slow-walk accumulation.",
            ))
        elif pr >= 0.80:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"High persistence ({pr_pct:.0f}%)",
                f"Active {int(active_days)}/{int(total_days)} days — sustained interest, not a one-print flash.",
            ))
        elif pr >= 0.50:
            bullets.append(_bullet(
                TONE_NEUTRAL,
                f"Moderate persistence ({pr_pct:.0f}%)",
                f"Active {int(active_days)}/{int(total_days)} days — some interest but uneven.",
            ))
        else:
            bullets.append(_bullet(
                TONE_WARNING,
                f"Low persistence ({pr_pct:.0f}%)",
                "Flow is bursty rather than accumulative — treat with caution.",
            ))

    # -- 3. Trend (accelerating / fading) -----------------------------------
    trend = str(row.get("trend") or "").lower()
    if trend == "accelerating":
        bullets.append(_bullet(
            TONE_POSITIVE,
            f"{side_word.capitalize()} flow is accelerating",
            "Daily premium is stepping higher — late arrivals are still chasing.",
        ))
    elif trend == "fading":
        bullets.append(_bullet(
            TONE_WARNING,
            "Flow is fading",
            "Daily premium is declining — the move may already be priced in.",
        ))
    elif trend in {"stable", "steady"}:
        bullets.append(_bullet(
            TONE_INFO,
            "Flow is stable day-over-day",
            "Consistent premium without acceleration — accumulation still in force.",
        ))

    # -- 4. Size context (prem/mcap + cumulative) ---------------------------
    bps = _f(row.get("prem_mcap_bps"))
    cum_prem = _f(row.get("cumulative_premium"))
    if bps is not None and bps > 0:
        if bps >= 50.0:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"Outsized vs market cap ({bps:.0f} bps)",
                f"Cumulative premium {_fmt_money(cum_prem)} — hard to hide this one from the tape.",
            ))
        elif bps >= 10.0:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"Notable size ({bps:.0f} bps of market cap)",
                f"Cumulative premium {_fmt_money(cum_prem)} — institutional-size flow.",
            ))
        elif bps >= 1.0:
            bullets.append(_bullet(
                TONE_INFO,
                f"Moderate size ({bps:.1f} bps of market cap)",
                f"Cumulative premium {_fmt_money(cum_prem)}.",
            ))

    # -- 5. 3-day percentile rank (institutional-scale gate) ----------------
    p3d = _f(row.get("perc_3_day_total"))
    if p3d is not None and p3d >= 90:
        bullets.append(_bullet(
            TONE_POSITIVE,
            f"Top {100 - p3d:.0f}% 3-day activity percentile",
            "Running hot vs its own recent history — not just big, but unusually big.",
        ))

    # -- 6. DTE bucket (intent) ---------------------------------------------
    dte_bucket = str(row.get("dominant_dte_bucket") or "").lower()
    if dte_bucket == "short":
        bullets.append(_bullet(
            TONE_WARNING,
            "Short-dated weighting dominates",
            "Most of the flow is 0-7 DTE — lotto / hedge character, high decay risk.",
        ))
    elif dte_bucket == "swing":
        bullets.append(_bullet(
            TONE_POSITIVE,
            "Swing-dated DTE dominates",
            "Flow is concentrated in 8-45 DTE — positioning for a real multi-day move.",
        ))
    elif dte_bucket == "leap":
        bullets.append(_bullet(
            TONE_INFO,
            "LEAP-dated flow",
            "Dominant DTE > 45 days — long-horizon institutional positioning.",
        ))

    # -- 7. Sweep / multileg character --------------------------------------
    sweep = _f(row.get("sweep_share"))
    multileg = _f(row.get("multileg_share"))
    if sweep is not None and sweep >= 0.40:
        bullets.append(_bullet(
            TONE_POSITIVE,
            f"Sweep-heavy ({sweep * 100:.0f}% sweeps)",
            "Urgency print — buyer lifted multiple exchanges simultaneously.",
        ))
    if multileg is not None and multileg >= 0.35:
        bullets.append(_bullet(
            TONE_INFO,
            f"Multileg share {multileg * 100:.0f}%",
            "A large fraction is structured (spreads / risk reversals) — reduces pure-directional read.",
        ))

    # -- 8. OI change -------------------------------------------------------
    oi = _f(row.get("latest_oi_change"))
    if oi is not None:
        if oi >= 20:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"OI stepping up (+{oi:.0f}%)",
                "New positions being built, not just intraday shuffle.",
            ))
        elif oi <= -20:
            bullets.append(_bullet(
                TONE_WARNING,
                f"OI is bleeding ({oi:.0f}%)",
                "Existing positions are being closed — flow may be exits, not opens.",
            ))

    # -- 9. Window return alignment ----------------------------------------
    wr = _f(row.get("window_return_pct"))
    if wr is not None and side:
        if side == "LONG" and wr >= 2.0:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"Underlying +{wr:.1f}% in window",
                "Price is confirming the call-side flow — trend is working.",
            ))
        elif side == "LONG" and wr <= -2.0:
            bullets.append(_bullet(
                TONE_WARNING,
                f"Underlying {wr:.1f}% in window (call flow fighting price)",
                "Buyers haven't been paid yet — late chase risk.",
            ))
        elif side == "SHORT" and wr <= -2.0:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"Underlying {wr:.1f}% in window",
                "Price confirming the put-side flow — trend is working.",
            ))
        elif side == "SHORT" and wr >= 2.0:
            bullets.append(_bullet(
                TONE_WARNING,
                f"Underlying +{wr:.1f}% in window (put flow fighting price)",
                "Bears haven't been paid yet — rally may squeeze premiums.",
            ))

    # -- 10. Dark pool alignment -------------------------------------------
    dp = row.get("dark_pool") or {}
    if isinstance(dp, dict) and dp.get("available"):
        dp_bias = _f(dp.get("bias"), default=0.5)
        dp_bps = _f(dp.get("notional_mcap_bps")) or 0.0
        if side == "LONG" and dp_bias is not None and dp_bias >= 0.60 and dp_bps >= 1.0:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"Dark pool lifting ({dp_bias * 100:.0f}% above mid)",
                f"{dp_bps:.1f} bps notional — institutional shares being accumulated alongside the calls.",
            ))
        elif side == "SHORT" and dp_bias is not None and dp_bias <= 0.40 and dp_bps >= 1.0:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"Dark pool hitting bids ({dp_bias * 100:.0f}% below mid)",
                f"{dp_bps:.1f} bps notional — institutional distribution alongside puts.",
            ))
        elif side == "LONG" and dp_bias is not None and dp_bias <= 0.40 and dp_bps >= 1.0:
            bullets.append(_bullet(
                TONE_WARNING,
                "Dark pool leaning sellers",
                "Calls say bullish but block prints hit bids — mixed institutional signal.",
            ))
        elif side == "SHORT" and dp_bias is not None and dp_bias >= 0.60 and dp_bps >= 1.0:
            bullets.append(_bullet(
                TONE_WARNING,
                "Dark pool leaning buyers",
                "Puts say bearish but block prints lift asks — mixed institutional signal.",
            ))

    # -- 11. Dealer hedge regime --------------------------------------------
    dealer = str(row.get("dealer_hedge_bias") or "").lower()
    if dealer == "chase":
        bullets.append(_bullet(
            TONE_POSITIVE,
            "Dealers are short gamma",
            "Dealer hedging flow amplifies the direction — breakouts tend to extend.",
        ))
    elif dealer == "suppress":
        bullets.append(_bullet(
            TONE_WARNING,
            "Dealers are long gamma (pinning)",
            "Dealer hedging dampens moves — expect slow grind, not rips.",
        ))

    # -- 12. Pin-risk proximity --------------------------------------------
    pin_pct = _f(row.get("pin_risk_distance_pct"))
    if pin_pct is not None and pin_pct <= 1.5:
        pin_k = row.get("pin_risk_strike")
        bullets.append(_bullet(
            TONE_WARNING,
            f"Pin risk {pin_pct:.1f}% away" + (f" ({pin_k})" if pin_k is not None else ""),
            "Heavy open interest at that strike — price may magnetise into expiry.",
        ))

    # -- 13. IV context -----------------------------------------------------
    iv = _f(row.get("latest_iv_rank"))
    iv_delta = _f(row.get("iv_rank_5d_delta"))
    if iv is not None:
        if iv >= 70:
            bullets.append(_bullet(
                TONE_WARNING,
                f"IV rank elevated ({iv:.0f})",
                "Options are rich — prefer debit spreads or short premium over long singles.",
            ))
        elif iv <= 15:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"IV rank cheap ({iv:.0f})",
                "Long-premium plays are reasonably priced — good asymmetric R/R available.",
            ))
    if iv_delta is not None and abs(iv_delta) >= 10:
        if iv_delta > 0:
            bullets.append(_bullet(
                TONE_WARNING,
                f"IV rank up {iv_delta:+.0f} pts over 5 days",
                "Expanding vol — someone is front-running news or repricing risk.",
            ))
        else:
            bullets.append(_bullet(
                TONE_POSITIVE,
                f"IV rank down {iv_delta:+.0f} pts over 5 days",
                "Vol is unwinding — options cheaper than a week ago.",
            ))

    # -- 14. Repeat-flow acceleration (Wave 2) ------------------------------
    if side == "LONG":
        accel = _f(row.get("bullish_accel_ratio"))
    elif side == "SHORT":
        accel = _f(row.get("bearish_accel_ratio"))
    else:
        accel = None
    if accel is not None and accel >= 0.40:
        bullets.append(_bullet(
            TONE_POSITIVE,
            f"Intraday repeat acceleration ({accel * 100:.0f}%)",
            "Multiple aligned prints within 2-hour windows — not a single one-off alert.",
        ))

    # -- 15. Sector context --------------------------------------------------
    sec_cnt = row.get("sector_accumulating_count")
    try:
        sec_cnt = int(sec_cnt) if sec_cnt is not None else None
    except (TypeError, ValueError):
        sec_cnt = None
    if sec_cnt is not None and sec_cnt >= 2:
        bullets.append(_bullet(
            TONE_POSITIVE,
            f"{sec_cnt} other names accumulating in the same sector",
            "Thematic / sector-wide positioning — more than an idiosyncratic print.",
        ))
    elif sec_cnt is not None and sec_cnt <= 0 and row.get("sector"):
        bullets.append(_bullet(
            TONE_INFO,
            "Sector peers not confirming",
            "Flow is idiosyncratic, not sector-wide — single-name bet.",
        ))

    # -- 16. Earnings proximity (catalyst) ----------------------------------
    earn = row.get("earnings") or {}
    if isinstance(earn, dict):
        due = _f(earn.get("days_until_earnings"))
    else:
        due = None
    if due is not None and 0 <= due <= 14:
        if due <= 5:
            bullets.append(_bullet(
                TONE_WARNING,
                f"Earnings in {int(due)} day(s)",
                "Binary event risk — size small and prefer defined-risk structures.",
            ))
        else:
            bullets.append(_bullet(
                TONE_INFO,
                f"Earnings in {int(due)} days",
                "Catalyst window — flow may be pre-positioning for the print.",
            ))

    return bullets[:_MAX_BULLETS]


# ===========================================================================
# Flow Feature narrative (per-day / per-ticker intraday row)
# ===========================================================================
def build_flow_feature_narrative(row: dict | None) -> list[dict]:
    """Return narrative bullets for a candidate / trader-card row.

    Consumes the dict shape produced by ``build_flow_feature_table`` and
    ``view_models.TraderCardView`` — expect keys like ``flow_score_scaled``,
    ``conviction_score``, ``dark_pool``, ``hot_chain``, ``insider``,
    ``session``, ``rs``, ``dealer_hedge_bias`` etc.

    Narrative focuses on *today's snapshot* (intraday tone, RS, hot chain,
    insider) rather than multi-day persistence, which is the Flow Tracker
    view's job.
    """
    if not row or not isinstance(row, dict):
        return []

    bullets: list[dict] = []
    side = _direction_side(row.get("direction"))
    side_word = "bullish" if side == "LONG" else ("bearish" if side == "SHORT" else "mixed")

    # -- 1. Conviction stack headline ---------------------------------------
    stack = row.get("conviction_stack") or {}
    if isinstance(stack, dict):
        s_score = stack.get("score")
        s_tier = stack.get("tier_label")
        if s_score is not None and s_tier:
            if s_score >= 80:
                bullets.append(_bullet(TONE_POSITIVE, f"{s_tier}: elite {side_word} setup ({s_score}/100)",
                                       "Flow, dark pool, chain and price all aligned."))
            elif s_score >= 65:
                bullets.append(_bullet(TONE_POSITIVE, f"{s_tier}: strong {side_word} setup ({s_score}/100)",
                                       "Multiple confirming signals — clean setup."))
            elif s_score >= 50:
                bullets.append(_bullet(TONE_NEUTRAL, f"{s_tier}: moderate {side_word} setup ({s_score}/100)"))
            else:
                bullets.append(_bullet(TONE_WARNING, f"{s_tier}: weak setup ({s_score}/100)",
                                       "Flow without broad confirmation."))

    # -- 2. Raw flow score --------------------------------------------------
    fs = _f(row.get("flow_score_scaled") or row.get("conviction_score"))
    if fs is not None:
        if fs >= 8.0:
            bullets.append(_bullet(TONE_POSITIVE, f"Flow score {fs:.1f}/10",
                                   "Large unusual premium, heavy sweep/multileg, well above ticker baseline."))
        elif fs <= 4.0:
            bullets.append(_bullet(TONE_WARNING, f"Flow score only {fs:.1f}/10",
                                   "Underlying flow is thin — don't over-weight other signals."))

    # -- 3. Dark pool confirmation -----------------------------------------
    dp = row.get("dark_pool") or {}
    if isinstance(dp, dict) and dp.get("available"):
        bias = _f(dp.get("bias"), default=0.5)
        bps = _f(dp.get("notional_mcap_bps")) or 0.0
        if side == "LONG" and bias is not None and bias >= 0.60 and bps >= 1.0:
            bullets.append(_bullet(TONE_POSITIVE, f"Dark pool lifting ({bias * 100:.0f}% > mid)",
                                   f"{bps:.1f} bps notional — institutions accumulating shares."))
        elif side == "SHORT" and bias is not None and bias <= 0.40 and bps >= 1.0:
            bullets.append(_bullet(TONE_POSITIVE, f"Dark pool hitting bids ({bias * 100:.0f}% < mid)",
                                   f"{bps:.1f} bps notional — institutions distributing."))
        elif side == "LONG" and bias is not None and bias <= 0.40 and bps >= 1.0:
            bullets.append(_bullet(TONE_WARNING, "Dark pool opposing long flow",
                                   "Calls are bullish but blocks hit bids — conflicting reads."))
        elif side == "SHORT" and bias is not None and bias >= 0.60 and bps >= 1.0:
            bullets.append(_bullet(TONE_WARNING, "Dark pool opposing short flow",
                                   "Puts are bearish but blocks lift asks — conflicting reads."))

    # -- 4. Hot chain / top chain ------------------------------------------
    hc = row.get("hot_chain") or row.get("top_chain")
    if isinstance(hc, dict):
        is_hot = bool(row.get("hot_chain"))
        call_p = _f(hc.get("call_premium") or hc.get("total_call_premium"))
        put_p = _f(hc.get("put_premium") or hc.get("total_put_premium"))
        dom = (hc.get("dominant_side") or "").upper()
        label = "Hot chain" if is_hot else "Top chain"
        if call_p is not None and put_p is not None:
            if side == "LONG" and call_p >= put_p:
                bullets.append(_bullet(TONE_POSITIVE, f"{label} is call-dominant",
                                       f"Calls {_fmt_money(call_p)} vs puts {_fmt_money(put_p)} — chain agrees with direction."))
            elif side == "SHORT" and put_p >= call_p:
                bullets.append(_bullet(TONE_POSITIVE, f"{label} is put-dominant",
                                       f"Puts {_fmt_money(put_p)} vs calls {_fmt_money(call_p)} — chain agrees with direction."))
            elif side == "LONG":
                bullets.append(_bullet(TONE_WARNING, f"{label} is put-heavy",
                                       "Calls on the chain are not dominant — shallow confirmation."))
            elif side == "SHORT":
                bullets.append(_bullet(TONE_WARNING, f"{label} is call-heavy",
                                       "Puts on the chain are not dominant — shallow confirmation."))
        elif dom:
            bullets.append(_bullet(TONE_INFO, f"{label} dominant side: {dom.title()}"))

    # -- 5. Insider -------------------------------------------------------
    ins = row.get("insider") or {}
    if isinstance(ins, dict):
        nd = (ins.get("net_direction") or "").lower()
        if nd == "buying" and side == "LONG":
            bullets.append(_bullet(TONE_POSITIVE, "Insiders are buying",
                                   "Slow-money directional signal agrees with long flow."))
        elif nd == "selling" and side == "SHORT":
            bullets.append(_bullet(TONE_POSITIVE, "Insiders are selling",
                                   "Slow-money directional signal agrees with short flow."))
        elif nd == "buying" and side == "SHORT":
            bullets.append(_bullet(TONE_WARNING, "Insiders are buying against short flow",
                                   "Opposed slow-money signal — re-check conviction."))
        elif nd == "selling" and side == "LONG":
            bullets.append(_bullet(TONE_WARNING, "Insiders are selling against long flow",
                                   "Opposed slow-money signal — re-check conviction."))

    # -- 6. Session tone ---------------------------------------------------
    sess = row.get("session") or {}
    if isinstance(sess, dict):
        tone = (sess.get("session_tone") or "").upper()
        if side == "LONG" and tone == "STRENGTH":
            bullets.append(_bullet(TONE_POSITIVE, "Intraday strength",
                                   "Tape is trading up — long flow is being rewarded today."))
        elif side == "SHORT" and tone == "WEAKNESS":
            bullets.append(_bullet(TONE_POSITIVE, "Intraday weakness",
                                   "Tape is trading down — short flow is being rewarded today."))
        elif side == "LONG" and tone == "WEAKNESS":
            bullets.append(_bullet(TONE_WARNING, "Fighting intraday weakness",
                                   "Longs fading into a weak tape — prefer small / patient entries."))
        elif side == "SHORT" and tone == "STRENGTH":
            bullets.append(_bullet(TONE_WARNING, "Fighting intraday strength",
                                   "Shorts leaning against a strong tape — squeeze risk."))

    # -- 7. Relative strength ----------------------------------------------
    rs = row.get("rs") or {}
    if isinstance(rs, dict):
        rs5 = _f(rs.get("rs_5d_pct"))
        rs20 = _f(rs.get("rs_20d_pct"))
        if side == "LONG" and rs5 is not None and rs5 > 0 and rs20 is not None and rs20 > 0:
            bullets.append(_bullet(TONE_POSITIVE, f"Outperforming SPY ({rs5:+.1f}% / {rs20:+.1f}% RS)",
                                   "Leading over both short- and medium-term windows."))
        elif side == "SHORT" and rs5 is not None and rs5 < 0 and rs20 is not None and rs20 < 0:
            bullets.append(_bullet(TONE_POSITIVE, f"Underperforming SPY ({rs5:+.1f}% / {rs20:+.1f}% RS)",
                                   "Lagging over both short- and medium-term windows."))
        elif side == "LONG" and rs20 is not None and rs20 < -2:
            bullets.append(_bullet(TONE_WARNING, f"Lagging SPY 20d ({rs20:+.1f}%)",
                                   "Call flow fighting a weak relative-strength backdrop."))
        elif side == "SHORT" and rs20 is not None and rs20 > 2:
            bullets.append(_bullet(TONE_WARNING, f"Leading SPY 20d ({rs20:+.1f}%)",
                                   "Put flow fighting a strong relative-strength backdrop."))

    # -- 8. Dealer hedge ---------------------------------------------------
    dealer = str(row.get("dealer_hedge_bias") or "").lower()
    if dealer == "chase":
        bullets.append(_bullet(TONE_POSITIVE, "Dealers short gamma",
                               "Dealer hedging amplifies moves — breakouts extend."))
    elif dealer == "suppress":
        bullets.append(_bullet(TONE_WARNING, "Dealers long gamma (pinning)",
                               "Dealer hedging dampens moves — expect grind, not rip."))

    # -- 9. IV context -----------------------------------------------------
    iv = _f(row.get("iv_rank"))
    if iv is not None:
        if iv >= 70:
            bullets.append(_bullet(TONE_WARNING, f"IV rank rich ({iv:.0f})",
                                   "Prefer spreads over long premium here."))
        elif iv <= 15:
            bullets.append(_bullet(TONE_POSITIVE, f"IV rank cheap ({iv:.0f})",
                                   "Long-premium plays are reasonably priced."))

    # -- 10. Earnings / catalyst -------------------------------------------
    earn = row.get("earnings") or {}
    if isinstance(earn, dict):
        due = _f(earn.get("days_until_earnings"))
        if due is not None and 0 <= due <= 14:
            if due <= 5:
                bullets.append(_bullet(TONE_WARNING, f"Earnings in {int(due)} day(s)",
                                       "Binary event — size small, prefer defined-risk structures."))
            else:
                bullets.append(_bullet(TONE_INFO, f"Earnings in {int(due)} days",
                                       "Catalyst window — flow may be pre-positioning."))

    return bullets[:_MAX_BULLETS]


__all__ = [
    "build_flow_tracker_narrative",
    "build_flow_feature_narrative",
    "TONE_POSITIVE",
    "TONE_NEGATIVE",
    "TONE_WARNING",
    "TONE_INFO",
    "TONE_NEUTRAL",
]

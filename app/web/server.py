"""Vector Alpha — swing-quant dashboard with sidebar navigation."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from app.config import (
    DP_TRACKER_LOOKBACK_DAYS,
    DP_TRACKER_MIN_ACTIVE_DAYS,
    FLOW_TRACKER_AUTO_WIDEN_MIN,
    FLOW_TRACKER_HORIZON_DEFAULT,
    FLOW_TRACKER_HORIZONS,
    FLOW_TRACKER_LOOKBACK_DAYS,
    FLOW_TRACKER_MIN_ACTIVE_DAYS,
    FLOW_TRACKER_MODE_DEFAULT,
    MIN_FINAL_SCORE,
    REGIME_THRESHOLD_BOOST,
)
from app.analytics.grade_backtest import format_header as _grade_stats_header, load_grade_stats
from app.features.decision_context import (
    compute_liquidity,
    compute_r_at_market,
    compute_session_context,
    compute_rs,
    suggest_expression,
    enrich_signal,
)
from app.features.dp_stats import attach_dp_z_tiers
from app.features.dark_pool_tracker import (
    aggregate_daily_accumulated,
    aggregate_dark_pool_prints,
    compute_multi_day_dp,
    load_daily_accumulated,
)
from app.features.flow_tracker import compute_multi_day_flow
from app.features.hottest_chains import aggregate_chains_by_ticker
from app.features.insider_tracker import classify_insider_activity
from app.features.sentiment_tracker import compute_sentiment_trend
from app.web.view_models import build_trader_card_rows
from app.web.data_access import (
    load_dark_pool_recent,
    load_equity_curve,
    load_equity_curve_agent,
    load_final_signals,
    load_flow_features,
    load_hottest_chains,
    load_insider_recent,
    load_positions,
    load_positions_agent,
    load_rejected,
    load_trade_log,
    load_trade_log_agent,
    load_trade_log_tail,
    load_watchlist,
)

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.after_request
def _add_cors(response):
    origin = request.headers.get("Origin", "")
    if origin and ("github.io" in origin or "localhost" in origin or "127.0.0.1" in origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

_REGIME_PATH = Path(__file__).resolve().parents[2] / "data" / "market_regime.json"


def load_market_regime() -> dict:
    """Load the latest market regime snapshot from disk."""
    try:
        return json.loads(_REGIME_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


TABLE_PAGE_SIZE = 7

ALERT_DISPLAY_COLS = [
    "event_ts",
    "ticker",
    "option_type",
    "strike",
    "expiration_date",
    "premium",
    "prem_mcap_bps",
    "contracts",
    "volume",
    "open_interest",
    "dte",
    "underlying_price",
    "direction",
    "execution_side",
    "alert_rule",
]

_BADGE_MAP = {
    "LONG": '<span class="badge badge-long">LONG</span>',
    "SHORT": '<span class="badge badge-short">SHORT</span>',
    "POSITIVE": '<span class="badge badge-positive">POSITIVE</span>',
    "NEGATIVE": '<span class="badge badge-negative">NEGATIVE</span>',
    "NEUTRAL": '<span class="badge badge-neutral">NEUTRAL</span>',
}

# ── Column tooltip descriptions ──────────────────────────────────────────────

_COLUMN_TOOLTIPS: dict[str, str] = {
    # Signals
    "ticker": "Stock symbol",
    "direction": "Trade direction (LONG / SHORT)",
    "pattern": "Price pattern that triggered the signal",
    "flow_score_raw": "Raw directional flow score before normalization",
    "flow_score_scaled": "Flow score normalized to 0\u201310 across current batch",
    "price_score": "Technical price-action score (0\u201310)",
    "final_score": "Combined conviction (60% flow + 40% price)",
    "gamma_regime": "Options gamma regime (POSITIVE dampens, NEGATIVE amplifies)",
    "net_gex": "Net gamma exposure across all strikes",
    "gamma_flip_level_estimate": "Estimated price where net gamma flips sign",
    "nearest_call_wall": "Highest call OI strike at or above spot",
    "nearest_put_wall": "Highest put OI strike at or below spot",
    "distance_to_call_wall_pct": "Distance to nearest overhead call OI wall (%)",
    "distance_to_put_wall_pct": "Distance to nearest downside put OI wall (%)",
    "ticker_call_oi": "Total call open interest",
    "ticker_put_oi": "Total put open interest",
    "ticker_put_call_ratio": "Put/call OI ratio (>1 = more put positioning)",
    "near_term_oi": "OI in expirations < 14 DTE",
    "swing_dte_oi": "OI in 14–90 DTE expirations",
    "long_dated_oi": "OI in expirations > 90 DTE",
    "entry_price": "Suggested entry price",
    "stop_price": "Initial stop-loss price",
    "target_1": "First profit target",
    "target_2": "Second profit target",
    "rr_ratio": "Reward-to-risk ratio",
    "time_stop_days": "Maximum holding period (days)",
    "source": "Signal origin (fresh scan or watchlist)",
    # Candidates
    "bullish_score": "Weighted bullish flow conviction (0\u201310)",
    "bearish_score": "Weighted bearish flow conviction (0\u201310)",
    "bullish_flow_intensity": "Bullish premium / market cap (basis points)",
    "bearish_flow_intensity": "Bearish premium / market cap (basis points)",
    "bullish_ppt_bps": "Bullish premium-per-trade (bps of market cap)",
    "bearish_ppt_bps": "Bearish premium-per-trade (bps of market cap)",
    "flow_imbalance_ratio": "Bullish prem \u00f7 bearish prem (\u003e1 = bullish dominant)",
    "dominant_direction": "Which side has stronger flow",
    "total_premium": "Total options premium across all flow",
    "total_count": "Count of qualifying options-flow prints rolled into this ticker row",
    "avg_dte": "Average days to expiration of flow",
    "dte_score": "DTE quality score (higher = better positioned)",
    "marketcap": "Market capitalisation; flow intensity = premium / market cap",
    # Alerts
    "event_ts": "Timestamp of the flow event",
    "option_type": "Call or put",
    "strike": "Option strike price",
    "expiration_date": "Option expiration date",
    "premium": "Total premium of the trade",
    "flow_intensity": "Premium / market cap (basis points)",
    "contracts": "Number of contracts traded",
    "volume": "Option volume",
    "open_interest": "Current open interest",
    "dte": "Days to expiration",
    "execution_side": "Executed at bid or ask",
    "alert_rule": "UW alert rule that triggered",
    # Trade log / positions
    "entry_date": "Date position was opened",
    "exit_date": "Date position was closed",
    "exit_price": "Price at exit",
    "shares": "Number of shares",
    "risk_pct": "Portfolio risk allocated (%)",
    "pnl_pct": "Profit/loss as percentage of entry",
    "pnl_dollar": "Profit/loss in dollars",
    "r_multiple": "P&L expressed as multiples of initial risk",
    "days_held": "Calendar days the position was open",
    "exit_reason": "Why the position was closed",
    "trail_method": "Trailing stop method used",
    "partial_pnl_pct": "P&L from partial exits (%)",
    "opened_at": "UTC timestamp when the position was opened from a signal",
    "initial_stop": "Original stop from the signal (before trailing)",
    "active_stop": "Current working stop (trailing may have tightened)",
    "position_value": "Notional at entry (shares × entry price)",
}

# ── Color-coding rules ───────────────────────────────────────────────────────

_SCORE_COLS = {
    "bullish_score", "bearish_score", "final_score",
    "price_score", "flow_score_scaled", "dte_score",
}
_PNL_COLS = {"pnl_dollar", "pnl_pct", "r_multiple"}
_BULLISH_INTENSITY_COLS = {"bullish_flow_intensity"}
_BEARISH_INTENSITY_COLS = {"bearish_flow_intensity"}
_INTENSITY_COLS = _BULLISH_INTENSITY_COLS | _BEARISH_INTENSITY_COLS | {"flow_intensity"}


def _wrap_color(val, positive: bool) -> str:
    cls = "positive" if positive else "negative"
    return f'<span class="{cls}">{val}</span>'


CONVICTION_HIGH = 7.5
CONVICTION_LOW = 5.0


def _wrap_conviction(val) -> str:
    """Green if >= CONVICTION_HIGH, red if < CONVICTION_LOW, else plain text."""
    if pd.isna(val) or str(val) in ("", "—"):
        return val if isinstance(val, str) else str(val)
    try:
        x = float(val)
    except (TypeError, ValueError):
        return str(val)
    if x >= CONVICTION_HIGH:
        return _wrap_color(val, True)
    if x < CONVICTION_LOW:
        return _wrap_color(val, False)
    return str(val)


def _wrap_conviction_intensity(val) -> str:
    """Like _wrap_conviction; leave zero / empty uncolored."""
    if pd.isna(val) or str(val) in ("", "—"):
        return val if isinstance(val, str) else str(val)
    try:
        x = float(val)
    except (TypeError, ValueError):
        return str(val)
    if x <= 0:
        return str(val)
    return _wrap_conviction(val)


def _conviction_span(val) -> str:
    """HTML span for card views (numeric formatting)."""
    if val is None or (isinstance(val, float) and pd.isna(val)) or str(val) == "":
        return "—"
    try:
        x = float(val)
    except (TypeError, ValueError):
        return html_escape(str(val))
    disp = f"{x:.2f}"
    if x >= CONVICTION_HIGH:
        return f'<span class="positive">{disp}</span>'
    if x < CONVICTION_LOW:
        return f'<span class="negative">{disp}</span>'
    return disp


def _fmt_card_num(val, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)) or str(val) == "":
        return "—"
    try:
        f = float(val)
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return html_escape(str(val))


def _fmt_iv_rank(val) -> str:
    """Format IV rank as a compact coloured string for card views."""
    if val is None or (isinstance(val, float) and pd.isna(val)) or str(val) in ("", "nan"):
        return "—"
    try:
        iv = float(val)
    except (TypeError, ValueError):
        return "—"
    label = f"{iv:.0f}"
    if 15 <= iv <= 45:
        return f'<span class="positive">{label}</span>'
    if iv > 75 or iv <= 5:
        return f'<span class="negative">{label}</span>'
    return label


def _prem_mcap_label(bps) -> tuple[str, str]:
    """Return (label, css_class) for a prem/mcap basis-point value."""
    if bps is None or (isinstance(bps, float) and pd.isna(bps)):
        return ("—", "")
    try:
        v = float(bps)
    except (TypeError, ValueError):
        return ("—", "")
    if v > 5:
        return ("Outsized", "positive")
    if v >= 1:
        return ("Notable", "")
    if v >= 0.1:
        return ("Normal", "")
    return ("Minor", "")


def _enrich_flow_tracker_ztier(flow_tracker: list[dict]) -> list[dict]:
    """Attach rolling z-score tiers from the latest flow_features snapshot.

    Reads ``data/flow_features/flow_features_*.csv`` (one file per pipeline
    run) and joins ``bullish_zscore_tier`` / ``bearish_zscore_tier`` onto
    each tracker row keyed by ticker.  Picks the worst (numerically highest)
    tier on the traded side so the UI chip reflects the weakest baseline
    behind the score.  Wave 0.5 C2.
    """
    if not flow_tracker:
        return flow_tracker

    try:
        from app.web.data_access import load_flow_features
        from app.features.flow_stats import TIER_LABELS

        ff, _name = load_flow_features()
        if ff is None or ff.empty or "ticker" not in ff.columns:
            return flow_tracker

        keep_cols = ["ticker"]
        for c in ("bullish_zscore_tier", "bearish_zscore_tier"):
            if c in ff.columns:
                keep_cols.append(c)
        ff = ff[keep_cols].copy()
        ff["ticker"] = ff["ticker"].astype(str).str.upper().str.strip()
        ff = ff.drop_duplicates(subset=["ticker"], keep="last")
        by_t: dict[str, dict] = {row["ticker"]: row for _, row in ff.iterrows()}

        for ft in flow_tracker:
            row = by_t.get(str(ft.get("ticker", "")).upper().strip())
            if row is None:
                ft["z_tier"] = None
                ft["z_tier_label"] = None
                continue

            direction = str(ft.get("direction") or "").upper()
            tier_int: int | None = None
            if direction == "BULLISH":
                val = row.get("bullish_zscore_tier")
            elif direction == "BEARISH":
                val = row.get("bearish_zscore_tier")
            else:
                # Unknown direction → pick the worst of the two.
                vals = [row.get("bullish_zscore_tier"), row.get("bearish_zscore_tier")]
                vals = [v for v in vals if v is not None and not pd.isna(v)]
                val = max(vals) if vals else None

            if val is not None and not pd.isna(val):
                try:
                    tier_int = int(val)
                except (TypeError, ValueError):
                    tier_int = None

            ft["z_tier"] = tier_int
            ft["z_tier_label"] = TIER_LABELS.get(tier_int) if tier_int is not None else None
    except Exception as e:
        print(f"  [flow-tracker] z-tier enrichment skipped: {e}")

    return flow_tracker


def _enrich_flow_tracker_sentiment(flow_tracker: list[dict]) -> list[dict]:
    """Merge sentiment trend data into each Flow Tracker ticker dict."""
    if not flow_tracker:
        return flow_tracker
    tickers = [ft["ticker"] for ft in flow_tracker]
    sentiment = compute_sentiment_trend(tickers)
    for ft in flow_tracker:
        sent = sentiment.get(ft["ticker"], {})
        ft["sentiment"] = sent if sent else {
            "combined_label": "neutral",
            "combined_sentiment": 0.5,
            "mention_spike": False,
            "st_messages": 0,
            "st_mention_trend": "stable",
            "rd_mentions": 0,
            "rd_mention_trend": "stable",
            "daily_mentions": [],
        }
    return flow_tracker


def _enrich_flow_tracker_dp(
    flow_tracker: list[dict],
    dp_tracker: list[dict],
) -> list[dict]:
    """Attach multi-day dark pool data to Flow Tracker entries for convergence."""
    if not flow_tracker or not dp_tracker:
        return flow_tracker

    dp_by_ticker = {d["ticker"]: d for d in dp_tracker}

    for ft in flow_tracker:
        dp = dp_by_ticker.get(ft["ticker"])
        if not dp:
            ft["dp"] = None
            ft["dp_aligned"] = False
            ft["dp_divergent"] = False
            continue

        ft["dp"] = {
            "bias": dp["bias"],
            "bias_label": dp["bias_label"],
            "bias_consistency_label": dp["bias_consistency_label"],
            "cumulative_notional": dp["cumulative_notional"],
            "notional_mcap_bps": dp["notional_mcap_bps"],
            "trend": dp["trend"],
            "active_days": dp["active_days"],
            "total_days": dp["total_days"],
            "daily_snapshots": dp["daily_snapshots"],
        }

        flow_dir = ft.get("direction", "")
        dp_bias = dp["bias"]
        if flow_dir == "BULLISH" and dp_bias >= 0.55:
            ft["dp_aligned"] = True
            ft["dp_divergent"] = False
        elif flow_dir == "BEARISH" and dp_bias <= 0.45:
            ft["dp_aligned"] = True
            ft["dp_divergent"] = False
        elif flow_dir == "BULLISH" and dp_bias <= 0.45:
            ft["dp_aligned"] = False
            ft["dp_divergent"] = True
        elif flow_dir == "BEARISH" and dp_bias >= 0.55:
            ft["dp_aligned"] = False
            ft["dp_divergent"] = True
        else:
            ft["dp_aligned"] = False
            ft["dp_divergent"] = False

        # Wave 0.5 A8 — dark-pool alignment bonus.  Options flow + dark pool
        # pointing the same way is institutional conviction you rarely see
        # without a catalyst; multiply conviction_score by (1 + bonus) capped
        # at DP_ALIGNMENT_MAX_BONUS and scaled by |bias - 0.5| × notional.
        from app.config import DP_ALIGNMENT_MAX_BONUS, DP_ALIGNMENT_MIN_NOTIONAL_BPS

        bonus = 0.0
        if ft.get("dp_aligned"):
            notional_bps = float(dp.get("notional_mcap_bps") or 0.0)
            if notional_bps >= DP_ALIGNMENT_MIN_NOTIONAL_BPS:
                # Strength: notional coverage (saturating at 20 bps) × bias conviction.
                notional_strength = min(notional_bps / 20.0, 1.0)
                bias_strength = min(abs(float(dp_bias) - 0.5) / 0.30, 1.0)
                bonus = DP_ALIGNMENT_MAX_BONUS * notional_strength * bias_strength
        ft["dp_alignment_bonus"] = round(bonus, 3)

        if bonus > 0 and "conviction_score" in ft:
            original = float(ft["conviction_score"])
            boosted = round(original * (1.0 + bonus), 1)
            ft["conviction_score"] = boosted
            ft["_dp_boost_delta"] = round(boosted - original, 1)
            # Re-grade with the boosted score so the UI badge stays coherent.
            try:
                from app.features.grade_explainer import conviction_grade
                ft["conviction_grade"] = conviction_grade(boosted)
            except Exception:
                pass

    return flow_tracker


def _enrich_flow_tracker_chains(
    flow_tracker: list[dict],
    chains_by_ticker: dict[str, dict],
) -> list[dict]:
    """Attach hottest chain data to Flow Tracker entries."""
    if not flow_tracker or not chains_by_ticker:
        return flow_tracker
    for ft in flow_tracker:
        hc = chains_by_ticker.get(ft["ticker"])
        if hc:
            ft["hot_chain"] = {
                "top_chain_label": hc.get("top_chain_label"),
                "top_chain_vol_oi": hc.get("top_chain_vol_oi", 0),
                "top_chain_ask_pct": hc.get("top_chain_ask_pct", 0),
                "top_chain_premium": hc.get("top_chain_premium", 0),
                "contract_count": hc.get("contract_count", 0),
                "total_premium": hc.get("total_premium", 0),
                "dominant_side": hc.get("dominant_side", "—"),
            }
        else:
            ft["hot_chain"] = None
    return flow_tracker


def _enrich_flow_tracker_insider(
    flow_tracker: list[dict],
    insider_by_ticker: dict[str, dict],
) -> list[dict]:
    """Attach insider transaction data to Flow Tracker entries."""
    if not flow_tracker or not insider_by_ticker:
        return flow_tracker
    for ft in flow_tracker:
        ins = insider_by_ticker.get(ft["ticker"])
        if ins and (ins["buy_count"] > 0 or ins["sell_count"] > 0):
            ft["insider"] = ins
        else:
            ft["insider"] = None
    return flow_tracker


def _enrich_flow_tracker_earnings(flow_tracker: list[dict]) -> list[dict]:
    """Attach earnings date data to Flow Tracker entries from cached pipeline data."""
    if not flow_tracker:
        return flow_tracker

    import json
    er_path = Path(__file__).resolve().parents[2] / "data" / "earnings_cache.json"
    try:
        earnings_cache = json.loads(er_path.read_text()) if er_path.is_file() else {}
    except Exception:
        earnings_cache = {}

    for ft in flow_tracker:
        er = earnings_cache.get(ft["ticker"])
        if er and er.get("next_earnings_date"):
            ft["earnings"] = er
        else:
            ft["earnings"] = None
    return flow_tracker


def _dir_badge_html(direction) -> str:
    d = str(direction or "").strip().upper()
    return _BADGE_MAP.get(d, html_escape(str(direction or "—")))


def _gamma_badge_html(gamma) -> str:
    if gamma is None or (isinstance(gamma, float) and pd.isna(gamma)) or str(gamma).strip() == "":
        return "—"
    g = str(gamma).strip().upper()
    return _BADGE_MAP.get(g, html_escape(str(gamma)))


def _scale_1_10(series: pd.Series) -> pd.Series:
    """Min-max rescale a numeric series to the 1-10 range for display."""
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return series.where(series.isna(), 5.0)
    return 1.0 + 9.0 * (series - mn) / (mx - mn)


def _scale_intensity_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale flow intensity columns to 1-10 for display readability."""
    df = df.copy()
    for col in _INTENSITY_COLS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            nonzero = numeric[numeric > 0]
            if not nonzero.empty:
                df[col] = np.where(numeric > 0, _scale_1_10(numeric.clip(lower=0)), 0.0)
    return df


_HIGH_PRECISION_COLS = {
    "bullish_flow_intensity", "bearish_flow_intensity",
    "bullish_ppt_bps", "bearish_ppt_bps",
    "iv_current",
}

def _round_floats(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round all float columns in a DataFrame for cleaner display."""
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols):
        df = df.copy()
        normal = [c for c in float_cols if c not in _HIGH_PRECISION_COLS]
        precise = [c for c in float_cols if c in _HIGH_PRECISION_COLS]
        if normal:
            df[normal] = df[normal].round(decimals)
        if precise:
            df[precise] = df[precise].round(6)
    return df


def _colorize_values(df: pd.DataFrame) -> pd.DataFrame:
    """Wrap numeric cell values with green/red color spans based on column semantics."""
    df = df.copy()
    for col in df.columns:
        if col in _SCORE_COLS:
            df[col] = df[col].apply(
                lambda v: _wrap_conviction(v) if pd.notna(v) and str(v) not in ("", "—") else v
            )
        elif col in _PNL_COLS:
            df[col] = df[col].apply(
                lambda v: _wrap_color(v, float(v) > 0)
                if pd.notna(v) and str(v) not in ("", "—") else v
            )
        elif col in _INTENSITY_COLS:
            df[col] = df[col].apply(
                lambda v: _wrap_conviction_intensity(v) if pd.notna(v) and str(v) not in ("", "—") else v
            )
        elif col == "flow_imbalance_ratio":

            def _imb_color(v):
                if pd.isna(v) or str(v) in ("", "—"):
                    return v
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    return v
                if x > 1.2:
                    return _wrap_color(v, True)
                if x < 0.83:
                    return _wrap_color(v, False)
                return v

            df[col] = df[col].apply(_imb_color)
        elif col == "dominant_direction":
            df[col] = df[col].map(
                lambda v: _BADGE_MAP.get(str(v).upper(), str(v))
            )
        elif col == "rr_ratio":
            df[col] = df[col].apply(
                lambda v: _wrap_color(v, float(v) >= 2)
                if pd.notna(v) and str(v) not in ("", "—") else v
            )
    return df


def _apply_badges(df: pd.DataFrame) -> pd.DataFrame:
    """Replace direction/gamma_regime values with colored HTML badge spans."""
    df = df.copy()
    for col in ("direction", "gamma_regime"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: _BADGE_MAP.get(str(v).upper(), str(v)))
    return df


def _inject_tooltips(html: str) -> str:
    """Add data-tip attributes to <th> elements for CSS tooltip display."""
    for col, tip in _COLUMN_TOOLTIPS.items():
        safe_tip = html_escape(tip, quote=True)
        html = html.replace(f"<th>{col}</th>", f'<th data-tip="{safe_tip}">{col}</th>')
    return html


def _df_to_detail_json(
    df: pd.DataFrame,
    *,
    risk_regime: dict | None = None,
) -> list[dict]:
    """Convert a DataFrame to a list of dicts suitable for JSON serialization.

    Wave 6 — every detail row is additionally decorated with
    ``conviction_stack`` and ``narrative`` so the Trader Card modal's
    "Why?" tab has prose to render.

    Wave 8 — also attaches ``sizing_context`` to the trade_structure
    payload so the Structure tab's modal shows the regime checks and
    HALT caveats identically to the Flow Tracker rows.
    """
    if df.empty:
        return []
    rounded = _round_floats(df)
    rows = rounded.fillna("").to_dict(orient="records")

    try:
        from app.features.conviction_stack import compute_conviction_stack
    except Exception:
        compute_conviction_stack = None  # type: ignore[assignment]
    try:
        from app.features.flow_narrative import build_flow_feature_narrative
    except Exception:
        build_flow_feature_narrative = None  # type: ignore[assignment]
    try:
        from app.features.trade_structure import recommend_structure
    except Exception:
        recommend_structure = None  # type: ignore[assignment]
    try:
        from app.web.view_models import attach_sizing_context
    except Exception:
        attach_sizing_context = None  # type: ignore[assignment]

    for r in rows:
        if compute_conviction_stack is not None:
            try:
                r["conviction_stack"] = compute_conviction_stack(r)
            except Exception:
                r.setdefault("conviction_stack", None)
        if build_flow_feature_narrative is not None:
            try:
                r["narrative"] = build_flow_feature_narrative(r)
            except Exception:
                r.setdefault("narrative", [])
        if recommend_structure is not None:
            try:
                r["trade_structure"] = recommend_structure(r)
            except Exception:
                r.setdefault("trade_structure", None)
        if attach_sizing_context is not None and risk_regime is not None:
            try:
                r["trade_structure"] = attach_sizing_context(
                    r.get("trade_structure"), risk_regime
                )
            except Exception:
                pass
    return rows





def _enrich_watchlist_from_rejected(
    watchlist: list[dict], rejected: pd.DataFrame
) -> list[dict]:
    """Fill missing price-check fields on watchlist rows from the latest rejected export."""
    if not watchlist or rejected.empty:
        return watchlist
    if not {"ticker", "direction"}.issubset(rejected.columns):
        return watchlist

    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in rejected.iterrows():
        k = (str(row["ticker"]).strip(), str(row["direction"]).strip())
        lookup[k] = row

    def _is_empty(v) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and not v.strip():
            return True
        return False

    out: list[dict] = []
    for w in watchlist:
        wd = dict(w)
        k = (str(wd.get("ticker", "")).strip(), str(wd.get("direction", "")).strip())
        row = lookup.get(k)
        if row is None:
            out.append(wd)
            continue
        for col in ("checks_passed", "checks_failed", "reject_reason"):
            if col not in rejected.columns or not _is_empty(wd.get(col)):
                continue
            val = row[col]
            if pd.isna(val):
                continue
            wd[col] = str(val).strip()
        if "price_score" in rejected.columns and _is_empty(wd.get("price_score")):
            val = row["price_score"]
            if not pd.isna(val):
                try:
                    wd["price_score"] = float(val)
                except (TypeError, ValueError):
                    wd["price_score"] = val
        out.append(wd)
    return out


def _filter_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Keep only the columns present in both the DataFrame and the whitelist."""
    keep = [c for c in cols if c in df.columns]
    return df[keep] if keep else df


def _int_query(name: str, default: int = 1) -> int:
    try:
        return max(1, int(request.args.get(name, default)))
    except (TypeError, ValueError):
        return default


SORT_COLUMNS = {"final_score", "flow_score_scaled", "price_score", "options_context_score"}


def _sort_df(df: pd.DataFrame, sort_key: str | None) -> pd.DataFrame:
    if df.empty or not sort_key or sort_key not in SORT_COLUMNS:
        return df
    if sort_key not in df.columns:
        return df
    return df.sort_values(sort_key, ascending=False, na_position="last")


def _paginate_slice(
    df: pd.DataFrame, page: int, per_page: int
) -> tuple[pd.DataFrame, int, int, int]:
    """Return (slice, total_rows, total_pages, clamped_page)."""
    n = len(df)
    if n == 0:
        return df, 0, 1, 1
    total_pages = (n + per_page - 1) // per_page
    page = min(max(1, page), total_pages)
    start = (page - 1) * per_page
    return df.iloc[start : start + per_page], n, total_pages, page


def _pager_query_url(param: str, target_page: int) -> str:
    args = request.args.to_dict()
    args[param] = str(target_page)
    qs = urlencode(sorted(args.items()))
    return f"{request.path}?{qs}" if qs else request.path


def _page_numbers(total_pages: int, current: int) -> list[int | None]:
    """Page indices for pager; None means ellipsis gap."""
    if total_pages <= 15:
        return list(range(1, total_pages + 1))
    keep = {
        1,
        total_pages,
        current,
        current - 1,
        current + 1,
        current - 2,
        current + 2,
    }
    pages = sorted(p for p in keep if 1 <= p <= total_pages)
    out: list[int | None] = []
    last = 0
    for p in pages:
        if last and p > last + 1:
            out.append(None)
        out.append(p)
        last = p
    return out


def _pager_html(param: str, page: int, total_pages: int, total_rows: int) -> str:
    if total_rows == 0:
        return ""
    row_word = "row" if total_rows == 1 else "rows"
    meta = f'<span class="pager-meta">{total_rows} {row_word}</span>'
    if total_pages <= 1:
        return f'<div class="table-pager" role="navigation" aria-label="Pagination">{meta}</div>'

    prev_el = (
        f'<a class="pager-btn" href="{html_escape(_pager_query_url(param, page - 1))}">Prev</a>'
        if page > 1
        else '<span class="pager-btn disabled" aria-hidden="true">Prev</span>'
    )
    next_el = (
        f'<a class="pager-btn" href="{html_escape(_pager_query_url(param, page + 1))}">Next</a>'
        if page < total_pages
        else '<span class="pager-btn disabled" aria-hidden="true">Next</span>'
    )

    nums: list[str] = []
    for item in _page_numbers(total_pages, page):
        if item is None:
            nums.append('<span class="pager-gap" aria-hidden="true">…</span>')
        elif item == page:
            nums.append(f'<span class="pager-num current" aria-current="page">{item}</span>')
        else:
            u = html_escape(_pager_query_url(param, item))
            nums.append(f'<a class="pager-num" href="{u}">{item}</a>')

    inner = (
        f'{meta}<div class="pager-controls">{prev_el}'
        f'<div class="pager-pages">{" ".join(nums)}</div>{next_el}</div>'
    )
    return f'<div class="table-pager" role="navigation" aria-label="Pagination">{inner}</div>'


def _empty_cards_html(msg: str) -> str:
    return f'<p class="empty">{html_escape(msg)}</p>'


def _row_scalar(row: pd.Series, key: str):
    if key not in row.index:
        return None
    v = row[key]
    if pd.isna(v):
        return None
    return v


def _fmt_money_short(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return html_escape(str(v))
    ax = abs(x)
    if ax >= 1e9:
        return f"${x / 1e9:.2f}B"
    if ax >= 1e6:
        return f"${x / 1e6:.2f}M"
    if ax >= 1e3:
        return f"${x / 1e3:.2f}K"
    return f"${x:.2f}"


def _card_metrics_dl(pairs: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for label, val in pairs:
        if val is None:
            continue
        s = str(val).strip()
        if not s or s == "—":
            continue
        parts.append(f"<dt>{html_escape(label)}</dt><dd>{val}</dd>")
    if not parts:
        return ""
    return '<dl class="card-metrics">' + "".join(parts) + "</dl>"


def _data_card_article(
    ticker: str,
    *,
    head_badges_html: str,
    metrics_html: str,
    extra_html: str = "",
    clickable: bool = False,
    direction: str = "",
    filter_attrs: dict[str, str] | None = None,
    extra_card_cls: str = "",
) -> str:
    tcls = "data-card-ticker ticker-link" if clickable else "data-card-ticker"
    icls = " data-card--interactive" if clickable else ""
    extra_cls = f" {extra_card_cls}" if extra_card_cls else ""
    dir_attr = f' data-direction="{html_escape(str(direction))}"' if direction else ""
    extra_attrs = ""
    if filter_attrs:
        for k, v in filter_attrs.items():
            extra_attrs += f' data-{k}="{html_escape(str(v))}"'
    return (
        f'<article class="data-card{icls}{extra_cls}" data-ticker="{html_escape(str(ticker))}"{dir_attr}{extra_attrs}>'
        f'<header class="data-card-head"><span class="{tcls}">{html_escape(str(ticker))}</span>'
        f"{head_badges_html}</header>"
        f'<div class="data-card-body">{metrics_html}{extra_html}</div></article>'
    )


def _filter_attrs_from_row(row) -> dict[str, str]:
    """Extract filterable data-* attributes from a candidate/rejected/signal row."""
    def _safe(col):
        v = _row_scalar(row, col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    pat = _safe("pattern")
    if pat in ("nan", "unknown"):
        pat = ""
    return {
        "pattern": pat,
        "reject": _safe("reject_reason").split("(")[0].strip(),
        "flow": _safe("flow_score_scaled"),
        "price": _safe("price_score"),
        "options": _safe("options_context_score"),
        "final": _safe("final_score"),
        "passed": _safe("checks_passed"),
        "failed": _safe("checks_failed"),
    }


def _paginate_card_fragments(
    fragments: list[str],
    *,
    page: int,
    per_page: int,
    page_param: str,
    empty_msg: str,
) -> str:
    n = len(fragments)
    if n == 0:
        return _empty_cards_html(empty_msg)
    total_pages = max(1, (n + per_page - 1) // per_page)
    page = min(max(1, page), total_pages)
    start = (page - 1) * per_page
    chunk = fragments[start : start + per_page]
    return f'<div class="card-grid">{"".join(chunk)}</div>' + _pager_html(
        page_param, page, total_pages, n
    )


def _signals_card_fragments(df: pd.DataFrame, *, clickable: bool = True) -> list[str]:
    if df.empty:
        return []
    view = _round_floats(df.copy())
    out: list[str] = []
    for _, row in view.iterrows():
        t = str(row.get("ticker", "")).strip()
        if not t:
            continue
        direc = _dir_badge_html(row.get("direction"))
        gamma = _gamma_badge_html(row.get("gamma_regime"))
        ct_badge = '<span class="badge badge-counter-trend">Counter-trend</span>' if _row_scalar(row, "counter_trend") else ""
        head = f'<span class="data-card-badges">{direc}{gamma}{ct_badge}</span>'
        pairs: list[tuple[str, str]] = [
            ("Final", _conviction_span(_row_scalar(row, "final_score"))),
            ("Flow", _conviction_span(_row_scalar(row, "flow_score_scaled"))),
            ("Price", _conviction_span(_row_scalar(row, "price_score"))),
        ]
        metrics = _card_metrics_dl(pairs)
        pat = _row_scalar(row, "pattern")
        pat_p = (
            f'<p class="data-card-thesis">{html_escape(str(pat))}</p>'
            if pat is not None and str(pat).strip() and str(pat) != "nan"
            else ""
        )
        ep = _fmt_card_num(_row_scalar(row, "entry_price"), 2)
        t1 = _fmt_card_num(_row_scalar(row, "target_1"), 2)
        rr = _fmt_card_num(_row_scalar(row, "rr_ratio"), 2)
        plan_p = f'<p class="data-card-plan">Entry {ep} → T1 {t1} (R:R {rr})</p>'
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=pat_p + plan_p,
                clickable=clickable,
                direction=str(row.get("direction", "")),
                filter_attrs=_filter_attrs_from_row(row),
            )
        )
    return out


def _candidates_card_fragments(
    df: pd.DataFrame,
    *,
    clickable: bool = True,
    is_near_miss: bool = False,
    near_threshold_long: float | None = None,
    near_threshold_short: float | None = None,
) -> list[str]:
    """Render validated candidate cards: 3 scores + pattern + status.

    When ``is_near_miss`` is True (plan §3), we swap the IV Rank metric for
    an explicit ``Δ to gate`` (distance to the promotion threshold) — that's
    the decision-critical number on Near Misses, IV rank rarely is.
    """
    if df.empty:
        return []
    view = _round_floats(df.copy())
    out: list[str] = []
    for _, row in view.iterrows():
        t = str(row.get("ticker", "")).strip()
        if not t:
            continue
        direc = _dir_badge_html(row.get("direction"))
        gamma = _gamma_badge_html(row.get("gamma_regime"))
        ct_badge = '<span class="badge badge-counter-trend">Counter-trend</span>' if _row_scalar(row, "counter_trend") else ""

        er_badge = ""
        days_er = _row_scalar(row, "days_until_earnings")
        if days_er is not None and str(days_er) not in ("nan", "None", ""):
            try:
                days_er_int = int(float(days_er))
                er_date = _row_scalar(row, "earnings_date") or ""
                if days_er_int <= 5:
                    er_badge = f'<span class="badge badge-earnings-imminent" title="Earnings on {html_escape(str(er_date))}">ER in {days_er_int}d</span>'
                elif days_er_int <= 10:
                    er_badge = f'<span class="badge badge-earnings-soon" title="Earnings on {html_escape(str(er_date))}">ER: {html_escape(str(er_date)[:10])}</span>'
            except (TypeError, ValueError):
                pass

        head = f'<span class="data-card-badges">{direc}{gamma}{ct_badge}{er_badge}</span>'
        pairs: list[tuple[str, str]] = [
            ("Flow", _conviction_span(_row_scalar(row, "flow_score_scaled"))),
            ("Price", _conviction_span(_row_scalar(row, "price_score"))),
            ("Options", _conviction_span(_row_scalar(row, "options_context_score"))),
            ("Final", _conviction_span(_row_scalar(row, "final_score"))),
        ]
        if is_near_miss:
            fs = _row_scalar(row, "final_score")
            dir_ = str(row.get("direction", "")).upper()
            threshold = near_threshold_long if dir_ == "LONG" else near_threshold_short
            if fs is not None and threshold is not None:
                try:
                    delta = float(fs) - float(threshold)
                    cls = "bad" if delta < 0 else "good"
                    pairs.append(
                        (
                            "Δ to gate",
                            f'<span class="{cls}">{delta:+.2f}</span>',
                        )
                    )
                except (TypeError, ValueError):
                    pairs.append(("Δ to gate", "—"))
            else:
                pairs.append(("Δ to gate", "—"))
        else:
            pairs.append(("IV Rank", _fmt_iv_rank(_row_scalar(row, "iv_rank"))))
        metrics = _card_metrics_dl(pairs)
        pat = _row_scalar(row, "pattern")
        pat_p = (
            f'<p class="data-card-thesis">{html_escape(str(pat))}</p>'
            if pat is not None and str(pat).strip() and str(pat) not in ("nan", "unknown")
            else ""
        )
        rr = _row_scalar(row, "reject_reason")
        if rr is not None and str(rr).strip() and str(rr) != "nan":
            reason_short = str(rr).split("(")[0].strip().replace("_", " ")
            status_p = f'<p class="data-card-meta"><span class="status-chip status-rejected">{html_escape(reason_short)}</span></p>'
        else:
            status_p = '<p class="data-card-meta"><span class="status-chip status-passed">PASSED</span></p>'
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=pat_p + status_p,
                clickable=clickable,
                direction=str(row.get("direction", "")),
                filter_attrs=_filter_attrs_from_row(row),
            )
        )
    return out


def _rejected_card_fragments(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    view = _round_floats(df.copy())
    out: list[str] = []
    for _, row in view.iterrows():
        t = str(row.get("ticker", "")).strip()
        if not t:
            continue
        direc = _dir_badge_html(row.get("direction"))
        head = f'<span class="data-card-badges">{direc}</span>'
        rr = _row_scalar(row, "reject_reason")
        reason_p = ""
        if rr is not None and str(rr).strip() and str(rr) != "nan":
            reason_short = str(rr).split("(")[0].strip().replace("_", " ")
            reason_p = f'<p class="data-card-meta"><span class="status-chip status-rejected">{html_escape(reason_short)}</span></p>'
        pairs: list[tuple[str, str]] = [
            ("Flow", _conviction_span(_row_scalar(row, "flow_score_scaled"))),
            ("Price", _conviction_span(_row_scalar(row, "price_score"))),
            ("Options", _conviction_span(_row_scalar(row, "options_context_score"))),
            ("Final", _conviction_span(_row_scalar(row, "final_score"))),
            ("IV Rank", _fmt_iv_rank(_row_scalar(row, "iv_rank"))),
        ]
        metrics = _card_metrics_dl(pairs)
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=reason_p,
                clickable=True,
                direction=str(row.get("direction", "")),
                filter_attrs=_filter_attrs_from_row(row),
            )
        )
    return out


def _flow_card_fragments(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    view = _scale_intensity_cols(df.copy())
    view = _round_floats(view)
    out: list[str] = []
    for _, row in view.iterrows():
        t = str(row.get("ticker", "")).strip()
        if not t:
            continue
        dom = _dir_badge_html(row.get("dominant_direction"))
        head = f'<span class="data-card-badges">{dom}</span>'
        pairs = [
            ("Premium", _fmt_money_short(_row_scalar(row, "total_premium"))),
            ("Flow prints (#)", _fmt_card_num(_row_scalar(row, "total_count"), 0)),
            ("Bull", _conviction_span(_row_scalar(row, "bullish_score"))),
            ("Bear", _conviction_span(_row_scalar(row, "bearish_score"))),
            ("Imbalance", _fmt_card_num(_row_scalar(row, "flow_imbalance_ratio"), 2)),
            ("DTE score", _conviction_span(_row_scalar(row, "dte_score"))),
        ]
        metrics = _card_metrics_dl(pairs)
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                clickable=False,
            )
        )
    return out


def _trades_card_fragments(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    view = _round_floats(df.copy())
    out: list[str] = []
    for _, row in view.iterrows():
        t = str(row.get("ticker", "")).strip()
        if not t:
            continue
        direc = _dir_badge_html(row.get("direction"))
        head = f'<span class="data-card-badges">{direc}</span>'
        pnl = _row_scalar(row, "pnl_dollar")
        pnl_s = "—"
        if pnl is not None:
            try:
                pnl_s = _wrap_color(_fmt_card_num(pnl, 2), float(pnl) > 0)
            except (TypeError, ValueError):
                pnl_s = html_escape(str(pnl))
        pairs = [
            ("Pattern", html_escape(str(_row_scalar(row, "pattern") or ""))),
            ("Entry → Exit", f"{_fmt_card_num(_row_scalar(row, 'entry_price'), 2)} → {_fmt_card_num(_row_scalar(row, 'exit_price'), 2)}"),
            ("P&amp;L", pnl_s),
            ("R", _fmt_card_num(_row_scalar(row, "r_multiple"), 2)),
            ("Days", _fmt_card_num(_row_scalar(row, "days_held"), 0)),
        ]
        metrics = _card_metrics_dl(pairs)
        er = _row_scalar(row, "exit_reason")
        ex = ""
        if er is not None and str(er).strip():
            ex = f'<p class="data-card-meta">Exit: {html_escape(str(er))}</p>'
        ed_in = _row_scalar(row, "entry_date")
        ed_out = _row_scalar(row, "exit_date")
        dates = ""
        if ed_in or ed_out:
            dates = f'<p class="data-card-plan">{html_escape(str(ed_in or "—"))} → {html_escape(str(ed_out or "—"))}</p>'
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=dates + ex,
                clickable=False,
            )
        )
    return out


def _alerts_card_fragments(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    out: list[str] = []
    for _, row in df.iterrows():
        t = str(row.get("ticker", "")).strip() or "—"
        direc = _dir_badge_html(row.get("direction"))
        ot = row.get("option_type")
        strike = _row_scalar(row, "strike")
        exp = _row_scalar(row, "expiration_date")
        spot = _row_scalar(row, "underlying_price")
        snip = "—"
        otm_badge = ""
        otm_suffix = ""
        if strike is not None and spot is not None:
            try:
                s, p = float(strike), float(spot)
                if p > 0:
                    otm_pct = abs(s - p) / p * 100
                    ot_str = str(ot or "").upper()
                    is_otm = (ot_str == "CALL" and s > p) or (ot_str == "PUT" and s < p)
                    if is_otm and otm_pct > 30:
                        otm_badge = ' <span class="badge badge-otm-deep">Deep OTM</span>'
                        otm_suffix = f" ({otm_pct:.0f}% OTM)"
                    elif is_otm and otm_pct > 15:
                        otm_badge = ' <span class="badge badge-otm-far">Far OTM</span>'
                        otm_suffix = f" ({otm_pct:.0f}% OTM)"
                    elif is_otm and otm_pct > 5:
                        otm_badge = ' <span class="badge badge-otm">OTM</span>'
                        otm_suffix = f" ({otm_pct:.0f}% OTM)"
            except (TypeError, ValueError):
                pass
        if strike is not None or exp is not None or ot is not None:
            snip = html_escape(
                f"{ot or ''} {strike or ''} @ {exp or ''}".strip()
            )
            if otm_suffix:
                snip += html_escape(otm_suffix)
        prem = _fmt_money_short(_row_scalar(row, "premium"))
        head = f'<span class="data-card-badges">{direc}{otm_badge}</span>'
        mcap_bps = _row_scalar(row, "prem_mcap_bps")
        label, label_cls = _prem_mcap_label(mcap_bps)
        if label == "—":
            mcap_html = "—"
        else:
            bps_num = f"{float(mcap_bps):.2f} bps" if mcap_bps is not None else ""
            cls_attr = f' class="{label_cls}"' if label_cls else ""
            mcap_html = f'<span{cls_attr}>{label}</span><br><small style="opacity:0.6">{bps_num}</small>'
        pairs = [
            ("Premium", prem),
            ("Prem/MCap", mcap_html),
            ("Strike / expiry", snip),
            ("DTE", _fmt_card_num(_row_scalar(row, "dte"), 0)),
        ]
        metrics = _card_metrics_dl(pairs)
        ev = row.get("event_ts")
        ev_p = (
            f'<p class="data-card-meta">{html_escape(str(ev))}</p>'
            if ev is not None and str(ev).strip()
            else ""
        )
        # Wave 5 — client-side sort/filter keys.  Strings because
        # ``_data_card_article`` will attribute-escape them and we
        # parse them back to numbers in JS.  Timestamp we coerce to
        # an ISO string for lexical sort parity with datetime ordering.
        try:
            _prem_f = float(_row_scalar(row, "premium") or 0) or 0.0
        except (TypeError, ValueError):
            _prem_f = 0.0
        try:
            _bps_f = float(mcap_bps) if mcap_bps is not None else float("nan")
        except (TypeError, ValueError):
            _bps_f = float("nan")
        try:
            _dte_f = float(_row_scalar(row, "dte") or 0) or 0.0
        except (TypeError, ValueError):
            _dte_f = 0.0
        _dir = str(row.get("direction") or "").upper().strip()
        _ot = str(ot or "").upper().strip()
        filter_attrs = {
            "uf-ticker": t,
            "uf-direction": _dir,
            "uf-option-type": _ot,
            "uf-premium": f"{_prem_f:.2f}",
            "uf-bps": "" if _bps_f != _bps_f else f"{_bps_f:.4f}",
            "uf-event-ts": str(ev or ""),
            "uf-dte": f"{_dte_f:.0f}",
        }
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=ev_p,
                clickable=False,
                filter_attrs=filter_attrs,
                extra_card_cls="uf-card",
            )
        )
    return out


def _watchlist_card_fragments(rows: list[dict]) -> list[str]:
    out: list[str] = []
    for r in rows:
        t = str(r.get("ticker", "")).strip()
        if not t:
            continue
        direc = _dir_badge_html(r.get("direction"))
        head = f'<span class="data-card-badges">{direc}</span>'
        pairs = [
            ("Flow (scaled)", _conviction_span(r.get("flow_score_scaled"))),
            ("IV Rank", _fmt_iv_rank(r.get("iv_rank"))),
            ("Seen", html_escape(str(r.get("first_seen", "")))),
        ]
        metrics = _card_metrics_dl(pairs)
        rr = r.get("reject_reason")
        ex = ""
        if rr:
            rr_full = str(rr)
            # Show only the top keyword (first comma- or pipe-separated token)
            # with full text in tooltip. Plan §3: "truncate to the top
            # rejection keyword + a full-text tooltip".
            primary = rr_full.split(",")[0].split("|")[0].split("(")[0].strip()
            primary = primary or rr_full[:40]
            ex += (
                f'<p class="data-card-thesis" title="{html_escape(rr_full)}">'
                f'{html_escape(primary)}</p>'
            )
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=ex,
                clickable=True,
                direction=str(r.get("direction", "")),
            )
        )
    return out


def _format_opened_ts(raw) -> str:
    """Parse an ISO timestamp and return a human-readable string."""
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(str(raw))
        return dt.strftime("%b %d, %I:%M %p UTC").lstrip("0")
    except (ValueError, TypeError):
        return html_escape(str(raw))


def _positions_card_fragments(rows: list[dict]) -> list[str]:
    out: list[str] = []
    for r in rows:
        t = str(r.get("ticker", "")).strip()
        if not t:
            continue
        direc = _dir_badge_html(r.get("direction"))
        hs = r.get("health_state")
        health_badge = ""
        if hs and str(hs).strip():
            cls = {"STRONG": "good", "NEUTRAL": "neutral", "WEAK": "warn", "FAILING": "bad"}.get(str(hs), "neutral")
            health_badge = f'<span class="health-badge health-{cls}">{html_escape(str(hs))}</span>'

        rm = r.get("r_at_market") or {}
        r_market_badge = ""
        if rm.get("r_available"):
            r_ent = rm.get("r_from_entry", 0.0)
            r_stp = rm.get("r_to_stop", 0.0)
            cls = "pos-pnl-pos" if r_ent >= 0 else "pos-pnl-neg"
            r_market_badge = (
                f'<span class="pos-r-market {cls}" title="R-multiple from entry · remaining R to stop">'
                f'{r_ent:+.2f}R · {r_stp:.2f}R to stop</span>'
            )

        heat_badge = ""
        heat_pct = r.get("heat_contribution_pct")
        if heat_pct is not None and heat_pct > 0:
            heat_badge = f'<span class="pos-heat" title="Risk as % of total open notional">{heat_pct:.1f}% heat</span>'

        head = f'<span class="data-card-badges">{direc}{health_badge}{r_market_badge}{heat_badge}</span>'
        pairs: list[tuple[str, str]] = [
            ("Health", _conviction_span(r.get("health"))),
            ("Unreal R", _fmt_card_num(r.get("unrealized_r"), 2)),
        ]
        metrics = _card_metrics_dl(pairs)

        ep_raw = r.get("entry_price")
        lp_raw = r.get("last_price")
        stop_raw = r.get("active_stop")
        t1_raw = r.get("target_1")
        t2_raw = r.get("target_2")
        direction = str(r.get("direction", "")).upper()

        ep = _fmt_card_num(ep_raw, 2)
        lp = _fmt_card_num(lp_raw, 2)
        stop = _fmt_card_num(stop_raw, 2)
        t1 = _fmt_card_num(t1_raw, 2)
        t2 = _fmt_card_num(t2_raw, 2)

        pnl_html = ""
        if ep_raw is not None and lp_raw is not None:
            try:
                ep_f, lp_f = float(ep_raw), float(lp_raw)
                if ep_f != 0:
                    pct = ((lp_f - ep_f) / ep_f) * 100
                    if direction == "SHORT":
                        pct = -pct
                    sign = "+" if pct >= 0 else ""
                    cls = "pos-pnl-pos" if pct >= 0 else "pos-pnl-neg"
                    pnl_html = f' <span class="{cls}">{sign}{pct:.1f}%</span>'
            except (TypeError, ValueError):
                pass

        ladder = (
            '<dl class="pos-price-ladder">'
            f"<dt>Entry</dt><dd>{ep}</dd>"
            f"<dt>Current</dt><dd>{lp}{pnl_html}</dd>"
            f"<dt>Stop</dt><dd>{stop}</dd>"
            f"<dt>T1</dt><dd>{t1}</dd>"
            f"<dt>T2</dt><dd>{t2}</dd>"
            "</dl>"
        )

        opened = r.get("opened_at") or r.get("entry_date")
        od = ""
        if opened:
            pretty = _format_opened_ts(opened)
            od = f'<p class="data-card-meta">Opened {pretty}</p>'
        proximity_cls = r.get("stop_proximity_cls") or ""
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=ladder + od,
                clickable=True,
                direction=str(r.get("direction", "")),
                extra_card_cls=proximity_cls,
            )
        )
    return out


def _table_with_pager(
    df: pd.DataFrame,
    *,
    page: int,
    per_page: int,
    page_param: str,
    badges: bool = False,
) -> str:
    """Render a data table for one page plus pager links (full data must stay in detail JSON)."""
    if df.empty:
        return _df_html(df, badges=badges)
    sliced, n, total_pages, page = _paginate_slice(df, page, per_page)
    return _df_html(sliced, badges=badges) + _pager_html(page_param, page, total_pages, n)


def _df_html(df: pd.DataFrame, *, max_rows: int | None = None, badges: bool = False) -> str:
    if df.empty:
        return '<p class="empty">No data. Run <code>python -m app.main --scan-only</code> to refresh outputs.</p>'
    view = df.head(max_rows) if max_rows else df
    view = _scale_intensity_cols(view)
    view = _round_floats(view)
    view = _colorize_values(view)
    if badges:
        view = _apply_badges(view)
    html = view.to_html(classes="data", index=False, border=0, escape=False, na_rep="—")
    return _inject_tooltips(html)


def _alerts_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "premium" in df.columns and "marketcap" in df.columns:
        mcap = pd.to_numeric(df["marketcap"], errors="coerce")
        prem = pd.to_numeric(df["premium"], errors="coerce")
        df["prem_mcap_bps"] = (prem / mcap * 10_000).round(2)
        df.loc[mcap < 1e8, "prem_mcap_bps"] = None

    cols = [c for c in ALERT_DISPLAY_COLS if c in df.columns]
    out = df[cols].copy() if cols else df.copy()
    if "event_ts" in out.columns:
        out["event_ts"] = pd.to_datetime(out["event_ts"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M UTC")
    if "expiration_date" in out.columns:
        out["expiration_date"] = pd.to_datetime(out["expiration_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sort_cols = [c for c in ["event_ts", "premium"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    return out


def _compute_performance(
    tl: pd.DataFrame | None = None,
    ec: pd.DataFrame | None = None,
) -> dict:
    """Build all performance analytics from the trade log and equity curve."""
    if tl is None:
        tl = load_trade_log()
    if ec is None:
        ec = load_equity_curve()
    perf: dict = {"has_data": not tl.empty, "equity_curve": []}

    if tl.empty:
        return perf

    wins = tl[tl["pnl_dollar"] > 0]
    losses = tl[tl["pnl_dollar"] <= 0]

    perf["total_trades"] = len(tl)
    perf["win_rate"] = round(len(wins) / len(tl) * 100, 1) if len(tl) else 0.0
    perf["avg_r"] = round(tl["r_multiple"].mean(), 2) if "r_multiple" in tl.columns else 0.0
    perf["total_pnl"] = round(tl["pnl_dollar"].sum(), 2) if "pnl_dollar" in tl.columns else 0.0
    gross_profit = wins["pnl_dollar"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["pnl_dollar"].sum()) if not losses.empty else 0.0
    perf["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")
    perf["avg_days_held"] = round(tl["days_held"].mean(), 1) if "days_held" in tl.columns else 0.0
    perf["best_trade"] = round(tl["pnl_dollar"].max(), 2)
    perf["worst_trade"] = round(tl["pnl_dollar"].min(), 2)

    def _group_stats(group: pd.DataFrame) -> dict:
        if group.empty:
            return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_pnl": 0.0}
        w = group[group["pnl_dollar"] > 0]
        return {
            "trades": len(group),
            "win_rate": round(len(w) / len(group) * 100, 1),
            "avg_r": round(group["r_multiple"].mean(), 2),
            "total_pnl": round(group["pnl_dollar"].sum(), 2),
        }

    if "pattern" in tl.columns:
        perf["by_pattern"] = {
            name: _group_stats(grp)
            for name, grp in tl.groupby("pattern")
        }
    else:
        perf["by_pattern"] = {}

    perf["by_direction"] = {
        name: _group_stats(grp)
        for name, grp in tl.groupby("direction")
    }

    if "final_score" in tl.columns:
        bins = [0, 7, 8, 9, 10.01]
        labels = ["<7", "7-8", "8-9", "9-10"]
        tl["score_tier"] = pd.cut(tl["final_score"], bins=bins, labels=labels, right=False)
        perf["by_score_tier"] = {
            str(name): _group_stats(grp)
            for name, grp in tl.groupby("score_tier", observed=True)
        }
    else:
        perf["by_score_tier"] = {}

    if "exit_reason" in tl.columns:
        perf["by_exit_reason"] = {
            name: _group_stats(grp)
            for name, grp in tl.groupby("exit_reason")
        }
    else:
        perf["by_exit_reason"] = {}

    if not ec.empty and "portfolio_value" in ec.columns:
        perf["equity_curve"] = ec.to_dict(orient="records")
    else:
        perf["equity_curve"] = []

    # Symbol breakdown
    if "ticker" in tl.columns:
        perf["by_symbol"] = {
            name: _group_stats(grp)
            for name, grp in tl.groupby("ticker")
        }
    else:
        perf["by_symbol"] = {}

    # Calendar data (daily P&L)
    if "exit_date" in tl.columns and "pnl_dollar" in tl.columns:
        cal = tl.copy()
        cal["exit_date"] = pd.to_datetime(cal["exit_date"], errors="coerce")
        cal = cal.dropna(subset=["exit_date"])
        cal["date_str"] = cal["exit_date"].dt.strftime("%Y-%m-%d")
        daily = cal.groupby("date_str")["pnl_dollar"].sum().reset_index()
        perf["calendar"] = daily.to_dict(orient="records")
    else:
        perf["calendar"] = []

    # Max drawdown
    if not ec.empty and "portfolio_value" in ec.columns:
        pv = ec["portfolio_value"].values
        peak = pd.Series(pv).cummax()
        dd = (pd.Series(pv) - peak) / peak * 100
        perf["max_drawdown"] = round(float(dd.min()), 2)
    else:
        perf["max_drawdown"] = 0.0

    return perf


def _build_overview(positions: list[dict], perf: dict, regime: dict) -> dict:
    """Build overview summary data for the landing page."""
    n_pos = len(positions)
    longs = sum(1 for p in positions if p.get("direction") == "LONG")
    shorts = n_pos - longs

    total_unreal_r = 0.0
    for p in positions:
        total_unreal_r += float(p.get("unrealized_r") or 0)

    return {
        "open_count": n_pos,
        "open_longs": longs,
        "open_shorts": shorts,
        "total_unrealized_r": round(total_unreal_r, 2),
        "total_trades": perf.get("total_trades", 0),
        "win_rate": perf.get("win_rate", 0.0),
        "avg_r": perf.get("avg_r", 0.0),
        "profit_factor": perf.get("profit_factor", 0.0),
        "total_pnl": perf.get("total_pnl", 0.0),
        "max_drawdown": perf.get("max_drawdown", 0.0),
    }


def _build_risk(positions: list[dict]) -> dict:
    """Build portfolio risk metrics from open positions."""
    if not positions:
        return {"has_data": False}

    long_exposure = 0.0
    short_exposure = 0.0
    rows = []

    for p in positions:
        entry = float(p.get("entry_price") or 0)
        shares = float(p.get("shares") or 0)
        current = float(p.get("last_price") or entry)
        direction = str(p.get("direction", "")).upper()
        notional = shares * entry

        if direction == "LONG":
            long_exposure += notional
            pnl_pct = ((current - entry) / entry * 100) if entry else 0
        else:
            short_exposure += notional
            pnl_pct = ((entry - current) / entry * 100) if entry else 0

        rows.append({
            "ticker": p.get("ticker"),
            "direction": direction,
            "entry": entry,
            "current": current,
            "shares": shares,
            "notional": round(notional, 2),
            "pnl_pct": round(pnl_pct, 1),
            "health_state": p.get("health_state", "—"),
            "days_held": p.get("days_held", 0),
            "active_stop": p.get("active_stop"),
        })

    gross = long_exposure + short_exposure
    net = long_exposure - short_exposure
    net_pct = (net / gross * 100) if gross else 0

    from collections import Counter
    health_dist = Counter(p.get("health_state", "NEUTRAL") for p in positions)

    return {
        "has_data": True,
        "long_exposure": round(long_exposure, 2),
        "short_exposure": round(short_exposure, 2),
        "gross_exposure": round(gross, 2),
        "net_exposure": round(net, 2),
        "net_pct": round(net_pct, 1),
        "position_count": len(positions),
        "rows": rows,
        "health_dist": dict(health_dist),
    }


# ---------------------------------------------------------------------------
# Trader dashboard helpers: action bar, actionable-now, flow-tracker enrichment
# ---------------------------------------------------------------------------

MAX_OPEN_POSITIONS = 6
# Default per-position risk assumption if caller doesn't supply one.
# Matches DEFAULT_RISK_PER_TRADE_PCT used downstream (~0.5-1% per trade).
DEFAULT_POSITION_RISK_PCT = 0.01


def _build_action_bar(
    positions: list[dict],
    regime: dict | None,
    signals_df: pd.DataFrame,
    watchlist: list[dict],
    grade_stats: dict | None,
) -> dict:
    """Top-of-page decision strip: regime, SPY, VIX, heat, open slots, new A-grades."""
    rs = (regime or {}).get("regime_score", 0.5)
    regime_cls = "ab-bull" if rs >= 0.65 else ("ab-bear" if rs <= 0.35 else "ab-neutral")
    regime_label = "Bull-leaning" if rs >= 0.65 else ("Bear-leaning" if rs <= 0.35 else "Neutral")

    spy_close = (regime or {}).get("spy_close")
    spy_prev = (regime or {}).get("spy_prev_close") or (regime or {}).get("spy_close_prev")
    spy_pct = None
    if spy_close and spy_prev:
        try:
            spy_pct = ((float(spy_close) - float(spy_prev)) / float(spy_prev)) * 100
        except (TypeError, ValueError, ZeroDivisionError):
            spy_pct = None
    if spy_pct is None:
        spy_pct = (regime or {}).get("spy_pct_today") or 0.0

    vix_close = (regime or {}).get("vix_close")
    vix_mult = (regime or {}).get("vix_sizing_mult")

    # Portfolio heat: sum of |entry - stop| * shares / equity.
    # Proxy equity from open-position notional when no explicit equity is known.
    total_notional = 0.0
    total_risk = 0.0
    for p in positions:
        try:
            ep = float(p.get("entry_price") or 0)
            stop = float(p.get("active_stop") or p.get("stop_price") or 0)
            sh = float(p.get("shares") or 0)
        except (TypeError, ValueError):
            continue
        if ep <= 0 or sh <= 0:
            continue
        total_notional += ep * sh
        if stop > 0:
            total_risk += abs(ep - stop) * sh

    heat_pct = (total_risk / total_notional * 100.0) if total_notional > 0 else 0.0
    # Rough sanity cap for display
    heat_pct = min(heat_pct, 99.9)
    heat_cls = "ab-neg" if heat_pct >= 4.0 else ("ab-warn" if heat_pct >= 2.5 else "ab-pos")

    open_count = len(positions)
    open_slots = max(0, MAX_OPEN_POSITIONS - open_count)

    # New A-grades = grade-A signals not in positions or watchlist.
    in_book = {str(p.get("ticker", "")).upper() for p in positions}
    in_book |= {str(w.get("ticker", "")).upper() for w in watchlist}
    new_a_grades = 0
    if not signals_df.empty and "final_score" in signals_df.columns:
        _fs = pd.to_numeric(signals_df["final_score"], errors="coerce")
        mask = _fs >= 7.5
        if mask.any():
            tickers = signals_df.loc[mask, "ticker"].astype(str).str.upper().tolist()
            new_a_grades = sum(1 for t in tickers if t not in in_book)

    # Z-score confidence breakdown: count how many A-grades rest on peer
    # baselines (Tier 3) or absolute fallback (Tier 4). Trader should size
    # these more cautiously.
    peer_a_count = 0
    total_a_count = 0
    if not signals_df.empty and "final_score" in signals_df.columns:
        _fs = pd.to_numeric(signals_df["final_score"], errors="coerce")
        a_mask = _fs >= 7.5
        total_a_count = int(a_mask.sum())
        if total_a_count and ("bullish_zscore_tier" in signals_df.columns or "bearish_zscore_tier" in signals_df.columns):
            a_rows = signals_df.loc[a_mask]
            for _, r in a_rows.iterrows():
                direction = str(r.get("direction", "")).upper()
                if direction == "LONG":
                    tier = r.get("bullish_zscore_tier")
                elif direction == "SHORT":
                    tier = r.get("bearish_zscore_tier")
                else:
                    continue
                try:
                    if int(tier) >= 3:
                        peer_a_count += 1
                except (TypeError, ValueError):
                    continue

    zscore_caveat_line = None
    if peer_a_count and total_a_count:
        zscore_caveat_line = f"{peer_a_count} of {total_a_count} A-grades use peer or absolute baseline"

    grade_stats_line = None
    if grade_stats and (grade_stats.get("stats") or {}).get("A", {}).get("count"):
        a = grade_stats["stats"]["A"]
        hr = int(round((a.get("hit_rate") or 0) * 100))
        grade_stats_line = f"{hr}% hit · {a.get('avg_r'):+.1f}R · n={a['count']}"

    # Wave 8 — compute the richer Risk Regime payload (VIX term, SPY RSI,
    # macro calendar, heat, concentration).  Falls back to the legacy VIX
    # multiplier when market indicators aren't available so the rest of
    # the Action Bar keeps rendering unchanged.
    risk_regime_payload = None
    try:
        from app.features.market_indicators import fetch_market_indicators
        from app.features.risk_regime import compute_risk_regime, summarise_for_ui

        mi = fetch_market_indicators()
        risk_regime_payload = compute_risk_regime(
            market_indicators=mi,
            positions=positions,
            heat_pct=heat_pct,
        )
        risk_regime_summary = summarise_for_ui(risk_regime_payload)
    except Exception:
        risk_regime_payload = None
        risk_regime_summary = None

    return {
        "regime_score": rs,
        "regime_cls": regime_cls,
        "regime_label": regime_label,
        "spy_pct_today": round(spy_pct, 2) if spy_pct is not None else 0.0,
        "spy_trend": (regime or {}).get("spy_trend"),
        "vix": float(vix_close) if vix_close else None,
        "vix_sizing_mult": float(vix_mult) if vix_mult else None,
        "heat_pct": round(heat_pct, 2),
        "heat_cls": heat_cls,
        "open_positions": open_count,
        "max_positions": MAX_OPEN_POSITIONS,
        "open_slots": open_slots,
        "new_a_grades": new_a_grades,
        "grade_stats_line": grade_stats_line,
        "zscore_caveat_line": zscore_caveat_line,
        # Wave 8 — full regime payload + condensed pill summary.
        "risk_regime": risk_regime_payload,
        "risk_regime_summary": risk_regime_summary,
        # Premium-Taxonomy plan — warm-up chip.  Only lit while the
        # tracker's history is rebuilding after a purge.
        "flow_tracker_warmup": _flow_tracker_warmup_state(),
    }


def _flow_tracker_warmup_state() -> dict | None:
    """Return ``{"active": bool, "day": int, "target": int}`` if the
    Flow Tracker is still warming up (fewer than ``FLOW_TRACKER_WARMUP_DAYS``
    distinct trading-day snapshots in ``screener_snapshots.csv``), else
    ``None``.

    Gated behind ``FLOW_TRACKER_WARMUP_BANNER_ENABLED`` so the chip can
    be switched off entirely once we're past the migration window.
    """
    try:
        from app.config import (
            FLOW_TRACKER_WARMUP_BANNER_ENABLED,
            FLOW_TRACKER_WARMUP_DAYS,
        )
        from app.features.flow_tracker import SNAPSHOTS_PATH
    except Exception:
        return None

    if not FLOW_TRACKER_WARMUP_BANNER_ENABLED:
        return None

    target = int(FLOW_TRACKER_WARMUP_DAYS)
    if target <= 0:
        return None

    try:
        if not SNAPSHOTS_PATH.exists():
            return {"active": True, "day": 0, "target": target}
        df = pd.read_csv(SNAPSHOTS_PATH, usecols=["date"])
    except Exception:
        return None

    if df is None or df.empty or "date" not in df.columns:
        return {"active": True, "day": 0, "target": target}

    distinct_days = df["date"].dropna().astype(str).nunique()
    if distinct_days >= target:
        return None
    return {"active": True, "day": int(distinct_days), "target": target}


def _actionable_now(
    signals_df: pd.DataFrame,
    positions: list[dict],
    watchlist: list[dict],
    limit: int = 5,
) -> list[dict]:
    """Top N final-score signals not already in book or watchlist.

    These are what a trader should consider taking _now_. Returns dicts that
    the `trader_card` macro can render directly.
    """
    if signals_df.empty:
        return []

    in_book = {str(p.get("ticker", "")).upper() for p in positions}
    in_book |= {str(w.get("ticker", "")).upper() for w in watchlist}

    df = signals_df.copy()
    if "final_score" in df.columns:
        df["_fs"] = pd.to_numeric(df["final_score"], errors="coerce").fillna(0)
        df = df.sort_values("_fs", ascending=False)

    out: list[dict] = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker or ticker in in_book:
            continue
        fs = float(row.get("final_score") or 0)
        if fs < 6.5:
            continue

        item = {k: row.get(k) for k in row.index if pd.notna(row.get(k))}
        out.append(item)
        if len(out) >= limit:
            break

    return out


def _enrich_flow_tracker_delta(
    rows: list[dict],
    flow_feature_df: pd.DataFrame,
) -> list[dict]:
    """Attach ``avg_delta`` + ``delta_source_mix`` from the latest flow-feature
    table to every Flow Tracker row.  Safe no-op when the feature table is
    empty or the ticker isn't present (e.g. tracker row survived gap days).
    """
    if not rows or flow_feature_df is None or flow_feature_df.empty:
        for r in rows:
            r.setdefault("avg_delta", None)
            r.setdefault("delta_source_mix", None)
        return rows

    need_cols = {
        "ticker",
        "bullish_avg_delta", "bearish_avg_delta",
        "bullish_delta_source_mix", "bearish_delta_source_mix",
    }
    have_cols = need_cols.intersection(flow_feature_df.columns)
    if "ticker" not in have_cols:
        for r in rows:
            r.setdefault("avg_delta", None)
            r.setdefault("delta_source_mix", None)
        return rows

    lookup: dict[str, dict] = {}
    for _, frow in flow_feature_df.iterrows():
        t = str(frow.get("ticker") or "").upper().strip()
        if not t:
            continue
        lookup[t] = {
            "bullish_avg_delta":         pd.to_numeric(frow.get("bullish_avg_delta"), errors="coerce"),
            "bearish_avg_delta":         pd.to_numeric(frow.get("bearish_avg_delta"), errors="coerce"),
            "bullish_delta_source_mix":  pd.to_numeric(frow.get("bullish_delta_source_mix"), errors="coerce"),
            "bearish_delta_source_mix":  pd.to_numeric(frow.get("bearish_delta_source_mix"), errors="coerce"),
        }

    for r in rows:
        ticker = str(r.get("ticker") or "").upper().strip()
        entry = lookup.get(ticker)
        if not entry:
            r["avg_delta"] = None
            r["delta_source_mix"] = None
            continue
        side = "bullish" if r.get("direction") == "BULLISH" else "bearish"
        ad = entry.get(f"{side}_avg_delta")
        sm = entry.get(f"{side}_delta_source_mix")
        r["avg_delta"] = float(ad) if ad is not None and not pd.isna(ad) else None
        r["delta_source_mix"] = float(sm) if sm is not None and not pd.isna(sm) else None
    return rows


def _build_flow_tracker_hero(rows: list[dict]) -> dict | None:
    """Compute the Wave 1 "Now What" 3-card hero strip.

    Returns a dict with three entries (``top``, ``sector``, ``setup``) or
    ``None`` if the tracker is empty.  Each card carries enough info for
    the UI to render a headline + 1-2 supporting lines and to link back to
    the detailed Flow Tracker card via anchor.

    The three picks are:
      * **top**    — highest ``conviction_score`` row (regardless of mode).
      * **sector** — the ``(sector, direction)`` pair with the most
                     accumulating rows.  Tie-break on cumulative premium.
      * **setup**  — most "ready to trade" row: passes the Strong gate,
                     DP-aligned, price action aligned with flow, IV rank
                     moderate (≤65, i.e. options not overpriced), no
                     earnings within 5 days.  Falls back to the top row
                     when no candidate satisfies all clauses.
    """
    if not rows:
        return None

    # ── Card 1: highest conviction.
    top = max(rows, key=lambda r: float(r.get("conviction_score", 0) or 0))

    # ── Card 2: hottest sector cluster.
    from collections import defaultdict
    sector_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    def _sector_key(v) -> str:
        # NaN/None-safe sector bucket so NaN floats don't create orphan groups.
        if v is None:
            return "—"
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return "—"
        except Exception:
            pass
        s = str(v).strip()
        return s if s and s.lower() != "nan" else "—"

    for r in rows:
        if r.get("passes_accumulation"):
            key = (_sector_key(r.get("sector")), str(r.get("direction") or ""))
            sector_groups[key].append(r)
    # Drop the "—" bucket; a cluster of unknown-sector rows isn't a signal.
    sector_groups.pop(("—", "BULLISH"), None)
    sector_groups.pop(("—", "BEARISH"), None)
    sector_card = None
    if sector_groups:
        best_key, best_rows = max(
            sector_groups.items(),
            key=lambda kv: (
                len(kv[1]),
                sum(float(x.get("cumulative_premium", 0) or 0) for x in kv[1]),
            ),
        )
        if len(best_rows) >= 2:  # Only meaningful when there's an actual cluster.
            sector_card = {
                "sector": best_key[0],
                "direction": best_key[1],
                "count": len(best_rows),
                "tickers": [x["ticker"] for x in sorted(
                    best_rows,
                    key=lambda x: float(x.get("conviction_score", 0) or 0),
                    reverse=True,
                )[:4]],
                "total_premium": sum(float(x.get("cumulative_premium", 0) or 0) for x in best_rows),
            }

    # ── Card 3: cleanest swing setup.
    def _ready_to_trade(r: dict) -> bool:
        if not r.get("passes_strong"):
            return False
        if not r.get("dp_aligned"):
            return False
        ret = r.get("window_return_pct")
        direction = str(r.get("direction") or "").upper()
        aligned = False
        if ret is not None:
            aligned = (direction == "BULLISH" and ret > 0) or (direction == "BEARISH" and ret < 0)
        if not aligned:
            return False
        iv_rank = r.get("latest_iv_rank")
        if iv_rank is not None:
            try:
                if float(iv_rank) > 65:
                    return False
            except (TypeError, ValueError):
                pass
        earnings = r.get("earnings") or {}
        dte = earnings.get("days_until_earnings")
        if dte is not None:
            try:
                if 0 <= float(dte) <= 5:
                    return False
            except (TypeError, ValueError):
                pass
        return True

    ready = [r for r in rows if _ready_to_trade(r)]
    setup_row = max(ready, key=lambda r: float(r.get("conviction_score", 0) or 0)) if ready else None
    setup_fallback = setup_row is None and top is not None

    # Assemble compact per-card payloads keyed by what the template needs.
    def _clean(v):
        # NaN-safe stringify so Jinja doesn't render literal "nan".
        if v is None:
            return None
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return None
        except Exception:
            pass
        s = str(v).strip()
        return None if (not s or s.lower() == "nan") else s

    def _compact(r: dict) -> dict:
        reasons = r.get("grade_reasons") or []
        top_reason = None
        for rr in reasons:
            if rr.get("kind") == "driver":
                top_reason = rr.get("label")
                break
        if top_reason is None and reasons:
            top_reason = reasons[0].get("label")
        return {
            "ticker": r.get("ticker"),
            "direction": r.get("direction"),
            "sector": _clean(r.get("sector")) or "—",
            "conviction_score": r.get("conviction_score"),
            "conviction_grade": r.get("conviction_grade"),
            "accumulation_score": r.get("accumulation_score"),
            "window_return_pct": r.get("window_return_pct"),
            "dp_aligned": bool(r.get("dp_aligned")),
            "dp_bias": (r.get("dp") or {}).get("bias") if isinstance(r.get("dp"), dict) else None,
            "iv_rank": r.get("latest_iv_rank"),
            "top_reason": _clean(top_reason),
            "earnings_in": ((r.get("earnings") or {}).get("days_until_earnings")),
        }

    return {
        "top": _compact(top) if top else None,
        "sector": sector_card,
        "setup": _compact(setup_row) if setup_row else (_compact(top) if setup_fallback and top else None),
        "setup_is_fallback": bool(setup_fallback),
    }


def _build_flow_tracker(
    *,
    lookback_days: int,
    min_active_days: int,
    flow: pd.DataFrame,
    hottest_chains_by_ticker: dict[str, dict],
    insider_by_ticker: dict[str, dict],
    risk_regime: dict | None = None,
) -> list[dict]:
    """Compute + enrich a Flow Tracker list for a given horizon.

    Wraps the multi-stage enrichment chain (sentiment → DP → chains →
    insider → earnings → delta → z-tier → decision) so the server can
    produce separate datasets for each horizon in ``FLOW_TRACKER_HORIZONS``
    without code duplication.  Wave 0.5 C1 / C2.

    Wave 8 — ``risk_regime`` is threaded through to the decision enricher
    so every row's ``trade_structure`` payload carries the current sizing
    context and HALT caveats.
    """
    base = compute_multi_day_flow(lookback_days, min_active_days)
    if not base:
        return base
    dp_tracker = compute_multi_day_dp(DP_TRACKER_LOOKBACK_DAYS, DP_TRACKER_MIN_ACTIVE_DAYS)
    rows = _enrich_flow_tracker_sentiment(base)
    rows = _enrich_flow_tracker_dp(rows, dp_tracker)
    rows = _enrich_flow_tracker_chains(rows, hottest_chains_by_ticker)
    rows = _enrich_flow_tracker_insider(rows, insider_by_ticker)
    rows = _enrich_flow_tracker_earnings(rows)
    rows = _enrich_flow_tracker_delta(rows, flow)
    rows = _enrich_flow_tracker_ztier(rows)
    rows = _enrich_flow_tracker_decision(rows, risk_regime=risk_regime)

    # Flow-Tracker-Swing-Radar: drop illiquid tickers from the radar so
    # the list stays swing-tradable.  ILLIQUID tier is ADV < $1M — any
    # fill there is essentially market-making the name.  Done after the
    # decision enrichment so liquidity is computed exactly once.
    from app.config import FLOW_TRACKER_HARD_ILLIQUID_FILTER
    if FLOW_TRACKER_HARD_ILLIQUID_FILTER:
        dropped = 0
        kept: list[dict] = []
        for r in rows:
            tier = str((r.get("liquidity") or {}).get("liquidity_tier") or "").upper()
            if tier == "ILLIQUID":
                dropped += 1
                continue
            kept.append(r)
        if dropped:
            print(f"  [flow-tracker] swing-radar: dropped {dropped} illiquid names")
        rows = kept
    return rows


def _enrich_flow_tracker_decision(
    rows: list[dict],
    *,
    risk_regime: dict | None = None,
) -> list[dict]:
    """Attach liquidity tier + trade-expression hint to every Flow Tracker row.

    Flow Tracker rows lack entry/stop plans so we can't produce full trader
    cards, but the liquidity + expression badges are still decision-critical.

    Wave 4 also attaches the composite ``conviction_stack`` (0-100) so the
    card chip replaces the legacy F/D/C/I dot row.
    """
    from app.features.conviction_stack import compute_conviction_stack

    for r in rows:
        ticker = r.get("ticker")
        if not ticker:
            continue
        try:
            r["liquidity"] = compute_liquidity(ticker, mcap=r.get("marketcap"))
        except Exception:
            r["liquidity"] = {"adv_dollar": None, "liquidity_tier": "UNKNOWN"}

        direction = "LONG" if r.get("direction") == "BULLISH" else "SHORT"
        earn = (r.get("earnings") or {})
        dte = earn.get("days_until_earnings")
        r["trade_expression"] = suggest_expression(
            direction,
            r.get("latest_iv_rank"),
            dte,
            adv_dollar=r["liquidity"].get("adv_dollar"),
        )

        try:
            r["conviction_stack"] = compute_conviction_stack(r)
        except Exception:
            r["conviction_stack"] = None

        # Wave 6 — attach natural-language narrative bullets so the Flow
        # Tracker modal's "Why?" tab can render without another trip
        # through the data.  Kept server-side so tests can assert on the
        # payload shape.
        try:
            from app.features.flow_narrative import build_flow_tracker_narrative
            r["narrative"] = build_flow_tracker_narrative(r)
        except Exception:
            r["narrative"] = []

        # Wave 7 — attach structure recommendation so the Flow Tracker
        # "Structure" tab can render a ranked trade-expression ladder
        # (primary + alternatives + avoid + caveats) without another
        # round trip through the pipeline.
        try:
            from app.features.trade_structure import recommend_structure
            r["trade_structure"] = recommend_structure(r)
        except Exception:
            r["trade_structure"] = None

        # Wave 8 — decorate the structure payload with the current risk
        # regime's sizing context so the Flow Tracker's Structure tab can
        # render the regime checks and any HALT caveat inline.
        if risk_regime is not None:
            try:
                from app.web.view_models import attach_sizing_context
                r["trade_structure"] = attach_sizing_context(
                    r.get("trade_structure"), risk_regime
                )
            except Exception:
                pass

        # Premium-Taxonomy plan — normalize the mix payload so the
        # Trader-Card modal can render the Premium Mix panel directly.
        try:
            from app.web.view_models import _build_premium_mix_ui
            r["premium_mix_ui"] = _build_premium_mix_ui(r)
        except Exception:
            r["premium_mix_ui"] = None
    return rows


def _enrich_positions_live(positions: list[dict], total_notional_hint: float | None = None) -> list[dict]:
    """Add R-at-market and heat-contribution to each open position.

    Called pre-render so templates can use `p.r_at_market`, `p.heat_contribution_pct`,
    and `p.stop_proximity_cls` without recomputing.
    """
    if not positions:
        return positions

    total_notional = total_notional_hint or 0.0
    if total_notional <= 0:
        for p in positions:
            try:
                total_notional += float(p.get("entry_price") or 0) * float(p.get("shares") or 0)
            except (TypeError, ValueError):
                continue
    if total_notional <= 0:
        total_notional = 1.0  # avoid div0; heat_pct will just be unusable

    for p in positions:
        direction = p.get("direction", "")
        entry = p.get("entry_price")
        stop = p.get("active_stop") or p.get("stop_price")
        spot = p.get("last_price") or entry
        p["r_at_market"] = compute_r_at_market(direction, entry, stop, spot)

        try:
            ep = float(entry or 0)
            sp = float(stop or 0)
            sh = float(p.get("shares") or 0)
            if ep > 0 and sp > 0 and sh > 0:
                risk_dollar = abs(ep - sp) * sh
                p["heat_contribution_pct"] = round(risk_dollar / total_notional * 100, 2)
            else:
                p["heat_contribution_pct"] = 0.0
        except (TypeError, ValueError):
            p["heat_contribution_pct"] = 0.0

        # Stop proximity classification for card border tinting.
        rm = p.get("r_at_market", {})
        r_to_stop = rm.get("r_to_stop")
        if r_to_stop is None:
            p["stop_proximity_cls"] = "stop-ok"
        elif r_to_stop < 0.25:
            p["stop_proximity_cls"] = "stop-danger"
        elif r_to_stop < 0.75:
            p["stop_proximity_cls"] = "stop-warn"
        else:
            p["stop_proximity_cls"] = "stop-ok"

    return positions


def _recent_activity(positions: list[dict], trades_df: pd.DataFrame, limit: int = 8) -> list[dict]:
    """Build a recent activity feed from positions and trade log."""
    events: list[dict] = []

    for p in positions:
        events.append({
            "type": "entry",
            "ticker": p.get("ticker"),
            "direction": p.get("direction"),
            "date": p.get("opened_at") or p.get("entry_date", ""),
            "detail": f"Entered at {_fmt_card_num(p.get('entry_price'), 2)} — {p.get('pattern', 'unknown')}",
        })

    if not trades_df.empty:
        for _, row in trades_df.tail(20).iterrows():
            pnl = row.get("pnl_dollar")
            pnl_str = f"${float(pnl):+,.0f}" if pd.notna(pnl) else ""
            events.append({
                "type": "exit",
                "ticker": row.get("ticker"),
                "direction": row.get("direction"),
                "date": str(row.get("exit_date", "")),
                "detail": f"Closed {pnl_str} — {row.get('exit_reason', '')}",
            })

    events.sort(key=lambda e: e.get("date", ""), reverse=True)
    return events[:limit]


# ---------------------------------------------------------------------------
# Tab insight summaries — natural language panels above card grids
# ---------------------------------------------------------------------------

def _line(dot: str, text: str) -> str:
    return f'<p class="insight-line"><span class="dot dot-{dot}"></span>{text}</p>'


def _signals_insights(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="tab-insights">' + _line("neutral", "No signals this scan.") + "</div>"

    n = len(df)
    longs = int((df["direction"] == "LONG").sum()) if "direction" in df.columns else 0
    shorts = n - longs

    lines: list[str] = []

    # Regime context
    rs_val = None
    if "regime_score" in df.columns:
        rs_series = pd.to_numeric(df["regime_score"], errors="coerce").dropna()
        if not rs_series.empty:
            rs_val = float(rs_series.iloc[0])
    regime_note = ""
    if rs_val is not None:
        if rs_val >= 0.65:
            regime_note = f" Regime {rs_val:.2f}, favoring longs."
        elif rs_val <= 0.35:
            regime_note = f" Regime {rs_val:.2f}, favoring shorts."
        else:
            regime_note = f" Regime {rs_val:.2f}, neutral."
    lines.append(_line("neutral", f"{n} signal{'s' if n != 1 else ''} — {longs} long, {shorts} short.{regime_note}"))

    # Best setup
    if "final_score" in df.columns:
        best_idx = pd.to_numeric(df["final_score"], errors="coerce").idxmax()
        best = df.loc[best_idx]
        ticker = best.get("ticker", "?")
        direction = best.get("direction", "?")
        score = _fmt_card_num(best.get("final_score"), 1)
        pattern = best.get("pattern", "")
        rr = _fmt_card_num(best.get("rr_ratio"), 1)
        pat_str = f" — {html_escape(str(pattern))}" if pattern and str(pattern) not in ("", "nan", "unknown") else ""
        rr_str = f", {rr}:1 RR" if rr != "—" else ""
        lines.append(_line("good", f"Strongest: {html_escape(str(ticker))} ({direction}, {score}){pat_str}{rr_str}"))

    # Common strengths
    strengths: list[str] = []
    if "flow_score_scaled" in df.columns:
        avg_flow = pd.to_numeric(df["flow_score_scaled"], errors="coerce").mean()
        if avg_flow >= 6.0:
            strengths.append("strong flow conviction")
    if "gamma_regime" in df.columns:
        pos_gamma = (df["gamma_regime"] == "POSITIVE").sum()
        if pos_gamma > n / 2:
            strengths.append("positive gamma")
    if "agg_premium_alignment" in df.columns:
        avg_align = pd.to_numeric(df["agg_premium_alignment"], errors="coerce").mean()
        if avg_align >= 0.6:
            strengths.append("premium-aligned")
    if "rr_ratio" in df.columns:
        avg_rr = pd.to_numeric(df["rr_ratio"], errors="coerce").mean()
        if avg_rr >= 2.5:
            strengths.append(f"good avg RR ({avg_rr:.1f}:1)")
    if strengths:
        lines.append(_line("good", "Strengths: " + ", ".join(strengths)))

    # Gaps / weaknesses from checks_failed
    if "checks_failed" in df.columns:
        from collections import Counter
        all_fails: list[str] = []
        for val in df["checks_failed"].dropna():
            parts = [p.strip() for p in str(val).split(",") if p.strip() and p.strip() != "none"]
            all_fails.extend(parts)
        if all_fails:
            counts = Counter(all_fails).most_common(3)
            gap_parts = [f"{count}/{n} lack {name}" for name, count in counts if count > 0]
            if gap_parts:
                lines.append(_line("warn", "Gaps: " + ", ".join(gap_parts)))

    return '<div class="tab-insights">' + "".join(lines) + "</div>"


def _candidates_insights(bull: pd.DataFrame, bear: pd.DataFrame, rejected: pd.DataFrame) -> str:
    n_bull = len(bull) if not bull.empty else 0
    n_bear = len(bear) if not bear.empty else 0

    if n_bull == 0 and n_bear == 0:
        return '<div class="tab-insights">' + _line("neutral", "No validated candidates this scan.") + "</div>"

    lines: list[str] = []

    # Count passed vs rejected
    all_cands = pd.concat([bull, bear], ignore_index=True) if n_bull + n_bear > 0 else pd.DataFrame()
    n_passed = 0
    n_rejected = 0
    if not all_cands.empty and "reject_reason" in all_cands.columns:
        n_passed = int(all_cands["reject_reason"].isna().sum() + (all_cands["reject_reason"] == "").sum())
        n_rejected = len(all_cands) - n_passed

    lines.append(_line("neutral", f"{n_bull} bullish, {n_bear} bearish validated. {n_passed} passed, {n_rejected} rejected."))

    # Near-misses: top rejected by flow score
    if not rejected.empty and "flow_score_scaled" in rejected.columns:
        rej_sorted = rejected.copy()
        rej_sorted["_fs"] = pd.to_numeric(rej_sorted["flow_score_scaled"], errors="coerce")
        rej_sorted = rej_sorted.dropna(subset=["_fs"]).sort_values("_fs", ascending=False).head(3)
        if not rej_sorted.empty:
            misses: list[str] = []
            for _, row in rej_sorted.iterrows():
                t = html_escape(str(row.get("ticker", "?")))
                fs = f"{float(row['_fs']):.1f}"
                reason = str(row.get("reject_reason", "")).split("(")[0].strip()
                misses.append(f"{t} (flow {fs}, {reason})")
            lines.append(_line("warn", "Near-misses: " + ", ".join(misses)))

    # Rejection breakdown
    if not rejected.empty and "reject_reason" in rejected.columns:
        from collections import Counter
        reasons = [str(r).split("(")[0].strip().replace("_", " ") for r in rejected["reject_reason"].dropna()]
        if reasons:
            counts = Counter(reasons).most_common(3)
            parts = [f"{name} ({count})" for name, count in counts]
            lines.append(_line("neutral", "Top rejection reasons: " + ", ".join(parts)))

    return '<div class="tab-insights">' + "".join(lines) + "</div>"


def _positions_insights(positions: list[dict]) -> str:
    n = len(positions)
    if n == 0:
        return '<div class="tab-insights">' + _line("neutral", "No open positions.") + "</div>"

    longs = sum(1 for p in positions if p.get("direction") == "LONG")
    shorts = n - longs

    lines: list[str] = []
    lines.append(_line("neutral", f"{n} open — {longs} long, {shorts} short."))

    # Health distribution
    from collections import Counter
    state_counts = Counter(p.get("health_state", "NEUTRAL") for p in positions)
    state_order = ["STRONG", "NEUTRAL", "WEAK", "FAILING"]
    state_dots = {"STRONG": "good", "NEUTRAL": "neutral", "WEAK": "warn", "FAILING": "bad"}
    health_parts: list[str] = []
    for s in state_order:
        c = state_counts.get(s, 0)
        if c > 0:
            health_parts.append(f"{c} {s}")
    if health_parts:
        lines.append(_line(state_dots.get(state_order[0] if state_counts.get("STRONG", 0) > 0 else "neutral", "neutral"),
                          "Health: " + ", ".join(health_parts)))

    # Action items: deteriorating, approaching time stop, FAILING
    actions: list[str] = []
    for p in positions:
        ticker = p.get("ticker", "?")
        state = p.get("health_state")
        delta = p.get("health_delta", 0.0)
        health = p.get("health")
        health_prev = p.get("health_prev")

        if state == "FAILING":
            actions.append(f"{ticker} FAILING (health {health:.0f})" if health is not None else f"{ticker} FAILING")
        elif delta is not None and delta < -2.0 and health is not None and health_prev is not None:
            actions.append(f"{ticker} deteriorating ({health_prev:.0f} → {health:.0f}, delta {delta:+.1f})")

        days = p.get("days_held", 0)
        time_stop = p.get("time_stop_days")
        if time_stop and days >= time_stop - 1:
            actions.append(f"{ticker} approaching time stop (day {days}/{time_stop})")

    for a in actions[:3]:
        lines.append(_line("bad", a))

    # Best runner
    best_r = -999.0
    best_pos = None
    for p in positions:
        entry = p.get("entry_price", 0)
        risk = p.get("risk_per_share", 0)
        best_price = p.get("best_price", entry)
        if risk > 0 and entry > 0:
            if p.get("direction") == "LONG":
                ur = (best_price - entry) / risk
            else:
                ur = (entry - best_price) / risk
            if ur > best_r:
                best_r = ur
                best_pos = p

    if best_pos and best_r > 0:
        t = best_pos.get("ticker", "?")
        state = best_pos.get("health_state", "?")
        lines.append(_line("good", f"Best runner: {t} at {best_r:+.1f}R ({state})"))

    return '<div class="tab-insights">' + "".join(lines) + "</div>"


def _rejected_insights(rejected: pd.DataFrame, watchlist: list[dict]) -> str:
    n_rej = len(rejected) if not rejected.empty else 0
    n_wl = len(watchlist)

    if n_rej == 0 and n_wl == 0:
        return '<div class="tab-insights">' + _line("neutral", "No rejected candidates or watchlist entries.") + "</div>"

    lines: list[str] = []

    # Rejection summary
    if n_rej > 0 and "reject_reason" in rejected.columns:
        from collections import Counter
        reasons = [str(r).split("(")[0].strip().replace("_", " ") for r in rejected["reject_reason"].dropna()]
        counts = Counter(reasons).most_common(3)
        parts = [f"{name} ({count})" for name, count in counts]
        lines.append(_line("neutral", f"{n_rej} rejected — top reasons: " + ", ".join(parts)))
    elif n_rej > 0:
        lines.append(_line("neutral", f"{n_rej} rejected this scan."))

    # Watchlist highlights
    if n_wl > 0:
        best_wl = max(watchlist, key=lambda w: float(w.get("flow_score_scaled", 0) or 0))
        best_ticker = best_wl.get("ticker", "?")
        best_flow = float(best_wl.get("flow_score_scaled", 0) or 0)
        lines.append(_line("warn", f"{n_wl} on watchlist. Closest to promotion: {html_escape(str(best_ticker))} (flow {best_flow:.1f})"))

        # Recurring names (3+ days on watchlist)
        from datetime import date
        today = date.today()
        persistent: list[str] = []
        for w in watchlist:
            first_seen = w.get("first_seen")
            if first_seen:
                try:
                    seen_date = date.fromisoformat(str(first_seen))
                    days_on = (today - seen_date).days
                    if days_on >= 3:
                        persistent.append(f"{w.get('ticker', '?')} ({days_on}d)")
                except (ValueError, TypeError):
                    pass
        if persistent:
            lines.append(_line("warn", "Persistent flow: " + ", ".join(persistent[:5])))

    return '<div class="tab-insights">' + "".join(lines) + "</div>"


def _build_dark_pool_screener(flow: pd.DataFrame) -> dict:
    """Load persisted dark pool prints and aggregate for the dashboard."""
    raw = load_dark_pool_recent()
    if not raw:
        return {"top_prints": [], "by_ticker": [], "by_mcap": []}

    smeta: dict[str, dict] = {}
    if not flow.empty and "ticker" in flow.columns:
        mcap_col = "marketcap" if "marketcap" in flow.columns else None
        sector_col = "sector" if "sector" in flow.columns else None
        for _, r in flow.iterrows():
            t = str(r.get("ticker", ""))
            entry: dict = {}
            if mcap_col:
                try:
                    entry["marketcap"] = float(r[mcap_col])
                except (TypeError, ValueError):
                    pass
            if sector_col:
                entry["sector"] = r[sector_col]
            if t and entry:
                smeta[t] = entry

    return aggregate_dark_pool_prints(raw, screener_meta=smeta)


def _build_daily_accumulated_dp(flow: pd.DataFrame) -> dict:
    """Build the daily accumulated dark pool view from today's deduped prints."""
    prints = load_daily_accumulated()
    if not prints:
        return {"by_ticker": [], "total_prints": 0, "total_notional": 0, "scan_count": 0}

    smeta: dict[str, dict] = {}
    if not flow.empty and "ticker" in flow.columns:
        mcap_col = "marketcap" if "marketcap" in flow.columns else None
        sector_col = "sector" if "sector" in flow.columns else None
        for _, r in flow.iterrows():
            t = str(r.get("ticker", ""))
            entry: dict = {}
            if mcap_col:
                try:
                    entry["marketcap"] = float(r[mcap_col])
                except (TypeError, ValueError):
                    pass
            if sector_col:
                entry["sector"] = r[sector_col]
            if t and entry:
                smeta[t] = entry

    return aggregate_daily_accumulated(prints, screener_meta=smeta)


@app.route("/")
def index():
    signals, signals_src = load_final_signals()
    rejected_all, rejected_src = load_rejected()
    flow, flow_src = load_flow_features()
    regime = load_market_regime()

    # Wave 0.5 C1 — selected horizon (?horizon=5d|15d).  Invalid keys fall
    # back to the default; the raw param (if any) is kept so the template
    # can pre-select the toggle button without re-parsing.
    _horizon_param = (request.args.get("horizon") or "").strip().lower()
    if _horizon_param in FLOW_TRACKER_HORIZONS:
        active_horizon = _horizon_param
    else:
        active_horizon = FLOW_TRACKER_HORIZON_DEFAULT

    # Filter out trivial rejections that never reached full price scoring.
    _TRIVIAL_PREFIXES = ("weak_bullish_flow", "weak_bearish_flow",
                         "trend_not_aligned", "price_over_extended", "error:",
                         "watchlist_reeval_failed")

    def _is_trivial(reason: str) -> bool:
        return any(str(reason).startswith(p) for p in _TRIVIAL_PREFIXES)

    if not rejected_all.empty and "reject_reason" in rejected_all.columns:
        _trivial_mask = rejected_all["reject_reason"].apply(_is_trivial)
        trivial_rejected_count = int(_trivial_mask.sum())
        rejected = rejected_all[~_trivial_mask].copy()
    else:
        trivial_rejected_count = 0
        rejected = rejected_all.copy()

    # Build validated candidates: signals + rejected merged, split by direction.
    # Enrich with flow features so rejected rows carry the underlying flow
    # metrics for the detail modal (bullish_score, total_premium, avg_dte, etc.).
    _parts = [df for df in (signals, rejected) if not df.empty]
    _validated = pd.concat(_parts, ignore_index=True) if _parts else pd.DataFrame()
    if not _validated.empty and not flow.empty and "ticker" in _validated.columns and "ticker" in flow.columns:
        _flow_cols = [c for c in flow.columns if c not in _validated.columns or c == "ticker"]
        _validated = _validated.merge(flow[_flow_cols], on="ticker", how="left")
    sort_bull = request.args.get("sort_bull", "")
    sort_bear = request.args.get("sort_bear", "")
    sort_rej = request.args.get("sort_rej", "")

    if not _validated.empty and "direction" in _validated.columns:
        bull = _validated[_validated["direction"] == "LONG"].copy()
        bear = _validated[_validated["direction"] == "SHORT"].copy()
        bull = _sort_df(bull, sort_bull) if sort_bull else _sort_df(bull, "flow_score_scaled")
        bear = _sort_df(bear, sort_bear) if sort_bear else _sort_df(bear, "flow_score_scaled")
    else:
        bull = pd.DataFrame()
        bear = pd.DataFrame()
    # Compute near-threshold candidates: those with final_score within 2 pts of threshold
    near_df = pd.DataFrame()
    rs = regime.get("regime_score", 0.5) if regime else 0.5
    _near_long_min = MIN_FINAL_SCORE + (1.0 - rs) * REGIME_THRESHOLD_BOOST
    _near_short_min = MIN_FINAL_SCORE + rs * REGIME_THRESHOLD_BOOST
    if not _validated.empty and "final_score" in _validated.columns:
        _validated["_fs_num"] = pd.to_numeric(_validated["final_score"], errors="coerce")
        _near_gap = 2.0
        mask = _validated["_fs_num"].notna() & (_validated["_fs_num"] > 0)
        if mask.any():
            long_mask = mask & (_validated["direction"] == "LONG") & (_validated["_fs_num"] >= _near_long_min - _near_gap)
            short_mask = mask & (_validated["direction"] == "SHORT") & (_validated["_fs_num"] >= _near_short_min - _near_gap)
            near_df = _validated[long_mask | short_mask].copy()
            if not near_df.empty:
                near_df = near_df.sort_values("_fs_num", ascending=False)
        _validated.drop(columns=["_fs_num"], inplace=True)
        if not near_df.empty and "_fs_num" in near_df.columns:
            near_df.drop(columns=["_fs_num"], inplace=True)

    page_near = _int_query("page_near")

    bull_src = rejected_src or signals_src
    bear_src = bull_src
    positions = load_positions()
    watchlist = load_watchlist()
    watchlist = _enrich_watchlist_from_rejected(watchlist, rejected_all)
    if not flow.empty and "ticker" in flow.columns:
        flow_lookup = flow.set_index("ticker").to_dict("index")
        for w in watchlist:
            t = w.get("ticker")
            if t and t in flow_lookup:
                for k, v in flow_lookup[t].items():
                    if k not in w or w[k] is None:
                        w[k] = v
    trades = load_trade_log_tail(80)

    # Auto-load unusual flow (top 30 by flow_intensity)
    from app.vendors.unusual_whales import fetch_recent_alert_flow

    alerts_raw = pd.DataFrame()
    alerts_error = ""
    # Wave 5 — allow the initial server render to honour a deep-linked
    # ``?opening_only=1`` so bookmarked / shared views stay consistent
    # with the client-side toggle.
    _oo_raw = request.args.get("opening_only")
    alerts_opening_only = (
        None if _oo_raw is None else str(_oo_raw).strip().lower() in {"1", "true", "yes", "on"}
    )
    try:
        alerts_raw = fetch_recent_alert_flow(
            limit=150,
            hours_back=24,
            opening_only=alerts_opening_only,
        )
    except Exception as e:
        alerts_error = str(e)
    alerts = _alerts_subset(alerts_raw)
    if not alerts.empty and "prem_mcap_bps" in alerts.columns:
        alerts = alerts.sort_values("prem_mcap_bps", ascending=False, na_position="last").head(30)
    elif not alerts.empty:
        alerts = alerts.head(30)

    page_alt = _int_query("page_alt")
    if alerts_error:
        alerts_html = f'<p class="error">Unusual flow unavailable: {html_escape(alerts_error)}</p>'
    else:
        alerts_html = _paginate_card_fragments(
            _alerts_card_fragments(alerts),
            page=page_alt,
            per_page=TABLE_PAGE_SIZE,
            page_param="page_alt",
            empty_msg="No unusual flow in last 24h.",
        )

    # Build top flow intensity summary from pipeline flow features (aggregated by ticker)
    # Wave 5 — augment each row with:
    #   • conviction_stack chip (from `app.features.conviction_stack`)
    #   • dealer_hedge_bias / pin_risk_distance_pct (from signals enrichment)
    #   • days_until_earnings as a coarse "catalyst" proxy (Wave-3 news chip
    #     was cancelled; earnings is the cheap, always-available signal).
    # The signals DataFrame is already loaded a few hundred lines up and
    # carries dealer + pin + earnings columns via pipeline.save_screener.
    _tf_ctx_lookup: dict[str, dict] = {}
    if "signals" in dir() and hasattr(signals, "to_dict"):
        try:
            _sig_rows = signals.to_dict(orient="records") if not signals.empty else []
        except Exception:
            _sig_rows = []
        for _sr in _sig_rows:
            _t = str(_sr.get("ticker") or "").strip().upper()
            if not _t or _t in _tf_ctx_lookup:
                continue
            _tf_ctx_lookup[_t] = {
                "dealer_hedge_bias": _sr.get("dealer_hedge_bias"),
                "dealer_hedge_label": _sr.get("dealer_hedge_label"),
                "pin_risk_strike": _sr.get("pin_risk_strike"),
                "pin_risk_distance_pct": _sr.get("pin_risk_distance_pct"),
                "pin_risk_concentration": _sr.get("pin_risk_concentration"),
                "days_until_earnings": _sr.get("days_until_earnings"),
            }

    top_flow: list[dict] = []
    if not flow.empty:
        _tf = flow.copy()
        for col in ("bullish_flow_intensity", "bearish_flow_intensity", "marketcap"):
            if col in _tf.columns:
                _tf[col] = pd.to_numeric(_tf[col], errors="coerce")
        if "marketcap" in _tf.columns:
            _tf = _tf[_tf["marketcap"] >= 1e8]
        if "bullish_flow_intensity" in _tf.columns and "bearish_flow_intensity" in _tf.columns:
            _tf["_peak"] = _tf[["bullish_flow_intensity", "bearish_flow_intensity"]].max(axis=1)
            _tf = _tf[_tf["_peak"] > 0].sort_values("_peak", ascending=False).head(15)
            from app.features.grade_explainer import build_flow_grade_reasons, conviction_grade
            for _, r in _tf.iterrows():
                bull_int = float(r.get("bullish_flow_intensity", 0) or 0)
                bear_int = float(r.get("bearish_flow_intensity", 0) or 0)
                is_bull = bull_int >= bear_int
                bps = (bull_int if is_bull else bear_int) * 10_000
                label, label_cls = _prem_mcap_label(bps)
                prem = float(r.get("bullish_premium_raw" if is_bull else "bearish_premium_raw", 0) or 0)
                count = int(r.get("bullish_count" if is_bull else "bearish_count", 0) or 0)
                side = "bullish" if is_bull else "bearish"
                flow_score = float(r.get(f"{side}_score", 0) or 0)
                flow_score_10 = flow_score * 10.0
                avg_delta_raw = r.get(f"{side}_avg_delta")
                try:
                    avg_delta_val = float(avg_delta_raw) if avg_delta_raw is not None and not pd.isna(avg_delta_raw) else 0.0
                except (TypeError, ValueError):
                    avg_delta_val = 0.0
                source_mix_raw = r.get(f"{side}_delta_source_mix")
                try:
                    source_mix_val = float(source_mix_raw) if source_mix_raw is not None and not pd.isna(source_mix_raw) else None
                except (TypeError, ValueError):
                    source_mix_val = None
                try:
                    reasons = build_flow_grade_reasons(r.to_dict(), side=side)
                except Exception:
                    reasons = []

                _tk = str(r.get("ticker") or "").strip().upper()
                _ctx = _tf_ctx_lookup.get(_tk, {}) or {}

                # Wave 5 — compose a dict rich enough for the conviction
                # stack to give a meaningful score even without the full
                # candidate enrichment (dark_pool / hot_chain live on the
                # candidate rows, not here, so stack will lean on flow_core
                # + dealer_regime on top_flow rows; that's expected).
                _stack_in = {
                    "direction": "LONG" if is_bull else "SHORT",
                    "flow_score_scaled": flow_score_10,
                    "dealer_hedge_bias": _ctx.get("dealer_hedge_bias"),
                }
                try:
                    from app.features.conviction_stack import compute_conviction_stack as _cs
                    _stack = _cs(_stack_in)
                except Exception:
                    _stack = None

                # "Catalyst" = earnings in the next 2 weeks.  News/congress
                # chips (Wave-3) were cancelled; earnings is what we already
                # fetch for free.
                _due = _ctx.get("days_until_earnings")
                try:
                    _due_f = float(_due) if _due is not None else None
                except (TypeError, ValueError):
                    _due_f = None
                _catalyst_label = None
                _catalyst_tone = None
                if _due_f is not None and 0 <= _due_f <= 14:
                    _catalyst_label = f"ER {int(_due_f)}d"
                    _catalyst_tone = "hot" if _due_f <= 5 else "warm"

                top_flow.append({
                    "ticker": r.get("ticker", "?"),
                    "direction": "LONG" if is_bull else "SHORT",
                    "bps": round(bps, 2),
                    "label": label,
                    "label_cls": label_cls,
                    "premium": prem,
                    "count": count,
                    "flow_score": flow_score,
                    "flow_score_10": round(flow_score_10, 2),
                    "conviction_grade": conviction_grade(flow_score_10),
                    "avg_delta": round(avg_delta_val, 3) if avg_delta_val else 0.0,
                    "delta_source_mix": round(source_mix_val, 3) if source_mix_val is not None else None,
                    "grade_reasons": reasons,
                    # ── Wave 5 chips ─────────────────────────────────
                    "conviction_stack": _stack,
                    "dealer_hedge_bias": _ctx.get("dealer_hedge_bias"),
                    "dealer_hedge_label": _ctx.get("dealer_hedge_label"),
                    "pin_risk_distance_pct": _ctx.get("pin_risk_distance_pct"),
                    "pin_risk_concentration": _ctx.get("pin_risk_concentration"),
                    "days_until_earnings": _due_f,
                    "catalyst_label": _catalyst_label,
                    "catalyst_tone": _catalyst_tone,
                })

    perf = _compute_performance()

    # Agent shadow portfolio
    positions_agent = load_positions_agent()
    tl_agent = load_trade_log_agent()
    ec_agent = load_equity_curve_agent()
    perf_agent = _compute_performance(tl=tl_agent, ec=ec_agent)

    last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _parse_scan_ts(src: str) -> str:
        """Extract a human-readable timestamp from a CSV filename like 'final_signals_20260322_131750.csv'."""
        import re
        m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", src)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)} UTC"
        return src or "unknown"
    scan_timestamp = _parse_scan_ts(signals_src or rejected_src or flow_src)

    # Cap rejected ("near misses") to top 15 by final_score
    NEAR_MISS_LIMIT = 15
    total_rejected_count = len(rejected)
    if not rejected.empty and "final_score" in rejected.columns:
        rejected_top = rejected.copy()
        rejected_top["_fs"] = pd.to_numeric(rejected_top["final_score"], errors="coerce")
        rejected_top = rejected_top.sort_values("_fs", ascending=False).head(NEAR_MISS_LIMIT)
        rejected_top.drop(columns=["_fs"], inplace=True, errors="ignore")
    else:
        rejected_top = rejected.head(NEAR_MISS_LIMIT)

    overview = _build_overview(positions, perf, regime)
    risk_data = _build_risk(positions)
    activity = _recent_activity(positions, load_trade_log_tail(20))

    # Enrich live positions with R-at-market and heat-contribution.
    positions = _enrich_positions_live(positions)
    positions_agent = _enrich_positions_live(positions_agent)

    # Grade-backtest header for Flow Tracker + action bar.
    grade_stats = load_grade_stats()
    grade_stats_header = _grade_stats_header(grade_stats)

    # Top-of-page action bar (uses signals DataFrame — signals already loaded above).
    action_bar_ctx = _build_action_bar(positions, regime, signals, watchlist, grade_stats)

    # "Trades to Take Now" — top 5 fresh signals not in book or watchlist.
    _vix_mult = (regime or {}).get("vix_sizing_mult")
    # Wave 8 — pull the freshly-computed risk_regime payload off the
    # action bar so Trader Cards layer the regime multiplier on top of
    # VIX sizing and expose the sizing-context block inside the
    # Structure tab.  Re-used below to decorate Flow Tracker rows.
    _risk_regime_for_tracker = (
        action_bar_ctx.get("risk_regime") if isinstance(action_bar_ctx, dict) else None
    )
    actionable_now = build_trader_card_rows(
        _actionable_now(signals, positions, watchlist, limit=5),
        vix_sizing_mult=float(_vix_mult) if _vix_mult else None,
        risk_regime=_risk_regime_for_tracker,
    )

    pp = TABLE_PAGE_SIZE
    page_bull = _int_query("page_bull")
    page_bear = _int_query("page_bear")
    page_near = _int_query("page_near")
    page_rej = _int_query("page_rej")
    page_tr = _int_query("page_tr")
    page_pos = _int_query("page_pos")
    page_wl = _int_query("page_wl")

    rejected_sorted = _sort_df(rejected_top, sort_rej) if sort_rej else rejected_top

    # Hottest chains and insider transactions (pre-compute for flow tracker enrichment)
    _hc_raw = load_hottest_chains()
    _hottest_chains_data = aggregate_chains_by_ticker(_hc_raw) if _hc_raw else {"by_ticker": [], "contracts": 0}
    _hottest_chains_by_ticker = {
        t["ticker"]: t for t in _hottest_chains_data.get("by_ticker", [])
    }

    _insider_raw = load_insider_recent()
    _insider_by_ticker = classify_insider_activity(_insider_raw) if _insider_raw else {}

    # Wave 1 — compute tracker rows once (needed both for the main list
    # and for the "Now What" 3-card hero strip at the top of the panel).
    _flow_tracker_rows = _build_flow_tracker(
        lookback_days=FLOW_TRACKER_HORIZONS[active_horizon]["lookback_days"],
        min_active_days=FLOW_TRACKER_HORIZONS[active_horizon]["min_active_days"],
        flow=flow,
        hottest_chains_by_ticker=_hottest_chains_by_ticker,
        insider_by_ticker=_insider_by_ticker,
        risk_regime=_risk_regime_for_tracker,
    )

    # Flow-Tracker-Swing-Radar: cross-reference chip.  Tag each tracker
    # row as either "also in signals" (the ticker appears on today's
    # bullish/bearish signal list) or "radar only" (the tracker surfaces
    # it from the multi-day regression but the signal pipeline passed).
    # Gives the trader a quick filter for "this is a new idea vs. this
    # confirms what we're already watching".
    try:
        _bull_tickers = set()
        _bear_tickers = set()
        try:
            if isinstance(bull, pd.DataFrame) and "ticker" in bull.columns:
                _bull_tickers = {str(t).upper() for t in bull["ticker"].dropna().tolist()}
            if isinstance(bear, pd.DataFrame) and "ticker" in bear.columns:
                _bear_tickers = {str(t).upper() for t in bear["ticker"].dropna().tolist()}
        except Exception:
            _bull_tickers = _bear_tickers = set()
        for r in _flow_tracker_rows:
            sym = str(r.get("ticker") or "").upper()
            direction = str(r.get("direction") or "").upper()
            signal_set = _bull_tickers if direction == "BULLISH" else _bear_tickers
            r["cross_ref"] = {
                "in_signals": sym in signal_set,
                "label": "Also in signals" if sym in signal_set else "Radar only",
            }
    except Exception:
        pass

    ctx = {
        "rejected_html": _paginate_card_fragments(
            _rejected_card_fragments(rejected_sorted),
            page=page_rej,
            per_page=pp,
            page_param="page_rej",
            empty_msg="No near misses.",
        ),
        "rejected_src": rejected_src,
        "rejected_detail": _df_to_detail_json(rejected_top, risk_regime=_risk_regime_for_tracker),
        "total_rejected_count": total_rejected_count,
        "watchlist_detail": watchlist,
        "bull_html": _paginate_card_fragments(
            _candidates_card_fragments(bull, clickable=True),
            page=page_bull,
            per_page=pp,
            page_param="page_bull",
            empty_msg="No bullish candidates.",
        ),
        "bull_src": bull_src,
        "bull_detail": _df_to_detail_json(bull, risk_regime=_risk_regime_for_tracker),
        "bear_html": _paginate_card_fragments(
            _candidates_card_fragments(bear, clickable=True),
            page=page_bear,
            per_page=pp,
            page_param="page_bear",
            empty_msg="No bearish candidates.",
        ),
        "bear_src": bear_src,
        "bear_detail": _df_to_detail_json(bear, risk_regime=_risk_regime_for_tracker),
        "near_html": _paginate_card_fragments(
            _candidates_card_fragments(
                near_df,
                clickable=True,
                is_near_miss=True,
                near_threshold_long=_near_long_min,
                near_threshold_short=_near_short_min,
            ),
            page=page_near,
            per_page=pp,
            page_param="page_near",
            empty_msg="No candidates near threshold.",
        ),
        "near_detail": _df_to_detail_json(near_df, risk_regime=_risk_regime_for_tracker),
        "near_threshold_long": round(_near_long_min, 1),
        "near_threshold_short": round(_near_short_min, 1),
        "positions_html": _paginate_card_fragments(
            _positions_card_fragments(positions),
            page=page_pos,
            per_page=pp,
            page_param="page_pos",
            empty_msg="No open positions.",
        ),
        "positions_detail": positions,
        "watchlist_html": _paginate_card_fragments(
            _watchlist_card_fragments(watchlist),
            page=page_wl,
            per_page=pp,
            page_param="page_wl",
            empty_msg="No tickers on watchlist.",
        ),
        "trades_html": _paginate_card_fragments(
            _trades_card_fragments(trades),
            page=page_tr,
            per_page=pp,
            page_param="page_tr",
            empty_msg="No trades in tail.",
        ),
        "alerts_html": alerts_html,
        "alerts_count": len(alerts),
        # Wave 5 — reflect the current opening-only state into the template
        # so the UI toggle can hydrate with the right checked state.
        "alerts_opening_only": bool(alerts_opening_only) if alerts_opening_only is not None else False,
        "top_flow": top_flow,
        "table_page_size": pp,
        "perf": perf,
        "overview": overview,
        "risk": risk_data,
        "activity": activity,
        "last_updated": last_updated,
        "candidates_insights": _candidates_insights(bull, bear, rejected_all),
        "positions_insights": _positions_insights(positions),
        "rejected_insights": _rejected_insights(rejected_all, watchlist),
        "trivial_rejected_count": trivial_rejected_count,
        "regime": regime,
        "sort_bull": sort_bull,
        "sort_bear": sort_bear,
        "sort_rej": sort_rej,
        "scan_timestamp": scan_timestamp,
        # Agent shadow portfolio
        "positions_agent": positions_agent,
        "positions_agent_html": _paginate_card_fragments(
            _positions_card_fragments(positions_agent),
            page=_int_query("page_pos_agent"),
            per_page=pp,
            page_param="page_pos_agent",
            empty_msg="No agent portfolio positions. Agent portfolio populates once the OpenAI API key is configured and agents run.",
        ),
        "perf_agent": perf_agent,
        # Multi-day flow tracker enriched with sentiment + dark pool + chains + insider + earnings + delta + decision context.
        # Wave 0.5 C1 — respects ?horizon=5d|15d on the URL.  Server-side
        # selection keeps the payload small and sidesteps the double-compute
        # cost of pre-building every horizon on every pageload.
        "flow_tracker": _flow_tracker_rows,
        "flow_tracker_active_horizon": active_horizon,
        "flow_tracker_horizons_config": FLOW_TRACKER_HORIZONS,
        "flow_tracker_hero": _build_flow_tracker_hero(_flow_tracker_rows),
        # Trader dashboard decision surfaces
        "ab": action_bar_ctx,
        "actionable_now": actionable_now,
        "grade_stats": grade_stats,
        "grade_stats_header": grade_stats_header,
        "flow_tracker_lookback": FLOW_TRACKER_HORIZONS[active_horizon]["lookback_days"],
        "flow_tracker_mode_default": FLOW_TRACKER_MODE_DEFAULT,
        "flow_tracker_auto_widen_min": FLOW_TRACKER_AUTO_WIDEN_MIN,
        # Market-wide dark pool screener (single-day)
        "dark_pool_screener": _build_dark_pool_screener(flow),
        # Daily accumulated dark pool (deduped across all intra-day scans)
        "dp_daily": _build_daily_accumulated_dp(flow),
        # Multi-day dark pool tracker (enriched with chains + insider + z-tier).
        # Wave 2 — attach_dp_z_tiers decorates each row with a per-ticker 30d
        # z-tier on today's notional so the UI can flag "genuinely unusual
        # for THIS name" vs "normal DP activity".
        "dp_tracker": attach_dp_z_tiers(
            _enrich_flow_tracker_insider(
                _enrich_flow_tracker_chains(
                    compute_multi_day_dp(DP_TRACKER_LOOKBACK_DAYS, DP_TRACKER_MIN_ACTIVE_DAYS),
                    _hottest_chains_by_ticker,
                ),
                _insider_by_ticker,
            )
        ),
        "dp_tracker_lookback": DP_TRACKER_LOOKBACK_DAYS,
        # Hottest chains screener
        "hottest_chains": _hottest_chains_data,
        # Insider data for candidate enrichment
        "insider_by_ticker": _insider_by_ticker,
    }
    return render_template("index.html", **ctx)


@app.route("/api/alerts")
def api_alerts():
    from app.vendors.unusual_whales import fetch_recent_alert_flow

    hours = int(request.args.get("hours", 24))
    limit = min(int(request.args.get("limit", 150)), 300)
    # Wave 5 — opening-only toggle.  Accepts 1/0, true/false, yes/no.
    # ``None`` (unset) preserves the existing behaviour dictated by the
    # module-wide ``FLOW_OPENING_ONLY`` config flag.
    oo_raw = request.args.get("opening_only")
    opening_only: bool | None
    if oo_raw is None:
        opening_only = None
    else:
        opening_only = str(oo_raw).strip().lower() in {"1", "true", "yes", "on"}
    try:
        df = fetch_recent_alert_flow(
            limit=limit,
            hours_back=hours,
            opening_only=opening_only,
        )
        sub = _alerts_subset(df)
        return jsonify(
            {
                "ok": True,
                "count": len(sub),
                "opening_only": bool(opening_only) if opening_only is not None else None,
                "rows": sub.fillna("").to_dict(orient="records"),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "rows": []}), 200


def _iv_rank_score(iv_rank: float | None) -> float:
    """Piecewise linear IV rank → score (0-2.5)."""
    if iv_rank is None:
        return 0.0
    r = float(iv_rank)
    if r <= 5:
        return 0.0
    if r <= 15:
        return (r - 5) / 10 * 1.0
    if r <= 45:
        return 1.0 + (r - 15) / 30 * 1.5
    if r <= 60:
        return 2.5 - (r - 45) / 15 * 1.0
    if r <= 75:
        return 1.5 - (r - 60) / 15 * 1.0
    if r <= 100:
        return 0.5 - (r - 75) / 25 * 0.5
    return 0.0


def _compute_options_score(direction: str, opts_ctx: dict | None) -> float | None:
    """Derive a 0-10 composite options context score (standalone, no yfinance)."""
    if not opts_ctx or not opts_ctx.get("options_context_available"):
        return None
    score = 0.0
    regime = opts_ctx.get("gamma_regime", "NEUTRAL")
    dist_call = opts_ctx.get("distance_to_call_wall_pct")
    dist_put = opts_ctx.get("distance_to_put_wall_pct")
    near_oi = opts_ctx.get("near_term_oi") or 0
    swing_oi = opts_ctx.get("swing_dte_oi") or 0
    pcr = opts_ctx.get("ticker_put_call_ratio")
    bull_prem = opts_ctx.get("daily_bullish_premium")
    bear_prem = opts_ctx.get("daily_bearish_premium")
    if direction == "LONG":
        if regime == "NEGATIVE": score += 2.5
        elif regime == "NEUTRAL": score += 1.25
    else:
        if regime == "NEGATIVE": score += 2.5
        elif regime == "NEUTRAL": score += 1.25
    if direction == "LONG":
        if dist_call is not None:
            if dist_call > 5.0: score += 2.5
            elif dist_call > 2.0: score += 1.25
    else:
        if dist_put is not None:
            if dist_put > 5.0: score += 2.5
            elif dist_put > 2.0: score += 1.25
    if swing_oi > near_oi and near_oi > 0:
        score += 0.75
    if pcr is not None:
        if direction == "LONG" and pcr < 0.7: score += 0.75
        elif direction == "SHORT" and pcr > 1.3: score += 0.75
        elif direction == "LONG" and pcr < 1.0: score += 0.375
        elif direction == "SHORT" and pcr > 1.0: score += 0.375
    if bull_prem is not None and bear_prem is not None:
        total = bull_prem + bear_prem
        if total > 0:
            aligned = bull_prem / total if direction == "LONG" else bear_prem / total
            score += min(1.0, aligned * 1.5)
    score += _iv_rank_score(opts_ctx.get("iv_rank"))
    return round(min(10.0, score), 1)


@app.route("/api/scan-ticker")
def api_scan_ticker():
    """Run flow + options scoring for a single ticker and return JSON."""
    from app.features.flow_features import build_flow_feature_table
    from app.features.options_context import clear_context_cache, fetch_options_context
    from app.vendors.unusual_whales import (
        fetch_dark_pool,
        fetch_flow_for_tickers,
        fetch_flow_recent,
        fetch_net_prem_ticks,
    )

    clear_context_cache()

    ticker = (request.args.get("ticker") or "").strip().upper()
    if not ticker or not ticker.isalpha():
        return jsonify({"ok": False, "error": "Invalid ticker symbol"}), 400

    result: dict = {"ok": True, "ticker": ticker}
    spot: float | None = None

    def _sf(val, digits=2):
        v = float(val) if val is not None else 0.0
        return round(v, digits) if v == v else 0.0

    try:
        # --- Flow alerts (lowered thresholds for overview) ---
        flow_data: dict = {"available": False}
        try:
            norm = fetch_flow_for_tickers([ticker])
            if not norm.empty:
                if "underlying_price" in norm.columns:
                    prices = pd.to_numeric(norm["underlying_price"], errors="coerce").dropna()
                    if not prices.empty:
                        spot = round(float(prices.iloc[-1]), 2)

                feature_table = build_flow_feature_table(
                    norm, min_premium=25_000, min_dte=1, max_dte=365,
                )

                # Attach z-score tier info (informational; doesn't change scoring
                # here because ticker-detail is a single-ticker scan).
                try:
                    from app.features.flow_features import rescore_with_z
                    from app.features.flow_stats import load_history as _load_flow_history
                    from app.config import ZSCORE_LOOKBACK_DAYS as _ZLOOKBACK
                    _hist = _load_flow_history(lookback_days=_ZLOOKBACK)
                    if not _hist.empty and not feature_table.empty:
                        feature_table = rescore_with_z(feature_table, _hist)
                except Exception:
                    pass

                if not feature_table.empty and ticker in feature_table["ticker"].values:
                    row = feature_table[feature_table["ticker"] == ticker].iloc[0]
                    from app.features.grade_explainer import (
                        build_flow_grade_reasons as _bgr,
                        conviction_grade as _cgrade,
                    )
                    _bull_reasons = _bgr(row.to_dict(), side="bullish")
                    _bear_reasons = _bgr(row.to_dict(), side="bearish")
                    flow_data = {
                        "available": True,
                        "bullish_raw": _sf(row.get("bullish_score", 0), 4),
                        "bearish_raw": _sf(row.get("bearish_score", 0), 4),
                        "bullish_scaled": _sf(row.get("bullish_score", 0) * 10),
                        "bearish_scaled": _sf(row.get("bearish_score", 0) * 10),
                        "bullish_premium": _sf(row.get("bullish_premium", 0)),
                        "bearish_premium": _sf(row.get("bearish_premium", 0)),
                        # Populated by _enrich_net_prem_ticks on signal records;
                        # 0.0 on raw feature-table rows.
                        "directional_momentum": _sf(row.get("directional_momentum", 0), 4),
                        "directional_momentum_pts": _sf(row.get("directional_momentum_pts", 0), 2),
                        "bullish_flow_intensity": _sf(row.get("bullish_flow_intensity", 0), 4),
                        "bearish_flow_intensity": _sf(row.get("bearish_flow_intensity", 0), 4),
                        # Delta-weighted flow (shadow when USE_DELTA_WEIGHTED_FLOW is False)
                        "bullish_delta_intensity": _sf(row.get("bullish_delta_intensity", 0), 6),
                        "bearish_delta_intensity": _sf(row.get("bearish_delta_intensity", 0), 6),
                        "bullish_avg_delta": _sf(row.get("bullish_avg_delta", 0), 3),
                        "bearish_avg_delta": _sf(row.get("bearish_avg_delta", 0), 3),
                        "bullish_delta_source_mix": _sf(row.get("bullish_delta_source_mix", 0), 3),
                        "bearish_delta_source_mix": _sf(row.get("bearish_delta_source_mix", 0), 3),
                        "bullish_sweep_count": int(row.get("bullish_sweep_count", 0) or 0),
                        "bearish_sweep_count": int(row.get("bearish_sweep_count", 0) or 0),
                        "bullish_breadth": _sf(row.get("bullish_breadth", 0), 3),
                        "bearish_breadth": _sf(row.get("bearish_breadth", 0), 3),
                        "bullish_vol_oi": _sf(row.get("bullish_vol_oi", 0)),
                        "bearish_vol_oi": _sf(row.get("bearish_vol_oi", 0)),
                        "bullish_repeat_count": int(row.get("bullish_repeat_count", 0) or 0),
                        "bearish_repeat_count": int(row.get("bearish_repeat_count", 0) or 0),
                        "flow_imbalance_ratio": _sf(row.get("flow_imbalance_ratio", 0)),
                        # Z-score tier info (None when z-path unavailable)
                        "bullish_zscore_tier": int(row.get("bullish_zscore_tier")) if row.get("bullish_zscore_tier") is not None and not pd.isna(row.get("bullish_zscore_tier")) else None,
                        "bearish_zscore_tier": int(row.get("bearish_zscore_tier")) if row.get("bearish_zscore_tier") is not None and not pd.isna(row.get("bearish_zscore_tier")) else None,
                        "component_tiers": {
                            comp: {
                                "bullish": int(row.get(f"bullish_{comp}_tier")) if row.get(f"bullish_{comp}_tier") is not None and not pd.isna(row.get(f"bullish_{comp}_tier")) else None,
                                "bearish": int(row.get(f"bearish_{comp}_tier")) if row.get(f"bearish_{comp}_tier") is not None and not pd.isna(row.get(f"bearish_{comp}_tier")) else None,
                            }
                            for comp in ("flow_intensity", "premium_per_trade", "vol_oi", "repeat", "sweep", "breadth", "dte")
                        },
                        "bullish_grade": _cgrade(_sf(row.get("bullish_score", 0) * 10)),
                        "bearish_grade": _cgrade(_sf(row.get("bearish_score", 0) * 10)),
                        "bullish_grade_reasons": _bull_reasons,
                        "bearish_grade_reasons": _bear_reasons,
                    }
        except Exception as fe:
            flow_data["error"] = str(fe)

        result["flow"] = flow_data
        result["spot"] = spot

        # --- Net premium ticks (intraday direction) ---
        net_prem: dict = {"available": False}
        try:
            raw_np = fetch_net_prem_ticks(ticker)
            if raw_np:
                net_prem = {"available": True, **raw_np}
        except Exception:
            pass
        result["net_premium"] = net_prem

        # --- Flow recent (all-size summary) ---
        flow_rec: dict = {"available": False}
        try:
            raw_fr = fetch_flow_recent(ticker)
            if raw_fr:
                flow_rec = {"available": True, **raw_fr}
        except Exception:
            pass
        result["flow_recent"] = flow_rec

        # --- Dark pool ---
        dp: dict = {"available": False}
        try:
            raw_dp = fetch_dark_pool(ticker)
            if raw_dp:
                dp = {"available": True, **raw_dp}
        except Exception:
            pass
        result["dark_pool"] = dp

        # --- Options context ---
        opts_out: dict = {"available": False}
        if spot and spot > 0:
            opts_ctx: dict | None = None
            try:
                opts_ctx = fetch_options_context(ticker, spot)
                opts_score_long = _compute_options_score("LONG", opts_ctx)
                opts_score_short = _compute_options_score("SHORT", opts_ctx)
                if opts_ctx and opts_ctx.get("options_context_available"):
                    opts_out = {
                        "available": True,
                        "long_score": opts_score_long,
                        "short_score": opts_score_short,
                        "gamma_regime": opts_ctx.get("gamma_regime"),
                        "net_gex": opts_ctx.get("net_gex"),
                        "nearest_call_wall": opts_ctx.get("nearest_call_wall"),
                        "nearest_put_wall": opts_ctx.get("nearest_put_wall"),
                        "distance_to_call_wall_pct": opts_ctx.get("distance_to_call_wall_pct"),
                        "distance_to_put_wall_pct": opts_ctx.get("distance_to_put_wall_pct"),
                        "ticker_put_call_ratio": opts_ctx.get("ticker_put_call_ratio"),
                        "near_term_oi": opts_ctx.get("near_term_oi"),
                        "swing_dte_oi": opts_ctx.get("swing_dte_oi"),
                        "daily_bullish_premium": opts_ctx.get("daily_bullish_premium"),
                        "daily_bearish_premium": opts_ctx.get("daily_bearish_premium"),
                        "call_volume_today": opts_ctx.get("call_volume_today"),
                        "put_volume_today": opts_ctx.get("put_volume_today"),
                        "call_volume_vs_30d_avg": opts_ctx.get("call_volume_vs_30d_avg"),
                        "put_volume_vs_30d_avg": opts_ctx.get("put_volume_vs_30d_avg"),
                        "call_ask_bid_ratio": opts_ctx.get("call_ask_bid_ratio"),
                        "put_ask_bid_ratio": opts_ctx.get("put_ask_bid_ratio"),
                        "iv_rank": opts_ctx.get("iv_rank"),
                        "iv_current": opts_ctx.get("iv_current"),
                    }
            except Exception:
                pass
        else:
            opts_out["note"] = "No spot price available — options context requires flow data"

        result["options"] = opts_out

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "ticker": ticker}), 200

    return jsonify(result)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PULL_INTERVAL = int(os.environ.get("AUTO_PULL_INTERVAL", "300"))  # seconds
_log = logging.getLogger(__name__)


def _git_auto_pull() -> None:
    """Pull latest data from origin/main on a timer.  Runs in a daemon thread."""
    import time

    while True:
        time.sleep(_PULL_INTERVAL)
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only", "origin", "main"],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                summary = result.stdout.strip().split("\n")[-1]
                if "Already up to date" not in summary:
                    _log.info("auto-pull: %s", summary)
            else:
                _log.warning("auto-pull failed: %s", result.stderr.strip())
        except Exception as exc:
            _log.warning("auto-pull error: %s", exc)


def main() -> None:
    port = int(os.environ.get("PORT", "5050"))

    if os.environ.get("DISABLE_AUTO_PULL") != "1":
        t = threading.Thread(target=_git_auto_pull, daemon=True)
        t.start()
        _log.info("auto-pull enabled every %ds (set DISABLE_AUTO_PULL=1 to disable)", _PULL_INTERVAL)

    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()

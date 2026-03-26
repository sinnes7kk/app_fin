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

from app.config import MIN_FINAL_SCORE, REGIME_THRESHOLD_BOOST
from app.web.data_access import (
    load_equity_curve,
    load_final_signals,
    load_flow_features,
    load_positions,
    load_rejected,
    load_trade_log,
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
    "flow_intensity",
    "contracts",
    "volume",
    "open_interest",
    "dte",
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
    "bullish_flow_intensity": "Bullish premium / ADDV (scaled 1\u201310)",
    "bearish_flow_intensity": "Bearish premium / ADDV (scaled 1\u201310)",
    "flow_imbalance_ratio": "Bullish prem \u00f7 bearish prem (\u003e1 = bullish dominant)",
    "dominant_direction": "Which side has stronger flow",
    "total_premium": "Total options premium across all flow",
    "total_count": "Count of qualifying options-flow prints rolled into this ticker row",
    "avg_dte": "Average days to expiration of flow",
    "dte_score": "DTE quality score (higher = better positioned)",
    "addv": "Average daily dollar volume (liquidity anchor); flow intensity = premium / ADDV",
    # Alerts
    "event_ts": "Timestamp of the flow event",
    "option_type": "Call or put",
    "strike": "Option strike price",
    "expiration_date": "Option expiration date",
    "premium": "Total premium of the trade",
    "flow_intensity": "Premium / ADDV (scaled 1\u201310)",
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


def _round_floats(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round all float columns in a DataFrame for cleaner display."""
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols):
        df = df.copy()
        df[float_cols] = df[float_cols].round(decimals)
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


def _df_to_detail_json(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of dicts suitable for JSON serialization."""
    if df.empty:
        return []
    rounded = _round_floats(df)
    return rounded.fillna("").to_dict(orient="records")


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
) -> str:
    tcls = "data-card-ticker ticker-link" if clickable else "data-card-ticker"
    icls = " data-card--interactive" if clickable else ""
    dir_attr = f' data-direction="{html_escape(str(direction))}"' if direction else ""
    extra_attrs = ""
    if filter_attrs:
        for k, v in filter_attrs.items():
            extra_attrs += f' data-{k}="{html_escape(str(v))}"'
    return (
        f'<article class="data-card{icls}" data-ticker="{html_escape(str(ticker))}"{dir_attr}{extra_attrs}>'
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
        head = f'<span class="data-card-badges">{direc}{gamma}</span>'
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


def _candidates_card_fragments(df: pd.DataFrame, *, clickable: bool = True) -> list[str]:
    """Render validated candidate cards: 3 scores + pattern + status."""
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
        head = f'<span class="data-card-badges">{direc}{gamma}</span>'
        pairs: list[tuple[str, str]] = [
            ("Flow", _conviction_span(_row_scalar(row, "flow_score_scaled"))),
            ("Price", _conviction_span(_row_scalar(row, "price_score"))),
            ("Options", _conviction_span(_row_scalar(row, "options_context_score"))),
            ("Final", _conviction_span(_row_scalar(row, "final_score"))),
        ]
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
            ("Final score", _conviction_span(_row_scalar(row, "final_score"))),
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
        snip = "—"
        if strike is not None or exp is not None or ot is not None:
            snip = html_escape(
                f"{ot or ''} {strike or ''} @ {exp or ''}".strip()
            )
        prem = _fmt_money_short(_row_scalar(row, "premium"))
        head = f'<span class="data-card-badges">{direc}</span>'
        pairs = [
            ("Premium", prem),
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
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=ev_p,
                clickable=False,
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
            ("Seen", html_escape(str(r.get("first_seen", "")))),
        ]
        metrics = _card_metrics_dl(pairs)
        rr = r.get("reject_reason")
        ex = ""
        if rr:
            ex += f'<p class="data-card-thesis">{html_escape(str(rr))}</p>'
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
        head = f'<span class="data-card-badges">{direc}{health_badge}</span>'
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
        out.append(
            _data_card_article(
                t,
                head_badges_html=head,
                metrics_html=metrics,
                extra_html=ladder + od,
                clickable=True,
                direction=str(r.get("direction", "")),
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

    # Compute flow_intensity per ticker (premium / ADDV)
    if "premium" in df.columns and "ticker" in df.columns:
        try:
            from app.features.price_features import fetch_addv

            tickers = df["ticker"].dropna().unique()
            addv_map = {}
            for t in tickers:
                try:
                    addv_map[t] = fetch_addv(str(t))
                except Exception:
                    addv_map[t] = None
            df = df.copy()
            df["flow_intensity"] = df.apply(
                lambda r: (
                    round(float(r["premium"]) / addv_map[r["ticker"]], 6)
                    if pd.notna(r.get("premium")) and addv_map.get(r["ticker"])
                    else None
                ),
                axis=1,
            )
        except Exception:
            df = df.copy()
            df["flow_intensity"] = None

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


def _compute_performance() -> dict:
    """Build all performance analytics from the trade log and equity curve."""
    tl = load_trade_log()
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


@app.route("/")
def index():
    signals, signals_src = load_final_signals()
    rejected_all, rejected_src = load_rejected()
    flow, flow_src = load_flow_features()
    regime = load_market_regime()

    # Filter out trivial rejections that never reached full price scoring.
    _TRIVIAL_PREFIXES = ("weak_bullish_flow", "weak_bearish_flow",
                         "trend_not_aligned", "price_over_extended", "error:")

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
    try:
        alerts_raw = fetch_recent_alert_flow(limit=150, hours_back=24)
    except Exception as e:
        alerts_error = str(e)
    alerts = _alerts_subset(alerts_raw)
    if not alerts.empty and "flow_intensity" in alerts.columns:
        alerts = alerts.sort_values("flow_intensity", ascending=False).head(30)
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

    perf = _compute_performance()
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

    pp = TABLE_PAGE_SIZE
    page_bull = _int_query("page_bull")
    page_bear = _int_query("page_bear")
    page_near = _int_query("page_near")
    page_rej = _int_query("page_rej")
    page_tr = _int_query("page_tr")
    page_pos = _int_query("page_pos")
    page_wl = _int_query("page_wl")

    rejected_sorted = _sort_df(rejected_top, sort_rej) if sort_rej else rejected_top

    ctx = {
        "rejected_html": _paginate_card_fragments(
            _rejected_card_fragments(rejected_sorted),
            page=page_rej,
            per_page=pp,
            page_param="page_rej",
            empty_msg="No near misses.",
        ),
        "rejected_src": rejected_src,
        "rejected_detail": _df_to_detail_json(rejected_top),
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
        "bull_detail": _df_to_detail_json(bull),
        "bear_html": _paginate_card_fragments(
            _candidates_card_fragments(bear, clickable=True),
            page=page_bear,
            per_page=pp,
            page_param="page_bear",
            empty_msg="No bearish candidates.",
        ),
        "bear_src": bear_src,
        "bear_detail": _df_to_detail_json(bear),
        "near_html": _paginate_card_fragments(
            _candidates_card_fragments(near_df, clickable=True),
            page=page_near,
            per_page=pp,
            page_param="page_near",
            empty_msg="No candidates near threshold.",
        ),
        "near_detail": _df_to_detail_json(near_df),
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
    }
    return render_template("index.html", **ctx)


@app.route("/api/alerts")
def api_alerts():
    from app.vendors.unusual_whales import fetch_recent_alert_flow

    hours = int(request.args.get("hours", 24))
    limit = min(int(request.args.get("limit", 150)), 300)
    try:
        df = fetch_recent_alert_flow(limit=limit, hours_back=hours)
        sub = _alerts_subset(df)
        return jsonify(
            {
                "ok": True,
                "count": len(sub),
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
    from app.features.options_context import fetch_options_context
    from app.vendors.unusual_whales import (
        fetch_dark_pool,
        fetch_flow_for_tickers,
        fetch_flow_recent,
        fetch_net_prem_ticks,
    )

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
                if not feature_table.empty and ticker in feature_table["ticker"].values:
                    row = feature_table[feature_table["ticker"] == ticker].iloc[0]
                    flow_data = {
                        "available": True,
                        "bullish_raw": _sf(row.get("bullish_score", 0), 4),
                        "bearish_raw": _sf(row.get("bearish_score", 0), 4),
                        "bullish_scaled": _sf(row.get("bullish_score", 0) * 10),
                        "bearish_scaled": _sf(row.get("bearish_score", 0) * 10),
                        "bullish_premium": _sf(row.get("bullish_premium", 0)),
                        "bearish_premium": _sf(row.get("bearish_premium", 0)),
                        "flow_velocity": _sf(row.get("flow_velocity", 0)),
                        "bullish_flow_intensity": _sf(row.get("bullish_flow_intensity", 0), 4),
                        "bearish_flow_intensity": _sf(row.get("bearish_flow_intensity", 0), 4),
                        "bullish_sweep_count": int(row.get("bullish_sweep_count", 0) or 0),
                        "bearish_sweep_count": int(row.get("bearish_sweep_count", 0) or 0),
                        "bullish_breadth": _sf(row.get("bullish_breadth", 0), 3),
                        "bearish_breadth": _sf(row.get("bearish_breadth", 0), 3),
                        "bullish_vol_oi": _sf(row.get("bullish_vol_oi", 0)),
                        "bearish_vol_oi": _sf(row.get("bearish_vol_oi", 0)),
                        "bullish_repeat_count": int(row.get("bullish_repeat_count", 0) or 0),
                        "bearish_repeat_count": int(row.get("bearish_repeat_count", 0) or 0),
                        "flow_imbalance_ratio": _sf(row.get("flow_imbalance_ratio", 0)),
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

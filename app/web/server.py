"""Flask dashboard: tabs for positions, candidates, signals, UW alerts, watchlist."""

from __future__ import annotations

import os

import pandas as pd
from flask import Flask, jsonify, render_template, request

from app.web.data_access import (
    load_final_signals,
    load_flow_features,
    load_positions,
    load_ranked_bearish,
    load_ranked_bullish,
    load_rejected,
    load_trade_log_tail,
    load_watchlist,
)

app = Flask(__name__, static_folder="static", template_folder="templates")

ALERT_DISPLAY_COLS = [
    "event_ts",
    "ticker",
    "option_type",
    "strike",
    "expiration_date",
    "premium",
    "contracts",
    "volume",
    "open_interest",
    "dte",
    "direction",
    "execution_side",
    "alert_rule",
]


def _df_html(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return '<p class="empty">No data. Run <code>python -m app.main --scan-only</code> to refresh outputs.</p>'
    view = df.head(max_rows) if max_rows else df
    return view.to_html(classes="data", index=False, border=0, escape=True, na_rep="—")


def _positions_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="empty">No open positions.</p>'
    df = pd.json_normalize(rows)
    return _df_html(df)


def _alerts_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
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


@app.route("/")
def index():
    signals, signals_src = load_final_signals()
    rejected, rejected_src = load_rejected()
    bull, bull_src = load_ranked_bullish()
    bear, bear_src = load_ranked_bearish()
    flow, flow_src = load_flow_features()
    positions = load_positions()
    watchlist = load_watchlist()
    trades = load_trade_log_tail(80)

    from app.vendors.unusual_whales import fetch_recent_alert_flow

    alerts_raw = pd.DataFrame()
    alerts_error = ""
    if request.args.get("live_alerts") == "1" or os.environ.get("DASHBOARD_LOAD_ALERTS") == "1":
        try:
            alerts_raw = fetch_recent_alert_flow(limit=150, hours_back=24)
        except Exception as e:
            alerts_error = str(e)
    alerts = _alerts_subset(alerts_raw)

    if alerts_raw.empty and not alerts_error and request.args.get("live_alerts") != "1":
        alerts_html = (
            '<p class="empty">Click <strong>Load / refresh alerts</strong> to query Unusual Whales '
            "(requires <code>UNUSUAL_WHALES_API_KEY</code> in <code>.env</code>), or open "
            '<a href="/?live_alerts=1">with alerts preloaded</a>.</p>'
        )
    elif alerts_error:
        alerts_html = f'<p class="error">{alerts_error}</p>'
    else:
        alerts_html = _df_html(alerts, max_rows=200)

    ctx = {
        "signals_html": _df_html(signals),
        "signals_src": signals_src,
        "rejected_html": _df_html(rejected, max_rows=200),
        "rejected_src": rejected_src,
        "bull_html": _df_html(bull),
        "bull_src": bull_src,
        "bear_html": _df_html(bear),
        "bear_src": bear_src,
        "flow_html": _df_html(flow, max_rows=150),
        "flow_src": flow_src,
        "positions_html": _positions_table(positions),
        "watchlist_html": _positions_table(watchlist),
        "trades_html": _df_html(trades),
        "alerts_html": alerts_html,
        "alerts_count": len(alerts_raw),
        "live_alerts_loaded": not alerts_raw.empty or bool(alerts_error),
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


def main() -> None:
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()

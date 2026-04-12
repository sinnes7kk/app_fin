"""Persist daily sentiment snapshots and compute multi-day sentiment trends."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from app.config import FLOW_TRACKER_LOOKBACK_DAYS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SENTIMENT_PATH = DATA_DIR / "sentiment_snapshots.csv"

SENTIMENT_COLS = [
    "snapshot_date",
    "ticker",
    "st_messages",
    "st_bullish",
    "st_bearish",
    "st_sentiment",
    "rd_mentions",
    "rd_posts",
    "rd_comments",
    "rd_top_sub",
    "rd_sentiment",
]


def fetch_and_save_sentiment(tickers: list[str]) -> None:
    """Fetch StockTwits + Reddit sentiment for *tickers* and persist to CSV.

    Upserts today's rows and prunes data outside the lookback window.
    Silently skips sources whose credentials are not configured.
    """
    from app.vendors.sentiment import fetch_reddit_mentions, fetch_stocktwits_sentiment

    if not tickers:
        return

    today_str = str(date.today())
    cutoff = str(date.today() - timedelta(days=FLOW_TRACKER_LOOKBACK_DAYS + 3))

    new_rows: list[dict] = []
    for ticker in tickers:
        row: dict = {"snapshot_date": today_str, "ticker": ticker.upper()}

        st = fetch_stocktwits_sentiment(ticker)
        if st:
            row.update(st)
        else:
            row.update({"st_messages": 0, "st_bullish": 0, "st_bearish": 0, "st_sentiment": 0.5})

        rd = fetch_reddit_mentions(ticker)
        if rd:
            row.update(rd)
        else:
            row.update({"rd_mentions": 0, "rd_posts": 0, "rd_comments": 0, "rd_top_sub": "—", "rd_sentiment": "neutral"})

        new_rows.append(row)

    if not new_rows:
        return

    existing: list[dict] = []
    if SENTIMENT_PATH.exists():
        try:
            with open(SENTIMENT_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    d = r.get("snapshot_date", "")
                    if d != today_str and d >= cutoff:
                        existing.append(r)
        except Exception:
            existing = []

    SENTIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = existing + new_rows
    with open(SENTIMENT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SENTIMENT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  [sentiment] saved {len(new_rows)} sentiment rows for {today_str} "
          f"({len(existing)} historical rows retained)")


def compute_sentiment_trend(
    tickers: list[str],
    lookback_days: Optional[int] = None,
) -> dict[str, dict]:
    """Aggregate sentiment snapshots and compute trends for each ticker.

    Returns {ticker: {combined_sentiment, st_mention_trend, rd_mention_trend,
    mention_spike, daily_mentions, ...}}.
    """
    lookback = lookback_days or FLOW_TRACKER_LOOKBACK_DAYS

    if not SENTIMENT_PATH.exists():
        return {}

    try:
        df = pd.read_csv(SENTIMENT_PATH)
    except Exception:
        return {}

    if df.empty or "snapshot_date" not in df.columns:
        return {}

    cutoff = str(date.today() - timedelta(days=lookback))
    df = df[df["snapshot_date"] >= cutoff].copy()
    if df.empty:
        return {}

    for col in ("st_messages", "st_bullish", "st_bearish", "st_sentiment",
                "rd_mentions", "rd_posts", "rd_comments"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    all_dates = sorted(df["snapshot_date"].unique())
    result: dict[str, dict] = {}

    for ticker in tickers:
        ticker = ticker.upper()
        grp = df[df["ticker"] == ticker]
        if grp.empty:
            result[ticker] = _empty_sentiment()
            continue

        sorted_grp = grp.sort_values("snapshot_date")
        latest = sorted_grp.iloc[-1]
        first = sorted_grp.iloc[0]

        # StockTwits mention trend
        st_first = float(first.get("st_messages") or 0)
        st_latest = float(latest.get("st_messages") or 0)
        st_trend = _growth_label(st_first, st_latest)

        # Reddit mention trend
        rd_first = float(first.get("rd_mentions") or 0)
        rd_latest = float(latest.get("rd_mentions") or 0)
        rd_trend = _growth_label(rd_first, rd_latest)

        # Combined sentiment: ST weight=0.4, RD weight=0.6
        st_sent = float(latest.get("st_sentiment") or 0.5)
        rd_sent_str = str(latest.get("rd_sentiment") or "neutral")
        rd_sent_num = {"bullish": 0.8, "bearish": 0.2, "neutral": 0.5}.get(rd_sent_str, 0.5)
        combined = round(st_sent * 0.4 + rd_sent_num * 0.6, 3)

        if combined >= 0.65:
            combined_label = "bullish"
        elif combined <= 0.35:
            combined_label = "bearish"
        else:
            combined_label = "neutral"

        # Mention spike: today's combined mentions > 2x window average
        combined_mentions_today = st_latest + rd_latest
        avg_mentions = (grp["st_messages"].sum() + grp["rd_mentions"].sum()) / max(len(sorted_grp), 1)
        mention_spike = combined_mentions_today > (2 * avg_mentions) if avg_mentions > 0 else False

        # Daily mention counts for sparkline
        daily_mentions: list[dict] = []
        for d in all_dates:
            day_row = grp[grp["snapshot_date"] == d]
            if day_row.empty:
                daily_mentions.append({"date": d, "st": 0, "rd": 0, "total": 0})
            else:
                dr = day_row.iloc[0]
                st_m = int(dr.get("st_messages") or 0)
                rd_m = int(dr.get("rd_mentions") or 0)
                daily_mentions.append({"date": d, "st": st_m, "rd": rd_m, "total": st_m + rd_m})

        result[ticker] = {
            "st_messages": int(st_latest),
            "st_bullish": int(latest.get("st_bullish") or 0),
            "st_bearish": int(latest.get("st_bearish") or 0),
            "st_sentiment": round(st_sent, 3),
            "st_mention_trend": st_trend,
            "rd_mentions": int(rd_latest),
            "rd_posts": int(latest.get("rd_posts") or 0),
            "rd_comments": int(latest.get("rd_comments") or 0),
            "rd_top_sub": str(latest.get("rd_top_sub") or "—"),
            "rd_sentiment": rd_sent_str,
            "rd_mention_trend": rd_trend,
            "combined_sentiment": combined,
            "combined_label": combined_label,
            "mention_spike": bool(mention_spike),
            "daily_mentions": daily_mentions,
        }

    return result


def _growth_label(first_val: float, latest_val: float) -> str:
    if first_val == 0 and latest_val > 0:
        return "new"
    if first_val == 0:
        return "stable"
    ratio = latest_val / first_val
    if ratio > 1.5:
        return "growing"
    if ratio < 0.5:
        return "declining"
    return "stable"


def _empty_sentiment() -> dict:
    return {
        "st_messages": 0,
        "st_bullish": 0,
        "st_bearish": 0,
        "st_sentiment": 0.5,
        "st_mention_trend": "stable",
        "rd_mentions": 0,
        "rd_posts": 0,
        "rd_comments": 0,
        "rd_top_sub": "—",
        "rd_sentiment": "neutral",
        "rd_mention_trend": "stable",
        "combined_sentiment": 0.5,
        "combined_label": "neutral",
        "mention_spike": False,
        "daily_mentions": [],
    }

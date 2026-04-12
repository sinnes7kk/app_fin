"""Vendor wrappers for social-sentiment APIs (StockTwits + Reddit public JSON)."""

from __future__ import annotations

import re
import time
from typing import Optional

import requests

from app.config import (
    REDDIT_SUBREDDITS,
    REDDIT_USER_AGENT,
    STOCKTWITS_API_KEY,
)

# ---------------------------------------------------------------------------
# StockTwits
# ---------------------------------------------------------------------------

_ST_BASE = "https://api.stocktwits.com/api/2"


def fetch_stocktwits_sentiment(ticker: str) -> Optional[dict]:
    """Fetch 24-hour message volume and sentiment for *ticker* from StockTwits.

    Uses the free public stream endpoint (200 req/hr).
    Returns a dict with keys: st_messages, st_bullish, st_bearish, st_sentiment
    or None on failure.
    """
    url = f"{_ST_BASE}/streams/symbol/{ticker.upper()}.json"
    params: dict = {}
    if STOCKTWITS_API_KEY:
        params["access_token"] = STOCKTWITS_API_KEY

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            print(f"  [sentiment] StockTwits rate-limited for {ticker}")
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"  [sentiment] StockTwits error for {ticker}: {exc}")
        return None

    messages = data.get("messages") or []
    bullish = sum(1 for m in messages if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bullish")
    bearish = sum(1 for m in messages if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bearish")
    total = len(messages)
    sentiment = round(bullish / total, 3) if total > 0 else 0.5

    return {
        "st_messages": total,
        "st_bullish": bullish,
        "st_bearish": bearish,
        "st_sentiment": sentiment,
    }


# ---------------------------------------------------------------------------
# Reddit (public JSON API — no credentials required)
# ---------------------------------------------------------------------------

_BULLISH_FLAIRS = {"dd", "gain", "due diligence", "technical analysis", "bullish", "catalyst"}
_BEARISH_FLAIRS = {"loss", "meme", "shitpost", "bearish", "yolo"}

_RD_RATE_DELAY = 1.2  # seconds between requests to stay under ~10 req/min unauthenticated


def fetch_reddit_mentions(
    ticker: str,
    subreddits: Optional[list] = None,
    lookback_hours: int = 24,
) -> Optional[dict]:
    """Search Reddit for recent mentions of *ticker* using the public JSON API.

    No API key needed — uses reddit.com/r/{sub}/search.json.
    Returns dict with keys: rd_mentions, rd_posts, rd_comments, rd_top_sub, rd_sentiment
    or None on complete failure.
    """
    subs = subreddits or REDDIT_SUBREDDITS
    sym = ticker.upper()
    pat = re.compile(rf"(?<![A-Za-z$])(?:\$)?{re.escape(sym)}(?![A-Za-z])", re.IGNORECASE)

    cutoff = time.time() - (lookback_hours * 3600)
    post_count = 0
    comment_count = 0
    sub_counts: dict[str, int] = {}
    bullish_hits = 0
    bearish_hits = 0
    headers = {"User-Agent": REDDIT_USER_AGENT}

    for sub_name in subs:
        url = f"https://www.reddit.com/r/{sub_name}/search.json"
        params = {
            "q": sym,
            "sort": "new",
            "t": "day",
            "restrict_sr": "on",
            "limit": 50,
        }
        try:
            time.sleep(_RD_RATE_DELAY)
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 429:
                print(f"  [sentiment] Reddit rate-limited on r/{sub_name}")
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception as exc:
            print(f"  [sentiment] Reddit r/{sub_name} error for {ticker}: {exc}")
            continue

        children = (data.get("data") or {}).get("children") or []
        for child in children:
            post = child.get("data") or {}
            created = post.get("created_utc", 0)
            if created < cutoff:
                continue

            title = post.get("title") or ""
            selftext = post.get("selftext") or ""
            if not pat.search(title) and not pat.search(selftext):
                continue

            post_count += 1
            sub_counts[sub_name] = sub_counts.get(sub_name, 0) + 1

            flair = (post.get("link_flair_text") or "").lower()
            if flair in _BULLISH_FLAIRS:
                bullish_hits += 1
            elif flair in _BEARISH_FLAIRS:
                bearish_hits += 1

            num_comments = post.get("num_comments", 0)
            comment_count += min(num_comments, 20)

    total_mentions = post_count + comment_count
    top_sub = max(sub_counts, key=sub_counts.get) if sub_counts else "—"

    if bullish_hits > bearish_hits:
        lean = "bullish"
    elif bearish_hits > bullish_hits:
        lean = "bearish"
    else:
        lean = "neutral"

    return {
        "rd_mentions": total_mentions,
        "rd_posts": post_count,
        "rd_comments": comment_count,
        "rd_top_sub": top_sub,
        "rd_sentiment": lean,
    }

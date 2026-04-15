"""Market-wide dark pool print aggregation, classification, and multi-day tracking."""

from __future__ import annotations

import csv
import glob as _glob
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.config import DP_TRACKER_LOOKBACK_DAYS, DP_TRACKER_MIN_ACTIVE_DAYS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DP_SNAPSHOTS_PATH = DATA_DIR / "dp_snapshots.csv"

# Intra-day accumulation: stores all unique prints seen today, keyed by tracking_id.
_DAILY_PREFIX = "dark_pool_daily_"
_scan_counter_path = DATA_DIR / "dp_scan_count.txt"

DP_SNAPSHOT_COLS = [
    "snapshot_date",
    "ticker",
    "total_notional",
    "total_volume",
    "print_count",
    "large_print_count",
    "buy_volume",
    "sell_volume",
    "bias",
    "largest_print_notional",
    "marketcap",
    "sector",
]


def classify_print(row: dict) -> dict:
    """Classify a single dark pool print as buy/sell/neutral and enrich it."""
    try:
        price = float(row.get("price", 0) or 0)
        size = float(row.get("size", 0) or 0)
        nbbo_ask = float(row.get("nbbo_ask", 0) or 0)
        nbbo_bid = float(row.get("nbbo_bid", 0) or 0)
        premium = float(row.get("premium", 0) or 0)
    except (TypeError, ValueError):
        return {}

    if price <= 0 or size <= 0:
        return {}

    notional = price * size

    if nbbo_ask > 0 and price >= nbbo_ask:
        side = "buy"
    elif nbbo_bid > 0 and price <= nbbo_bid:
        side = "sell"
    else:
        side = "neutral"

    ticker = (row.get("ticker") or "").upper().strip()
    executed_at = row.get("executed_at", "")
    try:
        ts = datetime.fromisoformat(executed_at.replace("Z", "+00:00"))
        time_str = ts.astimezone(timezone.utc).strftime("%H:%M:%S")
    except Exception:
        time_str = executed_at

    return {
        "ticker": ticker,
        "price": price,
        "size": size,
        "notional": notional,
        "premium": premium,
        "side": side,
        "nbbo_bid": nbbo_bid,
        "nbbo_ask": nbbo_ask,
        "executed_at": executed_at,
        "time_str": time_str,
        "volume": float(row.get("volume", 0) or 0),
    }


def aggregate_dark_pool_prints(
    raw_prints: list[dict],
    screener_meta: dict[str, dict] | None = None,
) -> dict:
    """Aggregate raw dark pool prints into per-ticker summaries and top prints.

    Parameters
    ----------
    raw_prints : list[dict]
        Raw response from ``fetch_dark_pool_recent()``.
    screener_meta : dict, optional
        Ticker -> screener row mapping (used for market cap enrichment).

    Returns
    -------
    dict with keys:
        top_prints  – list of individual large prints ($1M+ notional)
        by_ticker   – list of per-ticker aggregates sorted by notional desc
        by_mcap     – same list sorted by notional/mcap bps desc
    """
    screener_meta = screener_meta or {}

    classified: list[dict] = []
    for row in raw_prints:
        c = classify_print(row)
        if c and c["ticker"]:
            classified.append(c)

    if not classified:
        return {"top_prints": [], "by_ticker": [], "by_mcap": []}

    top_prints = sorted(
        [p for p in classified if p["notional"] >= 1_000_000],
        key=lambda p: p["notional"],
        reverse=True,
    )

    ticker_agg: dict[str, dict] = {}
    for p in classified:
        t = p["ticker"]
        if t not in ticker_agg:
            ticker_agg[t] = {
                "ticker": t,
                "total_volume": 0.0,
                "total_notional": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "print_count": 0,
                "large_print_count": 0,
                "largest_print_notional": 0.0,
            }
        a = ticker_agg[t]
        a["total_volume"] += p["size"]
        a["total_notional"] += p["notional"]
        a["print_count"] += 1
        if p["notional"] >= 1_000_000:
            a["large_print_count"] += 1
        if p["notional"] > a["largest_print_notional"]:
            a["largest_print_notional"] = p["notional"]
        if p["side"] == "buy":
            a["buy_volume"] += p["size"]
        elif p["side"] == "sell":
            a["sell_volume"] += p["size"]

    results: list[dict] = []
    for a in ticker_agg.values():
        tv = a["total_volume"]
        bias = a["buy_volume"] / tv if tv > 0 else 0.5
        a["bias"] = round(bias, 4)

        if bias >= 0.60:
            a["bias_label"] = "Buyers"
        elif bias <= 0.40:
            a["bias_label"] = "Sellers"
        else:
            a["bias_label"] = "Balanced"

        a["total_notional"] = round(a["total_notional"], 2)
        a["largest_print_notional"] = round(a["largest_print_notional"], 2)
        a["total_volume"] = round(a["total_volume"])

        sm = screener_meta.get(a["ticker"], {})
        mcap = 0.0
        try:
            mcap = float(sm.get("marketcap") or 0)
        except (TypeError, ValueError):
            pass
        a["marketcap"] = mcap

        if mcap > 0:
            a["notional_mcap_bps"] = round(a["total_notional"] / mcap * 10_000, 2)
        else:
            a["notional_mcap_bps"] = 0.0

        a["sector"] = sm.get("sector") or "—"
        results.append(a)

    by_notional = sorted(results, key=lambda x: x["total_notional"], reverse=True)
    by_mcap = sorted(
        [r for r in results if r["notional_mcap_bps"] > 0],
        key=lambda x: x["notional_mcap_bps"],
        reverse=True,
    )

    return {
        "top_prints": top_prints,
        "by_ticker": by_notional,
        "by_mcap": by_mcap,
    }


# ---------------------------------------------------------------------------
# Intra-day accumulation (deduped across multiple scans)
# ---------------------------------------------------------------------------


def _daily_path(d: date | None = None) -> Path:
    d = d or date.today()
    return DATA_DIR / f"{_DAILY_PREFIX}{d.isoformat()}.json"


def accumulate_daily_prints(new_prints: list[dict]) -> list[dict]:
    """Merge *new_prints* into today's accumulated file, deduped by tracking_id.

    Returns the full deduplicated list of prints for the day so far.
    Also increments a simple scan counter so the dashboard can report how many
    scans contributed to today's data.
    """
    if not new_prints:
        return load_daily_accumulated()

    today = date.today()
    path = _daily_path(today)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prune files from previous days (keep today only)
    for old in DATA_DIR.glob(f"{_DAILY_PREFIX}*.json"):
        if old != path:
            try:
                old.unlink()
            except OSError:
                pass

    existing: dict[int | str, dict] = {}
    if path.exists():
        try:
            for row in json.loads(path.read_text()):
                tid = row.get("tracking_id")
                if tid is not None:
                    existing[tid] = row
        except Exception:
            existing = {}

    before = len(existing)
    for row in new_prints:
        tid = row.get("tracking_id")
        if tid is not None and tid not in existing:
            existing[tid] = row

    all_prints = list(existing.values())
    path.write_text(json.dumps(all_prints, default=str))

    added = len(existing) - before
    print(f"  [dp-daily] {added} new prints merged → {len(all_prints)} total today")

    # Bump scan counter
    _increment_scan_counter(today)

    return all_prints


def load_daily_accumulated() -> list[dict]:
    """Read today's accumulated dark pool prints (for dashboard use)."""
    path = _daily_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_daily_scan_count() -> int:
    """How many scans contributed to today's accumulated data."""
    today_str = date.today().isoformat()
    if not _scan_counter_path.exists():
        return 0
    try:
        parts = _scan_counter_path.read_text().strip().split(":")
        if parts[0] == today_str:
            return int(parts[1])
    except Exception:
        pass
    return 0


def _increment_scan_counter(d: date) -> None:
    today_str = d.isoformat()
    current = 0
    if _scan_counter_path.exists():
        try:
            parts = _scan_counter_path.read_text().strip().split(":")
            if parts[0] == today_str:
                current = int(parts[1])
        except Exception:
            pass
    _scan_counter_path.write_text(f"{today_str}:{current + 1}")


def aggregate_daily_accumulated(
    prints: list[dict],
    screener_meta: dict[str, dict] | None = None,
) -> dict:
    """Aggregate the full day's accumulated prints into per-ticker summaries.

    Returns a dict with:
        by_ticker   – list of per-ticker aggregates sorted by notional desc
        total_prints – total print count
        total_notional – sum of all notional
        scan_count  – number of scans that contributed
    """
    screener_meta = screener_meta or {}

    classified: list[dict] = []
    for row in prints:
        c = classify_print(row)
        if c and c["ticker"]:
            classified.append(c)

    if not classified:
        return {"by_ticker": [], "total_prints": 0, "total_notional": 0, "scan_count": get_daily_scan_count()}

    ticker_agg: dict[str, dict] = {}
    for p in classified:
        t = p["ticker"]
        if t not in ticker_agg:
            ticker_agg[t] = {
                "ticker": t,
                "total_volume": 0.0,
                "total_notional": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "print_count": 0,
                "large_print_count": 0,
                "largest_print_notional": 0.0,
                "first_seen": p["executed_at"],
                "last_seen": p["executed_at"],
                "scan_appearances": 0,
            }
        a = ticker_agg[t]
        a["total_volume"] += p["size"]
        a["total_notional"] += p["notional"]
        a["print_count"] += 1
        if p["notional"] >= 1_000_000:
            a["large_print_count"] += 1
        if p["notional"] > a["largest_print_notional"]:
            a["largest_print_notional"] = p["notional"]
        if p["side"] == "buy":
            a["buy_volume"] += p["size"]
        elif p["side"] == "sell":
            a["sell_volume"] += p["size"]

        ts = p["executed_at"]
        if ts and ts < a["first_seen"]:
            a["first_seen"] = ts
        if ts and ts > a["last_seen"]:
            a["last_seen"] = ts

    results: list[dict] = []
    grand_notional = 0.0
    for a in ticker_agg.values():
        tv = a["total_volume"]
        bias = a["buy_volume"] / tv if tv > 0 else 0.5
        a["bias"] = round(bias, 4)

        if bias >= 0.60:
            a["bias_label"] = "Buyers"
        elif bias <= 0.40:
            a["bias_label"] = "Sellers"
        else:
            a["bias_label"] = "Balanced"

        a["total_notional"] = round(a["total_notional"], 2)
        a["largest_print_notional"] = round(a["largest_print_notional"], 2)
        a["total_volume"] = round(a["total_volume"])
        grand_notional += a["total_notional"]

        sm = screener_meta.get(a["ticker"], {})
        mcap = 0.0
        try:
            mcap = float(sm.get("marketcap") or 0)
        except (TypeError, ValueError):
            pass
        a["marketcap"] = mcap

        if mcap > 0:
            a["notional_mcap_bps"] = round(a["total_notional"] / mcap * 10_000, 2)
        else:
            a["notional_mcap_bps"] = 0.0

        a["sector"] = sm.get("sector") or "—"

        # Format timestamps for display
        for k in ("first_seen", "last_seen"):
            try:
                ts = datetime.fromisoformat(a[k].replace("Z", "+00:00"))
                a[k + "_str"] = ts.strftime("%H:%M:%S")
            except Exception:
                a[k + "_str"] = a[k] or ""

        results.append(a)

    by_notional = sorted(results, key=lambda x: x["total_notional"], reverse=True)

    return {
        "by_ticker": by_notional,
        "total_prints": len(classified),
        "total_notional": round(grand_notional, 2),
        "scan_count": get_daily_scan_count(),
    }


# ---------------------------------------------------------------------------
# Multi-day persistence
# ---------------------------------------------------------------------------

def save_dp_snapshot(
    ticker_aggregates: list[dict],
    screener_meta: dict[str, dict] | None = None,
) -> None:
    """Persist today's per-ticker dark pool aggregates to the rolling snapshot CSV.

    Upserts: replaces any rows with today's date, appends new ones.
    Prunes rows older than the lookback window + buffer.
    """
    if not ticker_aggregates:
        return

    screener_meta = screener_meta or {}
    today_str = str(date.today())
    cutoff = str(date.today() - timedelta(days=DP_TRACKER_LOOKBACK_DAYS + 3))

    new_rows: list[dict] = []
    for agg in ticker_aggregates:
        ticker = (agg.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        sm = screener_meta.get(ticker, {})
        mcap = agg.get("marketcap", 0.0)
        if not mcap:
            try:
                mcap = float(sm.get("marketcap") or 0)
            except (TypeError, ValueError):
                mcap = 0.0

        new_rows.append({
            "snapshot_date": today_str,
            "ticker": ticker,
            "total_notional": round(agg.get("total_notional", 0), 2),
            "total_volume": round(agg.get("total_volume", 0)),
            "print_count": agg.get("print_count", 0),
            "large_print_count": agg.get("large_print_count", 0),
            "buy_volume": round(agg.get("buy_volume", 0)),
            "sell_volume": round(agg.get("sell_volume", 0)),
            "bias": round(agg.get("bias", 0.5), 4),
            "largest_print_notional": round(agg.get("largest_print_notional", 0), 2),
            "marketcap": mcap,
            "sector": agg.get("sector") or sm.get("sector") or "—",
        })

    if not new_rows:
        return

    existing: list[dict] = []
    if DP_SNAPSHOTS_PATH.exists():
        try:
            with open(DP_SNAPSHOTS_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    d = r.get("snapshot_date", "")
                    if d != today_str and d >= cutoff:
                        existing.append(r)
        except Exception:
            existing = []

    DP_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = existing + new_rows
    with open(DP_SNAPSHOTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DP_SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"  [dp-tracker] saved {len(new_rows)} ticker snapshots for {today_str} "
        f"({len(existing)} historical rows retained)"
    )


def compute_multi_day_dp(
    lookback_days: int = DP_TRACKER_LOOKBACK_DAYS,
    min_active_days: int = DP_TRACKER_MIN_ACTIVE_DAYS,
) -> list[dict]:
    """Aggregate dark pool snapshots over the lookback window.

    Returns a list of dicts (one per qualifying ticker) sorted by
    persistence-weighted notional/mcap score.
    """
    if not DP_SNAPSHOTS_PATH.exists():
        return []

    try:
        df = pd.read_csv(DP_SNAPSHOTS_PATH)
    except Exception:
        return []

    if df.empty or "snapshot_date" not in df.columns:
        return []

    cutoff = str(date.today() - timedelta(days=lookback_days))
    df = df[df["snapshot_date"] >= cutoff].copy()
    if df.empty:
        return []

    for col in ("total_notional", "total_volume", "buy_volume", "sell_volume",
                 "bias", "largest_print_notional", "marketcap",
                 "print_count", "large_print_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    all_dates = sorted(df["snapshot_date"].unique())
    total_days = len(all_dates)
    if total_days == 0:
        return []

    results: list[dict] = []

    for ticker, grp in df.groupby("ticker"):
        active_dates = sorted(grp["snapshot_date"].unique())
        active_days = len(active_dates)
        if active_days < min_active_days:
            continue

        cum_notional = grp["total_notional"].sum()
        cum_volume = grp["total_volume"].sum()
        cum_buy = grp["buy_volume"].sum()
        cum_sell = grp["sell_volume"].sum()
        cum_prints = int(grp["print_count"].sum())
        cum_large = int(grp["large_print_count"].sum())
        max_print = grp["largest_print_notional"].max()

        # Volume-weighted bias
        vol_series = grp["total_volume"]
        bias_series = grp["bias"]
        total_vol = vol_series.sum()
        weighted_bias = (bias_series * vol_series).sum() / total_vol if total_vol > 0 else 0.5

        # Bias consistency: fraction of days that are clearly directional
        buy_days = int((bias_series > 0.55).sum())
        sell_days = int((bias_series < 0.45).sum())
        if buy_days >= (2 / 3) * active_days:
            bias_consistency_label = "Consistent buyer"
        elif sell_days >= (2 / 3) * active_days:
            bias_consistency_label = "Consistent seller"
        else:
            bias_consistency_label = "Mixed"

        # Bias label from weighted bias
        if weighted_bias >= 0.60:
            bias_label = "Buyers"
        elif weighted_bias <= 0.40:
            bias_label = "Sellers"
        else:
            bias_label = "Balanced"

        latest = grp.sort_values("snapshot_date").iloc[-1]
        mcap = float(latest.get("marketcap") or 0)

        if mcap > 0:
            notional_mcap_bps = round(cum_notional / mcap * 10_000, 2)
        else:
            notional_mcap_bps = 0.0

        persistence_ratio = round(active_days / total_days, 2)
        persistence_weighted = round(notional_mcap_bps * persistence_ratio, 4)

        # Trend: second-half vs first-half notional
        mid = len(active_dates) // 2
        if mid > 0:
            first_half = grp[grp["snapshot_date"].isin(active_dates[:mid])]
            second_half = grp[grp["snapshot_date"].isin(active_dates[mid:])]
            n1 = first_half["total_notional"].sum()
            n2 = second_half["total_notional"].sum()
            if n1 > 0:
                ratio = n2 / n1
                if ratio > 1.3:
                    trend = "accelerating"
                elif ratio < 0.7:
                    trend = "fading"
                else:
                    trend = "steady"
            else:
                trend = "accelerating" if n2 > 0 else "steady"
        else:
            trend = "steady"

        # Daily snapshots for sparkline
        daily_snaps: list[dict] = []
        for d in all_dates:
            day_row = grp[grp["snapshot_date"] == d]
            if day_row.empty:
                daily_snaps.append({
                    "date": d, "notional": 0, "buy_notional": 0,
                    "sell_notional": 0, "bias": 0.5, "active": False,
                })
            else:
                dr = day_row.iloc[0]
                day_not = float(dr.get("total_notional") or 0)
                day_bias = float(dr.get("bias") or 0.5)
                day_buy_vol = float(dr.get("buy_volume") or 0)
                day_sell_vol = float(dr.get("sell_volume") or 0)
                day_total_vol = day_buy_vol + day_sell_vol
                buy_frac = day_buy_vol / day_total_vol if day_total_vol > 0 else 0.5
                daily_snaps.append({
                    "date": d,
                    "notional": day_not,
                    "buy_notional": round(day_not * buy_frac, 2),
                    "sell_notional": round(day_not * (1 - buy_frac), 2),
                    "bias": day_bias,
                    "active": True,
                })

        results.append({
            "ticker": str(ticker),
            "sector": latest.get("sector") or "—",
            "bias": round(weighted_bias, 4),
            "bias_label": bias_label,
            "bias_consistency_label": bias_consistency_label,
            "buy_days": buy_days,
            "sell_days": sell_days,
            "cumulative_notional": round(cum_notional, 2),
            "cumulative_volume": round(cum_volume),
            "cumulative_prints": cum_prints,
            "cumulative_large_prints": cum_large,
            "largest_print": round(max_print, 2),
            "notional_mcap_bps": notional_mcap_bps,
            "marketcap": mcap,
            "active_days": active_days,
            "total_days": total_days,
            "persistence_ratio": persistence_ratio,
            "persistence_weighted": persistence_weighted,
            "trend": trend,
            "daily_snapshots": daily_snaps,
        })

    results.sort(key=lambda x: x["persistence_weighted"], reverse=True)
    return results

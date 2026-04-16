"""Multi-day flow tracker — persist UW screener snapshots and surface repeat unusual activity."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import FLOW_TRACKER_LOOKBACK_DAYS, FLOW_TRACKER_MIN_ACTIVE_DAYS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SNAPSHOTS_PATH = DATA_DIR / "screener_snapshots.csv"

SNAPSHOT_COLS = [
    "snapshot_date",
    "ticker",
    "sector",
    "close",
    "marketcap",
    "bullish_premium",
    "bearish_premium",
    "net_premium",
    "call_volume",
    "put_volume",
    "volume",
    "call_open_interest",
    "put_open_interest",
    "total_oi_change_perc",
    "call_oi_change_perc",
    "put_oi_change_perc",
    "put_call_ratio",
    "iv_rank",
    "iv30d",
    "perc_3_day_total",
    "perc_30_day_total",
]


def save_screener_snapshot(screener_data: list[dict]) -> None:
    """Persist today's screener response to the rolling snapshots CSV.

    Upserts: replaces any rows with today's date, appends new ones.
    Prunes rows older than the lookback window + buffer.
    """
    if not screener_data:
        return

    today_str = str(date.today())
    cutoff = str(date.today() - timedelta(days=FLOW_TRACKER_LOOKBACK_DAYS + 3))

    new_rows: list[dict] = []
    for sr in screener_data:
        ticker = (sr.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        row: dict = {"snapshot_date": today_str, "ticker": ticker}
        for col in SNAPSHOT_COLS[2:]:
            row[col] = sr.get(col)
        new_rows.append(row)

    if not new_rows:
        return

    existing: list[dict] = []
    if SNAPSHOTS_PATH.exists():
        try:
            with open(SNAPSHOTS_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    d = r.get("snapshot_date", "")
                    if d != today_str and d >= cutoff:
                        existing.append(r)
        except Exception:
            existing = []

    SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = existing + new_rows
    with open(SNAPSHOTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  [flow-tracker] saved {len(new_rows)} screener rows for {today_str} "
          f"({len(existing)} historical rows retained)")


def save_flow_feature_snapshot(feature_table: pd.DataFrame) -> None:
    """Merge flow-feature tickers into screener_snapshots.csv.

    The UW stock screener only returns ~30 tickers per day, missing many
    tickers that have clear unusual flow in the pipeline.  This function
    fills the gaps by appending flow-feature tickers that are **not already
    present** for today's date, using the metrics available from flow scoring.
    Screener data is richer, so it always takes priority.
    """
    if feature_table is None or feature_table.empty:
        return

    today_str = str(date.today())

    # Read existing today rows to find which tickers are already covered
    existing_today: set[str] = set()
    if SNAPSHOTS_PATH.exists():
        try:
            with open(SNAPSHOTS_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("snapshot_date") == today_str:
                        existing_today.add((r.get("ticker") or "").upper().strip())
        except Exception:
            pass

    new_rows: list[dict] = []
    for _, row in feature_table.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker or ticker in existing_today:
            continue

        bull_prem = float(row.get("bullish_premium", 0) or 0)
        bear_prem = float(row.get("bearish_premium", 0) or 0)
        mcap = float(row.get("marketcap", 0) or 0)

        new_rows.append({
            "snapshot_date": today_str,
            "ticker": ticker,
            "sector": None,
            "close": None,
            "marketcap": mcap if mcap > 0 else None,
            "bullish_premium": round(bull_prem, 2),
            "bearish_premium": round(bear_prem, 2),
            "net_premium": round(bull_prem - bear_prem, 2),
            "call_volume": None,
            "put_volume": None,
            "volume": row.get("total_count"),
            "call_open_interest": None,
            "put_open_interest": None,
            "total_oi_change_perc": None,
            "call_oi_change_perc": None,
            "put_oi_change_perc": None,
            "put_call_ratio": None,
            "iv_rank": None,
            "iv30d": None,
            "perc_3_day_total": None,
            "perc_30_day_total": None,
        })

    if not new_rows:
        return

    # Append to the existing file (screener snapshot already wrote today's rows)
    all_rows: list[dict] = []
    if SNAPSHOTS_PATH.exists():
        try:
            with open(SNAPSHOTS_PATH, "r", newline="") as f:
                all_rows = list(csv.DictReader(f))
        except Exception:
            all_rows = []

    all_rows.extend(new_rows)
    SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  [flow-tracker] merged {len(new_rows)} flow-feature tickers "
          f"(skipped {len(existing_today)} already from screener)")


def compute_multi_day_flow(
    lookback_days: int = FLOW_TRACKER_LOOKBACK_DAYS,
    min_active_days: int = FLOW_TRACKER_MIN_ACTIVE_DAYS,
) -> list[dict]:
    """Aggregate screener snapshots over the lookback window.

    Returns a list of dicts (one per qualifying ticker) sorted by
    persistence-weighted prem/mcap score.  Only tickers with flow on
    >= min_active_days distinct days are included.
    """
    if not SNAPSHOTS_PATH.exists():
        return []

    try:
        df = pd.read_csv(SNAPSHOTS_PATH)
    except Exception:
        return []

    if df.empty or "snapshot_date" not in df.columns:
        return []

    cutoff = str(date.today() - timedelta(days=lookback_days))
    df = df[df["snapshot_date"] >= cutoff].copy()
    if df.empty:
        return []

    for col in ("bullish_premium", "bearish_premium", "net_premium",
                 "marketcap", "close", "iv_rank", "iv30d",
                 "total_oi_change_perc", "call_oi_change_perc", "put_oi_change_perc",
                 "put_call_ratio", "perc_3_day_total", "perc_30_day_total",
                 "call_volume", "put_volume", "volume",
                 "call_open_interest", "put_open_interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

        cum_bull = grp["bullish_premium"].sum()
        cum_bear = grp["bearish_premium"].sum()
        cum_net = grp["net_premium"].sum()
        cum_total = cum_bull + cum_bear

        latest = grp.sort_values("snapshot_date").iloc[-1]
        mcap = float(latest.get("marketcap") or 0)

        if cum_total > 0 and mcap > 0:
            dominant_prem = max(cum_bull, cum_bear)
            prem_mcap_bps = round(dominant_prem / mcap * 10_000, 2)
        else:
            prem_mcap_bps = 0.0

        persistence_ratio = round(active_days / total_days, 2)
        persistence_weighted = round(prem_mcap_bps * persistence_ratio, 4)

        # Flow trend: compare second half vs first half of the window
        mid = len(active_dates) // 2
        if mid > 0:
            first_half = grp[grp["snapshot_date"].isin(active_dates[:mid])]
            second_half = grp[grp["snapshot_date"].isin(active_dates[mid:])]
            p1 = first_half["bullish_premium"].sum() + first_half["bearish_premium"].sum()
            p2 = second_half["bullish_premium"].sum() + second_half["bearish_premium"].sum()
            if p1 > 0:
                ratio = p2 / p1
                if ratio > 1.3:
                    trend = "accelerating"
                elif ratio < 0.7:
                    trend = "fading"
                else:
                    trend = "steady"
            else:
                trend = "accelerating" if p2 > 0 else "steady"
        else:
            trend = "steady"

        # Daily snapshots for sparkline
        daily_snaps: list[dict] = []
        for d in all_dates:
            day_row = grp[grp["snapshot_date"] == d]
            if day_row.empty:
                daily_snaps.append({"date": d, "premium": 0, "active": False})
            else:
                dr = day_row.iloc[0]
                dp = float(dr.get("bullish_premium") or 0) + float(dr.get("bearish_premium") or 0)
                daily_snaps.append({"date": d, "premium": dp, "active": True})

        direction = "BULLISH" if cum_bull >= cum_bear else "BEARISH"
        bull_pct = round(cum_bull / cum_total * 100, 1) if cum_total > 0 else 50.0

        avg_vol_30d = grp["perc_30_day_total"].mean()
        if pd.isna(avg_vol_30d):
            avg_vol_30d = 0.0

        results.append({
            "ticker": ticker,
            "sector": latest.get("sector") or "—",
            "direction": direction,
            "bull_pct": bull_pct,
            "cumulative_premium": round(cum_total, 2),
            "cumulative_bull": round(cum_bull, 2),
            "cumulative_bear": round(cum_bear, 2),
            "cumulative_net": round(cum_net, 2),
            "prem_mcap_bps": prem_mcap_bps,
            "active_days": active_days,
            "total_days": total_days,
            "persistence_ratio": persistence_ratio,
            "persistence_weighted": persistence_weighted,
            "trend": trend,
            "avg_vol_ratio_30d": round(float(avg_vol_30d), 2),
            "latest_oi_change": round(float(latest.get("total_oi_change_perc") or 0), 2),
            "latest_iv_rank": round(float(latest.get("iv_rank") or 0), 1),
            "latest_close": round(float(latest.get("close") or 0), 2),
            "latest_put_call_ratio": round(float(latest.get("put_call_ratio") or 0), 2),
            "marketcap": mcap,
            "daily_snapshots": daily_snaps,
        })

    results.sort(key=lambda x: x["persistence_weighted"], reverse=True)
    return results

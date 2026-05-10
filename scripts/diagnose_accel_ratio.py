"""Phase 1 diagnostic for the ``accel_ratio_today`` 93% zero share.

Reads the most recent ``data/raw_flow/raw_flow_*.csv`` files, today's
``data/screener_snapshots.csv`` slice, and ``data/grade_history.csv``
to characterize whether the zeros are "true zeros from the global
``_now_ref`` window" or a persistence/merge bug.

Run: ``.venv/bin/python scripts/diagnose_accel_ratio.py``
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RAW_DIR = DATA / "raw_flow"
SNAP_PATH = DATA / "screener_snapshots.csv"
GH_PATH = DATA / "grade_history.csv"


def _fmt_pct(num: int, den: int) -> str:
    if not den:
        return "0/0 (—)"
    return f"{num}/{den} ({num/den*100:.1f}%)"


def _summarize_event_ts(df: pd.DataFrame, label: str) -> None:
    if "event_ts" not in df.columns:
        print(f"  [{label}] event_ts column missing")
        return
    ts = pd.to_datetime(df["event_ts"], errors="coerce", utc=True)
    nn = ts.notna().sum()
    print(f"  [{label}] rows={len(df)}, event_ts non-null={_fmt_pct(int(nn), len(df))}")
    if nn == 0:
        return
    print(f"    min={ts.min()}  max={ts.max()}  span={ts.max() - ts.min()}")
    # Hourly histogram of last 12h.
    nowref = ts.max()
    bins = []
    for hours_back in (1, 2, 4, 8, 12, 24):
        cutoff = nowref - pd.Timedelta(hours=hours_back)
        n_in = int(((ts > cutoff) & ts.notna()).sum())
        bins.append((hours_back, n_in))
    print("    rows-within-Nh-of-max:")
    for h, n in bins:
        print(f"      last {h:>2}h: {_fmt_pct(n, int(nn))}")


def _per_ticker_window_audit(df: pd.DataFrame) -> None:
    """Compare global vs per-ticker last-2h window coverage."""
    if df.empty or "ticker" not in df.columns or "event_ts" not in df.columns:
        return
    work = df.copy()
    work["event_ts"] = pd.to_datetime(work["event_ts"], errors="coerce", utc=True)
    work = work[work["event_ts"].notna()]
    if work.empty:
        return
    if "direction" not in work.columns:
        print("    [audit] direction column missing; skipping per-side breakdown")
        return

    # Global last-2h window.
    g_max = work["event_ts"].max()
    g_cutoff = g_max - pd.Timedelta(hours=2)
    work["in_global_2h"] = work["event_ts"] >= g_cutoff

    # Per-ticker last-2h window.
    work["t_max"] = work.groupby("ticker")["event_ts"].transform("max")
    work["in_ticker_2h"] = (work["t_max"] - work["event_ts"]) <= pd.Timedelta(hours=2)

    by_ticker = work.groupby("ticker").agg(
        n_total=("event_ts", "size"),
        n_long=("direction", lambda s: int((s == "LONG").sum())),
        n_short=("direction", lambda s: int((s == "SHORT").sum())),
        n_long_global_2h=("direction", lambda s: int(((s == "LONG") & work.loc[s.index, "in_global_2h"]).sum())),
        n_long_ticker_2h=("direction", lambda s: int(((s == "LONG") & work.loc[s.index, "in_ticker_2h"]).sum())),
        n_short_global_2h=("direction", lambda s: int(((s == "SHORT") & work.loc[s.index, "in_global_2h"]).sum())),
        n_short_ticker_2h=("direction", lambda s: int(((s == "SHORT") & work.loc[s.index, "in_ticker_2h"]).sum())),
    )

    n_tickers = len(by_ticker)
    print(f"    tickers in batch: {n_tickers}")

    # Bullish coverage stats.
    bull_active = by_ticker[by_ticker["n_long"] > 0]
    print(f"    LONG-active tickers: {len(bull_active)}")
    bull_global_zero = (bull_active["n_long_global_2h"] == 0).sum()
    bull_ticker_zero = (bull_active["n_long_ticker_2h"] == 0).sum()
    would_help_bull = ((bull_active["n_long_global_2h"] == 0) & (bull_active["n_long_ticker_2h"] > 0)).sum()
    print(f"      bullish_accel_ratio = 0 with global _now_ref: "
          f"{_fmt_pct(int(bull_global_zero), len(bull_active))}")
    print(f"      bullish_accel_ratio = 0 with per-ticker _now_ref: "
          f"{_fmt_pct(int(bull_ticker_zero), len(bull_active))}")
    print(f"      tickers that switch from 0 -> non-zero (Option A wins): "
          f"{_fmt_pct(int(would_help_bull), len(bull_active))}")

    # Bearish coverage stats.
    bear_active = by_ticker[by_ticker["n_short"] > 0]
    print(f"    SHORT-active tickers: {len(bear_active)}")
    bear_global_zero = (bear_active["n_short_global_2h"] == 0).sum()
    bear_ticker_zero = (bear_active["n_short_ticker_2h"] == 0).sum()
    would_help_bear = ((bear_active["n_short_global_2h"] == 0) & (bear_active["n_short_ticker_2h"] > 0)).sum()
    print(f"      bearish_accel_ratio = 0 with global _now_ref: "
          f"{_fmt_pct(int(bear_global_zero), len(bear_active))}")
    print(f"      bearish_accel_ratio = 0 with per-ticker _now_ref: "
          f"{_fmt_pct(int(bear_ticker_zero), len(bear_active))}")
    print(f"      tickers that switch from 0 -> non-zero (Option A wins): "
          f"{_fmt_pct(int(would_help_bear), len(bear_active))}")


def diagnose_raw_flow_files(n_recent: int = 5) -> None:
    print("=" * 78)
    print(f"PHASE 1A — raw_flow files (last {n_recent})")
    print("=" * 78)
    files = sorted(RAW_DIR.glob("raw_flow_*.csv"))
    if not files:
        print("No raw_flow files found.")
        return
    for path in files[-n_recent:]:
        print(f"\n  --- {path.name} ---")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  read failed: {e}")
            continue
        _summarize_event_ts(df, "raw_flow")
        _per_ticker_window_audit(df)


def diagnose_screener_snapshots() -> None:
    print("\n" + "=" * 78)
    print("PHASE 1B — screener_snapshots.csv (today's slice)")
    print("=" * 78)
    if not SNAP_PATH.exists():
        print("Missing screener_snapshots.csv")
        return
    df = pd.read_csv(SNAP_PATH)
    if "snapshot_date" not in df.columns:
        print("snapshot_date column missing")
        return
    latest_date = df["snapshot_date"].max()
    today = df[df["snapshot_date"] == latest_date].copy()
    print(f"  latest snapshot_date: {latest_date}, rows: {len(today)}")
    for col in ("bullish_accel_ratio", "bearish_accel_ratio",
                "sweep_share", "multileg_share", "dominant_dte_bucket"):
        if col not in today.columns:
            print(f"    [missing column] {col}")
            continue
        nn = today[col].notna().sum()
        # Categorical / string column — show value counts only.
        if today[col].dtype == object:
            counts = today[col].fillna("NULL").value_counts().head(5).to_dict()
            print(f"    {col:30s} populated={_fmt_pct(int(nn), len(today))} top={counts}")
            continue
        as_num = pd.to_numeric(today[col], errors="coerce")
        zeros = (as_num.fillna(0) == 0).sum()
        nonzero = as_num[as_num.fillna(0) > 0]
        nonzero_mean = float(nonzero.mean()) if len(nonzero) else 0.0
        print(f"    {col:30s} populated={_fmt_pct(int(nn), len(today))} "
              f"zero/empty={_fmt_pct(int(zeros), len(today))} "
              f"non-zero mean={nonzero_mean:.4f} (n={len(nonzero)})")


def diagnose_grade_history() -> None:
    print("\n" + "=" * 78)
    print("PHASE 1C — grade_history.csv chain end-to-end")
    print("=" * 78)
    if not GH_PATH.exists():
        print("Missing grade_history.csv")
        return
    gh = pd.read_csv(GH_PATH)
    latest = gh["as_of"].max()
    today = gh[gh["as_of"] == latest].copy()
    print(f"  latest as_of: {latest}, rows: {len(today)}")
    if "accel_ratio_today" not in today.columns:
        print("  accel_ratio_today column missing")
        return
    zero_share = (today["accel_ratio_today"].fillna(0) == 0).mean() * 100
    print(f"  accel_ratio_today zero share: {zero_share:.1f}%")
    nonzero = today[today["accel_ratio_today"].fillna(0) > 0]
    print(f"  rows w/ non-zero accel_ratio_today: {len(nonzero)}/{len(today)}")
    if len(nonzero):
        cols = ["ticker", "direction", "accel_ratio_today",
                "cumulative_premium", "dominant_dte_bucket"]
        cols = [c for c in cols if c in nonzero.columns]
        print(nonzero[cols].to_string(index=False))

    # Cross-reference with screener_snapshots latest persisted accel ratio
    # to confirm the read-back chain is intact.
    if SNAP_PATH.exists():
        snap = pd.read_csv(SNAP_PATH)
        if "snapshot_date" in snap.columns:
            snap_latest = snap[snap["snapshot_date"] == snap["snapshot_date"].max()].copy()
            join_cols = ["ticker"]
            if not nonzero.empty and not snap_latest.empty:
                merged = nonzero.merge(
                    snap_latest[join_cols + [c for c in ("bullish_accel_ratio", "bearish_accel_ratio")
                                              if c in snap_latest.columns]],
                    on="ticker", how="left",
                )
                print("\n  cross-check (nonzero grade_history rows ↔ persisted snapshot accel ratios):")
                show_cols = ["ticker", "direction", "accel_ratio_today",
                             "bullish_accel_ratio", "bearish_accel_ratio"]
                show_cols = [c for c in show_cols if c in merged.columns]
                print(merged[show_cols].to_string(index=False))


def main() -> int:
    print("\nDiagnostic run: accel_ratio_today")
    print(f"  workspace: {ROOT}")
    print(f"  now (UTC): {datetime.now(timezone.utc).isoformat()}")
    diagnose_raw_flow_files()
    diagnose_screener_snapshots()
    diagnose_grade_history()
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        "Look at the 'tickers that switch from 0 -> non-zero (Option A wins)'\n"
        "lines under PHASE 1A. If those values are large fractions of the\n"
        "active-ticker base, the per-ticker _now_ref fix (Option A) is the\n"
        "correct intervention. If small, the zeros are 'real' (no recent\n"
        "activity in any window) and Option B is needed.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

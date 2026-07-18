"""Backfill aggressor-signed premium features on ``feature_lab.csv``.

The four aggressor columns (``aggressor_bull_share``,
``aggressor_net_prem_bps``, ``ask_side_ratio``, ``directional_sweep_share``)
are computed from the per-print ``data/raw_flow/raw_flow_*.csv`` archive.

For each historical ``feature_lab`` row we locate the *last* raw_flow
snapshot dated the same calendar day (matching the backtest engine's
"latest snapshot per day wins" convention in
``app/backtest/engine.py::_load_flow_snapshots``), slice it to the row's
ticker, and recompute the four features. This lets the momentum-score
head-to-head run on a fully populated column instead of waiting weeks for
it to accrue live.

Usage::

    python scripts/backfill_aggressor_features.py            # dry-run report
    python scripts/backfill_aggressor_features.py --apply     # writes the file
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import feature_lab as fl  # noqa: E402

FEATURE_LAB = fl.FEATURE_LAB_PATH
RAW_FLOW_DIR = fl.RAW_FLOW_DIR
AGG_COLS = fl.AGGRESSOR_FEATURE_COLS


def _fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(n / d * 100 if d else 0):.1f}%)"


def _latest_raw_flow_by_day() -> dict[str, Path]:
    """Map ``YYYY-MM-DD`` -> latest raw_flow path for that calendar day."""
    by_day: dict[str, Path] = {}
    if not RAW_FLOW_DIR.exists():
        return by_day
    for path in sorted(RAW_FLOW_DIR.glob("raw_flow_*.csv")):
        stem = path.stem.replace("raw_flow_", "")
        if len(stem) < 8 or not stem[:8].isdigit():
            continue
        day = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
        by_day[day] = path  # sorted -> later timestamp overwrites (last wins)
    return by_day


def _populated(seq) -> int:
    n = 0
    for v in seq:
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none"}:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write feature_lab.csv (default: dry-run)")
    args = ap.parse_args()

    if not FEATURE_LAB.exists():
        print(f"ERROR: {FEATURE_LAB} not found")
        return 1

    with open(FEATURE_LAB, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("feature_lab.csv is empty; nothing to backfill")
        return 0
    print(f"Loaded {len(rows)} feature_lab rows")

    by_day = _latest_raw_flow_by_day()
    if not by_day:
        print(f"ERROR: no raw_flow files under {RAW_FLOW_DIR}")
        return 1
    print(f"Raw-flow archive: {len(by_day)} distinct days, "
          f"{min(by_day)} .. {max(by_day)}")

    # Per-day cache: day -> {ticker(upper) -> feature dict}. Loaded lazily.
    day_features: dict[str, dict[str, dict]] = {}
    missing_days: set[str] = set()

    def _features_for(day: str, ticker: str) -> dict | None:
        if day in missing_days:
            return None
        if day not in day_features:
            path = by_day.get(day)
            if path is None:
                missing_days.add(day)
                return None
            try:
                df = pd.read_csv(path)
            except Exception:
                missing_days.add(day)
                return None
            cache: dict[str, dict] = {}
            if not df.empty and "ticker" in df.columns:
                up = df["ticker"].astype(str).str.upper().str.strip()
                for tk, sub in df.groupby(up):
                    cache[tk] = fl._aggressor_signed_features(sub)
            day_features[day] = cache
        return day_features[day].get(ticker)

    before = {c: _populated(r.get(c) for r in rows) for c in AGG_COLS}

    changed = 0
    for r in rows:
        ticker = str(r.get("ticker") or "").upper().strip()
        as_of = str(r.get("as_of") or "").strip()[:10]
        if not ticker or not as_of:
            continue
        feats = _features_for(as_of, ticker)
        if feats is None:
            continue
        row_changed = False
        for c in AGG_COLS:
            v = feats.get(c)
            new = "" if v is None else round(float(v), 8)
            old = str(r.get(c) or "").strip()
            if str(new) != old and not (old == "" and new == ""):
                row_changed = True
            r[c] = new
        if row_changed:
            changed += 1

    after = {c: _populated(r.get(c) for r in rows) for c in AGG_COLS}

    print("\n--- backfill summary ---")
    for c in AGG_COLS:
        print(f"{c:26s} BEFORE {_fmt_pct(before[c], len(rows)):>18s}  "
              f"AFTER {_fmt_pct(after[c], len(rows))}")
    print(f"rows changed: {changed}")
    if missing_days:
        print(f"days with no raw_flow match: {len(missing_days)} "
              f"({', '.join(sorted(missing_days)[:6])}{'...' if len(missing_days) > 6 else ''})")

    if not args.apply:
        print("\nDRY-RUN — re-run with --apply to write feature_lab.csv")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = FEATURE_LAB.with_suffix(f".csv.bak.{ts}")
    shutil.copy2(FEATURE_LAB, backup)
    print(f"\nBacked up to {backup.name}")

    fieldnames = list(fl.LAB_COLS)
    with open(FEATURE_LAB, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {len(rows)} rows to {FEATURE_LAB.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

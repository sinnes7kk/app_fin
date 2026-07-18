"""Backfill ``prem_momentum_z3d`` on existing ``feature_lab.csv`` rows.

``prem_momentum_z3d`` used to draw its trailing baseline from
``grade_history.csv`` (graded tickers only, ~300 names), so it only
populated for the ~21% of rows whose ticker had accumulated three graded
days.  It now sources the baseline from the screener snapshot universe
(2000+ tickers) via ``feature_lab.load_screener_premium_history``.

This script recomputes the column for every historical row using the
full ``snapshots_archive.csv.gz`` history so the panel is internally
consistent for the next Spearman ranking — no need to wait weeks for the
column to re-accrue live.

Usage::

    python scripts/backfill_prem_momentum.py            # dry-run report
    python scripts/backfill_prem_momentum.py --apply    # writes the file
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


def _fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(n / d * 100 if d else 0):.1f}%)"


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

    # Full-history premium series (live rolling file + gzip archive).
    hist = fl.load_screener_premium_history(days=None, include_archive=True)
    if hist is None or hist.empty:
        print("ERROR: no screener premium history available (archive + live both empty)")
        return 1
    print(
        f"Premium history: {len(hist)} (ticker, day) rows, "
        f"{hist['ticker'].nunique()} tickers, "
        f"{hist['day'].min().date()} .. {hist['day'].max().date()}"
    )

    def _populated(seq) -> int:
        n = 0
        for v in seq:
            s = str(v).strip()
            if s and s.lower() not in {"nan", "none"}:
                n += 1
        return n

    before = _populated(r.get("prem_momentum_z3d") for r in rows)

    changed = 0
    new_populated = 0
    for r in rows:
        ticker = str(r.get("ticker") or "").upper().strip()
        as_of = r.get("as_of")
        if not ticker or not as_of:
            continue
        z = fl._prem_momentum_z3d(ticker, as_of, hist)
        old = str(r.get("prem_momentum_z3d") or "").strip()
        new = "" if z is None else repr(round(z, 6))
        if new:
            new_populated += 1
        if old != new and not (old == "" and new == ""):
            changed += 1
        r["prem_momentum_z3d"] = "" if z is None else round(z, 6)

    print("\n--- backfill summary ---")
    print(f"prem_momentum_z3d populated BEFORE: {_fmt_pct(before, len(rows))}")
    print(f"prem_momentum_z3d populated AFTER:  {_fmt_pct(new_populated, len(rows))}")
    print(f"rows changed: {changed}")

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

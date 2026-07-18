"""Backfill the shadow ``momentum_score`` on ``feature_lab.csv``.

The momentum score is cross-sectional: it ranks each candidate against
the *other candidates on the same ``as_of`` day*. This script recomputes
it per day over the full history so the walk-forward shadow report has a
populated column to evaluate immediately (no need to wait weeks).

Run this AFTER ``backfill_aggressor_features.py`` so the aggressor
inputs are present.

Usage::

    python scripts/backfill_momentum_score.py            # dry-run report
    python scripts/backfill_momentum_score.py --apply     # writes the file
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features import feature_lab as fl  # noqa: E402
from app.features.momentum_score import compute_day_scores  # noqa: E402

FEATURE_LAB = fl.FEATURE_LAB_PATH


def _fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({(n / d * 100 if d else 0):.1f}%)"


def _populated(seq) -> int:
    n = 0
    for v in seq:
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none"}:
            n += 1
    return n


def _num(v):
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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

    before = _populated(r.get("momentum_score") for r in rows)

    # Group row indices by as_of day.
    by_day: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        day = str(r.get("as_of") or "").strip()[:10]
        if day:
            by_day[day].append(i)

    changed = 0
    for day, idxs in by_day.items():
        day_rows = []
        for i in idxs:
            r = rows[i]
            dr = {"ticker": r.get("ticker"), "direction": r.get("direction")}
            for c in ("bullish_premium_share", "aggressor_bull_share", "ask_side_ratio",
                      "sector_relative_pct", "aggressor_net_prem_bps",
                      "directional_sweep_share", "dollar_delta_weighted_flow",
                      "realized_vol_regime", "far_otm_call_share", "far_otm_put_share"):
                dr[c] = _num(r.get(c))
            day_rows.append(dr)
        scores = compute_day_scores(day_rows)
        for pos, i in enumerate(idxs):
            sc = scores[pos]
            comp = sc.get("momentum_composite")
            score = sc.get("momentum_score")
            old = str(rows[i].get("momentum_score") or "").strip()
            new = "" if score is None else str(round(score, 4))
            if old != new and not (old == "" and new == ""):
                changed += 1
            rows[i]["momentum_composite"] = "" if comp is None else round(comp, 6)
            rows[i]["momentum_score"] = "" if score is None else round(score, 4)

    after = _populated(r.get("momentum_score") for r in rows)

    print("\n--- backfill summary ---")
    print(f"momentum_score populated BEFORE: {_fmt_pct(before, len(rows))}")
    print(f"momentum_score populated AFTER:  {_fmt_pct(after, len(rows))}")
    print(f"rows changed: {changed}")
    print(f"days scored: {len(by_day)}")

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

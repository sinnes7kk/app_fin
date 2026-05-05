"""Backfill ``dominant_dte_bucket`` on existing ``grade_history.csv`` rows.

The screener-source path persists rows without DTE info, leaving ~58% of
``data/grade_history.csv`` rows with ``dominant_dte_bucket`` null. This
script recovers what it can from two on-disk sources:

1. ``data/snapshots_archive.csv.gz`` — has DTE for per-ticker-API rows
2. ``data/flow_features/*.csv`` — has DTE for every row (full coverage)

Rows that still can't be recovered (typically: screener-only rows whose
ticker never made it into a per-ticker enrichment that day) are left
with ``dominant_dte_bucket = "unknown"`` — a documented fallback bucket
in the per-bucket exit configuration (Stage C).

Usage::

    python scripts/backfill_dte_bucket.py            # dry-run report
    python scripts/backfill_dte_bucket.py --apply    # writes the file
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
GRADE_HISTORY = DATA_DIR / "grade_history.csv"
SNAPSHOTS_ARCHIVE = DATA_DIR / "snapshots_archive.csv.gz"
FLOW_FEATURES_DIR = DATA_DIR / "flow_features"

UNKNOWN_BUCKET = "unknown"


def _files_by_date() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for p in sorted(FLOW_FEATURES_DIR.glob("*.csv")):
        m = re.match(r"flow_features_(\d{8})_", p.name)
        if not m:
            continue
        d = m.group(1)
        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        out.setdefault(iso, []).append(p)
    return out


def _build_archive_lookup() -> dict[tuple[str, str], str]:
    """Return {(snapshot_date, ticker): dominant_dte_bucket} from the archive."""
    if not SNAPSHOTS_ARCHIVE.exists():
        return {}
    df = pd.read_csv(SNAPSHOTS_ARCHIVE)
    if "dominant_dte_bucket" not in df.columns:
        return {}
    df = df[df["dominant_dte_bucket"].notna()]
    return {
        (str(r["snapshot_date"]), str(r["ticker"])): str(r["dominant_dte_bucket"])
        for _, r in df.iterrows()
    }


def _build_flow_features_lookup() -> dict[tuple[str, str], str]:
    """Walk every flow_features csv and build {(date, ticker): bucket}.

    When the same ticker appears in multiple snapshots for one date, prefer
    the latest non-null bucket.
    """
    out: dict[tuple[str, str], str] = {}
    for d, files in _files_by_date().items():
        for fp in files:
            try:
                df = pd.read_csv(fp, usecols=["ticker", "dominant_dte_bucket"])
            except Exception:
                continue
            if "dominant_dte_bucket" not in df.columns:
                continue
            for _, r in df.iterrows():
                v = r.get("dominant_dte_bucket")
                if pd.isna(v):
                    continue
                key = (d, str(r["ticker"]))
                out[key] = str(v)
    return out


def _read_grade_history() -> list[dict[str, str]]:
    if not GRADE_HISTORY.exists():
        return []
    with open(GRADE_HISTORY, newline="") as f:
        return list(csv.DictReader(f))


def _write_grade_history(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with open(GRADE_HISTORY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="write changes (else dry-run)")
    args = p.parse_args(argv)

    rows = _read_grade_history()
    if not rows:
        print("grade_history.csv not found or empty", flush=True)
        return 1

    fieldnames = list(rows[0].keys())
    if "dominant_dte_bucket" not in fieldnames:
        print("dominant_dte_bucket column missing — schema unexpected", flush=True)
        return 1

    print("Building lookups…", flush=True)
    archive = _build_archive_lookup()
    print(f"  archive entries: {len(archive)}", flush=True)
    ff = _build_flow_features_lookup()
    print(f"  flow_features entries: {len(ff)}", flush=True)

    fill_archive = 0
    fill_ff = 0
    fill_unknown = 0
    already_filled = 0
    sources: Counter = Counter()

    for r in rows:
        cur = (r.get("dominant_dte_bucket") or "").strip()
        if cur:
            already_filled += 1
            continue
        key = (str(r.get("as_of", "")), str(r.get("ticker", "")))
        if key in archive:
            r["dominant_dte_bucket"] = archive[key]
            fill_archive += 1
            sources["archive"] += 1
            continue
        if key in ff:
            r["dominant_dte_bucket"] = ff[key]
            fill_ff += 1
            sources["flow_features"] += 1
            continue
        r["dominant_dte_bucket"] = UNKNOWN_BUCKET
        fill_unknown += 1
        sources["unknown"] += 1

    total = len(rows)
    print()
    print("Backfill report")
    print(f"  total rows:          {total}")
    print(f"  already filled:      {already_filled}")
    print(f"  filled from archive: {fill_archive}")
    print(f"  filled from flow_ff: {fill_ff}")
    print(f"  marked 'unknown':    {fill_unknown}")
    coverage_real = (already_filled + fill_archive + fill_ff) / total
    print(f"  real DTE coverage:   {coverage_real:.1%}")

    print()
    print("Bucket distribution after backfill:")
    for k, v in Counter(r["dominant_dte_bucket"] for r in rows).most_common():
        print(f"  {k}: {v}")

    if args.apply:
        backup = GRADE_HISTORY.with_suffix(
            f".csv.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.copy2(GRADE_HISTORY, backup)
        print(f"\nBackup: {backup}", flush=True)
        _write_grade_history(rows, fieldnames)
        print(f"Wrote: {GRADE_HISTORY}", flush=True)
    else:
        print("\nDry-run only. Pass --apply to write the file.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

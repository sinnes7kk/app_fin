"""One-shot migration: re-stamp snapshot rows whose ``snapshot_date`` fell on
a non-NYSE-trading day to the prior trading day, and dedupe.

Before this migration existed, every tracker stamped rows with
``date.today()``.  A scan run on Saturday therefore produced rows dated
Saturday even though the data was Friday's close — duplicating the
prior-session row in the multi-day regression.

The migration walks each tracker CSV, re-stamps weekend/holiday rows to
the most recent prior NYSE session, then collapses
``(snapshot_date, ticker)`` duplicates with a "prefer pre-existing
correct row" rule (we keep the first occurrence seen when iterating in
original CSV order — i.e. the row that was already correctly stamped
for that trading day, because it was written before the mis-stamped
one).  Writes a ``.bak`` alongside each file first.

Safe to run multiple times — idempotent once all non-session rows are
re-stamped.

Usage::

    python -m scripts.restamp_snapshot_weekends           # dry-run
    python -m scripts.restamp_snapshot_weekends --apply   # write changes
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from app.utils.market_calendar import _is_session_day, _previous_session

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TARGETS = [
    DATA_DIR / "screener_snapshots.csv",
    DATA_DIR / "dp_snapshots.csv",
    DATA_DIR / "sentiment_snapshots.csv",
]


def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def _restamp(d: date) -> date:
    """Return ``d`` if it's an NYSE session day, else the prior session."""
    if _is_session_day(d):
        return d
    return _previous_session(d)


def _process(path: Path, apply: bool) -> dict:
    if not path.exists():
        return {"path": str(path), "status": "missing"}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "snapshot_date" not in fieldnames:
        return {"path": str(path), "status": "no_snapshot_date_col"}

    restamps: dict[str, str] = {}
    kept: list[dict] = []
    seen: set[tuple[str, str]] = set()
    dropped_dupes = 0
    restamped = 0

    for row in rows:
        original = (row.get("snapshot_date") or "").strip()
        parsed = _parse_date(original)
        if parsed is None:
            kept.append(row)
            continue
        new = _restamp(parsed)
        new_str = new.isoformat()
        if new_str != original:
            restamps[original] = new_str
            row = {**row, "snapshot_date": new_str}
            restamped += 1
        key = (new_str, (row.get("ticker") or "").upper().strip())
        if key in seen:
            dropped_dupes += 1
            continue
        seen.add(key)
        kept.append(row)

    result = {
        "path": str(path),
        "rows_in": len(rows),
        "rows_out": len(kept),
        "restamped": restamped,
        "dropped_dupes": dropped_dupes,
        "restamps": restamps,
    }

    if apply and (restamped or dropped_dupes):
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
        tmp.replace(path)
        result["backup"] = str(backup)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes.  Without this flag, only prints what would happen.",
    )
    args = parser.parse_args(argv)

    for path in TARGETS:
        r = _process(path, apply=args.apply)
        if r.get("status") == "missing":
            print(f"[skip] {path.name}: file does not exist")
            continue
        if r.get("status") == "no_snapshot_date_col":
            print(f"[skip] {path.name}: no snapshot_date column")
            continue
        verb = "rewrote" if args.apply else "would rewrite"
        print(
            f"[{path.name}] rows={r['rows_in']}->{r['rows_out']} "
            f"restamped={r['restamped']} dropped_dupes={r['dropped_dupes']}"
        )
        if r["restamps"]:
            for old, new in sorted(r["restamps"].items()):
                print(f"    {verb} {old} -> {new}")
        if args.apply and "backup" in r:
            print(f"    backup: {r['backup']}")

    if not args.apply:
        print("\nDry run.  Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

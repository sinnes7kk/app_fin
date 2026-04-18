#!/usr/bin/env python
"""One-shot migration — back up and truncate ``screener_snapshots.csv``.

Premium-Taxonomy plan
=====================

Before this migration, ``screener_snapshots.csv`` mixed two
incompatible definitions of ``bullish_premium`` / ``bearish_premium``:

*   Rows written by ``save_screener_snapshot`` used UW's aggregate
    (full ASK-call + BID-put premium across every DTE).
*   Rows written by ``save_flow_feature_snapshot`` used the narrow
    post-filter subset (only unusual flow, only 0-60 DTE).

Mixing the two in the multi-day regression skewed the Flow Tracker's
accumulation score and pulled in tickers whose "jump" was really just
a change in which writer ran last.  The schema has now been promoted
to a full premium taxonomy (``total_*_premium`` + per-bucket
``lottery_*`` / ``swing_*`` / ``leap_*`` + ``premium_source``) so both
writers persist comparable numbers — but every legacy row in the CSV
is still ambiguous.

This script:

1.  Copies the existing CSV to a timestamped backup under
    ``data/_archive/screener_snapshots_<UTC-ISO>.csv``.
2.  Rewrites the CSV with just the header row so the next scan starts
    from a clean slate.
3.  Prints the new header + row count so the operator can eyeball the
    result before kicking off a scan.

Run once, manually, after the Premium-Taxonomy code lands:

    python -m scripts.purge_snapshot_history

The Flow Tracker needs ``FLOW_TRACKER_WARMUP_DAYS`` trading days of
fresh snapshots (default 3) before the multi-day regression lights up
again — the Action Bar's "Rebuilding history" banner will surface
that to the UI while it ramps.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import the canonical column list + path from flow_tracker so we stay
# in sync with whatever the writer is persisting today.
from app.features.flow_tracker import SNAPSHOTS_PATH, SNAPSHOT_COLS


def _archive_dir(root: Path) -> Path:
    """Return (and create) ``data/_archive/`` beside ``screener_snapshots.csv``."""
    archive = root.parent / "_archive"
    archive.mkdir(parents=True, exist_ok=True)
    return archive


def _backup(source: Path) -> Path | None:
    """Copy ``source`` to ``_archive/screener_snapshots_<utc>.csv``.

    Returns the backup path, or ``None`` if the source file doesn't
    exist (nothing to back up).
    """
    if not source.exists():
        return None
    archive = _archive_dir(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = archive / f"{source.stem}_{stamp}.csv"
    shutil.copy2(source, dest)
    return dest


def _write_empty_snapshot(path: Path) -> int:
    """Rewrite ``path`` with just the header row.

    Returns the number of columns written so the caller can sanity-
    check the new schema.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(SNAPSHOT_COLS) + "\n"
    path.write_text(header, encoding="utf-8")
    return len(SNAPSHOT_COLS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (for CI / scripted use).",
    )
    args = parser.parse_args(argv)

    target = Path(SNAPSHOTS_PATH)

    if target.exists():
        row_count = sum(1 for _ in target.open("r", encoding="utf-8")) - 1
        print(f"screener_snapshots.csv   : {target}")
        print(f"existing rows (excl hdr) : {max(row_count, 0):,}")
    else:
        print(f"screener_snapshots.csv   : {target} (does not exist, will be created empty)")

    print(f"target column count      : {len(SNAPSHOT_COLS)}")

    if not args.yes:
        reply = input("Back up and truncate? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return 1

    backup_path = _backup(target)
    if backup_path is not None:
        print(f"backup written           : {backup_path}")
    else:
        print("backup written           : (no existing file, nothing to archive)")

    col_count = _write_empty_snapshot(target)
    print(f"truncated CSV, wrote     : {col_count} header columns")
    print("Done.  Flow Tracker will rebuild history on the next scan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

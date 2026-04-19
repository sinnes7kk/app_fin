#!/usr/bin/env python
"""One-shot migration — back up and truncate the daily-tracker snapshot CSVs.

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

1.  Copies each tracker CSV to a timestamped backup under
    ``data/_archive/<name>_<UTC-ISO>.csv``.
2.  Rewrites each CSV with just the header row so the next scan starts
    from a clean slate.
3.  Prints the new header + row count so the operator can eyeball the
    result before kicking off a scan.

By default it purges all three daily-tracker CSVs (screener, dark pool,
sentiment) so their multi-day regressions reset together — a mid-run
schema change in one contaminates the other trackers that read from
the same data directory in the same cadence.

Run once, manually, after a schema change:

    python -m scripts.purge_snapshot_history                # all 3
    python -m scripts.purge_snapshot_history --only screener
    python -m scripts.purge_snapshot_history --only screener dp

Each tracker needs ``FLOW_TRACKER_WARMUP_DAYS`` trading days of fresh
snapshots (default 3) before its multi-day regression lights up again
— the Action Bar's "Rebuilding history" banner surfaces that to the
UI while it ramps.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class _Target:
    key: str
    path: Path
    cols: list[str]


def _load_targets() -> list[_Target]:
    """Resolve the (key, path, cols) for each daily-tracker CSV.

    Imported lazily so a broken feature module doesn't keep the script
    from purging the other trackers.
    """
    targets: list[_Target] = []
    try:
        from app.features.flow_tracker import SNAPSHOTS_PATH, SNAPSHOT_COLS
        targets.append(_Target("screener", Path(SNAPSHOTS_PATH), list(SNAPSHOT_COLS)))
    except Exception as e:
        print(f"[warn] cannot resolve screener target: {e}")
    try:
        from app.features.dark_pool_tracker import DP_SNAPSHOTS_PATH, DP_SNAPSHOT_COLS
        targets.append(_Target("dp", Path(DP_SNAPSHOTS_PATH), list(DP_SNAPSHOT_COLS)))
    except Exception as e:
        print(f"[warn] cannot resolve dp target: {e}")
    try:
        from app.features.sentiment_tracker import SENTIMENT_PATH, SENTIMENT_COLS
        targets.append(_Target("sentiment", Path(SENTIMENT_PATH), list(SENTIMENT_COLS)))
    except Exception as e:
        print(f"[warn] cannot resolve sentiment target: {e}")
    return targets


def _archive_dir(root: Path) -> Path:
    """Return (and create) ``data/_archive/`` beside the tracker CSV."""
    archive = root.parent / "_archive"
    archive.mkdir(parents=True, exist_ok=True)
    return archive


def _backup(source: Path) -> Path | None:
    """Copy ``source`` to ``_archive/<stem>_<utc>.csv``.

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


def _write_empty_snapshot(path: Path, cols: list[str]) -> int:
    """Rewrite ``path`` with just the header row.

    Returns the number of columns written so the caller can sanity-
    check the new schema.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(cols) + "\n"
    path.write_text(header, encoding="utf-8")
    return len(cols)


def _purge_one(t: _Target) -> None:
    if t.path.exists():
        row_count = sum(1 for _ in t.path.open("r", encoding="utf-8")) - 1
        print(f"[{t.key}] {t.path}")
        print(f"[{t.key}] existing rows (excl hdr) : {max(row_count, 0):,}")
    else:
        print(f"[{t.key}] {t.path} (does not exist, will be created empty)")
    print(f"[{t.key}] target column count      : {len(t.cols)}")

    backup_path = _backup(t.path)
    if backup_path is not None:
        print(f"[{t.key}] backup written           : {backup_path}")
    else:
        print(f"[{t.key}] backup written           : (no existing file, nothing to archive)")

    col_count = _write_empty_snapshot(t.path, t.cols)
    print(f"[{t.key}] truncated CSV, wrote     : {col_count} header columns")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (for CI / scripted use).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["screener", "dp", "sentiment"],
        help="Restrict the purge to a subset of trackers.  Default: all three.",
    )
    args = parser.parse_args(argv)

    targets = _load_targets()
    if args.only:
        wanted = set(args.only)
        targets = [t for t in targets if t.key in wanted]
    if not targets:
        print("No targets resolved — nothing to do.")
        return 1

    print("About to back up and truncate the following tracker CSVs:")
    for t in targets:
        print(f"  * {t.key:9s} {t.path}")

    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return 1

    for t in targets:
        print()
        _purge_one(t)

    print("\nDone.  Each tracker will rebuild history on the next scan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

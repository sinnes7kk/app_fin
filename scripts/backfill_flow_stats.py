"""One-off backfill for ``data/flow_stats_meta.json``.

Walks ``data/flow_features/flow_features_*.csv`` chronologically and records the
first date each ticker was observed. Used by the z-score fallback ladder to
distinguish "new to screener" tickers (recent first_observed_date) from
tickers with gap days in their series.

Usage:
    python -m scripts.backfill_flow_stats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.features.flow_stats import FLOW_FEATURES_DIR, META_PATH, _parse_snapshot_date  # noqa: E402


def main() -> int:
    if not FLOW_FEATURES_DIR.exists():
        print(f"No flow_features directory at {FLOW_FEATURES_DIR} — nothing to backfill.")
        return 1

    files = sorted(FLOW_FEATURES_DIR.glob("flow_features_*.csv"))
    if not files:
        print("No flow_features snapshots found — nothing to backfill.")
        return 1

    first_seen: dict[str, str] = {}
    processed_files = 0
    for p in files:
        d = _parse_snapshot_date(p)
        if d is None:
            continue
        try:
            df = pd.read_csv(p, usecols=["ticker"])
        except Exception as e:
            print(f"  skipped {p.name}: {e}")
            continue
        if df.empty or "ticker" not in df.columns:
            continue
        for t in df["ticker"].dropna().astype(str).str.upper().unique():
            t = t.strip()
            if not t:
                continue
            if t not in first_seen or d < first_seen[t]:
                first_seen[t] = d
        processed_files += 1

    if META_PATH.exists():
        try:
            with open(META_PATH, "r") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}
    else:
        existing = {}

    merged = dict(existing)
    added = 0
    for t, d in first_seen.items():
        if t not in merged:
            merged[t] = d
            added += 1
        elif d < merged[t]:
            merged[t] = d

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)

    print(
        f"Backfill complete: scanned {processed_files} files, "
        f"{len(first_seen)} unique tickers, added {added} new entries. "
        f"Total meta entries: {len(merged)}. Written to {META_PATH}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

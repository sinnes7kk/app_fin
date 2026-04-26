"""'Saw it, couldn't trade it' panel.

For every scan day we already write ``data/final_signals/rejected_*.csv``
— one file per pipeline run with every ticker that was scored but did
not make it through to a final signal. The vast majority of rejects
are uninteresting (low flow, no setup) but a non-trivial slice has
**high flow** that was killed at the price-validation gate. Those are
the trades we *saw* but *couldn't take*, and they're the ones worth
auditing to retune the gate.

This module builds a per-day panel of those high-flow rejects, plus
an append-only history rollup that the audit script consumes.

v1 deliberately stops short of computing forward returns — the
codebase doesn't have a centralised price-history store keyed by date,
and pulling one in is a separate piece of work. The audit script
notes the gap and falls back to "rejection-mix-over-time" as the
useful read until forward returns land.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

# Per-side score floor for inclusion in the panel. Same scale as
# bullish_score / bearish_score (0-1); 0.5 = the "elevated" line we
# also use for the sector-heat aggregator. Below this we don't think
# the system "saw" the trade — it just scored it.
HIGH_FLOW_THRESHOLD = 0.5

# z-shadow scores live on 0-10 scale (they are already pre-multiplied
# inside ``rescore_with_z``). 5.0 mirrors the 0.5 cutoff above so a
# ticker whose abs-path was mid but whose z-shadow flagged it as a
# strong basket-relative outlier still surfaces.
HIGH_FLOW_THRESHOLD_Z_SHADOW = 5.0

# Stamp pattern in rejected file names: ``rejected_YYYYMMDD_HHMMSS.csv``.
_STAMP_RE = re.compile(r"rejected_(\d{8})_(\d{6})\.csv$")


def _parse_scan_ts(path: Path) -> tuple[str, str] | None:
    """Return ``(YYYY-MM-DD, HH:MM:SS)`` parsed from ``path.name`` or None."""
    m = _STAMP_RE.search(path.name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    iso_d = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    iso_t = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}"
    return iso_d, iso_t


def list_rejected_files_for_day(
    final_signals_dir: Path,
    date_str: str,
) -> list[Path]:
    """All rejected_*.csv files whose stamped date matches ``date_str``."""
    final_signals_dir = Path(final_signals_dir)
    if not final_signals_dir.exists():
        return []
    out: list[Path] = []
    for p in sorted(final_signals_dir.glob("rejected_*.csv")):
        parsed = _parse_scan_ts(p)
        if parsed and parsed[0] == date_str:
            out.append(p)
    return out


def load_rejected_for_day(
    final_signals_dir: Path,
    date_str: str,
) -> pd.DataFrame:
    """Concatenate every rejected_*.csv for ``date_str`` and stamp scan_ts.

    Returns an empty frame (without scan_ts) if no files exist for
    that date.
    """
    files = list_rejected_files_for_day(final_signals_dir, date_str)
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        parsed = _parse_scan_ts(path)
        if parsed is None:
            continue
        scan_iso = f"{parsed[0]}T{parsed[1]}"
        df = df.copy()
        df["scan_ts"] = scan_iso
        df["scan_date"] = parsed[0]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def dedupe_latest_per_ticker(rejected: pd.DataFrame) -> pd.DataFrame:
    """Keep the row with the highest ``scan_ts`` per (ticker, direction).

    Multiple scans per day will surface the same ticker repeatedly; we
    only care about its most recent rejection state.
    """
    if rejected.empty or "scan_ts" not in rejected.columns:
        return rejected.copy()
    df = rejected.sort_values("scan_ts").copy()
    keys = ["ticker", "direction"] if "direction" in df.columns else ["ticker"]
    deduped = df.groupby(keys, as_index=False, sort=False).tail(1)
    return deduped.reset_index(drop=True)


def normalize_reject_reason(raw: object) -> str:
    """Bucket a raw ``reject_reason`` into a clean enum string.

    Handles trailing parenthetical detail (``poor_rr (1.7:1)`` ->
    ``poor_rr``) and unmapped values fall through to ``other``.
    """
    if raw is None:
        return "none"
    s = str(raw).strip()
    if not s or s.lower() == "none" or s.lower() == "nan":
        return "none"
    s = re.sub(r"\s*\(.*\)$", "", s).strip()
    known = {
        "price_validation_failed",
        "price_over_extended",
        "poor_rr",
        "trend_not_aligned",
        "weak_bullish_flow",
        "weak_bearish_flow",
        "weak_relative_strength",
        "watchlist_reeval_failed",
        "no_options_context",
    }
    if s in known:
        return s
    return "other"


# Reject reasons we treat as "high-flow but blocked" — i.e. the system
# did see strong flow, something downstream killed it. Excludes the
# weak-flow buckets (those are correct rejections by definition) and
# watchlist reevaluations (those are stale-state rejections, separate
# concern).
HIGH_FLOW_REJECT_REASONS: set[str] = {
    "price_validation_failed",
    "price_over_extended",
    "poor_rr",
    "trend_not_aligned",
    "no_options_context",
    "weak_relative_strength",
    "other",
}

PANEL_COLUMNS = [
    "scan_date",
    "scan_ts",
    "ticker",
    "direction",
    "flow_score_raw",
    "flow_score_scaled",
    "bullish_score_z_shadow",
    "bearish_score_z_shadow",
    "price_score",
    "final_score",
    "reject_reason",
    "reject_reason_norm",
    "checks_failed",
    "iv_rank",
    "gamma_regime",
]


def build_panel(
    final_signals_dir: Path,
    date_str: str,
    threshold_raw: float = HIGH_FLOW_THRESHOLD,
    threshold_z_shadow: float = HIGH_FLOW_THRESHOLD_Z_SHADOW,
) -> pd.DataFrame:
    """Build the saw-it-couldn't-trade panel for one trading day.

    A ticker enters the panel when:

    * ``flow_score_raw >= threshold_raw`` *or*
      ``z-shadow score for the relevant side >= threshold_z_shadow``,
    * **and** the normalized ``reject_reason`` lives in
      ``HIGH_FLOW_REJECT_REASONS``.

    Each (ticker, direction) is represented once, by the row from the
    latest scan of the day.
    """
    rejected = load_rejected_for_day(final_signals_dir, date_str)
    if rejected.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    deduped = dedupe_latest_per_ticker(rejected)
    if deduped.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    # Coerce numerics defensively — the rejected files are written
    # with mixed types and pandas can land columns as object when a
    # row writes an empty string.
    for col in (
        "flow_score_raw",
        "flow_score_scaled",
        "bullish_score_z_shadow",
        "bearish_score_z_shadow",
        "price_score",
        "final_score",
        "iv_rank",
    ):
        if col in deduped.columns:
            deduped[col] = pd.to_numeric(deduped[col], errors="coerce")

    deduped["reject_reason_norm"] = deduped.get(
        "reject_reason", pd.Series([""] * len(deduped))
    ).map(normalize_reject_reason)

    if "direction" in deduped.columns:
        bull_z = deduped.get("bullish_score_z_shadow")
        bear_z = deduped.get("bearish_score_z_shadow")
        side_z = pd.Series(0.0, index=deduped.index)
        if bull_z is not None:
            side_z = side_z.where(
                deduped["direction"].astype(str).str.upper() != "LONG",
                bull_z.fillna(0.0),
            )
        if bear_z is not None:
            side_z = side_z.where(
                deduped["direction"].astype(str).str.upper() != "SHORT",
                bear_z.fillna(0.0),
            )
    else:
        side_z = pd.Series(0.0, index=deduped.index)

    flow_raw = deduped.get("flow_score_raw")
    flow_raw_filled = flow_raw.fillna(0.0) if flow_raw is not None else pd.Series(
        0.0, index=deduped.index
    )

    high_flow_mask = (flow_raw_filled >= threshold_raw) | (
        side_z >= threshold_z_shadow
    )
    blocked_mask = deduped["reject_reason_norm"].isin(HIGH_FLOW_REJECT_REASONS)
    panel = deduped[high_flow_mask & blocked_mask].copy()
    if panel.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    for col in PANEL_COLUMNS:
        if col not in panel.columns:
            panel[col] = pd.NA
    panel = panel[PANEL_COLUMNS].sort_values(
        ["flow_score_raw"], ascending=False
    ).reset_index(drop=True)
    return panel


def append_history(
    panel: pd.DataFrame,
    history_path: Path,
) -> None:
    """Append today's panel rows to the long-running history CSV.

    Idempotent on (scan_date, ticker, direction): re-appending the
    same key set replaces the prior rows so we always reflect the
    latest scan's rejection state.
    """
    history_path = Path(history_path)
    if panel is None or panel.empty:
        return
    key_cols = [c for c in ("scan_date", "ticker", "direction") if c in panel.columns]
    if history_path.exists():
        try:
            existing = pd.read_csv(history_path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty and all(c in existing.columns for c in key_cols):
            new_keys = set(panel[key_cols].astype(str).apply(tuple, axis=1).tolist())
            existing_keys = existing[key_cols].astype(str).apply(tuple, axis=1)
            keep = ~existing_keys.isin(new_keys)
            combined = pd.concat([existing[keep], panel], ignore_index=True)
        else:
            combined = panel
    else:
        combined = panel
    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)


def emit_daily(
    final_signals_dir: Path,
    output_dir: Path,
    history_path: Path,
    date_str: str | None = None,
) -> tuple[Path | None, int]:
    """End-to-end: build today's panel, write per-day CSV + append history.

    Returns ``(daily_path or None, n_rows)``. ``daily_path`` is None
    when there is nothing to write.
    """
    final_signals_dir = Path(final_signals_dir)
    output_dir = Path(output_dir)
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    panel = build_panel(final_signals_dir, date_str)
    if panel.empty:
        return None, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"saw_couldnt_trade_{date_str}.csv"
    panel.to_csv(out_path, index=False)
    append_history(panel, history_path)
    return out_path, len(panel)

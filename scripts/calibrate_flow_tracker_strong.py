"""Calibrate Flow Tracker Strong + Early thresholds against history.

Sweeps the relevant gates over the last N trading days of
``data/snapshots_archive.csv.gz`` and reports per-config pass counts so we
can pick thresholds knowing exactly what each setting would have produced
historically. Two cohorts are evaluated:

  * Strong (multi-day, directional purity, accumulation-class)
  * Early (2-day same-direction confirmation, looser size floors)

The values currently in ``FLOW_TRACKER_MODES["strong_accumulation"]`` and
``FLOW_TRACKER_MODES["early_accumulation"]`` were picked from the
``wide-10-min4`` and ``early-mid`` rows of the 2026-05-09 sweep (see
``data/diagnostic_strong_calibration_2026-05-09.md``). Re-run periodically
as the data window grows; if a different candidate dominates, update the
config block in ``app/config.py`` accordingly.

Usage::

    python scripts/calibrate_flow_tracker_strong.py
    python scripts/calibrate_flow_tracker_strong.py --days 15 --top 5
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.features.flow_tracker as ft  # noqa: E402
from app.features.flow_tracker import compute_multi_day_flow  # noqa: E402

ARCHIVE_PATH = ROOT / "data" / "snapshots_archive.csv.gz"
REPORT_PATH = ROOT / "data" / f"diagnostic_strong_calibration_{date.today().isoformat()}.md"

# Cohort sweep grids ----------------------------------------------------

STRONG_CANDIDATES: list[dict[str, Any]] = [
    # label, lookback (cal days), day_skew_floor, min_active, min_cum_M,
    # min_bps, min_consistency, min_day_persist, require_no_flips,
    # min_accel_t, min_grade_rank
    {"label": "current",          "lookback": 5,  "skew": 0.20, "min_act": 5, "cum_M": 25, "bps": 5.0, "cons": 0.30, "day_p": 0.60, "no_flip": True,  "accel_t": 0.5, "grade": 4},
    {"label": "wide-window-7",    "lookback": 7,  "skew": 0.15, "min_act": 5, "cum_M": 25, "bps": 5.0, "cons": 0.20, "day_p": 0.60, "no_flip": True,  "accel_t": 0.0, "grade": 4},
    {"label": "wide-window-10",   "lookback": 10, "skew": 0.10, "min_act": 5, "cum_M": 25, "bps": 5.0, "cons": 0.20, "day_p": 0.60, "no_flip": True,  "accel_t": 0.0, "grade": 4},
    {"label": "wide-10-low-skew", "lookback": 10, "skew": 0.08, "min_act": 5, "cum_M": 25, "bps": 5.0, "cons": 0.10, "day_p": 0.60, "no_flip": True,  "accel_t": 0.0, "grade": 4},
    {"label": "wide-10-min4",     "lookback": 10, "skew": 0.08, "min_act": 4, "cum_M": 25, "bps": 5.0, "cons": 0.10, "day_p": 0.60, "no_flip": True,  "accel_t": 0.0, "grade": 4},
    {"label": "wide-10-min4-A_minus","lookback": 10, "skew": 0.08, "min_act": 4, "cum_M": 25, "bps": 5.0, "cons": 0.10, "day_p": 0.60, "no_flip": True,  "accel_t": 0.0, "grade": 4},
    {"label": "wide-10-min3",     "lookback": 10, "skew": 0.08, "min_act": 3, "cum_M": 25, "bps": 5.0, "cons": 0.10, "day_p": 0.60, "no_flip": True,  "accel_t": 0.0, "grade": 4},
    {"label": "tight-but-fixed",  "lookback": 10, "skew": 0.08, "min_act": 5, "cum_M": 25, "bps": 5.0, "cons": 0.20, "day_p": 0.70, "no_flip": True,  "accel_t": 0.0, "grade": 4},
    {"label": "loose-grade",      "lookback": 10, "skew": 0.08, "min_act": 4, "cum_M": 15, "bps": 3.0, "cons": 0.10, "day_p": 0.60, "no_flip": True,  "accel_t": 0.0, "grade": 3},
]

EARLY_CANDIDATES: list[dict[str, Any]] = [
    # 2-day mode candidates. ``min_act`` should always be 2; the other
    # gates control how strict the 2-day confirmation is.
    {"label": "early-strict",     "lookback": 4, "skew": 0.20, "min_act": 2, "cum_M": 10, "bps": 3.0, "cons": 0.30, "day_p": 1.00, "no_flip": True,  "accel_t": -99.0, "grade": 3},
    {"label": "early-mid",        "lookback": 4, "skew": 0.10, "min_act": 2, "cum_M": 5,  "bps": 2.0, "cons": 0.10, "day_p": 1.00, "no_flip": True,  "accel_t": -99.0, "grade": 3},
    {"label": "early-loose",      "lookback": 4, "skew": 0.08, "min_act": 2, "cum_M": 5,  "bps": 2.0, "cons": 0.05, "day_p": 1.00, "no_flip": True,  "accel_t": -99.0, "grade": 2},
    {"label": "early-permissive", "lookback": 4, "skew": 0.08, "min_act": 2, "cum_M": 3,  "bps": 1.5, "cons": 0.05, "day_p": 1.00, "no_flip": True,  "accel_t": -99.0, "grade": 0},
    {"label": "early-no-flip",    "lookback": 4, "skew": 0.10, "min_act": 2, "cum_M": 5,  "bps": 2.0, "cons": 0.05, "day_p": 0.50, "no_flip": True,  "accel_t": -99.0, "grade": 2},
]

# ----------------------------------------------------------------------


def _patch_global_constants(lookback: int, skew_floor: float) -> None:
    """Patch the two module-level constants that compute_multi_day_flow reads.

    Default-bound parameters of ``_compute_day_persistence`` need a
    separate fix: we rewrite ``__defaults__`` so the new floor takes
    effect everywhere the function is called without an explicit arg.
    """
    ft.FLOW_TRACKER_LOOKBACK_DAYS = lookback
    ft.FLOW_TRACKER_DAY_SKEW_FLOOR = skew_floor
    fn = ft._compute_day_persistence
    if fn.__defaults__:
        fn.__defaults__ = (skew_floor,) + fn.__defaults__[1:]


def _gate_dict(c: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "min_active_days": int(c["min_act"]),
        "min_cum_premium": float(c["cum_M"]) * 1e6,
        "min_prem_mcap_bps": float(c["bps"]),
        "min_consistency": float(c["cons"]),
        "min_day_persistence": float(c["day_p"]),
        "require_no_flips": bool(c["no_flip"]),
        "min_accel_t": float(c["accel_t"]),
        "exclude_hedging": True,
        "min_grade_rank": int(c["grade"]),
        "intro": "calibration sweep",
    }


def _patch_strong_mode(c: dict[str, Any]) -> None:
    """Rewrite Strong's gate dict in-place so passes_strong reflects the candidate."""
    ft.FLOW_TRACKER_MODES["strong_accumulation"] = _gate_dict(c, "Strong")


def _patch_early_mode(c: dict[str, Any]) -> None:
    """Rewrite Early's gate dict in-place so passes_early reflects the candidate."""
    ft.FLOW_TRACKER_MODES["early_accumulation"] = _gate_dict(c, "Early")


def trading_days_from_archive(n_back: int) -> list[str]:
    df = pd.read_csv(ARCHIVE_PATH)
    dates = sorted(df["snapshot_date"].unique())
    return dates[-n_back:]


def evaluate_candidate(
    candidate: dict[str, Any],
    asof_dates: list[str],
    cohort: str = "strong",
) -> dict[str, Any]:
    """Evaluate ``candidate`` for the requested ``cohort`` (``strong`` or ``early``).

    Re-uses ``passes_strong`` / ``passes_early`` per-row flags by patching
    the corresponding entry in ``FLOW_TRACKER_MODES`` before each scoring
    pass. Cohorts are evaluated independently so each can sweep its own
    lookback / skew settings without interference.
    """
    _patch_global_constants(candidate["lookback"], candidate["skew"])
    if cohort == "strong":
        _patch_strong_mode(candidate)
        flag = "passes_strong"
    else:
        _patch_early_mode(candidate)
        flag = "passes_early"

    per_day_pass: dict[str, int] = {}
    per_day_total: dict[str, int] = {}
    survivors: list[tuple[str, str, str]] = []  # (date, ticker, direction)

    for asof in asof_dates:
        rows = compute_multi_day_flow(
            mode=None, as_of=asof, snapshots_path=ARCHIVE_PATH,
        )
        per_day_total[asof] = len(rows)
        passers = [r for r in rows if r.get(flag)]
        per_day_pass[asof] = len(passers)
        for r in passers:
            survivors.append((asof, r["ticker"], r["direction"]))

    n_days_with_hits = sum(1 for n in per_day_pass.values() if n >= 1)
    avg_per_day = sum(per_day_pass.values()) / max(len(per_day_pass), 1)
    max_per_day = max(per_day_pass.values()) if per_day_pass else 0
    total_hits = sum(per_day_pass.values())

    return {
        "candidate": candidate,
        "per_day_pass": per_day_pass,
        "per_day_total": per_day_total,
        "n_days_with_hits": n_days_with_hits,
        "avg_per_day": avg_per_day,
        "max_per_day": max_per_day,
        "total_hits": total_hits,
        "survivors": survivors,
    }


def render_markdown(
    strong_results: list[dict[str, Any]],
    early_results: list[dict[str, Any]],
    asof_dates: list[str],
    n_top: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# Flow Tracker — Strong + Early Calibration Sweep ({date.today().isoformat()})")
    lines.append("")
    lines.append(
        f"Sweep window: **{asof_dates[0]} → {asof_dates[-1]}** ({len(asof_dates)} trading days)."
    )
    lines.append(
        "Source: `data/snapshots_archive.csv.gz` (append-only screener history)."
    )
    lines.append("")
    lines.append(
        "Each row evaluates `compute_multi_day_flow` with the candidate's "
        "lookback / day-skew floor patched in, then counts how many tickers "
        "passed `passes_strong` on each historical `as_of`."
    )
    lines.append("")

    # ---------------- Strong section ----------------
    lines.append("## Strong cohort")
    lines.append("")
    lines.append(
        "| label | LB | skew | min_act | cum$M | bps | cons | dayP | accel_t | grade | days≥1 | avg/d | max/d | total |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for res in strong_results:
        c = res["candidate"]
        lines.append(
            f"| {c['label']} | {c['lookback']} | {c['skew']:.2f} | {c['min_act']} | "
            f"{c['cum_M']} | {c['bps']:.1f} | {c['cons']:.2f} | {c['day_p']:.2f} | "
            f"{c['accel_t']:.1f} | {c['grade']} | {res['n_days_with_hits']}/{len(asof_dates)} | "
            f"{res['avg_per_day']:.1f} | {res['max_per_day']} | {res['total_hits']} |"
        )
    lines.append("")
    lines.append("### Strong — per-day pass counts")
    lines.append("")
    header = "| date | " + " | ".join(r["candidate"]["label"] for r in strong_results) + " |"
    sep = "|---|" + "|".join(["---:"] * len(strong_results)) + "|"
    lines.append(header)
    lines.append(sep)
    for d in asof_dates:
        cells = [str(r["per_day_pass"].get(d, 0)) for r in strong_results]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### Strong — top survivors per candidate")
    lines.append("")
    for res in strong_results:
        if not res["survivors"]:
            continue
        lines.append(f"**{res['candidate']['label']}** ({len(res['survivors'])} hits across window):")
        # group by ticker — direction
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for _, t, dir_ in res["survivors"]:
            counts[(t, dir_)] += 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:n_top]
        for (t, dir_), n in ranked:
            lines.append(f"- {t} ({dir_}): {n} day(s)")
        lines.append("")

    # ---------------- Early section ----------------
    lines.append("## Early cohort (2-day flow)")
    lines.append("")
    lines.append(
        "Same gate, but `min_active_days = 2` and (for most candidates) "
        "`min_day_persistence = 1.00` + `require_no_flips = True` — i.e. **both** "
        "active days lean the same direction. The lookback is 4 calendar days "
        "so the window holds ~2-3 trading days plus today, keeping the 2-day "
        "confirmation tight."
    )
    lines.append("")
    lines.append(
        "| label | LB | skew | min_act | cum$M | bps | cons | dayP | accel_t | grade | days≥1 | avg/d | max/d | total |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for res in early_results:
        c = res["candidate"]
        lines.append(
            f"| {c['label']} | {c['lookback']} | {c['skew']:.2f} | {c['min_act']} | "
            f"{c['cum_M']} | {c['bps']:.1f} | {c['cons']:.2f} | {c['day_p']:.2f} | "
            f"{c['accel_t']:.1f} | {c['grade']} | {res['n_days_with_hits']}/{len(asof_dates)} | "
            f"{res['avg_per_day']:.1f} | {res['max_per_day']} | {res['total_hits']} |"
        )
    lines.append("")
    lines.append("### Early — per-day pass counts")
    lines.append("")
    header = "| date | " + " | ".join(r["candidate"]["label"] for r in early_results) + " |"
    sep = "|---|" + "|".join(["---:"] * len(early_results)) + "|"
    lines.append(header)
    lines.append(sep)
    for d in asof_dates:
        cells = [str(r["per_day_pass"].get(d, 0)) for r in early_results]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### Early — top survivors per candidate")
    lines.append("")
    for res in early_results:
        if not res["survivors"]:
            continue
        lines.append(f"**{res['candidate']['label']}** ({len(res['survivors'])} hits across window):")
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for _, t, dir_ in res["survivors"]:
            counts[(t, dir_)] += 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:n_top]
        for (t, dir_), n in ranked:
            lines.append(f"- {t} ({dir_}): {n} day(s)")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=15, help="trading days to sweep (default 15)")
    parser.add_argument("--top", type=int, default=10, help="top survivors per cohort to print")
    args = parser.parse_args()

    if not ARCHIVE_PATH.exists():
        print(f"ERROR: {ARCHIVE_PATH} not found")
        sys.exit(1)

    asof_dates = trading_days_from_archive(args.days)
    print(f"Sweeping {len(asof_dates)} trading days: {asof_dates[0]} → {asof_dates[-1]}")
    print(f"Strong cohort: {len(STRONG_CANDIDATES)} candidates")
    print(f"Early cohort:  {len(EARLY_CANDIDATES)} candidates")
    print()

    # Snapshot the originals so the script doesn't permanently corrupt the
    # module's constants in long-running test environments.
    orig_lookback = ft.FLOW_TRACKER_LOOKBACK_DAYS
    orig_skew = ft.FLOW_TRACKER_DAY_SKEW_FLOOR
    orig_strong = dict(ft.FLOW_TRACKER_MODES["strong_accumulation"])
    orig_early = dict(ft.FLOW_TRACKER_MODES.get("early_accumulation", {}))
    orig_defaults = ft._compute_day_persistence.__defaults__

    try:
        strong_results = []
        for c in STRONG_CANDIDATES:
            print(f"  evaluating Strong: {c['label']}")
            strong_results.append(evaluate_candidate(c, asof_dates, cohort="strong"))

        early_results = []
        for c in EARLY_CANDIDATES:
            print(f"  evaluating Early:  {c['label']}")
            early_results.append(evaluate_candidate(c, asof_dates, cohort="early"))
    finally:
        ft.FLOW_TRACKER_LOOKBACK_DAYS = orig_lookback
        ft.FLOW_TRACKER_DAY_SKEW_FLOOR = orig_skew
        ft.FLOW_TRACKER_MODES["strong_accumulation"] = orig_strong
        if orig_early:
            ft.FLOW_TRACKER_MODES["early_accumulation"] = orig_early
        ft._compute_day_persistence.__defaults__ = orig_defaults

    md = render_markdown(strong_results, early_results, asof_dates, args.top)
    REPORT_PATH.write_text(md)

    print()
    print(f"Report written → {REPORT_PATH}")
    print()
    print("Quick summary:")
    print()
    print(f"  Strong cohort:")
    for res in strong_results:
        c = res["candidate"]
        print(
            f"    {c['label']:<24} days≥1={res['n_days_with_hits']:>2}/{len(asof_dates)}  "
            f"avg={res['avg_per_day']:.1f}  total={res['total_hits']}"
        )
    print()
    print(f"  Early cohort:")
    for res in early_results:
        c = res["candidate"]
        print(
            f"    {c['label']:<24} days≥1={res['n_days_with_hits']:>2}/{len(asof_dates)}  "
            f"avg={res['avg_per_day']:.1f}  total={res['total_hits']}"
        )


if __name__ == "__main__":
    main()

"""Audit data/grade_history.csv for proxy-feature data quality.

For each of the six raw inputs that feed conviction_recalibration's
proxies (persistence_ratio, prem_mcap_bps, accumulation_score,
accel_ratio_today, cumulative_premium, latest_oi_change), this script
counts:

  - total rows
  - blank/null rows
  - exact-zero rows (often equivalent to "no data" once the producing
    feature falls back to the dataclass default)
  - non-zero rows

A column where >50% of rows are blank/zero is treated as a likely
upstream bug and flagged loudly, because the recalibration fit will
silently treat those rows as low-signal.

Usage:

    python scripts/audit_grade_history.py
    python scripts/audit_grade_history.py --out data/audit_grade_history.md
    python scripts/audit_grade_history.py --append data/diagnostic_recalibration_2026-05-06.md

The ``--append`` mode tacks the audit table onto an existing
recalibration diagnostic so the human reviewing the report sees both
the fit results and the data-quality context together.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
GRADE_HISTORY = DATA_DIR / "grade_history.csv"

PROXY_INPUTS = (
    "persistence_ratio",
    "prem_mcap_bps",
    "accumulation_score",
    "accel_ratio_today",
    "cumulative_premium",
    "latest_oi_change",
)

# Threshold above which a column's blank+zero share is treated as a bug
# rather than a normal sparse-feature outcome.
SUSPICIOUS_BLANK_OR_ZERO_FRAC = 0.50


def _is_blank(v: str | None) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ("", "nan", "none", "null")


def _is_zero(v: str | None) -> bool:
    if _is_blank(v):
        return False
    try:
        return float(v) == 0.0
    except (TypeError, ValueError):
        return False


def audit(history_path: Path = GRADE_HISTORY) -> dict:
    """Compute per-column blank/zero/non-zero counts. Returns a dict
    suitable for serialization or markdown rendering.
    """
    if not history_path.exists():
        return {"error": f"missing: {history_path}", "columns": []}
    with open(history_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    n = len(rows)

    columns: list[dict] = []
    for col in PROXY_INPUTS:
        blank = sum(1 for r in rows if _is_blank(r.get(col)))
        zero = sum(1 for r in rows if _is_zero(r.get(col)))
        non_zero = n - blank - zero
        blank_or_zero_frac = (blank + zero) / n if n else 0.0
        suspicious = (
            n > 0 and blank_or_zero_frac > SUSPICIOUS_BLANK_OR_ZERO_FRAC
        )
        columns.append({
            "column": col,
            "n": n,
            "blank": blank,
            "zero": zero,
            "non_zero": non_zero,
            "blank_or_zero_frac": blank_or_zero_frac,
            "suspicious": suspicious,
        })
    return {"n": n, "columns": columns}


def render_markdown(report: dict) -> str:
    if "error" in report:
        return f"## Grade-history audit\n\n_{report['error']}_\n"

    lines = [
        "## Grade-history input audit",
        "",
        f"Inspecting `data/grade_history.csv` ({report['n']} rows). For each "
        "of the six raw inputs that feed conviction_recalibration's proxies, "
        "we count blank/null vs exact-zero vs non-zero rows. Columns flagged "
        f"as `⚠ suspicious` have > {int(SUSPICIOUS_BLANK_OR_ZERO_FRAC * 100)}% "
        "blank-or-zero values, which usually means an upstream feature is "
        "silently producing zeros and the fit is treating them as low-signal.",
        "",
        "| Column | n | blank | zero | non-zero | blank+zero % | flag |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in report["columns"]:
        flag = "⚠ suspicious" if c["suspicious"] else "ok"
        lines.append(
            f"| `{c['column']}` | {c['n']} | {c['blank']} | {c['zero']} | "
            f"{c['non_zero']} | {c['blank_or_zero_frac']:.1%} | {flag} |"
        )
    lines.append("")

    bad = [c["column"] for c in report["columns"] if c["suspicious"]]
    if bad:
        lines.append(
            "**Action:** investigate the producers of "
            f"{', '.join('`' + c + '`' for c in bad)}. Most likely the "
            "feature is being computed before its inputs are populated, or "
            "is hitting a default branch in `compute_multi_day_flow`."
        )
    else:
        lines.append("All six proxy inputs look populated. Recalibration fits "
                     "can trust the panel.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(GRADE_HISTORY))
    ap.add_argument("--out", default=None,
                    help="Optional output file. If omitted, prints to stdout.")
    ap.add_argument("--append", default=None,
                    help="Append the audit section to an existing markdown "
                         "report (e.g. diagnostic_recalibration_*.md).")
    args = ap.parse_args(argv)

    report = audit(Path(args.input))
    md = render_markdown(report)

    if args.append:
        target = Path(args.append)
        if not target.exists():
            print(f"--append target missing: {target}", file=sys.stderr)
            return 1
        existing = target.read_text()
        sep = "\n\n---\n\n" if not existing.endswith("\n") else "\n---\n\n"
        target.write_text(existing + sep + md)
        print(f"Appended audit to {target}")
        return 0

    if args.out:
        Path(args.out).write_text(md)
        print(f"Wrote {args.out}")
        return 0

    print(md)
    print(f"\n_(generated {datetime.now().isoformat(timespec='seconds')})_")
    return 0


if __name__ == "__main__":
    sys.exit(main())

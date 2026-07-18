"""Shadow meta-labeling report.

Trains the L2-logistic meta-model (``app.backtest.meta_label``) on the
joined feature_lab × replay panel using purged walk-forward CV and writes
a markdown summary: OOS AUC, calibration (Brier), the realized-R lift
between high- and low-P(win) candidates, and a decile sizing curve.

The question it answers: *if we sized trades by predicted P(win) instead
of taking every candidate flat, would we have earned more per unit
risk?* A positive OOS R-lift and AUC > ~0.55 is the signal that
meta-labeling is worth wiring into position sizing.

Shadow-only. Writes ``data/meta_label_shadow_<YYYY-MM-DD>.md``.

Usage::

    python scripts/meta_label_report.py
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.meta_label import train_meta_walk_forward  # noqa: E402
from scripts.feature_lab_report import build_panel  # noqa: E402

DATA_DIR = ROOT / "data"


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.3f}"


def build_report(res, *, label_days: int, n_splits: int) -> str:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Meta-labeling — shadow report — {today}",
        "",
        f"Rows used: **{res.n}**  |  OOS folds: **{res.n_folds}**  |  "
        f"walk-forward {n_splits} splits, {label_days}d label horizon.",
        "",
        f"Features: {', '.join('`' + f + '`' for f in res.features) or '—'}",
        "",
        "---",
        "",
        "## Out-of-sample metrics",
        "",
        f"- **ROC-AUC:** {_fmt(res.auc)}  _(0.50 = coin flip; >0.55 = usable)_",
        f"- **Brier score:** {_fmt(res.brier)}  _(lower = better calibrated)_",
        f"- **Base win rate:** {_fmt(res.base_win_rate)}",
        "",
        "## Realized-R lift by P(win)",
        "",
        f"- Top⅓ P(win) mean realized R: **{_fmt(res.top_tercile_r)}**",
        f"- Bottom⅓ P(win) mean realized R: **{_fmt(res.bottom_tercile_r)}**",
        f"- **Lift (top − bottom): {_fmt(res.r_lift)}**",
        "",
    ]
    if res.decile_r:
        lines.append("## Decile sizing curve (mean realized R, low→high P(win))")
        lines.append("")
        cells = " | ".join(f"{v:+.2f}" for v in res.decile_r)
        lines.append("| " + " | ".join(f"D{i+1}" for i in range(len(res.decile_r))) + " |")
        lines.append("| " + " | ".join(["---"] * len(res.decile_r)) + " |")
        lines.append("| " + cells + " |")
        lines.append("")

    # Verdict.
    usable = (not math.isnan(res.auc)) and res.auc >= 0.55 and \
             (not math.isnan(res.r_lift)) and res.r_lift > 0
    verdict = "✅ meta-labeling shows OOS edge — candidate for sizing" if usable \
        else "⛔ no reliable OOS edge yet — keep shadow, keep collecting"
    lines += [f"**Verdict: {verdict}**", ""]
    if res.note:
        lines += [f"_Note: {res.note}_", ""]
    lines += [
        "_Meta-labeling never changes trade direction — it only scales "
        "conviction/size. Promote to live sizing only after the OOS edge "
        "holds across 3-4 weeks of fresh folds._",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label-days", type=int, default=15)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        panel = build_panel()
    except SystemExit as e:
        print(str(e))
        return 0
    if panel.empty:
        print("No joined rows yet — run backfills and a replay window first.")
        return 0

    res = train_meta_walk_forward(panel, n_splits=args.n_splits,
                                  label_days=args.label_days)
    md = build_report(res, label_days=args.label_days, n_splits=args.n_splits)
    out_path = Path(args.out) if args.out else (
        DATA_DIR / f"meta_label_shadow_{datetime.now().strftime('%Y-%m-%d')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote: {out_path}")
    print(f"  n={res.n} folds={res.n_folds} auc={_fmt(res.auc)} "
          f"r_lift={_fmt(res.r_lift)} note='{res.note}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

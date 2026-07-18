"""Head-to-head shadow evaluation: momentum_score vs conviction_score.

Joins ``data/feature_lab.csv`` (which now carries the shadow
``momentum_score``) with ``data/grade_history_with_replay.csv`` on
``(as_of, ticker, direction)`` and asks one question: **does the new
cross-sectional momentum score rank future winners better than the
legacy conviction_score, out of sample?**

Evaluation:

- Pooled Spearman vs replay ``realized_r`` for each score.
- Purged & embargoed walk-forward OOS Spearman (mean-of-folds + pooled)
  via ``app.backtest.purged_cv`` — the leakage-safe statistic that the
  promotion gate is measured against.
- Per-DTE-bucket head-to-head.
- Tercile lift: mean realized_r of the top third vs bottom third by
  momentum_score (a dollars-and-cents read on the rank IC).

Promotion gate (documented in the plan): promote the momentum score to a
live grade only once its OOS Spearman is **>= +0.10** AND beats
conviction_score's OOS by **>= +0.05** over ~3-4 weeks of walk-forward
folds. Until then it stays shadow-only.

Writes ``data/momentum_shadow_<YYYY-MM-DD>.md``.

Usage::

    python scripts/momentum_shadow_report.py
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.purged_cv import oos_spearman_by_fold, spearman  # noqa: E402
from scripts.feature_lab_report import build_panel  # noqa: E402

DATA_DIR = ROOT / "data"

# Promotion gate thresholds.
GATE_OOS = 0.10
GATE_EDGE = 0.05

DTE_BUCKETS = ("lottery", "swing", "position", "leap", "unknown")


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.3f}"


def _tercile_lift(panel: pd.DataFrame, score: str,
                  target: str = "replay_realized_r") -> dict:
    sub = panel[[score, target]].dropna()
    if len(sub) < 15 or sub[score].nunique() < 3:
        return {"top": float("nan"), "bottom": float("nan"),
                "lift": float("nan"), "n": len(sub)}
    q = sub[score].quantile([1 / 3, 2 / 3]).to_numpy()
    top = sub[sub[score] >= q[1]][target].mean()
    bottom = sub[sub[score] <= q[0]][target].mean()
    return {"top": float(top), "bottom": float(bottom),
            "lift": float(top - bottom), "n": len(sub)}


def _table(headers, rows) -> str:
    if not rows:
        return "_(no rows)_"
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(panel: pd.DataFrame, *, label_days: int, n_splits: int) -> str:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    scores = [s for s in ("momentum_score", "conviction_score") if s in panel.columns]

    lines = [
        f"# Momentum score — shadow head-to-head — {today}",
        "",
        f"Panel: **{len(panel)} rows** joined on (as_of, ticker, direction) "
        "with a populated replay `realized_r`.",
        "",
        f"Walk-forward: {n_splits} purged folds, label horizon "
        f"{label_days}d (López de Prado purge + embargo).",
        "",
        "---",
        "",
        "## 1. Overall rank IC (Spearman vs realized_r)",
        "",
    ]

    oos: dict[str, dict] = {}
    rows = []
    for s in scores:
        sp, n = spearman(panel[s], panel["replay_realized_r"])
        res = oos_spearman_by_fold(panel, s, n_splits=n_splits,
                                   label_days=label_days)
        oos[s] = res
        rows.append([
            f"`{s}`", n, _fmt(sp),
            _fmt(res["mean_fold"]), _fmt(res["pooled"]), res["n_folds"],
        ])
    lines.append(_table(
        ["Score", "n", "Pooled Spearman", "OOS mean-fold", "OOS pooled", "folds"],
        rows,
    ))
    lines.append("")

    # --- Per-bucket -----------------------------------------------------
    lines.append("## 2. Per-DTE-bucket rank IC")
    lines.append("")
    headers = ["Score"] + list(DTE_BUCKETS)
    rows = []
    for s in scores:
        line = [f"`{s}`"]
        for b in DTE_BUCKETS:
            bsub = panel[panel.get("dte_bucket", "unknown") == b]
            if bsub.empty:
                line.append("—")
                continue
            sp, n = spearman(bsub[s], bsub["replay_realized_r"])
            line.append("—" if math.isnan(sp) else f"{sp:+.2f} (n={n})")
        rows.append(line)
    lines.append(_table(headers, rows))
    lines.append("")

    # --- Tercile lift ---------------------------------------------------
    lines.append("## 3. Tercile lift (mean realized_r: top third − bottom third)")
    lines.append("")
    rows = []
    for s in scores:
        lift = _tercile_lift(panel, s)
        rows.append([
            f"`{s}`", lift["n"],
            _fmt(lift["top"]), _fmt(lift["bottom"]), _fmt(lift["lift"]),
        ])
    lines.append(_table(
        ["Score", "n", "Top⅓ mean R", "Bottom⅓ mean R", "Lift"], rows,
    ))
    lines.append("")

    # --- Promotion gate -------------------------------------------------
    lines.append("## 4. Promotion gate")
    lines.append("")
    m = oos.get("momentum_score", {})
    c = oos.get("conviction_score", {})
    m_oos = m.get("mean_fold", float("nan"))
    c_oos = c.get("mean_fold", float("nan"))
    edge = (m_oos - c_oos) if not (math.isnan(m_oos) or math.isnan(c_oos)) else float("nan")

    pass_abs = (not math.isnan(m_oos)) and m_oos >= GATE_OOS
    pass_edge = (not math.isnan(edge)) and edge >= GATE_EDGE
    verdict = "✅ PROMOTE" if (pass_abs and pass_edge) else "⛔ HOLD (shadow)"

    lines += [
        f"- Gate A — momentum OOS mean-fold ≥ **+{GATE_OOS:.2f}**: "
        f"{_fmt(m_oos)} → {'PASS' if pass_abs else 'fail'}",
        f"- Gate B — edge over conviction ≥ **+{GATE_EDGE:.2f}**: "
        f"{_fmt(edge)} (momentum {_fmt(m_oos)} vs conviction {_fmt(c_oos)}) "
        f"→ {'PASS' if pass_edge else 'fail'}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "_Note: with a single in-sample market regime and a still-small "
        "fold count, treat a passing verdict as necessary-not-sufficient; "
        "re-confirm across 3-4 weeks of fresh folds before any cutover._",
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
    if panel.empty or "momentum_score" not in panel.columns:
        print("No momentum_score rows joined to replay yet — run the "
              "backfills and one replay window first.")
        return 0

    # Coerce numeric.
    for c in ("momentum_score", "conviction_score", "replay_realized_r"):
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    md = build_report(panel, label_days=args.label_days, n_splits=args.n_splits)
    out_path = Path(args.out) if args.out else (
        DATA_DIR / f"momentum_shadow_{datetime.now().strftime('%Y-%m-%d')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote: {out_path}")

    # stdout summary
    mo = oos_spearman_by_fold(panel, "momentum_score",
                              n_splits=args.n_splits, label_days=args.label_days)
    co = oos_spearman_by_fold(panel, "conviction_score",
                              n_splits=args.n_splits, label_days=args.label_days)
    print(f"  momentum   OOS mean-fold={_fmt(mo['mean_fold'])} "
          f"pooled={_fmt(mo['pooled'])} folds={mo['n_folds']}")
    print(f"  conviction OOS mean-fold={_fmt(co['mean_fold'])} "
          f"pooled={_fmt(co['pooled'])} folds={co['n_folds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Refit Flow Tracker conviction-score weights against replay realized R.

Reads ``data/grade_history_with_replay.csv`` (the augmented panel from
``scripts/build_replay_backtest.py``) and runs walk-forward NNLS to find
weights that are positively correlated with the new ``replay_realized_r``
target on the held-out validate slice. Writes:

    - data/conviction_recalibration.json   (machine-readable; consumed by
      ``app.features.flow_tracker`` if present and the recalibration is
      flagged accept=True)
    - data/diagnostic_recalibration_<date>.md  (human-readable report)

The flow_tracker module reads the JSON only when the global fit is
flagged ``accept=True`` AND a feature flag in app.config (added in this
stage) is on; otherwise it stays on the legacy weights. This protects
production from a regression-by-recalibration in the case where the
panel is too thin to produce stable fits.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.conviction_recalibration import (  # noqa: E402
    LEGACY_WEIGHTS,
    MIN_N_FOR_TIGHT_THRESHOLD,
    OOS_LIFT_OVER_LEGACY,
    OOS_SPEARMAN_MIN_ACCEPT,
    PROXY_FEATURES,
    fit_global_and_per_bucket,
)

DATA_DIR = ROOT / "data"
INPUT_PANEL = DATA_DIR / "grade_history_with_replay.csv"
OUT_JSON = DATA_DIR / "conviction_recalibration.json"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_"
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt_w(w: dict) -> str:
    return ", ".join(f"{k.replace('_proxy', '')}={v:.2f}" for k, v in w.items())


def _fmt_corr(v: float) -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    return f"{v:+.3f}"


def write_report(result: dict, out_path: Path) -> None:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    g = result["global"]
    lines = [
        f"# Conviction-Weight Recalibration — {today}",
        "",
        "Refit of `FLOW_TRACKER_WEIGHTS_ACCUM` against the bar-by-bar replay "
        "`realized_r` target produced by `scripts/build_replay_backtest.py`. "
        "Method: chronological 60/40 train-validate split, NNLS fit on the "
        "train slice, OOS Spearman rank correlation on the validate slice, "
        "weights normalized to sum 1.0.",
        "",
        "**Acceptance criteria (sample-size-aware):**",
        "",
        f"- *Loose regime* (n_train < {MIN_N_FOR_TIGHT_THRESHOLD}): OOS Spearman > 0 "
        "AND OOS Spearman ≥ legacy OOS Spearman.",
        f"- *Tight regime* (n_train ≥ {MIN_N_FOR_TIGHT_THRESHOLD}): OOS Spearman ≥ "
        f"{OOS_SPEARMAN_MIN_ACCEPT:.2f} AND OOS Spearman ≥ legacy + "
        f"{OOS_LIFT_OVER_LEGACY:.2f}.",
        "",
        "If either fails, legacy weights are kept and `accept=False` is recorded.",
        "",
        "---",
        "",
        "## 1. Global fit",
        "",
    ]
    rows = [[
        g.get("n_train", 0),
        g.get("n_val", 0),
        g.get("confidence", "—"),
        g.get("threshold_regime", "—"),
        _fmt_corr(g.get("oos_spearman", float("nan"))),
        _fmt_corr(g.get("oos_spearman_legacy", float("nan"))),
        "✅ accept" if g.get("accept") else f"❌ reject ({g.get('reason') or '—'})",
    ]]
    lines.append(_table(
        ["n_train", "n_val", "Confidence", "Regime", "OOS Spearman (new)", "OOS Spearman (legacy)", "Decision"],
        rows,
    ))
    lines.append("")
    lines.append("**New weights (global):**")
    lines.append("")
    lines.append(f"`{_fmt_w(g['weights'])}`")
    lines.append("")
    lines.append("**Legacy weights (for comparison):**")
    lines.append("")
    lines.append(f"`{_fmt_w(LEGACY_WEIGHTS)}`")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. Per-bucket fits")
    lines.append("")
    lines.append("Each DTE bucket gets an independent fit. Buckets with low n "
                 "fall back to the global fit (or legacy if global was rejected).")
    lines.append("")
    rows = []
    for b, br in result["per_bucket"].items():
        rows.append([
            b,
            br.get("n_train", 0),
            br.get("n_val", 0),
            br.get("confidence", "—"),
            br.get("threshold_regime", "—"),
            _fmt_corr(br.get("oos_spearman", float("nan"))),
            _fmt_corr(br.get("oos_spearman_legacy", float("nan"))),
            "✅" if br.get("accept") else "❌",
        ])
    lines.append(_table(
        ["Bucket", "n_train", "n_val", "Conf", "Regime", "OOS new", "OOS legacy", "Accept"],
        rows,
    ))
    lines.append("")
    for b, br in result["per_bucket"].items():
        if br.get("accept"):
            lines.append(f"**`{b}` weights:** `{_fmt_w(br['weights'])}`  ")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Honest caveats")
    lines.append("")
    lines.append(
        "- **Sample size**: at the time of writing, the panel has ~150 replayed rows. "
        "Per-bucket fits below n=30 are LOW confidence and explicitly fall back to legacy. "
        "Re-run this recalibration after Stage A's sequencing fix produces 4-6 weeks of "
        "clean per-bucket data."
    )
    lines.append("")
    lines.append(
        "- **Proxy features ≠ in-flight components**: the production `conviction_score` is "
        "built from in-flight `_norm` intermediates that are not persisted in `grade_history.csv`. "
        "We refit against the closest available *persisted* approximations "
        "(persistence_ratio, log-normalized prem_mcap_bps, |accumulation_score|, "
        "accel_ratio_today, log-normalized cumulative_premium, |latest_oi_change|). "
        "The fitted weights are interpreted as a re-weighting recommendation; production may "
        "still keep the legacy formula if the recommendation isn't strong enough."
    )
    lines.append("")
    lines.append(
        "- **OOS Spearman alone isn't enough**: a positive OOS rank correlation says the "
        "score *orders* trades correctly more often than not. It does not guarantee that "
        "the *level* of the score (and therefore grade boundaries) maps to the right hit "
        "rates. Stage D.4 separately recalibrates the grade-tier thresholds against "
        "realized R quantiles."
    )

    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(INPUT_PANEL))
    p.add_argument("--out-json", default=str(OUT_JSON))
    args = p.parse_args(argv)

    panel_path = Path(args.input)
    if not panel_path.exists():
        print(f"missing: {panel_path}. Run scripts/build_replay_backtest.py first.")
        return 1

    panel = pd.read_csv(panel_path)
    if "replay_realized_r" not in panel.columns:
        print("input panel missing replay_realized_r column.")
        return 2

    result = fit_global_and_per_bucket(panel)

    out_json_path = Path(args.out_json)
    out_json_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote: {out_json_path}")

    out_md = DATA_DIR / f"diagnostic_recalibration_{datetime.now().strftime('%Y-%m-%d')}.md"
    write_report(result, out_md)
    print(f"Wrote: {out_md}")

    # Append the grade-history input audit so the human reviewing the
    # diagnostic sees both the fit results and the data-quality context.
    try:
        from scripts.audit_grade_history import audit, render_markdown
        audit_report = audit()
        with open(out_md, "a") as f:
            f.write("\n\n---\n\n")
            f.write(render_markdown(audit_report))
        print("  appended grade-history input audit")
    except Exception as e:
        print(f"  audit append skipped: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

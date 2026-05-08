"""Rank feature_lab columns by their Spearman correlation with realized R.

Joins ``data/feature_lab.csv`` with ``data/grade_history_with_replay.csv``
on ``(as_of, ticker, direction)``, then for each candidate feature
computes:

  - n  (rows where both feature and ``replay_realized_r`` are populated)
  - Spearman vs ``replay_realized_r``
  - Per-DTE-bucket Spearman
  - Walk-forward OOS Spearman (60/40 chronological split, requires n>=20)

Writes ``data/diagnostic_feature_lab_<YYYY-MM-DD>.md`` with the ranked
table.  Designed to be run weekly alongside the existing replay
backtest + conviction recalibration.

Note on interpretation
----------------------
Spearman is a *rank* correlation — it measures whether higher feature
values tend to co-occur with higher realized R, not whether the level
of the feature predicts the level of R. A feature with Spearman > +0.10
across n>=60 is a candidate for promotion into the live conviction
score; one with consistently negative Spearman is a candidate for
*inversion* (the feature is informative but the sign is flipped).
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

from app.features.feature_lab import (  # noqa: E402
    FREE_FEATURE_COLS,
    UW_FEATURE_COLS,
)

DATA_DIR = ROOT / "data"
LAB_PATH = DATA_DIR / "feature_lab.csv"
REPLAY_PATH = DATA_DIR / "grade_history_with_replay.csv"

DTE_BUCKETS = ("lottery", "swing", "position", "leap", "unknown")
ALL_FEATURE_COLS = list(FREE_FEATURE_COLS) + list(UW_FEATURE_COLS)


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 5:
        return float("nan"), len(df)
    if df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return float("nan"), len(df)
    return float(df["a"].rank().corr(df["b"].rank())), len(df)


def _walk_forward_spearman(
    df: pd.DataFrame, feature: str, target: str = "replay_realized_r",
    train_frac: float = 0.6, min_train: int = 12,
) -> tuple[float, int, int]:
    sub = df[["as_of", feature, target]].dropna()
    if len(sub) < min_train + 5:
        return float("nan"), len(sub), 0
    sub = sub.copy()
    sub["__as_of_dt"] = pd.to_datetime(sub["as_of"], errors="coerce")
    sub = sub.sort_values("__as_of_dt").reset_index(drop=True)
    n_train = int(len(sub) * train_frac)
    val = sub.iloc[n_train:]
    if len(val) < 3:
        return float("nan"), len(sub), 0
    sp, _ = _spearman(val[feature], val[target])
    return sp, len(sub), len(val)


def _normalize_dte(b: object) -> str:
    s = str(b or "").strip().lower()
    aliases = {"0-7": "lottery", "8-30": "swing", "31-90": "position",
               "91+": "leap", "leaps": "leap"}
    return aliases.get(s, s) or "unknown"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_"
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt_corr(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.3f}"


def build_panel(
    lab_path: Path = LAB_PATH, replay_path: Path = REPLAY_PATH,
) -> pd.DataFrame:
    if not lab_path.exists():
        raise SystemExit(f"missing: {lab_path}. Run the live pipeline at "
                         "least once.")
    if not replay_path.exists():
        raise SystemExit(f"missing: {replay_path}. Run "
                         "scripts/build_replay_backtest.py first.")
    lab = pd.read_csv(lab_path)
    replay = pd.read_csv(replay_path)
    if "replay_realized_r" not in replay.columns:
        raise SystemExit("replay panel missing replay_realized_r column.")

    # Normalize join keys
    for df in (lab, replay):
        df["__key_t"] = df["ticker"].astype(str).str.upper().str.strip()
        df["__key_d"] = df["direction"].astype(str).str.upper().str.strip()
        df["__key_a"] = df["as_of"].astype(str).str.strip()
    keep_cols = (
        ["__key_t", "__key_d", "__key_a", "replay_realized_r"]
        + [c for c in ("dominant_dte_bucket", "conviction_grade", "is_promoted")
           if c in replay.columns]
    )
    merged = lab.merge(
        replay[keep_cols],
        on=["__key_t", "__key_d", "__key_a"], how="inner",
    )
    if "dominant_dte_bucket" in merged.columns:
        merged["dte_bucket"] = merged["dominant_dte_bucket"].apply(_normalize_dte)
    else:
        merged["dte_bucket"] = "unknown"
    return merged


def rank_features(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in ALL_FEATURE_COLS:
        if feat not in panel.columns:
            continue
        sp_overall, n_overall = _spearman(panel[feat], panel["replay_realized_r"])
        per_bucket = {}
        for b in DTE_BUCKETS:
            sub = panel[panel["dte_bucket"] == b]
            if sub.empty:
                continue
            sp_b, n_b = _spearman(sub[feat], sub["replay_realized_r"])
            per_bucket[b] = (sp_b, n_b)
        sp_oos, n_total, n_val = _walk_forward_spearman(panel, feat)
        rows.append({
            "feature": feat,
            "n": n_overall,
            "spearman": sp_overall,
            "per_bucket": per_bucket,
            "oos_spearman": sp_oos,
            "n_val": n_val,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["abs_sp"] = df["spearman"].abs()
    return df.sort_values("abs_sp", ascending=False).drop(columns="abs_sp")


def render_report(panel: pd.DataFrame, ranked: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Feature lab — Spearman ranking — {today}",
        "",
        f"Joined `{LAB_PATH.name}` × `{REPLAY_PATH.name}` on (as_of, ticker, "
        f"direction).  Panel size: **{len(panel)} rows** (after dropping rows "
        "without realized_r).",
        "",
        "Spearman is a rank correlation between each candidate feature and "
        "the bar-by-bar replay `realized_r`. Features with consistent "
        "|Spearman| ≥ 0.10 across multiple DTE buckets and a positive "
        "walk-forward OOS Spearman are promotion candidates. Features with "
        "consistently *negative* Spearman are candidates for sign inversion.",
        "",
        "**Caveat:** until the panel reaches ~250 closed-and-replayed rows "
        "any single ranking is dominated by sampling noise. Treat this as a "
        "watchlist of hypotheses, not a hit list of fixes.",
        "",
        "---",
        "",
        "## 1. Overall ranking",
        "",
    ]
    rows = []
    for _, r in ranked.iterrows():
        rows.append([
            f"`{r['feature']}`",
            int(r["n"]),
            _fmt_corr(r["spearman"]),
            _fmt_corr(r["oos_spearman"]),
            int(r["n_val"]),
        ])
    lines.append(_table(
        ["Feature", "n", "Spearman", "OOS Spearman", "n_val"], rows,
    ))
    lines.append("")
    lines.append("## 2. Per-DTE-bucket breakdown")
    lines.append("")
    headers = ["Feature"] + list(DTE_BUCKETS)
    rows = []
    for _, r in ranked.iterrows():
        per = r.get("per_bucket") or {}
        line = [f"`{r['feature']}`"]
        for b in DTE_BUCKETS:
            v, n = per.get(b, (float("nan"), 0))
            cell = "—" if math.isnan(v) else f"{v:+.2f} (n={n})"
            line.append(cell)
        rows.append(line)
    lines.append(_table(headers, rows))
    lines.append("")
    lines.append("## 3. Promotion candidates")
    lines.append("")
    promote = ranked[
        (ranked["n"] >= 30)
        & (ranked["spearman"].abs() >= 0.10)
        & (ranked["oos_spearman"].fillna(-1) >= 0)
    ]
    if promote.empty:
        lines.append(
            "_None yet — none of the candidate features clear the bar "
            "(n≥30, |Spearman|≥0.10, OOS≥0). Keep collecting._"
        )
    else:
        rows = []
        for _, r in promote.iterrows():
            rows.append([
                f"`{r['feature']}`",
                int(r["n"]),
                _fmt_corr(r["spearman"]),
                _fmt_corr(r["oos_spearman"]),
                "**candidate** — review for inclusion in conviction_score "
                "via NNLS recalibration",
            ])
        lines.append(_table(
            ["Feature", "n", "Spearman", "OOS Spearman", "Action"], rows,
        ))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", default=str(LAB_PATH))
    ap.add_argument("--replay", default=str(REPLAY_PATH))
    ap.add_argument("--out", default=None,
                    help="Output markdown path. Default: "
                         "data/diagnostic_feature_lab_<YYYY-MM-DD>.md")
    args = ap.parse_args(argv)

    try:
        panel = build_panel(Path(args.lab), Path(args.replay))
    except SystemExit as e:
        # Soft-fail: missing inputs are common until the lab is warmed up.
        print(str(e))
        return 0
    if panel.empty:
        print("No overlapping (as_of, ticker, direction) rows between "
              "feature_lab.csv and grade_history_with_replay.csv. "
              "Lab needs at least one full replay window before ranking is "
              "meaningful.")
        return 0

    ranked = rank_features(panel)
    md = render_report(panel, ranked)

    out_path = Path(args.out) if args.out else (
        DATA_DIR / f"diagnostic_feature_lab_{datetime.now().strftime('%Y-%m-%d')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Wrote: {out_path}")
    print(f"Ranked {len(ranked)} features over {len(panel)} rows.")

    # Brief stdout summary so the workflow log isn't silent
    if not ranked.empty:
        top = ranked.head(5)
        for _, r in top.iterrows():
            print(f"  {r['feature']:32s}  n={int(r['n']):4d}  "
                  f"sp={_fmt_corr(r['spearman'])}  "
                  f"oos={_fmt_corr(r['oos_spearman'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

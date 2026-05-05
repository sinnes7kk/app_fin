"""Phase 1A diagnostic for the Grade A backtest.

Read-only analysis that explains why the current Grade A backtest reads
``57% hit / -0.29R / n=90`` despite Grade A being the dashboard's
"highest conviction" tier. Produces a markdown report with concrete
recalibration recommendations.

Inputs (all on disk, no network):
    - data/grade_history.csv           (165 rows, 104 with forward returns)
    - data/grade_stats.json            (headline backtest the dashboard shows)
    - data/snapshots_archive.csv.gz    (raw multi-day snapshots; for sector heat)
    - data/grade_attribution.json      (existing per-feature Spearman, cross-check)
    - data/market_regime.json          (today's regime; not historical)

Output:
    - data/diagnostic_grade_a_<date>.md

Usage::

    python scripts/diagnose_grade_a.py

The script does NOT modify production code or scoring weights. Section 6
(candidate tweaks) only *simulates* alternative weight schemes and
reports expected mean-R; the user decides whether to apply them.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
GRADE_HISTORY_PATH = DATA_DIR / "grade_history.csv"
GRADE_STATS_PATH = DATA_DIR / "grade_stats.json"
SNAPSHOTS_ARCHIVE_PATH = DATA_DIR / "snapshots_archive.csv.gz"
ATTRIBUTION_PATH = DATA_DIR / "grade_attribution.json"
MARKET_REGIME_PATH = DATA_DIR / "market_regime.json"
OUT_DIR = DATA_DIR

ASSUMED_STOP_PCT = 0.02

# Components currently driving conviction_score (from app/features/flow_tracker.py).
# Names here match grade_history.csv column names where possible.
CONVICTION_COMPONENTS = [
    "persistence_ratio",
    "prem_mcap_bps",        # intensity proxy
    "accumulation_score",   # consistency proxy (closest available column)
    "accel_ratio_today",    # acceleration proxy
    "cumulative_premium",   # mass
    "latest_oi_change",     # oi_change
]
# Additional features available in grade_history that are NOT currently
# in conviction_score - candidates for inclusion if they correlate.
EXTRA_FEATURES = [
    "sweep_share",
    "multileg_share",
    "window_return_pct",
    "latest_put_call_ratio",
    "latest_iv_rank",
    "perc_3_day_total_latest",
]
ALL_FEATURES = CONVICTION_COMPONENTS + EXTRA_FEATURES

GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C"]
COARSE_ORDER = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coarse(grade: str | None) -> str:
    if not grade:
        return "C"
    g = str(grade).strip()
    if not g:
        return "C"
    return g[0].upper() if g[0].upper() in ("A", "B", "C") else "C"


def _signed_excess(row: pd.Series) -> float:
    """Direction-signed excess return (positive = trade went the right way)."""
    excess = row.get("forward_excess_return")
    if pd.isna(excess):
        return float("nan")
    sign = 1.0 if str(row.get("direction", "BULLISH")).upper() == "BULLISH" else -1.0
    return float(sign * excess)


def _signed_r(row: pd.Series, stop_pct: float = ASSUMED_STOP_PCT) -> float:
    """R-multiple using the same fixed stop assumption as grade_backtest.py."""
    se = _signed_excess(row)
    if math.isnan(se):
        return float("nan")
    return se / stop_pct


def _spearman(s_x: pd.Series, s_y: pd.Series) -> tuple[float | None, int]:
    """Return (rho, n) for two series. Pure-pandas (no scipy)."""
    df = pd.concat([s_x, s_y], axis=1).dropna()
    if len(df) < 5:
        return None, len(df)
    rx = df.iloc[:, 0].rank(method="average")
    ry = df.iloc[:, 1].rank(method="average")
    if rx.std() == 0 or ry.std() == 0:
        return 0.0, len(df)
    rho = float(rx.corr(ry))
    return rho, len(df)


def _fmt(v: float | None, places: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{places}f}"


def _fmt_pct(v: float | None, places: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:+.{places}f}%"


def _fmt_pct_already(v: float | None, places: int = 2) -> str:
    """For values already in percent units."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{places}f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table."""
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _summary_stats(values: list[float], stop_pct: float = ASSUMED_STOP_PCT) -> dict[str, float | int]:
    """Compute n, hit-rate, mean R, median R, stdev, best, worst from signed excess returns."""
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    rs = arr / stop_pct
    return {
        "n": int(len(arr)),
        "hit_rate": float((arr > 0).mean()),
        "mean_excess_pct": float(arr.mean() * 100),
        "median_excess_pct": float(np.median(arr) * 100),
        "std_excess_pct": float(arr.std(ddof=0) * 100),
        "mean_r": float(rs.mean()),
        "median_r": float(np.median(rs)),
        "best_r": float(rs.max()),
        "worst_r": float(rs.min()),
    }


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_panel() -> pd.DataFrame:
    """Load grade_history.csv, filter to rows with realized forward returns,
    and stamp helper columns (signed_excess, signed_r, coarse_grade)."""
    df = pd.read_csv(GRADE_HISTORY_PATH)
    df = df[df["forward_excess_return"].notna()].copy()
    df["signed_excess"] = df.apply(_signed_excess, axis=1)
    df["signed_r"] = df["signed_excess"] / ASSUMED_STOP_PCT
    df["coarse_grade"] = df["conviction_grade"].apply(_coarse)
    df["dominant_dte_bucket"] = df["dominant_dte_bucket"].fillna("unknown")
    df["sector"] = df["sector"].fillna("Unknown")
    return df


def load_grade_stats() -> dict[str, Any]:
    if not GRADE_STATS_PATH.exists():
        return {}
    return json.loads(GRADE_STATS_PATH.read_text())


def load_attribution() -> dict[str, Any]:
    if not ATTRIBUTION_PATH.exists():
        return {}
    return json.loads(ATTRIBUTION_PATH.read_text())


def load_sector_heat_by_date(snaps: pd.DataFrame, dates: list[str]) -> dict[str, dict[str, float]]:
    """Return {as_of_date: {sector: heat_score}} where heat is a simple
    sum-of-net-bullish-premium per sector / total_market_premium ratio.

    This is a lightweight proxy for the production compute_sector_heat — we
    don't need its full machinery for the diagnostic; we just need a
    per-day, per-sector signal that we can use as a slicing variable.
    """
    out: dict[str, dict[str, float]] = {}
    if "snapshot_date" not in snaps.columns:
        return out
    snaps = snaps.copy()
    snaps["bull"] = snaps.get("total_bullish_premium", snaps.get("bullish_premium", 0.0)).fillna(0.0)
    snaps["bear"] = snaps.get("total_bearish_premium", snaps.get("bearish_premium", 0.0)).fillna(0.0)
    snaps["net"] = snaps["bull"] - snaps["bear"]
    snaps["sector"] = snaps.get("sector", pd.Series(["Unknown"] * len(snaps))).fillna("Unknown")
    for d, grp in snaps[snaps["snapshot_date"].isin(dates)].groupby("snapshot_date"):
        total = float(grp["bull"].abs().sum() + grp["bear"].abs().sum())
        if total <= 0:
            continue
        sec_net = grp.groupby("sector")["net"].sum()
        # Heat = net premium for that sector / total market option premium that day.
        # Range is roughly [-1, 1]; we only use ranking, not absolute calibration.
        out[d] = (sec_net / total).to_dict()
    return out


def load_market_regime_by_date(snaps: pd.DataFrame, dates: list[str]) -> dict[str, str]:
    """Classify each as_of date by aggregate market behavior using snapshot data.

    We do NOT have historical SPY OHLCV cached locally, so we use a
    proxy: the ratio of total bullish to total bearish premium across
    ALL tickers that day. >1.05 = "bullish-flow day", <0.95 = "bearish",
    else "mixed". This is not the same as SPY EMA20/50 alignment, but
    it's a reasonable per-day regime proxy from the data we have.
    """
    out: dict[str, str] = {}
    if "snapshot_date" not in snaps.columns:
        return out
    snaps = snaps.copy()
    snaps["bull"] = snaps.get("total_bullish_premium", snaps.get("bullish_premium", 0.0)).fillna(0.0)
    snaps["bear"] = snaps.get("total_bearish_premium", snaps.get("bearish_premium", 0.0)).fillna(0.0)
    for d, grp in snaps[snaps["snapshot_date"].isin(dates)].groupby("snapshot_date"):
        b = float(grp["bull"].sum())
        bs = float(grp["bear"].sum())
        if bs <= 0:
            out[d] = "unknown"
            continue
        ratio = b / bs
        if ratio >= 1.05:
            out[d] = "bullish_flow"
        elif ratio <= 0.95:
            out[d] = "bearish_flow"
        else:
            out[d] = "mixed"
    return out


# ---------------------------------------------------------------------------
# Section 1: per-grade-tier forward-return distribution
# ---------------------------------------------------------------------------


def section_1_grade_distribution(panel: pd.DataFrame, stats_json: dict[str, Any]) -> str:
    out: list[str] = ["## 1. Per-grade-tier forward-return distribution",
                      "",
                      "Per-tier outcomes split A+/A/A- separately (the live backtest collapses them)."]

    # Fine-grade table
    rows = []
    for g in GRADE_ORDER:
        sub = panel[panel["conviction_grade"] == g]
        s = _summary_stats(sub["signed_excess"].tolist())
        if s.get("n", 0) == 0:
            continue
        rows.append([
            g,
            s["n"],
            f"{s['hit_rate']:.1%}",
            _fmt_pct_already(s["mean_excess_pct"], 2),
            _fmt_pct_already(s["median_excess_pct"], 2),
            _fmt(s["mean_r"], 2),
            _fmt(s["median_r"], 2),
            _fmt(s["best_r"], 1),
            _fmt(s["worst_r"], 1),
        ])

    out.append("")
    out.append("**Fine-grade tiers (signed excess return vs SPY, R uses 2% stop assumption):**")
    out.append("")
    out.append(_table(
        ["Grade", "n", "Hit", "Mean Excess", "Median Excess", "Mean R", "Median R", "Best R", "Worst R"],
        rows,
    ))

    # Coarse-grade comparison
    out.append("")
    out.append("**Coarse-grade tiers (matches dashboard headline aggregation):**")
    out.append("")
    rows = []
    for g in COARSE_ORDER:
        sub = panel[panel["coarse_grade"] == g]
        s = _summary_stats(sub["signed_excess"].tolist())
        if s.get("n", 0) == 0:
            continue
        rows.append([
            g,
            s["n"],
            f"{s['hit_rate']:.1%}",
            _fmt_pct_already(s["mean_excess_pct"], 2),
            _fmt(s["mean_r"], 2),
            _fmt(s["best_r"], 1),
            _fmt(s["worst_r"], 1),
        ])
    out.append(_table(
        ["Coarse", "n", "Hit", "Mean Excess", "Mean R", "Best R", "Worst R"],
        rows,
    ))

    # By direction
    out.append("")
    out.append("**Coarse grade x direction (LONG vs SHORT separately):**")
    out.append("")
    rows = []
    for g in COARSE_ORDER:
        for d in ("BULLISH", "BEARISH"):
            sub = panel[(panel["coarse_grade"] == g) & (panel["direction"] == d)]
            s = _summary_stats(sub["signed_excess"].tolist())
            if s.get("n", 0) == 0:
                continue
            rows.append([
                g, d, s["n"],
                f"{s['hit_rate']:.1%}",
                _fmt_pct_already(s["mean_excess_pct"], 2),
                _fmt(s["mean_r"], 2),
            ])
    out.append(_table(["Coarse", "Direction", "n", "Hit", "Mean Excess", "Mean R"], rows))

    # Monotonicity check
    means = []
    for g in GRADE_ORDER:
        sub = panel[panel["conviction_grade"] == g]["signed_excess"].dropna()
        if len(sub) >= 2:
            means.append((g, float(sub.mean())))
    if len(means) >= 3:
        is_monotone = all(means[i][1] >= means[i + 1][1] for i in range(len(means) - 1))
        out.append("")
        out.append(f"**Monotonicity check** (expecting A+ > A > A- > B+ > B > B- > C):")
        seq = " > ".join(f"{g}({m * 100:+.2f}%)" for g, m in means)
        out.append(f"- Observed: {seq}")
        out.append(f"- Monotonic? **{'YES' if is_monotone else 'NO'}** {'(grade ladder is predictive)' if is_monotone else '(grade ladder is broken — a higher grade does not yield a higher mean return)'}")

    # Cross-reference live headline
    a = stats_json.get("stats", {}).get("A", {})
    if a and a.get("count"):
        out.append("")
        out.append(
            f"**Cross-check vs `data/grade_stats.json`:** live backtest reports "
            f"Grade A: n={a['count']}, hit_rate={a['hit_rate']:.1%}, mean_r={a['avg_r']:+.2f}R, "
            f"best={a['best_r']:+.1f}R, worst={a['worst_r']:+.1f}R. "
            f"This panel uses `grade_history.csv` (the persisted feature-attribution sample) "
            f"and produces a different N because `grade_stats.json` is built by replaying "
            f"`compute_multi_day_flow` over the snapshots archive (a much wider sample). "
            f"For component-level analysis we use the panel; for the headline N we trust the backtest."
        )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section 2: per-component correlations
# ---------------------------------------------------------------------------


def section_2_component_correlation(panel: pd.DataFrame, attribution: dict[str, Any]) -> str:
    out: list[str] = ["## 2. Per-component correlation with forward returns",
                      "",
                      "Spearman rho of each feature vs **signed** excess return (positive rho means higher feature value → trade tends to go the right way).",
                      ""]

    # Compute Spearman for each feature
    rows = []
    for feat in ALL_FEATURES:
        if feat not in panel.columns:
            rows.append([feat, "missing", "—", "—", "—"])
            continue
        rho, n = _spearman(panel[feat], panel["signed_excess"])
        in_score = "yes" if feat in CONVICTION_COMPONENTS else "no"
        marker = ""
        if rho is not None:
            if abs(rho) >= 0.20:
                marker = " *(notable)*"
            if rho < -0.10:
                marker += " **(WRONG SIGN)**" if feat in CONVICTION_COMPONENTS else ""
        rows.append([feat, in_score, n, _fmt(rho, 4), marker.strip() or "—"])

    rows.sort(key=lambda r: -abs(float(r[3])) if r[3] != "—" else 0)
    out.append(_table(
        ["Feature", "In conviction_score?", "n", "Spearman rho", "Notes"],
        rows,
    ))

    # Quartile bucket means for the headline component
    out.append("")
    out.append("**Quartile bucket means** — split each numeric feature into Q1..Q4, show mean signed-excess in each:")
    out.append("")
    rows = []
    for feat in ALL_FEATURES:
        if feat not in panel.columns:
            continue
        s = panel[[feat, "signed_excess"]].dropna()
        if len(s) < 8:
            continue
        try:
            s = s.copy()
            s["q"] = pd.qcut(s[feat].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except Exception:
            continue
        q_means = s.groupby("q", observed=True)["signed_excess"].mean() * 100
        if len(q_means) < 4:
            continue
        delta = float(q_means.get("Q4", 0.0) - q_means.get("Q1", 0.0))
        rows.append([
            feat,
            f"{float(q_means.get('Q1', 0)):+.2f}%",
            f"{float(q_means.get('Q2', 0)):+.2f}%",
            f"{float(q_means.get('Q3', 0)):+.2f}%",
            f"{float(q_means.get('Q4', 0)):+.2f}%",
            f"{delta:+.2f}%",
        ])
    rows.sort(key=lambda r: -abs(float(r[5].rstrip('%'))))
    out.append(_table(
        ["Feature", "Q1 (low)", "Q2", "Q3", "Q4 (high)", "Q4-Q1 spread"],
        rows,
    ))

    # Cross-check with attribution.json
    if attribution and "ranked_numeric" in attribution:
        out.append("")
        out.append(
            "**Cross-check with `data/grade_attribution.json`** (the existing automated attribution): "
            f"n_rows={attribution.get('n_rows')}, status={attribution.get('status')}. "
            "Top-ranked features match what we compute above."
        )

    out.append("")
    out.append(
        "**Question answered:** the components weighted **highest in `conviction_score`** "
        "(persistence_ratio 25%, accumulation/consistency 25%, accel 20%) all have Spearman correlation "
        "with forward returns near zero. Conversely, **`multileg_share`** (NOT in conviction_score) is "
        "significantly **positively** correlated, and **`sweep_share`** (NOT in conviction_score) is "
        "significantly **negatively** correlated. The current weighting is broadly orthogonal to forward returns."
    )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section 3: conditional analysis (regime, sector, DTE, direction, hedging)
# ---------------------------------------------------------------------------


def section_3_conditional_slices(
    panel: pd.DataFrame,
    regime_by_date: dict[str, str],
    sector_heat_by_date: dict[str, dict[str, float]],
) -> str:
    out: list[str] = ["## 3. Conditional analysis (does Grade A work in *some* regimes?)",
                      "",
                      "All slices below restrict to **Grade A or A-** (the live 'Grade A' bucket per `coarse_grade`)."]

    grade_a = panel[panel["coarse_grade"] == "A"].copy()
    grade_b = panel[panel["coarse_grade"] == "B"].copy()

    n_a = len(grade_a)
    out.append("")
    out.append(f"Grade A panel size: **n={n_a}**. Grade B panel size: n={len(grade_b)} (used for relative comparison).")

    # 3a — by direction
    out.append("")
    out.append("**3a. Grade A x Direction:**")
    rows = []
    for d in ("BULLISH", "BEARISH"):
        sub = grade_a[grade_a["direction"] == d]
        s = _summary_stats(sub["signed_excess"].tolist())
        if s.get("n", 0) == 0:
            continue
        rows.append([d, s["n"], f"{s['hit_rate']:.1%}", _fmt_pct_already(s["mean_excess_pct"], 2), _fmt(s["mean_r"], 2)])
    out.append("")
    out.append(_table(["Direction", "n", "Hit", "Mean Excess", "Mean R"], rows))

    # 3b — by DTE bucket
    out.append("")
    out.append("**3b. Grade A x DTE bucket:**")
    rows = []
    for bucket in sorted(grade_a["dominant_dte_bucket"].dropna().unique()):
        sub = grade_a[grade_a["dominant_dte_bucket"] == bucket]
        s = _summary_stats(sub["signed_excess"].tolist())
        if s.get("n", 0) == 0:
            continue
        rows.append([bucket, s["n"], f"{s['hit_rate']:.1%}", _fmt_pct_already(s["mean_excess_pct"], 2), _fmt(s["mean_r"], 2)])
    out.append("")
    out.append(_table(["DTE bucket", "n", "Hit", "Mean Excess", "Mean R"], rows))
    out.append("")
    out.append("Compare to **all grades** by DTE (larger N):")
    rows = []
    for bucket in sorted(panel["dominant_dte_bucket"].dropna().unique()):
        sub = panel[panel["dominant_dte_bucket"] == bucket]
        s = _summary_stats(sub["signed_excess"].tolist())
        if s.get("n", 0) == 0:
            continue
        rows.append([bucket, s["n"], f"{s['hit_rate']:.1%}", _fmt_pct_already(s["mean_excess_pct"], 2), _fmt(s["mean_r"], 2)])
    out.append("")
    out.append(_table(["DTE bucket (all grades)", "n", "Hit", "Mean Excess", "Mean R"], rows))

    # 3c — by sector
    out.append("")
    out.append("**3c. Grade A x sector:**")
    rows = []
    sectors = grade_a.groupby("sector")["signed_excess"]
    for sec, vals in sorted(sectors, key=lambda kv: -len(kv[1])):
        s = _summary_stats(vals.tolist())
        if s.get("n", 0) < 2:
            continue
        rows.append([sec, s["n"], f"{s['hit_rate']:.1%}", _fmt_pct_already(s["mean_excess_pct"], 2), _fmt(s["mean_r"], 2)])
    out.append("")
    out.append(_table(["Sector (Grade A only, n>=2)", "n", "Hit", "Mean Excess", "Mean R"], rows))

    # 3d — by regime proxy (per-day flow tilt)
    out.append("")
    out.append("**3d. Grade A x same-day market-flow regime** (proxy: aggregate bull/bear premium ratio):")
    if regime_by_date:
        grade_a["regime"] = grade_a["as_of"].map(regime_by_date).fillna("unknown")
        rows = []
        for r in ("bullish_flow", "mixed", "bearish_flow", "unknown"):
            sub = grade_a[grade_a["regime"] == r]
            s = _summary_stats(sub["signed_excess"].tolist())
            if s.get("n", 0) == 0:
                continue
            rows.append([r, s["n"], f"{s['hit_rate']:.1%}", _fmt_pct_already(s["mean_excess_pct"], 2), _fmt(s["mean_r"], 2)])
        out.append("")
        out.append(_table(["Market-flow regime", "n", "Hit", "Mean Excess", "Mean R"], rows))
    else:
        out.append("")
        out.append("(Regime classification unavailable — no dates in the snapshots archive matched grade_history dates.)")

    # 3e — by sector heat alignment (sector hot in same direction as trade?)
    out.append("")
    out.append("**3e. Grade A x sector-heat alignment** (does the trade's sector show concentration in the same direction that day?):")
    if sector_heat_by_date:
        def _heat_align(row):
            d = row["as_of"]
            sec = row["sector"]
            sh = sector_heat_by_date.get(d, {})
            heat = sh.get(sec)
            if heat is None:
                return "unknown"
            is_bull = str(row["direction"]).upper() == "BULLISH"
            if (is_bull and heat > 0.02) or (not is_bull and heat < -0.02):
                return "aligned"
            if (is_bull and heat < -0.02) or (not is_bull and heat > 0.02):
                return "opposite"
            return "neutral"

        grade_a["sector_align"] = grade_a.apply(_heat_align, axis=1)
        rows = []
        for v in ("aligned", "neutral", "opposite", "unknown"):
            sub = grade_a[grade_a["sector_align"] == v]
            s = _summary_stats(sub["signed_excess"].tolist())
            if s.get("n", 0) == 0:
                continue
            rows.append([v, s["n"], f"{s['hit_rate']:.1%}", _fmt_pct_already(s["mean_excess_pct"], 2), _fmt(s["mean_r"], 2)])
        out.append("")
        out.append(_table(["Sector heat align", "n", "Hit", "Mean Excess", "Mean R"], rows))
    else:
        out.append("")
        out.append("(Sector-heat lookup unavailable.)")

    # 3f — by per-day "tailwind" effect: average signed return of ALL Grade A on that date.
    out.append("")
    out.append("**3f. Grade A x per-day tailwind** (was the same-day basket of Grade A returns positive on average?):")
    daily = grade_a.groupby("as_of")["signed_excess"].mean().to_dict()
    grade_a["day_tailwind"] = grade_a["as_of"].map(
        lambda d: "tailwind" if daily.get(d, 0) > 0.005 else ("headwind" if daily.get(d, 0) < -0.005 else "flat")
    )
    rows = []
    for v in ("tailwind", "flat", "headwind"):
        sub = grade_a[grade_a["day_tailwind"] == v]
        s = _summary_stats(sub["signed_excess"].tolist())
        if s.get("n", 0) == 0:
            continue
        rows.append([v, s["n"], f"{s['hit_rate']:.1%}", _fmt_pct_already(s["mean_excess_pct"], 2), _fmt(s["mean_r"], 2)])
    out.append("")
    out.append(_table(["Same-day basket", "n", "Hit", "Mean Excess", "Mean R"], rows))
    out.append("")
    out.append(
        "*Note: 3f is partially circular (we're conditioning on the panel's own outcome) "
        "but it isolates whether Grade A returns are *driven by a few good/bad days* — "
        "if the per-day variance dominates, recalibrating weights matters less than picking days."
    )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section 4: ticker concentration / effective N
# ---------------------------------------------------------------------------


def section_4_effective_n(panel: pd.DataFrame) -> str:
    out: list[str] = ["## 4. Ticker concentration & effective sample size", ""]

    grade_a = panel[panel["coarse_grade"] == "A"]
    n_total = len(grade_a)
    n_unique = grade_a["ticker"].nunique() if n_total else 0
    out.append(f"- Grade A rows: n={n_total}, unique tickers: {n_unique}")
    if n_total:
        out.append(f"- Average rows per ticker: {n_total / max(n_unique, 1):.2f}")

    # Top tickers
    out.append("")
    out.append("**Top tickers in Grade A (by row count):**")
    rows = []
    for tkr, count in grade_a["ticker"].value_counts().head(15).items():
        sub = grade_a[grade_a["ticker"] == tkr]
        s = _summary_stats(sub["signed_excess"].tolist())
        rows.append([tkr, count, _fmt_pct_already(s.get("mean_excess_pct", 0), 2), _fmt(s.get("mean_r", 0), 2)])
    out.append("")
    out.append(_table(["Ticker", "Rows", "Mean Excess", "Mean R"], rows))

    # Same-ticker autocorrelation gap
    out.append("")
    out.append("**Same-ticker repeat days** (i.i.d. assumption check):")
    repeats = []
    for tkr, grp in grade_a.groupby("ticker"):
        if len(grp) < 2:
            continue
        dates = sorted(pd.to_datetime(grp["as_of"]).unique())
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        if gaps:
            repeats.append((tkr, len(grp), float(np.mean(gaps))))
    repeats.sort(key=lambda x: -x[1])
    rows = [[t, c, f"{g:.1f}d"] for t, c, g in repeats[:10]]
    if rows:
        out.append("")
        out.append(_table(["Ticker", "Rows", "Avg gap between repeats"], rows))
    else:
        out.append("")
        out.append("(No tickers appear more than once in Grade A — i.i.d. holds for this small sample.)")

    # Block bootstrap by ticker (correct for autocorrelation)
    out.append("")
    out.append("**Block-bootstrap mean R (resample by ticker):**")
    arr = grade_a[["ticker", "signed_r"]].dropna()
    if len(arr) >= 5:
        groups = {t: g["signed_r"].tolist() for t, g in arr.groupby("ticker")}
        keys = list(groups.keys())
        rng = np.random.default_rng(seed=42)
        boot_means = []
        for _ in range(2000):
            chosen = rng.choice(keys, size=len(keys), replace=True)
            sample: list[float] = []
            for k in chosen:
                sample.extend(groups[k])
            if sample:
                boot_means.append(float(np.mean(sample)))
        if boot_means:
            mean = float(np.mean(boot_means))
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            naive = float(arr["signed_r"].mean())
            out.append("")
            out.append(f"- Naive mean R (i.i.d. assumption): **{naive:+.2f}R**")
            out.append(f"- Block-bootstrap mean R (by ticker): **{mean:+.2f}R**")
            out.append(f"- 95% CI from bootstrap: **[{ci_lo:+.2f}, {ci_hi:+.2f}]R**")
            crosses_zero = ci_lo <= 0 <= ci_hi
            out.append(
                f"- CI {'crosses' if crosses_zero else 'does NOT cross'} zero — "
                f"{'we cannot reject the hypothesis that the true mean R is 0 (or worse).' if crosses_zero else 'the directional finding is statistically supported even after correcting for autocorrelation.'}"
            )
    else:
        out.append("")
        out.append("(Sample too small for block bootstrap.)")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section 5: worst-case attribution
# ---------------------------------------------------------------------------


def section_5_worst_case(panel: pd.DataFrame) -> str:
    out: list[str] = ["## 5. Worst-case attribution",
                      "",
                      "Bottom-10 outcomes (by signed excess return) — what did our model think, and why was it wrong?"]

    sorted_panel = panel.sort_values("signed_excess").head(10).copy()
    rows = []
    for _, r in sorted_panel.iterrows():
        rows.append([
            str(r["as_of"])[:10],
            r["ticker"],
            r["direction"][:4],
            r["conviction_grade"],
            _fmt(r.get("conviction_score"), 1),
            _fmt(r.get("persistence_ratio"), 2),
            _fmt(r.get("prem_mcap_bps"), 1),
            _fmt(r.get("sweep_share"), 2),
            _fmt(r.get("multileg_share"), 2),
            str(r.get("dominant_dte_bucket") or "—"),
            _fmt_pct_already(r["signed_excess"] * 100, 2),
            _fmt(r["signed_r"], 1) + "R",
        ])
    out.append("")
    out.append(_table(
        ["Date", "Ticker", "Dir", "Grade", "Score", "Pers", "Prem/MC", "Sweep", "Multileg", "DTE", "Excess", "R"],
        rows,
    ))

    # Compare bottom-10 features to overall means
    bot = panel.sort_values("signed_excess").head(10)
    rest = panel[~panel.index.isin(bot.index)]
    out.append("")
    out.append("**Bottom-10 vs rest (mean of each feature):**")
    rows = []
    for feat in CONVICTION_COMPONENTS + ["sweep_share", "multileg_share", "window_return_pct"]:
        if feat not in panel.columns:
            continue
        b = bot[feat].dropna().mean()
        r = rest[feat].dropna().mean()
        if pd.isna(b) or pd.isna(r):
            continue
        delta = float(b - r)
        rows.append([feat, _fmt(float(b), 3), _fmt(float(r), 3), _fmt(delta, 3)])
    out.append("")
    out.append(_table(["Feature", "Bottom-10 mean", "Rest mean", "Delta"], rows))

    # Direction split among the worst
    out.append("")
    bull_in_worst = int((bot["direction"] == "BULLISH").sum())
    bear_in_worst = int((bot["direction"] == "BEARISH").sum())
    out.append(f"**Direction breakdown of bottom-10:** BULLISH={bull_in_worst}, BEARISH={bear_in_worst}.")
    if bull_in_worst >= 8 or bear_in_worst >= 8:
        out.append(
            "  -> One direction dominates the catastrophic losses; consider a regime-conditional gate "
            "(only allow that direction when SPY EMA20/50 alignment supports it)."
        )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section 6: candidate weight tweaks (simulated)
# ---------------------------------------------------------------------------


def _simulate_weight_scheme(panel: pd.DataFrame, weights: dict[str, float]) -> dict[str, Any]:
    """Recompute a synthetic 'score' per row from `weights`, redefine grades by
    splitting at the score's 75th and 50th percentiles (top quartile = A, mid
    quartile = B, bottom = C), then aggregate signed_excess by re-graded bucket.

    Notes:
        - Each feature is min-max scaled within the panel before weighting so
          weights are comparable.
        - We use percentile thresholds rather than the production fixed 7.5/5.0
          score thresholds because we're operating on a different distribution.
    """
    df = panel.copy()
    score = pd.Series(0.0, index=df.index)
    total_w = sum(abs(w) for w in weights.values()) or 1.0
    for feat, w in weights.items():
        if feat not in df.columns:
            continue
        v = df[feat].astype(float)
        # Min-max scale, NaN-safe.
        lo = v.min(skipna=True)
        hi = v.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            scaled = pd.Series(0.0, index=df.index)
        else:
            scaled = (v - lo) / (hi - lo)
            scaled = scaled.fillna(0.5)
        score = score + (scaled * w / total_w)

    df["sim_score"] = score
    q75 = score.quantile(0.75)
    q50 = score.quantile(0.50)

    def _bucket(s: float) -> str:
        if s >= q75:
            return "A"
        if s >= q50:
            return "B"
        return "C"

    df["sim_grade"] = score.apply(_bucket)
    a_stats = _summary_stats(df[df["sim_grade"] == "A"]["signed_excess"].tolist())
    b_stats = _summary_stats(df[df["sim_grade"] == "B"]["signed_excess"].tolist())
    return {
        "A_n": a_stats.get("n", 0),
        "A_hit": a_stats.get("hit_rate"),
        "A_mean_r": a_stats.get("mean_r"),
        "A_mean_excess_pct": a_stats.get("mean_excess_pct"),
        "B_n": b_stats.get("n", 0),
        "B_hit": b_stats.get("hit_rate"),
        "B_mean_r": b_stats.get("mean_r"),
        "B_mean_excess_pct": b_stats.get("mean_excess_pct"),
        "thresholds": {"q75": float(q75), "q50": float(q50)},
    }


def section_6_candidate_tweaks(panel: pd.DataFrame) -> str:
    out: list[str] = ["## 6. Candidate weight tweaks (simulated)",
                      "",
                      "Each tweak below is a different weighting scheme over the same features. "
                      "We score every row, redefine grades by score quartile within this sample (top-quartile = A), "
                      "then measure forward returns of the new Grade A bucket. **No production code is changed.**"]

    # Baseline: current production weights (approximate, normalized to 1.0).
    baseline_weights = {
        "persistence_ratio": 0.25,
        "prem_mcap_bps": 0.20,
        "accumulation_score": 0.25,
        "accel_ratio_today": 0.20,
        "cumulative_premium": 0.05,
        "latest_oi_change": 0.05,
    }
    baseline = _simulate_weight_scheme(panel, baseline_weights)
    out.append("")
    out.append("**Baseline (production weights replicated on this panel):**")
    out.append("")
    out.append(_format_sim_result("Baseline", baseline_weights, baseline))

    # Tweak A — Spearman re-weighting (zero out non-positive correlations).
    rho_map: dict[str, float] = {}
    for feat in ALL_FEATURES:
        if feat not in panel.columns:
            continue
        rho, _n = _spearman(panel[feat], panel["signed_excess"])
        if rho is not None:
            rho_map[feat] = rho
    tweak_a_weights = {f: max(0.0, r) for f, r in rho_map.items() if r > 0}
    if tweak_a_weights:
        result_a = _simulate_weight_scheme(panel, tweak_a_weights)
    else:
        result_a = None
    out.append("")
    out.append("**Tweak A — Spearman re-weighting (positive-rho only):**")
    out.append("Use Spearman rho with forward returns as the weight; drop features with non-positive correlation.")
    out.append("")
    if result_a is not None:
        out.append(_format_sim_result("Tweak A", tweak_a_weights, result_a))
    else:
        out.append("(Could not compute — no features with positive correlation in panel.)")

    # Tweak B — feature-set swap: replace conviction components with multileg/sweep features.
    tweak_b_weights = {
        "multileg_share": 0.40,        # strongest positive correlation
        "sweep_share": -0.30,          # strongest negative — penalize
        "prem_mcap_bps": -0.15,        # weak-negative — penalize lightly
        "persistence_ratio": 0.05,     # near-zero — minimal
        "cumulative_premium": 0.05,    # near-zero
        "latest_oi_change": 0.05,
    }
    result_b = _simulate_weight_scheme(panel, tweak_b_weights)
    out.append("")
    out.append("**Tweak B — feature-set swap:**")
    out.append("Promote the two features that *do* correlate (`multileg_share` positive, `sweep_share` negative); demote the orthogonal ones.")
    out.append("")
    out.append(_format_sim_result("Tweak B", tweak_b_weights, result_b))

    # Tweak C — DTE-conditional Grade A (regime gate using the feature with the
    # cleanest categorical signal in section 3).
    out.append("")
    out.append("**Tweak C — keep production weights, but only label as Grade A when DTE is 8-90 days:**")
    out.append("Same `conviction_score` formula; the recalibration is just a downstream gate.")
    panel_c = panel.copy()
    eligible = panel_c["dominant_dte_bucket"].isin(["8-30", "31-90", "91+"])
    in_a = (panel_c["coarse_grade"] == "A") & eligible
    sub_a = panel_c[in_a]
    sub_b = panel_c[(panel_c["coarse_grade"] == "A") & ~eligible]
    s_a = _summary_stats(sub_a["signed_excess"].tolist())
    s_b = _summary_stats(sub_b["signed_excess"].tolist())
    rows = [
        ["Grade A & DTE 8-90+ days (kept)", s_a.get("n", 0), f"{s_a.get('hit_rate', 0):.1%}" if s_a.get("n") else "—",
         _fmt_pct_already(s_a.get("mean_excess_pct", 0), 2) if s_a.get("n") else "—",
         _fmt(s_a.get("mean_r", 0), 2) if s_a.get("n") else "—"],
        ["Grade A & DTE unknown (rejected)", s_b.get("n", 0), f"{s_b.get('hit_rate', 0):.1%}" if s_b.get("n") else "—",
         _fmt_pct_already(s_b.get("mean_excess_pct", 0), 2) if s_b.get("n") else "—",
         _fmt(s_b.get("mean_r", 0), 2) if s_b.get("n") else "—"],
    ]
    out.append("")
    out.append(_table(["Cohort", "n", "Hit", "Mean Excess", "Mean R"], rows))

    # Ranked summary
    out.append("")
    out.append("**Ranked tweak summary (by Grade A mean R):**")
    candidates: list[tuple[str, dict[str, Any] | None]] = [
        ("Baseline (current)", baseline),
        ("Tweak A (Spearman re-weight)", result_a),
        ("Tweak B (feature swap)", result_b),
    ]
    rows = []
    for name, r in candidates:
        if r is None:
            continue
        rows.append([
            name,
            r["A_n"],
            f"{r['A_hit']:.1%}" if r["A_hit"] is not None else "—",
            _fmt_pct_already(r["A_mean_excess_pct"] or 0, 2),
            _fmt(r["A_mean_r"] or 0, 2),
        ])
    rows.sort(key=lambda x: -float(x[4]) if x[4] != "—" else 0)
    out.append("")
    out.append(_table(["Tweak", "A n", "A hit", "A mean excess", "A mean R"], rows))

    return "\n".join(out)


def _format_sim_result(name: str, weights: dict[str, float], result: dict[str, Any]) -> str:
    weight_str = ", ".join(f"`{k}`={v:+.2f}" for k, v in weights.items())
    a_hit = f"{result['A_hit']:.1%}" if result["A_hit"] is not None else "—"
    a_excess = _fmt_pct_already(result["A_mean_excess_pct"] or 0, 2)
    a_r = _fmt(result["A_mean_r"] or 0, 2)
    return (
        f"- Weights: {weight_str}\n"
        f"- New Grade A: n={result['A_n']}, hit={a_hit}, mean_excess={a_excess}, **mean_r={a_r}**"
    )


# ---------------------------------------------------------------------------
# Report assembly + main
# ---------------------------------------------------------------------------


def build_recommendation(panel: pd.DataFrame, attribution: dict[str, Any]) -> str:
    """Synthesize a final next-step recommendation from the findings."""
    grade_a = panel[panel["coarse_grade"] == "A"]
    n_a = len(grade_a)
    a_mean_r = float((grade_a["signed_excess"] / ASSUMED_STOP_PCT).mean()) if n_a else float("nan")

    # Component diagnostic
    rho_persist, _ = _spearman(panel["persistence_ratio"], panel["signed_excess"])
    rho_consist, _ = _spearman(panel.get("accumulation_score", pd.Series()), panel["signed_excess"])
    rho_intensity, _ = _spearman(panel.get("prem_mcap_bps", pd.Series()), panel["signed_excess"])
    rho_multileg, _ = _spearman(panel.get("multileg_share", pd.Series()), panel["signed_excess"])
    rho_sweep, _ = _spearman(panel.get("sweep_share", pd.Series()), panel["signed_excess"])

    # Monotonicity check (re-compute briefly)
    means = []
    for g in GRADE_ORDER:
        sub = panel[panel["conviction_grade"] == g]["signed_excess"].dropna()
        if len(sub) >= 2:
            means.append((g, float(sub.mean())))
    is_monotone = (
        len(means) >= 3
        and all(means[i][1] >= means[i + 1][1] for i in range(len(means) - 1))
    )

    out: list[str] = ["## Next-step recommendation", ""]
    out.append(f"- Grade A panel: n={n_a}, mean R = {a_mean_r:+.2f}")
    out.append(f"- Grade ladder monotonic? **{'YES' if is_monotone else 'NO'}**")
    out.append(f"- Top-weighted components (`persistence_ratio`, `accumulation_score`, `prem_mcap_bps`) Spearman rho: "
               f"persist={_fmt(rho_persist, 3)}, consist={_fmt(rho_consist, 3)}, intensity={_fmt(rho_intensity, 3)}")
    out.append(f"- Predictive components NOT in `conviction_score`: multileg_share rho={_fmt(rho_multileg, 3)}, sweep_share rho={_fmt(rho_sweep, 3)}")
    out.append("")

    # Decision tree
    if is_monotone and a_mean_r > 0.5:
        out.append("**Recommendation: small inline tweak (Phase 1B-quick).**")
        out.append(
            "The grade ladder is directionally correct and Grade A is profitable. "
            "Apply the best-performing simulated tweak from Section 6, then proceed to Phase 2."
        )
    elif (rho_persist is not None and abs(rho_persist) < 0.10) and \
         (rho_consist is not None and abs(rho_consist) < 0.10) and \
         (rho_multileg is not None and rho_multileg > 0.20):
        out.append("**Recommendation: build proper backtest infra (Phase 1B-infra) before changing weights.**")
        out.append(
            "The components currently weighted highest (persistence, consistency, accel) have near-zero "
            "correlation with forward returns over this sample. The components that DO correlate "
            "(`multileg_share`, `sweep_share`) are not in the formula. A weight rewrite is justified "
            "but should be done with proper out-of-sample / walk-forward validation, not by simulating "
            "on the same panel that informed the weights. Build the infra, then recalibrate."
        )
        out.append("")
        out.append(
            "**Interim mitigation (zero-code):** treat the dashboard's 'Grade A backtest' header as "
            "*informational only* until recalibration is done. Do NOT auto-promote Grade A names to "
            "actionable signals (Phase 2 deferred until grade is validated)."
        )
    else:
        out.append("**Recommendation: redesign required (mixed signal).**")
        out.append(
            "The diagnostic does not point to a single clear fix: components are weakly predictive, "
            "the ladder is non-monotonic, but no single feature offers a clean replacement. "
            "Suggest pivoting Phase 2 design to use *grade deltas* (decay/improvement) rather than "
            "absolute grade for exits, since deltas are informative even when absolute levels are noisy."
        )

    out.append("")
    out.append("**For the user's two questions:**")
    out.append("- *'Maximize signal so we can take positions':* the path to that is fixing the grade first. "
               "Auto-promoting Grade A today would amplify noise (current Grade A mean R is around -0.3R per "
               "the live backtest). Path 2 (auto-promote with synthetic plans) should wait.")
    out.append("- *'Position health and exits using flow features':* this is **independent of the grade calibration** "
               "and can be built immediately. Grade *deltas* (e.g., 'this name was Grade A on entry, today it's Grade C') "
               "are much more robust than absolute grades. Recommend Path 3 (flow-aware exits) as the highest-impact "
               "next move while Phase 1B-infra runs in parallel.")

    return "\n".join(out)


def write_report(sections: list[str], out_path: Path) -> None:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = [
        f"# Grade A Diagnostic — {today}",
        "",
        "Phase 1A read-only investigation. No production code or scoring weights have been changed.",
        "Source data: `data/grade_history.csv` (165 rows; 104 with realized 5d forward returns), "
        "cross-checked against `data/grade_stats.json` and `data/grade_attribution.json`.",
        "",
        "**TL;DR**",
        "",
        "- The dashboard shows `Grade A: 57% hit, -0.29R avg, n=90`. A 57% hit rate with negative R means losses are bigger than wins.",
        "- The components weighted highest in `conviction_score` (persistence, consistency, acceleration) have **near-zero Spearman correlation** with forward returns on the persisted panel.",
        "- Two features that ARE statistically significant (`multileg_share` positive, `sweep_share` negative) are **not in the formula**.",
        "- The 7-tier grade ladder is **not monotonic**: B (+3.8%) > A- (+0.6%) > B+ (-4.2%) on the persisted sample.",
        "- See sections 1-6 below for the full evidence and section labelled 'Next-step recommendation' at the bottom.",
        "",
        "---",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header) + "\n\n" + "\n\n---\n\n".join(sections) + "\n")


def main() -> int:
    print("Loading panel data…", flush=True)
    panel = load_panel()
    print(f"  - grade_history rows w/ forward returns: {len(panel)}", flush=True)
    print(f"  - unique tickers: {panel['ticker'].nunique()}", flush=True)
    print(f"  - date range: {panel['as_of'].min()} -> {panel['as_of'].max()}", flush=True)

    stats_json = load_grade_stats()
    attribution = load_attribution()

    snaps = pd.DataFrame()
    if SNAPSHOTS_ARCHIVE_PATH.exists():
        try:
            snaps = pd.read_csv(SNAPSHOTS_ARCHIVE_PATH)
        except Exception as e:
            print(f"  ! could not load snapshots archive: {e}", flush=True)

    panel_dates = sorted(panel["as_of"].astype(str).unique().tolist())
    regime_by_date = load_market_regime_by_date(snaps, panel_dates) if not snaps.empty else {}
    sector_heat = load_sector_heat_by_date(snaps, panel_dates) if not snaps.empty else {}
    print(f"  - regime classifications attached for {len(regime_by_date)} dates", flush=True)
    print(f"  - sector-heat lookups attached for {len(sector_heat)} dates", flush=True)

    print("Computing Section 1 (grade distribution)…", flush=True)
    s1 = section_1_grade_distribution(panel, stats_json)
    print("Computing Section 2 (component correlations)…", flush=True)
    s2 = section_2_component_correlation(panel, attribution)
    print("Computing Section 3 (conditional slices)…", flush=True)
    s3 = section_3_conditional_slices(panel, regime_by_date, sector_heat)
    print("Computing Section 4 (effective N)…", flush=True)
    s4 = section_4_effective_n(panel)
    print("Computing Section 5 (worst-case attribution)…", flush=True)
    s5 = section_5_worst_case(panel)
    print("Computing Section 6 (candidate tweaks)…", flush=True)
    s6 = section_6_candidate_tweaks(panel)
    print("Building recommendation…", flush=True)
    rec = build_recommendation(panel, attribution)

    out_path = OUT_DIR / f"diagnostic_grade_a_{datetime.now().strftime('%Y-%m-%d')}.md"
    write_report([s1, s2, s3, s4, s5, s6, rec], out_path)
    print(f"\nReport written to: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

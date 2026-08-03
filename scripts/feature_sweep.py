"""Unified feature sweep — rank EVERY tracked feature by realized-R edge.

Joins the two panels that carry an outcome
(``grade_history_with_replay.csv`` × ``feature_lab.csv``) on
``(as_of, ticker, direction)`` and evaluates every numeric feature from
every scorer against the bar-by-bar ``replay_realized_r``:

  - the Flow Tracker ``conviction_score`` components (LIVE in the A/B/C grade),
  - the flow-tracker fields that are computed but feed no score,
  - the pipeline ``final_score`` flow input we can observe (``flow_intensity``),
  - the feature-lab free features, the aggressor family, the UW options
    features, and the shadow composites.

Each feature is scored against *two* labels, because they answer different
questions: ``replay_realized_r`` is the outcome after the replay's exit
policy (what the account earns, but confounded with stop/target quality),
while ``replay_forward_return_5d`` is a plain 5-day close-to-close move that
no entry or exit rule touches (does the feature predict price at all). A
feature that scores well on forward return but poorly on realized R points
at the exit policy, not the feature.

For each feature it reports:

  - ``n``            matured rows with both feature and realized_r populated
  - ``spearman``     pooled rank IC vs realized_r (in-sample)
  - ``oos_spearman`` chronological 60/40 walk-forward rank IC (leakage-safe-ish)
  - ``r_spread``     mean realized R of the top tercile minus the bottom
                     tercile (the actual $ edge, in R units)
  - ``p``            two-sided p-value on the pooled rank IC
  - ``n_fwd`` / ``fwd_spearman`` / ``fwd_oos`` — the same rank ICs against the
                     plan-free forward return, over all rows (no maturity
                     filter, so newly-added features become readable sooner)

Sorted by a robustness score = sign-agreeing OOS rank IC, so features that
only look good in-sample sink. Writes
``data/diagnostic_feature_sweep_<YYYY-MM-DD>.md``.

Usage::

    python scripts/feature_sweep.py
    python scripts/feature_sweep.py --min-n 40
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

DATA_DIR = ROOT / "data"
REPLAY_PATH = DATA_DIR / "grade_history_with_replay.csv"
LAB_PATH = DATA_DIR / "feature_lab.csv"
TARGET = "replay_realized_r"
# Plan-free companion label. ``replay_realized_r`` is the outcome *after* the
# replay's stop/target/trail policy, so a feature that predicts price can look
# flat purely because the exit policy ate the move. The raw forward return
# answers the narrower question — does the feature predict where price goes —
# and needs no matured trade, only N more bars.
FWD_TARGET = "replay_forward_return_5d"

# Feature -> (family, in_score?) provenance. "family" groups the output;
# "role" says whether the feature actually feeds a live score today.
#   conv   = Flow Tracker conviction_score (A/B/C) component
#   ft     = flow-tracker field, computed but feeds NO score
#   final  = pipeline final_score flow input (observable subset)
#   lab    = feature-lab free feature (shadow / informational)
#   aggr   = aggressor-signed family (shadow)
#   uw     = Unusual Whales options feature (shadow)
#   comp   = composite score (shadow)
FEATURE_PROVENANCE: dict[str, tuple[str, str]] = {
    # --- Flow Tracker conviction_score components (LIVE A/B/C) ---
    "persistence_ratio": ("conviction_score component", "LIVE"),
    "prem_mcap_bps": ("conviction_score component", "LIVE"),
    "cumulative_premium": ("conviction_score component", "LIVE"),
    "latest_oi_change": ("conviction_score component", "LIVE"),
    # flow_intensity ≈ intensity input AND the top final_score flow weight
    "flow_intensity": ("conviction_score + final_score", "LIVE"),
    # --- Flow-tracker fields computed but NOT in any score ---
    "sweep_share": ("flow-tracker (unused)", "—"),
    "multileg_share": ("flow-tracker (unused)", "—"),
    "accel_ratio_today": ("flow-tracker (unused)", "—"),
    "window_return_pct": ("flow-tracker (unused)", "—"),
    "accumulation_score": ("flow-tracker (unused)", "—"),
    "latest_put_call_ratio": ("flow-tracker (unused)", "—"),
    "latest_iv_rank": ("flow-tracker (unused)", "—"),
    "perc_3_day_total_latest": ("flow-tracker (unused)", "—"),
    "perc_30_day_total_latest": ("flow-tracker (unused)", "—"),
    # --- feature-lab free features (shadow) ---
    "bullish_premium_share": ("feature-lab", "shadow"),
    "unusual_premium_share": ("feature-lab", "shadow"),
    "vrp_proxy": ("feature-lab", "shadow"),
    "far_otm_call_share": ("feature-lab", "shadow"),
    "far_otm_put_share": ("feature-lab", "shadow"),
    "dollar_delta_weighted_flow": ("feature-lab", "shadow"),
    "sector_relative_pct": ("feature-lab", "shadow"),
    "prem_momentum_z3d": ("feature-lab", "shadow"),
    "realized_vol_regime": ("feature-lab", "shadow"),
    # --- aggressor family (shadow) ---
    "aggressor_bull_share": ("aggressor", "shadow"),
    "aggressor_net_prem_bps": ("aggressor", "shadow"),
    "ask_side_ratio": ("aggressor", "shadow"),
    "directional_sweep_share": ("aggressor", "shadow"),
    # --- UW options features (shadow) ---
    "gex_total": ("UW options", "shadow"),
    "vanna_total": ("UW options", "shadow"),
    "charm_total": ("UW options", "shadow"),
    "iv_skew_25d": ("UW options", "shadow"),
    "atm_iv_30d": ("UW options", "shadow"),
    "atm_iv_60d": ("UW options", "shadow"),
    "atm_iv_90d": ("UW options", "shadow"),
    "term_slope_30_90": ("UW options", "shadow"),
    "expiry_concentration_top1": ("UW options", "shadow"),
    "max_pain_dist_pct": ("UW options", "shadow"),
    "dealer_net_delta_at_spot": ("UW options", "shadow"),
    "dealer_net_gamma_at_spot": ("UW options", "shadow"),
    # --- price / volume / volatility technicals (Tier 1, shadow) ---
    "ret_5d": ("price/vol (T1)", "shadow"),
    "ret_21d": ("price/vol (T1)", "shadow"),
    "ret_63d": ("price/vol (T1)", "shadow"),
    "ret_126d": ("price/vol (T1)", "shadow"),
    "dist_52w_high": ("price/vol (T1)", "shadow"),
    "px_vs_sma50": ("price/vol (T1)", "shadow"),
    "px_vs_sma200": ("price/vol (T1)", "shadow"),
    "rsi_14": ("price/vol (T1)", "shadow"),
    "bollinger_z": ("price/vol (T1)", "shadow"),
    "rel_volume": ("price/vol (T1)", "shadow"),
    "atr_pct": ("price/vol (T1)", "shadow"),
    "gap_pct": ("price/vol (T1)", "shadow"),
    # --- cross-sectional / cross-asset residual signals (Tier 2, shadow) ---
    "beta_63d": ("cross-sectional (T2)", "shadow"),
    "resid_mom_21d": ("cross-sectional (T2)", "shadow"),
    "rel_strength_spy_63d": ("cross-sectional (T2)", "shadow"),
    "rel_strength_sector_63d": ("cross-sectional (T2)", "shadow"),
    # --- composites (shadow) ---
    "conviction_score": ("composite", "LIVE grade"),
    "conviction_stack": ("composite", "LIVE"),
    "momentum_score": ("composite", "shadow"),
    "momentum_composite": ("composite", "shadow"),
}


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    df = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                       "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(df) < 5 or df["a"].nunique() < 2 or df["b"].nunique() < 2:
        return float("nan"), len(df)
    return float(df["a"].rank().corr(df["b"].rank())), len(df)


def _spearman_p(rho: float, n: int) -> float:
    if math.isnan(rho) or n < 4:
        return float("nan")
    z = abs(rho) * math.sqrt(max(n - 1, 1))
    return math.erfc(z / math.sqrt(2.0))


def _oos_spearman(df: pd.DataFrame, feat: str, train_frac: float = 0.6,
                  min_train: int = 12, target: str = TARGET) -> tuple[float, int]:
    sub = df[["__dt", feat, target]].dropna().sort_values("__dt")
    if len(sub) < min_train + 5:
        return float("nan"), 0
    n_train = int(len(sub) * train_frac)
    val = sub.iloc[n_train:]
    if len(val) < 5:
        return float("nan"), 0
    sp, _ = _spearman(val[feat], val[target])
    return sp, len(val)


def _tercile_spread(df: pd.DataFrame, feat: str) -> tuple[float, int]:
    sub = pd.DataFrame({
        "x": pd.to_numeric(df[feat], errors="coerce"),
        "r": pd.to_numeric(df[TARGET], errors="coerce"),
    }).dropna()
    if len(sub) < 15 or sub["x"].nunique() < 3:
        return float("nan"), len(sub)
    q = sub["x"].quantile([1 / 3, 2 / 3]).tolist()
    top = sub[sub["x"] >= q[1]]["r"]
    bot = sub[sub["x"] <= q[0]]["r"]
    if len(top) < 3 or len(bot) < 3:
        return float("nan"), len(sub)
    return float(top.mean() - bot.mean()), len(sub)


_REAL_EXITS = ("T2", "T1_then_stop", "stop", "time_stop")


def _matured_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of matured (completed / held-to-horizon) replay rows.

    Prefers the explicit ``replay_is_matured`` flag stamped by the replay
    engine. Falls back (for CSVs written before that flag existed) to
    deriving maturity from ``replay_exit_reason`` — a real exit fired, or a
    ``no_exit_yet`` row that nonetheless held the full ``max_hold`` horizon —
    while dropping ``days_held == 0`` degenerate marks.
    """
    if "replay_is_matured" in df.columns:
        raw = df["replay_is_matured"].astype(str).str.strip().str.lower()
        m = raw.map({"true": True, "false": False, "1": True, "0": False})
        if m.notna().any():
            return m.fillna(False).astype(bool)

    exit_reason = df.get("replay_exit_reason")
    if exit_reason is None:
        return pd.Series(True, index=df.index)  # nothing to filter on
    er = exit_reason.astype(str).str.strip()
    real_exit = er.isin(_REAL_EXITS)
    days_held = pd.to_numeric(df.get("replay_days_held"), errors="coerce").fillna(0)
    if "replay_max_hold_days_used" in df.columns:
        max_hold = pd.to_numeric(df["replay_max_hold_days_used"], errors="coerce").fillna(1e9)
    else:
        # Column absent (pre-flag CSV): can't confirm held-to-horizon, so no
        # no_exit_yet row qualifies — only real exits count as matured.
        max_hold = pd.Series(1e9, index=df.index)
    held_to_horizon = (er == "no_exit_yet") & (days_held >= max_hold)
    return (real_exit | held_to_horizon) & (days_held > 0)


def build_panel() -> pd.DataFrame:
    rep = pd.read_csv(REPLAY_PATH)
    lab = pd.read_csv(LAB_PATH)
    for d in (rep, lab):
        d["__t"] = d["ticker"].astype(str).str.upper().str.strip()
        d["__d"] = d["direction"].astype(str).str.upper().str.strip()
        d["__a"] = d["as_of"].astype(str).str.strip()
    # Keep replay outcome + flow-tracker fields; bring lab features in.
    lab_only = [c for c in lab.columns
                if c not in rep.columns and c not in ("__t", "__d", "__a")]
    merged = rep.merge(
        lab[["__t", "__d", "__a"] + lab_only],
        on=["__t", "__d", "__a"], how="left",
    )
    if FWD_TARGET not in merged.columns:
        merged[FWD_TARGET] = float("nan")
    has_r = pd.to_numeric(merged[TARGET], errors="coerce").notna()
    has_fwd = pd.to_numeric(merged[FWD_TARGET], errors="coerce").notna()
    merged = merged[has_r | has_fwd].copy()

    # Matured-only applies to the realized-R label only. Trades whose OHLCV ran
    # out before an exit fired carry a truncated mark-to-market R (typically
    # 0-3 days held) and, being the most recent signals, would otherwise
    # dominate the time-based OOS window and bias every feature verdict. The
    # forward-return label has no such failure mode — it just needs N more bars
    # — so immature rows stay in the panel and are masked per-label instead.
    merged["__matured"] = _matured_mask(merged)
    merged["__dt"] = pd.to_datetime(merged["as_of"], errors="coerce")
    n_r = int((merged["__matured"] & pd.to_numeric(
        merged[TARGET], errors="coerce").notna()).sum())
    n_fwd = int(pd.to_numeric(merged[FWD_TARGET], errors="coerce").notna().sum())
    print(f"  [feature_sweep] panel {len(merged)} rows: "
          f"{n_r} matured realized_r, {n_fwd} with {FWD_TARGET}")
    return merged


def sweep(panel: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    matured = panel[panel["__matured"]]
    rows = []
    for feat, (family, role) in FEATURE_PROVENANCE.items():
        if feat not in panel.columns:
            continue
        sp, n = _spearman(matured[feat], matured[TARGET])
        fwd_sp, n_fwd = _spearman(panel[feat], panel[FWD_TARGET])
        if n < min_n and n_fwd < min_n:
            continue
        oos, n_val = _oos_spearman(matured, feat)
        fwd_oos, _ = _oos_spearman(panel, feat, target=FWD_TARGET)
        spread, _ = _tercile_spread(matured, feat)
        p = _spearman_p(sp, n)
        # Robustness: OOS rank IC, but only credited when it agrees in sign
        # with the pooled IC (kills in-sample-only flukes).
        agree = (not math.isnan(oos) and not math.isnan(sp)
                 and np.sign(oos) == np.sign(sp))
        robust = oos if agree else min(oos, 0.0) if not math.isnan(oos) else float("nan")
        rows.append({
            "feature": feat, "family": family, "role": role,
            "n": n, "spearman": sp, "p": p,
            "oos_spearman": oos, "n_val": n_val,
            "r_spread": spread, "robust": robust,
            "n_fwd": n_fwd, "fwd_spearman": fwd_sp, "fwd_oos": fwd_oos,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["__sortkey"] = df["robust"].fillna(-9).where(
        df["robust"].notna(), df["spearman"].abs().mul(0.1) - 9)
    return df.sort_values("__sortkey", ascending=False).drop(columns="__sortkey")


def _f(v, nd=3, signed=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return (f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}")


def render(panel: pd.DataFrame, ranked: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_rows = len(panel)
    n_mat = int(panel["__matured"].sum())
    n_fwd = int(pd.to_numeric(panel[FWD_TARGET], errors="coerce").notna().sum())
    lines = [
        f"# Unified feature sweep — {today}",
        "",
        f"Panel: **{n_rows} rows** joined `grade_history_with_replay.csv` × "
        "`feature_lab.csv` on (as_of, ticker, direction) — "
        f"**{n_mat}** matured rows carry `replay_realized_r`, "
        f"**{n_fwd}** carry `{FWD_TARGET}`.",
        "",
        "Every feature is scored against **two labels**:",
        "",
        f"- **realized R** (`{TARGET}`, matured rows only) — the outcome after "
        "the replay's stop/target/trail policy. This is what the account "
        "actually earns, but it confounds feature quality with exit-policy "
        "quality: a feature can predict the move correctly and still score "
        "flat because the stop took the trade out first.",
        f"- **forward return** (`{FWD_TARGET}`, all rows) — plain 5-day "
        "close-to-close, independent of any entry or exit rule. This isolates "
        "*does the feature predict price*, and needs no matured trade, so it "
        "reaches a usable sample far sooner on newly-added features.",
        "",
        "Read them together. Agreement in sign is the strong signal. A feature "
        "with a good forward IC but a poor realized-R IC is a hint that the "
        "**exit policy**, not the feature, is the thing to fix.",
        "",
        "`spearman` = pooled rank IC (in-sample). "
        "`oos` = chronological 60/40 walk-forward rank IC. "
        "`r_spread` = mean realized R of top tercile − bottom tercile "
        "(the $ edge, in R). Sorted by sign-agreeing OOS IC on realized R so "
        "in-sample-only flukes sink.",
        "",
        "**Caveat:** one bull-market regime, small OOS slices. Treat this as a "
        "hypothesis watchlist, not a hit list. A feature needs |IC| that holds "
        "OOS across fresh weeks before it earns a place in a live score.",
        "",
        "---",
        "",
        "## Full ranking (all scorers, both targets)",
        "",
    ]
    headers = ["Feature", "Family", "Live?", "n", "Spearman", "p",
               "OOS", "n_val", "R-spread", "n (fwd)", "Fwd IC", "Fwd OOS"]
    body = ["| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, r in ranked.iterrows():
        body.append("| " + " | ".join(str(c) for c in [
            f"`{r['feature']}`", r["family"], r["role"], int(r["n"]),
            _f(r["spearman"]), _f(r["p"], nd=3, signed=False),
            _f(r["oos_spearman"]), int(r["n_val"]),
            _f(r["r_spread"], nd=2),
            int(r["n_fwd"]), _f(r["fwd_spearman"]), _f(r["fwd_oos"]),
        ]) + " |")
    lines.append("\n".join(body))
    lines.append("")

    # Highlight: what clears a minimal bar.
    lines.append("## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)")
    lines.append("")
    good = ranked[
        (ranked["n"] >= 40)
        & (ranked["spearman"].abs() >= 0.10)
        & (ranked["oos_spearman"].notna())
        & (np.sign(ranked["oos_spearman"]) == np.sign(ranked["spearman"]))
    ]
    if good.empty:
        lines.append("_None clear it yet on current data._")
    else:
        for _, r in good.iterrows():
            lines.append(
                f"- `{r['feature']}` ({r['family']}, {r['role']}): "
                f"Spearman {_f(r['spearman'])}, OOS {_f(r['oos_spearman'])}, "
                f"R-spread {_f(r['r_spread'], nd=2)}, n={int(r['n'])}"
            )
    lines.append("")

    # Where the two labels disagree, the exit policy is the suspect.
    lines.append("## Predicts price but not realized R (exit-policy suspects)")
    lines.append("")
    lines.append(
        "Features whose forward-return IC is meaningfully positive while their "
        "realized-R IC is flat or negative. The feature is calling the move; "
        "the stop/target/trail is giving it back."
    )
    lines.append("")
    gap = ranked[
        (ranked["n_fwd"] >= 40)
        & (ranked["fwd_spearman"] >= 0.10)
        & (ranked["spearman"].fillna(0) < 0.05)
    ]
    if gap.empty:
        lines.append("_No feature shows that split on current data._")
    else:
        for _, r in gap.iterrows():
            lines.append(
                f"- `{r['feature']}` ({r['family']}, {r['role']}): "
                f"forward IC {_f(r['fwd_spearman'])} (n={int(r['n_fwd'])}) vs "
                f"realized-R IC {_f(r['spearman'])} (n={int(r['n'])})"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not REPLAY_PATH.exists() or not LAB_PATH.exists():
        print("missing inputs (need replay panel + feature_lab).")
        return 0
    panel = build_panel()
    if panel.empty:
        print("no rows with realized_r yet.")
        return 0
    ranked = sweep(panel, min_n=args.min_n)
    md = render(panel, ranked)
    out = Path(args.out) if args.out else (
        DATA_DIR / f"diagnostic_feature_sweep_{datetime.now().strftime('%Y-%m-%d')}.md")
    out.write_text(md)
    print(f"Wrote: {out}  ({len(ranked)} features over {len(panel)} rows)")
    for _, r in ranked.head(15).iterrows():
        print(f"  {r['feature']:28s} {r['family']:28s} "
              f"n={int(r['n']):4d} sp={_f(r['spearman'])} "
              f"oos={_f(r['oos_spearman'])} Rspread={_f(r['r_spread'],nd=2)} "
              f"| fwd n={int(r['n_fwd']):4d} ic={_f(r['fwd_spearman'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

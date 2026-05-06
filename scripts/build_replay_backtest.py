"""Run the faithful equity-trade replay backtest over grade_history.csv.

Reads ``data/grade_history.csv``, calls
``app.analytics.trade_replay.replay_trade_plan`` for every row, joins the
realized exit metrics back to the original schema, and writes:

    - data/grade_history_with_replay.csv  (augmented panel)
    - data/diagnostic_replay_<date>.md    (7-section analysis report)

The report intentionally mirrors the structure of ``data/diagnostic_grade_a_*.md``
but uses the new ``realized_r`` (from bar-by-bar exit replay) as the
target metric instead of the old ``forward_excess_return / 0.02``.

OHLCV is fetched on-demand via ``app.features.price_features.fetch_ohlcv``
(disk-cached at 24h TTL inside the same module). SPY is fetched once
per run.

Usage::

    python scripts/build_replay_backtest.py
    python scripts/build_replay_backtest.py --tickers AAPL,NVDA   # filter
    python scripts/build_replay_backtest.py --max-rows 50         # debug
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analytics.trade_replay import replay_trade_plan  # noqa: E402

DATA_DIR = ROOT / "data"
GRADE_HISTORY = DATA_DIR / "grade_history.csv"
OUT_PANEL = DATA_DIR / "grade_history_with_replay.csv"
OHLCV_CACHE_DIR = DATA_DIR / "_ohlcv_cache"
OHLCV_CACHE_TTL_S = 24 * 3600

ASSUMED_STOP_PCT = 0.02  # legacy metric we cross-reference

# ---- helpers ----------------------------------------------------------


def _ohlcv_cache_path(ticker: str) -> Path:
    safe = "".join(c for c in str(ticker or "").upper() if c.isalnum() or c in "._-")
    return OHLCV_CACHE_DIR / f"{safe}.csv"


def _load_ohlcv(ticker: str, lookback_days: int = 120) -> pd.DataFrame | None:
    """Disk-cached OHLCV fetch via yfinance (fail soft)."""
    OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _ohlcv_cache_path(ticker)
    if p.exists():
        age = time.time() - p.stat().st_mtime
        if age < OHLCV_CACHE_TTL_S:
            try:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
            except Exception:
                pass
    try:
        from app.features.price_features import fetch_ohlcv
        df = fetch_ohlcv(ticker, lookback_days=lookback_days, include_partial=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    try:
        df.to_csv(p)
    except Exception:
        pass
    return df


def _coarse(grade: str | None) -> str:
    if not grade:
        return "C"
    g = str(grade).strip()
    if not g:
        return "C"
    return g[0].upper() if g[0].upper() in ("A", "B", "C") else "C"


def _normalize_dte_bucket(b: Any) -> str:
    s = str(b or "").strip().lower()
    if not s or s == "nan":
        return "unknown"
    if s in ("0-7",):
        return "lottery"
    if s in ("8-30",):
        return "swing"
    if s in ("31-90",):
        return "position"
    if s in ("91+", "leap", "leaps"):
        return "leap"
    return s


def _summary_signed_r(rs: list[float]) -> dict[str, float]:
    arr = np.array([r for r in rs if not (isinstance(r, float) and math.isnan(r))], dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "hit_rate": float((arr > 0).mean()),
        "mean_r": float(arr.mean()),
        "median_r": float(np.median(arr)),
        "std_r": float(arr.std(ddof=0)),
        "best_r": float(arr.max()),
        "worst_r": float(arr.min()),
    }


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_"
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt_r(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.2f}"


def _fmt_pct_already(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:+.2f}%"


# ---- replay over grade_history ----------------------------------------


def replay_panel(panel: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """Run replay_trade_plan for each row, merge results back to the panel.

    Skips rows where OHLCV fetch fails. Returns a new DataFrame.
    """
    spy = _load_ohlcv("SPY", lookback_days=120)

    rows = panel.to_dict("records")
    if max_rows is not None:
        rows = rows[:max_rows]

    out_rows: list[dict[str, Any]] = []
    skipped = 0
    fetch_fail: Counter = Counter()
    for i, r in enumerate(rows, start=1):
        ticker = str(r.get("ticker") or "").upper().strip()
        as_of = str(r.get("as_of") or "")
        direction = str(r.get("direction") or "BULLISH").upper()
        if not ticker or not as_of:
            skipped += 1
            fetch_fail["missing_id"] += 1
            continue
        df = _load_ohlcv(ticker, lookback_days=120)
        if df is None or df.empty:
            skipped += 1
            fetch_fail[ticker] += 1
            continue
        try:
            res = replay_trade_plan(
                ticker, as_of, direction, df, spy,
                dominant_dte_bucket=r.get("dominant_dte_bucket"),
            )
        except Exception as e:
            print(f"  [{i}/{len(rows)}] {ticker} {as_of} replay failed: {e}", flush=True)
            skipped += 1
            continue
        merged = {**r, **{f"replay_{k}": v for k, v in res.items()}}
        out_rows.append(merged)
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] processed (skipped so far: {skipped})", flush=True)

    print(f"\nReplay complete: {len(out_rows)} replayed, {skipped} skipped.", flush=True)
    if fetch_fail:
        offenders = sorted(
            ((t, c) for t, c in fetch_fail.items() if t != "missing_id"),
            key=lambda kv: -kv[1],
        )
        if offenders:
            preview = ", ".join(f"{t}({c})" for t, c in offenders[:15])
            more = "" if len(offenders) <= 15 else f" … +{len(offenders) - 15} more"
            print(
                f"OHLCV fetch failures: {len(offenders)} tickers — {preview}{more}",
                flush=True,
            )
            print(
                "  hint: cache lives in data/_ohlcv_cache/. If this list is large, "
                "yfinance is likely rate-limited on this runner — pre-warm the cache "
                "locally and commit, or let hourly_scan keep it fresh.",
                flush=True,
            )
        if fetch_fail.get("missing_id"):
            print(
                f"  ({fetch_fail['missing_id']} rows skipped due to missing ticker/as_of)",
                flush=True,
            )
    return pd.DataFrame(out_rows)


# ---- report sections ---------------------------------------------------


def section_1_replay_summary(panel: pd.DataFrame) -> str:
    out = ["## 1. Replay summary by exit_reason", ""]
    counts = Counter(panel["replay_exit_reason"].fillna("error").tolist())
    rows = []
    n_total = sum(counts.values())
    for k in (
        "T2", "T1_then_stop", "stop", "ema20_trail", "time_stop", "no_exit_yet"
    ):
        c = counts.get(k, 0)
        pct = (c / n_total * 100) if n_total else 0
        rows.append([k, c, f"{pct:.1f}%"])
    # Catch any unexpected reasons
    for k, c in counts.items():
        if k in ("T2", "T1_then_stop", "stop", "ema20_trail", "time_stop", "no_exit_yet"):
            continue
        rows.append([k, c, f"{c / n_total * 100:.1f}%"])
    out.append(_table(["Exit reason", "n", "% of replayed"], rows))
    out.append("")

    # Realized R distribution
    rs = panel["replay_realized_r"].dropna().tolist()
    s = _summary_signed_r(rs)
    if s.get("n"):
        out.append("**Aggregate realized-R (all rows):**")
        out.append("")
        out.append(_table(
            ["n", "Hit", "Mean R", "Median R", "Std", "Best", "Worst"],
            [[s["n"], f"{s['hit_rate']:.1%}", _fmt_r(s["mean_r"]), _fmt_r(s["median_r"]),
              _fmt_r(s["std_r"]), _fmt_r(s["best_r"]), _fmt_r(s["worst_r"])]],
        ))
    return "\n".join(out)


def section_2_per_grade(panel: pd.DataFrame) -> str:
    out = ["## 2. Per-grade tier with realized R (vs old 5d close-to-close)", ""]
    out.append(
        "Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) "
        "vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade "
        "plan would have exited intraday before the 5d close was reached."
    )
    out.append("")
    panel = panel.copy()
    panel["coarse"] = panel["conviction_grade"].apply(_coarse)
    panel["legacy_r"] = (
        panel["forward_excess_return"].astype(float).abs().fillna(np.nan)
    )
    # Compute signed legacy R the same way grade_backtest.py does
    sign = panel["direction"].map(lambda d: 1.0 if str(d).upper() == "BULLISH" else -1.0)
    panel["legacy_signed_r"] = (
        panel["forward_excess_return"].astype(float) * sign / ASSUMED_STOP_PCT
    )

    rows = []
    for g in ("A+", "A", "A-", "B+", "B", "B-", "C"):
        sub = panel[panel["conviction_grade"] == g]
        if sub.empty:
            continue
        legacy = _summary_signed_r(sub["legacy_signed_r"].dropna().tolist())
        new = _summary_signed_r(sub["replay_realized_r"].dropna().tolist())
        if not legacy.get("n") and not new.get("n"):
            continue
        rows.append([
            g,
            new.get("n", 0),
            f"{new.get('hit_rate', 0):.1%}" if new.get("n") else "—",
            _fmt_r(new.get("mean_r")) if new.get("n") else "—",
            _fmt_r(legacy.get("mean_r")) if legacy.get("n") else "—",
            _fmt_r(new.get("mean_r", 0) - legacy.get("mean_r", 0)) if new.get("n") and legacy.get("n") else "—",
        ])
    out.append("")
    out.append(_table(
        ["Grade", "n", "Hit (replay)", "Mean R (replay)", "Mean R (legacy 5d)", "Δ (new - legacy)"],
        rows,
    ))

    # Coarse tier
    rows = []
    for g in ("A", "B", "C"):
        sub = panel[panel["coarse"] == g]
        if sub.empty:
            continue
        legacy = _summary_signed_r(sub["legacy_signed_r"].dropna().tolist())
        new = _summary_signed_r(sub["replay_realized_r"].dropna().tolist())
        rows.append([
            g,
            new.get("n", 0),
            f"{new.get('hit_rate', 0):.1%}" if new.get("n") else "—",
            _fmt_r(new.get("mean_r")) if new.get("n") else "—",
            _fmt_r(legacy.get("mean_r")) if legacy.get("n") else "—",
        ])
    out.append("")
    out.append("**Coarse-grade view (matches dashboard headline):**")
    out.append("")
    out.append(_table(["Coarse", "n", "Hit (replay)", "Mean R (replay)", "Mean R (legacy)"], rows))
    return "\n".join(out)


def section_3_per_dte(panel: pd.DataFrame) -> str:
    out = ["## 3. Per-DTE-bucket performance", ""]
    panel = panel.copy()
    panel["bucket"] = panel["dominant_dte_bucket"].apply(_normalize_dte_bucket)

    rows = []
    for b in ("lottery", "swing", "position", "leap", "unknown"):
        sub = panel[panel["bucket"] == b]
        if sub.empty:
            continue
        s = _summary_signed_r(sub["replay_realized_r"].dropna().tolist())
        mfe = sub["replay_mfe_r"].dropna()
        days = sub["replay_days_held"].dropna()
        hit_t1 = sub["replay_partial_filled"].fillna(False).astype(bool).sum() / max(len(sub), 1)
        hit_t2 = (sub["replay_exit_reason"] == "T2").sum() / max(len(sub), 1)
        stopped = sub["replay_exit_reason"].isin(["stop", "T1_then_stop"]).sum() / max(len(sub), 1)
        rows.append([
            b,
            s.get("n", 0),
            f"{s.get('hit_rate', 0):.1%}",
            _fmt_r(s.get("mean_r")),
            _fmt_r(float(mfe.mean()) if len(mfe) else None),
            f"{float(days.mean()):.1f}" if len(days) else "—",
            f"{hit_t1:.1%}",
            f"{hit_t2:.1%}",
            f"{stopped:.1%}",
        ])
    out.append(_table(
        ["Bucket", "n", "Hit", "Mean R", "Mean MFE", "Avg days", "% T1 hit", "% T2 hit", "% stopped"],
        rows,
    ))
    return "\n".join(out)


def section_4_grade_x_bucket(panel: pd.DataFrame) -> str:
    out = ["## 4. DTE-bucket × grade interaction", ""]
    panel = panel.copy()
    panel["bucket"] = panel["dominant_dte_bucket"].apply(_normalize_dte_bucket)
    panel["coarse"] = panel["conviction_grade"].apply(_coarse)
    rows = []
    for g in ("A", "B"):
        for b in ("lottery", "swing", "position", "leap", "unknown"):
            sub = panel[(panel["coarse"] == g) & (panel["bucket"] == b)]
            if sub.empty:
                continue
            s = _summary_signed_r(sub["replay_realized_r"].dropna().tolist())
            rows.append([
                g, b, s.get("n", 0),
                f"{s.get('hit_rate', 0):.1%}" if s.get("n") else "—",
                _fmt_r(s.get("mean_r")) if s.get("n") else "—",
            ])
    out.append(_table(["Grade", "Bucket", "n", "Hit", "Mean R"], rows))
    out.append("")
    out.append(
        "**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. "
        "Sparse rows (low n) are inconclusive — *do not* read trends from them."
    )
    return "\n".join(out)


def section_5_time_to_mfe(panel: pd.DataFrame) -> str:
    out = ["## 5. Time-to-MFE distribution per bucket", ""]
    panel = panel.copy()
    panel["bucket"] = panel["dominant_dte_bucket"].apply(_normalize_dte_bucket)
    rows = []
    for b in ("lottery", "swing", "position", "leap", "unknown"):
        sub = panel[panel["bucket"] == b]
        days = sub["replay_mfe_day"].dropna()
        if days.empty:
            continue
        rows.append([
            b, len(sub),
            f"{float(days.mean()):.1f}",
            f"{float(days.median()):.1f}",
            f"{float(days.quantile(0.75)):.1f}",
            f"{float(days.max()):.0f}",
        ])
    out.append(_table(["Bucket", "n", "Mean d-to-MFE", "Median", "p75", "Max"], rows))
    out.append("")
    out.append(
        "**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket "
        "`MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` "
        "is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out."
    )
    return "\n".join(out)


def section_6_path_metrics(panel: pd.DataFrame) -> str:
    out = ["## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)", ""]
    panel = panel.copy()
    panel["bucket"] = panel["dominant_dte_bucket"].apply(_normalize_dte_bucket)
    rows = []
    for b in ("lottery", "swing", "position", "leap", "unknown"):
        sub = panel[panel["bucket"] == b]
        if sub.empty:
            continue
        n = len(sub)
        p_05 = sub["replay_hit_0_5r_within_3d"].fillna(False).sum() / n
        p_1 = sub["replay_hit_1r_within_5d"].fillna(False).sum() / n
        p_2 = sub["replay_hit_2r_within_5d"].fillna(False).sum() / n
        p_3 = sub["replay_hit_3r_within_10d"].fillna(False).sum() / n
        rows.append([b, n, f"{p_05:.1%}", f"{p_1:.1%}", f"{p_2:.1%}", f"{p_3:.1%}"])
    out.append(_table(
        ["Bucket", "n", "+0.5R/3d", "+1R/5d", "+2R/5d", "+3R/10d"],
        rows,
    ))
    out.append("")
    out.append(
        "Conditional probability: of trades that hit +1R, what fraction then go on to +2R? "
        "This separates 'small wins' from 'runners.'"
    )
    rows = []
    for b in ("lottery", "swing", "position", "leap", "unknown"):
        sub = panel[panel["bucket"] == b]
        n_1r = sub["replay_hit_1r_within_5d"].fillna(False).sum()
        if n_1r == 0:
            continue
        n_2r = (sub["replay_hit_1r_within_5d"].fillna(False) & sub["replay_hit_2r_within_5d"].fillna(False)).sum()
        rows.append([b, int(n_1r), int(n_2r), f"{n_2r / n_1r:.1%}" if n_1r else "—"])
    out.append("")
    out.append(_table(["Bucket", "Hit +1R", "Hit +2R", "P(+2R | +1R)"], rows))
    return "\n".join(out)


def section_7_recommendations(panel: pd.DataFrame) -> str:
    out = ["## 7. Concrete per-bucket config recommendations", ""]
    out.append(
        "Recommended values are derived from observed time-to-MFE distributions and "
        "exit-reason mix. **Where sample size is small (n < 15), the recommendation is "
        "marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived "
        "after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**"
    )
    out.append("")
    panel = panel.copy()
    panel["bucket"] = panel["dominant_dte_bucket"].apply(_normalize_dte_bucket)

    recs: dict[str, dict[str, Any]] = {}
    for b in ("lottery", "swing", "position", "leap", "unknown"):
        sub = panel[panel["bucket"] == b]
        if sub.empty:
            continue
        days = sub["replay_mfe_day"].dropna()
        n = len(sub)
        # MAX_HOLD recommendation: 75th percentile of time-to-MFE rounded up,
        # capped to a sane range per bucket.
        if len(days) >= 3:
            p75 = float(days.quantile(0.75))
            max_hold_rec = int(np.ceil(p75 * 1.2))  # 20% slack
        else:
            # Defaults by bucket if no data
            max_hold_rec = {"lottery": 3, "swing": 7, "position": 15, "leap": 25, "unknown": 7}.get(b, 7)

        # Cap recommendations into expected ranges to prevent thin samples
        # from producing wild values
        caps = {
            "lottery": (2, 5),
            "swing": (5, 12),
            "position": (10, 25),
            "leap": (15, 40),
            "unknown": (5, 15),
        }
        lo, hi = caps.get(b, (5, 15))
        max_hold_rec = max(lo, min(hi, max_hold_rec))

        # TIME_STOP_MIN_R: if hit-rate at +0.5R/3d is high, demand more (1R);
        # if low, demand less (0.5R) so we don't kill borderline runners.
        p_05 = sub["replay_hit_0_5r_within_3d"].fillna(False).sum() / n
        time_stop_min_r = 1.0 if p_05 >= 0.4 else 0.5

        # ATR_TRAIL_MULT: from MAE distribution. If 75th-percentile MAE is
        # tighter (smaller magnitude), we can use a tighter trail.
        mae = sub["replay_mae_r"].dropna().abs()
        if len(mae) >= 3:
            mae_p75 = float(mae.quantile(0.75))
            atr_trail_rec = round(max(1.5, min(4.0, mae_p75 + 1.0)), 1)
        else:
            atr_trail_rec = {"lottery": 1.5, "swing": 2.5, "position": 3.5, "leap": 4.0, "unknown": 2.5}.get(b, 2.5)

        confidence = "high" if n >= 30 else "medium" if n >= 15 else "low"

        recs[b] = {
            "n": n,
            "MAX_HOLD_DAYS": max_hold_rec,
            "TIME_STOP_MIN_R": time_stop_min_r,
            "ATR_TRAIL_MULT": atr_trail_rec,
            "confidence": confidence,
            "observed_median_d_to_mfe": float(days.median()) if len(days) else None,
            "observed_mean_r": float(sub["replay_realized_r"].dropna().mean()) if not sub.empty else None,
        }

    rows = []
    for b, v in recs.items():
        rows.append([
            b, v["n"], v["confidence"].upper(),
            v["MAX_HOLD_DAYS"], v["TIME_STOP_MIN_R"], v["ATR_TRAIL_MULT"],
            f"{v['observed_median_d_to_mfe']:.1f}" if v["observed_median_d_to_mfe"] is not None else "—",
            _fmt_r(v["observed_mean_r"]),
        ])
    out.append(_table(
        ["Bucket", "n", "Confidence", "MAX_HOLD_DAYS", "TIME_STOP_MIN_R", "ATR_TRAIL_MULT",
         "Median d-to-MFE", "Observed Mean R"],
        rows,
    ))

    # Save recs as machine-readable JSON for Stage C to consume.
    recs_path = DATA_DIR / "replay_recommended_config.json"
    try:
        with open(recs_path, "w") as f:
            json.dump(recs, f, indent=2)
        out.append("")
        out.append(f"Machine-readable config written to: `{recs_path.relative_to(ROOT)}` "
                   f"(consumed by Stage C config refactor).")
    except Exception:
        pass

    out.append("")
    out.append("**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; "
               "~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations "
               "are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; "
               "`lottery`/`leap` should be re-derived after the sequencing fix produces "
               "4-6 weeks of clean data.")
    return "\n".join(out)


def write_report(panel: pd.DataFrame, out_path: Path) -> None:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_replayed = len(panel[panel["replay_realized_r"].notna()])
    sections = [
        section_1_replay_summary(panel),
        section_2_per_grade(panel),
        section_3_per_dte(panel),
        section_4_grade_x_bucket(panel),
        section_5_time_to_mfe(panel),
        section_6_path_metrics(panel),
        section_7_recommendations(panel),
    ]
    header = [
        f"# Faithful Replay Backtest — {today}",
        "",
        "Source: `data/grade_history.csv` replayed bar-by-bar via "
        "`app/analytics/trade_replay.py`. Production exit logic (T2 hit, "
        "ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, "
        "time stop) is faithfully reproduced; health-based / gamma / wall "
        "exits are skipped (no historical data).",
        "",
        f"**Rows replayed: {n_replayed} / {len(panel)}**.",
        "",
        "---",
        "",
    ]
    out_path.write_text("\n".join(header) + "\n\n" + "\n\n---\n\n".join(sections) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rows", type=int, default=None, help="cap rows for debugging")
    p.add_argument("--tickers", type=str, default=None, help="comma-separated filter")
    args = p.parse_args(argv)

    if not GRADE_HISTORY.exists():
        print(f"missing: {GRADE_HISTORY}", flush=True)
        return 1

    panel = pd.read_csv(GRADE_HISTORY)
    if args.tickers:
        wants = {t.strip().upper() for t in args.tickers.split(",")}
        panel = panel[panel["ticker"].str.upper().isin(wants)]

    print(f"Replaying {len(panel)} rows…", flush=True)
    augmented = replay_panel(panel, max_rows=args.max_rows)

    if augmented.empty:
        print("Nothing replayed (likely OHLCV fetch failures). Aborting.", flush=True)
        return 2

    augmented.to_csv(OUT_PANEL, index=False)
    print(f"Wrote: {OUT_PANEL}", flush=True)

    out_path = DATA_DIR / f"diagnostic_replay_{datetime.now().strftime('%Y-%m-%d')}.md"
    write_report(augmented, out_path)
    print(f"Wrote: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

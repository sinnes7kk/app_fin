"""Roll the saw-it-couldn't-trade history into a per-reason audit report.

Reads ``data/saw_couldnt_trade_history.csv`` and emits a markdown
report under ``data/audit_reports/rejections_<period>.md`` summarising:

* Counts per ``reject_reason_norm`` over the period.
* Top tickers per reason (by ``flow_score_raw``).
* Daily mix-over-time so trends in which gates fire most are visible.

Forward returns are intentionally NOT computed in v1: there is no
single price-history source we can trust per (ticker, scan_date)
across the full set of rejected tickers (many never get final-signal
treatment, so they don't enter the price-features cache). The script
flags the gap and recommends adding a price-history join in v2.

Usage::

    python -m scripts.audit_rejections                    # last 30d
    python -m scripts.audit_rejections --days 7
    python -m scripts.audit_rejections --since 2026-04-01

Outputs ``data/audit_reports/rejections_<period>.md``. The hourly CI
commits ``data/audit_reports/`` so the latest report lands on main.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HISTORY = ROOT / "data" / "saw_couldnt_trade_history.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "audit_reports"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY,
        help=f"Path to saw_couldnt_trade_history.csv (default: {DEFAULT_HISTORY})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for the markdown report (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history to include, ending today (default: 30)",
    )
    p.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD) to start the window from. Overrides --days when set.",
    )
    return p.parse_args(argv)


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"failed to read {path}: {e}")
        return pd.DataFrame()
    if df.empty or "scan_date" not in df.columns:
        return pd.DataFrame()
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce").dt.date
    df = df.dropna(subset=["scan_date"])
    return df


def _filter_window(
    df: pd.DataFrame,
    days: int,
    since: str | None,
) -> tuple[pd.DataFrame, str, str]:
    today = datetime.utcnow().date()
    if since:
        start = pd.to_datetime(since).date()
    else:
        start = today - timedelta(days=days)
    window = df[(df["scan_date"] >= start) & (df["scan_date"] <= today)].copy()
    return window, start.isoformat(), today.isoformat()


def _per_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("reject_reason_norm", dropna=False)
    out = grp.agg(
        n_total=("ticker", "size"),
        n_unique_tickers=("ticker", "nunique"),
        mean_flow_score=("flow_score_raw", "mean"),
        median_flow_score=("flow_score_raw", "median"),
        max_flow_score=("flow_score_raw", "max"),
    ).reset_index()
    out["mean_flow_score"] = out["mean_flow_score"].round(3)
    out["median_flow_score"] = out["median_flow_score"].round(3)
    out["max_flow_score"] = out["max_flow_score"].round(3)
    return out.sort_values("n_total", ascending=False).reset_index(drop=True)


def _top_tickers_per_reason(df: pd.DataFrame, top_n: int = 5) -> dict[str, list[dict]]:
    if df.empty:
        return {}
    out: dict[str, list[dict]] = {}
    for reason, grp in df.groupby("reject_reason_norm"):
        sub = grp.sort_values("flow_score_raw", ascending=False).head(top_n)
        out[reason] = sub[
            ["scan_date", "ticker", "direction", "flow_score_raw", "checks_failed"]
        ].to_dict(orient="records")
    return out


def _daily_mix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    counts = df.groupby(["scan_date", "reject_reason_norm"]).size().unstack(fill_value=0)
    counts.index = counts.index.astype(str)
    return counts.sort_index()


def _df_to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_\n"
    return df.to_markdown(index=False)


def _build_markdown(
    summary: pd.DataFrame,
    top_per_reason: dict[str, list[dict]],
    daily_mix: pd.DataFrame,
    start: str,
    end: str,
    n_days_with_data: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# Rejection audit report — {start} to {end}")
    lines.append("")
    lines.append(
        f"Built from `data/saw_couldnt_trade_history.csv` covering "
        f"{n_days_with_data} day(s) with data."
    )
    lines.append("")
    lines.append(
        "> **Forward returns are not yet attached.** v1 of the audit ships "
        "the rejection-mix tables; the killed-winner analysis needs a "
        "centralised price-history join (see TODO at the bottom)."
    )
    lines.append("")

    lines.append("## Summary by reject reason")
    lines.append("")
    lines.append(_df_to_md_table(summary))
    lines.append("")

    lines.append("## Top high-flow rejects per reason")
    lines.append("")
    for reason, rows in top_per_reason.items():
        lines.append(f"### {reason}")
        lines.append("")
        if rows:
            lines.append(_df_to_md_table(pd.DataFrame(rows)))
        else:
            lines.append("_(no rows)_")
        lines.append("")

    lines.append("## Daily reject-reason mix")
    lines.append("")
    if daily_mix.empty:
        lines.append("_(no daily data in window)_")
    else:
        lines.append(_df_to_md_table(daily_mix.reset_index()))
    lines.append("")

    lines.append("## TODO (v2)")
    lines.append("")
    lines.append(
        "- Attach forward 1d / 3d / 5d returns per (ticker, scan_date) "
        "so each reject reason can be scored on killed-winners count and "
        "false-rejection rate. Will require an OHLCV-by-date helper that "
        "covers the long tail of rejected tickers — many never reach the "
        "options-context fetch and therefore don't sit in the existing "
        "price-features cache."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df = _load_history(args.history_path)
    if df.empty:
        print(
            f"No data in {args.history_path}. Run the pipeline a few times "
            "to generate saw_couldnt_trade_history.csv before auditing."
        )
        return 1

    window, start, end = _filter_window(df, args.days, args.since)
    summary = _per_reason_summary(window)
    top_per_reason = _top_tickers_per_reason(window)
    daily_mix = _daily_mix(window)
    n_days = window["scan_date"].nunique() if not window.empty else 0

    md = _build_markdown(
        summary=summary,
        top_per_reason=top_per_reason,
        daily_mix=daily_mix,
        start=start,
        end=end,
        n_days_with_data=n_days,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.since:
        period = f"since_{start}_to_{end}"
    else:
        period = f"last_{args.days}d_to_{end}"
    out_path = args.out_dir / f"rejections_{period}.md"
    out_path.write_text(md)
    print(f"wrote {out_path} (rows in window: {len(window)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

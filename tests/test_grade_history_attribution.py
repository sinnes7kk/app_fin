"""Tests for grade history persistence + attribution (Grade Backtest
B1 / B2 / B3).

Synthetic-data approach: we seed a history file with ~30 rows where one
feature is *perfectly positively* correlated with the forward return
and another is *perfectly negatively* correlated.  After
``refresh_attribution`` runs, the Spearman ρ for those two features
should land near +1 and -1 respectively, while a random feature should
land near 0.

We also verify ``persist_grade_history`` is idempotent on
``(as_of, ticker, direction)`` so hourly re-scans don't inflate the
sample count.

Run with:

    python -m pytest tests/test_grade_history_attribution.py -v
    python tests/test_grade_history_attribution.py           # standalone
"""

from __future__ import annotations

import csv
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _grade_row(
    ticker: str,
    conviction_score: float,
    direction: str = "BULLISH",
    window_return_pct: float = 0.0,
    flow_intensity: float = 1.0,
) -> dict:
    return {
        "ticker": ticker,
        "direction": direction,
        "sector": "Tech",
        "conviction_score": conviction_score,
        "conviction_grade": "A" if conviction_score >= 7 else "B",
        "conviction_stack": conviction_score * 10,
        "flow_intensity": flow_intensity,
        "prem_mcap_bps": flow_intensity,
        "persistence_ratio": 0.8,
        "accumulation_score": 75.0,
        "sweep_share": 0.5,
        "multileg_share": 0.1,
        "accel_ratio_today": 1.2,
        "window_return_pct": window_return_pct,
        "cumulative_premium": 10_000_000.0,
        "latest_put_call_ratio": 0.3,
        "latest_iv_rank": 40.0,
        "latest_oi_change": 5.0,
        "perc_3_day_total_latest": 80.0,
        "perc_30_day_total_latest": 82.0,
        "dominant_dte_bucket": "swing",
        "premium_source": "screener",
    }


def test_persist_grade_history_is_idempotent_for_same_day():
    """Running twice in the same day must not double the row count."""
    from app.analytics import grade_history as gh

    tmp_path = Path(tempfile.mkdtemp()) / "grade_history.csv"
    with patch.object(gh, "GRADE_HISTORY_PATH", tmp_path):
        grades = [_grade_row("AAPL", 8.0), _grade_row("MSFT", 7.5)]
        gh.persist_grade_history(grades, as_of="2025-01-15")
        gh.persist_grade_history(grades, as_of="2025-01-15")

        rows = gh.load_history()

    same_day = [r for r in rows if r["as_of"] == "2025-01-15"]
    assert len(same_day) == 2, (
        f"idempotency broken: expected 2 rows for 2025-01-15, got {len(same_day)}"
    )


def test_persist_grade_history_appends_across_days():
    from app.analytics import grade_history as gh

    tmp_path = Path(tempfile.mkdtemp()) / "grade_history.csv"
    with patch.object(gh, "GRADE_HISTORY_PATH", tmp_path):
        gh.persist_grade_history([_grade_row("AAPL", 8.0)], as_of="2025-01-14")
        gh.persist_grade_history([_grade_row("AAPL", 8.0)], as_of="2025-01-15")

        rows = gh.load_history()

    dates = {r["as_of"] for r in rows}
    assert dates == {"2025-01-14", "2025-01-15"}, f"got {dates}"


def test_attach_forward_returns_fills_matured_rows():
    """``attach_forward_returns`` must:
      - fill rows whose ``as_of`` is far enough in the past,
      - leave recent rows untouched,
      - be idempotent (a second call attaches nothing new).
    """
    from app.analytics import grade_history as gh

    tmp_path = Path(tempfile.mkdtemp()) / "grade_history.csv"
    today = date.today()
    old = (today - timedelta(days=14)).isoformat()
    new = (today - timedelta(days=1)).isoformat()

    with patch.object(gh, "GRADE_HISTORY_PATH", tmp_path):
        gh.persist_grade_history([_grade_row("AAPL", 8.0)], as_of=old)
        gh.persist_grade_history([_grade_row("MSFT", 7.5)], as_of=new)

        def _fake_fwd(ticker, as_of, window=5):
            return 0.02 if ticker == "AAPL" else 0.01

        with patch(
            "app.analytics.grade_backtest._forward_excess_return",
            side_effect=_fake_fwd,
        ):
            attached_first = gh.attach_forward_returns(window=5)
            attached_second = gh.attach_forward_returns(window=5)

        rows = gh.load_history()
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["AAPL"]["forward_excess_return"].startswith("0.02"), (
        f"matured AAPL row not back-filled: {by_ticker['AAPL']}"
    )
    assert by_ticker["MSFT"]["forward_excess_return"] == "", (
        f"recent MSFT row back-filled prematurely: {by_ticker['MSFT']}"
    )
    assert attached_first == 1, f"first attach should fill 1 row, got {attached_first}"
    assert attached_second == 0, f"attach must be idempotent, got {attached_second}"


def test_attribution_recovers_known_correlations():
    """Seed a history where conviction_score is perfectly positively
    correlated with forward excess return, and window_return_pct is
    perfectly negatively correlated.  Attribution should find both
    signs."""
    from app.analytics import grade_attribution as ga
    from app.analytics import grade_history as gh

    tmp_hist = Path(tempfile.mkdtemp()) / "grade_history.csv"
    tmp_attr = tmp_hist.parent / "grade_attribution.json"

    today = date.today()
    n = 35  # > default min_samples
    old_dates = [(today - timedelta(days=10 + i)).isoformat() for i in range(n)]

    # Seed rows directly by writing to the CSV — bypasses the
    # persist_grade_history + attach_forward_returns plumbing so the
    # attribution test is focused.
    from app.analytics.grade_history import HISTORY_COLS

    rows = []
    for i, d in enumerate(old_dates):
        # conviction in [1..n]; forward in same order -> perfect +rank
        # window_return in reverse -> perfect -rank
        conviction = float(i + 1)
        window_ret = float(n - i)
        fwd = (i + 1) / 100.0  # strictly monotonic positive
        row = _grade_row(
            f"T{i:02d}",
            conviction_score=conviction,
            direction="BULLISH",
            window_return_pct=window_ret,
            flow_intensity=5.0,  # constant -> rho undefined / None
        )
        row_full = {c: "" for c in HISTORY_COLS}
        row_full.update({
            "as_of": d,
            "ticker": row["ticker"],
            "direction": row["direction"],
            "sector": row["sector"],
            "conviction_score": row["conviction_score"],
            "conviction_grade": row["conviction_grade"],
            "conviction_stack": row["conviction_stack"],
            "flow_intensity": row["flow_intensity"],
            "persistence_ratio": row["persistence_ratio"],
            "accumulation_score": row["accumulation_score"],
            "sweep_share": row["sweep_share"],
            "multileg_share": row["multileg_share"],
            "accel_ratio_today": row["accel_ratio_today"],
            "window_return_pct": row["window_return_pct"],
            "cumulative_premium": row["cumulative_premium"],
            "prem_mcap_bps": row["prem_mcap_bps"],
            "latest_put_call_ratio": row["latest_put_call_ratio"],
            "latest_iv_rank": row["latest_iv_rank"],
            "latest_oi_change": row["latest_oi_change"],
            "perc_3_day_total_latest": row["perc_3_day_total_latest"],
            "perc_30_day_total_latest": row["perc_30_day_total_latest"],
            "dominant_dte_bucket": row["dominant_dte_bucket"],
            "premium_source": row["premium_source"],
            "forward_excess_return": f"{fwd:.6f}",
            "forward_attached_at": "2025-01-01T00:00:00Z",
        })
        rows.append(row_full)

    with open(tmp_hist, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with patch.object(gh, "GRADE_HISTORY_PATH", tmp_hist), \
         patch.object(ga, "ATTRIBUTION_PATH", tmp_attr):
        report = ga.refresh_attribution(window_days=3650, min_samples=10)

    assert report.get("status") == "ok", f"attribution report failed: {report}"
    numeric = report.get("numeric", {})
    rho_cs = numeric.get("conviction_score", {}).get("rho")
    rho_wr = numeric.get("window_return_pct", {}).get("rho")

    assert rho_cs is not None, "conviction_score ρ missing"
    assert rho_wr is not None, "window_return_pct ρ missing"
    assert rho_cs > 0.9, (
        f"conviction_score should show strong +ρ, got {rho_cs}"
    )
    assert rho_wr < -0.9, (
        f"window_return_pct should show strong -ρ, got {rho_wr}"
    )


def test_attribution_reports_insufficient_history_below_threshold():
    from app.analytics import grade_attribution as ga
    from app.analytics import grade_history as gh

    tmp_hist = Path(tempfile.mkdtemp()) / "grade_history.csv"
    tmp_attr = tmp_hist.parent / "grade_attribution.json"

    with patch.object(gh, "GRADE_HISTORY_PATH", tmp_hist), \
         patch.object(ga, "ATTRIBUTION_PATH", tmp_attr):
        report = ga.refresh_attribution(min_samples=30)

    assert report.get("status") == "insufficient_history", (
        f"expected insufficient_history; got {report.get('status')}"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_persist_grade_history_is_idempotent_for_same_day,
        test_persist_grade_history_appends_across_days,
        test_attach_forward_returns_fills_matured_rows,
        test_attribution_recovers_known_correlations,
        test_attribution_reports_insufficient_history_below_threshold,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)

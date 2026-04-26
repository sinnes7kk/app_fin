"""Tests for ``app.reports.saw_couldnt_trade``.

Covers:

1. Filename parsing for ``rejected_YYYYMMDD_HHMMSS.csv``.
2. Multi-scan dedupe — each (ticker, direction) keeps only its
   latest-scan row.
3. ``normalize_reject_reason`` strips trailing parens, recognises
   the known enum, and falls through to ``other``.
4. ``build_panel`` includes high-flow rejects whose reason is in
   ``HIGH_FLOW_REJECT_REASONS`` and excludes weak-flow rejections.
5. The z-shadow path catches rows whose abs-path score is below the
   threshold but whose z-shadow is high.
6. ``append_history`` is idempotent on (scan_date, ticker, direction).

Run with either:

    python -m pytest tests/test_saw_couldnt_trade.py -v
    python -m tests.test_saw_couldnt_trade        # standalone (no pytest)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore

from app.reports.saw_couldnt_trade import (
    HIGH_FLOW_REJECT_REASONS,
    HIGH_FLOW_THRESHOLD,
    HIGH_FLOW_THRESHOLD_Z_SHADOW,
    PANEL_COLUMNS,
    _parse_scan_ts,
    append_history,
    build_panel,
    dedupe_latest_per_ticker,
    list_rejected_files_for_day,
    load_rejected_for_day,
    normalize_reject_reason,
)


def _write_rejected(
    directory: Path,
    stamp: str,
    rows: list[dict],
) -> Path:
    """Write a synthetic rejected_<stamp>.csv with the columns the panel reads."""
    df = pd.DataFrame(rows)
    path = directory / f"rejected_{stamp}.csv"
    df.to_csv(path, index=False)
    return path


def test_parse_scan_ts():
    p = Path("data/final_signals/rejected_20260422_230523.csv")
    out = _parse_scan_ts(p)
    assert out == ("2026-04-22", "23:05:23")
    assert _parse_scan_ts(Path("not_a_match.csv")) is None
    assert _parse_scan_ts(Path("rejected_2026.csv")) is None


def test_normalize_reject_reason():
    assert normalize_reject_reason("price_validation_failed") == "price_validation_failed"
    assert normalize_reject_reason("poor_rr (1.7:1)") == "poor_rr"
    assert normalize_reject_reason("poor_rr  (3.5:1) ") == "poor_rr"
    assert normalize_reject_reason("none") == "none"
    assert normalize_reject_reason("") == "none"
    assert normalize_reject_reason(None) == "none"
    assert normalize_reject_reason("something_we_have_not_seen") == "other"


def test_list_and_load_for_day():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # Two scans on 2026-04-22, one on 2026-04-23.
        _write_rejected(d, "20260422_140000", [
            {"ticker": "NVDA", "direction": "LONG", "flow_score_raw": 0.7,
             "reject_reason": "price_validation_failed"},
        ])
        _write_rejected(d, "20260422_220000", [
            {"ticker": "NVDA", "direction": "LONG", "flow_score_raw": 0.85,
             "reject_reason": "price_over_extended"},
            {"ticker": "AMD", "direction": "LONG", "flow_score_raw": 0.6,
             "reject_reason": "price_validation_failed"},
        ])
        _write_rejected(d, "20260423_140000", [
            {"ticker": "ANYTHING", "direction": "LONG", "flow_score_raw": 0.9,
             "reject_reason": "price_validation_failed"},
        ])

        files = list_rejected_files_for_day(d, "2026-04-22")
        assert len(files) == 2

        rej = load_rejected_for_day(d, "2026-04-22")
        assert len(rej) == 3
        assert "scan_ts" in rej.columns
        # Both stamp patterns should be present.
        assert {"2026-04-22T14:00:00", "2026-04-22T22:00:00"} == set(rej["scan_ts"].unique())


def test_dedupe_latest_per_ticker():
    rej = pd.DataFrame([
        {"ticker": "NVDA", "direction": "LONG", "scan_ts": "2026-04-22T14:00:00",
         "flow_score_raw": 0.7, "reject_reason": "price_validation_failed"},
        {"ticker": "NVDA", "direction": "LONG", "scan_ts": "2026-04-22T22:00:00",
         "flow_score_raw": 0.85, "reject_reason": "price_over_extended"},
        {"ticker": "AMD", "direction": "LONG", "scan_ts": "2026-04-22T22:00:00",
         "flow_score_raw": 0.6, "reject_reason": "price_validation_failed"},
    ])
    deduped = dedupe_latest_per_ticker(rej)
    assert len(deduped) == 2
    nvda_row = deduped[deduped["ticker"] == "NVDA"].iloc[0]
    assert nvda_row["flow_score_raw"] == 0.85
    assert nvda_row["reject_reason"] == "price_over_extended"


def test_build_panel_filters_high_flow_blocked():
    """High-flow rejects should land in the panel; weak-flow rejects must not."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_rejected(d, "20260422_220000", [
            # Should land in panel — high flow + blocked
            {"ticker": "NVDA", "direction": "LONG",
             "flow_score_raw": 0.8, "flow_score_scaled": 8.0,
             "reject_reason": "price_validation_failed"},
            {"ticker": "AMD", "direction": "LONG",
             "flow_score_raw": 0.6, "flow_score_scaled": 6.0,
             "reject_reason": "poor_rr (1.7:1)"},
            # Excluded — weak flow (correct rejection)
            {"ticker": "XYZ", "direction": "LONG",
             "flow_score_raw": 0.2, "flow_score_scaled": 2.0,
             "reject_reason": "weak_bullish_flow"},
            # Excluded — flow above threshold but reason is weak-flow
            {"ticker": "ABC", "direction": "LONG",
             "flow_score_raw": 0.6, "flow_score_scaled": 6.0,
             "reject_reason": "weak_bullish_flow"},
            # Excluded — high flow but watchlist re-eval (separate concern)
            {"ticker": "DEF", "direction": "LONG",
             "flow_score_raw": 0.7, "flow_score_scaled": 7.0,
             "reject_reason": "watchlist_reeval_failed"},
            # Excluded — flow below threshold
            {"ticker": "GHI", "direction": "LONG",
             "flow_score_raw": 0.4, "flow_score_scaled": 4.0,
             "reject_reason": "price_validation_failed"},
        ])
        panel = build_panel(d, "2026-04-22")
        tickers_in_panel = set(panel["ticker"].tolist())
        assert tickers_in_panel == {"NVDA", "AMD"}, (
            f"unexpected panel set: {tickers_in_panel}"
        )

        # Schema is intact and includes the normalized reason.
        assert set(panel.columns) == set(PANEL_COLUMNS)
        nvda = panel[panel["ticker"] == "NVDA"].iloc[0]
        assert nvda["reject_reason_norm"] == "price_validation_failed"
        amd = panel[panel["ticker"] == "AMD"].iloc[0]
        assert amd["reject_reason_norm"] == "poor_rr"


def test_z_shadow_path_catches_low_abs_score():
    """A row with abs flow_score_raw below the threshold but a high
    z-shadow score on the relevant side should still surface."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_rejected(d, "20260422_220000", [
            {"ticker": "STEALTH", "direction": "LONG",
             "flow_score_raw": 0.3,
             "bullish_score_z_shadow": 7.5,
             "bearish_score_z_shadow": 1.0,
             "reject_reason": "price_validation_failed"},
            # Same low abs but z is for the WRONG side -> should NOT land.
            {"ticker": "WRONGSIDE", "direction": "LONG",
             "flow_score_raw": 0.3,
             "bullish_score_z_shadow": 1.0,
             "bearish_score_z_shadow": 7.5,
             "reject_reason": "price_validation_failed"},
        ])
        panel = build_panel(d, "2026-04-22")
        assert "STEALTH" in panel["ticker"].tolist()
        assert "WRONGSIDE" not in panel["ticker"].tolist()


def test_known_high_flow_reasons_complete():
    """Sanity check: the most important blocked reasons we care about
    are all in HIGH_FLOW_REJECT_REASONS."""
    must_include = {
        "price_validation_failed",
        "price_over_extended",
        "poor_rr",
    }
    assert must_include.issubset(HIGH_FLOW_REJECT_REASONS)


def test_thresholds_visible():
    assert HIGH_FLOW_THRESHOLD == 0.5
    assert HIGH_FLOW_THRESHOLD_Z_SHADOW == 5.0


def test_append_history_idempotent():
    panel = pd.DataFrame([
        {
            "scan_date": "2026-04-22", "scan_ts": "2026-04-22T22:00:00",
            "ticker": "NVDA", "direction": "LONG",
            "flow_score_raw": 0.8, "flow_score_scaled": 8.0,
            "bullish_score_z_shadow": None, "bearish_score_z_shadow": None,
            "price_score": 5.0, "final_score": 6.0,
            "reject_reason": "price_validation_failed",
            "reject_reason_norm": "price_validation_failed",
            "checks_failed": "not_extended", "iv_rank": 50.0, "gamma_regime": "NEUTRAL",
        },
    ])

    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / "history.csv"
        append_history(panel, h)
        a = pd.read_csv(h)
        append_history(panel, h)
        b = pd.read_csv(h)
        assert len(a) == len(b), "duplicate appends should not grow the history"

        panel2 = panel.copy()
        panel2["scan_date"] = "2026-04-23"
        append_history(panel2, h)
        c = pd.read_csv(h)
        assert len(c) == len(b) + len(panel2)


def test_empty_inputs():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        # No files at all.
        assert list_rejected_files_for_day(d, "2026-04-22") == []
        assert load_rejected_for_day(d, "2026-04-22").empty
        assert build_panel(d, "2026-04-22").empty


def _run_all() -> int:
    fns = [
        test_parse_scan_ts,
        test_normalize_reject_reason,
        test_list_and_load_for_day,
        test_dedupe_latest_per_ticker,
        test_build_panel_filters_high_flow_blocked,
        test_z_shadow_path_catches_low_abs_score,
        test_known_high_flow_reasons_complete,
        test_thresholds_visible,
        test_append_history_idempotent,
        test_empty_inputs,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # pragma: no cover
            failures += 1
            print(f"  FAIL {fn.__name__}: unexpected {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())

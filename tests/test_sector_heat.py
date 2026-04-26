"""Tests for ``app.features.sector_heat``.

Covers:

1. Sub-sector overrides win over UW broad sector — semis-as-a-basket
   pulls NVDA / AMD / AVGO / LRCX out of the catch-all "Technology" GICS
   bucket so they aggregate together.
2. Single-ticker dominance is suppressed (n_tickers < MIN_TICKERS_FOR_HEAT).
3. Heat-score formula stays in 0-10 and is monotonic in both terms.
4. Empty / null inputs return an empty frame with the documented columns.
5. ``append_sector_heat_history`` is idempotent under repeat calls
   for the same ``(date, sector, direction)`` key.

Run with either:

    python -m pytest tests/test_sector_heat.py -v
    python -m tests.test_sector_heat            # standalone (no pytest)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore

from app.features import sector_heat
from app.features.sector_heat import (
    ELEVATED_SCORE_THRESHOLD,
    MIN_TICKERS_FOR_HEAT,
    append_sector_heat_history,
    compute_sector_heat,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Wrap rows in a DataFrame with sane defaults for missing fields."""
    base = []
    for r in rows:
        base.append(
            {
                "ticker": r["ticker"],
                "bullish_score": r.get("bullish_score", 0.0),
                "bearish_score": r.get("bearish_score", 0.0),
                "bullish_premium": r.get("bullish_premium", 0.0),
                "bearish_premium": r.get("bearish_premium", 0.0),
            }
        )
    return pd.DataFrame(base)


def test_sub_sector_overrides_win_over_broad_sector():
    """Semi names tagged 'Technology' by UW must aggregate as
    'Semiconductors' so the basket signal works."""
    ft = _frame(
        [
            {"ticker": "NVDA", "bullish_score": 0.7, "bullish_premium": 5_000_000},
            {"ticker": "AMD",  "bullish_score": 0.6, "bullish_premium": 3_000_000},
            {"ticker": "AVGO", "bullish_score": 0.55, "bullish_premium": 2_000_000},
            {"ticker": "LRCX", "bullish_score": 0.52, "bullish_premium": 1_500_000},
            # A non-semi tech name should land in the broad Technology bucket.
            {"ticker": "MSFT", "bullish_score": 0.3,  "bullish_premium": 800_000},
            {"ticker": "ORCL", "bullish_score": 0.2,  "bullish_premium": 500_000},
            {"ticker": "CRM",  "bullish_score": 0.25, "bullish_premium": 600_000},
        ]
    )
    meta = {
        # UW returns "Technology" for all of these.
        "NVDA": {"sector": "Technology"},
        "AMD":  {"sector": "Technology"},
        "AVGO": {"sector": "Technology"},
        "LRCX": {"sector": "Technology"},
        "MSFT": {"sector": "Technology"},
        "ORCL": {"sector": "Technology"},
        "CRM":  {"sector": "Technology"},
    }
    out = compute_sector_heat(ft, screener_meta=meta, snapshot_date="2026-04-22")
    assert not out.empty, "expected at least one heat row"

    sectors = set(out["sector"].tolist())
    assert "Semiconductors" in sectors, f"sub-sector override missing: {sectors}"
    assert "Technology" in sectors, "non-semi tech names should still bucket as Technology"

    bull_semi = out[
        (out["sector"] == "Semiconductors") & (out["direction"] == "bullish")
    ].iloc[0]
    assert bull_semi["n_tickers"] == 4
    assert bull_semi["n_above_thresh"] >= 3, (
        f"expected >=3 names above threshold, got {bull_semi['n_above_thresh']}"
    )
    expected_top = {"NVDA", "AMD", "AVGO", "LRCX"}
    actual_top = set(bull_semi["top_tickers"].split(","))
    assert actual_top == expected_top, f"top tickers wrong: {actual_top}"


def test_single_ticker_sector_suppressed():
    """A sector with fewer than MIN_TICKERS_FOR_HEAT names must not emit a row."""
    ft = _frame([{"ticker": "GOLD", "bullish_score": 0.95, "bullish_premium": 9_000_000}])
    out = compute_sector_heat(ft, screener_meta=None, snapshot_date="2026-04-22")
    assert out.empty, f"single-ticker sector should be suppressed but got: {out}"


def test_min_tickers_boundary():
    """Exactly MIN_TICKERS_FOR_HEAT names should fire; one fewer should not."""
    semis = ["NVDA", "AMD", "AVGO"]  # MIN_TICKERS_FOR_HEAT = 3
    assert len(semis) == MIN_TICKERS_FOR_HEAT
    ft = _frame([{"ticker": t, "bullish_score": 0.6} for t in semis])
    out = compute_sector_heat(ft, snapshot_date="2026-04-22")
    assert (out["sector"] == "Semiconductors").any(), (
        f"boundary case (n=={MIN_TICKERS_FOR_HEAT}) should fire: {out}"
    )

    ft2 = _frame([{"ticker": t, "bullish_score": 0.6} for t in semis[:-1]])
    out2 = compute_sector_heat(ft2, snapshot_date="2026-04-22")
    assert out2.empty, "below-threshold case should suppress"


def test_heat_score_bounded_and_monotonic():
    """Heat score must land in [0, 10] and grow with score + breadth."""
    cold = _frame([{"ticker": t, "bullish_score": 0.1} for t in ["NVDA", "AMD", "AVGO"]])
    warm = _frame([{"ticker": t, "bullish_score": 0.5} for t in ["NVDA", "AMD", "AVGO"]])
    hot  = _frame([{"ticker": t, "bullish_score": 0.95} for t in ["NVDA", "AMD", "AVGO"]])

    h_cold = compute_sector_heat(cold, snapshot_date="2026-04-22")
    h_warm = compute_sector_heat(warm, snapshot_date="2026-04-22")
    h_hot  = compute_sector_heat(hot,  snapshot_date="2026-04-22")

    s_cold = h_cold[h_cold["direction"] == "bullish"]["sector_heat_score"].iloc[0]
    s_warm = h_warm[h_warm["direction"] == "bullish"]["sector_heat_score"].iloc[0]
    s_hot  = h_hot[h_hot["direction"] == "bullish"]["sector_heat_score"].iloc[0]

    assert 0.0 <= s_cold <= 10.0
    assert 0.0 <= s_warm <= 10.0
    assert 0.0 <= s_hot  <= 10.0
    assert s_cold < s_warm < s_hot, (
        f"heat-score should be monotonic: cold={s_cold}, warm={s_warm}, hot={s_hot}"
    )

    # Threshold check: at exactly the threshold, share_above is 1.0 and
    # mean_top is 0.5 -> heat = 0.6 * 5.0 + 0.4 * 10.0 = 7.0.
    assert abs(s_warm - 7.0) < 0.01, f"expected ~7.0 at threshold, got {s_warm}"


def test_breadth_dominates_when_top_score_equal():
    """Two sectors with the same top scores but different breadths
    should rank by breadth."""
    # Both sectors have 3 strong names; sector A has 2 weak names dragging
    # share_above down, sector B has 2 strong names extending breadth.
    a = _frame(
        [
            {"ticker": "NVDA", "bullish_score": 0.8},
            {"ticker": "AMD",  "bullish_score": 0.8},
            {"ticker": "AVGO", "bullish_score": 0.8},
            {"ticker": "MU",   "bullish_score": 0.1},
            {"ticker": "LRCX", "bullish_score": 0.1},
        ]
    )
    b = _frame(
        [
            {"ticker": "GILD", "bullish_score": 0.8},
            {"ticker": "REGN", "bullish_score": 0.8},
            {"ticker": "VRTX", "bullish_score": 0.8},
            {"ticker": "MRNA", "bullish_score": 0.8},
            {"ticker": "BIIB", "bullish_score": 0.8},
        ]
    )
    ha = compute_sector_heat(a, snapshot_date="2026-04-22")
    hb = compute_sector_heat(b, snapshot_date="2026-04-22")
    sa = ha[ha["direction"] == "bullish"]["sector_heat_score"].iloc[0]
    sb = hb[hb["direction"] == "bullish"]["sector_heat_score"].iloc[0]
    assert sb > sa, f"breadth should boost B over A: a={sa}, b={sb}"


def test_elevated_threshold_count():
    ft = _frame(
        [
            {"ticker": "NVDA", "bullish_score": ELEVATED_SCORE_THRESHOLD + 0.1},
            {"ticker": "AMD",  "bullish_score": ELEVATED_SCORE_THRESHOLD + 0.05},
            {"ticker": "AVGO", "bullish_score": ELEVATED_SCORE_THRESHOLD - 0.01},
            {"ticker": "LRCX", "bullish_score": 0.05},
        ]
    )
    out = compute_sector_heat(ft, snapshot_date="2026-04-22")
    bull = out[out["direction"] == "bullish"].iloc[0]
    assert bull["n_above_thresh"] == 2, (
        f"only 2 should clear threshold, got {bull['n_above_thresh']}"
    )


def test_empty_input_returns_empty_frame_with_schema():
    out = compute_sector_heat(pd.DataFrame(), snapshot_date="2026-04-22")
    assert out.empty
    expected = {
        "snapshot_date", "sector", "direction", "n_tickers",
        "n_above_thresh", "share_above_thresh", "mean_score_topk",
        "max_score", "total_directional_premium", "top_tickers",
        "sector_heat_score",
    }
    assert set(out.columns) == expected, f"schema drift: {set(out.columns)}"


def test_append_history_idempotent():
    """Two appends of the same (date, sector, direction) keys should yield
    a single row each — the second append must overwrite, not duplicate."""
    ft = _frame([{"ticker": t, "bullish_score": 0.6} for t in ["NVDA", "AMD", "AVGO"]])
    h1 = compute_sector_heat(ft, snapshot_date="2026-04-22")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "history.csv"
        append_sector_heat_history(h1, path)
        rows1 = pd.read_csv(path)
        append_sector_heat_history(h1, path)
        rows2 = pd.read_csv(path)
        assert len(rows1) == len(rows2), (
            f"idempotency broken: {len(rows1)} vs {len(rows2)}"
        )

        # Different date -> rows should accumulate.
        h2 = compute_sector_heat(ft, snapshot_date="2026-04-23")
        append_sector_heat_history(h2, path)
        rows3 = pd.read_csv(path)
        assert len(rows3) == len(rows1) + len(h2), (
            "different-date append should add rows"
        )


def _run_all() -> int:
    """Standalone runner so this file works without pytest."""
    fns = [
        test_sub_sector_overrides_win_over_broad_sector,
        test_single_ticker_sector_suppressed,
        test_min_tickers_boundary,
        test_heat_score_bounded_and_monotonic,
        test_breadth_dominates_when_top_score_equal,
        test_elevated_threshold_count,
        test_empty_input_returns_empty_frame_with_schema,
        test_append_history_idempotent,
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

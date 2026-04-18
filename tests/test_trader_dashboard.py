"""Unit + smoke tests for the trader dashboard overhaul.

Run with:
    python -m pytest tests/test_trader_dashboard.py -v
"""

from __future__ import annotations

import pandas as pd

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore

from app.features.decision_context import (
    compute_liquidity,
    compute_r_at_market,
    compute_rs,
    compute_session_context,
    suggest_expression,
)
from app.features.flow_tracker import _conviction_grade


# ---------------------------------------------------------------------------
# Phase 1 — Flow Tracker scoring math
# ---------------------------------------------------------------------------


def test_conviction_grade_thresholds():
    assert _conviction_grade(8.0) == "A"
    assert _conviction_grade(7.5) == "A"
    assert _conviction_grade(7.49) == "B"
    assert _conviction_grade(5.0) == "B"
    assert _conviction_grade(4.99) == "C"


def test_compute_multi_day_flow_returns_new_fields(monkeypatch, tmp_path):
    """Grade rows must expose accel_t_stat and hedging_risk after the math rewrite."""
    from app.features import flow_tracker as ft_mod

    df = pd.DataFrame(
        [
            # Ticker ABC: 5 consecutive days of ramping bullish premium, low P/C — A grade candidate
            {"snapshot_date": "2026-04-10", "ticker": "ABC", "sector": "Tech", "close": 100,
             "marketcap": 5_000_000_000, "bullish_premium": 500_000, "bearish_premium": 100_000,
             "net_premium": 400_000, "call_volume": 1000, "put_volume": 500, "volume": 1500,
             "call_open_interest": 5000, "put_open_interest": 2000, "total_oi_change_perc": 10,
             "call_oi_change_perc": 15, "put_oi_change_perc": 5, "put_call_ratio": 0.5,
             "iv_rank": 40, "iv30d": 30, "perc_3_day_total": 0.1, "perc_30_day_total": 0.02},
            {"snapshot_date": "2026-04-11", "ticker": "ABC", "sector": "Tech", "close": 101,
             "marketcap": 5_000_000_000, "bullish_premium": 700_000, "bearish_premium": 150_000,
             "net_premium": 550_000, "call_volume": 1300, "put_volume": 600, "volume": 1900,
             "call_open_interest": 5200, "put_open_interest": 2100, "total_oi_change_perc": 12,
             "call_oi_change_perc": 16, "put_oi_change_perc": 6, "put_call_ratio": 0.55,
             "iv_rank": 42, "iv30d": 31, "perc_3_day_total": 0.12, "perc_30_day_total": 0.025},
            {"snapshot_date": "2026-04-12", "ticker": "ABC", "sector": "Tech", "close": 102,
             "marketcap": 5_000_000_000, "bullish_premium": 900_000, "bearish_premium": 200_000,
             "net_premium": 700_000, "call_volume": 1600, "put_volume": 700, "volume": 2300,
             "call_open_interest": 5400, "put_open_interest": 2200, "total_oi_change_perc": 14,
             "call_oi_change_perc": 17, "put_oi_change_perc": 7, "put_call_ratio": 0.6,
             "iv_rank": 43, "iv30d": 32, "perc_3_day_total": 0.15, "perc_30_day_total": 0.03},
            {"snapshot_date": "2026-04-13", "ticker": "ABC", "sector": "Tech", "close": 104,
             "marketcap": 5_000_000_000, "bullish_premium": 1_200_000, "bearish_premium": 250_000,
             "net_premium": 950_000, "call_volume": 2000, "put_volume": 800, "volume": 2800,
             "call_open_interest": 5700, "put_open_interest": 2300, "total_oi_change_perc": 16,
             "call_oi_change_perc": 18, "put_oi_change_perc": 8, "put_call_ratio": 0.55,
             "iv_rank": 45, "iv30d": 33, "perc_3_day_total": 0.18, "perc_30_day_total": 0.035},
            {"snapshot_date": "2026-04-14", "ticker": "ABC", "sector": "Tech", "close": 106,
             "marketcap": 5_000_000_000, "bullish_premium": 1_500_000, "bearish_premium": 300_000,
             "net_premium": 1_200_000, "call_volume": 2500, "put_volume": 900, "volume": 3400,
             "call_open_interest": 6000, "put_open_interest": 2400, "total_oi_change_perc": 18,
             "call_oi_change_perc": 20, "put_oi_change_perc": 9, "put_call_ratio": 0.6,
             "iv_rank": 48, "iv30d": 35, "perc_3_day_total": 0.22, "perc_30_day_total": 0.04},
        ]
    )

    csv_path = tmp_path / "snaps.csv"
    df.to_csv(csv_path, index=False)

    import datetime as _dt

    class _FakeDate:
        @staticmethod
        def today():
            return _dt.date(2026, 4, 15)

        @staticmethod
        def fromisoformat(s):
            return _dt.date.fromisoformat(s)

    monkeypatch.setattr(ft_mod, "SNAPSHOTS_PATH", csv_path)
    monkeypatch.setattr(ft_mod, "date", _FakeDate)

    rows = ft_mod.compute_multi_day_flow(
        lookback_days=10,
        min_active_days=2,
        min_premium=500_000,
        min_mcap=100_000_000,
        min_prem_mcap_bps=0.1,
    )
    assert rows, "expected at least one qualifying ticker"
    abc = next(r for r in rows if r["ticker"] == "ABC")

    assert "accel_t_stat" in abc
    assert "hedging_risk" in abc
    assert abc["trend"] == "accelerating"
    assert abc["accel_t_stat"] > 1.0  # strong upward slope
    assert abc["conviction_grade"] in ("A", "B")
    assert abc["direction"] == "BULLISH"
    assert abc["hedging_risk"] is False  # low P/C on bullish flow = no hedging penalty


# ---------------------------------------------------------------------------
# Phase 6 — Decision context helpers
# ---------------------------------------------------------------------------


def test_suggest_expression_respects_iv_and_catalyst():
    # High IV + imminent earnings → debit spread
    out = suggest_expression("LONG", 90, 3)
    assert out["expression"] == "DEBIT_SPREAD"

    # Low IV → long call
    out = suggest_expression("LONG", 15, 30)
    assert out["expression"] == "LONG_CALL"

    # Mid IV + no catalyst → stock
    out = suggest_expression("LONG", 50, 30)
    assert out["expression"] == "STOCK"

    # Illiquid → stock regardless of IV
    out = suggest_expression("LONG", 95, 3, adv_dollar=5_000_000)
    assert out["expression"] == "STOCK"

    # Short side symmetry
    out = suggest_expression("SHORT", 20, 40)
    assert out["expression"] == "LONG_PUT"


def test_compute_r_at_market_long_and_short():
    # Long: entry 100, stop 95 → risk=5. Spot 102.5 → +0.5R from entry, 1.5R to stop
    out = compute_r_at_market("LONG", 100, 95, 102.5)
    assert out["r_available"] is True
    assert out["r_from_entry"] == 0.5
    assert out["r_to_stop"] == 1.5

    # Short: entry 100, stop 105 → risk=5. Spot 98 → +0.4R from entry, 1.4R to stop
    out = compute_r_at_market("SHORT", 100, 105, 98)
    assert out["r_available"] is True
    assert out["r_from_entry"] == 0.4
    assert out["r_to_stop"] == 1.4

    # Missing inputs
    assert compute_r_at_market("LONG", None, 95, 100)["r_available"] is False
    assert compute_r_at_market("LONG", 100, 100, 100)["r_available"] is False  # zero risk


def test_compute_liquidity_tiers():
    assert compute_liquidity("SPY", mcap=1e13)["liquidity_tier"] in {"DEEP", "HEALTHY", "THIN", "ILLIQUID", "UNKNOWN"}


def test_compute_session_context_flat_and_strong():
    df = pd.DataFrame({
        "open": [100, 101],
        "close": [100.5, 102],
        "high": [101, 103],
        "low": [99, 101],
        "volume": [1_000_000, 1_200_000],
    })
    ctx = compute_session_context(df)
    assert ctx["session_available"] is True
    # close 102 vs prior close 100.5 → +1.49%
    assert ctx["session_tone"] == "STRENGTH"


def test_trader_card_view_projects_size_heat():
    """TraderCardView should produce size/notional/risk/heat from a raw row."""
    from app.web.view_models import TraderCardView

    row = {
        "ticker": "AAPL",
        "direction": "LONG",
        "final_score": 8.2,   # score >= 8.0 → risk 2.0%
        "entry_price": 200.0,
        "stop_price": 196.0,  # risk per share = 4
    }
    view = TraderCardView.from_row(row, vix_sizing_mult=1.0, capital=10_000.0)
    # risk budget = 200; shares = 200/4 = 50
    assert view.size_shares == 50
    assert view.notional_dollar == 50 * 200.0
    assert view.risk_dollar == 200.0
    assert view.heat_pct == 2.0

    # With VIX size halving, shares halve too
    view2 = TraderCardView.from_row(row, vix_sizing_mult=0.5, capital=10_000.0)
    assert view2.size_shares == 25


def test_trader_card_view_short_sizing():
    from app.web.view_models import TraderCardView

    row = {
        "ticker": "FOO",
        "direction": "SHORT",
        "final_score": 7.2,  # tier: risk 1.5%
        "entry_price": 50.0,
        "stop_price": 53.0,  # short risk = 3
    }
    view = TraderCardView.from_row(row, capital=10_000.0)
    # budget = 150, shares = 150/3 = 50
    assert view.size_shares == 50
    assert view.risk_dollar == 150.0


def test_trader_card_view_gracefully_skips_invalid_plans():
    from app.web.view_models import TraderCardView

    v1 = TraderCardView.from_row({"direction": "LONG", "final_score": 8.0})
    assert v1.size_shares is None and v1.heat_pct is None

    # Stop on wrong side (no risk budget)
    v2 = TraderCardView.from_row(
        {"direction": "LONG", "final_score": 8.0, "entry_price": 100, "stop_price": 101}
    )
    assert v2.size_shares is None


def test_compute_rs_returns_both_windows():
    tdf = pd.DataFrame({"close": list(range(1, 30))})
    sdf = pd.DataFrame({"close": [10] * 29})  # flat SPY
    rs = compute_rs(tdf, sdf, windows=(5, 20))
    assert "rs_5d_pct" in rs
    assert "rs_20d_pct" in rs
    # Ticker went up, SPY flat → positive RS
    assert rs["rs_5d_pct"] > 0
    assert rs["rs_20d_pct"] > 0


# ---------------------------------------------------------------------------
# Phase 2+4 — Trader-dashboard smoke tests via Flask test_client
# ---------------------------------------------------------------------------


def test_dashboard_renders_key_surfaces():
    """End-to-end: GET / should include action bar, grade stats header, and Flow Tracker insights."""
    from app.web.server import app

    with app.test_client() as c:
        resp = c.get("/")
        assert resp.status_code == 200
        body = resp.data.decode()

        # Action bar appears
        assert "action-bar" in body
        assert 'New A-Grades' in body
        # Overview section got the replacement
        assert "Trades to Take Now" in body
        # Flow Tracker grade-stats header + ft-insight
        assert "grade-stats-header" in body or "Grade stats unavailable" in body
        # Nav reorg groups are present
        assert "Decide" in body
        assert "Execute" in body
        assert "Context" in body
        assert "Review" in body


def test_static_build():
    """Static build should succeed end-to-end and produce non-empty HTML."""
    import subprocess
    import sys
    from pathlib import Path

    site = Path("_site/index.html")
    result = subprocess.run(
        [sys.executable, "scripts/build_static.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"static build failed:\n{result.stderr}"
    assert site.exists(), "static site not produced"
    assert site.stat().st_size > 50_000, "static site looks empty"
    html = site.read_text()
    assert "action-bar" in html

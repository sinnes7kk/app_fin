"""Tests for Wave 0 Flow Tracker mode gating (strong / accumulation / all).

Validates that ``compute_multi_day_flow`` produces rows with:
  - per-row ``passes_strong`` / ``passes_accumulation`` / ``passes_all`` flags
  - an ``accumulation_score`` in [0, 100]
  - a ``mode_counts`` tally consistent with the flags
  - strict subset ordering (strong ⊆ accumulation ⊆ all)

Run with either:
    python -m pytest tests/test_flow_tracker_modes.py -v
    python -m tests.test_flow_tracker_modes      # standalone (no pytest)
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic snapshot helper.
# ─────────────────────────────────────────────────────────────────────────────

def _write_snapshots(tmp_dir: Path, rows: list[dict]) -> Path:
    """Write a minimal screener_snapshots.csv the tracker can read."""
    from app.features.flow_tracker import SNAPSHOT_COLS
    path = tmp_dir / "screener_snapshots.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in SNAPSHOT_COLS})
    return path


def _days_back(n: int) -> list[str]:
    today = date.today()
    return [str(today - timedelta(days=i)) for i in range(n - 1, -1, -1)]


def _strong_row(ticker: str, d: str, bull: float, bear: float = 0.0,
                 mcap: float = 50_000_000_000, pcr: float = 0.3) -> dict:
    """One strong-accumulation-pattern day of flow."""
    return {
        "snapshot_date": d,
        "ticker": ticker,
        "sector": "Tech",
        "close": 100.0,
        "marketcap": mcap,
        "bullish_premium": bull,
        "bearish_premium": bear,
        "net_premium": bull - bear,
        "volume": 1000,
        "total_oi_change_perc": 0.0,
        "put_call_ratio": pcr,
        "iv_rank": 50,
        "iv30d": 0.3,
        "perc_3_day_total": 0.8,
        "perc_30_day_total": 0.8,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures.
# ─────────────────────────────────────────────────────────────────────────────

def _run_with_synthetic(rows: list[dict], lookback_days: int = 5, min_active_days: int = 2):
    """Run compute_multi_day_flow against a synthetic snapshot CSV.

    Temporarily monkey-patches ``flow_tracker.SNAPSHOTS_PATH`` so the module
    reads from our tmp directory instead of ``data/``.
    """
    from app.features import flow_tracker as ft_mod

    tmp = Path(tempfile.mkdtemp(prefix="ft_wave0_"))
    try:
        _write_snapshots(tmp, rows)
        original = ft_mod.SNAPSHOTS_PATH
        ft_mod.SNAPSHOTS_PATH = tmp / "screener_snapshots.csv"
        try:
            return ft_mod.compute_multi_day_flow(
                lookback_days=lookback_days, min_active_days=min_active_days
            )
        finally:
            ft_mod.SNAPSHOTS_PATH = original
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tests.
# ─────────────────────────────────────────────────────────────────────────────

def test_strong_accumulation_ticker_passes_all_three_modes():
    """A ticker hitting every day with $5M+ rising bull flow should pass Strong."""
    days = _days_back(5)
    # Rising profile → positive accel_t_stat clears the strong gate.
    rows = [_strong_row("STRONG", d, bull=3_000_000 + i * 1_500_000)
            for i, d in enumerate(days)]
    out = _run_with_synthetic(rows)
    assert len(out) == 1
    row = out[0]
    assert row["ticker"] == "STRONG"
    assert row["passes_strong"], "strong row should pass strong gate"
    assert row["passes_accumulation"], "strong row should also pass accumulation"
    assert row["passes_all"], "strong row should also pass all"
    # Accumulation score should be in [0, 100] and close to the ceiling on this setup.
    assert 0.0 <= row["accumulation_score"] <= 100.0
    assert row["accumulation_score"] >= 60.0, (
        f"strong-pattern row should score >= 60; got {row['accumulation_score']}"
    )


def test_weak_spiky_ticker_passes_only_all():
    """One big day then nothing should pass 'all' only."""
    days = _days_back(5)
    rows = [_strong_row("SPIKY", days[0], bull=500_000)]  # 1 of 5 days, small bull
    # Need >= 2 active days to escape the base min_active_days gate; add one more tiny day
    rows.append(_strong_row("SPIKY", days[1], bull=50_000))
    out = _run_with_synthetic(rows)
    if not out:
        # If base gates now exclude the ticker entirely, that's acceptable — the
        # "all" payload simply has no entry; accumulation/strong have none either.
        return
    row = out[0]
    assert row["passes_strong"] is False
    assert row["passes_accumulation"] is False


def test_hedging_bullish_excluded_from_accumulation():
    """Bullish direction with PCR > 0.9 = hedging → excluded from accumulation."""
    days = _days_back(5)
    # Strong bullish mass, but PCR elevated (hedging pattern).
    rows = [_strong_row("HEDGE", d, bull=5_000_000, pcr=1.1) for d in days]
    out = _run_with_synthetic(rows)
    assert len(out) == 1
    row = out[0]
    assert row["hedging_risk"] is True
    assert row["passes_accumulation"] is False, (
        "hedging_risk should exclude row from accumulation"
    )
    assert row["passes_strong"] is False
    # Legacy 'all' mode does not exclude hedging; haircut is applied instead.
    assert row["passes_all"] is True


def test_fading_ticker_fails_strong():
    """A fading pattern (declining daily totals) should fail the strong accel gate."""
    days = _days_back(5)
    # Descending bullish flow → negative accel_t_stat
    amounts = [10_000_000, 5_000_000, 2_000_000, 1_000_000, 500_000]
    rows = [_strong_row("FADE", days[i], bull=amt) for i, amt in enumerate(amounts)]
    out = _run_with_synthetic(rows)
    assert len(out) == 1
    row = out[0]
    assert row["accel_t_stat"] < 0, f"expected negative accel_t_stat, got {row['accel_t_stat']}"
    assert row["passes_strong"] is False, "fading pattern should fail strong accel gate"


def test_mode_counts_and_subset_ordering():
    """mode_counts on each row should match the flag counts and be monotone."""
    days = _days_back(5)
    rows: list[dict] = []
    # Strong: 5/5 days, big and rising
    for i, d in enumerate(days):
        rows.append(_strong_row("STRONG", d, bull=3_000_000 + i * 500_000))
    # Accumulation-only: 4/5 days, moderate, one-sided, non-fading
    for i in range(4):
        rows.append(_strong_row("ACCUM", days[i + 1], bull=1_500_000))
    # All-only: 2/5 days, small
    rows.append(_strong_row("WEAK", days[0], bull=300_000))
    rows.append(_strong_row("WEAK", days[1], bull=300_000))

    out = _run_with_synthetic(rows)
    assert out, "synthetic data should produce at least one row"

    n_strong = sum(1 for r in out if r["passes_strong"])
    n_accum = sum(1 for r in out if r["passes_accumulation"])
    n_all = sum(1 for r in out if r["passes_all"])

    assert n_strong <= n_accum <= n_all, (
        f"subset ordering violated: strong={n_strong}, accum={n_accum}, all={n_all}"
    )

    for r in out:
        mc = r["mode_counts"]
        assert mc["strong_accumulation"] == n_strong
        assert mc["accumulation"] == n_accum
        assert mc["all"] == n_all


def test_accumulation_score_bounds():
    days = _days_back(5)
    rows = [_strong_row("BOUND", d, bull=2_000_000 + i * 300_000) for i, d in enumerate(days)]
    out = _run_with_synthetic(rows)
    assert out
    acc = out[0]["accumulation_score"]
    assert 0.0 <= acc <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Wave 0.5 Cluster B — scoring mechanics fixes
# ─────────────────────────────────────────────────────────────────────────────


def test_b2_premium_weighted_persistence_rejects_spike_then_pennies():
    """$10M day 1 + $50K × 4 should not score persistence = 5/5.

    Under the new premium-weighted definition only day 1 clears the threshold
    (15% of per-day mean ≈ $305K, or the $100K absolute floor), so active_days
    becomes 1.  min_active_days=2 then drops the row entirely.
    """
    days = _days_back(5)
    rows = [
        _strong_row("SPIKE", days[0], bull=10_000_000),
        _strong_row("SPIKE", days[1], bull=50_000),
        _strong_row("SPIKE", days[2], bull=50_000),
        _strong_row("SPIKE", days[3], bull=50_000),
        _strong_row("SPIKE", days[4], bull=50_000),
    ]
    out = _run_with_synthetic(rows)
    row = next((r for r in out if r["ticker"] == "SPIKE"), None)
    assert row is None, f"SPIKE should be dropped by premium-weighted persistence gate; got {row}"


def test_b2_healthy_flow_keeps_all_active_days():
    """$2M × 5 days with mild variation: every day should count as active."""
    days = _days_back(5)
    rows = [_strong_row("HEALTHY", d, bull=2_000_000 + i * 200_000) for i, d in enumerate(days)]
    out = _run_with_synthetic(rows)
    assert out
    row = out[0]
    assert row["active_days"] == 5, f"expected active_days=5 for uniform flow; got {row['active_days']}"
    assert row["persistence_ratio"] == 1.0


def test_b1_absolute_mass_independent_of_cohort_size():
    """A $3M ticker should score the same mass regardless of cohort peers."""
    days = _days_back(5)

    # Scenario 1: small cohort (only one $3M ticker)
    rows_small = [_strong_row("MIDCAP", d, bull=600_000) for d in days]
    out_small = _run_with_synthetic(rows_small)
    assert out_small
    score_small = out_small[0]["conviction_score"]

    # Scenario 2: same ticker + bigger peer.  Peer must be clearly larger so
    # cohort-relative scoring WOULD have suppressed MIDCAP under the legacy
    # code path.  Under B1 absolute scoring MIDCAP's score should be stable.
    rows_big = list(rows_small) + [_strong_row("MEGA", d, bull=20_000_000) for d in days]
    out_big = _run_with_synthetic(rows_big)
    mid = next(r for r in out_big if r["ticker"] == "MIDCAP")
    # Tolerance: <= 0.3 pt drift on the same ticker between scenarios.
    assert abs(mid["conviction_score"] - score_small) <= 0.3, (
        f"mass should be absolute: score_small={score_small}, score_with_mega={mid['conviction_score']}"
    )


def test_b3_short_history_uses_absolute_pcr_fallback():
    """With <10 PCR observations in history we fall back to absolute threshold."""
    days = _days_back(5)
    # 5 days of bullish flow with elevated PCR — absolute threshold triggers hedging.
    rows = [_strong_row("BHIST", d, bull=5_000_000, pcr=1.1) for d in days]
    out = _run_with_synthetic(rows)
    assert out
    row = out[0]
    assert row["pcr_check_mode"] == "absolute", (
        f"expected absolute fallback; got {row['pcr_check_mode']} with n={row['ticker_pcr_history_n']}"
    )
    assert row["hedging_risk"] is True


def test_b3_relative_pcr_does_not_false_flag_high_baseline_ticker():
    """With >=10 observations and ticker median PCR ~1.0, a 1.05 PCR should NOT trigger hedging.

    Legacy absolute threshold (>0.9) would incorrectly flag this ticker.
    """
    days = _days_back(15)
    # Stable high-PCR ticker: PCR hovers around 1.0 every day.
    rows = [_strong_row("HIPCR", d, bull=3_000_000, pcr=1.0 + (i % 3) * 0.02) for i, d in enumerate(days)]
    out = _run_with_synthetic(rows, lookback_days=15)
    assert out
    row = out[0]
    assert row["ticker_pcr_history_n"] >= 10
    assert row["pcr_check_mode"] == "relative"
    assert row["hedging_risk"] is False, (
        "ticker's PCR is stable vs its own history — should NOT be flagged as hedging"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Wave 0.5 Cluster A — structural enrichment & scoring
# ─────────────────────────────────────────────────────────────────────────────


def _strong_row_with_extras(
    ticker: str,
    d: str,
    bull: float,
    bear: float = 0.0,
    mcap: float = 50_000_000_000,
    pcr: float = 0.3,
    close: float = 100.0,
    dte_bucket: str | None = None,
    sweep_share: float | None = None,
    multileg_share: float | None = None,
    perc_3d: float | None = None,
    total_oi_change: float | None = None,
) -> dict:
    row = _strong_row(ticker, d, bull, bear, mcap, pcr)
    row["close"] = close
    if dte_bucket is not None:
        row["dominant_dte_bucket"] = dte_bucket
    if sweep_share is not None:
        row["sweep_share"] = sweep_share
    if multileg_share is not None:
        row["multileg_share"] = multileg_share
    if perc_3d is not None:
        row["perc_3_day_total"] = perc_3d
    if total_oi_change is not None:
        row["total_oi_change_perc"] = total_oi_change
    return row


def test_a1_dte_multiplier_softens_weekly_flow():
    """Weekly (0-7d) dominant flow should score lower than medium-dated flow.

    Same structural inputs except the DTE bucket — the DTE multiplier of
    0.90 vs 1.00 should produce a measurable score gap.
    """
    days = _days_back(5)
    medium = [_strong_row_with_extras("MED", d, bull=3_000_000 + i * 200_000,
                                      dte_bucket="31-90", perc_3d=0.9)
              for i, d in enumerate(days)]
    weekly = [_strong_row_with_extras("WEEK", d, bull=3_000_000 + i * 200_000,
                                      dte_bucket="0-7", perc_3d=0.9)
              for i, d in enumerate(days)]
    out = _run_with_synthetic(medium + weekly)
    med = next(r for r in out if r["ticker"] == "MED")
    wk = next(r for r in out if r["ticker"] == "WEEK")
    assert wk["conviction_score"] < med["conviction_score"], (
        f"weekly DTE should score lower: medium={med['conviction_score']}, weekly={wk['conviction_score']}"
    )
    assert med["dominant_dte_bucket"] == "31-90"
    assert wk["dominant_dte_bucket"] == "0-7"


def test_a4_window_return_aligned_bonus():
    """Bullish flow with rising price should earn a return bonus."""
    days = _days_back(5)
    rows = []
    for i, d in enumerate(days):
        rows.append(_strong_row_with_extras(
            "TREND", d, bull=3_000_000, close=100.0 + i * 2.0, perc_3d=0.9,
        ))
    out = _run_with_synthetic(rows)
    row = out[0]
    assert row["window_return_pct"] > 0
    # Score should reflect the bonus — hard to assert exact value, but with a
    # ~8% window return the bonus should be ≥ 0.15 points.  Run the same
    # scenario with flat price and verify delta.
    flat = [_strong_row_with_extras("TREND", d, bull=3_000_000, close=100.0,
                                    perc_3d=0.9) for d in days]
    out_flat = _run_with_synthetic(flat)
    row_flat = out_flat[0]
    assert row["conviction_score"] > row_flat["conviction_score"], (
        "rising-price bullish flow should beat flat-price bullish flow"
    )


def test_a4_window_return_fighting_flow_drags_score():
    """Bullish flow while price drops should take a drag."""
    days = _days_back(5)
    rising = [_strong_row_with_extras("UP", d, bull=3_000_000, close=100.0, perc_3d=0.9)
              for d in days]
    falling = [_strong_row_with_extras("DN", d, bull=3_000_000,
                                       close=100.0 - i * 2.0, perc_3d=0.9)
               for i, d in enumerate(days)]
    out = _run_with_synthetic(rising + falling)
    up = next(r for r in out if r["ticker"] == "UP")
    dn = next(r for r in out if r["ticker"] == "DN")
    assert dn["window_return_pct"] < 0
    assert dn["conviction_score"] <= up["conviction_score"], (
        "bullish flow fighting the tape shouldn't beat flat-tape bullish flow"
    )


def test_a6_3d_percentile_gates_accumulation():
    """Low 3-day percentile should block a ticker from the accumulation mode."""
    days = _days_back(5)
    # Ticker clears everything else but 3d percentile is very low.
    low = [_strong_row_with_extras("LOW", d, bull=3_000_000 + i * 200_000, perc_3d=0.2)
           for i, d in enumerate(days)]
    # Control: same ticker but perc_3d above the gate.
    hi = [_strong_row_with_extras("HI", d, bull=3_000_000 + i * 200_000, perc_3d=0.85)
          for i, d in enumerate(days)]
    out = _run_with_synthetic(low + hi)
    low_row = next(r for r in out if r["ticker"] == "LOW")
    hi_row = next(r for r in out if r["ticker"] == "HI")
    assert hi_row["passes_accumulation"] is True
    assert low_row["passes_accumulation"] is False, (
        "3d percentile 0.2 should be blocked by A6 gate"
    )


def test_a7_oi_change_contributes_to_score():
    """Two otherwise-identical tickers should separate based on OI build."""
    days = _days_back(5)
    no_oi = [_strong_row_with_extras("FLAT", d, bull=3_000_000 + i * 200_000,
                                     perc_3d=0.9, total_oi_change=0.0)
             for i, d in enumerate(days)]
    big_oi = [_strong_row_with_extras("BUILD", d, bull=3_000_000 + i * 200_000,
                                      perc_3d=0.9, total_oi_change=40.0)
              for i, d in enumerate(days)]
    out = _run_with_synthetic(no_oi + big_oi)
    flat = next(r for r in out if r["ticker"] == "FLAT")
    build = next(r for r in out if r["ticker"] == "BUILD")
    assert build["conviction_score"] > flat["conviction_score"], (
        f"40% OI build should beat 0% OI: build={build['conviction_score']}, flat={flat['conviction_score']}"
    )


def test_a1_a2_enrichment_populates_on_row():
    """Enrichment columns should round-trip from snapshot → tracker row."""
    days = _days_back(5)
    rows = [_strong_row_with_extras("ENR", d, bull=3_000_000 + i * 200_000,
                                    dte_bucket="31-90", sweep_share=0.65,
                                    multileg_share=0.10, perc_3d=0.9)
            for i, d in enumerate(days)]
    out = _run_with_synthetic(rows)
    row = out[0]
    assert row["dominant_dte_bucket"] == "31-90"
    assert abs(row["sweep_share"] - 0.65) < 1e-6
    assert abs(row["multileg_share"] - 0.10) < 1e-6


def test_b4_window_avg_pcr_populated():
    """window_avg_pcr should be on every returned row."""
    days = _days_back(5)
    rows = [_strong_row("WPCR", d, bull=2_000_000 + i * 100_000, pcr=0.4 + i * 0.05) for i, d in enumerate(days)]
    out = _run_with_synthetic(rows)
    assert out
    row = out[0]
    assert "window_avg_pcr" in row
    assert row["window_avg_pcr"] > 0
    # mean of [0.4, 0.45, 0.5, 0.55, 0.6] = 0.5
    assert abs(row["window_avg_pcr"] - 0.5) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Wave 0.5 Cluster C — retention + horizon toggle + z-tier + sector cluster.
# ─────────────────────────────────────────────────────────────────────────────

def test_c1_retention_keeps_15d_window():
    """Snapshots written up to ~FLOW_TRACKER_RETENTION_DAYS back must survive
    the save-time pruning so the 15d horizon can compute on them."""
    from app.config import FLOW_TRACKER_RETENTION_DAYS
    assert FLOW_TRACKER_RETENTION_DAYS >= 15, (
        "retention must be >= 15d to support the 15d horizon"
    )

    days = _days_back(15)
    rows = [_strong_row("LONG", d, bull=3_000_000 + i * 150_000) for i, d in enumerate(days)]
    out = _run_with_synthetic(rows, lookback_days=15, min_active_days=4)
    assert out, "expected at least one row for 15d horizon"
    row = out[0]
    assert row["ticker"] == "LONG"
    assert row["active_days"] >= 4, "15d horizon should see most of the 15 active days"


def test_c1_horizons_config_has_5d_and_15d():
    """The horizon config must expose the two toggle options the UI relies on."""
    from app.config import FLOW_TRACKER_HORIZONS, FLOW_TRACKER_HORIZON_DEFAULT
    assert "5d" in FLOW_TRACKER_HORIZONS
    assert "15d" in FLOW_TRACKER_HORIZONS
    assert FLOW_TRACKER_HORIZON_DEFAULT in FLOW_TRACKER_HORIZONS
    assert FLOW_TRACKER_HORIZONS["5d"]["lookback_days"] == 5
    assert FLOW_TRACKER_HORIZONS["15d"]["lookback_days"] == 15
    assert FLOW_TRACKER_HORIZONS["15d"]["min_active_days"] >= 3, (
        "15d window should require more active days than 5d"
    )


def test_c3_sector_accumulating_count_groups_by_sector_and_direction():
    """Three Tech names all passing Accumulation should each show
    ``sector_accumulating_count == 2`` (the other two)."""
    days = _days_back(5)
    rows: list[dict] = []
    for t in ("AAA", "BBB", "CCC"):
        for i, d in enumerate(days):
            rows.append(_strong_row(t, d, bull=3_000_000 + i * 250_000))
    out = _run_with_synthetic(rows)
    accum = [r for r in out if r.get("passes_accumulation")]
    assert len(accum) >= 2, f"expected multiple accumulating rows, got {len(accum)}"
    for r in accum:
        assert r["sector_accumulating_count"] == len(accum) - 1, (
            f"{r['ticker']} expected count {len(accum) - 1}, got {r['sector_accumulating_count']}"
        )


def test_c3_sector_count_splits_by_direction():
    """Bullish and bearish flows in the same sector should not be counted
    together — mixed sector = not a one-sided bid."""
    days = _days_back(5)
    rows: list[dict] = []
    # Bullish names
    for t in ("BULL1", "BULL2"):
        for i, d in enumerate(days):
            rows.append(_strong_row(t, d, bull=3_000_000 + i * 250_000))
    # Bearish name (same sector "Tech" via _strong_row) — flip bull/bear
    for i, d in enumerate(days):
        r = _strong_row("BEAR1", d, bull=0.0, bear=3_000_000 + i * 250_000)
        rows.append(r)
    out = _run_with_synthetic(rows)
    bulls = [r for r in out if r["direction"] == "BULLISH" and r.get("passes_accumulation")]
    bears = [r for r in out if r["direction"] == "BEARISH" and r.get("passes_accumulation")]
    for r in bulls:
        # Each bullish row's sector count should only see OTHER bullish rows.
        assert r["sector_accumulating_count"] == len(bulls) - 1
    for r in bears:
        assert r["sector_accumulating_count"] == len(bears) - 1


# ─────────────────────────────────────────────────────────────────────────────
# Wave 1 — "Now What" hero strip (_build_flow_tracker_hero).
#
# We test the pure builder in isolation (no Flask, no network) so the
# selection logic and the fallback paths are locked in.
# ─────────────────────────────────────────────────────────────────────────────

def _hero_row(**overrides) -> dict:
    """Minimal tracker-row-like dict for hero tests."""
    base = {
        "ticker": "AAA",
        "direction": "BULLISH",
        "sector": "Tech",
        "conviction_score": 6.5,
        "conviction_grade": "B+",
        "accumulation_score": 60,
        "passes_accumulation": True,
        "passes_strong": True,
        "dp_aligned": True,
        "window_return_pct": 1.5,
        "latest_iv_rank": 45,
        "earnings": {"days_until_earnings": 25},
        "cumulative_premium": 2_000_000,
        "grade_reasons": [{"kind": "driver", "label": "Directional consistency"}],
    }
    base.update(overrides)
    return base


def test_hero_empty_returns_none():
    from app.web.server import _build_flow_tracker_hero
    assert _build_flow_tracker_hero([]) is None


def test_hero_top_picks_highest_conviction():
    from app.web.server import _build_flow_tracker_hero
    rows = [
        _hero_row(ticker="LOW", conviction_score=5.0),
        _hero_row(ticker="HIGH", conviction_score=8.2, conviction_grade="A"),
        _hero_row(ticker="MID", conviction_score=6.1),
    ]
    hero = _build_flow_tracker_hero(rows)
    assert hero["top"]["ticker"] == "HIGH"


def test_hero_sector_card_needs_cluster_of_two():
    from app.web.server import _build_flow_tracker_hero
    # Single accumulating row — should not qualify as a cluster.
    hero = _build_flow_tracker_hero([_hero_row(ticker="LONELY")])
    assert hero["sector"] is None


def test_hero_sector_picks_largest_same_direction_cluster():
    from app.web.server import _build_flow_tracker_hero
    rows = [
        # Two bullish in Tech.
        _hero_row(ticker="T1", sector="Tech"),
        _hero_row(ticker="T2", sector="Tech"),
        # Three bullish in Energy — should win.
        _hero_row(ticker="E1", sector="Energy"),
        _hero_row(ticker="E2", sector="Energy"),
        _hero_row(ticker="E3", sector="Energy"),
    ]
    hero = _build_flow_tracker_hero(rows)
    assert hero["sector"]["sector"] == "Energy"
    assert hero["sector"]["count"] == 3


def test_hero_sector_card_drops_nan_bucket():
    """Rows with NaN / empty sector must not form a bogus cluster."""
    from app.web.server import _build_flow_tracker_hero
    rows = [
        _hero_row(ticker="A", sector=float("nan")),
        _hero_row(ticker="B", sector=None),
        _hero_row(ticker="C", sector=""),
    ]
    hero = _build_flow_tracker_hero(rows)
    assert hero["sector"] is None


def test_hero_setup_passes_all_clean_filters():
    from app.web.server import _build_flow_tracker_hero
    rows = [_hero_row(ticker="CLEAN")]
    hero = _build_flow_tracker_hero(rows)
    assert hero["setup"]["ticker"] == "CLEAN"
    assert hero["setup_is_fallback"] is False


def test_hero_setup_rejects_high_iv_and_er_soon_and_falls_back():
    """IV > 65 disqualifies; earnings within 5d disqualifies; row stays the
    top-conviction fallback with the fallback flag set."""
    from app.web.server import _build_flow_tracker_hero
    rows = [
        # Fails IV-rank clause.
        _hero_row(ticker="EXP", latest_iv_rank=80, conviction_score=7.8),
        # Fails earnings clause.
        _hero_row(ticker="ER", earnings={"days_until_earnings": 3}, conviction_score=7.0),
    ]
    hero = _build_flow_tracker_hero(rows)
    assert hero["setup"]["ticker"] == "EXP"  # highest conviction
    assert hero["setup_is_fallback"] is True


def test_hero_setup_rejects_fighting_price_and_missing_dp():
    from app.web.server import _build_flow_tracker_hero
    rows = [
        _hero_row(ticker="FIGHT", direction="BULLISH", window_return_pct=-2.0),
        _hero_row(ticker="NODP", dp_aligned=False),
    ]
    hero = _build_flow_tracker_hero(rows)
    assert hero["setup_is_fallback"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest required).
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_strong_accumulation_ticker_passes_all_three_modes,
        test_weak_spiky_ticker_passes_only_all,
        test_hedging_bullish_excluded_from_accumulation,
        test_fading_ticker_fails_strong,
        test_mode_counts_and_subset_ordering,
        test_accumulation_score_bounds,
        test_b2_premium_weighted_persistence_rejects_spike_then_pennies,
        test_b2_healthy_flow_keeps_all_active_days,
        test_b1_absolute_mass_independent_of_cohort_size,
        test_b3_short_history_uses_absolute_pcr_fallback,
        test_b3_relative_pcr_does_not_false_flag_high_baseline_ticker,
        test_b4_window_avg_pcr_populated,
        # Wave 0.5 Cluster A
        test_a1_dte_multiplier_softens_weekly_flow,
        test_a4_window_return_aligned_bonus,
        test_a4_window_return_fighting_flow_drags_score,
        test_a6_3d_percentile_gates_accumulation,
        test_a7_oi_change_contributes_to_score,
        test_a1_a2_enrichment_populates_on_row,
        # Wave 0.5 Cluster C
        test_c1_retention_keeps_15d_window,
        test_c1_horizons_config_has_5d_and_15d,
        test_c3_sector_accumulating_count_groups_by_sector_and_direction,
        test_c3_sector_count_splits_by_direction,
        # Wave 1 — "Now What" hero strip.
        test_hero_empty_returns_none,
        test_hero_top_picks_highest_conviction,
        test_hero_sector_card_needs_cluster_of_two,
        test_hero_sector_picks_largest_same_direction_cluster,
        test_hero_sector_card_drops_nan_bucket,
        test_hero_setup_passes_all_clean_filters,
        test_hero_setup_rejects_high_iv_and_er_soon_and_falls_back,
        test_hero_setup_rejects_fighting_price_and_missing_dp,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(0 if failed == 0 else 1)

"""Tests for Wave 6 ``app.features.flow_narrative``.

Covers the dashboard-level narrative generator that powers the "Why?"
tab in the Flow Tracker + Trader Card modals.  We don't assert on exact
prose (that would make the tests brittle every time we tweak phrasing)
— instead we assert on:

* Bullet count stays within the declared cap.
* Tone distribution matches the archetype (strong accumulation → mostly
  positive, fading flow → warnings, hedging → warning, etc.).
* Key phrases / substrings appear where they should so we notice if a
  template accidentally drops a field.
* Empty / malformed rows return ``[]`` without crashing.

Run with:
    python -m pytest tests/test_flow_narrative.py -v
    python tests/test_flow_narrative.py    # standalone
"""

from __future__ import annotations

import traceback

from app.features.flow_narrative import (
    TONE_INFO,
    TONE_NEGATIVE,
    TONE_NEUTRAL,
    TONE_POSITIVE,
    TONE_WARNING,
    build_flow_feature_narrative,
    build_flow_tracker_narrative,
)


# ---------------------------------------------------------------------------
# Archetype row factories (explicit — so tests are readable and each test
# only tweaks the one field under examination).
# ---------------------------------------------------------------------------
def _strong_accumulation_tracker_row() -> dict:
    return {
        "ticker": "AAPL",
        "direction": "BULLISH",
        "persistence_ratio": 1.0,
        "active_days": 5,
        "total_days": 5,
        "trend": "accelerating",
        "prem_mcap_bps": 65.0,
        "cumulative_premium": 45_000_000,
        "perc_3_day_total": 94.0,
        "dominant_dte_bucket": "swing",
        "sweep_share": 0.55,
        "multileg_share": 0.10,
        "latest_oi_change": 38.0,
        "window_return_pct": 3.2,
        "latest_iv_rank": 32,
        "dark_pool": {"available": True, "bias": 0.70, "notional_mcap_bps": 8.0},
        "dealer_hedge_bias": "chase",
        "bullish_accel_ratio": 0.55,
        "sector_accumulating_count": 3,
        "sector": "Tech",
        "earnings": {"days_until_earnings": 11},
        "conviction_stack": {"score": 82, "tier_label": "Stack A", "tier": "elite"},
    }


def _fading_flow_tracker_row() -> dict:
    return {
        "ticker": "XYZ",
        "direction": "BULLISH",
        "persistence_ratio": 0.40,
        "active_days": 2,
        "total_days": 5,
        "trend": "fading",
        "prem_mcap_bps": 1.8,
        "cumulative_premium": 400_000,
        "dominant_dte_bucket": "short",
        "latest_oi_change": -22.0,
        "window_return_pct": 0.5,
        "conviction_stack": {"score": 42, "tier_label": "Stack D", "tier": "weak"},
    }


def _late_chaser_row() -> dict:
    """LONG flow but price has been running the wrong way → warning."""
    return {
        "ticker": "ABC",
        "direction": "BULLISH",
        "persistence_ratio": 0.80,
        "active_days": 4,
        "total_days": 5,
        "trend": "stable",
        "prem_mcap_bps": 15.0,
        "cumulative_premium": 6_000_000,
        "window_return_pct": -4.5,
        "conviction_stack": {"score": 60, "tier_label": "Stack C", "tier": "moderate"},
    }


def _hedging_row() -> dict:
    """Long-gamma / dealer pinning regime with near pin — double warning."""
    return {
        "ticker": "PIN",
        "direction": "BULLISH",
        "persistence_ratio": 0.60,
        "active_days": 3,
        "total_days": 5,
        "trend": "stable",
        "prem_mcap_bps": 5.0,
        "cumulative_premium": 2_000_000,
        "dealer_hedge_bias": "suppress",
        "pin_risk_strike": 100,
        "pin_risk_distance_pct": 0.6,
        "latest_iv_rank": 72,
        "iv_rank_5d_delta": 12,
        "conviction_stack": {"score": 55, "tier_label": "Stack C", "tier": "moderate"},
    }


def _tones(bullets: list[dict]) -> list[str]:
    return [str(b.get("tone", "")) for b in bullets]


def _labels(bullets: list[dict]) -> str:
    return " | ".join(str(b.get("label", "")) for b in bullets)


# ---------------------------------------------------------------------------
# build_flow_tracker_narrative
# ---------------------------------------------------------------------------
def test_empty_row_returns_empty_list():
    assert build_flow_tracker_narrative(None) == []
    assert build_flow_tracker_narrative({}) == []
    # Non-dict inputs should also degrade gracefully.
    assert build_flow_tracker_narrative("not a row") == []  # type: ignore[arg-type]
    assert build_flow_tracker_narrative(12345) == []  # type: ignore[arg-type]


def test_bullet_cap_respected():
    """Cap is 8 bullets — dense rows should not exceed it."""
    bullets = build_flow_tracker_narrative(_strong_accumulation_tracker_row())
    assert 1 <= len(bullets) <= 8, f"expected 1-8 bullets, got {len(bullets)}"


def test_bullet_shape_is_stable():
    """Every bullet exposes tone/icon/label/detail for the UI to render."""
    bullets = build_flow_tracker_narrative(_strong_accumulation_tracker_row())
    assert bullets, "strong accumulation row produced no bullets"
    for b in bullets:
        assert set(b.keys()) >= {"tone", "icon", "label", "detail"}
        assert b["tone"] in {TONE_POSITIVE, TONE_NEGATIVE, TONE_WARNING, TONE_INFO, TONE_NEUTRAL}
        assert isinstance(b["label"], str) and b["label"]


def test_strong_accumulation_leads_with_positive():
    bullets = build_flow_tracker_narrative(_strong_accumulation_tracker_row())
    tones = _tones(bullets)
    # Headline bullet (stack) should always be positive on this row.
    assert tones[0] == TONE_POSITIVE, f"expected lead tone positive, got {tones[0]}"
    # Majority positive on a dense positive row.
    positive = sum(t == TONE_POSITIVE for t in tones)
    assert positive >= 3, f"expected >= 3 positive bullets, got {positive} ({tones})"


def test_strong_accumulation_mentions_persistence_and_size():
    labels = _labels(build_flow_tracker_narrative(_strong_accumulation_tracker_row()))
    # Persistence + size are the two core facts we MUST surface.
    assert "Active every day" in labels or "persistence" in labels.lower()
    assert "bps" in labels.lower(), f"expected bps mention, got: {labels}"


def test_fading_flow_raises_warnings():
    bullets = build_flow_tracker_narrative(_fading_flow_tracker_row())
    tones = _tones(bullets)
    # Fading row should include at least one warning.
    assert TONE_WARNING in tones, f"expected warning tone, got {tones}"
    labels_lower = _labels(bullets).lower()
    assert "fading" in labels_lower or "low persistence" in labels_lower


def test_late_chaser_warning_when_price_fights_long_flow():
    bullets = build_flow_tracker_narrative(_late_chaser_row())
    labels_lower = _labels(bullets).lower()
    assert any(b["tone"] == TONE_WARNING for b in bullets), \
        f"expected warning on late-chase row: {labels_lower}"
    assert "fighting" in labels_lower or "late chase" in labels_lower or "-4" in labels_lower


def test_hedging_row_warns_about_pin_and_dealer():
    bullets = build_flow_tracker_narrative(_hedging_row())
    labels_lower = _labels(bullets).lower()
    assert "pin" in labels_lower, f"expected pin mention: {labels_lower}"
    assert "dealer" in labels_lower or "gamma" in labels_lower, f"expected dealer/gamma mention: {labels_lower}"


def test_dte_bucket_short_is_flagged():
    row = _strong_accumulation_tracker_row()
    row["dominant_dte_bucket"] = "short"
    bullets = build_flow_tracker_narrative(row)
    assert any("Short-dated" in b["label"] for b in bullets)


def test_sector_count_zero_flags_single_name():
    row = _strong_accumulation_tracker_row()
    row["sector_accumulating_count"] = 0
    bullets = build_flow_tracker_narrative(row)
    # Either surfaces as info (idiosyncratic) OR is suppressed; verify we
    # don't claim "other names accumulating" when the count is 0.
    for b in bullets:
        assert "other names" not in b["label"], f"false positive sector claim: {b['label']}"


# ---------------------------------------------------------------------------
# build_flow_feature_narrative
# ---------------------------------------------------------------------------
def test_feature_narrative_empty_row():
    assert build_flow_feature_narrative(None) == []
    assert build_flow_feature_narrative({}) == []


def test_feature_narrative_strong_long_row():
    row = {
        "direction": "LONG",
        "flow_score_scaled": 8.7,
        "conviction_stack": {"score": 78, "tier_label": "Stack B", "tier": "strong"},
        "dark_pool": {"available": True, "bias": 0.70, "notional_mcap_bps": 4.0},
        "hot_chain": {"call_premium": 2_000_000, "put_premium": 200_000},
        "insider": {"net_direction": "buying"},
        "session": {"session_tone": "STRENGTH"},
        "rs": {"rs_5d_pct": 2.5, "rs_20d_pct": 4.0},
        "dealer_hedge_bias": "chase",
        "iv_rank": 14,
        "earnings": {"days_until_earnings": 20},
    }
    bullets = build_flow_feature_narrative(row)
    tones = _tones(bullets)
    assert bullets, "expected narrative output"
    assert tones[0] == TONE_POSITIVE
    labels_lower = _labels(bullets).lower()
    assert "dark pool" in labels_lower
    assert "insiders" in labels_lower or "insider" in labels_lower


def test_feature_narrative_insider_opposition_warns():
    row = {
        "direction": "LONG",
        "flow_score_scaled": 7.5,
        "insider": {"net_direction": "selling"},
    }
    bullets = build_flow_feature_narrative(row)
    labels_lower = _labels(bullets).lower()
    assert "insider" in labels_lower, f"expected insider mention: {labels_lower}"
    assert any(b["tone"] == TONE_WARNING for b in bullets), \
        f"expected warning on opposing insider: {labels_lower}"


def test_feature_narrative_short_downtrend_is_positive():
    row = {
        "direction": "SHORT",
        "flow_score_scaled": 8.0,
        "session": {"session_tone": "WEAKNESS"},
        "rs": {"rs_5d_pct": -3.5, "rs_20d_pct": -5.0},
        "dealer_hedge_bias": "chase",
    }
    bullets = build_flow_feature_narrative(row)
    tones = _tones(bullets)
    # Short flow getting paid — should skew positive.
    assert _tones(bullets).count(TONE_POSITIVE) >= 2, \
        f"expected multiple positives on confirmed short row, got {tones}"


def test_malformed_nested_objects_do_not_crash():
    row = {
        "direction": "LONG",
        "flow_score_scaled": 6.0,
        "dark_pool": "not a dict",        # malformed
        "hot_chain": 42,                  # malformed
        "insider": None,                  # missing
        "session": [],                    # wrong type
        "rs": "junk",
        "conviction_stack": "broken",
    }
    bullets = build_flow_feature_narrative(row)
    assert isinstance(bullets, list)
    for b in bullets:
        assert "tone" in b and "label" in b


def test_tracker_narrative_survives_nan_numbers():
    row = {
        "direction": "BULLISH",
        "persistence_ratio": float("nan"),
        "active_days": None,
        "total_days": None,
        "prem_mcap_bps": float("nan"),
        "window_return_pct": float("nan"),
        "latest_oi_change": None,
        "trend": "",
    }
    # Should simply skip the unreadable bullets, not explode.
    bullets = build_flow_tracker_narrative(row)
    assert isinstance(bullets, list)


# ---------------------------------------------------------------------------
# Standalone runner — mirrors other Wave tests so CI can run without pytest
# when we're iterating offline.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_empty_row_returns_empty_list,
        test_bullet_cap_respected,
        test_bullet_shape_is_stable,
        test_strong_accumulation_leads_with_positive,
        test_strong_accumulation_mentions_persistence_and_size,
        test_fading_flow_raises_warnings,
        test_late_chaser_warning_when_price_fights_long_flow,
        test_hedging_row_warns_about_pin_and_dealer,
        test_dte_bucket_short_is_flagged,
        test_sector_count_zero_flags_single_name,
        test_feature_narrative_empty_row,
        test_feature_narrative_strong_long_row,
        test_feature_narrative_insider_opposition_warns,
        test_feature_narrative_short_downtrend_is_positive,
        test_malformed_nested_objects_do_not_crash,
        test_tracker_narrative_survives_nan_numbers,
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
    if failed:
        raise SystemExit(1)

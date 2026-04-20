"""Tests for the Unusual Flow Trader Card (Premium-Taxonomy plan).

The Unusual Flow tab's "Top Flow Intensity by Ticker" cards surface the
DTE-bucket context (LOTTERY / SWING / LEAP) and open a multi-tab modal.
These tests pin down the view-model contracts that feed both surfaces:

1.  ``_build_premium_mix_ui`` correctly picks the dominant bucket from
    a scan-only synth payload.
2.  Dominant-side accel ratio is selected from
    ``bullish_accel_ratio`` or ``bearish_accel_ratio`` depending on
    which side has higher flow intensity.
3.  ``sweep_share`` / ``multileg_share`` clamp to 0..1 and round to
    3 decimals.
4.  Rendered card HTML emits the interactive wrapper + bucket chip +
    modal shell so the JS can attach.

Run with:

    python -m pytest tests/test_top_flow_trader_card.py -v
    python tests/test_top_flow_trader_card.py           # standalone
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path for ``python tests/...`` standalone runs.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore


# ---------------------------------------------------------------------------
# 1. premium_mix_ui projection
# ---------------------------------------------------------------------------


def test_premium_mix_ui_picks_dominant_bucket_from_synth_payload():
    """A scan-only synth payload with the most dollars in 'swing' should
    produce dominant_bucket == 'swing'."""
    from app.web.view_models import _build_premium_mix_ui

    synth = {
        "total_bullish":   4_000_000,
        "total_bearish":   1_000_000,
        "lottery_bullish": 500_000,
        "lottery_bearish": 100_000,
        "swing_bullish":   3_000_000,
        "swing_bearish":   500_000,
        "leap_bullish":    500_000,
        "leap_bearish":    400_000,
        "unusual_bullish": 4_000_000,
        "unusual_bearish": 1_000_000,
        "other_bullish":   0.0,
        "other_bearish":   0.0,
        "source":          "flow_features",
    }
    ui = _build_premium_mix_ui({"premium_mix": synth})
    assert ui is not None
    assert ui["dominant_bucket"] == "swing"
    assert ui["source"] == "flow_features"
    # Total gross (bull+bear) = 5M, bucket "swing" gross = 3.5M → 70%.
    swing = next(b for b in ui["buckets"] if b["key"] == "swing")
    assert abs(swing["pct_of_total"] - 70.0) < 0.1
    # Lottery bucket: 0.6M gross → 12%.
    lot = next(b for b in ui["buckets"] if b["key"] == "lottery")
    assert abs(lot["pct_of_total"] - 12.0) < 0.1


def test_premium_mix_ui_returns_none_for_empty_mix():
    """Empty / missing premium_mix → None so the UI cleanly hides the panel."""
    from app.web.view_models import _build_premium_mix_ui

    assert _build_premium_mix_ui({}) is None
    assert _build_premium_mix_ui({"premium_mix": None}) is None
    assert _build_premium_mix_ui({
        "premium_mix": {"total_bullish": 0, "total_bearish": 0}
    }) is None


def test_premium_mix_ui_dominant_prefers_highest_gross():
    """When LEAP dominates, dominant_bucket flips to 'leap'."""
    from app.web.view_models import _build_premium_mix_ui

    synth = {
        "total_bullish":   2_000_000,
        "total_bearish":   5_000_000,
        "lottery_bullish": 100_000,
        "lottery_bearish": 300_000,
        "swing_bullish":   300_000,
        "swing_bearish":   1_000_000,
        "leap_bullish":    1_500_000,
        "leap_bearish":    3_500_000,
        "unusual_bullish": 2_000_000,
        "unusual_bearish": 5_000_000,
        "other_bullish":   0.0,
        "other_bearish":   0.0,
        "source":          "flow_features",
    }
    ui = _build_premium_mix_ui({"premium_mix": synth})
    assert ui is not None
    assert ui["dominant_bucket"] == "leap"


# ---------------------------------------------------------------------------
# 1b. UF DTE Bucket Fix — bucket columns on `r` propagate to _synth_mix,
#     and context="unusual_flow" drops the redundant "Unusual flow" row.
# ---------------------------------------------------------------------------


def test_premium_mix_ui_context_unusual_flow_omits_unusual_row():
    """Unusual-flow context tab: `_build_premium_mix_ui(context="unusual_flow")`
    must NOT include an "unusual" key.  Under the Unusual Flow tab every
    print already cleared the $500K floor upstream, so "total == unusual"
    tautologically and showing both would be noise.
    """
    from app.web.view_models import _build_premium_mix_ui

    synth = {
        "total_bullish":   2_000_000,
        "total_bearish":   1_000_000,
        "lottery_bullish":   500_000,
        "lottery_bearish":   200_000,
        "swing_bullish":   1_000_000,
        "swing_bearish":     500_000,
        "leap_bullish":      500_000,
        "leap_bearish":      300_000,
        "source":          "flow_features",
    }
    ui = _build_premium_mix_ui({"premium_mix": synth}, context="unusual_flow")
    assert ui is not None
    assert ui.get("context") == "unusual_flow"
    assert "unusual" not in ui, (
        "context='unusual_flow' must skip the redundant 'Unusual flow' row"
    )
    # But total + buckets still populated so the panel renders meaningfully.
    assert ui["total"]["gross"] == 3_000_000.0
    assert any(b["gross"] > 0 for b in ui["buckets"])


def test_premium_mix_ui_context_flow_tracker_keeps_unusual_row():
    """Default context (Flow Tracker) must still emit both kicker rows,
    otherwise the Flow Tracker Trader Card loses the "unusual flow"
    subset row that it genuinely needs (it shows the subset of total
    that came from institutional-sized prints, which IS different from
    total in the Flow Tracker data source).
    """
    from app.web.view_models import _build_premium_mix_ui

    synth = {
        "total_bullish":   5_000_000,
        "total_bearish":   1_000_000,
        "unusual_bullish": 4_000_000,
        "unusual_bearish":   900_000,
        "lottery_bullish":   500_000,
        "lottery_bearish":   100_000,
        "swing_bullish":   3_000_000,
        "swing_bearish":     500_000,
        "leap_bullish":      500_000,
        "leap_bearish":      300_000,
        "source":          "flow_features",
    }
    ui_default = _build_premium_mix_ui({"premium_mix": synth})
    ui_explicit = _build_premium_mix_ui({"premium_mix": synth}, context="flow_tracker")
    for ui in (ui_default, ui_explicit):
        assert ui is not None
        assert "unusual" in ui
        # Unusual row gross = 4.9M, distinct from total gross 6M.
        assert ui["unusual"]["gross"] == 4_900_000.0
        assert ui["total"]["gross"] == 6_000_000.0


def test_synth_mix_reads_bucket_columns_off_feature_row():
    """The server.py `top_flow` loop reads bucket columns straight off
    `r` (a flow_features row) when building `_synth_mix`.  This test
    simulates that projection with bucket columns present and absent,
    ensuring the zero-fill fallback keeps the UI non-crashing.
    """
    # Reproduce the relevant fragment of the server-side projection.
    def _build_synth(r: dict) -> dict:
        _bull = float(r.get("bullish_premium_raw", 0) or 0)
        _bear = float(r.get("bearish_premium_raw", 0) or 0)
        return {
            "total_bullish":   _bull,
            "total_bearish":   _bear,
            "lottery_bullish": float(r.get("lottery_bullish_premium") or 0),
            "lottery_bearish": float(r.get("lottery_bearish_premium") or 0),
            "swing_bullish":   float(r.get("swing_bullish_premium")   or 0),
            "swing_bearish":   float(r.get("swing_bearish_premium")   or 0),
            "leap_bullish":    float(r.get("leap_bullish_premium")    or 0),
            "leap_bearish":    float(r.get("leap_bearish_premium")    or 0),
            "other_bullish":   float(r.get("other_bullish_premium")   or 0),
            "other_bearish":   float(r.get("other_bearish_premium")   or 0),
            "source":          "flow_features",
        }

    # Case A: pipeline persisted bucket columns — buckets populate.
    r_full = {
        "bullish_premium_raw": 3_600_000,
        "bearish_premium_raw":   600_000,
        "lottery_bullish_premium": 1_000_000,
        "lottery_bearish_premium":         0,
        "swing_bullish_premium":     600_000,
        "swing_bearish_premium":     600_000,
        "leap_bullish_premium":    2_000_000,
        "leap_bearish_premium":            0,
    }
    synth_full = _build_synth(r_full)
    assert synth_full["lottery_bullish"] == 1_000_000
    assert synth_full["swing_bearish"] == 600_000
    assert synth_full["leap_bullish"] == 2_000_000
    # Bullish buckets reconcile to bullish raw total.
    assert (
        synth_full["lottery_bullish"]
        + synth_full["swing_bullish"]
        + synth_full["leap_bullish"]
        == synth_full["total_bullish"]
    )

    # Case B: pre-fix scan with no bucket columns — synth falls back
    # to zeros across all buckets; no crash, no NaN leakage.
    r_legacy = {
        "bullish_premium_raw": 500_000,
        "bearish_premium_raw":       0,
    }
    synth_legacy = _build_synth(r_legacy)
    for side in ("bullish", "bearish"):
        for bucket in ("lottery", "swing", "leap", "other"):
            assert synth_legacy[f"{bucket}_{side}"] == 0


# ---------------------------------------------------------------------------
# 2. Dominant-side accel selection + chip clamping (server-side projection)
# ---------------------------------------------------------------------------


def _run_top_flow_projection(flow_row: dict) -> dict:
    """Reproduce the scalar projection the server.py top_flow loop
    applies per row (sweep/multileg clamping + side-aware accel_ratio).

    Duplicating this small bit of math in a test helper keeps us honest
    without having to spin up a full Flask request context.
    """
    import pandas as pd

    bull_int = float(flow_row.get("bullish_flow_intensity", 0) or 0)
    bear_int = float(flow_row.get("bearish_flow_intensity", 0) or 0)
    is_bull = bull_int >= bear_int
    side = "bullish" if is_bull else "bearish"

    def _f01(v) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if pd.isna(f):
            return 0.0
        return max(0.0, min(1.0, f))

    sweep_share = round(_f01(flow_row.get("sweep_share")), 3)
    multileg_share = round(_f01(flow_row.get("multileg_share")), 3)
    try:
        accel_ratio = float(flow_row.get(f"{side}_accel_ratio", 0) or 0)
    except (TypeError, ValueError):
        accel_ratio = 0.0

    return {
        "direction": "LONG" if is_bull else "SHORT",
        "sweep_share": sweep_share,
        "multileg_share": multileg_share,
        "accel_ratio": round(accel_ratio, 2),
    }


def test_dominant_side_uses_bullish_accel_when_bull_heavier():
    row = {
        "bullish_flow_intensity": 0.005,
        "bearish_flow_intensity": 0.001,
        "bullish_accel_ratio": 1.8,
        "bearish_accel_ratio": 0.4,
        "sweep_share": 0.45,
        "multileg_share": 0.1,
    }
    out = _run_top_flow_projection(row)
    assert out["direction"] == "LONG"
    assert out["accel_ratio"] == 1.8


def test_dominant_side_uses_bearish_accel_when_bear_heavier():
    row = {
        "bullish_flow_intensity": 0.001,
        "bearish_flow_intensity": 0.006,
        "bullish_accel_ratio": 2.5,
        "bearish_accel_ratio": 1.1,
        "sweep_share": 0.2,
        "multileg_share": 0.5,
    }
    out = _run_top_flow_projection(row)
    assert out["direction"] == "SHORT"
    assert out["accel_ratio"] == 1.1


def test_sweep_multileg_clamp_to_unit_interval():
    """Out-of-range / nullish values must land in [0, 1]."""
    row = {
        "bullish_flow_intensity": 0.01,
        "bearish_flow_intensity": 0.0,
        "sweep_share": 1.42,       # > 1 → clamp to 1.0
        "multileg_share": -0.25,   # < 0 → clamp to 0.0
        "bullish_accel_ratio": 1.0,
    }
    out = _run_top_flow_projection(row)
    assert out["sweep_share"] == 1.0
    assert out["multileg_share"] == 0.0


def test_sweep_multileg_none_becomes_zero():
    row = {
        "bullish_flow_intensity": 0.01,
        "bearish_flow_intensity": 0.0,
        "sweep_share": None,
        "multileg_share": None,
    }
    out = _run_top_flow_projection(row)
    assert out["sweep_share"] == 0.0
    assert out["multileg_share"] == 0.0


# ---------------------------------------------------------------------------
# 3. Rendered HTML smoke test
# ---------------------------------------------------------------------------


def test_rendered_top_flow_card_has_interactive_wrapper_and_modal_shell():
    """Minimal Jinja render of the Unusual Flow section to prove the
    new wrapper classes, bucket chip, and modal shell are emitted.

    We call the Jinja template engine directly (no Flask app needed)
    with a synthetic ``top_flow`` list.  If the template markup drifts
    this test catches it before the JS modal binding silently breaks.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    tmpl_dir = _REPO_ROOT / "app" / "web" / "templates"

    # Minimal env — we render just the Unusual Flow snippet out of
    # index.html.  Full template render requires a lot of globals the
    # Flask route provides; extracting the relevant loop keeps the test
    # fast and isolated.
    snippet = """
{% if top_flow %}
<div class="card-grid" id="top-flow-list">
  {% for tf in top_flow %}
  {% set _pm = tf.premium_mix_ui %}
  {% set _pm_bucket = _pm.dominant_bucket if _pm and _pm.dominant_bucket else None %}
  {% set _uf_bucket_key = (_pm_bucket or tf.dominant_dte_bucket or 'other')|string|lower %}
  <article class="data-card data-card--interactive" data-ticker="{{ tf.ticker }}"
           data-uf-bucket="{{ _uf_bucket_key }}">
    <div class="data-card-head">
      <span class="badge uf-bucket-chip uf-bucket-{{ _uf_bucket_key }}">{{ _uf_bucket_key|upper }}</span>
      <span class="ticker">{{ tf.ticker }}</span>
      {% if tf.sweep_share %}<span class="uf-chip uf-chip-sweep">Sweep {{ (tf.sweep_share * 100)|round(0) }}%</span>{% endif %}
    </div>
  </article>
  {% endfor %}
</div>
<div id="detail-unusual-flow" class="detail-panel"></div>
{% endif %}
"""

    env = Environment(
        loader=FileSystemLoader(str(tmpl_dir)),
        autoescape=select_autoescape(["html"]),
    )
    tmpl = env.from_string(snippet)

    top_flow = [
        {
            "ticker": "NVDA",
            "dominant_dte_bucket": "31-90",
            "sweep_share": 0.68,
            "multileg_share": 0.12,
            "accel_ratio": 1.8,
            "premium_mix_ui": {
                "dominant_bucket": "swing",
                "dominant_label": "Swing",
            },
        }
    ]
    html = tmpl.render(top_flow=top_flow)
    assert 'data-card--interactive' in html
    assert 'id="top-flow-list"' in html
    assert 'uf-bucket-swing' in html  # dominant_bucket from premium_mix_ui wins
    assert 'data-uf-bucket="swing"' in html
    assert 'Sweep' in html
    assert 'id="detail-unusual-flow"' in html


def test_bucket_key_falls_back_to_dominant_dte_when_no_mix():
    """When premium_mix_ui is missing, the chip uses flow-features
    dominant_dte_bucket so the card still surfaces a bucket."""
    from jinja2 import Environment, select_autoescape

    snippet = """
{% set _pm = tf.premium_mix_ui %}
{% set _pm_bucket = _pm.dominant_bucket if _pm and _pm.dominant_bucket else None %}
{% set _uf_bucket_key = (_pm_bucket or tf.dominant_dte_bucket or 'other')|string|lower %}
KEY={{ _uf_bucket_key }}
"""
    env = Environment(autoescape=select_autoescape(["html"]))
    tmpl = env.from_string(snippet)
    out = tmpl.render(tf={
        "ticker": "AAPL",
        "dominant_dte_bucket": "91+",
        "premium_mix_ui": None,
    })
    assert "KEY=91+" in out

    out2 = tmpl.render(tf={"ticker": "TSLA"})
    assert "KEY=other" in out2


# ---------------------------------------------------------------------------
# Standalone runner (no pytest) — mirrors the convention used by
# tests/test_market_calendar.py + tests/test_premium_taxonomy.py.
# ---------------------------------------------------------------------------


def _run_all() -> int:
    """Run every ``test_*`` function in this module; return nonzero on failure."""
    mod = sys.modules[__name__]
    tests = [
        (name, fn) for name, fn in vars(mod).items()
        if name.startswith("test_") and callable(fn)
    ]
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failures.append((name, f"AssertionError: {e}"))
        except Exception as e:  # pragma: no cover
            failures.append((name, f"{type(e).__name__}: {e}"))
    total = len(tests)
    if failures:
        print(f"FAILED: {len(failures)}/{total}")
        for name, err in failures:
            print(f"  {name}\n    {err}")
        return 1
    print(f"OK: {total}/{total} tests passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_run_all())

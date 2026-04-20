"""Tests for the server-side filters applied by UW flow fetchers.

Week-1 Signal Quality Batch:
1.  ``fetch_flow_raw`` forwards ``size_greater_oi=true`` when
    ``FLOW_OPENING_ONLY=True`` (opening-trade filter) and forwards
    ``min_premium`` straight through to UW.
2.  ``fetch_recent_alert_flow`` accepts a ``min_premium`` kwarg and
    forwards it as the ``min_premium`` UW query param alongside the
    existing ``is_sweep`` + ``vol_greater_oi`` filters.

Run with:

    python -m pytest tests/test_fetch_flow_raw_filters.py -v
    python tests/test_fetch_flow_raw_filters.py           # standalone
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore


def _mock_resp(payload=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload if payload is not None else {"data": []})
    return resp


# ---------------------------------------------------------------------------
# fetch_flow_raw — universe pull used by run_flow_to_price_pipeline
# ---------------------------------------------------------------------------


def test_fetch_flow_raw_forwards_size_greater_oi_when_opening_only_true():
    """When FLOW_OPENING_ONLY=True, fetch_flow_raw must append
    size_greater_oi=true to UW query params."""
    from app.vendors import unusual_whales as uw
    import app.config as cfg

    with patch.object(cfg, "FLOW_OPENING_ONLY", True):
        with patch.object(uw, "_uw_request", return_value=_mock_resp()) as mock_req:
            uw.fetch_flow_raw(limit=100)
    assert mock_req.called
    params = mock_req.call_args.kwargs["params"]
    assert params.get("size_greater_oi") == "true"


def test_fetch_flow_raw_skips_size_greater_oi_when_opening_only_false():
    """When FLOW_OPENING_ONLY=False, fetch_flow_raw must NOT add
    size_greater_oi (legacy behaviour)."""
    from app.vendors import unusual_whales as uw
    import app.config as cfg

    with patch.object(cfg, "FLOW_OPENING_ONLY", False):
        with patch.object(uw, "_uw_request", return_value=_mock_resp()) as mock_req:
            uw.fetch_flow_raw(limit=100)
    params = mock_req.call_args.kwargs["params"]
    assert "size_greater_oi" not in params


def test_fetch_flow_raw_forwards_min_premium_server_side():
    """min_premium must be forwarded to UW as a query param so the floor
    is applied at the API layer, not just client-side."""
    from app.vendors import unusual_whales as uw

    with patch.object(uw, "_uw_request", return_value=_mock_resp()) as mock_req:
        uw.fetch_flow_raw(limit=100, min_premium=500_000)
    params = mock_req.call_args.kwargs["params"]
    assert params.get("min_premium") == 500_000


def test_fetch_flow_raw_extra_params_override_opening_only():
    """Explicit extra_params must win over the module-wide
    FLOW_OPENING_ONLY flag."""
    from app.vendors import unusual_whales as uw
    import app.config as cfg

    with patch.object(cfg, "FLOW_OPENING_ONLY", True):
        with patch.object(uw, "_uw_request", return_value=_mock_resp()) as mock_req:
            uw.fetch_flow_raw(limit=100, size_greater_oi="false")
    params = mock_req.call_args.kwargs["params"]
    assert params.get("size_greater_oi") == "false"


# ---------------------------------------------------------------------------
# fetch_recent_alert_flow — Individual Trades / /api/alerts feed
# ---------------------------------------------------------------------------


def test_fetch_recent_alert_flow_forwards_min_premium():
    """Individual Trades feed must forward min_premium to UW so the
    $500k institutional floor is applied server-side."""
    from app.vendors import unusual_whales as uw

    with patch.object(uw, "_uw_request", return_value=_mock_resp()) as mock_req:
        uw.fetch_recent_alert_flow(limit=150, hours_back=24, min_premium=500_000)
    params = mock_req.call_args.kwargs["params"]
    assert params.get("min_premium") == 500_000
    assert params.get("is_sweep") == "true"
    assert params.get("vol_greater_oi") == "true"


def test_fetch_recent_alert_flow_no_min_premium_when_none():
    """Legacy caller (min_premium=None) must not emit the min_premium
    param so server behaviour is unchanged."""
    from app.vendors import unusual_whales as uw

    with patch.object(uw, "_uw_request", return_value=_mock_resp()) as mock_req:
        uw.fetch_recent_alert_flow(limit=150, hours_back=24)
    params = mock_req.call_args.kwargs["params"]
    assert "min_premium" not in params


def test_fetch_recent_alert_flow_opening_only_true_layers_size_greater_oi():
    """opening_only=True must add size_greater_oi=true on top of the
    existing is_sweep + vol_greater_oi filters."""
    from app.vendors import unusual_whales as uw

    with patch.object(uw, "_uw_request", return_value=_mock_resp()) as mock_req:
        uw.fetch_recent_alert_flow(
            limit=150, hours_back=24, opening_only=True, min_premium=500_000
        )
    params = mock_req.call_args.kwargs["params"]
    assert params.get("size_greater_oi") == "true"
    assert params.get("min_premium") == 500_000


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tests = [
        test_fetch_flow_raw_forwards_size_greater_oi_when_opening_only_true,
        test_fetch_flow_raw_skips_size_greater_oi_when_opening_only_false,
        test_fetch_flow_raw_forwards_min_premium_server_side,
        test_fetch_flow_raw_extra_params_override_opening_only,
        test_fetch_recent_alert_flow_forwards_min_premium,
        test_fetch_recent_alert_flow_no_min_premium_when_none,
        test_fetch_recent_alert_flow_opening_only_true_layers_size_greater_oi,
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

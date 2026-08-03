"""Per-feature grade attribution.

Consumes the matured ``grade_history.csv`` (rows with
``forward_excess_return`` back-filled) and computes **Spearman rank
correlation** between each flow feature and the signed forward excess
return.  Writes ``data/grade_attribution.json``.

Why Spearman?
-------------
Our features are heavy-tailed (premium $, bps-of-mcap) and many are
effectively discrete (dominant_dte_bucket, persistence_ratio over a
small window), so Pearson assumptions break down.  Spearman handles
both by ranking first.

scipy is optional — we fall back to a stdlib-only rank-correlation
implementation so the hourly scan never pulls scipy just for this.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ATTRIBUTION_PATH = DATA_DIR / "grade_attribution.json"
# Bar-by-bar replay panel. Preferred target source: its ``replay_realized_r``
# is directionally-signed and is refreshed every weekly backtest, whereas the
# legacy ``forward_excess_return`` path (5d close-to-close vs SPY) depends on a
# live OHLCV fetch inside the hourly scan and silently stopped populating
# recent rows on 2026-05-22 — which is why grade_attribution.json went empty
# (0 matured rows inside the 60d window) despite 900+ graded rows on disk.
REPLAY_PATH = DATA_DIR / "grade_history_with_replay.csv"

# Features evaluated by attribution.  Categorical columns live in
# CATEGORICAL_FEATURES and get one-hot-into-indicator treatment instead
# of a numeric correlation.
NUMERIC_FEATURES = [
    "conviction_score",
    "conviction_stack",
    "flow_intensity",
    "persistence_ratio",
    "accumulation_score",
    "sweep_share",
    "multileg_share",
    "accel_ratio_today",
    "window_return_pct",
    "cumulative_premium",
    "prem_mcap_bps",
    "latest_put_call_ratio",
    "latest_iv_rank",
    "latest_oi_change",
    "perc_3_day_total_latest",
    "perc_30_day_total_latest",
]

CATEGORICAL_FEATURES = [
    "dominant_dte_bucket",
    "premium_source",
    "conviction_grade",
]

DEFAULT_WINDOW_DAYS = 60
DEFAULT_MIN_SAMPLES = 30


def _to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _rank(values: list[float]) -> list[float]:
    """Fractional ranks (average for ties) — same as scipy's 'average'."""
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _spearman(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    """Return (rho, two-sided p-value).  p is a normal-approximation
    (sufficient for n >= 30).  Uses scipy when available for exact p."""
    if len(x) < 3:
        return None, None
    try:
        from scipy.stats import spearmanr  # type: ignore

        res = spearmanr(x, y)
        rho = float(getattr(res, "correlation", None) or res[0])
        pval = float(getattr(res, "pvalue", None) or res[1])
        if math.isnan(rho):
            return None, None
        return rho, pval
    except Exception:
        pass

    # Stdlib fallback.
    rx = _rank(x)
    ry = _rank(y)
    rho = _pearson(rx, ry)
    if rho is None:
        return None, None

    # Normal-approx p-value.  z = rho * sqrt(n-1), two-sided.
    n = len(x)
    z = rho * math.sqrt(max(n - 1, 1))
    # Standard normal survival function — simple erf-based approx.
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return rho, p


def _filter_window(rows: list[dict[str, Any]], window_days: int) -> list[dict[str, Any]]:
    if window_days <= 0:
        return rows
    today = date.today()
    cutoff = (today - timedelta(days=window_days)).isoformat()
    out: list[dict[str, Any]] = []
    for r in rows:
        as_of = str(r.get("as_of") or "")
        if as_of >= cutoff:
            out.append(r)
    return out


def _signed_forward(r: dict[str, Any]) -> float | None:
    """Forward excess return signed by predicted direction.

    Aligns with how the backtest converts to R-multiples — a BEARISH
    grade that loses 2% against SPY scores positively (the short
    worked).
    """
    raw = _to_float(r.get("forward_excess_return"))
    if raw is None:
        return None
    direction = (r.get("direction") or "BULLISH").strip().upper()
    sign = -1.0 if direction == "BEARISH" else 1.0
    return sign * raw


def _replay_realized_r(r: dict[str, Any]) -> float | None:
    """Target from the replay panel — already directionally-signed R.

    ``replay_realized_r`` is the bar-by-bar trade R in the trade's own
    direction (a bearish trade that profits is positive), so unlike
    ``forward_excess_return`` it does *not* need re-signing.
    """
    return _to_float(r.get("replay_realized_r"))


def _attrib_numeric(
    feature: str,
    rows: list[dict[str, Any]],
    y_of: "Any" = _signed_forward,
) -> dict[str, Any]:
    pairs = []
    for r in rows:
        y = y_of(r)
        x = _to_float(r.get(feature))
        if y is None or x is None:
            continue
        pairs.append((x, y))
    if not pairs:
        return {"n": 0, "rho": None, "p": None}
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    rho, p = _spearman(x_vals, y_vals)
    return {
        "n": len(pairs),
        "rho": round(rho, 4) if rho is not None else None,
        "p": round(p, 4) if p is not None else None,
    }


def _attrib_categorical(
    feature: str,
    rows: list[dict[str, Any]],
    y_of: "Any" = _signed_forward,
) -> dict[str, Any]:
    """Per-level mean signed forward return (robust for small cohorts)."""
    buckets: dict[str, list[float]] = {}
    for r in rows:
        y = y_of(r)
        if y is None:
            continue
        key = str(r.get(feature) or "").strip() or "—"
        buckets.setdefault(key, []).append(y)
    levels = {}
    for k, vs in buckets.items():
        if not vs:
            continue
        levels[k] = {
            "n": len(vs),
            "mean_signed_excess_return_pct": round(sum(vs) / len(vs) * 100, 3),
            "hit_rate": round(sum(1 for v in vs if v > 0) / len(vs), 3),
        }
    return {"levels": levels, "n": sum(lv["n"] for lv in levels.values())}


_REAL_EXITS = ("T2", "T1_then_stop", "stop", "time_stop")


def _row_is_matured(r: dict[str, Any]) -> bool:
    """Matured = a real exit fired, or held the full max-hold horizon.

    Excludes ``no_exit_yet`` rows whose OHLCV ran out early (truncated
    mark-to-market R, usually 0-3 days held) so they can't bias attribution.
    Prefers the explicit ``replay_is_matured`` flag, falls back to deriving
    it from exit reason + days_held for pre-flag CSVs.
    """
    flag = str(r.get("replay_is_matured") or "").strip().lower()
    if flag in ("true", "1"):
        return True
    if flag in ("false", "0"):
        return False
    exit_reason = str(r.get("replay_exit_reason") or "").strip()
    days_held = _to_float(r.get("replay_days_held")) or 0.0
    if days_held <= 0:
        return False
    if exit_reason in _REAL_EXITS:
        return True
    max_hold = _to_float(r.get("replay_max_hold_days_used"))
    return exit_reason == "no_exit_yet" and max_hold is not None and days_held >= max_hold


def _load_replay_rows() -> list[dict[str, Any]]:
    """Matured replay-panel rows that carry a realized R.

    Immature ``no_exit_yet`` rows (OHLCV ran out before an exit) are dropped
    — their truncated R would otherwise pollute the correlation, and because
    they are the most recent signals they would dominate any recent window.
    """
    if not REPLAY_PATH.exists():
        return []
    try:
        import csv

        with open(REPLAY_PATH, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    return [
        r for r in rows
        if _to_float(r.get("replay_realized_r")) is not None and _row_is_matured(r)
    ]


def _persisted_report_has_rows() -> bool:
    """True when the on-disk report already holds a populated result."""
    if not ATTRIBUTION_PATH.exists():
        return False
    try:
        with open(ATTRIBUTION_PATH, "r") as f:
            return int(json.load(f).get("n_rows") or 0) > 0
    except Exception:
        return False


def refresh_attribution(
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    write: bool = True,
    source: str = "auto",
) -> dict[str, Any]:
    """Compute + (optionally) persist the attribution report.

    ``source`` selects the target metric:
      - ``"replay"`` — Spearman vs the bar-by-bar ``replay_realized_r``
        (preferred; already directional; refreshed every backtest).
      - ``"forward"`` — the legacy 5d ``forward_excess_return`` path.
      - ``"auto"`` (default) — use the replay panel when it has at least
        ``min_samples`` realized rows, else fall back to ``forward``.

    The replay source is *not* window-filtered — the replay panel only
    contains closed/replayed trades already, so every row is matured and a
    trailing-window cut would just throw away signal.

    Returns a dict with ``status`` ``"insufficient_history"`` when neither
    source clears ``min_samples``.
    """
    from app.analytics.grade_history import load_history

    y_of = _signed_forward
    used_source = "forward_excess_return"
    rows: list[dict[str, Any]] = []

    if source in ("auto", "replay"):
        replay_rows = _load_replay_rows()
        if len(replay_rows) >= min_samples or source == "replay":
            rows = replay_rows
            y_of = _replay_realized_r
            used_source = "replay_realized_r"

    if used_source == "forward_excess_return":
        rows = load_history(with_returns_only=True)
        rows = _filter_window(rows, window_days)

    if len(rows) < min_samples:
        result = {
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "window_days": window_days,
            "min_samples": min_samples,
            "n_rows": len(rows),
            "target": used_source,
            "status": "insufficient_history",
            "numeric": {},
            "categorical": {},
        }
        # The hourly scan calls this without a replay panel — REPLAY_PATH is
        # produced (and gitignored) by the weekly backtest job only — so it
        # always lands on the empty legacy branch and would clobber the
        # backtest's populated report. An empty result never replaces a
        # populated one on disk.
        if write and not _persisted_report_has_rows():
            ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ATTRIBUTION_PATH, "w") as f:
                json.dump(result, f, indent=2)
        return result

    numeric: dict[str, dict[str, Any]] = {}
    for feat in NUMERIC_FEATURES:
        numeric[feat] = _attrib_numeric(feat, rows, y_of)

    categorical: dict[str, dict[str, Any]] = {}
    for feat in CATEGORICAL_FEATURES:
        categorical[feat] = _attrib_categorical(feat, rows, y_of)

    # Rank numeric features by |rho| so the UI can show the top
    # predictors without client-side sorting.
    ranked = sorted(
        [
            {"feature": k, **v}
            for k, v in numeric.items()
            if v.get("rho") is not None
        ],
        key=lambda d: abs(d["rho"]),
        reverse=True,
    )

    result = {
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "window_days": window_days,
        "min_samples": min_samples,
        "n_rows": len(rows),
        "target": used_source,
        "status": "ok",
        "numeric": numeric,
        "categorical": categorical,
        "ranked_numeric": ranked,
    }

    if write:
        ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ATTRIBUTION_PATH, "w") as f:
            json.dump(result, f, indent=2)

    return result


def load_attribution() -> dict[str, Any] | None:
    if not ATTRIBUTION_PATH.exists():
        return None
    try:
        with open(ATTRIBUTION_PATH) as f:
            return json.load(f)
    except Exception:
        return None

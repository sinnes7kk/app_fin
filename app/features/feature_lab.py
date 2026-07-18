"""Feature lab — shadow log of candidate flow/options features.

Why this module exists
----------------------
The recalibration run on 2026-05-06 showed that the six existing
``conviction_score`` proxies have OOS Spearman barely distinguishable
from random.  Re-weighting features that are individually weak will
not produce a strong score, so we need a *wider* feature search.

This module is the data-collection step of that search:

  - Computes a battery of research-backed candidate features per
    ``(as_of, ticker, direction)`` row, drawn from the same scan we
    already run.
  - Persists them to ``data/feature_lab.csv`` — a separate file from
    ``grade_history.csv`` so we can churn the schema freely without
    breaking the existing recalibration pipeline.
  - Has zero impact on live scoring or promotion.

After 4-6 weeks of accumulated data, ``scripts/feature_lab_report.py``
runs a Spearman ranking of every candidate feature against the bar-by-
bar replay ``realized_r``, surfacing which features actually predict
forward outcomes.

Schema
------
- IDs: ``as_of``, ``ticker``, ``direction``, ``conviction_grade``,
  ``conviction_score``, ``sector``.
- Free features (computable from data the scan already has): see
  ``FREE_FEATURE_COLS``.
- UW endpoint features (gated to top-N by conviction_score): see
  ``UW_FEATURE_COLS``.

Tickers below the top-N cutoff get NaN UW columns but full free-feature
columns. This keeps the row count consistent across hourly scans.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FEATURE_LAB_PATH = DATA_DIR / "feature_lab.csv"
RAW_FLOW_DIR = DATA_DIR / "raw_flow"
GRADE_HISTORY_PATH = DATA_DIR / "grade_history.csv"
# Screener premium history feeds ``prem_momentum_z3d``.  The live rolling
# file covers ~15 trading days (enough for a 3-day trailing window); the
# gzip archive holds the full multi-day history and is only loaded for
# backfills.  Sourcing momentum from these (2000+ tickers) instead of
# ``grade_history.csv`` (~300 graded tickers) is what lets the feature
# populate for the transient names that never accumulate 3 graded days.
SCREENER_SNAPSHOTS_PATH = DATA_DIR / "screener_snapshots.csv"
SNAPSHOTS_ARCHIVE_PATH = DATA_DIR / "snapshots_archive.csv.gz"

# Top-N candidates by conviction_score that get the expensive UW endpoint
# fetches.  At ~10 hourly scans/day this caps the extra UW load at ~1800
# calls/day across 6 endpoints. Configurable via the env var
# ``FEATURE_LAB_TOPN`` for testing.
FEATURE_LAB_TOPN_DEFAULT: int = 30

# A "far OTM" option is one whose strike is more than this fraction away
# from spot.  Chosen so that for a 30-DTE option, a far-OTM strike is
# roughly the 3-delta region for typical IV regimes.
FAR_OTM_THRESHOLD: float = 0.10

# --- Schema ------------------------------------------------------------

ID_COLS = (
    "as_of",
    "ticker",
    "direction",
    "conviction_grade",
    "conviction_score",
    "sector",
)

FREE_FEATURE_COLS = (
    "bullish_premium_share",
    "unusual_premium_share",
    "vrp_proxy",
    "far_otm_call_share",
    "far_otm_put_share",
    "dollar_delta_weighted_flow",
    "sector_relative_pct",
    "prem_momentum_z3d",
    "realized_vol_regime",
)

# Aggressor-signed premium features (Spec 1). Computed from the per-ticker
# raw_flow slice, which already classifies every print into LONG/SHORT via
# aggressor side (CALL@ASK / PUT@BID = bullish; PUT@ASK / CALL@BID = bearish;
# see app/vendors/unusual_whales.py _infer_direction). These distinguish
# genuine directional aggression from the crude call/put split that feeds
# bullish_premium_share. Shadow-logged; feed the momentum score once proven.
AGGRESSOR_FEATURE_COLS = (
    "aggressor_bull_share",     # LONG premium / (LONG + SHORT) premium, 0..1
    "aggressor_net_prem_bps",   # signed net bullish aggression / marketcap, bps
    "ask_side_ratio",           # buyer-initiated premium share (ASK / (ASK+BID))
    "directional_sweep_share",  # signed sweep pressure / total premium
)

UW_FEATURE_COLS = (
    "gex_total",
    "vanna_total",
    "charm_total",
    "iv_skew_25d",
    "atm_iv_30d",
    "atm_iv_60d",
    "atm_iv_90d",
    "term_slope_30_90",
    "expiry_concentration_top1",
    "max_pain_dist_pct",
    "dealer_net_delta_at_spot",
    "dealer_net_gamma_at_spot",
)

# Shadow momentum score (Spec 2): cross-sectional composite of the
# validated free features + aggressor family. Never affects live grades
# until it clears the promotion gate; see app/features/momentum_score.py.
MOMENTUM_COLS = (
    "momentum_composite",  # mean cross-sectional percentile, 0..1
    "momentum_score",      # momentum_composite * 100, 0..100
)

LAB_COLS = (
    ID_COLS
    + FREE_FEATURE_COLS
    + AGGRESSOR_FEATURE_COLS
    + UW_FEATURE_COLS
    + MOMENTUM_COLS
)


# --- Free feature helpers ---------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    n, d = _safe_float(num), _safe_float(den)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _bullish_premium_share(g: dict) -> float | None:
    """Share of total directional premium that is bullish.

    ``g`` is a row from ``compute_multi_day_flow``. The cumulative
    bull/bear totals live at ``cumulative_bull`` / ``cumulative_bear``;
    the older flat key names (``bullish_premium``, etc.) only exist on
    raw screener payloads and were the cause of the 100% NULL bug
    surfaced 2026-05-09. We accept both shapes so the helper still
    works on legacy fixtures.
    """
    bull = (
        _safe_float(g.get("cumulative_bull"))
        or _safe_float(g.get("bullish_premium"))
        or _safe_float(g.get("total_bullish_premium"))
        or 0.0
    )
    bear = (
        _safe_float(g.get("cumulative_bear"))
        or _safe_float(g.get("bearish_premium"))
        or _safe_float(g.get("total_bearish_premium"))
        or 0.0
    )
    total = bull + bear
    if total <= 0:
        return None
    return bull / total


def _unusual_premium_share(g: dict) -> float | None:
    """Share of *unusual* premium that is bullish.

    Reads from the nested ``premium_mix`` dict that
    ``compute_multi_day_flow`` emits (``unusual_bullish`` /
    ``unusual_bearish``). Falls back to legacy flat keys for fixtures.
    """
    mix = g.get("premium_mix") or {}
    ub = (
        _safe_float(mix.get("unusual_bullish"))
        or _safe_float(g.get("unusual_bullish_premium"))
        or 0.0
    )
    ud = (
        _safe_float(mix.get("unusual_bearish"))
        or _safe_float(g.get("unusual_bearish_premium"))
        or 0.0
    )
    total = ub + ud
    if total <= 0:
        return None
    return ub / total


def _vrp_proxy(g: dict, ohlcv: pd.DataFrame | None) -> float | None:
    """Volatility risk premium proxy = IV rank − realized vol percentile.

    Uses ``latest_iv_rank`` (already in the grade dict, 0–100) and
    computes a 30-day realized vol percentile from the ticker's OHLCV
    cache window. Returns the difference in percentile points. A
    positive value means options are pricing more vol than recent
    realized — a bearish setup for vol buyers but often a bullish
    setup for the underlying when call premium dominates.
    """
    iv_rank = _safe_float(g.get("latest_iv_rank"))
    if iv_rank is None:
        return None
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return None
    closes = pd.to_numeric(ohlcv["close"], errors="coerce").dropna()
    if len(closes) < 30:
        return None
    rets = closes.pct_change().dropna()
    if len(rets) < 30:
        return None
    rv30 = rets.tail(30).std() * math.sqrt(252) * 100
    rv_history = rets.rolling(30).std() * math.sqrt(252) * 100
    rv_history = rv_history.dropna()
    if rv_history.empty:
        return None
    rv_pct = float((rv_history <= rv30).mean()) * 100
    return float(iv_rank) - rv_pct


def _far_otm_shares(rows: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return (far_otm_call_share, far_otm_put_share) for a ticker's
    raw flow rows. Each share is fraction of total premium where
    |strike − spot| / spot > FAR_OTM_THRESHOLD, split by call vs put.
    """
    if rows.empty:
        return None, None
    if not {"strike", "underlying_price", "premium", "option_type"}.issubset(rows.columns):
        return None, None
    df = rows.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    df = df.dropna(subset=["strike", "underlying_price", "premium"])
    if df.empty:
        return None, None
    df["moneyness"] = (df["strike"] - df["underlying_price"]).abs() / df["underlying_price"]
    df["is_far"] = df["moneyness"] > FAR_OTM_THRESHOLD
    total = df["premium"].sum()
    if total <= 0:
        return None, None
    far_calls = df.loc[df["is_far"] & (df["option_type"].str.upper() == "CALL"), "premium"].sum()
    far_puts = df.loc[df["is_far"] & (df["option_type"].str.upper() == "PUT"), "premium"].sum()
    return float(far_calls / total), float(far_puts / total)


def _dollar_delta_weighted_flow(rows: pd.DataFrame, direction: str) -> float | None:
    """Approximate Σ(premium × delta × direction_sign).

    No greek lookup — uses a moneyness-based delta proxy:
        call_delta ≈ 0.5 + clip(5 × (S − K) / S, −0.45, +0.45)
        put_delta  ≈ call_delta − 1
    This is intentionally rough; the goal is to amplify near-the-money
    flow over far-OTM lottery tickets in a way that does not require an
    extra UW round-trip per row.
    """
    if rows.empty:
        return None
    if not {"strike", "underlying_price", "premium", "option_type"}.issubset(rows.columns):
        return None
    df = rows.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    df = df.dropna(subset=["strike", "underlying_price", "premium"])
    if df.empty:
        return None
    moneyness = (df["underlying_price"] - df["strike"]) / df["underlying_price"]
    delta_proxy = 0.5 + (5 * moneyness).clip(-0.45, 0.45)
    is_put = df["option_type"].str.upper() == "PUT"
    delta_signed = np.where(is_put, delta_proxy - 1.0, delta_proxy)
    direction_sign = 1.0 if str(direction).upper() == "BULLISH" else -1.0
    weighted = (df["premium"] * delta_signed * direction_sign).sum()
    return float(weighted)


def _aggressor_signed_features(rows: pd.DataFrame) -> dict[str, float | None]:
    """Aggressor-signed premium features for a ticker's raw flow rows.

    Uses the per-print ``direction`` (LONG/SHORT from aggressor side) that
    ``normalize_flow_response`` already derives, plus ``ask_side_premium`` /
    ``bid_side_premium`` and the ``is_sweep`` / ``is_multileg`` flags.

    All four are direction-neutral in storage (positive = net bullish); the
    momentum score orients them to the trade's direction downstream.

    - ``aggressor_bull_share``: LONG premium / (LONG + SHORT) premium.
    - ``aggressor_net_prem_bps``: confidence-weighted (LONG - SHORT) premium
      / marketcap, in basis points.
    - ``ask_side_ratio``: buyer-initiated share = ASK / (ASK + BID) premium.
    - ``directional_sweep_share``: (LONG - SHORT) *sweep* premium / total
      premium; captures urgency-weighted directional pressure.
    """
    empty = {c: None for c in AGGRESSOR_FEATURE_COLS}
    if rows is None or rows.empty:
        return empty

    df = rows.copy()
    prem = pd.to_numeric(df.get("premium"), errors="coerce")

    # --- ask_side_ratio (direction-agnostic buyer urgency) -------------
    ask = pd.to_numeric(df.get("ask_side_premium"), errors="coerce").fillna(0.0)
    bid = pd.to_numeric(df.get("bid_side_premium"), errors="coerce").fillna(0.0)
    ask_bid_total = float(ask.sum() + bid.sum())
    ask_side_ratio = float(ask.sum() / ask_bid_total) if ask_bid_total > 0 else None

    out = dict(empty)
    out["ask_side_ratio"] = ask_side_ratio

    if "direction" not in df.columns or prem.isna().all():
        return out

    df["_prem"] = prem.fillna(0.0)
    df["_dir"] = df["direction"].astype(str).str.upper().str.strip()

    # Exclude multi-leg prints from directional attribution: a spread's
    # short leg misattributes direction. ask_side_ratio above keeps them.
    directional = df
    if "is_multileg" in df.columns:
        ml = df["is_multileg"].astype(str).str.lower().isin(["true", "1", "1.0"])
        directional = df[~ml]

    conf = pd.to_numeric(directional.get("direction_confidence"), errors="coerce").fillna(1.0)
    long_mask = directional["_dir"] == "LONG"
    short_mask = directional["_dir"] == "SHORT"

    long_prem = float(directional.loc[long_mask, "_prem"].sum())
    short_prem = float(directional.loc[short_mask, "_prem"].sum())
    dir_total = long_prem + short_prem
    if dir_total > 0:
        out["aggressor_bull_share"] = long_prem / dir_total

    # Confidence-weighted net, normalized by marketcap (bps).
    long_w = float((directional.loc[long_mask, "_prem"] * conf[long_mask]).sum())
    short_w = float((directional.loc[short_mask, "_prem"] * conf[short_mask]).sum())
    mcap = pd.to_numeric(df.get("marketcap"), errors="coerce").dropna()
    mcap_val = float(mcap.iloc[0]) if not mcap.empty else 0.0
    if mcap_val > 0:
        out["aggressor_net_prem_bps"] = (long_w - short_w) / mcap_val * 1e4

    # Directional sweep pressure.
    total_prem = float(df["_prem"].sum())
    if "is_sweep" in directional.columns and total_prem > 0:
        sw = directional["is_sweep"].astype(str).str.lower().isin(["true", "1", "1.0"])
        long_sw = float(directional.loc[sw & long_mask, "_prem"].sum())
        short_sw = float(directional.loc[sw & short_mask, "_prem"].sum())
        out["directional_sweep_share"] = (long_sw - short_sw) / total_prem

    return out


def _sector_relative_pct_lookup(grades: Iterable[dict]) -> dict[str, float]:
    """Build a per-ticker sector-relative percentile of prem_mcap_bps.

    Returns a {ticker: pct} dict where pct = (ticker_value − sector_p50)
    / (sector_p90 − sector_p10), clipped to [-2, 2]. NaN/missing
    inputs are skipped.
    """
    rows = []
    for g in grades:
        prem = _safe_float(g.get("prem_mcap_bps"))
        sector = str(g.get("sector") or "").strip()
        if prem is None or not sector:
            continue
        rows.append({"ticker": g.get("ticker"), "sector": sector, "prem_mcap_bps": prem})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for sector, group in df.groupby("sector"):
        if len(group) < 3:
            continue
        p10, p50, p90 = group["prem_mcap_bps"].quantile([0.10, 0.50, 0.90])
        spread = p90 - p10
        if spread <= 0:
            continue
        for _, r in group.iterrows():
            val = (r["prem_mcap_bps"] - p50) / spread
            out[r["ticker"]] = float(np.clip(val, -2.0, 2.0))
    return out


_PREM_MOMENTUM_WINDOW = 3


def _prem_momentum_z3d(
    ticker: str,
    as_of_day: object,
    premium_history_df: pd.DataFrame | None,
    window: int = _PREM_MOMENTUM_WINDOW,
) -> float | None:
    """Z-score of ``as_of_day``'s daily premium vs the trailing ``window`` days.

    ``premium_history_df`` is the tidy frame produced by
    ``load_screener_premium_history`` — one ``daily_premium`` value per
    ``(ticker, day)``.  Today's premium is scored against the mean/std of
    the ``window`` *prior* trading days (strictly before ``as_of_day``),
    so the statistic answers "is today's options premium unusually high
    versus this ticker's own recent baseline?".

    Sourcing the baseline from the screener snapshot universe (2000+
    tickers) rather than the graded-only ``grade_history`` (~300 tickers)
    is what lets this populate for the transient names that never
    accumulate three graded days.
    """
    if premium_history_df is None or premium_history_df.empty:
        return None
    if not {"ticker", "day", "daily_premium"}.issubset(premium_history_df.columns):
        return None

    as_of_ts = pd.to_datetime(as_of_day, errors="coerce")
    if pd.isna(as_of_ts):
        return None
    as_of_ts = as_of_ts.normalize()

    sub = premium_history_df[
        premium_history_df["ticker"].astype(str).str.upper() == ticker.upper()
    ]
    if sub.empty:
        return None
    sub = sub.dropna(subset=["daily_premium"]).sort_values("day")

    today = sub[sub["day"] == as_of_ts]
    if today.empty:
        return None
    today_premium = float(today["daily_premium"].iloc[-1])

    prior = sub[sub["day"] < as_of_ts].tail(window)
    if len(prior) < window:
        return None
    mu = prior["daily_premium"].mean()
    sigma = prior["daily_premium"].std()
    if sigma is None or sigma == 0 or math.isnan(sigma):
        return None
    return float((today_premium - mu) / sigma)


def _realized_vol_regime(ohlcv: pd.DataFrame | None) -> float | None:
    """Ratio of std of 5-day returns to std of 20-day returns.

    Values > 1 mean recent volatility is rising relative to the
    medium-term — typical of breakouts / regime changes. Values < 1
    mean compression / mean reversion likely.
    """
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return None
    closes = pd.to_numeric(ohlcv["close"], errors="coerce").dropna()
    if len(closes) < 25:
        return None
    rets = closes.pct_change().dropna()
    if len(rets) < 25:
        return None
    s5 = rets.tail(5).std()
    s20 = rets.tail(20).std()
    if s20 is None or s20 == 0 or math.isnan(s20):
        return None
    return float(s5 / s20)


# --- Public API --------------------------------------------------------


def compute_lab_features(
    grades: list[dict],
    *,
    raw_flow_df: pd.DataFrame | None = None,
    grade_history_df: pd.DataFrame | None = None,
    premium_history_df: pd.DataFrame | None = None,
    as_of: object | None = None,
    fetch_uw: bool = True,
    topn_cutoff: int = FEATURE_LAB_TOPN_DEFAULT,
    ohlcv_loader=None,
    uw_loader=None,
) -> list[dict]:
    """Compute one feature_lab row per ``grade`` dict.

    Parameters
    ----------
    grades:
        Output of ``app.features.flow_tracker.compute_multi_day_flow``.
    raw_flow_df:
        Optional DataFrame of today's raw_flow rows (single combined
        frame across all tickers) used for far-OTM share and dollar-
        delta-weighted flow.  If None, those columns are NaN.
    grade_history_df:
        Deprecated.  Formerly the source for the momentum z-score; kept
        for backward compatibility but no longer used (the momentum
        baseline now comes from ``premium_history_df``).
    premium_history_df:
        Optional tidy frame from ``load_screener_premium_history`` used
        for the ``prem_momentum_z3d`` column.  If None, that column is
        NaN.
    as_of:
        The scan day (date/str/Timestamp) that today's premium is scored
        against the trailing baseline for.  Defaults to the latest day in
        ``premium_history_df`` when omitted.
    fetch_uw:
        If True, fetch the 6 UW endpoints for top-N candidates.
    topn_cutoff:
        Only the top-N candidates by conviction_score get UW fetches.
    ohlcv_loader:
        Callable ``ticker -> pd.DataFrame`` returning daily OHLCV bars.
        Defaults to ``app.features.price_features.fetch_ohlcv``.
    uw_loader:
        Callable ``ticker, spot -> dict`` returning the 12 UW
        feature columns for a ticker. Defaults to
        ``app.features.feature_lab_uw.fetch_uw_features``.

    Returns
    -------
    list of dicts, each with all ``LAB_COLS`` keys.
    """
    if not grades:
        return []

    if ohlcv_loader is None:
        try:
            from app.features.price_features import fetch_ohlcv as _fetch
            ohlcv_loader = lambda t: _fetch(t, lookback_days=120, include_partial=False)  # noqa: E731
        except Exception:
            ohlcv_loader = lambda t: None  # noqa: E731
    if uw_loader is None and fetch_uw:
        try:
            from app.features.feature_lab_uw import fetch_uw_features as _uw
            uw_loader = _uw
        except Exception:
            uw_loader = None

    # Resolve the reference day for the premium-momentum baseline.
    momentum_as_of = pd.to_datetime(as_of, errors="coerce") if as_of is not None else pd.NaT
    if pd.isna(momentum_as_of) and premium_history_df is not None and not premium_history_df.empty:
        momentum_as_of = pd.to_datetime(premium_history_df["day"], errors="coerce").max()
    momentum_as_of = momentum_as_of.normalize() if not pd.isna(momentum_as_of) else pd.NaT

    sector_rel = _sector_relative_pct_lookup(grades)

    ranked = sorted(
        grades,
        key=lambda g: float(g.get("conviction_score") or 0.0),
        reverse=True,
    )
    topn_tickers = {
        (str(g.get("ticker") or "").upper(), str(g.get("direction") or "BULLISH").upper())
        for g in ranked[:topn_cutoff]
    }

    out_rows: list[dict] = []
    for g in grades:
        ticker = str(g.get("ticker") or "").upper().strip()
        direction = str(g.get("direction") or "BULLISH").upper().strip()
        if not ticker:
            continue

        try:
            ohlcv = ohlcv_loader(ticker)
        except Exception:
            ohlcv = None

        ticker_flow = pd.DataFrame()
        if raw_flow_df is not None and not raw_flow_df.empty and "ticker" in raw_flow_df.columns:
            ticker_flow = raw_flow_df[raw_flow_df["ticker"].astype(str).str.upper() == ticker]

        far_call, far_put = _far_otm_shares(ticker_flow)
        ddw = _dollar_delta_weighted_flow(ticker_flow, direction)
        aggressor = _aggressor_signed_features(ticker_flow)

        row: dict[str, Any] = {
            "as_of": "",
            "ticker": ticker,
            "direction": direction,
            "conviction_grade": g.get("conviction_grade"),
            "conviction_score": _safe_float(g.get("conviction_score")),
            "sector": g.get("sector"),
            "bullish_premium_share": _bullish_premium_share(g),
            "unusual_premium_share": _unusual_premium_share(g),
            "vrp_proxy": _vrp_proxy(g, ohlcv),
            "far_otm_call_share": far_call,
            "far_otm_put_share": far_put,
            "dollar_delta_weighted_flow": ddw,
            "sector_relative_pct": sector_rel.get(g.get("ticker")),
            "prem_momentum_z3d": _prem_momentum_z3d(
                ticker,
                momentum_as_of,
                premium_history_df,
            ),
            "realized_vol_regime": _realized_vol_regime(ohlcv),
            **{c: aggressor.get(c) for c in AGGRESSOR_FEATURE_COLS},
        }
        # UW columns — populated for top-N only.
        for col in UW_FEATURE_COLS:
            row[col] = None
        if fetch_uw and uw_loader is not None and (ticker, direction) in topn_tickers:
            spot: float | None = None
            if ohlcv is not None and not ohlcv.empty and "close" in ohlcv.columns:
                try:
                    spot = float(ohlcv["close"].dropna().iloc[-1])
                except Exception:
                    spot = None
            try:
                uw_data = uw_loader(ticker, spot)
            except Exception:
                uw_data = {}
            for col in UW_FEATURE_COLS:
                if col in (uw_data or {}):
                    row[col] = _safe_float(uw_data.get(col))

        out_rows.append(row)

    # Shadow momentum score: cross-sectional across this day's universe.
    try:
        from app.features.momentum_score import compute_day_scores
        day_scores = compute_day_scores(out_rows)
        for row, sc in zip(out_rows, day_scores):
            row["momentum_composite"] = sc.get("momentum_composite")
            row["momentum_score"] = sc.get("momentum_score")
    except Exception:
        for row in out_rows:
            row.setdefault("momentum_composite", None)
            row.setdefault("momentum_score", None)

    return out_rows


def _coerce(value: Any) -> Any:
    """Collapse nested dicts / lists to blank so CSV stays flat."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    return value


def _load_existing_rows() -> list[dict]:
    if not FEATURE_LAB_PATH.exists():
        return []
    try:
        with open(FEATURE_LAB_PATH, "r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _write_rows(rows: list[dict]) -> None:
    FEATURE_LAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_LAB_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LAB_COLS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def persist_feature_lab(rows: list[dict], as_of: str) -> int:
    """Write today's feature_lab rows to ``data/feature_lab.csv``.

    Idempotent on ``as_of``: re-running replaces today's rows so
    repeated hourly scans don't double the panel.  Returns the number
    of rows written.
    """
    if not rows:
        return 0
    existing = _load_existing_rows()
    existing = [r for r in existing if str(r.get("as_of", "")) != str(as_of)]

    new_rows = []
    for r in rows:
        out = dict(r)
        out["as_of"] = as_of
        new_rows.append({k: _coerce(out.get(k)) for k in LAB_COLS})
    _write_rows(existing + new_rows)
    return len(new_rows)


def load_recent_grade_history(days: int = 7) -> pd.DataFrame | None:
    """Load the trailing N days of ``grade_history.csv`` for momentum calc.

    Returns None if the file is missing or unreadable.
    """
    if not GRADE_HISTORY_PATH.exists():
        return None
    try:
        df = pd.read_csv(GRADE_HISTORY_PATH)
    except Exception:
        return None
    if "as_of" not in df.columns:
        return None
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df = df.dropna(subset=["as_of"])
    if df.empty:
        return None
    cutoff = df["as_of"].max() - pd.Timedelta(days=days)
    return df[df["as_of"] >= cutoff].copy()


def load_latest_raw_flow() -> pd.DataFrame | None:
    """Load the most recent raw_flow CSV from ``data/raw_flow/``.

    Returns None if no raw_flow files exist.
    """
    if not RAW_FLOW_DIR.exists():
        return None
    files = sorted(RAW_FLOW_DIR.glob("raw_flow_*.csv"))
    if not files:
        return None
    try:
        return pd.read_csv(files[-1])
    except Exception:
        return None


_SNAP_PREMIUM_COLS = ["snapshot_date", "ticker", "total_bullish_premium", "total_bearish_premium"]


def _read_snapshot_premium(path: Path) -> pd.DataFrame | None:
    """Read the premium columns from a screener-snapshot file (csv or gz)."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=lambda c: c in _SNAP_PREMIUM_COLS)
    except Exception:
        return None
    if df.empty or "snapshot_date" not in df.columns or "ticker" not in df.columns:
        return None
    return df


def load_screener_premium_history(
    days: int | None = 60,
    *,
    include_archive: bool = False,
) -> pd.DataFrame | None:
    """Load a tidy per-``(ticker, day)`` daily-premium series for momentum.

    ``daily_premium`` mirrors ``flow_tracker``'s ``cum_total`` for a single
    day: ``total_bullish_premium + total_bearish_premium``.  The live
    ``screener_snapshots.csv`` (rolling ~15 trading days) is enough for the
    3-day trailing window used live; pass ``include_archive=True`` to also
    fold in ``snapshots_archive.csv.gz`` for full-history backfills.

    Returns a frame with columns ``ticker`` (upper), ``day`` (normalized
    Timestamp), ``daily_premium`` (float), deduped to one row per
    ``(ticker, day)`` keeping the last observation.  Returns None when no
    source file is available.
    """
    frames = []
    live = _read_snapshot_premium(SCREENER_SNAPSHOTS_PATH)
    if live is not None:
        frames.append(live)
    if include_archive:
        arch = _read_snapshot_premium(SNAPSHOTS_ARCHIVE_PATH)
        if arch is not None:
            frames.append(arch)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["day"])
    if df.empty:
        return None
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    bull = pd.to_numeric(df.get("total_bullish_premium"), errors="coerce").fillna(0.0)
    bear = pd.to_numeric(df.get("total_bearish_premium"), errors="coerce").fillna(0.0)
    df["daily_premium"] = bull + bear

    # One row per (ticker, day); keep the last observation of the day.
    df = df.sort_values("day").drop_duplicates(subset=["ticker", "day"], keep="last")

    if days is not None:
        cutoff = df["day"].max() - pd.Timedelta(days=days)
        df = df[df["day"] >= cutoff]

    return df[["ticker", "day", "daily_premium"]].reset_index(drop=True)

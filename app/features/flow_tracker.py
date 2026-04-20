"""Multi-day flow tracker — persist UW screener snapshots and surface repeat unusual activity."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.market_calendar import current_trading_day
from app.config import (
    FLOW_TRACKER_DTE_BUCKETS,
    FLOW_TRACKER_ETF_EXCLUDE,
    FLOW_TRACKER_HARD_MODE_FILTER,
    FLOW_TRACKER_LOOKBACK_DAYS,
    FLOW_TRACKER_MAX_RESULTS,
    FLOW_TRACKER_MIN_3D_PERCENTILE,
    FLOW_TRACKER_MIN_ACTIVE_DAYS,
    FLOW_TRACKER_MIN_MCAP,
    FLOW_TRACKER_MIN_PREM_MCAP_BPS,
    FLOW_TRACKER_MIN_PREMIUM,
    FLOW_TRACKER_MODES,
    FLOW_TRACKER_RETENTION_DAYS,
    FLOW_TRACKER_WEIGHTS_ACCUM,
)


# Grade rank used for `min_grade_rank` gates in FLOW_TRACKER_MODES.  Higher
# rank = better grade.  Keep in sync with `grade_explainer.conviction_grade`.
_GRADE_RANK = {"C": 0, "B-": 1, "B": 2, "B+": 3, "A-": 4, "A": 5, "A+": 6}


def _num(v, default: float = 0.0) -> float:
    """Coerce ``v`` to ``float``, mapping ``None`` / ``NaN`` / non-numeric to ``default``.

    Prevents the "NaN is truthy" trap where ``float(x or 0)`` returns
    ``nan`` when ``x`` is ``float('nan')`` (pandas-originated missing
    values).  Any UI dict value that flows into ``"%+.1f"|format`` /
    ``toFixed`` must go through this helper so the template never
    renders ``+nan%`` / ``$nan``.
    """
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN check
        return default
    return x


def _mode_passes(row: dict, mode_cfg: dict) -> bool:
    """Return True if a scored row clears every gate defined by ``mode_cfg``.

    ``row`` must already carry ``active_days``, ``_cum_total`` (or
    ``cumulative_premium``), ``prem_mcap_bps``, ``_consistency_raw``,
    ``_accel_t_stat``, ``hedging_risk``, ``conviction_grade``.
    """
    if row.get("active_days", 0) < mode_cfg["min_active_days"]:
        return False
    cum_total = row.get("_cum_total")
    if cum_total is None:
        cum_total = row.get("cumulative_premium", 0.0)
    if _num(cum_total) < mode_cfg["min_cum_premium"]:
        return False
    if _num(row.get("prem_mcap_bps")) < mode_cfg["min_prem_mcap_bps"]:
        return False
    if _num(row.get("_consistency_raw")) < mode_cfg["min_consistency"]:
        return False
    if _num(row.get("_accel_t_stat")) < mode_cfg["min_accel_t"]:
        return False
    if mode_cfg["exclude_hedging"] and row.get("hedging_risk"):
        return False
    grade = str(row.get("conviction_grade", "C") or "C")
    if _GRADE_RANK.get(grade, 0) < mode_cfg["min_grade_rank"]:
        return False
    # Wave 0.5 A6 — 3-day percentile gate for accumulation modes only.  Uses
    # the window max so a ticker that hit top 30% unusual activity even once
    # during the window stays in; keeps the gate from blocking tickers whose
    # unusualness cools off on the final day of the window.
    if mode_cfg.get("exclude_hedging"):
        perc_3d = row.get("perc_3_day_total_max")
        if perc_3d is not None and float(perc_3d) > 0 and float(perc_3d) < FLOW_TRACKER_MIN_3D_PERCENTILE:
            return False
    return True


def _accumulation_score(
    active_days: int,
    lookback_days: int,
    consistency_raw: float,
    accel_t_stat: float,
    prem_mcap_bps: float,
) -> float:
    """Purpose-built accumulation-ness (0-100) per [0.3] in the plan.

    0.35 persistence + 0.30 one-sidedness + 0.25 rising + 0.10 per-bps.
    Each component clipped to [0, 1] before weighting.
    """
    lb = max(int(lookback_days), 1)
    days_norm = min(float(active_days) / lb, 1.0)
    consistency_norm = float(np.clip(consistency_raw, 0.0, 1.0))
    accel_norm = float(np.clip((float(accel_t_stat) + 0.5) / 2.5, 0.0, 1.0))
    bps_norm = float(np.clip(float(prem_mcap_bps) / 5.0, 0.0, 1.0))
    return round(
        100.0 * (
            0.35 * days_norm
            + 0.30 * consistency_norm
            + 0.25 * accel_norm
            + 0.10 * bps_norm
        ),
        1,
    )

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SNAPSHOTS_PATH = DATA_DIR / "screener_snapshots.csv"
STATS_META_PATH = DATA_DIR / "flow_stats_meta.json"
# Append-only full-history archive used by the grade backtest.  The hot
# CSV above is pruned at FLOW_TRACKER_RETENTION_DAYS; this archive is
# strictly additive so backtests can regress over 60-90+ days without
# the retention policy capping them.
SNAPSHOTS_ARCHIVE_PATH = DATA_DIR / "snapshots_archive.csv.gz"


def _append_rows_to_archive(rows: list[dict]) -> None:
    """Append ``rows`` to the append-only snapshot archive.

    Writes a header on first create.  Silently tolerates schema
    evolution: new columns not in ``SNAPSHOT_COLS`` are ignored
    (``extrasaction='ignore'``); columns present in the header but not
    in a given row are written as empty strings.  Gzipped to keep the
    artifact small in git.
    """
    if not rows:
        return
    import gzip

    SNAPSHOTS_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    first_write = not SNAPSHOTS_ARCHIVE_PATH.exists()
    mode = "wt" if first_write else "at"
    try:
        with gzip.open(SNAPSHOTS_ARCHIVE_PATH, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
            if first_write:
                writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"  [flow-tracker] archive append failed (continuing): {e}")


def _load_stats_meta() -> dict:
    """Read ``data/flow_stats_meta.json`` (ticker → first_observed_date).

    Returns an empty dict if the file is missing or malformed.
    """
    import json
    if not STATS_META_PATH.exists():
        return {}
    try:
        with open(STATS_META_PATH, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_stats_meta(meta: dict) -> None:
    import json
    STATS_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_META_PATH, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def update_stats_meta(tickers) -> None:
    """Record ``first_observed_date`` for any ticker seen for the first time.

    Used by ``flow_stats`` to distinguish genuine short-history tickers (new
    IPOs / new screener entries) from tickers with gap days in their series.
    """
    if not tickers:
        return
    today_str = current_trading_day().isoformat()
    meta = _load_stats_meta()
    changed = False
    for t in tickers:
        t = (str(t) or "").upper().strip()
        if not t:
            continue
        if t not in meta:
            meta[t] = today_str
            changed = True
    if changed:
        _save_stats_meta(meta)

SNAPSHOT_COLS = [
    "snapshot_date",
    "ticker",
    "sector",
    "close",
    "marketcap",
    # Premium taxonomy (Flow Tracker - Premium Taxonomy plan):
    #   bullish_premium / bearish_premium       — legacy alias, == total_*_premium
    #   total_*_premium                          — UW aggregate directional (all
    #                                              trades / all DTE) from the
    #                                              screener or options-volume
    #                                              endpoint.  Authoritative for
    #                                              Flow Tracker's multi-day
    #                                              regression.
    #   unusual_*_premium                        — flow-alert-derived, premium
    #                                              >= $500K, no DTE filter.
    #   lottery_*_premium / swing_*_premium /
    #   leap_*_premium                           — unusual flow sliced into
    #                                              DTE buckets (config
    #                                              FLOW_TRACKER_PREMIUM_BUCKETS).
    #                                              Powers the Trader Card
    #                                              Premium Mix panel.
    #   premium_source                           — "screener" | "per_ticker_api"
    #                                              for data-lineage debugging.
    "bullish_premium",
    "bearish_premium",
    "total_bullish_premium",
    "total_bearish_premium",
    "unusual_bullish_premium",
    "unusual_bearish_premium",
    "lottery_bullish_premium",
    "lottery_bearish_premium",
    "swing_bullish_premium",
    "swing_bearish_premium",
    "leap_bullish_premium",
    "leap_bearish_premium",
    "premium_source",
    "net_premium",
    "call_volume",
    "put_volume",
    "volume",
    "call_open_interest",
    "put_open_interest",
    "total_oi_change_perc",
    "call_oi_change_perc",
    "put_oi_change_perc",
    "put_call_ratio",
    "iv_rank",
    "iv30d",
    "perc_3_day_total",
    "perc_30_day_total",
    # Wave 0.5 A5 — call/put premium split (usually returned by the screener
    # as `call_premium`/`put_premium`).  Falls back to None when absent.
    "call_premium",
    "put_premium",
    # Wave 0.5 A1 / A2 — flow-alert-derived enrichment attached by the
    # pipeline via `save_screener_snapshot(..., flow_enrichment=...)` or
    # `save_flow_feature_snapshot`.
    "dominant_dte_bucket",
    "sweep_share",
    "multileg_share",
    # Wave 2 — intraday repeat-flow acceleration per side.  Today's
    # ``repeat_2h / directional_count`` ratio; enriched the same way as
    # sweep_share.
    "bullish_accel_ratio",
    "bearish_accel_ratio",
]


# Zero-valued skeleton for the premium-taxonomy columns.  Used by the
# shared ``_build_snapshot_row`` helper when a row has no bucket breakdown
# available (e.g. a screener row for a ticker without any qualifying flow
# alerts today).
_EMPTY_BUCKET_ROW = {
    "lottery_bullish_premium": 0.0,
    "lottery_bearish_premium": 0.0,
    "swing_bullish_premium":   0.0,
    "swing_bearish_premium":   0.0,
    "leap_bullish_premium":    0.0,
    "leap_bearish_premium":    0.0,
    "unusual_bullish_premium": 0.0,
    "unusual_bearish_premium": 0.0,
}


def _build_snapshot_row(
    *,
    snapshot_date: str,
    ticker: str,
    source: str,
    total_bullish_premium: float,
    total_bearish_premium: float,
    buckets: dict | None = None,
    base: dict | None = None,
    enrichment: dict | None = None,
) -> dict:
    """Assemble one screener-snapshot row with the full premium taxonomy.

    ``base`` is the raw screener / enrichment payload (fields copied as-is
    for columns that aren't part of the premium taxonomy).  ``buckets`` is
    the per-ticker DTE-bucket breakdown (keys: ``lottery_bullish_premium``,
    ``swing_bullish_premium``, etc.) from
    :func:`app.features.flow_features.aggregate_premium_by_dte_bucket`.
    ``enrichment`` is optional flow-alert structural enrichment (dominant
    DTE bucket, sweep share, multileg share, accel ratios).

    The row is guaranteed to have every column in ``SNAPSHOT_COLS`` present
    so the CSV writer never emits ragged rows.
    """
    base = base or {}
    enrichment = enrichment or {}
    buckets = {**_EMPTY_BUCKET_ROW, **(buckets or {})}

    total_bull = _num(total_bullish_premium)
    total_bear = _num(total_bearish_premium)

    swing_bull = _num(buckets.get("swing_bullish_premium"))
    swing_bear = _num(buckets.get("swing_bearish_premium"))

    unusual_bull = _num(buckets.get("unusual_bullish_premium")) or swing_bull
    unusual_bear = _num(buckets.get("unusual_bearish_premium")) or swing_bear

    row: dict = {col: None for col in SNAPSHOT_COLS}
    row.update({
        "snapshot_date": snapshot_date,
        "ticker": ticker,
        "sector": base.get("sector"),
        "close": base.get("close"),
        "marketcap": base.get("marketcap"),
        "bullish_premium": round(total_bull, 2),
        "bearish_premium": round(total_bear, 2),
        "total_bullish_premium": round(total_bull, 2),
        "total_bearish_premium": round(total_bear, 2),
        "unusual_bullish_premium": round(unusual_bull, 2),
        "unusual_bearish_premium": round(unusual_bear, 2),
        "lottery_bullish_premium": round(_num(buckets.get("lottery_bullish_premium")), 2),
        "lottery_bearish_premium": round(_num(buckets.get("lottery_bearish_premium")), 2),
        "swing_bullish_premium":   round(swing_bull, 2),
        "swing_bearish_premium":   round(swing_bear, 2),
        "leap_bullish_premium":    round(_num(buckets.get("leap_bullish_premium")), 2),
        "leap_bearish_premium":    round(_num(buckets.get("leap_bearish_premium")), 2),
        "premium_source": source,
        "net_premium": base.get("net_premium"),
        "call_volume": base.get("call_volume"),
        "put_volume": base.get("put_volume"),
        "volume": base.get("volume"),
        "call_open_interest": base.get("call_open_interest"),
        "put_open_interest": base.get("put_open_interest"),
        "total_oi_change_perc": base.get("total_oi_change_perc"),
        "call_oi_change_perc": base.get("call_oi_change_perc"),
        "put_oi_change_perc": base.get("put_oi_change_perc"),
        "put_call_ratio": base.get("put_call_ratio"),
        "iv_rank": base.get("iv_rank"),
        "iv30d": base.get("iv30d"),
        "perc_3_day_total": base.get("perc_3_day_total"),
        "perc_30_day_total": base.get("perc_30_day_total"),
        "call_premium": base.get("call_premium"),
        "put_premium": base.get("put_premium"),
    })

    for key in (
        "dominant_dte_bucket",
        "sweep_share",
        "multileg_share",
        "bullish_accel_ratio",
        "bearish_accel_ratio",
    ):
        if enrichment.get(key) not in (None, ""):
            row[key] = enrichment.get(key)
        elif base.get(key) not in (None, ""):
            row[key] = base.get(key)

    return row


def save_screener_snapshot(
    screener_data: list[dict],
    flow_enrichment: dict[str, dict] | None = None,
    premium_buckets: dict[str, dict] | None = None,
) -> None:
    """Persist today's screener response to the rolling snapshots CSV.

    Upserts: replaces any rows with today's date, appends new ones.
    Prunes rows older than the lookback window + buffer.

    Parameters
    ----------
    screener_data
        Raw screener rows from UW's ``/screener/stocks`` endpoint.
    flow_enrichment
        Optional ``{ticker: {key: value}}`` mapping used to fill in columns
        the screener doesn't provide (dominant DTE bucket, sweep share,
        multileg share).  Only keys already in :data:`SNAPSHOT_COLS` are
        copied; missing tickers are left as ``None``.
    premium_buckets
        Optional ``{ticker: {lottery_bullish_premium: ..., ...}}`` mapping
        from :func:`app.features.flow_features.aggregate_premium_by_dte_bucket`.
        When absent, bucket columns are written as zeros.
    """
    if not screener_data:
        return

    today = current_trading_day()
    today_str = today.isoformat()
    # Wave 0.5 C1 — retention extended from LOOKBACK+3 (8d) to 21d so the
    # 15d horizon toggle has its history and the B3 relative-PCR baseline
    # can reach the ≥10-observation threshold.
    cutoff = (today - timedelta(days=FLOW_TRACKER_RETENTION_DAYS)).isoformat()
    flow_enrichment = flow_enrichment or {}
    premium_buckets = premium_buckets or {}

    new_rows: list[dict] = []
    for sr in screener_data:
        ticker = (sr.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        # UW screener's `bullish_premium` / `bearish_premium` are the
        # authoritative aggregate-directional total for the day.  We keep
        # both `bullish_premium` (legacy column) and `total_bullish_premium`
        # populated so downstream consumers can migrate without breakage.
        row = _build_snapshot_row(
            snapshot_date=today_str,
            ticker=ticker,
            source="screener",
            total_bullish_premium=sr.get("bullish_premium") or 0.0,
            total_bearish_premium=sr.get("bearish_premium") or 0.0,
            buckets=premium_buckets.get(ticker),
            base=sr,
            enrichment=flow_enrichment.get(ticker),
        )
        new_rows.append(row)

    if not new_rows:
        return

    existing: list[dict] = []
    if SNAPSHOTS_PATH.exists():
        try:
            with open(SNAPSHOTS_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    d = r.get("snapshot_date", "")
                    if d != today_str and d >= cutoff:
                        existing.append(r)
        except Exception:
            existing = []

    SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = existing + new_rows
    with open(SNAPSHOTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Append-only archive (never pruned) for the grade backtest.  The
    # hot CSV above is self-pruning at FLOW_TRACKER_RETENTION_DAYS, but
    # backtests need longer history.
    _append_rows_to_archive(new_rows)

    try:
        update_stats_meta(r["ticker"] for r in new_rows)
    except Exception as e:
        print(f"  [flow-tracker] stats-meta update failed: {e}")

    print(f"  [flow-tracker] saved {len(new_rows)} screener rows for {today_str} "
          f"({len(existing)} historical rows retained)")


def save_flow_feature_snapshot(
    feature_table: pd.DataFrame,
    premium_buckets: dict[str, dict] | None = None,
) -> None:
    """Merge flow-feature tickers into screener_snapshots.csv.

    The UW stock screener only returns ~340 tickers per day, missing many
    tickers that have clear unusual flow in the pipeline.  This function
    fills the gaps by appending flow-feature tickers that are **not already
    present** for today's date, using the metrics available from flow scoring.
    Screener data is richer, so it always takes priority.

    Premium-taxonomy policy: every gap-filler row carries the full UW
    aggregate (total_*_premium) from ``/stock/{ticker}/options-volume`` so
    its premium semantics match screener rows exactly.  When the per-ticker
    fetch fails, the ticker is **skipped for this scan** rather than
    silently falling back to the narrow flow-alert premium — keeps the
    multi-day regression consistent across days.

    ``premium_buckets`` is an optional
    ``{ticker: {lottery_bullish_premium: ..., ...}}`` mapping produced by
    :func:`app.features.flow_features.aggregate_premium_by_dte_bucket`.
    """
    if feature_table is None or feature_table.empty:
        return

    today_str = current_trading_day().isoformat()

    # Read existing today rows to find which tickers are already covered
    existing_today: set[str] = set()
    if SNAPSHOTS_PATH.exists():
        try:
            with open(SNAPSHOTS_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("snapshot_date") == today_str:
                        existing_today.add((r.get("ticker") or "").upper().strip())
        except Exception:
            pass

    # Build a list of tickers we'll enrich with per-ticker UW calls.  We do
    # this up-front (and in parallel) so the slow network calls overlap
    # rather than blocking one-by-one.
    premium_buckets = premium_buckets or {}
    gap_tickers: list[str] = []
    gap_feature_rows: dict[str, dict] = {}
    for _, row in feature_table.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not ticker or ticker in existing_today:
            continue
        gap_tickers.append(ticker)
        gap_feature_rows[ticker] = {
            "mcap": float(row.get("marketcap", 0) or 0),
            "dominant_dte_bucket": row.get("dominant_dte_bucket"),
            "sweep_share": row.get("sweep_share"),
            "multileg_share": row.get("multileg_share"),
            "bullish_accel_ratio": row.get("bullish_accel_ratio"),
            "bearish_accel_ratio": row.get("bearish_accel_ratio"),
            "total_count": row.get("total_count"),
        }

    if not gap_tickers:
        return

    # Per-ticker enrichment — UW screener only returns ~340 tickers per
    # day (due to the relative-volume filter / 500-row page cap), but
    # flow-features routinely surfaces 100+ tickers below that cut-off.
    # Without this, those rows land with close=None / OI=None / IV=None
    # and every Trader Card for them renders zeros.  We fan-out to
    # /stock/{ticker}/options-volume + /iv-rank + /info concurrently.
    #
    # Premium-taxonomy plan: when the per-ticker fetch fails for a
    # ticker, we **skip** that ticker rather than silently persisting a
    # row with narrow flow-alert premium.  This keeps the daily series
    # apples-to-apples for ``compute_multi_day_flow``'s log-linear
    # regression — mixing UW-aggregate rows with flow-alert-narrow rows
    # distorts acceleration / persistence / bps calculations.
    enrichment_map: dict[str, dict] = {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        from app.vendors.unusual_whales import fetch_ticker_options_snapshot

        max_workers = min(8, max(1, len(gap_tickers)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for ticker, snap in zip(
                gap_tickers, pool.map(fetch_ticker_options_snapshot, gap_tickers)
            ):
                if snap:
                    enrichment_map[ticker] = snap
    except Exception as e:
        print(f"  [flow-tracker] per-ticker enrichment failed: {e}")

    new_rows: list[dict] = []
    skipped_tickers: list[str] = []
    for ticker in gap_tickers:
        ft = gap_feature_rows[ticker]
        enriched = enrichment_map.get(ticker)
        if not enriched:
            skipped_tickers.append(ticker)
            continue

        mcap = ft["mcap"]
        # Merge the flow-feature marketcap back onto the enriched base
        # dict so _build_snapshot_row's base fields work uniformly.
        base = dict(enriched)
        if mcap > 0 and not base.get("marketcap"):
            base["marketcap"] = mcap
        if base.get("volume") in (None, 0):
            base["volume"] = ft.get("total_count")
        if base.get("net_premium") is None:
            total_b = _num(base.get("bullish_premium"))
            total_s = _num(base.get("bearish_premium"))
            base["net_premium"] = round(total_b - total_s, 2)

        new_rows.append(
            _build_snapshot_row(
                snapshot_date=today_str,
                ticker=ticker,
                source="per_ticker_api",
                total_bullish_premium=base.get("bullish_premium") or 0.0,
                total_bearish_premium=base.get("bearish_premium") or 0.0,
                buckets=premium_buckets.get(ticker),
                base=base,
                enrichment=ft,
            )
        )

    if not new_rows:
        if skipped_tickers:
            print(
                f"  [flow-tracker] WARN all {len(skipped_tickers)} flow-feature gap-filler "
                f"tickers skipped (per-ticker fetch failed). No gap-filler rows persisted "
                f"this scan."
            )
        return

    enriched_count = len(new_rows)
    print(
        f"  [flow-tracker] per-ticker enrichment: {enriched_count}/{len(gap_tickers)} "
        f"gap-filler rows back-filled with close/OI/IV from UW"
    )
    if skipped_tickers:
        print(
            f"  [flow-tracker] WARN skipped {len(skipped_tickers)} gap-filler tickers "
            f"(per-ticker fetch failed — would have used narrow premium, policy is to drop): "
            f"{', '.join(skipped_tickers[:8])}"
            f"{'...' if len(skipped_tickers) > 8 else ''}"
        )

    # Append to the existing file (screener snapshot already wrote today's rows)
    all_rows: list[dict] = []
    if SNAPSHOTS_PATH.exists():
        try:
            with open(SNAPSHOTS_PATH, "r", newline="") as f:
                all_rows = list(csv.DictReader(f))
        except Exception:
            all_rows = []

    all_rows.extend(new_rows)
    SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Mirror the gap-filler rows into the append-only archive so the
    # backtest sees a complete universe (not just screener tickers).
    _append_rows_to_archive(new_rows)

    try:
        update_stats_meta(r["ticker"] for r in new_rows)
    except Exception as e:
        print(f"  [flow-tracker] stats-meta update failed: {e}")

    print(f"  [flow-tracker] merged {len(new_rows)} flow-feature tickers "
          f"(skipped {len(existing_today)} already from screener)")


def _conviction_grade(score: float) -> str:
    """Thin re-export of ``grade_explainer.conviction_grade``.

    Kept as a module-level name so existing imports
    (``from app.features.flow_tracker import _conviction_grade``) keep
    working. Implements the 7-tier ladder (A+/A/A-/B+/B/B-/C).
    """
    from app.features.grade_explainer import conviction_grade
    return conviction_grade(score)


def compute_multi_day_flow(
    lookback_days: int = FLOW_TRACKER_LOOKBACK_DAYS,
    min_active_days: int = FLOW_TRACKER_MIN_ACTIVE_DAYS,
    min_premium: float = FLOW_TRACKER_MIN_PREMIUM,
    min_mcap: float = FLOW_TRACKER_MIN_MCAP,
    min_prem_mcap_bps: float = FLOW_TRACKER_MIN_PREM_MCAP_BPS,
    max_results: int = FLOW_TRACKER_MAX_RESULTS,
    mode: str | None = None,
    as_of: "date | str | None" = None,
    snapshots_path: "Path | None" = None,
) -> list[dict]:
    """Aggregate screener snapshots over the lookback window.

    Returns a list of dicts (one per qualifying ticker) sorted by a
    composite conviction score.  Four-layer filter:
      1. ETF/ETP exclusion
      2. Minimum cumulative premium + market-cap floor
      3. Minimum prem/mcap bps floor
      4. min_active_days persistence gate
    Capped to max_results by conviction score.

    ``mode`` — when supplied and ``FLOW_TRACKER_HARD_MODE_FILTER`` is
    ``True``, rows failing the mode's gate are dropped **before** the
    conviction sort + cap, so the swing-candidate radar stays tight.
    Legacy callers that pass ``mode=None`` get the original behaviour
    (every row tagged with ``passes_*`` flags).
    """
    source_path = snapshots_path if snapshots_path is not None else SNAPSHOTS_PATH
    if not source_path.exists():
        return []

    try:
        # pandas transparently handles `.csv.gz` via the extension.
        df = pd.read_csv(source_path)
    except Exception:
        return []

    if df.empty or "snapshot_date" not in df.columns:
        return []

    # Layer 1: ETF exclusion
    if "ticker" in df.columns:
        df = df[~df["ticker"].isin(FLOW_TRACKER_ETF_EXCLUDE)]

    # Resolve `as_of` — supports date, ISO string, or None (live scan).
    # When `as_of` is explicit, we're in backtest-replay mode and must
    # drop rows strictly after `as_of` (point-in-time replay).  When
    # `as_of` is None we're live; we use current_trading_day() as the
    # window anchor but do NOT apply the future-row filter — existing
    # tests and weekend scans rely on being able to seed/load
    # snapshots dated slightly ahead of the current NYSE trading day.
    if as_of is None:
        today = current_trading_day()
    else:
        if isinstance(as_of, str):
            today = date.fromisoformat(as_of)
        else:
            today = as_of
        today_str = today.isoformat()
        df = df[df["snapshot_date"] <= today_str]

    cutoff = (today - timedelta(days=lookback_days)).isoformat()
    df = df[df["snapshot_date"] >= cutoff].copy()
    if df.empty:
        return []

    for col in ("bullish_premium", "bearish_premium", "net_premium",
                 "total_bullish_premium", "total_bearish_premium",
                 "unusual_bullish_premium", "unusual_bearish_premium",
                 "lottery_bullish_premium", "lottery_bearish_premium",
                 "swing_bullish_premium", "swing_bearish_premium",
                 "leap_bullish_premium", "leap_bearish_premium",
                 "marketcap", "close", "iv_rank", "iv30d",
                 "total_oi_change_perc", "call_oi_change_perc", "put_oi_change_perc",
                 "put_call_ratio", "perc_3_day_total", "perc_30_day_total",
                 "call_volume", "put_volume", "volume",
                 "call_open_interest", "put_open_interest",
                 # Wave 0.5 A1/A2/A5 enrichment
                 "sweep_share", "multileg_share",
                 "call_premium", "put_premium",
                 # Wave 2 — repeat-flow acceleration per side.
                 "bullish_accel_ratio", "bearish_accel_ratio"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Premium-taxonomy plan: regress on the UW aggregate-directional
    # premium (``total_*_premium``).  Legacy rows (pre-taxonomy scans)
    # only have ``bullish_premium`` / ``bearish_premium`` which were
    # *already* the UW aggregate for screener rows — so fillna from the
    # legacy columns is safe.  Gap-filler rows written with the old
    # narrow definition will show up here too; the purge script in
    # scripts/purge_snapshot_history.py drops those before first run.
    if "total_bullish_premium" not in df.columns:
        df["total_bullish_premium"] = df.get("bullish_premium")
    else:
        df["total_bullish_premium"] = df["total_bullish_premium"].fillna(df.get("bullish_premium"))
    if "total_bearish_premium" not in df.columns:
        df["total_bearish_premium"] = df.get("bearish_premium")
    else:
        df["total_bearish_premium"] = df["total_bearish_premium"].fillna(df.get("bearish_premium"))
    df["total_bullish_premium"] = df["total_bullish_premium"].fillna(0.0)
    df["total_bearish_premium"] = df["total_bearish_premium"].fillna(0.0)

    all_dates = sorted(df["snapshot_date"].unique())
    total_days = len(all_dates)
    if total_days == 0:
        return []

    raw_results: list[dict] = []

    # Wave 0.5 B2 — premium-weighted persistence.  A day only counts toward
    # active_days when its premium clears max(15% × per-day mean, $100K).
    # Stops `$10M day 1 + $50K × 4` from scoring persistence 5/5.
    _PERSISTENCE_ABS_FLOOR = 100_000.0
    _PERSISTENCE_REL_FLOOR = 0.15

    for ticker, grp in df.groupby("ticker"):
        raw_active_dates = sorted(grp["snapshot_date"].unique())
        raw_active_days = len(raw_active_dates)

        if raw_active_days < min_active_days:
            continue

        cum_bull = float(grp["total_bullish_premium"].sum())
        cum_bear = float(grp["total_bearish_premium"].sum())
        cum_net = float(grp["net_premium"].sum()) if "net_premium" in grp.columns else (cum_bull - cum_bear)
        cum_total = cum_bull + cum_bear

        latest = grp.sort_values("snapshot_date").iloc[-1]
        mcap = float(latest.get("marketcap") or 0)

        # Layer 2: premium + mcap floors
        if cum_total < min_premium:
            continue
        if mcap < min_mcap:
            continue

        if cum_total > 0 and mcap > 0:
            dominant_prem = max(cum_bull, cum_bear)
            prem_mcap_bps = round(dominant_prem / mcap * 10_000, 2)
        else:
            prem_mcap_bps = 0.0

        # Layer 3: prem/mcap bps floor
        if prem_mcap_bps < min_prem_mcap_bps:
            continue

        # Build per-day premium series for regression / concentration metrics.
        # Includes every day the ticker had data so trend regression and the
        # consistency calc see the full picture.
        raw_daily: list[tuple[int, float, float, float, str]] = []  # (day_idx, bull, bear, total, date)
        for idx, d in enumerate(raw_active_dates):
            day_rows = grp[grp["snapshot_date"] == d]
            db = float(day_rows["total_bullish_premium"].sum())
            dbe = float(day_rows["total_bearish_premium"].sum())
            raw_daily.append((idx, db, dbe, db + dbe, d))

        # Wave 0.5 B2 — premium-weighted active day count.
        gross_window = sum(t[3] for t in raw_daily)
        per_day_mean = gross_window / raw_active_days if raw_active_days else 0.0
        day_threshold = max(_PERSISTENCE_ABS_FLOOR, per_day_mean * _PERSISTENCE_REL_FLOOR)
        significant_dates = {t[4] for t in raw_daily if t[3] >= day_threshold}
        active_days = len(significant_dates)
        if active_days < min_active_days:
            # Premium-weighted gate failed → drop.  Keeps single-blast tickers
            # ($10M day 1 + pennies) out of the tracker even if raw day-count
            # cleared min_active_days.
            continue
        persistence_ratio = round(active_days / total_days, 2)

        daily_totals = [(i, b, be, tot) for (i, b, be, tot, _) in raw_daily]

        # --- Acceleration via log-linear regression ---
        # Regress log1p(daily_total) on day index, compute a t-statistic for the slope.
        # Maps smoothly to 0-1 via clip((t + 1) / 3, 0, 1). Robust to sparse/unequal days.
        accel_t_stat = 0.0
        trend = "steady"
        if active_days >= 2:
            day_idx_arr = np.array([t[0] for t in daily_totals], dtype=float)
            day_tot_arr = np.array([t[3] for t in daily_totals], dtype=float)
            y = np.log1p(day_tot_arr)
            x = day_idx_arr
            x_mean = x.mean()
            y_mean = y.mean()
            xv = x - x_mean
            yv = y - y_mean
            ss_xx = float((xv * xv).sum())
            if ss_xx > 0:
                slope = float((xv * yv).sum() / ss_xx)
                y_pred = y_mean + slope * xv
                residuals = y - y_pred
                dof = max(active_days - 2, 1)
                s_err = float(np.sqrt((residuals * residuals).sum() / dof))
                se_slope = s_err / np.sqrt(ss_xx) if ss_xx > 0 else 0.0
                if se_slope > 1e-9:
                    accel_t_stat = slope / se_slope
                else:
                    accel_t_stat = 0.0 if slope == 0 else (5.0 if slope > 0 else -5.0)
                if accel_t_stat > 1.0:
                    trend = "accelerating"
                elif accel_t_stat < -1.0:
                    trend = "fading"
                else:
                    trend = "steady"
        accel_raw = float(np.clip((accel_t_stat + 1.0) / 3.0, 0.0, 1.0))

        # --- Premium-weighted directional concentration ---
        # consistency = |Σ(bull - bear)| / Σ(bull + bear) across active days.
        # Naturally down-weights thin days; 1.0 = one-sided, 0.0 = perfectly mixed.
        net_signed = sum(b - be for _, b, be, _ in daily_totals)
        gross = sum(t for _, _, _, t in daily_totals)
        consistency_raw = abs(net_signed) / gross if gross > 0 else 0.0

        # Daily snapshots for sparkline
        daily_snaps: list[dict] = []
        for d in all_dates:
            day_row = grp[grp["snapshot_date"] == d]
            if day_row.empty:
                daily_snaps.append({"date": d, "premium": 0, "active": False})
            else:
                dr = day_row.iloc[0]
                dp = float(dr.get("total_bullish_premium") or 0) + float(dr.get("total_bearish_premium") or 0)
                daily_snaps.append({"date": d, "premium": dp, "active": True})

        direction = "BULLISH" if cum_bull >= cum_bear else "BEARISH"
        bull_pct = round(cum_bull / cum_total * 100, 1) if cum_total > 0 else 50.0

        avg_vol_30d = grp["perc_30_day_total"].mean()
        if pd.isna(avg_vol_30d):
            avg_vol_30d = 0.0

        # Wave 0.5 B4 — window-averaged PCR for the hedging-risk check.  Latest
        # day is noisy; the mean better reflects the week's protective posture.
        pcr_series = grp["put_call_ratio"].dropna() if "put_call_ratio" in grp.columns else pd.Series(dtype=float)
        window_avg_pcr = float(pcr_series.mean()) if not pcr_series.empty else 0.0

        # Wave 0.5 B3 — relative PCR baseline for the hedging haircut.  When
        # we have enough per-ticker history (≥10 observations) we compare to
        # the ticker's own median PCR; otherwise we fall back to the absolute
        # threshold.  Counts observations across the full CSV (retention
        # window) rather than the lookback slice so this kicks in naturally
        # once C1 extends retention.
        pcr_median = float(pcr_series.median()) if not pcr_series.empty else 0.0
        pcr_history_n = int(pcr_series.shape[0])

        # Wave 0.5 A1 — dominant DTE bucket across the window.  Takes the
        # most recent non-null value since screener publishes today's bucket
        # fresh each scan; falls back to the latest available historical
        # value so the chip stays populated when the latest screener row
        # doesn't surface flow detail.
        dte_bucket = None
        if "dominant_dte_bucket" in grp.columns:
            buckets = grp["dominant_dte_bucket"].dropna()
            if not buckets.empty:
                dte_bucket = str(buckets.mode().iloc[0])

        # Wave 0.5 A2 — window-average sweep / multileg share.
        sweep_share_win = float(grp["sweep_share"].mean()) if "sweep_share" in grp.columns else 0.0
        if pd.isna(sweep_share_win):
            sweep_share_win = 0.0
        multileg_share_win = float(grp["multileg_share"].mean()) if "multileg_share" in grp.columns else 0.0
        if pd.isna(multileg_share_win):
            multileg_share_win = 0.0

        # Wave 2 — today's repeat-flow acceleration ratio on the dominant side.
        # Intraday signal: fraction of today's directional prints that landed in
        # the last 2h.  Flat-session baseline is 2/6.5 ≈ 0.308 — we compute a
        # centred score so the UI can threshold as: >= +0.12 accelerating,
        # <= −0.12 fading, else steady.
        _FLAT = 2.0 / 6.5
        latest_bull_accel = float(latest.get("bullish_accel_ratio") or 0.0)
        latest_bear_accel = float(latest.get("bearish_accel_ratio") or 0.0)
        if pd.isna(latest_bull_accel):
            latest_bull_accel = 0.0
        if pd.isna(latest_bear_accel):
            latest_bear_accel = 0.0
        accel_ratio_today = latest_bull_accel if cum_bull >= cum_bear else latest_bear_accel
        accel_score_today = round(max(-0.5, min(0.7, accel_ratio_today - _FLAT)), 3)
        if accel_score_today >= 0.12:
            accel_label_today = "accelerating"
        elif accel_score_today <= -0.12:
            accel_label_today = "fading"
        else:
            accel_label_today = "steady"

        # Wave 0.5 A5 — window call/put premium split (takes the latest
        # non-null observation; summed across the window doesn't really
        # make sense because the screener reports rolling 24h values).
        call_prem_latest = None
        put_prem_latest = None
        if "call_premium" in grp.columns:
            cp = grp["call_premium"].dropna()
            if not cp.empty:
                call_prem_latest = float(cp.iloc[-1])
        if "put_premium" in grp.columns:
            pp = grp["put_premium"].dropna()
            if not pp.empty:
                put_prem_latest = float(pp.iloc[-1])

        # Wave 0.5 A4 — window return %.  Uses first/last close in the
        # window so we can compare flow direction to realised price action.
        window_return_pct = 0.0
        if "close" in grp.columns:
            closes = grp.sort_values("snapshot_date")["close"].dropna()
            if len(closes) >= 2 and float(closes.iloc[0]) > 0:
                window_return_pct = round(
                    (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1.0) * 100.0, 2
                )

        # Wave 0.5 A6 — 3-day percentile.  Gate uses the latest snapshot's
        # value; we also expose the window max for richer reasons copy.
        perc_3d_latest = float(latest.get("perc_3_day_total") or 0.0)
        perc_3d_window_max = float(grp["perc_3_day_total"].max()) if "perc_3_day_total" in grp.columns else 0.0
        if pd.isna(perc_3d_window_max):
            perc_3d_window_max = 0.0

        # Wave 0.5 A7 — window-average absolute OI change, normalised to [0,1]
        # via a 50% ceiling (50%+ daily OI growth is saturating-rare).
        oi_change_series = grp["total_oi_change_perc"].dropna() if "total_oi_change_perc" in grp.columns else pd.Series(dtype=float)
        oi_change_window_avg = float(oi_change_series.mean()) if not oi_change_series.empty else 0.0
        oi_change_norm = float(np.clip(abs(oi_change_window_avg) / 50.0, 0.0, 1.0))

        # Premium-taxonomy plan — per-ticker window totals by DTE bucket.
        # Reported as today's latest snapshot values (screener / UW
        # options-volume endpoint both report rolling-24h numbers that
        # don't sum meaningfully across the window) PLUS window sums for
        # the unusual-flow-derived lottery/swing/leap buckets (those DO
        # sum cleanly because each day's aggregation is independent).
        def _latest_col(col: str) -> float:
            if col not in grp.columns:
                return 0.0
            series = grp[col].dropna()
            return float(series.iloc[-1]) if not series.empty else 0.0

        def _sum_col(col: str) -> float:
            if col not in grp.columns:
                return 0.0
            return float(pd.to_numeric(grp[col], errors="coerce").fillna(0.0).sum())

        lottery_bull_sum = _sum_col("lottery_bullish_premium")
        lottery_bear_sum = _sum_col("lottery_bearish_premium")
        swing_bull_sum   = _sum_col("swing_bullish_premium")
        swing_bear_sum   = _sum_col("swing_bearish_premium")
        leap_bull_sum    = _sum_col("leap_bullish_premium")
        leap_bear_sum    = _sum_col("leap_bearish_premium")
        unusual_bull_sum = _sum_col("unusual_bullish_premium") or swing_bull_sum
        unusual_bear_sum = _sum_col("unusual_bearish_premium") or swing_bear_sum

        # "Other DTE" = unusual flow outside lottery/swing/leap (DTE 15-29,
        # 121-179).  Derived as max(0, unusual - lottery - swing - leap).
        other_bull = max(0.0, unusual_bull_sum - lottery_bull_sum - swing_bull_sum - leap_bull_sum)
        other_bear = max(0.0, unusual_bear_sum - lottery_bear_sum - swing_bear_sum - leap_bear_sum)

        # Premium source tag — latest row's source (tracks what feeds today).
        source_tag = None
        if "premium_source" in grp.columns:
            src_series = grp["premium_source"].dropna()
            if not src_series.empty:
                source_tag = str(src_series.iloc[-1])

        premium_mix = {
            "total_bullish":   round(cum_bull, 2),
            "total_bearish":   round(cum_bear, 2),
            "lottery_bullish": round(lottery_bull_sum, 2),
            "lottery_bearish": round(lottery_bear_sum, 2),
            "swing_bullish":   round(swing_bull_sum, 2),
            "swing_bearish":   round(swing_bear_sum, 2),
            "leap_bullish":    round(leap_bull_sum, 2),
            "leap_bearish":    round(leap_bear_sum, 2),
            "unusual_bullish": round(unusual_bull_sum, 2),
            "unusual_bearish": round(unusual_bear_sum, 2),
            "other_bullish":   round(other_bull, 2),
            "other_bearish":   round(other_bear, 2),
            "source":          source_tag,
        }

        raw_results.append({
            "ticker": ticker,
            "sector": latest.get("sector") or "—",
            "direction": direction,
            "bull_pct": bull_pct,
            "cumulative_premium": round(cum_total, 2),
            "cumulative_bull": round(cum_bull, 2),
            "cumulative_bear": round(cum_bear, 2),
            "cumulative_net": round(cum_net, 2),
            "prem_mcap_bps": prem_mcap_bps,
            "active_days": active_days,
            "raw_active_days": raw_active_days,
            "total_days": total_days,
            "persistence_ratio": persistence_ratio,
            "trend": trend,
            "avg_vol_ratio_30d": round(_num(avg_vol_30d), 2),
            # NaN-safe coercion — pandas snapshots frequently carry NaN
            # for tickers where UW returned a partial row.  Without the
            # guard these flowed into templates as "+nan%" / "$nan".
            "latest_oi_change": round(_num(latest.get("total_oi_change_perc")), 2),
            "latest_iv_rank": round(_num(latest.get("iv_rank")), 1),
            "latest_close": round(_num(latest.get("close")), 2),
            "latest_put_call_ratio": round(_num(latest.get("put_call_ratio")), 2),
            "window_avg_pcr": round(window_avg_pcr, 3),
            "ticker_pcr_median": round(pcr_median, 3),
            "ticker_pcr_history_n": pcr_history_n,
            "marketcap": mcap,
            "daily_snapshots": daily_snaps,
            "dominant_dte_bucket": dte_bucket,
            "sweep_share": round(sweep_share_win, 3),
            "multileg_share": round(multileg_share_win, 3),
            # Wave 2 — today's repeat-flow acceleration on the dominant side.
            "accel_ratio_today": round(accel_ratio_today, 3),
            "accel_score_today": accel_score_today,
            "accel_label_today": accel_label_today,
            "call_premium_latest": call_prem_latest,
            "put_premium_latest": put_prem_latest,
            "window_return_pct": window_return_pct,
            "perc_3_day_total_latest": round(perc_3d_latest, 3),
            "perc_3_day_total_max": round(perc_3d_window_max, 3),
            "oi_change_window_avg": round(oi_change_window_avg, 2),
            "premium_mix": premium_mix,
            "premium_source": source_tag,
            "_consistency_raw": consistency_raw,
            "_accel_raw": accel_raw,
            "_accel_t_stat": accel_t_stat,
            "_oi_change_norm": oi_change_norm,
            "_cum_total": cum_total,
        })

    if not raw_results:
        return []

    # --- Layer 3: Composite conviction score (0–10 scale) ---
    # Every component now uses ABSOLUTE anchors so grades are comparable
    # across cohort sizes and regime shifts.
    #
    # Absolute intensity scale: 1 bps → 0, 30 bps → 1 (log-spaced).
    _INTENSITY_FLOOR = np.log1p(1.0)
    _INTENSITY_CEIL = np.log1p(30.0)

    # Wave 0.5 B1 — absolute mass scale: $500K → 0, $50M → 1.  Replaces the
    # cohort-relative `log_prem / log_prem.max()` which penalized a ticker
    # every time peers happened to be bigger that day.
    _MASS_FLOOR = np.log1p(5e5)
    _MASS_CEIL = np.log1p(5e7)

    # Wave 0: always score with accumulation-oriented weights so one-sidedness
    # and acceleration carry their weight across every mode.  The 7-tier
    # ladder stays on the same 0-10 scale.
    w = FLOW_TRACKER_WEIGHTS_ACCUM

    # Wave 0.5 B3 — relative PCR haircut thresholds.  Falls back to absolute
    # floor when the ticker has <10 observations in the available history.
    _PCR_RELATIVE_MULT = 1.3                # window avg / median > 1.3 → elevated
    _PCR_RELATIVE_MIN_HISTORY = 10
    _PCR_BULLISH_ABS = 0.9
    _PCR_BEARISH_ABS = 0.5

    for i, r in enumerate(raw_results):
        persistence_norm = r["persistence_ratio"]
        log_bps = np.log1p(max(r["prem_mcap_bps"], 0.0))
        intensity_norm = float(np.clip(
            (log_bps - _INTENSITY_FLOOR) / (_INTENSITY_CEIL - _INTENSITY_FLOOR),
            0.0, 1.0,
        ))
        consistency_norm = r["_consistency_raw"]
        accel_norm = r["_accel_raw"]
        log_mass = np.log1p(max(r["_cum_total"], 0.0))
        mass_norm = float(np.clip(
            (log_mass - _MASS_FLOOR) / (_MASS_CEIL - _MASS_FLOOR),
            0.0, 1.0,
        ))
        oi_change_norm = r["_oi_change_norm"]

        score = (
            w["persistence"] * persistence_norm
            + w["intensity"]   * intensity_norm
            + w["consistency"] * consistency_norm
            + w["accel"]       * accel_norm
            + w["mass"]        * mass_norm
            + w.get("oi_change", 0.0) * oi_change_norm
        ) * 10.0

        # Wave 0.5 A1 — DTE multiplier.  Short-dated bullish flow (<=7d) is
        # often market-maker hedging or lottery gambling, not accumulation;
        # LEAPs (91d+) signal structural commitment.  Applied AFTER the
        # component sum but BEFORE PCR haircut so the two haircuts compose.
        dte_multiplier = 1.0
        dte_bucket_val = r.get("dominant_dte_bucket")
        if dte_bucket_val:
            for label, _lo, _hi, mult in FLOW_TRACKER_DTE_BUCKETS:
                if label == dte_bucket_val:
                    dte_multiplier = mult
                    break
        score *= dte_multiplier
        r["_dte_multiplier"] = dte_multiplier

        # --- Wave 0.5 B3/B4 — PCR haircut (relative-first, absolute fallback) ---
        # Bullish with elevated PCR or bearish with suppressed PCR = protective
        # hedging pattern, not directional conviction.  Uses the window average
        # (B4) vs the ticker's own median when we have enough history (B3),
        # falling back to the absolute threshold on short history so high-PCR
        # stocks like HOOD/MSTR/biotech don't false-flag.
        window_pcr = float(r.get("window_avg_pcr") or 0.0)
        pcr_median = float(r.get("ticker_pcr_median") or 0.0)
        pcr_n = int(r.get("ticker_pcr_history_n") or 0)
        hedging_risk = False
        pcr_check_mode = None
        if window_pcr > 0:
            if pcr_n >= _PCR_RELATIVE_MIN_HISTORY and pcr_median > 0:
                ratio = window_pcr / pcr_median
                pcr_check_mode = "relative"
                if r["direction"] == "BULLISH" and ratio > _PCR_RELATIVE_MULT:
                    hedging_risk = True
                elif r["direction"] == "BEARISH" and ratio < (1.0 / _PCR_RELATIVE_MULT):
                    hedging_risk = True
            else:
                pcr_check_mode = "absolute"
                if r["direction"] == "BULLISH" and window_pcr > _PCR_BULLISH_ABS:
                    hedging_risk = True
                elif r["direction"] == "BEARISH" and window_pcr < _PCR_BEARISH_ABS:
                    hedging_risk = True
        if hedging_risk:
            score *= 0.85

        # Decouple-Score plan — window_return_pct no longer adjusts
        # conviction_score.  Price confirmation lives in the stack's
        # _score_price component (up to 12 pts); conviction_score is
        # a pure flow-quality metric.
        r["conviction_score"] = round(max(score, 0.0), 1)
        r["conviction_grade"] = _conviction_grade(r["conviction_score"])
        r["accel_t_stat"] = round(r["_accel_t_stat"], 2)
        r["hedging_risk"] = hedging_risk
        r["pcr_check_mode"] = pcr_check_mode

        r["_intensity_norm"] = intensity_norm
        r["_consistency_norm"] = consistency_norm
        r["_accel_norm"] = accel_norm
        r["_mass_norm"] = mass_norm
        r["_oi_change_norm_kept"] = oi_change_norm

        # Wave 0.3 — purpose-built accumulation score (0-100).
        r["accumulation_score"] = _accumulation_score(
            active_days=r["active_days"],
            lookback_days=lookback_days,
            consistency_raw=r["_consistency_raw"],
            accel_t_stat=r["_accel_t_stat"],
            prem_mcap_bps=r["prem_mcap_bps"],
        )

        # Wave 0.1 — per-mode pass flags.  Modes are strict subsets of each
        # other (all ⊇ accumulation ⊇ strong) so the UI can filter client-side.
        r["passes_all"] = _mode_passes(r, FLOW_TRACKER_MODES["all"])
        r["passes_accumulation"] = _mode_passes(r, FLOW_TRACKER_MODES["accumulation"])
        r["passes_strong"] = _mode_passes(r, FLOW_TRACKER_MODES["strong_accumulation"])

        try:
            from app.features.grade_explainer import build_tracker_grade_reasons
            r["grade_reasons"] = build_tracker_grade_reasons(r)
        except Exception:
            r["grade_reasons"] = []

        for _k in (
            "_consistency_raw", "_accel_raw", "_accel_t_stat", "_cum_total",
            "_intensity_norm", "_consistency_norm", "_accel_norm", "_mass_norm",
            "_oi_change_norm", "_oi_change_norm_kept",
            "_dte_multiplier",
        ):
            r.pop(_k, None)

    raw_results.sort(key=lambda x: x["conviction_score"], reverse=True)

    total_qualified = len(raw_results)
    # Funnel counts exposed to the UI for the [Strong][Accum][All] toggle.
    # These count the full population **before** mode hard-filtering so
    # the UI can still show "3 accumulation, 1 strong" even when the
    # caller requests a narrow mode.
    count_strong = sum(1 for r in raw_results if r["passes_strong"])
    count_accum = sum(1 for r in raw_results if r["passes_accumulation"])
    count_all = sum(1 for r in raw_results if r["passes_all"])

    # Flow-Tracker-Swing-Radar: mode hard-filter before the cap.  Legacy
    # callers (mode=None) keep the old "return all, tag per mode" shape.
    if mode and FLOW_TRACKER_HARD_MODE_FILTER:
        mode_key = str(mode).lower()
        mode_flag_map = {
            "all": "passes_all",
            "accumulation": "passes_accumulation",
            "strong_accumulation": "passes_strong",
            "strong": "passes_strong",
        }
        flag = mode_flag_map.get(mode_key)
        if flag is not None:
            raw_results = [r for r in raw_results if r.get(flag)]

    if max_results and len(raw_results) > max_results:
        raw_results = raw_results[:max_results]

    # Wave 0.5 C3 — sector accumulation count.  For each qualifying row,
    # count how many OTHER rows in the same sector AND direction passed
    # the accumulation mode.  Signals sector-wide bid (e.g. "3 energy
    # names accumulating bullish").  Excludes the row itself so lonely
    # sector-leaders show 0, not 1.
    by_sector_dir: dict[tuple[str, str], int] = {}
    for r in raw_results:
        if r.get("passes_accumulation"):
            key = (str(r.get("sector") or "—"), str(r.get("direction") or ""))
            by_sector_dir[key] = by_sector_dir.get(key, 0) + 1

    for r in raw_results:
        r["total_qualified"] = total_qualified
        r["mode_counts"] = {
            "strong_accumulation": count_strong,
            "accumulation": count_accum,
            "all": count_all,
        }
        key = (str(r.get("sector") or "—"), str(r.get("direction") or ""))
        total_in_sector = by_sector_dir.get(key, 0)
        # Exclude self only when this row is itself counted in the sector tally.
        r["sector_accumulating_count"] = (
            total_in_sector - 1 if r.get("passes_accumulation") else total_in_sector
        )

    return raw_results

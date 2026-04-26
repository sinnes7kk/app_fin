"""Sector-level flow heat aggregation.

This module turns the per-ticker flow scores from
``build_flow_feature_table`` into a per-(sector × direction) basket signal.
The goal is to surface coordinated sector runs that single-ticker scoring
misses — e.g. when AVGO + LRCX + MRVL + MU + ARM are all elevated on the
same morning, the basket should light up even when no individual name
breaks the LONG-signal threshold downstream.

Output is informational in v1: a separate CSV per scan plus an
append-only history file. No existing scoring logic is touched.

The heat-score is bounded ~0-10 and combines two terms:

* ``mean_score_topk``  — robust to tail noise; rewards sectors with a
  small handful of very strong names.
* ``share_above_thresh`` — rewards breadth (many names elevated).

Premium concentration is preserved as an inspectable column
(``total_directional_premium``) but deliberately kept out of the score
itself in v1; raw dollars don't normalise across sectors of very
different size and scaling them well needs more history than we have.

Suppression: a sector must have at least
``MIN_TICKERS_FOR_HEAT`` (=3) names in the feature table to be emitted.
This prevents single-name dominance dressed up as a basket signal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import pandas as pd

# Minimum tickers in a sector for the basket signal to fire.
MIN_TICKERS_FOR_HEAT = 3

# Per-side score threshold for "elevated". Scores live on a 0-1 scale
# (see ``flow_features._weighted_flow_score_mixed``); 0.5 maps to the
# scaled-by-10 readout of 5.0 we already eyeball in the trader-card UI.
ELEVATED_SCORE_THRESHOLD = 0.5

# How many of the strongest names per sector feed ``mean_score_topk``.
TOP_K = 5

# Sub-sector overrides. UW's ``sector`` field is the broad GICS bucket
# (Technology, Healthcare, ...) which is too coarse for actionable
# basket trades — Semis, Biotech, Regional Banks, Gold Miners, China
# ADRs all live inside one of those buckets and move on different
# catalysts. The map below pulls them out into named sub-sectors that
# the heat aggregator can rank independently.
#
# Keep this list small and curated. The point is precision on the
# baskets we actually trade against (SOXX/SMH/XBI/KRE/GDX/KWEB), not
# coverage of every sub-industry.
SUB_SECTOR_MAP: dict[str, str] = {
    # Semiconductors (SOXX/SMH constituents that show up in flow most often)
    "NVDA": "Semiconductors",
    "AMD": "Semiconductors",
    "AVGO": "Semiconductors",
    "TSM": "Semiconductors",
    "MU": "Semiconductors",
    "AMAT": "Semiconductors",
    "LRCX": "Semiconductors",
    "KLAC": "Semiconductors",
    "MRVL": "Semiconductors",
    "ARM": "Semiconductors",
    "QCOM": "Semiconductors",
    "TXN": "Semiconductors",
    "INTC": "Semiconductors",
    "ON": "Semiconductors",
    "MCHP": "Semiconductors",
    "ADI": "Semiconductors",
    "NXPI": "Semiconductors",
    "STX": "Semiconductors",
    "WDC": "Semiconductors",
    "SNDK": "Semiconductors",
    "ASML": "Semiconductors",
    "AMKR": "Semiconductors",
    "POET": "Semiconductors",
    "ALAB": "Semiconductors",
    "RMBS": "Semiconductors",
    "SMTC": "Semiconductors",
    "POWI": "Semiconductors",
    "AMBA": "Semiconductors",
    "CRDO": "Semiconductors",
    "GFS": "Semiconductors",
    "AOSL": "Semiconductors",
    "AXTI": "Semiconductors",
    "VECO": "Semiconductors",
    "COHU": "Semiconductors",
    "SOXX": "Semiconductors",
    "SMH": "Semiconductors",
    "SOXL": "Semiconductors",
    # Biotech (XBI flow basket)
    "MRNA": "Biotech",
    "BNTX": "Biotech",
    "VRTX": "Biotech",
    "REGN": "Biotech",
    "GILD": "Biotech",
    "BIIB": "Biotech",
    "INCY": "Biotech",
    "AMGN": "Biotech",
    "ALNY": "Biotech",
    "BMRN": "Biotech",
    "SRPT": "Biotech",
    "ARWR": "Biotech",
    "CRSP": "Biotech",
    "BEAM": "Biotech",
    "EDIT": "Biotech",
    "NTLA": "Biotech",
    "AGIO": "Biotech",
    "HALO": "Biotech",
    "KRYS": "Biotech",
    "SMMT": "Biotech",
    "RCUS": "Biotech",
    "MIRM": "Biotech",
    "AUPH": "Biotech",
    "TARS": "Biotech",
    "CPRX": "Biotech",
    "INBX": "Biotech",
    "EWTX": "Biotech",
    "GHRS": "Biotech",
    "CRVS": "Biotech",
    "TRVI": "Biotech",
    "TSHA": "Biotech",
    "URGN": "Biotech",
    "NKTR": "Biotech",
    "IDYA": "Biotech",
    "ONC": "Biotech",
    "XBI": "Biotech",
    # Regional Banks (KRE basket)
    "ZION": "RegionalBanks",
    "CFR": "RegionalBanks",
    "EWBC": "RegionalBanks",
    "TCBI": "RegionalBanks",
    "FLG": "RegionalBanks",
    "PNFP": "RegionalBanks",
    "OZK": "RegionalBanks",
    "RF": "RegionalBanks",
    "BUSE": "RegionalBanks",
    "OSBC": "RegionalBanks",
    "TFIN": "RegionalBanks",
    "KRE": "RegionalBanks",
    # Gold Miners (GDX basket)
    "GOLD": "GoldMiners",
    "NEM": "GoldMiners",
    "AEM": "GoldMiners",
    "FNV": "GoldMiners",
    "WPM": "GoldMiners",
    "RGLD": "GoldMiners",
    "AU": "GoldMiners",
    "KGC": "GoldMiners",
    "EGO": "GoldMiners",
    "ORLA": "GoldMiners",
    "CGAU": "GoldMiners",
    "GDX": "GoldMiners",
    "GDXJ": "GoldMiners",
    # China ADRs (KWEB basket)
    "BABA": "ChinaADRs",
    "JD": "ChinaADRs",
    "PDD": "ChinaADRs",
    "BIDU": "ChinaADRs",
    "NTES": "ChinaADRs",
    "TME": "ChinaADRs",
    "BILI": "ChinaADRs",
    "TCOM": "ChinaADRs",
    "LI": "ChinaADRs",
    "XPEV": "ChinaADRs",
    "NIO": "ChinaADRs",
    "PONY": "ChinaADRs",
    "KWEB": "ChinaADRs",
    "FXI": "ChinaADRs",
}

# Broad ETF / index instruments that arrive with empty/missing GICS sector
# from UW. Map them to a synthetic "Index" bucket so they don't pollute
# the named sectors but still produce a heat row of their own when several
# index ETFs run together (rare but useful as a regime tell).
ETF_SECTOR_OVERRIDES: dict[str, str] = {
    "SPY": "Index",
    "QQQ": "Index",
    "DIA": "Index",
    "IWM": "Index",
    "RSP": "Index",
    "VTI": "Index",
    "EFA": "Index",
    "EEM": "Index",
    "EWZ": "Index",
    "HYG": "Bonds",
    "TLT": "Bonds",
    "LQD": "Bonds",
    "IBIT": "Crypto",
    "FBTC": "Crypto",
    "ETHE": "Crypto",
    "ETHA": "Crypto",
    "GLD": "GoldMiners",
    "SLV": "GoldMiners",
    "USO": "Energy",
    "XLE": "Energy",
    "XLF": "Financial Services",
    "XLK": "Technology",
    "XLI": "Industrials",
    "XLV": "Healthcare",
    "XLY": "Consumer Cyclical",
    "XLP": "Consumer Defensive",
    "XLU": "Utilities",
    "XLB": "Basic Materials",
    "XLRE": "Real Estate",
    "IGV": "Technology",
}


def _resolve_sector(ticker: str, broad_sector: str | None) -> str | None:
    """Return the sector label to bucket ``ticker`` under.

    Precedence:
    1. ``SUB_SECTOR_MAP`` (curated sub-sector for actionable baskets)
    2. ``ETF_SECTOR_OVERRIDES`` (synthetic buckets for index/bond/crypto ETFs)
    3. Broad GICS sector from the UW screener.

    Returns ``None`` for tickers we can't classify so callers can drop them.
    """
    t = (ticker or "").upper().strip()
    if not t:
        return None
    sub = SUB_SECTOR_MAP.get(t)
    if sub:
        return sub
    etf = ETF_SECTOR_OVERRIDES.get(t)
    if etf:
        return etf
    if broad_sector and isinstance(broad_sector, str) and broad_sector.strip():
        return broad_sector.strip()
    return None


def _build_sector_lookup(
    feature_table: pd.DataFrame,
    screener_meta: Mapping[str, Mapping] | None,
) -> dict[str, str]:
    """Resolve each ticker in ``feature_table`` to a sector label.

    Pulls the broad GICS sector from ``screener_meta`` when available
    (UW's ``sector`` field), then runs every ticker through
    ``_resolve_sector`` so sub-sector overrides win.
    """
    lookup: dict[str, str] = {}
    if "ticker" not in feature_table.columns:
        return lookup
    for raw_t in feature_table["ticker"].astype(str).tolist():
        t = raw_t.upper().strip()
        if not t or t in lookup:
            continue
        broad = None
        if screener_meta:
            meta = screener_meta.get(t)
            if isinstance(meta, Mapping):
                broad = meta.get("sector")
        resolved = _resolve_sector(t, broad)
        if resolved:
            lookup[t] = resolved
    return lookup


def _per_side_aggregate(
    df: pd.DataFrame,
    direction: str,
    score_col: str,
    premium_col: str,
) -> pd.DataFrame:
    """Aggregate one side (bullish or bearish) into per-sector rows."""
    if df.empty or score_col not in df.columns:
        return pd.DataFrame()

    work = df[["ticker", "sector", score_col]].copy()
    work["score"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    if premium_col in df.columns:
        work["premium"] = pd.to_numeric(df[premium_col], errors="coerce").fillna(0.0)
    else:
        work["premium"] = 0.0

    rows: list[dict] = []
    for sector, grp in work.groupby("sector", sort=False):
        n = len(grp)
        if n < MIN_TICKERS_FOR_HEAT:
            continue
        scored = grp.sort_values("score", ascending=False)
        top_k = scored.head(TOP_K)
        n_above = int((scored["score"] >= ELEVATED_SCORE_THRESHOLD).sum())
        mean_top = float(top_k["score"].mean()) if not top_k.empty else 0.0
        max_score = float(scored["score"].max())
        total_prem = float(scored["premium"].sum())
        top_tickers = ",".join(top_k["ticker"].astype(str).tolist())

        # Heat-score: 0.6 * (mean_topk * 10) + 0.4 * (share_above * 10).
        # Both terms land in 0-10; weighted sum stays in 0-10 with the
        # max coming from a sector where every name screams (share=1)
        # AND the top-K all have score >= 1.0 (extremely rare).
        share_above = (n_above / n) if n else 0.0
        heat = round(0.6 * (mean_top * 10.0) + 0.4 * (share_above * 10.0), 3)

        rows.append(
            {
                "sector": sector,
                "direction": direction,
                "n_tickers": n,
                "n_above_thresh": n_above,
                "share_above_thresh": round(share_above, 3),
                "mean_score_topk": round(mean_top, 4),
                "max_score": round(max_score, 4),
                "total_directional_premium": round(total_prem, 2),
                "top_tickers": top_tickers,
                "sector_heat_score": heat,
            }
        )
    return pd.DataFrame(rows)


def compute_sector_heat(
    feature_table: pd.DataFrame,
    screener_meta: Mapping[str, Mapping] | None = None,
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-ticker flow scores into per-(sector × direction) heat rows.

    Parameters
    ----------
    feature_table
        Output of ``build_flow_feature_table`` — must contain at minimum
        ``ticker``, ``bullish_score``, ``bearish_score``. Premium columns
        (``bullish_premium`` / ``bearish_premium``) are optional but recommended.
    screener_meta
        ``{ticker: {...}}`` from ``fetch_stock_screener``. Used purely
        for the broad GICS ``sector`` field; pass ``None`` to rely
        entirely on ``SUB_SECTOR_MAP`` / ``ETF_SECTOR_OVERRIDES``.
    snapshot_date
        ISO date string (``YYYY-MM-DD``) to stamp on every row. Defaults
        to today's UTC date.

    Returns
    -------
    pd.DataFrame
        Columns: ``snapshot_date, sector, direction, n_tickers,
        n_above_thresh, share_above_thresh, mean_score_topk, max_score,
        total_directional_premium, top_tickers, sector_heat_score``.
        Empty frame if the feature table has no usable scores.
    """
    cols = [
        "snapshot_date",
        "sector",
        "direction",
        "n_tickers",
        "n_above_thresh",
        "share_above_thresh",
        "mean_score_topk",
        "max_score",
        "total_directional_premium",
        "top_tickers",
        "sector_heat_score",
    ]
    if feature_table is None or feature_table.empty:
        return pd.DataFrame(columns=cols)

    if snapshot_date is None:
        snapshot_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    sector_lookup = _build_sector_lookup(feature_table, screener_meta)
    if not sector_lookup:
        return pd.DataFrame(columns=cols)

    work = feature_table.copy()
    work["ticker"] = work["ticker"].astype(str).str.upper().str.strip()
    work["sector"] = work["ticker"].map(sector_lookup)
    work = work[work["sector"].notna() & (work["sector"] != "")]
    if work.empty:
        return pd.DataFrame(columns=cols)

    bull = _per_side_aggregate(
        work, direction="bullish",
        score_col="bullish_score", premium_col="bullish_premium",
    )
    bear = _per_side_aggregate(
        work, direction="bearish",
        score_col="bearish_score", premium_col="bearish_premium",
    )
    out = pd.concat([bull, bear], ignore_index=True) if (not bull.empty or not bear.empty) else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=cols)

    out.insert(0, "snapshot_date", snapshot_date)
    out = out[cols].sort_values(
        ["direction", "sector_heat_score"], ascending=[True, False]
    ).reset_index(drop=True)
    return out


def append_sector_heat_history(
    heat_df: pd.DataFrame,
    history_path,
) -> None:
    """Append today's heat rows to the long-running history CSV.

    Idempotent: if a row already exists for the same
    ``(snapshot_date, sector, direction)`` it is replaced rather than
    duplicated. Writes header on first use.
    """
    from pathlib import Path

    path = Path(history_path)
    if heat_df is None or heat_df.empty:
        return

    key_cols = ["snapshot_date", "sector", "direction"]
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame()
        if not existing.empty and all(c in existing.columns for c in key_cols):
            new_keys = set(
                heat_df[key_cols].astype(str).apply(tuple, axis=1).tolist()
            )
            existing_keys = existing[key_cols].astype(str).apply(tuple, axis=1)
            keep_mask = ~existing_keys.isin(new_keys)
            combined = pd.concat(
                [existing[keep_mask], heat_df], ignore_index=True
            )
        else:
            combined = heat_df
    else:
        combined = heat_df

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)

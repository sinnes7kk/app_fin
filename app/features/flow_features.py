"""Flow-derived ticker features from normalized Unusual Whales data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_qualifying_flow(
    df: pd.DataFrame,
    min_premium: float = 500_000,
    min_dte: int = 30,
    max_dte: int = 120,
    require_ask_side: bool = True,
) -> pd.DataFrame:
    """
    Filter normalized flow events down to the ones relevant for the V1 swing system.

    Rules:
    - premium >= min_premium
    - min_dte <= dte <= max_dte
    - option_type is CALL or PUT
    - execution_side == ASK if require_ask_side=True
    """
    if df.empty:
        return df.copy()

    required_cols = ["premium", "dte", "option_type", "execution_side"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for flow filtering: {missing}")

    out = df.copy()

    out = out[
        (out["premium"] >= min_premium)
        & (out["dte"] >= min_dte)
        & (out["dte"] <= max_dte)
        & (out["option_type"].isin(["CALL", "PUT"]))
    ]

    if require_ask_side:
        out = out[out["execution_side"] == "ASK"]

    return out


def add_volume_oi_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume/open_interest ratio.

    If open_interest is missing or zero, ratio is NaN.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if "volume" not in out.columns or "open_interest" not in out.columns:
        out["volume_oi_ratio"] = pd.NA
        return out

    oi = pd.to_numeric(out["open_interest"], errors="coerce")
    vol = pd.to_numeric(out["volume"], errors="coerce")

    out["volume_oi_ratio"] = vol / oi.replace(0, pd.NA)

    return out


def add_repeat_flow_count(
    df: pd.DataFrame,
    group_col: str = "ticker",
) -> pd.DataFrame:
    """
    Add per-row repeat count within the filtered dataset.

    For V1 this is simply the count of qualifying events for that ticker.
    """
    if df.empty:
        return df.copy()

    if group_col not in df.columns:
        raise ValueError(f"Missing grouping column: {group_col}")

    out = df.copy()
    out["repeat_flow_count"] = out.groupby(group_col)[group_col].transform("count")
    return out


def _dte_score(dte: float) -> float:
    """Score DTE quality for swing continuation."""
    if pd.isna(dte):
        return 0.0
    if 30 <= dte <= 90:
        return 1.0
    if 90 < dte <= 150:
        return 0.7
    return 0.3


def aggregate_flow_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate filtered normalized flow to one row per ticker.

    Output includes:
    - direction-specific premiums/counts
    - directional vol/OI
    - sweep counts/premium
    - premium-per-trade metrics
    - concentration metrics
    - DTE quality score
    - directional scores for ranking
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "bullish_premium",
                "bearish_premium",
                "bullish_count",
                "bearish_count",
                "total_premium",
                "total_count",
                "avg_dte",
                "avg_volume_oi_ratio",
                "bullish_vol_oi",
                "bearish_vol_oi",
                "repeat_flow_count",
                "bullish_sweep_count",
                "bearish_sweep_count",
                "bullish_sweep_premium",
                "bearish_sweep_premium",
                "bullish_premium_per_trade",
                "bearish_premium_per_trade",
                "bullish_max_trade_premium",
                "bearish_max_trade_premium",
                "bullish_concentration",
                "bearish_concentration",
                "net_premium_bias",
                "flow_imbalance_ratio",
                "dominant_direction",
                "dte_score",
                "bullish_score",
                "bearish_score",
            ]
        )

    required_cols = ["ticker", "premium", "dte", "direction"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for aggregation: {missing}")

    out = df.copy()

    if "volume_oi_ratio" not in out.columns:
        out = add_volume_oi_ratio(out)

    if "repeat_flow_count" not in out.columns:
        out = add_repeat_flow_count(out)

    if "is_sweep" not in out.columns:
        out["is_sweep"] = False

    # Ensure numeric / boolean stability
    out["premium"] = pd.to_numeric(out["premium"], errors="coerce").fillna(0.0)
    out["dte"] = pd.to_numeric(out["dte"], errors="coerce")
    out["is_sweep"] = out["is_sweep"].fillna(False).astype(bool)

    # Direction-specific components
    out["bullish_premium_component"] = out["premium"].where(out["direction"] == "LONG", 0.0)
    out["bearish_premium_component"] = out["premium"].where(out["direction"] == "SHORT", 0.0)

    out["bullish_count_component"] = (out["direction"] == "LONG").astype(int)
    out["bearish_count_component"] = (out["direction"] == "SHORT").astype(int)

    out["bullish_sweep_component"] = (
        (out["direction"] == "LONG") & (out["is_sweep"])
    ).astype(int)
    out["bearish_sweep_component"] = (
        (out["direction"] == "SHORT") & (out["is_sweep"])
    ).astype(int)

    out["bullish_sweep_premium_component"] = out["premium"].where(
        (out["direction"] == "LONG") & (out["is_sweep"]),
        0.0,
    )
    out["bearish_sweep_premium_component"] = out["premium"].where(
        (out["direction"] == "SHORT") & (out["is_sweep"]),
        0.0,
    )

    out["bullish_vol_oi_component"] = out["volume_oi_ratio"].where(out["direction"] == "LONG", np.nan)
    out["bearish_vol_oi_component"] = out["volume_oi_ratio"].where(out["direction"] == "SHORT", np.nan)

    out["bullish_trade_premium_component"] = out["premium"].where(out["direction"] == "LONG", np.nan)
    out["bearish_trade_premium_component"] = out["premium"].where(out["direction"] == "SHORT", np.nan)

    grouped = out.groupby("ticker", dropna=False)

    agg = grouped.agg(
        bullish_premium=("bullish_premium_component", "sum"),
        bearish_premium=("bearish_premium_component", "sum"),
        bullish_count=("bullish_count_component", "sum"),
        bearish_count=("bearish_count_component", "sum"),
        total_premium=("premium", "sum"),
        total_count=("ticker", "size"),
        avg_dte=("dte", "mean"),
        avg_volume_oi_ratio=("volume_oi_ratio", "mean"),
        bullish_vol_oi=("bullish_vol_oi_component", "mean"),
        bearish_vol_oi=("bearish_vol_oi_component", "mean"),
        repeat_flow_count=("repeat_flow_count", "max"),
        bullish_sweep_count=("bullish_sweep_component", "sum"),
        bearish_sweep_count=("bearish_sweep_component", "sum"),
        bullish_sweep_premium=("bullish_sweep_premium_component", "sum"),
        bearish_sweep_premium=("bearish_sweep_premium_component", "sum"),
        bullish_max_trade_premium=("bullish_trade_premium_component", "max"),
        bearish_max_trade_premium=("bearish_trade_premium_component", "max"),
    ).reset_index()

    # Fill direction-specific nulls
    fill_zero_cols = [
        "bullish_premium",
        "bearish_premium",
        "bullish_count",
        "bearish_count",
        "total_premium",
        "total_count",
        "repeat_flow_count",
        "bullish_sweep_count",
        "bearish_sweep_count",
        "bullish_sweep_premium",
        "bearish_sweep_premium",
        "bullish_max_trade_premium",
        "bearish_max_trade_premium",
    ]
    agg[fill_zero_cols] = agg[fill_zero_cols].fillna(0)

    # Core directional metrics
    agg["net_premium_bias"] = agg["bullish_premium"] - agg["bearish_premium"]
    agg["dominant_direction"] = agg["net_premium_bias"].apply(
        lambda x: "LONG" if x > 0 else ("SHORT" if x < 0 else "NEUTRAL")
    )

    # Premium per trade
    agg["bullish_premium_per_trade"] = np.where(
        agg["bullish_count"] > 0,
        agg["bullish_premium"] / agg["bullish_count"],
        0.0,
    )
    agg["bearish_premium_per_trade"] = np.where(
        agg["bearish_count"] > 0,
        agg["bearish_premium"] / agg["bearish_count"],
        0.0,
    )

    # Concentration: max individual trade premium / total directional premium
    agg["bullish_concentration"] = np.where(
        agg["bullish_premium"] > 0,
        agg["bullish_max_trade_premium"] / agg["bullish_premium"],
        0.0,
    )
    agg["bearish_concentration"] = np.where(
        agg["bearish_premium"] > 0,
        agg["bearish_max_trade_premium"] / agg["bearish_premium"],
        0.0,
    )

    # Flow imbalance ratio
    agg["flow_imbalance_ratio"] = np.where(
        agg["bearish_premium"] > 0,
        agg["bullish_premium"] / agg["bearish_premium"],
        np.where(agg["bullish_premium"] > 0, np.inf, 1.0),
    )

    # DTE score
    agg["dte_score"] = agg["avg_dte"].apply(_dte_score)

    # Round selected columns
    round_cols = [
        "avg_dte",
        "avg_volume_oi_ratio",
        "bullish_vol_oi",
        "bearish_vol_oi",
        "bullish_concentration",
        "bearish_concentration",
        "dte_score",
    ]
    for col in round_cols:
        if col in agg.columns:
            agg[col] = agg[col].round(2)

    # Directional scores
    # Favor:
    # - larger but log-scaled directional premium
    # - higher directional vol/oi
    # - repeat flow
    # - sweep participation
    # - good DTE
    # - larger premium per trade
    # - moderate concentration (single huge print helpful, but not everything)
    agg["bullish_score"] = (
        0.35 * np.log1p(agg["bullish_premium"])
        + 0.15 * np.log1p(agg["bullish_premium_per_trade"])
        + 0.20 * agg["bullish_vol_oi"].fillna(0)
        + 0.15 * agg["repeat_flow_count"]
        + 0.10 * agg["bullish_sweep_count"]
        + 0.05 * agg["dte_score"]
        + 0.05 * agg["bullish_concentration"]
    )

    agg["bearish_score"] = (
        0.35 * np.log1p(agg["bearish_premium"])
        + 0.15 * np.log1p(agg["bearish_premium_per_trade"])
        + 0.20 * agg["bearish_vol_oi"].fillna(0)
        + 0.15 * agg["repeat_flow_count"]
        + 0.10 * agg["bearish_sweep_count"]
        + 0.05 * agg["dte_score"]
        + 0.05 * agg["bearish_concentration"]
    )

    return agg


def rank_flow_candidates(
    agg_df: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Rank bullish and bearish candidates separately from aggregated ticker flow.

    Uses directional score first, then premium and sweep metrics as tie-breakers.
    """
    if agg_df.empty:
        empty = agg_df.copy()
        return {"bullish": empty, "bearish": empty}

    bullish = agg_df[agg_df["bullish_premium"] > 0].copy()
    bearish = agg_df[agg_df["bearish_premium"] > 0].copy()

    bullish = bullish.sort_values(
        by=[
            "bullish_score",
            "bullish_premium",
            "bullish_vol_oi",
            "bullish_sweep_premium",
        ],
        ascending=[False, False, False, False],
    ).head(top_n)

    bearish = bearish.sort_values(
        by=[
            "bearish_score",
            "bearish_premium",
            "bearish_vol_oi",
            "bearish_sweep_premium",
        ],
        ascending=[False, False, False, False],
    ).head(top_n)

    return {
        "bullish": bullish.reset_index(drop=True),
        "bearish": bearish.reset_index(drop=True),
    }


def build_flow_feature_table(
    normalized_df: pd.DataFrame,
    min_premium: float = 500_000,
    min_dte: int = 30,
    max_dte: int = 120,
    require_ask_side: bool = True,
) -> pd.DataFrame:
    """
    Full V1 pipeline for flow features:
    1. filter qualifying flow
    2. add derived metrics
    3. aggregate by ticker
    """
    filtered = filter_qualifying_flow(
        normalized_df,
        min_premium=min_premium,
        min_dte=min_dte,
        max_dte=max_dte,
        require_ask_side=require_ask_side,
    )

    filtered = add_volume_oi_ratio(filtered)
    filtered = add_repeat_flow_count(filtered)

    agg = aggregate_flow_by_ticker(filtered)
    return agg
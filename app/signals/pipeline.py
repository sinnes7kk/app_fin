"""End-to-end pipeline: flow candidates -> price validation -> final ranked setups."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.features.flow_features import build_flow_feature_table, rank_flow_candidates
from app.features.price_features import clean_ohlcv, compute_features, fetch_ohlcv
from app.signals.scoring import score_long_setup, score_short_setup
from app.signals.trade_plan import build_long_trade_plan, build_short_trade_plan
from app.vendors.unusual_whales import (
    fetch_flow_for_tickers,
    fetch_flow_raw,
    fetch_uw_alerts,
    normalize_flow_response,
)


def minmax_scale(series: pd.Series, target_max: float = 10.0) -> pd.Series:
    """Scale a Series to 0–target_max via min-max normalization."""
    min_v, max_v = series.min(), series.max()
    if max_v == min_v:
        return series.apply(lambda _: target_max / 2)
    return target_max * (series - min_v) / (max_v - min_v)


def combine_scores(flow_score: float, price_score: float) -> float:
    """Combine flow conviction (0-10) and price conviction (0-10) into one final score."""
    return round(0.6 * flow_score + 0.4 * price_score, 4)


def has_strong_bullish_flow(row, min_ratio: float = 1.5) -> bool:
    return row["bullish_premium"] > row["bearish_premium"] * min_ratio


def has_strong_bearish_flow(row, min_ratio: float = 1.5) -> bool:
    return row["bearish_premium"] > row["bullish_premium"] * min_ratio


def dedupe_final_results(results: list[dict]) -> list[dict]:
    """Keep only the highest-scoring direction per ticker."""
    best_by_ticker: dict[str, dict] = {}
    for result in results:
        ticker = result["ticker"]
        if ticker not in best_by_ticker or result["final_score"] > best_by_ticker[ticker]["final_score"]:
            best_by_ticker[ticker] = result
    return list(best_by_ticker.values())


LONG_ALL_REASONS = {
    "trend_aligned", "pullback_to_support",
    "bullish_strong_close", "bullish_rejection_wick",
    "healthy_pullback_volume", "confirmation_volume",
}
SHORT_ALL_REASONS = {
    "trend_aligned", "pullback_to_resistance",
    "bearish_strong_close", "bearish_rejection_wick",
    "healthy_pullback_volume", "confirmation_volume",
}


def _build_rejection_row(
    ticker: str,
    direction: str,
    flow_score_raw: float,
    price_signal: dict,
    all_reasons: set[str],
    reject_reason: str = "price_validation_failed",
    flow_score_scaled: float | None = None,
) -> dict:
    passed = set(price_signal.get("reasons", []))
    failed = sorted(all_reasons - passed)
    return {
        "ticker": ticker,
        "direction": direction,
        "flow_score_raw": flow_score_raw,
        "flow_score_scaled": flow_score_scaled if flow_score_scaled is not None else flow_score_raw,
        "price_score": price_signal.get("score", 0),
        "reject_reason": reject_reason,
        "checks_passed": ", ".join(sorted(passed)) or "none",
        "checks_failed": ", ".join(failed) or "none",
    }


def run_price_validation_for_bullish_candidates(bullish_df) -> tuple[list[dict], list[dict]]:
    """Run long-side price validation. Returns (accepted, rejected)."""
    if bullish_df.empty:
        return [], []

    accepted: list[dict] = []
    rejected: list[dict] = []

    for _, row in bullish_df.iterrows():
        ticker = row["ticker"]
        flow_scaled = float(row["bullish_score"])
        flow_raw = float(row.get("bullish_score_raw", flow_scaled))

        if not has_strong_bullish_flow(row):
            rejected.append(_build_rejection_row(
                ticker, "LONG", flow_raw, {}, LONG_ALL_REASONS,
                reject_reason="weak_bullish_flow",
                flow_score_scaled=flow_scaled,
            ))
            continue

        try:
            df = fetch_ohlcv(ticker)
            df = clean_ohlcv(df)
            df = compute_features(df)

            price_signal = score_long_setup(df)
            price_signal["ticker"] = ticker

            if not price_signal["is_valid"]:
                rejected.append(_build_rejection_row(
                    ticker, "LONG", flow_raw, price_signal, LONG_ALL_REASONS,
                    flow_score_scaled=flow_scaled,
                ))
                continue

            trade_plan = build_long_trade_plan(df, price_signal)

            accepted.append({
                "ticker": ticker,
                "direction": "LONG",
                "flow_score_raw": flow_raw,
                "flow_score_scaled": flow_scaled,
                "price_score": float(price_signal["score"]),
                "final_score": combine_scores(flow_scaled, float(price_signal["score"])),
                "entry_price": trade_plan["entry_price"],
                "stop_price": trade_plan["stop_price"],
                "target_1": trade_plan["target_1"],
                "target_2": trade_plan["target_2"],
                "time_stop_days": trade_plan["time_stop_days"],
                "source": "fresh",
                "trade_plan": trade_plan,
                "flow_snapshot": row.to_dict(),
                "price_snapshot": price_signal,
            })

        except Exception as e:
            rejected.append(_build_rejection_row(
                ticker, "LONG", flow_raw, {}, LONG_ALL_REASONS,
                reject_reason=f"error: {e}",
                flow_score_scaled=flow_scaled,
            ))

    return accepted, rejected


def run_price_validation_for_bearish_candidates(bearish_df) -> tuple[list[dict], list[dict]]:
    """Run short-side price validation. Returns (accepted, rejected)."""
    if bearish_df.empty:
        return [], []

    accepted: list[dict] = []
    rejected: list[dict] = []

    for _, row in bearish_df.iterrows():
        ticker = row["ticker"]
        flow_scaled = float(row["bearish_score"])
        flow_raw = float(row.get("bearish_score_raw", flow_scaled))

        if not has_strong_bearish_flow(row):
            rejected.append(_build_rejection_row(
                ticker, "SHORT", flow_raw, {}, SHORT_ALL_REASONS,
                reject_reason="weak_bearish_flow",
                flow_score_scaled=flow_scaled,
            ))
            continue

        try:
            df = fetch_ohlcv(ticker)
            df = clean_ohlcv(df)
            df = compute_features(df)

            price_signal = score_short_setup(df)
            price_signal["ticker"] = ticker

            if not price_signal["is_valid"]:
                rejected.append(_build_rejection_row(
                    ticker, "SHORT", flow_raw, price_signal, SHORT_ALL_REASONS,
                    flow_score_scaled=flow_scaled,
                ))
                continue

            trade_plan = build_short_trade_plan(df, price_signal)

            accepted.append({
                "ticker": ticker,
                "direction": "SHORT",
                "flow_score_raw": flow_raw,
                "flow_score_scaled": flow_scaled,
                "price_score": float(price_signal["score"]),
                "final_score": combine_scores(flow_scaled, float(price_signal["score"])),
                "entry_price": trade_plan["entry_price"],
                "stop_price": trade_plan["stop_price"],
                "target_1": trade_plan["target_1"],
                "target_2": trade_plan["target_2"],
                "time_stop_days": trade_plan["time_stop_days"],
                "source": "fresh",
                "trade_plan": trade_plan,
                "flow_snapshot": row.to_dict(),
                "price_snapshot": price_signal,
            })

        except Exception as e:
            rejected.append(_build_rejection_row(
                ticker, "SHORT", flow_raw, {}, SHORT_ALL_REASONS,
                reject_reason=f"error: {e}",
                flow_score_scaled=flow_scaled,
            ))

    return accepted, rejected


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def _run_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_run_outputs(
    feature_table: pd.DataFrame,
    bullish_ranked: pd.DataFrame,
    bearish_ranked: pd.DataFrame,
    signals_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    stamp: str | None = None,
) -> dict[str, Path]:
    """Save all intermediate, final, and rejected DataFrames to CSV under data/."""
    stamp = stamp or _run_stamp()
    paths: dict[str, Path] = {}

    for name, df, subdir in [
        ("flow_features", feature_table, "flow_features"),
        ("ranked_bullish", bullish_ranked, "ranked_candidates"),
        ("ranked_bearish", bearish_ranked, "ranked_candidates"),
        ("final_signals", signals_df, "final_signals"),
        ("rejected", rejected_df, "final_signals"),
    ]:
        out_dir = DATA_ROOT / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}_{stamp}.csv"
        df.to_csv(path, index=False)
        paths[name] = path

    return paths


def run_flow_to_price_pipeline(
    flow_limit: int = 500,
    top_n: int = 10,
    min_premium: float = 500_000,
    save: bool = True,
    use_uw_alerts: bool = True,
    alert_hours_back: int = 24,
) -> dict:
    """
    Full V1 pipeline:
    1. Load + prune persistent watchlist
    2. Pull raw UW flow + UW curated alert tickers
    3. Normalize + build flow features
    4. Rank bullish/bearish candidates
    5. Scale flow scores, run price validation
    6. Re-evaluate watchlist entries (frozen flow, fresh price)
    7. Merge fresh + promoted signals, dedupe
    8. Update watchlist with newly rejected, save to disk
    """
    from app.signals.watchlist import (
        add_candidates,
        load_watchlist,
        prune_expired,
        reevaluate_watchlist,
        save_watchlist,
    )

    prev_watchlist = load_watchlist()
    active_watchlist, expired_watchlist = prune_expired(prev_watchlist)

    payload = fetch_flow_raw(limit=flow_limit)
    normalized = normalize_flow_response(payload)

    alert_stats = {"alert_tickers": 0, "new_tickers": 0}
    if use_uw_alerts:
        alert_tickers = fetch_uw_alerts(hours_back=alert_hours_back)
        alert_stats["alert_tickers"] = len(alert_tickers)

        existing_tickers = set(normalized["ticker"].unique()) if not normalized.empty else set()
        new_tickers = [t for t in alert_tickers if t not in existing_tickers]
        alert_stats["new_tickers"] = len(new_tickers)

        if new_tickers:
            alert_flow = fetch_flow_for_tickers(new_tickers)
            if not alert_flow.empty:
                normalized = pd.concat([normalized, alert_flow], ignore_index=True)

    feature_table = build_flow_feature_table(normalized, min_premium=min_premium)
    ranked = rank_flow_candidates(feature_table, top_n=top_n)

    if not ranked["bullish"].empty:
        ranked["bullish"]["bullish_score_raw"] = ranked["bullish"]["bullish_score"]
        ranked["bullish"]["bullish_score"] = minmax_scale(ranked["bullish"]["bullish_score_raw"])
    if not ranked["bearish"].empty:
        ranked["bearish"]["bearish_score_raw"] = ranked["bearish"]["bearish_score"]
        ranked["bearish"]["bearish_score"] = minmax_scale(ranked["bearish"]["bearish_score_raw"])

    bull_accepted, bull_rejected = run_price_validation_for_bullish_candidates(ranked["bullish"])
    bear_accepted, bear_rejected = run_price_validation_for_bearish_candidates(ranked["bearish"])

    fresh_results = bull_accepted + bear_accepted
    all_rejected = bull_rejected + bear_rejected

    fresh_tickers = {(r["ticker"], r["direction"]) for r in fresh_results + all_rejected}
    promoted, still_watching, watch_rejected = reevaluate_watchlist(
        active_watchlist, fresh_tickers=fresh_tickers,
    )

    final_results = fresh_results + promoted
    final_results = dedupe_final_results(final_results)
    final_results = sorted(final_results, key=lambda x: x["final_score"], reverse=True)

    signals_df = results_to_dataframe(final_results)

    all_rejected_combined = all_rejected + watch_rejected
    rejected_df = pd.DataFrame(all_rejected_combined) if all_rejected_combined else pd.DataFrame(
        columns=REJECTED_COLUMNS
    )

    updated_watchlist = add_candidates(still_watching, all_rejected)
    save_watchlist(updated_watchlist)

    saved_paths = {}
    if save:
        saved_paths = save_run_outputs(
            feature_table=feature_table,
            bullish_ranked=ranked["bullish"],
            bearish_ranked=ranked["bearish"],
            signals_df=signals_df,
            rejected_df=rejected_df,
        )
        for name, path in saved_paths.items():
            print(f"  saved {name} -> {path}")

    return {
        "results": final_results,
        "signals_df": signals_df,
        "rejected_df": rejected_df,
        "feature_table": feature_table,
        "ranked_bullish": ranked["bullish"],
        "ranked_bearish": ranked["bearish"],
        "saved_paths": saved_paths,
        "watchlist": {
            "previous_count": len(prev_watchlist),
            "expired_count": len(expired_watchlist),
            "promoted_count": len(promoted),
            "still_watching_count": len(still_watching),
            "new_rejects_added": len(all_rejected),
            "current_count": len(updated_watchlist),
        },
        "alert_stats": alert_stats,
    }


REJECTED_COLUMNS = [
    "ticker",
    "direction",
    "flow_score_raw",
    "price_score",
    "reject_reason",
    "checks_passed",
    "checks_failed",
]

SIGNAL_COLUMNS = [
    "ticker",
    "direction",
    "flow_score_raw",
    "flow_score_scaled",
    "price_score",
    "final_score",
    "entry_price",
    "stop_price",
    "target_1",
    "target_2",
    "time_stop_days",
    "source",
]


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert pipeline results into a clean signals DataFrame."""
    if not results:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return pd.DataFrame(results)[SIGNAL_COLUMNS]
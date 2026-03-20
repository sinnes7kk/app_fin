"""Replay backtester — steps through archived flow snapshots day by day."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from app.backtest.price_loader import PriceCache
from app.config import (
    MAX_HOLD_DAYS,
    MAX_PORTFOLIO_HEAT,
    MAX_POSITIONS,
    MIN_FINAL_SCORE,
    PARTIAL_EXIT_PCT,
    PORTFOLIO_CAPITAL,
    ROTATION_SCORE_MARGIN,
    SIZING_TIERS,
)
from app.features.flow_features import build_flow_feature_table, rank_flow_candidates
from app.signals.pipeline import (
    combine_scores,
    has_strong_bearish_flow,
    has_strong_bullish_flow,
    minmax_scale,
)
from app.signals.positions import (
    TRADE_LOG_COLUMNS,
    _check_exits,
    _check_partial,
    _identify_exit_trail,
    close_position,
)
from app.signals.scoring import score_long_setup, score_short_setup
from app.signals.trade_plan import (
    MIN_RR,
    build_long_trade_plan,
    build_short_trade_plan,
    compute_trailing_stops,
)

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
RAW_FLOW_DIR = DATA_ROOT / "raw_flow"


@dataclass
class BacktestResult:
    trade_log: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _load_flow_snapshots(start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load archived raw flow CSVs, grouped by date. Latest snapshot per day wins."""
    snapshots: dict[str, pd.DataFrame] = {}
    if not RAW_FLOW_DIR.exists():
        return snapshots

    for path in sorted(RAW_FLOW_DIR.glob("raw_flow_*.csv")):
        stem = path.stem.replace("raw_flow_", "")
        try:
            file_date = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
        except (IndexError, ValueError):
            continue

        if file_date < start or file_date > end:
            continue

        try:
            df = pd.read_csv(path)
            if not df.empty:
                if "event_ts" in df.columns:
                    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
                if "expiration_date" in df.columns:
                    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce")
                for col in df.columns:
                    if col in ("ticker", "option_type", "alert_rule",
                               "execution_side", "direction", "event_ts",
                               "expiration_date"):
                        continue
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                snapshots[file_date] = df
        except Exception:
            continue

    return snapshots


def _all_tickers_in_snapshots(snapshots: dict[str, pd.DataFrame]) -> list[str]:
    """Collect all unique tickers across all snapshots."""
    tickers: set[str] = set()
    for df in snapshots.values():
        if "ticker" in df.columns:
            tickers.update(df["ticker"].dropna().unique())
    return sorted(tickers)


def _risk_pct_for_score(score: float) -> float:
    for threshold, pct in SIZING_TIERS:
        if score >= threshold:
            return pct
    return 0.0


def _current_heat(positions: list[dict]) -> float:
    return sum(p.get("risk_dollar", 0.0) for p in positions) / PORTFOLIO_CAPITAL


def _build_position(sig: dict, risk_pct: float, entry_date: str) -> dict | None:
    risk_per_share = (
        round(sig["entry_price"] - sig["stop_price"], 2)
        if sig["direction"] == "LONG"
        else round(sig["stop_price"] - sig["entry_price"], 2)
    )
    if risk_per_share <= 0:
        return None

    risk_dollar = PORTFOLIO_CAPITAL * risk_pct
    shares = int(risk_dollar / risk_per_share)
    if shares <= 0:
        return None

    position_value = shares * sig["entry_price"]
    actual_risk_dollar = shares * risk_per_share

    return {
        "ticker": sig["ticker"],
        "direction": sig["direction"],
        "entry_price": sig["entry_price"],
        "entry_date": entry_date,
        "initial_stop": sig["stop_price"],
        "risk_per_share": risk_per_share,
        "best_price": sig["entry_price"],
        "days_held": 0,
        "trail_atr": sig["stop_price"],
        "trail_ema": sig["stop_price"],
        "trail_hybrid": sig["stop_price"],
        "active_stop": sig["stop_price"],
        "target_1": sig["target_1"],
        "target_2": sig["target_2"],
        "partial_filled": False,
        "final_score": sig["final_score"],
        "flow_score_scaled": sig.get("flow_score_scaled", 0.0),
        "source": "backtest",
        "partial_pnl_pct": 0.0,
        "risk_pct": risk_pct,
        "risk_dollar": round(actual_risk_dollar, 2),
        "shares": shares,
        "position_value": round(position_value, 2),
    }


def _unrealized_r(pos: dict) -> float:
    entry = pos["entry_price"]
    risk = pos["risk_per_share"]
    if risk <= 0:
        return 0.0
    best = pos.get("best_price", entry)
    if pos["direction"] == "LONG":
        return (best - entry) / risk
    return (entry - best) / risk


def _find_replaceable(existing: list[dict], new_score: float) -> dict | None:
    candidates = []
    for pos in existing:
        if pos.get("partial_filled", False):
            continue
        ur = _unrealized_r(pos)
        if ur <= 0.0:
            candidates.append(pos)
        elif new_score >= pos["final_score"] + ROTATION_SCORE_MARGIN:
            candidates.append(pos)
    if not candidates:
        return None
    return min(candidates, key=lambda p: p["final_score"])


def _close_position_bt(pos: dict, exit_price: float, exit_reason: str,
                        exit_date: str, trail_method: str = "") -> dict:
    """close_position variant that accepts an explicit exit_date."""
    entry = pos["entry_price"]
    risk = pos["risk_per_share"]
    direction = pos["direction"]
    shares = pos.get("shares", 0)

    if direction == "LONG":
        pnl_pct = (exit_price - entry) / entry
        r_multiple = (exit_price - entry) / risk if risk > 0 else 0.0
        pnl_dollar = (exit_price - entry) * shares
    else:
        pnl_pct = (entry - exit_price) / entry
        r_multiple = (entry - exit_price) / risk if risk > 0 else 0.0
        pnl_dollar = (entry - exit_price) * shares

    return {
        "ticker": pos["ticker"],
        "direction": direction,
        "entry_date": pos["entry_date"],
        "exit_date": exit_date,
        "entry_price": entry,
        "exit_price": round(exit_price, 2),
        "shares": shares,
        "risk_pct": pos.get("risk_pct", 0.0),
        "pnl_pct": round(pnl_pct, 4),
        "pnl_dollar": round(pnl_dollar, 2),
        "r_multiple": round(r_multiple, 2),
        "days_held": pos["days_held"],
        "exit_reason": exit_reason,
        "trail_method": trail_method,
        "final_score": pos["final_score"],
        "flow_score_scaled": pos["flow_score_scaled"],
        "partial_pnl_pct": round(pos.get("partial_pnl_pct", 0.0), 4),
    }


def _generate_signals(flow_df: pd.DataFrame, price_cache: PriceCache,
                       as_of: str, top_n: int = 20,
                       min_premium: float = 50_000) -> list[dict]:
    """Run the flow → score → trade plan pipeline for a single day."""
    feature_table = build_flow_feature_table(flow_df, min_premium=min_premium)
    ranked = rank_flow_candidates(feature_table, top_n=top_n)

    if not ranked["bullish"].empty:
        ranked["bullish"]["bullish_score_raw"] = ranked["bullish"]["bullish_score"]
        ranked["bullish"]["bullish_score"] = minmax_scale(ranked["bullish"]["bullish_score_raw"])
    if not ranked["bearish"].empty:
        ranked["bearish"]["bearish_score_raw"] = ranked["bearish"]["bearish_score"]
        ranked["bearish"]["bearish_score"] = minmax_scale(ranked["bearish"]["bearish_score_raw"])

    signals: list[dict] = []

    for _, row in ranked["bullish"].iterrows():
        ticker = row["ticker"]
        flow_scaled = float(row["bullish_score"])
        flow_raw = float(row.get("bullish_score_raw", flow_scaled))

        if not has_strong_bullish_flow(row):
            continue

        df = price_cache.get_as_of(ticker, as_of)
        if df is None:
            continue

        try:
            price_signal = score_long_setup(df)
            price_signal["ticker"] = ticker
            if not price_signal["is_valid"]:
                continue

            trade_plan = build_long_trade_plan(df, price_signal)
            if trade_plan["rr_ratio"] < MIN_RR:
                continue

            signals.append({
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
                "rr_ratio": trade_plan["rr_ratio"],
                "time_stop_days": trade_plan["time_stop_days"],
                "source": "backtest",
            })
        except Exception:
            continue

    for _, row in ranked["bearish"].iterrows():
        ticker = row["ticker"]
        flow_scaled = float(row["bearish_score"])
        flow_raw = float(row.get("bearish_score_raw", flow_scaled))

        if not has_strong_bearish_flow(row):
            continue

        df = price_cache.get_as_of(ticker, as_of)
        if df is None:
            continue

        try:
            price_signal = score_short_setup(df)
            price_signal["ticker"] = ticker
            if not price_signal["is_valid"]:
                continue

            trade_plan = build_short_trade_plan(df, price_signal)
            if trade_plan["rr_ratio"] < MIN_RR:
                continue

            signals.append({
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
                "rr_ratio": trade_plan["rr_ratio"],
                "time_stop_days": trade_plan["time_stop_days"],
                "source": "backtest",
            })
        except Exception:
            continue

    signals.sort(key=lambda s: s["final_score"], reverse=True)
    return signals


def _open_signals(signals: list[dict], positions: list[dict],
                   current_date: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Open new positions from signals. Returns (opened, rotated_out, updated_positions)."""
    open_tickers = {(p["ticker"], p["direction"]) for p in positions}
    heat = _current_heat(positions)

    opened: list[dict] = []
    rotated_out: list[dict] = []

    for sig in signals:
        key = (sig["ticker"], sig["direction"])
        if key in open_tickers:
            continue

        score = sig["final_score"]
        if score < MIN_FINAL_SCORE:
            continue

        risk_pct = _risk_pct_for_score(score)
        pos = _build_position(sig, risk_pct, current_date)
        if pos is None:
            continue

        new_heat = pos["risk_dollar"] / PORTFOLIO_CAPITAL
        book_full = len(positions) >= MAX_POSITIONS

        if book_full:
            victim = _find_replaceable(positions, score)
            if victim is None:
                continue
            exit_price = victim.get("best_price", victim["entry_price"])
            log_row = _close_position_bt(victim, exit_price, "rotated_out", current_date)
            rotated_out.append(log_row)
            heat -= victim.get("risk_dollar", 0.0) / PORTFOLIO_CAPITAL
            positions = [p for p in positions if not (
                p["ticker"] == victim["ticker"] and p["direction"] == victim["direction"]
            )]
            open_tickers.discard((victim["ticker"], victim["direction"]))

        if heat + new_heat > MAX_PORTFOLIO_HEAT:
            continue

        heat += new_heat
        opened.append(pos)
        positions.append(pos)
        open_tickers.add(key)

    return opened, rotated_out, positions


def _update_positions(positions: list[dict], price_cache: PriceCache,
                       current_date: str) -> tuple[list[dict], list[dict]]:
    """Daily position update. Returns (still_open, closed_rows)."""
    still_open: list[dict] = []
    closed_rows: list[dict] = []

    for pos in positions:
        ticker = pos["ticker"]
        df = price_cache.get_as_of(ticker, current_date)
        if df is None:
            still_open.append(pos)
            continue

        bar = df.iloc[-1]
        pos["days_held"] += 1

        trail_update = compute_trailing_stops(pos, df)
        pos.update(trail_update)

        if pos["entry_date"] == current_date:
            still_open.append(pos)
            continue

        partial_triggered, partial_price = _check_partial(pos, bar)
        if partial_triggered:
            entry = pos["entry_price"]
            if pos["direction"] == "LONG":
                partial_pnl = (partial_price - entry) / entry
            else:
                partial_pnl = (entry - partial_price) / entry
            pos["partial_filled"] = True
            pos["partial_pnl_pct"] = round(partial_pnl * PARTIAL_EXIT_PCT, 4)

        exit_reason, exit_price, trail_method = _check_exits(pos, bar)
        if exit_reason:
            log_row = _close_position_bt(pos, exit_price, exit_reason, current_date, trail_method)
            closed_rows.append(log_row)
        else:
            still_open.append(pos)

    return still_open, closed_rows


def _compute_equity(positions: list[dict], price_cache: PriceCache,
                     current_date: str, realized_pnl: float) -> float:
    """Compute total portfolio value: capital + realized P&L + unrealized P&L."""
    unrealized = 0.0
    for pos in positions:
        df = price_cache.get_as_of(pos["ticker"], current_date)
        if df is None:
            continue
        close = float(df.iloc[-1]["close"])
        shares = pos.get("shares", 0)
        if pos["direction"] == "LONG":
            unrealized += (close - pos["entry_price"]) * shares
        else:
            unrealized += (pos["entry_price"] - close) * shares
    return PORTFOLIO_CAPITAL + realized_pnl + unrealized


def _compute_summary(trade_log: list[dict], equity_curve: list[dict]) -> dict:
    if not trade_log:
        return {"total_trades": 0, "note": "no trades executed"}

    total = len(trade_log)
    winners = [t for t in trade_log if t["pnl_pct"] > 0]
    losers = [t for t in trade_log if t["pnl_pct"] <= 0]
    win_rate = len(winners) / total if total else 0

    total_pnl = sum(t["pnl_dollar"] for t in trade_log)
    avg_r = sum(t["r_multiple"] for t in trade_log) / total
    avg_winner_r = (sum(t["r_multiple"] for t in winners) / len(winners)) if winners else 0
    avg_loser_r = (sum(t["r_multiple"] for t in losers) / len(losers)) if losers else 0

    max_dd = 0.0
    peak = PORTFOLIO_CAPITAL
    for pt in equity_curve:
        val = pt["equity"]
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "total_trades": total,
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": round(win_rate, 3),
        "total_pnl_dollar": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / PORTFOLIO_CAPITAL, 4),
        "avg_r_multiple": round(avg_r, 2),
        "avg_winner_r": round(avg_winner_r, 2),
        "avg_loser_r": round(avg_loser_r, 2),
        "max_drawdown_pct": round(max_dd, 4),
        "days_tested": len(equity_curve),
    }


def run_backtest(
    start_date: str,
    end_date: str,
    capital: float = PORTFOLIO_CAPITAL,
) -> BacktestResult:
    """Run a full replay backtest over archived flow snapshots."""
    print(f"Loading flow snapshots from {start_date} to {end_date}...")
    snapshots = _load_flow_snapshots(start_date, end_date)

    if not snapshots:
        print("  No archived flow snapshots found in date range.")
        print(f"  Run the live pipeline first to populate data/raw_flow/")
        return BacktestResult(summary={"error": "no flow snapshots available"})

    dates = sorted(snapshots.keys())
    print(f"  Found {len(dates)} trading day(s) with flow data: {dates[0]} → {dates[-1]}")

    all_tickers = _all_tickers_in_snapshots(snapshots)
    print(f"  Pre-fetching OHLCV for {len(all_tickers)} tickers...")
    price_cache = PriceCache()
    price_cache.prefetch(all_tickers)

    positions: list[dict] = []
    trade_log: list[dict] = []
    equity_curve: list[dict] = []
    realized_pnl = 0.0

    for day_date in dates:
        flow_df = snapshots[day_date]
        print(f"\n  === Day: {day_date} ===")

        signals = _generate_signals(flow_df, price_cache, day_date)
        print(f"    Signals generated: {len(signals)}")

        opened, rotated, positions = _open_signals(signals, positions, day_date)
        trade_log.extend(rotated)
        realized_pnl += sum(r["pnl_dollar"] for r in rotated)

        if opened:
            print(f"    Opened: {', '.join(p['ticker'] for p in opened)}")
        if rotated:
            print(f"    Rotated out: {', '.join(r['ticker'] for r in rotated)}")

        positions, closed = _update_positions(positions, price_cache, day_date)
        trade_log.extend(closed)
        realized_pnl += sum(c["pnl_dollar"] for c in closed)

        if closed:
            for c in closed:
                print(f"    Closed {c['ticker']} {c['direction']} {c['pnl_pct']:+.2%} "
                      f"({c['r_multiple']:+.1f}R) — {c['exit_reason']}")

        equity = _compute_equity(positions, price_cache, day_date, realized_pnl)
        equity_curve.append({"date": day_date, "equity": round(equity, 2),
                             "open_positions": len(positions)})
        print(f"    Portfolio: ${equity:,.2f}  ({len(positions)} open)")

    # Force-close remaining positions at last available price
    if positions:
        last_date = dates[-1]
        print(f"\n  Force-closing {len(positions)} remaining position(s) at end of backtest...")
        for pos in positions:
            df = price_cache.get_as_of(pos["ticker"], last_date)
            if df is not None:
                exit_price = float(df.iloc[-1]["close"])
            else:
                exit_price = pos["entry_price"]
            log_row = _close_position_bt(pos, exit_price, "backtest_end", last_date)
            trade_log.append(log_row)
            realized_pnl += log_row["pnl_dollar"]

    summary = _compute_summary(trade_log, equity_curve)
    return BacktestResult(trade_log=trade_log, equity_curve=equity_curve, summary=summary)


def save_backtest_results(result: BacktestResult) -> Path:
    """Save backtest trade log to data/backtest/."""
    out_dir = DATA_ROOT / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = date.today().strftime("%Y%m%d")
    path = out_dir / f"backtest_{stamp}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result.trade_log)

    return path


def print_summary(result: BacktestResult) -> None:
    s = result.summary
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    if "error" in s:
        print(f"  {s['error']}")
        return

    print(f"  Days tested:      {s['days_tested']}")
    print(f"  Total trades:     {s['total_trades']}")
    print(f"  Winners:          {s['winners']}  ({s['win_rate']:.0%})")
    print(f"  Losers:           {s['losers']}")
    print(f"  Total P&L:        ${s['total_pnl_dollar']:+,.2f}  ({s['total_pnl_pct']:+.2%})")
    print(f"  Avg R-multiple:   {s['avg_r_multiple']:+.2f}R")
    print(f"  Avg winner:       {s['avg_winner_r']:+.2f}R")
    print(f"  Avg loser:        {s['avg_loser_r']:+.2f}R")
    print(f"  Max drawdown:     {s['max_drawdown_pct']:.2%}")

    if result.equity_curve:
        first = result.equity_curve[0]["equity"]
        last = result.equity_curve[-1]["equity"]
        print(f"  Equity curve:     ${first:,.2f} → ${last:,.2f}")

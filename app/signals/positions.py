"""Position tracker with trailing stops and P&L logging."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

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
from app.features.price_features import compute_features, fetch_ohlcv
from app.signals.trade_plan import compute_trailing_stops

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
POSITIONS_PATH = DATA_DIR / "positions.json"
TRADE_LOG_PATH = DATA_DIR / "trade_log.csv"

TRADE_LOG_COLUMNS = [
    "ticker",
    "direction",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "shares",
    "risk_pct",
    "pnl_pct",
    "pnl_dollar",
    "r_multiple",
    "days_held",
    "exit_reason",
    "trail_method",
    "final_score",
    "flow_score_scaled",
    "partial_pnl_pct",
]


def _load_positions() -> list[dict]:
    if POSITIONS_PATH.exists():
        return json.loads(POSITIONS_PATH.read_text())
    return []


def _save_positions(positions: list[dict]) -> None:
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_PATH.write_text(json.dumps(positions, indent=2, default=str))


def _append_trade_log(rows: list[dict]) -> None:
    if not rows:
        return
    TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not TRADE_LOG_PATH.exists()
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _risk_pct_for_score(score: float) -> float:
    """Return the per-trade risk percentage based on final_score tier."""
    for threshold, pct in SIZING_TIERS:
        if score >= threshold:
            return pct
    return 0.0


def _current_portfolio_heat(positions: list[dict]) -> float:
    """Sum of risk_dollar / PORTFOLIO_CAPITAL across all open positions."""
    return sum(p.get("risk_dollar", 0.0) for p in positions) / PORTFOLIO_CAPITAL


def _unrealized_r(pos: dict) -> float:
    """Estimate current unrealized R-multiple from stored best_price."""
    entry = pos["entry_price"]
    risk = pos["risk_per_share"]
    if risk <= 0:
        return 0.0
    best = pos.get("best_price", entry)
    if pos["direction"] == "LONG":
        return (best - entry) / risk
    return (entry - best) / risk


def _find_replaceable(existing: list[dict], new_score: float) -> dict | None:
    """Find the weakest position eligible for rotation, or None.

    A position is replaceable if:
      - Not partial-filled (T1 already hit = working trade, keep it)
      - AND either losing/flat (unrealized_r <= 0) OR the new signal's score
        exceeds the position's score by at least ROTATION_SCORE_MARGIN.
    Returns the single weakest eligible position sorted by final_score.
    """
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


def _build_position(sig: dict, risk_pct: float) -> dict | None:
    """Size a signal and return a position dict, or None if sizing fails."""
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
        "entry_date": str(date.today()),
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
        "source": sig.get("source", "fresh"),
        "partial_pnl_pct": 0.0,
        "risk_pct": risk_pct,
        "risk_dollar": round(actual_risk_dollar, 2),
        "shares": shares,
        "position_value": round(position_value, 2),
    }


def open_positions(signals: list[dict]) -> dict:
    """Convert pipeline signals into new position entries.

    Filters by MIN_FINAL_SCORE, sizes by conviction tier, respects
    MAX_PORTFOLIO_HEAT and MAX_POSITIONS. When the book is full, rotates
    out the weakest eligible position if the new signal is strong enough.

    Returns a dict with 'opened', 'rotated_out', and 'skipped' lists.
    """
    existing = _load_positions()
    open_tickers = {(p["ticker"], p["direction"]) for p in existing}
    heat = _current_portfolio_heat(existing)

    opened: list[dict] = []
    rotated_out: list[dict] = []
    skipped: list[dict] = []

    for sig in signals:
        key = (sig["ticker"], sig["direction"])
        if key in open_tickers:
            continue

        score = sig["final_score"]
        if score < MIN_FINAL_SCORE:
            skipped.append({"ticker": sig["ticker"], "reason": f"score {score:.2f} < {MIN_FINAL_SCORE}"})
            continue

        risk_pct = _risk_pct_for_score(score)
        pos = _build_position(sig, risk_pct)
        if pos is None:
            continue

        new_heat = pos["risk_dollar"] / PORTFOLIO_CAPITAL
        book_full = len(existing) >= MAX_POSITIONS

        if book_full:
            victim = _find_replaceable(existing, score)
            if victim is None:
                skipped.append({"ticker": sig["ticker"], "reason": "book full, no replaceable position"})
                continue

            # Close the victim at its current best_price (approximate market)
            exit_price = victim.get("best_price", victim["entry_price"])
            log_row = close_position(victim, exit_price, "rotated_out")
            rotated_out.append(log_row)

            # Free the victim's heat and slot
            heat -= victim.get("risk_dollar", 0.0) / PORTFOLIO_CAPITAL
            existing = [p for p in existing if not (
                p["ticker"] == victim["ticker"] and p["direction"] == victim["direction"]
            )]
            open_tickers.discard((victim["ticker"], victim["direction"]))

        if heat + new_heat > MAX_PORTFOLIO_HEAT:
            skipped.append({"ticker": sig["ticker"], "reason": "portfolio heat cap reached"})
            continue

        heat += new_heat
        opened.append(pos)
        existing.append(pos)
        open_tickers.add(key)

    if skipped:
        for s in skipped:
            print(f"  [sizing] skipped {s['ticker']}: {s['reason']}")

    _save_positions(existing)
    _append_trade_log(rotated_out)

    return {"opened": opened, "rotated_out": rotated_out, "skipped": skipped}


def close_position(pos: dict, exit_price: float, exit_reason: str, trail_method: str = "") -> dict:
    """Build a trade-log row from a closed position."""
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
        "exit_date": str(date.today()),
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


def _identify_exit_trail(pos: dict, exit_price: float) -> str:
    """Determine which trailing stop method triggered the exit."""
    direction = pos["direction"]
    is_long = direction == "LONG"

    candidates = []
    if is_long:
        if exit_price <= pos["trail_atr"]:
            candidates.append("atr_chandelier")
        if exit_price <= pos["trail_ema"]:
            candidates.append("ema_trail")
        if exit_price <= pos["trail_hybrid"]:
            candidates.append("hybrid")
    else:
        if exit_price >= pos["trail_atr"]:
            candidates.append("atr_chandelier")
        if exit_price >= pos["trail_ema"]:
            candidates.append("ema_trail")
        if exit_price >= pos["trail_hybrid"]:
            candidates.append("hybrid")

    return candidates[0] if candidates else "active_stop"


def _check_exits(pos: dict, bar: pd.Series) -> tuple[str | None, float, str]:
    """Check if the current bar triggers any exit condition.

    Returns (exit_reason, exit_price, trail_method) or (None, 0, "") if no exit.
    """
    close = float(bar["close"])
    low = float(bar["low"])
    high = float(bar["high"])
    direction = pos["direction"]
    is_long = direction == "LONG"

    # T2 hit → full close at target
    if is_long and high >= pos["target_2"]:
        return "target_2", pos["target_2"], ""
    if not is_long and low <= pos["target_2"]:
        return "target_2", pos["target_2"], ""

    # Active stop hit
    if is_long and low <= pos["active_stop"]:
        trail_method = _identify_exit_trail(pos, pos["active_stop"])
        return "stop", pos["active_stop"], trail_method
    if not is_long and high >= pos["active_stop"]:
        trail_method = _identify_exit_trail(pos, pos["active_stop"])
        return "stop", pos["active_stop"], trail_method

    # Time stop: close if held too long and not trailing profitably
    risk = pos["risk_per_share"]
    if is_long:
        unrealized_r = (close - pos["entry_price"]) / risk if risk > 0 else 0.0
    else:
        unrealized_r = (pos["entry_price"] - close) / risk if risk > 0 else 0.0

    if pos["days_held"] >= MAX_HOLD_DAYS and unrealized_r < 1.0:
        return "time_stop", close, ""

    return None, 0.0, ""


def _check_partial(pos: dict, bar: pd.Series) -> tuple[bool, float]:
    """Check if T1 was hit for a partial exit."""
    if pos["partial_filled"]:
        return False, 0.0

    high = float(bar["high"])
    low = float(bar["low"])
    is_long = pos["direction"] == "LONG"

    if is_long and high >= pos["target_1"]:
        return True, pos["target_1"]
    if not is_long and low <= pos["target_1"]:
        return True, pos["target_1"]

    return False, 0.0


def update_positions() -> dict:
    """Daily update loop: refresh stops, check exits, log trades.

    Returns a summary dict with lists of closed trades and updated positions.
    """
    positions = _load_positions()
    if not positions:
        return {"closed": [], "still_open": [], "partial_fills": []}

    still_open: list[dict] = []
    closed_rows: list[dict] = []
    partial_fills: list[str] = []

    for pos in positions:
        ticker = pos["ticker"]
        try:
            df = compute_features(fetch_ohlcv(ticker))
        except Exception as e:
            print(f"  [positions] skipping {ticker}: {e}")
            still_open.append(pos)
            continue

        bar = df.iloc[-1]
        pos["days_held"] += 1

        # Compute fresh trailing stops (always, so the stored levels are current)
        trail_update = compute_trailing_stops(pos, df)
        pos.update(trail_update)

        # Don't check exits on entry day — give the trade at least one full bar
        if pos["entry_date"] == str(date.today()):
            still_open.append(pos)
            continue

        # Check for partial exit at T1
        partial_triggered, partial_price = _check_partial(pos, bar)
        if partial_triggered:
            entry = pos["entry_price"]
            if pos["direction"] == "LONG":
                partial_pnl = (partial_price - entry) / entry
            else:
                partial_pnl = (entry - partial_price) / entry

            pos["partial_filled"] = True
            pos["partial_pnl_pct"] = round(partial_pnl * PARTIAL_EXIT_PCT, 4)
            partial_fills.append(f"{ticker} T1 hit @ {partial_price:.2f} ({partial_pnl:.1%})")

        # Check for full exit
        exit_reason, exit_price, trail_method = _check_exits(pos, bar)
        if exit_reason:
            log_row = close_position(pos, exit_price, exit_reason, trail_method)
            closed_rows.append(log_row)
        else:
            still_open.append(pos)

    _save_positions(still_open)
    _append_trade_log(closed_rows)

    return {
        "closed": closed_rows,
        "still_open": still_open,
        "partial_fills": partial_fills,
    }

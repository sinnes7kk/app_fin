"""Position tracker with trailing stops and P&L logging."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import (
    DRAWDOWN_HALT_PCT,
    DRAWDOWN_SIZING_MULT,
    DRAWDOWN_THROTTLE_PCT,
    MAX_HOLD_DAYS,
    MAX_PORTFOLIO_HEAT,
    MAX_POSITIONS,
    MAX_SECTOR_PER_DIRECTION,
    MIN_FINAL_SCORE,
    PARTIAL_EXIT_PCT,
    PARTIAL_TIERS,
    PORTFOLIO_CAPITAL,
    ROTATION_COOLDOWN_DAYS,
    ROTATION_HEALTH_MARGIN,
    SIZING_TIERS,
)
from app.features.options_context import fetch_options_context
from app.features.price_features import compute_features, fetch_ohlcv
from app.features.sector_map import get_sector
from app.signals.position_health import compute_position_health
from app.signals.scoring import score_long_setup, score_short_setup
from app.signals.trade_plan import compute_trailing_stops
from app.vendors.unusual_whales import fetch_dark_pool, fetch_net_prem_ticks

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
POSITIONS_PATH = DATA_DIR / "positions.json"
TRADE_LOG_PATH = DATA_DIR / "trade_log.csv"
EQUITY_CURVE_PATH = DATA_DIR / "equity_curve.csv"

EQUITY_CURVE_COLUMNS = [
    "date",
    "portfolio_value",
    "cash",
    "positions_value",
    "open_count",
    "daily_realized_pnl",
]

TRADE_LOG_COLUMNS = [
    "ticker",
    "direction",
    "pattern",
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


def _append_equity_curve(
    still_open: list[dict],
    realized_pnl: float,
) -> None:
    """Append one row to the equity curve CSV with today's portfolio snapshot."""
    positions_value = 0.0
    for pos in still_open:
        price = pos.get("best_price", pos["entry_price"])
        shares = pos.get("shares", 0)
        if pos["direction"] == "LONG":
            positions_value += shares * price
        else:
            positions_value += shares * (2 * pos["entry_price"] - price)

    invested = sum(p.get("position_value", 0.0) for p in still_open)
    cash = PORTFOLIO_CAPITAL - invested
    portfolio_value = cash + positions_value

    row = {
        "date": str(date.today()),
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "open_count": len(still_open),
        "daily_realized_pnl": round(realized_pnl, 2),
    }

    EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # With 8 runs/day, keep only the latest snapshot per calendar date
    # (overwrite today's row if it exists to avoid distorting drawdown math).
    today_str = row["date"]
    existing_rows: list[dict] = []
    if EQUITY_CURVE_PATH.exists():
        with open(EQUITY_CURVE_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r.get("date") != today_str]

    with open(EQUITY_CURVE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EQUITY_CURVE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)


def _current_drawdown() -> float:
    """Compute current drawdown from peak as a positive fraction (0.0 = no drawdown).

    Reads the equity curve CSV.  Returns 0.0 if data is insufficient.
    """
    if not EQUITY_CURVE_PATH.exists():
        return 0.0
    try:
        eq = pd.read_csv(EQUITY_CURVE_PATH)
        if eq.empty or "portfolio_value" not in eq.columns:
            return 0.0
        values = pd.to_numeric(eq["portfolio_value"], errors="coerce").dropna()
        if values.empty:
            return 0.0
        peak = values.cummax().iloc[-1]
        current = values.iloc[-1]
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - current) / peak)
    except Exception:
        return 0.0


def _risk_pct_for_score(score: float) -> float:
    """Return the per-trade risk percentage based on final_score tier."""
    for threshold, pct in SIZING_TIERS:
        if score >= threshold:
            return pct
    return 0.0


def _partial_pct_for_score(score: float) -> float:
    """Return the fraction of shares to sell at T1, scaled by conviction."""
    for threshold, pct in PARTIAL_TIERS:
        if score >= threshold:
            return pct
    return PARTIAL_EXIT_PCT


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


def _find_replaceable_by_health(existing: list[dict], new_score: float) -> dict | None:
    """Find the weakest position eligible for conviction-based rotation, or None.

    Compares the new candidate's final_score against each open position's
    composite conviction (0.5 * re_entry_score + 0.5 * health).  Falls back
    to raw health when conviction hasn't been computed yet.

      - STRONG: never replaceable
      - NEUTRAL: replaceable only if new_score exceeds conviction by a margin
        (adjusted by delta — deteriorating positions have a lower bar)
      - WEAK: replaceable if new_score exceeds conviction
      - FAILING: will be auto-exited separately, skip here

    Also respects anti-whipsaw cooldown and preserves partial-filled positions.
    """
    today = date.today()
    candidates: list[tuple[dict, float]] = []

    for pos in existing:
        if pos.get("partial_filled", False):
            continue

        last_rot = pos.get("last_rotation_date")
        if last_rot:
            try:
                days_since = (today - date.fromisoformat(str(last_rot))).days
            except (ValueError, TypeError):
                days_since = 999
            if days_since < ROTATION_COOLDOWN_DAYS:
                continue

        health = pos.get("health", 5.0)
        conv = pos.get("conviction") or health
        state = pos.get("health_state", "NEUTRAL")
        delta = pos.get("health_delta", 0.0)

        if state == "STRONG":
            continue

        if state == "FAILING":
            continue

        if state == "WEAK":
            if new_score > conv:
                candidates.append((pos, conv))
            continue

        # NEUTRAL: delta-adjusted margin
        margin = ROTATION_HEALTH_MARGIN
        if delta < -1.0:
            margin = max(margin - 1.0, 0.5)
        elif delta > 0:
            margin = margin + 0.5

        if new_score - conv > margin:
            candidates.append((pos, conv))

    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def _normalize_checks_snapshot(sig: dict, key: str) -> str | None:
    """String snapshot for dashboard chips (matches pipeline CSV format)."""
    v = sig.get(key)
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, list):
        joined = ", ".join(sorted(str(x) for x in v))
        return joined if joined else "none"
    s = str(v).strip()
    return s if s else None


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
        "opened_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "initial_stop": sig["stop_price"],
        "risk_per_share": risk_per_share,
        "best_price": sig["entry_price"],
        "last_price": sig["entry_price"],
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
        "price_score": sig.get("price_score"),
        "pattern": sig.get("pattern", "unknown"),
        "checks_passed": _normalize_checks_snapshot(sig, "checks_passed"),
        "checks_failed": _normalize_checks_snapshot(sig, "checks_failed"),
        "flow_score_raw": sig.get("flow_score_raw"),
        "source": sig.get("source", "fresh"),
        "gamma_regime": sig.get("gamma_regime"),
        "net_gex": sig.get("net_gex"),
        "nearest_call_wall": sig.get("nearest_call_wall"),
        "nearest_put_wall": sig.get("nearest_put_wall"),
        "gamma_flip_level_estimate": sig.get("gamma_flip_level_estimate"),
        "partial_pnl_pct": 0.0,
        "risk_pct": risk_pct,
        "risk_dollar": round(actual_risk_dollar, 2),
        "shares": shares,
        "position_value": round(position_value, 2),
        "health": None,
        "health_state": None,
        "health_at_entry": None,
        "health_prev": None,
        "health_delta": 0.0,
        "last_rotation_date": None,
        "re_entry_score": None,
        "conviction": None,
    }


def open_positions(signals: list[dict]) -> dict:
    """Convert pipeline signals into new position entries.

    Filters by MIN_FINAL_SCORE, sizes by conviction tier, respects
    MAX_PORTFOLIO_HEAT and MAX_POSITIONS. When the book is full, rotates
    out the weakest eligible position if the new signal is strong enough.

    Returns a dict with 'opened', 'rotated_out', and 'skipped' lists.
    """
    drawdown = _current_drawdown()
    if drawdown >= DRAWDOWN_HALT_PCT:
        print(f"  [circuit-breaker] drawdown {drawdown:.1%} >= halt threshold — no new entries")
        return {"opened": [], "rotated_out": [], "skipped": [
            {"ticker": "*", "reason": f"drawdown halt ({drawdown:.1%})"}
        ]}
    drawdown_mult = DRAWDOWN_SIZING_MULT if drawdown >= DRAWDOWN_THROTTLE_PCT else 1.0
    if drawdown_mult < 1.0:
        print(f"  [circuit-breaker] drawdown {drawdown:.1%} — sizing reduced to {drawdown_mult:.0%}")

    existing = _load_positions()
    open_tickers = {(p["ticker"], p["direction"]) for p in existing}
    heat = _current_portfolio_heat(existing)

    # Build sector counts for the sector concentration guard
    sector_counts: dict[tuple[str, str], int] = {}
    for p in existing:
        key_sd = (get_sector(p["ticker"]), p["direction"])
        sector_counts[key_sd] = sector_counts.get(key_sd, 0) + 1

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

        sector = get_sector(sig["ticker"])
        sector_key = (sector, sig["direction"])
        if sector_counts.get(sector_key, 0) >= MAX_SECTOR_PER_DIRECTION:
            skipped.append({"ticker": sig["ticker"], "reason": f"sector cap ({sector} {sig['direction']})"})
            continue

        risk_pct = _risk_pct_for_score(score)
        vix_mult = sig.get("vix_sizing_mult", 1.0)
        risk_pct = risk_pct * vix_mult * drawdown_mult
        pos = _build_position(sig, risk_pct)
        if pos is None:
            continue

        new_heat = pos["risk_dollar"] / PORTFOLIO_CAPITAL
        book_full = len(existing) >= MAX_POSITIONS

        rotated_slot = False
        if book_full:
            victim = _find_replaceable_by_health(existing, score)
            if victim is None:
                skipped.append({"ticker": sig["ticker"], "reason": "book full, no replaceable position"})
                continue

            exit_price = victim.get("best_price", victim["entry_price"])
            log_row = close_position(victim, exit_price, "rotated_out")
            rotated_out.append(log_row)

            heat -= victim.get("risk_dollar", 0.0) / PORTFOLIO_CAPITAL
            existing = [p for p in existing if not (
                p["ticker"] == victim["ticker"] and p["direction"] == victim["direction"]
            )]
            open_tickers.discard((victim["ticker"], victim["direction"]))
            rotated_slot = True

        if heat + new_heat > MAX_PORTFOLIO_HEAT:
            skipped.append({"ticker": sig["ticker"], "reason": "portfolio heat cap reached"})
            continue

        if rotated_slot:
            pos["last_rotation_date"] = str(date.today())

        heat += new_heat
        opened.append(pos)
        existing.append(pos)
        open_tickers.add(key)
        sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1

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
        "pattern": pos.get("pattern", "unknown"),
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

    # Health-based auto-exit: FAILING positions or rapidly-deteriorating WEAK
    health_state = pos.get("health_state")
    health_delta = pos.get("health_delta", 0.0)
    if health_state == "FAILING":
        return "health_failing", close, ""
    if health_state == "WEAK" and health_delta < -2.0:
        return "health_weak_deteriorating", close, ""

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

    # One-time migration for positions missing newer fields
    for pos in positions:
        if "health" not in pos:
            print(f"  [migration] {pos['ticker']}: adding health fields (first V2 run)")
            pos.setdefault("health", None)
            pos.setdefault("health_state", None)
            pos.setdefault("health_at_entry", None)
            pos.setdefault("health_prev", None)
            pos.setdefault("health_delta", 0.0)
            pos.setdefault("last_rotation_date", None)
        if "conviction" not in pos:
            pos.setdefault("re_entry_score", None)
            pos.setdefault("conviction", None)

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
        close = float(bar["close"])
        entry_dt = date.fromisoformat(pos["entry_date"])
        pos["days_held"] = (date.today() - entry_dt).days
        pos["last_price"] = close

        try:
            opts_ctx = fetch_options_context(ticker, close)
        except Exception:
            opts_ctx = None

        if opts_ctx and opts_ctx.get("options_context_available"):
            pos["gamma_regime"] = opts_ctx.get("gamma_regime")
            pos["net_gex"] = opts_ctx.get("net_gex")
            pos["nearest_call_wall"] = opts_ctx.get("nearest_call_wall")
            pos["nearest_put_wall"] = opts_ctx.get("nearest_put_wall")
            pos["gamma_flip_level_estimate"] = opts_ctx.get("gamma_flip_level_estimate")

        trail_update = compute_trailing_stops(pos, df, options_ctx=opts_ctx)
        pos.update(trail_update)

        # Fetch live enrichment data for position health scoring
        enrichment: dict = {}
        try:
            npt = fetch_net_prem_ticks(ticker)
            if npt:
                enrichment["net_prem_ticks"] = npt
        except Exception:
            pass
        try:
            dp = fetch_dark_pool(ticker)
            if dp:
                enrichment["dark_pool"] = dp
        except Exception:
            pass

        health_result = compute_position_health(pos, enrichment=enrichment or None)
        pos["health"] = health_result["health"]
        pos["health_state"] = health_result["health_state"]
        pos["health_at_entry"] = health_result["health_at_entry"]
        pos["health_prev"] = health_result["health_prev"]
        pos["health_delta"] = health_result["health_delta"]
        pos["health_components"] = health_result.get("components", {})

        if pos["direction"] == "LONG":
            re_entry = score_long_setup(df)
        else:
            re_entry = score_short_setup(df)
        pos["re_entry_score"] = re_entry["score"]
        pos["conviction"] = round(0.5 * re_entry["score"] + 0.5 * pos["health"], 2)

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

            p_pct = _partial_pct_for_score(pos["final_score"])
            pos["partial_filled"] = True
            pos["partial_exit_pct"] = p_pct
            pos["partial_pnl_pct"] = round(partial_pnl * p_pct, 4)
            partial_fills.append(f"{ticker} T1 hit @ {partial_price:.2f} ({partial_pnl:.1%}, exit {p_pct:.0%})")

        # Check for full exit
        exit_reason, exit_price, trail_method = _check_exits(pos, bar)
        if exit_reason:
            log_row = close_position(pos, exit_price, exit_reason, trail_method)
            closed_rows.append(log_row)
        else:
            still_open.append(pos)

    _save_positions(still_open)
    _append_trade_log(closed_rows)

    realized_pnl = sum(r.get("pnl_dollar", 0.0) for r in closed_rows)
    _append_equity_curve(still_open, realized_pnl)

    return {
        "closed": closed_rows,
        "still_open": still_open,
        "partial_fills": partial_fills,
    }

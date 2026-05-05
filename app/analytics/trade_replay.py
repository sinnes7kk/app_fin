"""Faithful equity-trade replay backtest.

Replays the production trade plan (entry / stop / T1 / T2) and exit logic
(T2 hit, ATR-chandelier trail, EMA20 trail, hybrid trail, T1 partial +
post-T1 tighten, per-bucket time stop) bar-by-bar over historical OHLCV.

This is the *real* ground truth that should replace the misleading
``5d close-to-close excess return / fixed 2% stop`` metric used by
``app/analytics/grade_backtest.py``.

What it includes:
    - ATR-14 derived plan (v1 approximation; production uses
      ``score_long_setup`` + ``build_long_trade_plan``, which require
      structural support / resistance — we use ATR-derived levels here
      for a clean, OHLCV-only replay).
    - Bar-by-bar exit checks in the same priority order as
      ``app/signals/positions.py`` ``_check_exits``.
    - MFE / MAE excursions with day-indexed timing.
    - Hit-at-R levels (did the trade reach +0.5R within 3d? +1R / +2R /
      +3R within typical hold windows?).
    - Multi-horizon close-to-close returns (kept for back-compat with
      the old metric).

What it omits (data-limited; documented):
    - Health-based exits — no historical ``net_prem_ticks`` /
      ``dark_pool``.
    - Gamma-regime trail adjustments — no historical ``options_ctx``.
    - Wall blends — same data-availability gap.

Stage C wires this engine to the same per-DTE-bucket
``resolve_hold_config`` / ``resolve_trail_config`` helpers production
uses, so production and replay never diverge.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# These will be wired to the shared helpers in Stage C; for now we
# accept hold + trail config as parameters with sensible defaults.

DEFAULT_ATR_PERIOD = 14
# Stop/T1/T2 multipliers are calibrated so that R-multiples match the
# production trade plan: 1.0 ATR stop = 1R risk, T1 at +1.5R, T2 at +3R.
# Trail moves with best price at 2.5*ATR distance, post-T1 trail at T1-1*ATR.
DEFAULT_STOP_ATR_MULT = 1.0
DEFAULT_T1_ATR_MULT = 1.5
DEFAULT_T2_ATR_MULT = 3.0
DEFAULT_TRAIL_ATR_MULT = 2.5
DEFAULT_POST_T1_TRAIL_ATR_MULT = 1.0
DEFAULT_PARTIAL_PCT = 0.5
DEFAULT_MAX_HOLD_DAYS = 20  # generous default; per-bucket overrides via Stage C
DEFAULT_TIME_STOP_MIN_R = 1.0


def _atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """True-range based ATR (Wilder smoothing)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _empty_result(reason: str = "no_data") -> dict[str, Any]:
    return {
        "exit_reason": "no_exit_yet",
        "reason_detail": reason,
        "entry_price": None,
        "stop_price": None,
        "target_1": None,
        "target_2": None,
        "risk_per_share": None,
        "atr_at_entry": None,
        "exit_price": None,
        "exit_date": None,
        "days_held": 0,
        "partial_filled": False,
        "partial_pnl_pct": 0.0,
        "realized_r": float("nan"),
        "realized_excess_pct": float("nan"),
        "mfe_pct": float("nan"),
        "mfe_day": None,
        "mfe_r": float("nan"),
        "mae_pct": float("nan"),
        "mae_day": None,
        "mae_r": float("nan"),
        "hit_0_5r_within_3d": False,
        "hit_1r_within_5d": False,
        "hit_2r_within_5d": False,
        "hit_3r_within_10d": False,
        "forward_return_1d": float("nan"),
        "forward_return_3d": float("nan"),
        "forward_return_5d": float("nan"),
        "forward_return_10d": float("nan"),
        "forward_return_20d": float("nan"),
    }


def _spy_excess(
    spy: pd.DataFrame | None,
    entry_idx: int,
    exit_idx: int,
    ticker_return: float,
) -> float:
    """Direction-agnostic ticker_return - SPY_return over the same window.

    Caller is responsible for pre-signing ticker_return by direction.
    """
    if spy is None or spy.empty:
        return float("nan")
    try:
        s_start = float(spy.iloc[min(entry_idx, len(spy) - 1)]["close"])
        s_end = float(spy.iloc[min(exit_idx, len(spy) - 1)]["close"])
        if s_start <= 0:
            return float("nan")
        spy_ret = s_end / s_start - 1.0
        return ticker_return - spy_ret
    except Exception:
        return float("nan")


def _multi_horizon_returns(
    df: pd.DataFrame,
    entry_idx: int,
    is_long: bool,
) -> dict[str, float]:
    """Direction-signed close-to-close returns at fixed horizons (kept for
    back-compat with the legacy 5d metric)."""
    out: dict[str, float] = {}
    if entry_idx >= len(df):
        return {f"forward_return_{h}d": float("nan") for h in (1, 3, 5, 10, 20)}
    entry_close = float(df.iloc[entry_idx]["close"])
    sign = 1.0 if is_long else -1.0
    for h in (1, 3, 5, 10, 20):
        end_idx = min(entry_idx + h, len(df) - 1)
        if end_idx <= entry_idx:
            out[f"forward_return_{h}d"] = float("nan")
            continue
        end_close = float(df.iloc[end_idx]["close"])
        if entry_close <= 0:
            out[f"forward_return_{h}d"] = float("nan")
            continue
        raw = end_close / entry_close - 1.0
        out[f"forward_return_{h}d"] = sign * raw
    return out


def replay_trade_plan(
    ticker: str,
    as_of: str,
    direction: str,
    df_ohlcv: pd.DataFrame,
    spy_ohlcv: pd.DataFrame | None = None,
    *,
    dominant_dte_bucket: str | None = None,
    max_hold_days: int | None = None,
    time_stop_min_r: float | None = None,
    atr_period: int = DEFAULT_ATR_PERIOD,
    stop_atr_mult: float = DEFAULT_STOP_ATR_MULT,
    t1_atr_mult: float = DEFAULT_T1_ATR_MULT,
    t2_atr_mult: float = DEFAULT_T2_ATR_MULT,
    trail_atr_mult: float | None = None,
    post_t1_trail_atr_mult: float = DEFAULT_POST_T1_TRAIL_ATR_MULT,
    partial_pct: float = DEFAULT_PARTIAL_PCT,
) -> dict[str, Any]:
    """Replay a trade plan from the OHLCV bar at ``as_of`` forward.

    Args:
        ticker: For diagnostic only (not used in math).
        as_of: ISO date string (entry decision day; entry executes at the
            close of the *next* bar to avoid look-ahead).
        direction: ``"BULLISH"`` / ``"BEARISH"`` / ``"LONG"`` / ``"SHORT"``.
        df_ohlcv: DataFrame with at least ``open / high / low / close``
            columns and a DatetimeIndex (or a ``date`` column).
        spy_ohlcv: Optional same-shape SPY frame for excess-return calc.

    Returns:
        Dict with plan, exit, MFE/MAE, hit-at-R, multi-horizon return
        fields. Always returns a dict (use ``exit_reason="no_exit_yet"``
        for incomplete data instead of raising).
    """
    # Bucket-aware defaults so production trade-management and the replay
    # backtest stay locked together. Explicit kwargs still override.
    if max_hold_days is None or time_stop_min_r is None or trail_atr_mult is None:
        try:
            from app.signals.hold_config import (
                resolve_hold_config,
                resolve_trail_config,
            )
            mh, tsr = resolve_hold_config(dominant_dte_bucket)
            tm = resolve_trail_config(dominant_dte_bucket)
        except Exception:
            mh, tsr, tm = (
                DEFAULT_MAX_HOLD_DAYS,
                DEFAULT_TIME_STOP_MIN_R,
                DEFAULT_TRAIL_ATR_MULT,
            )
        if max_hold_days is None:
            max_hold_days = mh
        if time_stop_min_r is None:
            time_stop_min_r = tsr
        if trail_atr_mult is None:
            trail_atr_mult = tm

    if df_ohlcv is None or df_ohlcv.empty:
        return _empty_result("no_ohlcv")

    df = df_ohlcv.copy()
    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    is_long = str(direction).upper() in ("BULLISH", "LONG")

    # Find the bar at or after as_of - this is the "decision bar".
    # Entry executes at the NEXT bar's open (no look-ahead on the decision).
    entry_dt = pd.to_datetime(as_of)
    future = df[df.index >= entry_dt]
    if len(future) < 2:
        return _empty_result("not_enough_forward_bars")

    decision_idx = df.index.get_loc(future.index[0])
    if isinstance(decision_idx, slice):
        decision_idx = decision_idx.start
    entry_idx = int(decision_idx) + 1
    if entry_idx >= len(df):
        return _empty_result("entry_at_last_bar")

    # Compute ATR/EMA on the full series so we can index into them.
    atr_series = _atr(df, period=atr_period)
    ema20 = _ema(df["close"].astype(float), period=20)

    atr_at_entry = float(atr_series.iloc[decision_idx])
    if not math.isfinite(atr_at_entry) or atr_at_entry <= 0:
        return _empty_result("atr_unavailable")

    entry_price = float(df.iloc[entry_idx]["open"])
    if entry_price <= 0:
        return _empty_result("entry_price_invalid")

    # Build the plan (ATR-based v1 approximation).
    if is_long:
        stop_price = entry_price - stop_atr_mult * atr_at_entry
        target_1 = entry_price + t1_atr_mult * atr_at_entry
        target_2 = entry_price + t2_atr_mult * atr_at_entry
    else:
        stop_price = entry_price + stop_atr_mult * atr_at_entry
        target_1 = entry_price - t1_atr_mult * atr_at_entry
        target_2 = entry_price - t2_atr_mult * atr_at_entry
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return _empty_result("risk_per_share_invalid")

    # Bar-by-bar replay.
    best_price = entry_price
    partial_filled = False
    partial_pnl_pct = 0.0
    active_stop = stop_price
    initial_stop = stop_price

    mfe_pct = 0.0
    mae_pct = 0.0
    mfe_day: int | None = None
    mae_day: int | None = None
    hit_flags = {
        "hit_0_5r_within_3d": False,
        "hit_1r_within_5d": False,
        "hit_2r_within_5d": False,
        "hit_3r_within_10d": False,
    }

    exit_reason = "no_exit_yet"
    reason_detail = ""
    exit_price: float | None = None
    exit_idx_actual = entry_idx
    days_held = 0

    for offset in range(1, max_hold_days + 1):
        bar_idx = entry_idx + offset
        if bar_idx >= len(df):
            break
        days_held = offset
        bar = df.iloc[bar_idx]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        if is_long:
            best_price = max(best_price, high)
            bar_excursion_high = (high - entry_price) / risk_per_share
            bar_excursion_low = (low - entry_price) / risk_per_share
        else:
            best_price = min(best_price, low)
            bar_excursion_high = (entry_price - low) / risk_per_share
            bar_excursion_low = (entry_price - high) / risk_per_share

        if bar_excursion_high > mfe_pct:
            mfe_pct = bar_excursion_high
            mfe_day = offset
        if bar_excursion_low < mae_pct:
            mae_pct = bar_excursion_low
            mae_day = offset

        if offset <= 3 and bar_excursion_high >= 0.5:
            hit_flags["hit_0_5r_within_3d"] = True
        if offset <= 5 and bar_excursion_high >= 1.0:
            hit_flags["hit_1r_within_5d"] = True
        if offset <= 5 and bar_excursion_high >= 2.0:
            hit_flags["hit_2r_within_5d"] = True
        if offset <= 10 and bar_excursion_high >= 3.0:
            hit_flags["hit_3r_within_10d"] = True

        # ----- Order matches positions.py _check_exits -----

        # 1. T2 hit -> full close at target
        if is_long and high >= target_2:
            exit_reason = "T2"
            exit_price = target_2
            exit_idx_actual = bar_idx
            break
        if not is_long and low <= target_2:
            exit_reason = "T2"
            exit_price = target_2
            exit_idx_actual = bar_idx
            break

        # 2. T1 partial (intraday) — fills before stop check on same bar
        if not partial_filled:
            t1_hit = (is_long and high >= target_1) or (not is_long and low <= target_1)
            if t1_hit:
                partial_filled = True
                partial_pnl_pct = partial_pct
                # After partial, tighten trail to T1 ± post_t1_trail_atr*ATR
                if is_long:
                    new_trail = target_1 - post_t1_trail_atr_mult * atr_at_entry
                    initial_stop = max(initial_stop, new_trail)
                else:
                    new_trail = target_1 + post_t1_trail_atr_mult * atr_at_entry
                    initial_stop = min(initial_stop, new_trail)

        # 3. Update active stop using ATR / EMA / hybrid trail
        # EMA trail only counts when it is on the *favorable* side of entry
        # (below entry for longs, above for shorts). This matches production
        # in spirit: real long breakout entries are above EMA20, so EMA20
        # is always a meaningful trail. In synthetic / degenerate flat data
        # the EMA may straddle entry, in which case it shouldn't trigger a
        # stop on the entry bar.
        ema_now = float(ema20.iloc[bar_idx])
        if is_long:
            trail_atr = best_price - trail_atr_mult * atr_at_entry
            trail_ema = ema_now if ema_now < entry_price else stop_price
            unr_r_close = (close - entry_price) / risk_per_share
            if unr_r_close < 1.0:
                trail_hybrid = stop_price
            elif unr_r_close < 2.0:
                trail_hybrid = entry_price
            else:
                trail_hybrid = best_price - 2.0 * atr_at_entry
            active_stop = max(initial_stop, trail_atr, trail_ema, trail_hybrid)
        else:
            trail_atr = best_price + trail_atr_mult * atr_at_entry
            trail_ema = ema_now if ema_now > entry_price else stop_price
            unr_r_close = (entry_price - close) / risk_per_share
            if unr_r_close < 1.0:
                trail_hybrid = stop_price
            elif unr_r_close < 2.0:
                trail_hybrid = entry_price
            else:
                trail_hybrid = best_price + 2.0 * atr_at_entry
            active_stop = min(initial_stop, trail_atr, trail_ema, trail_hybrid)

        # 4. Stop hit
        if is_long and low <= active_stop:
            exit_reason = "stop" if not partial_filled else "T1_then_stop"
            exit_price = active_stop
            exit_idx_actual = bar_idx
            break
        if not is_long and high >= active_stop:
            exit_reason = "stop" if not partial_filled else "T1_then_stop"
            exit_price = active_stop
            exit_idx_actual = bar_idx
            break

        # 5. Time stop — only after max_hold_days AND below time_stop_min_r
        if offset >= max_hold_days and unr_r_close < time_stop_min_r:
            exit_reason = "time_stop"
            exit_price = close
            exit_idx_actual = bar_idx
            break

    # If we ran out of bars without hitting any exit, mark "no_exit_yet"
    if exit_reason == "no_exit_yet":
        exit_idx_actual = min(entry_idx + max_hold_days, len(df) - 1)
        exit_price = float(df.iloc[exit_idx_actual]["close"])
        days_held = exit_idx_actual - entry_idx

    # Realized R
    if is_long:
        full_pnl = (exit_price - entry_price) / risk_per_share
    else:
        full_pnl = (entry_price - exit_price) / risk_per_share

    if partial_filled:
        # Partial at T1 (locked at +t1_atr_mult / stop_atr_mult R), remainder at exit
        if is_long:
            partial_r = (target_1 - entry_price) / risk_per_share
            remainder_r = (exit_price - entry_price) / risk_per_share
        else:
            partial_r = (entry_price - target_1) / risk_per_share
            remainder_r = (entry_price - exit_price) / risk_per_share
        realized_r = partial_pct * partial_r + (1.0 - partial_pct) * remainder_r
    else:
        realized_r = full_pnl

    # Realized excess vs SPY over the same hold window
    if is_long:
        ticker_ret = (exit_price - entry_price) / entry_price
    else:
        ticker_ret = (entry_price - exit_price) / entry_price
    realized_excess = _spy_excess(spy_ohlcv, entry_idx, exit_idx_actual, ticker_ret)

    # MFE/MAE in R
    mfe_r = mfe_pct  # already in R units (we divided by risk_per_share above)
    mae_r = mae_pct
    # Convert MFE/MAE to percent-of-entry for human-readable output
    mfe_pct_price = mfe_r * (risk_per_share / entry_price)
    mae_pct_price = mae_r * (risk_per_share / entry_price)

    # Multi-horizon close-to-close (legacy parity)
    multi = _multi_horizon_returns(df, entry_idx, is_long)

    return {
        "ticker": ticker,
        "as_of": as_of,
        "direction": "BULLISH" if is_long else "BEARISH",
        "entry_price": round(entry_price, 4),
        "stop_price": round(stop_price, 4),
        "target_1": round(target_1, 4),
        "target_2": round(target_2, 4),
        "risk_per_share": round(risk_per_share, 4),
        "atr_at_entry": round(atr_at_entry, 4),
        "exit_reason": exit_reason,
        "reason_detail": reason_detail,
        "exit_price": round(exit_price, 4) if exit_price is not None else None,
        "exit_date": df.index[exit_idx_actual].strftime("%Y-%m-%d"),
        "days_held": int(days_held),
        "partial_filled": bool(partial_filled),
        "partial_pnl_pct": float(partial_pnl_pct),
        "realized_r": round(float(realized_r), 4),
        "realized_excess_pct": round(float(realized_excess), 6) if not math.isnan(realized_excess) else float("nan"),
        "mfe_pct": round(float(mfe_pct_price), 6),
        "mfe_r": round(float(mfe_r), 4),
        "mfe_day": mfe_day,
        "mae_pct": round(float(mae_pct_price), 6),
        "mae_r": round(float(mae_r), 4),
        "mae_day": mae_day,
        "hit_0_5r_within_3d": bool(hit_flags["hit_0_5r_within_3d"]),
        "hit_1r_within_5d": bool(hit_flags["hit_1r_within_5d"]),
        "hit_2r_within_5d": bool(hit_flags["hit_2r_within_5d"]),
        "hit_3r_within_10d": bool(hit_flags["hit_3r_within_10d"]),
        **{k: (round(float(v), 6) if not math.isnan(v) else float("nan"))
           for k, v in multi.items()},
    }

import argparse

from app.config import MAX_POSITIONS
from app.signals.pipeline import apply_agent_filter, run_flow_to_price_pipeline
from app.signals.positions import open_positions, update_positions


def _print_signals(out: dict) -> None:
    signals_df = out["signals_df"]
    rejected_df = out["rejected_df"]
    wl = out["watchlist"]

    print("=" * 60)
    print("FINAL SIGNALS")
    print("=" * 60)
    if not signals_df.empty:
        print(signals_df.to_string())
    else:
        print("  No tickers passed both flow + price validation")

    print()
    print("=" * 60)
    print("REJECTED CANDIDATES")
    print("=" * 60)
    if not rejected_df.empty:
        print(rejected_df.to_string())
    else:
        print("  No candidates were rejected (none reached price validation)")

    mr = out.get("market_regime", {})
    if mr.get("available"):
        print()
        print("=" * 60)
        print("MARKET REGIME")
        print("=" * 60)
        print(f"  Regime score:  {mr['regime_score']:.2f}  (0=risk-off, 1=risk-on)")
        print(f"  SPY trend:     {mr['spy_trend']}  (close={mr['spy_close']}  EMA20={mr['spy_ema20']}  EMA50={mr['spy_ema50']})")
        print(f"  VIX:           {mr.get('vix_close', '?')}  sizing mult={mr['vix_sizing_mult']:.2f}")
        print(f"  Components:    spy_align={mr.get('_spy_alignment', '?')}  vix={mr.get('_vix_component', '?')}  rvol={mr.get('_rvol_component', '?')}  tide={mr.get('_tide_component', '?')}")

    al = out["alert_stats"]
    print()
    print("=" * 60)
    print("UW ALERTS")
    print("=" * 60)
    print(f"  Alert tickers:     {al['alert_tickers']}")
    print(f"  New (not in bulk): {al['new_tickers']}")

    print()
    print("=" * 60)
    print("WATCHLIST STATUS")
    print("=" * 60)
    print(f"  Previous entries:  {wl['previous_count']}")
    print(f"  Expired (pruned):  {wl['expired_count']}")
    print(f"  Promoted to signal:{wl['promoted_count']}")
    print(f"  Still watching:    {wl['still_watching_count']}")
    print(f"  New rejects added: {wl['new_rejects_added']}")
    print(f"  Current total:     {wl['current_count']}")


def _print_positions(open_result: dict, update_result: dict) -> None:
    from app.config import PORTFOLIO_CAPITAL

    print()
    print("=" * 60)
    print("POSITION TRACKER")
    print("=" * 60)

    new_positions = open_result.get("opened", [])
    rotated = open_result.get("rotated_out", [])

    if rotated:
        print(f"\n  Rotated out {len(rotated)} position(s):")
        for c in rotated:
            r_str = f"{c['r_multiple']:+.1f}R"
            pnl_str = f"{c['pnl_pct']:+.2%}"
            dollar_str = f"${c.get('pnl_dollar', 0):+,.0f}"
            print(f"    {c['direction']:5s} {c['ticker']:6s}  {pnl_str} ({dollar_str})  {r_str}  "
                  f"held {c['days_held']}d  → replaced by higher conviction")

    if new_positions:
        print(f"\n  Opened {len(new_positions)} new position(s):")
        for p in new_positions:
            print(f"    {p['direction']:5s} {p['ticker']:6s}  entry={p['entry_price']:.2f}"
                  f"  stop={p['initial_stop']:.2f}  T1={p['target_1']:.2f}  T2={p['target_2']:.2f}"
                  f"  |  {p['shares']} shares  risk={p['risk_pct']:.1%}  ${p['position_value']:,.0f}")
    else:
        print("\n  No new positions opened (all signals already in book or none generated)")

    for msg in update_result.get("partial_fills", []):
        print(f"  PARTIAL: {msg}")

    closed = update_result.get("closed", [])
    if closed:
        print(f"\n  Closed {len(closed)} position(s):")
        for c in closed:
            r_str = f"{c['r_multiple']:+.1f}R"
            pnl_str = f"{c['pnl_pct']:+.2%}"
            dollar_str = f"${c.get('pnl_dollar', 0):+,.0f}"
            reason = c["exit_reason"]
            trail = f" [{c['trail_method']}]" if c["trail_method"] else ""
            print(f"    {c['direction']:5s} {c['ticker']:6s}  {pnl_str} ({dollar_str})  {r_str}  "
                  f"held {c['days_held']}d  reason={reason}{trail}")
    else:
        print("\n  No positions closed today")

    still_open = update_result.get("still_open", [])
    if still_open:
        total_heat = sum(p.get("risk_dollar", 0) for p in still_open) / PORTFOLIO_CAPITAL
        total_value = sum(p.get("position_value", 0) for p in still_open)
        print(f"\n  Open positions ({len(still_open)}/{MAX_POSITIONS})  "
              f"portfolio heat: {total_heat:.1%}  exposure: ${total_value:,.0f}")
        for p in still_open:
            risk = p["risk_per_share"]
            close_approx = p["best_price"]
            if p["direction"] == "LONG":
                ur = (close_approx - p["entry_price"]) / risk if risk > 0 else 0.0
            else:
                ur = (p["entry_price"] - close_approx) / risk if risk > 0 else 0.0
            partial_tag = " [T1 filled]" if p["partial_filled"] else ""
            shares = p.get("shares", 0)
            h = p.get("health")
            h_state = p.get("health_state", "?")
            h_delta = p.get("health_delta", 0.0)
            health_tag = f"  H={h:.1f} {h_state}" if h is not None else ""
            delta_tag = f" ({h_delta:+.1f})" if h is not None else ""
            print(f"    {p['direction']:5s} {p['ticker']:6s}  {shares:>4} shares  day {p['days_held']}  "
                  f"stop={p['active_stop']:.2f}  ~{ur:+.1f}R{partial_tag}{health_tag}{delta_tag}")
    else:
        print("\n  No open positions")


def main() -> None:
    parser = argparse.ArgumentParser(description="app_fin signal pipeline")
    parser.add_argument("--scan-only", action="store_true", help="Run scanner without position management")
    parser.add_argument("--update-only", action="store_true", help="Only update existing positions (skip scanning)")
    parser.add_argument("--backtest", action="store_true", help="Run replay backtest over archived flow")
    parser.add_argument("--start", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward backtest mode")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start local HTTP dashboard (default http://127.0.0.1:5050)",
    )
    args = parser.parse_args()

    if args.serve:
        from app.web.server import main as serve_dashboard

        serve_dashboard()
        return

    if args.backtest:
        from app.backtest.engine import print_summary, run_backtest, run_walk_forward, save_backtest_results

        start = args.start or "2020-01-01"
        end = args.end or "2099-12-31"
        if args.walk_forward:
            result = run_walk_forward(start_date=start, end_date=end)
        else:
            result = run_backtest(start_date=start, end_date=end)
        print_summary(result)
        if result.trade_log:
            path = save_backtest_results(result)
            print(f"\n  Trade log saved to {path}")
        return

    if args.update_only:
        print("Updating existing positions...")
        result = update_positions()
        _print_positions({"opened": [], "rotated_out": [], "skipped": []}, result)
        # Agent shadow portfolio
        agent_result = update_positions(portfolio="agent")
        if agent_result["closed"] or agent_result["still_open"]:
            print(f"  [agent portfolio] {len(agent_result['still_open'])} open, {len(agent_result['closed'])} closed")
        return

    out = run_flow_to_price_pipeline(flow_limit=2000, top_n=50, min_premium=500_000, alert_hours_back=48)
    _print_signals(out)

    if args.scan_only:
        return

    results = out["results"]
    open_result = open_positions(results)

    update_result = update_positions()
    _print_positions(open_result, update_result)

    # Agent shadow portfolio: same universe, filtered by orchestrator decisions
    agent_signals = apply_agent_filter(results)
    agent_open = open_positions(agent_signals, portfolio="agent")
    agent_update = update_positions(portfolio="agent")
    n_vetoed = len(results) - len(agent_signals)
    if n_vetoed > 0 or agent_open["opened"] or agent_update["closed"]:
        print(f"\n  [agent portfolio] vetoed {n_vetoed} signals, "
              f"opened {len(agent_open['opened'])}, "
              f"closed {len(agent_update['closed'])}, "
              f"open {len(agent_update['still_open'])}")


if __name__ == "__main__":
    main()

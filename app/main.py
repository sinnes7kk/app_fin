from app.signals.pipeline import run_flow_to_price_pipeline


def main() -> None:
    out = run_flow_to_price_pipeline(flow_limit=500, top_n=20, min_premium=50_000)

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


if __name__ == "__main__":
    main()

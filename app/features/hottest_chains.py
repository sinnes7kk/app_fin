"""Hottest option chains aggregation — contract-level unusual activity."""

from __future__ import annotations


def aggregate_chains_by_ticker(raw_chains: list[dict]) -> dict:
    """Group hottest chain contracts by underlying ticker.

    Returns a dict with:
        by_ticker  - list of per-ticker summaries sorted by total premium
        contracts  - total contract count
    """
    if not raw_chains:
        return {"by_ticker": [], "contracts": 0}

    ticker_agg: dict[str, dict] = {}

    for row in raw_chains:
        ticker = (row.get("ticker_symbol") or row.get("ticker") or "").upper().strip()
        if not ticker:
            continue

        if ticker not in ticker_agg:
            ticker_agg[ticker] = {
                "ticker": ticker,
                "total_premium": 0.0,
                "call_premium": 0.0,
                "put_premium": 0.0,
                "contract_count": 0,
                "top_contracts": [],
            }

        agg = ticker_agg[ticker]
        agg["contract_count"] += 1

        try:
            prem = float(row.get("total_premium") or row.get("premium") or 0)
        except (TypeError, ValueError):
            prem = 0.0
        agg["total_premium"] += prem

        opt_type = (row.get("type") or row.get("option_type") or "").upper().strip()
        if "CALL" in opt_type:
            agg["call_premium"] += prem
        elif "PUT" in opt_type:
            agg["put_premium"] += prem

        try:
            vol = float(row.get("volume") or 0)
            oi = float(row.get("open_interest") or 0)
            vol_oi = round(vol / oi, 2) if oi > 0 else 0.0
        except (TypeError, ValueError):
            vol_oi = 0.0

        try:
            ask_vol = float(row.get("ask_side_volume") or row.get("total_ask_side_prem") or 0)
            total_vol = float(row.get("volume") or 1)
            ask_pct = round(ask_vol / total_vol * 100, 1) if total_vol > 0 else 0.0
        except (TypeError, ValueError):
            ask_pct = 0.0

        contract = {
            "strike": row.get("strike"),
            "expiry": row.get("expiry") or row.get("expiration_date"),
            "type": "Call" if "CALL" in opt_type else "Put" if "PUT" in opt_type else opt_type,
            "premium": round(prem, 2),
            "vol_oi": vol_oi,
            "ask_pct": ask_pct,
            "volume": int(vol) if vol else 0,
            "oi": int(oi) if oi else 0,
        }
        agg["top_contracts"].append(contract)

    results: list[dict] = []
    for agg in ticker_agg.values():
        agg["total_premium"] = round(agg["total_premium"], 2)
        agg["call_premium"] = round(agg["call_premium"], 2)
        agg["put_premium"] = round(agg["put_premium"], 2)

        total = agg["call_premium"] + agg["put_premium"]
        if total > 0:
            agg["dominant_side"] = "Calls" if agg["call_premium"] > agg["put_premium"] else "Puts"
            agg["call_pct"] = round(agg["call_premium"] / total * 100, 1)
        else:
            agg["dominant_side"] = "—"
            agg["call_pct"] = 50.0

        agg["top_contracts"] = sorted(
            agg["top_contracts"],
            key=lambda c: c["premium"],
            reverse=True,
        )[:5]

        if agg["top_contracts"]:
            top = agg["top_contracts"][0]
            agg["top_chain_label"] = (
                f"${top['strike']} {top['type']} {str(top['expiry'] or '')[:10]}"
            )
            agg["top_chain_vol_oi"] = top["vol_oi"]
            agg["top_chain_ask_pct"] = top["ask_pct"]
            agg["top_chain_premium"] = top["premium"]
        else:
            agg["top_chain_label"] = None

        results.append(agg)

    results.sort(key=lambda x: x["total_premium"], reverse=True)

    return {
        "by_ticker": results,
        "contracts": sum(a["contract_count"] for a in results),
    }

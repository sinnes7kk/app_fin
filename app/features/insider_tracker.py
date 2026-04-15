"""Insider transaction classification and per-ticker aggregation."""

from __future__ import annotations

from datetime import date, timedelta

NOTABLE_TITLES = {
    "ceo", "cfo", "coo", "cto", "president", "chairman",
    "director", "chief", "officer", "vp", "evp", "svp",
}


def classify_insider_activity(
    transactions: list[dict],
    lookback_days: int = 30,
) -> dict[str, dict]:
    """Aggregate insider transactions by ticker over a lookback window.

    Returns a dict keyed by ticker with buy/sell counts, net direction,
    total notional, notable buyers, and most recent transaction date.
    """
    if not transactions:
        return {}

    cutoff = str(date.today() - timedelta(days=lookback_days))
    ticker_agg: dict[str, dict] = {}

    for row in transactions:
        ticker = (row.get("ticker") or row.get("symbol") or "").upper().strip()
        if not ticker:
            continue

        filing_date = str(row.get("filing_date") or row.get("date") or "")[:10]
        if filing_date and filing_date < cutoff:
            continue

        tx_type = (
            row.get("transaction_type") or row.get("acquisition_or_disposition") or ""
        ).upper().strip()

        is_buy = any(kw in tx_type for kw in ("BUY", "PURCHASE", "A", "ACQUISITION"))
        is_sell = any(kw in tx_type for kw in ("SELL", "SALE", "D", "DISPOSITION"))

        if not is_buy and not is_sell:
            if tx_type in ("A", "P"):
                is_buy = True
            elif tx_type in ("D", "S"):
                is_sell = True
            else:
                continue

        try:
            shares = abs(float(row.get("shares") or row.get("quantity") or 0))
        except (TypeError, ValueError):
            shares = 0.0
        try:
            value = abs(float(row.get("value") or row.get("total_value") or 0))
        except (TypeError, ValueError):
            value = 0.0

        insider_name = (row.get("full_name") or row.get("insider_name") or row.get("name") or "").strip()
        insider_title = (row.get("title") or row.get("insider_title") or "").strip()

        is_notable = any(
            t in insider_title.lower() for t in NOTABLE_TITLES
        ) if insider_title else False

        if ticker not in ticker_agg:
            ticker_agg[ticker] = {
                "ticker": ticker,
                "buy_count": 0,
                "sell_count": 0,
                "buy_notional": 0.0,
                "sell_notional": 0.0,
                "buy_shares": 0.0,
                "sell_shares": 0.0,
                "notable_buyers": [],
                "notable_sellers": [],
                "last_date": "",
            }

        agg = ticker_agg[ticker]

        if is_buy:
            agg["buy_count"] += 1
            agg["buy_notional"] += value
            agg["buy_shares"] += shares
            if is_notable and insider_name:
                label = f"{insider_name}"
                if insider_title:
                    label += f" ({insider_title})"
                if value >= 10_000:
                    label += f" ${value:,.0f}"
                agg["notable_buyers"].append(label)
        elif is_sell:
            agg["sell_count"] += 1
            agg["sell_notional"] += value
            agg["sell_shares"] += shares
            if is_notable and insider_name:
                label = f"{insider_name}"
                if insider_title:
                    label += f" ({insider_title})"
                agg["notable_sellers"].append(label)

        if filing_date > agg["last_date"]:
            agg["last_date"] = filing_date

    results: dict[str, dict] = {}
    for ticker, agg in ticker_agg.items():
        net_notional = agg["buy_notional"] - agg["sell_notional"]
        if agg["buy_count"] > agg["sell_count"]:
            net_direction = "buying"
        elif agg["sell_count"] > agg["buy_count"]:
            net_direction = "selling"
        else:
            net_direction = "mixed"

        results[ticker] = {
            "ticker": ticker,
            "buy_count": agg["buy_count"],
            "sell_count": agg["sell_count"],
            "buy_notional": round(agg["buy_notional"], 2),
            "sell_notional": round(agg["sell_notional"], 2),
            "net_notional": round(net_notional, 2),
            "net_direction": net_direction,
            "notable_buyers": agg["notable_buyers"][:3],
            "notable_sellers": agg["notable_sellers"][:3],
            "last_date": agg["last_date"],
        }

    return results

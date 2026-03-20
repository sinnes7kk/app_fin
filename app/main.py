from app.features.price_features import fetch_ohlcv, clean_ohlcv, compute_features
from app.rules.continuation_rules import evaluate_long_setup


def main():
    ticker = "AAPL"
    df = fetch_ohlcv(ticker)
    df = clean_ohlcv(df)
    df = compute_features(df)

    result = evaluate_long_setup(df)
    print(df.tail(5)[["close", "ema20", "ema50", "atr14", "volume", "vol_ma20", "rel_volume"]])
    print(result)


if __name__ == "__main__":
    main()
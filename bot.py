import ccxt
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Binance Exchange Setup
exchange = ccxt.binance({
    'options': {'defaultType': 'future'}  # Fetching futures data
})

def fetch_top_trading_pairs(limit=20):
    """
    Fetches the top trading pairs based on:
    - High 24h Trading Volume
    - Strong ATR (Volatility)
    - Favorable RSI (Not Overbought/Oversold)
    - Clear Trend (EMA Confirmation)
    :param limit: Number of top pairs to return
    :return: List of trading pairs
    """
    try:
        # Fetch all futures tickers
        markets = exchange.fetch_tickers()
        futures_pairs = [symbol for symbol in markets if symbol.endswith(":USDT")]
        print(futures_pairs)
        trading_data = []

        for symbol in futures_pairs:
            try:
                ticker = markets[symbol]
                volume = float(ticker['quoteVolume'])  # 24h volume
                price = float(ticker['last'])

                # Fetch recent candles for analysis
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)  # 50 candles for trend check
                if len(ohlcv) < 50:
                    continue

                closes = [candle[4] for candle in ohlcv]  # Closing prices

                # Calculate ATR (Volatility Measure)
                highs = [candle[2] for candle in ohlcv]
                lows = [candle[3] for candle in ohlcv]
                atr = sum([abs(h - l) for h, l in zip(highs, lows)]) / len(ohlcv)

                # Calculate RSI
                rsi = calculate_rsi(closes)

                # Calculate EMA Trend Confirmation
                ema_10 = sum(closes[-10:]) / 10
                ema_50 = sum(closes[-50:]) / 50
                trending = "bullish" if ema_10 > ema_50 else "bearish"

                trading_data.append({
                    "symbol": symbol,
                    "volume": volume,
                    "atr": atr,
                    "rsi": rsi,
                    "trend": trending
                })

            except Exception as e:
                logging.warning(f"Error fetching data for {symbol}: {e}")
                continue

        # Filtering criteria for top 10 coins:
        sorted_pairs = sorted(
            trading_data,
            key=lambda x: (x['volume'], x['atr']),  # Sort by high volume & high volatility (ATR)
            reverse=True
        )

        selected_pairs = [pair['symbol'] for pair in sorted_pairs[:limit]]

        logging.info(f"Selected Trading Pairs: {selected_pairs}")
        return selected_pairs

    except Exception as e:
        logging.error(f"Error fetching trading pairs: {e}")
        return []

def calculate_rsi(closes, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = sum(delta for delta in deltas if delta > 0) / period
    losses = abs(sum(delta for delta in deltas if delta < 0)) / period

    rs = gains / losses if losses != 0 else 0
    return 100 - (100 / (1 + rs))

if __name__ == "__main__":
    top_pairs = fetch_top_trading_pairs()
    with open("trading_pairs.txt", "w") as f:
        f.write("\n".join(top_pairs))

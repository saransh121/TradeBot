import ccxt
exchange = ccxt.binance()

print(exchange.fetch_ohlcv('TRX/USDT', '15m', limit=100))
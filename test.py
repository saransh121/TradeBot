import ccxt
exchange = ccxt.binance()

#USUAL/USDT
#MOVE/USDT
#VELODROME/USDT
#TROY/USDT
#KOMA/USDT
#BIGTIME/USDT
#FLUX/USDT
#ETH/USDT
print(len(exchange.fetch_ohlcv('ETH/USDT', '3m', limit=1000)))
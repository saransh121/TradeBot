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
#PEPE/USDT
print((exchange.fetch_ohlcv('SHIB/USDT', '3m', limit=1000)))

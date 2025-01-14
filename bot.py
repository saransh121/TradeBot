import ccxt
import time
import pandas as pd
import numpy as np
import logging
import joblib
import os
import threading
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# Initialize Binance Futures API
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'options': {'defaultType': 'future'},
})

# Logging setup
logging.basicConfig(level=logging.INFO, filename='trading_bot.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
LEVERAGE = 35
POSITION_SIZE_PERCENT = 3  # % of wallet balance to trade per coin
TIMEFRAME = '3m'
PROFIT_TARGET_PERCENT = 0.1  # 10% profit target
N_STEPS = 60  # For LSTM input sequence length

# Trading Pairs
TRADING_PAIRS = ["XRP/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT","ENA/USDT","USUAL/USDT","AIXBT/USDT"]
#TRADING_PAIRS = ["XRP/USDT", 
#                     "DOGE/USDT", "ADA/USDT"]
# Fetch wallet balance
def fetch_wallet_balance():
    try:
        balance = exchange.fetch_balance({'type': 'future'})
        usdt_free = balance['free']['USDT']
        logging.info(f"Available USDT balance: {usdt_free} USDT")
        return usdt_free
    except Exception as e:
        logging.error(f"Error fetching wallet balance: {e}")
        return 0

# Fetch OHLCV data
def fetch_data(symbol, timeframe, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# Add Technical Indicators
def add_indicators(data):
    try:
        data['MA_10'] = data['close'].rolling(window=10).mean()
        data['MA_30'] = data['close'].rolling(window=30).mean()
        data['RSI'] = calculate_rsi(data['close'])
        data['ATR'] = calculate_atr(data)
        data['EMA_7'] = data['close'].ewm(span=7, adjust=False).mean()
        data['EMA_25'] = data['close'].ewm(span=25, adjust=False).mean()
        data['EMA_99'] = data['close'].ewm(span=99, adjust=False).mean()
        data = calculate_macd(data)
        # data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
        # data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
        # data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['close'])
        return data.dropna()
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        return None

def calculate_bollinger_bands(series, window=20):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, lower_band

# RSI Calculation
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ATR Calculation
def calculate_atr(data, period=14):
    data['TR'] = data[['high', 'low', 'close']].apply(
        lambda row: max(row['high'] - row['low'], abs(row['high'] - row['close']), abs(row['low'] - row['close'])),
        axis=1
    )
    return data['TR'].rolling(window=period).mean()

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD and Signal line.
    """
    data['EMA_12'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

def detect_crossover(data, short_ema_col='EMA_7', long_ema_col='EMA_25', trend_ema_col='EMA_99'):
    """
    Enhanced EMA crossover detection with support, trend, breakout, wick, and volume analysis.
    
    :param data: DataFrame containing price, EMA, and volume columns.
    :return: 'buy', 'sell', 'watch', or None.
    """
    if len(data) < 4:
        return None  # Not enough data

    # Extract EMA values
    short_prev, short_curr = data[short_ema_col].iloc[-2], data[short_ema_col].iloc[-1]
    long_prev, long_curr = data[long_ema_col].iloc[-2], data[long_ema_col].iloc[-1]
    trend_curr = data[trend_ema_col].iloc[-1]

    # Current and previous candle data
    open_curr, close_curr, low_curr, high_curr = data['open'].iloc[-1], data['close'].iloc[-1], data['low'].iloc[-1], data['high'].iloc[-1]
    close_prev = data['close'].iloc[-2]
    
    # Volume data
    volume_curr = data['volume'].iloc[-1]
    avg_volume = data['volume'].iloc[-20:].mean()  # Average of last 20 periods

    # EMA slopes
    short_slope = short_curr - short_prev
    long_slope = long_curr - long_prev

    # Candle characteristics
    is_red_candle = close_curr < open_curr
    is_green_candle = close_curr > open_curr

    # Support/Resistance threshold
    support_threshold = 0.001 * close_curr  # 0.1% buffer

    # EMA Compression threshold
    ema_gap = abs(short_curr - long_curr)
    compression_threshold = 0.0005 * close_curr  # 0.05% gap

    # Wick sizes
    upper_wick = high_curr - max(open_curr, close_curr)
    lower_wick = min(open_curr, close_curr) - low_curr

    # Volume conditions
    is_high_volume = volume_curr > 1.1 * avg_volume  # 50% higher than average
    is_low_volume = volume_curr < 0.9 * avg_volume   # 20% lower than average

    # --- New Logic with Volume Analysis ---

    # 1. High Volume Breakout Above EMA → Strong Buy
    if close_prev < short_prev and close_curr > short_curr and close_curr > long_curr and is_high_volume:
        logging.info("High volume breakout above EMA resistance. Strong BUY signal.")
        return 'buy'

    # 2. High Volume Breakdown Below EMA → Strong Sell
    if close_prev > short_prev and close_curr < short_curr and close_curr < long_curr and is_high_volume:
        logging.info("High volume breakdown below EMA support. Strong SELL signal.")
        return 'sell'

    # 3. Low Volume Breakout → Ignore Signal
    if (close_prev < short_prev and close_curr > short_curr) and is_low_volume:
        logging.info("Low volume breakout detected. Ignoring weak BUY signal.")
        return None

    # 4. EMA Compression (Squeeze) → Trend Reversal Alert
    if ema_gap <= compression_threshold:
        logging.info("EMA compression detected. Potential breakout or reversal ahead. Signal: WATCH")
        return 'watch'  # New return statement added here

    # 5. Long Lower Wick Near EMA + High Volume → Buy Signal
    if lower_wick > upper_wick and abs(low_curr - short_curr) <= support_threshold and is_green_candle and is_high_volume:
        logging.info("Long lower wick near EMA with high volume. Strong BUY signal.")
        return 'buy'

    # 6. Long Upper Wick Near EMA + High Volume → Sell Signal
    if upper_wick > lower_wick and abs(high_curr - short_curr) <= support_threshold and is_red_candle and is_high_volume:
        logging.info("Long upper wick near EMA with high volume. Strong SELL signal.")
        return 'sell'

    # --- Existing Crossover Logic ---

    # Bearish crossover
#    if short_prev >= long_prev and short_curr < long_curr:
#       if short_curr < trend_curr and is_high_volume:
 #           logging.info("Bearish crossover with high volume. Signal: SELL")
  #         return 'sell'

    # Bullish crossover
 #   elif short_prev <= long_prev and short_curr > long_curr:
  #      if short_curr > trend_curr and is_high_volume:
  #          logging.info("Bullish crossover with high volume. Signal: BUY")
   #         return 'buy'

    return None




# Prepare LSTM Input Data
def prepare_lstm_input(data, scaler, n_steps=60):
    try:
        features = ['open', 'high', 'low', 'close', 'volume', 'MA_10', 'MA_30', 'RSI', 'ATR','EMA_12','EMA_26','MACD','EMA_7','EMA_25','EMA_99','Upper_Band','Lower_Band']
        scaled_data = scaler.transform(data[features].tail(n_steps))
        return scaled_data.reshape(1, n_steps, len(features))
    except Exception as e:
        logging.error(f"Error preparing LSTM input: {e}")
        return None

def place_order(symbol, side, size):
    """
    Place an order with stop-loss functionality.
    """
    try:
        # Fetch the current price for calculating stop loss
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']

        # Calculate stop price: 5% below for buy and 5% above for sell
        if side == 'buy':
            stop_price = current_price * 0.99  # Stop-loss price 2% below current price
        elif side == 'sell':
            stop_price = current_price * 1.01  # Stop-loss price 2% above current price
        else:
            raise ValueError(f"Invalid order side: {side}")

        # Ensure leverage is set
        exchange.set_leverage(LEVERAGE, symbol)

        # Place the market order
        market_order = exchange.create_order(symbol, 'market', side, size)
        logging.info(f"Market order placed: {side} {size} {symbol} at {current_price}")

        # Place the stop-loss order
        params = {'stopPrice': stop_price}
        stop_order = exchange.create_order(symbol, 'stop_market', 'sell' if side == 'buy' else 'buy', size, None, params)
        logging.info(f"Stop-loss order placed: {'sell' if side == 'buy' else 'buy'} {size} {symbol} with stop price: {stop_price}")

        return market_order, stop_order['id']

    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")
        return None, None

    

def cancel_order(order_id):
    """
    Cancel a specific order by ID.
    """
    try:
        exchange.cancel_order(order_id)
        logging.info(f"Stop-loss order {order_id} canceled successfully.")
    except Exception as e:
        logging.error(f"Error canceling order {order_id}: {e}")



#close order
def close_order(symbol, side, size):
    try:
        exchange.set_leverage(LEVERAGE, symbol)
        # exchange.create_order_with_take_profit_and_stop_loss()
        order = exchange.create_order(symbol, 'market', side, size)
        logging.info(f"Order placed: {side} {size} {symbol}")
        return order
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")





def validate_position_size(symbol, size, current_price):
    try:
        market = exchange.markets[symbol]  # Get market info for the trading pair
        min_qty = market['limits']['amount']['min']  # Minimum quantity allowed
        min_notional = market['limits']['cost']['min']  # Minimum notional value allowed (e.g., $10)

        # Adjust size if it doesn't meet the minimum quantity
        if size < min_qty:
            size = min_qty

        # Adjust size if it doesn't meet the minimum notional value
        if size * current_price < min_notional:
            size = min_notional / current_price

        return size
    except Exception as e:
        logging.error(f"Error validating position size for {symbol}: {e}")
        return 0


# Determine Trade Signal
def should_trade(symbol, model, scaler, data, balance):
    try:
        data = add_indicators(data)  # Add indicators before preparing input
        if data is None:
            logging.warning(f"Missing indicators for {symbol}. Skipping.")
            return None, 0

        current_price = data['close'].iloc[-1]
        lstm_input = prepare_lstm_input(data, scaler, n_steps=N_STEPS)
        if lstm_input is None:
            return None, 0

        predicted_price = 1
        #model.predict(lstm_input)[0][0]
        dummy_row = np.zeros((1, 17)) 
        dummy_row[0, 3] = predicted_price
        predicted_price = scaler.inverse_transform(dummy_row)[0][3]
        logging.info(f"LSTM Prediction for {symbol}: {predicted_price}, Current Price: {current_price}")

        position_size = (POSITION_SIZE_PERCENT * balance) / current_price
        position_size = validate_position_size(symbol, position_size, current_price)
        atr = data['ATR'].iloc[-1]
        buy_threshold = 1
        #1.002 + (atr / current_price * 0.005)  # Adjust by 10% of ATR
        sell_threshold = 1
        #0.998 - (atr / current_price * 0.005)

        crossover_signal = detect_crossover(data)

        logging.info(f"Trade conditions for {symbol} - Predicted: {predicted_price}, Current: {current_price}, MA_10: {data['MA_10'].iloc[-1]}, MA_30: {data['MA_30'].iloc[-1]}, RSI: {data['RSI'].iloc[-1]} , MACD : {data['MACD'].iloc[-1]}, Signal {data['Signal'].iloc[-1]}")
        logging.info(f"buy threshold {buy_threshold} - sell threshold {sell_threshold}")
        logging.info(f"cross over signal {crossover_signal}")
        # Remove the proximity condition for buy and sell
        # Buy Condition
        if ((crossover_signal == 'buy' )
           #   (predicted_price > (current_price * buy_threshold))
            #   and (crossover_signal == 'buy' ))
                
                #  or ((data['MA_10'].iloc[-1] > data['MA_30'].iloc[-1]) 
                #  and (data['MACD'].iloc[-1] > data['Signal'].iloc[-1]) 
                #and (30 < data['RSI'].iloc[-1] < 50) 
                and (data['MACD'].iloc[-1] > 0)
        ):
            return 'buy', position_size

        # Sell Condition
        elif ((crossover_signal == 'sell' )
            #(predicted_price < (current_price * sell_threshold))
            #     and (crossover_signal == 'sell' ))
                #  or ((data['MA_10'].iloc[-1] < data['MA_30'].iloc[-1]) 
                # and (data['MACD'].iloc[-1] < data['Signal'].iloc[-1])
                #and (data['RSI'].iloc[-1] > 65)
                and (data['MACD'].iloc[-1] < 0)
        ):
            return 'sell', position_size


        return None, 0
    except Exception as e:
        logging.error(f"Error determining trade signal for {symbol}: {e}")
        return None, 0

def monitor_positions():
    """
    Monitor open positions and close them when the profit target is achieved or ROI is below -15%.
    """
    try:
        positions = exchange.fetch_positions()
        for position in positions:
            if float(position['contracts']) > 0:  # Active positions only
                symbol = position['symbol']
                unrealized_profit = float(position['unrealizedPnl'])
                notional_value = float(position['initialMargin'])
                fee_adjusted_profit = PROFIT_TARGET_PERCENT - 0.001  # Account for fees

                logging.info(f"Monitoring {symbol}: Unrealized PnL={unrealized_profit}, Notional Value={notional_value}")

                # Close position if profit target is achieved or ROI is below -15%
                if (unrealized_profit >= notional_value * fee_adjusted_profit) or (unrealized_profit <= -notional_value * 0.10)  :
                    if unrealized_profit >= notional_value * fee_adjusted_profit:
                        logging.info(f"Profit target hit for {symbol}. Closing position.")
                    elif unrealized_profit <= -notional_value * 0.10:
                        logging.info(f"ROI below -15% for {symbol}. Closing position.")

                    side = 'sell' if position['side'] == 'long' else 'buy'
                    size = abs(float(position['contracts']))
                    
                    # Place the closing order
                    close_order = exchange.create_order(symbol, 'market', side, size)
                    logging.info(f"Position closed: {side} {size} {symbol}")

                    # Cancel any active stop-loss orders
                    open_orders = exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        if order['type'] == 'stop_market':
                            logging.info(f"Cancelling stop-loss order for {symbol}: {order['id']}")
                            exchange.cancel_order(order['id'], symbol)
    except Exception as e:
        logging.error(f"Error monitoring positions: {e}")



# Main Trading Function
def trade():
    logging.info("Starting bot...")
    while True:
        try:
            balance = fetch_wallet_balance()
            if balance > 0:
                for symbol in TRADING_PAIRS:
                    logging.info(f"Processing pair: {symbol}")
                    data = fetch_data(symbol, TIMEFRAME)
                    if data is not None:
                        model_path = f"models_lstm/lstm_{symbol.replace('/', '_')}.h5"
                        scaler_path = f"models_lstm/scaler_{symbol.replace('/', '_')}.pkl"

                        if os.path.exists(model_path) and os.path.exists(scaler_path):
                            model = load_model(model_path, compile=False)
                            scaler = joblib.load(scaler_path)
                            action, size = should_trade(symbol, model, scaler, data, balance)
                            if action == 'buy':
                                place_order(symbol, 'buy', size)
                            elif action == 'sell':
                                place_order(symbol, 'sell', size)
                        else:
                            logging.warning(f"No LSTM model or scaler found for {symbol}")

                # Monitor positions after trading
                # monitor_positions()
            else:
                logging.info("Insufficient balance. Waiting for funds.")
            time.sleep(25)  # Adjust as needed
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)

def monitor_thread():
    while True:
        try:
            monitor_positions()
            time.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logging.error(f"Error in monitor thread: {e}")
            time.sleep(10)

threading.Thread(target=monitor_thread, daemon=True).start()


if __name__ == "__main__":
    trade()

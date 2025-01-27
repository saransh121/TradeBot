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
LEVERAGE = 15
POSITION_SIZE_PERCENT = 0.35  # % of wallet balance to trade per coin
TIMEFRAME = '15m'
PROFIT_TARGET_PERCENT = 0.1  # 10% profit target
N_STEPS = 60  # For LSTM input sequence length

# Trading Pairs
def load_trading_pairs(file_path="trading_pairs.txt"):
    """
    Loads trading pairs from a text file.
    
    :param file_path: Path to the file containing trading pairs.
    :return: List of trading pairs.
    """
    try:
        with open(file_path, 'r') as file:
            pairs = [line.strip() for line in file if line.strip()]
            logging.info(f"Loaded trading pairs: {pairs}")
            return pairs
    except FileNotFoundError:
        logging.error(f"File {file_path} not found. Please check the file path.")
        return []
    except Exception as e:
        logging.error(f"Error loading trading pairs: {e}")
        return []

# Load trading pairs from file
TRADING_PAIRS = load_trading_pairs()


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
        #'EMA_10', 'EMA_25', 'EMA_50', 'EMA_100', 'EMA_200'
        data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
        data['EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()
        data['EMA_100'] = data['close'].ewm(span=100, adjust=False).mean()
        data['EMA_200'] = data['close'].ewm(span=200, adjust=False).mean()
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

import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# def detect_crossover(data, short_ema_col='EMA_7', long_ema_col='EMA_25', trend_ema_col='EMA_99'):
#     """
#     Optimized EMA crossover detection with support, trend, breakout, wick, and volume analysis.
    
#     :param data: DataFrame containing price, EMA, and volume columns.
#     :return: 'buy', 'sell', 'watch', or None.
#     """
#     if len(data) < 4:
#         return None  # Not enough data

#     # Extract EMA values
#     short_prev, short_curr = data[short_ema_col].iloc[-2], data[short_ema_col].iloc[-1]
#     long_prev, long_curr = data[long_ema_col].iloc[-2], data[long_ema_col].iloc[-1]
#     trend_curr = data[trend_ema_col].iloc[-1]

#     # Current and previous candle data
#     open_curr, close_curr, low_curr, high_curr = data['open'].iloc[-1], data['close'].iloc[-1], data['low'].iloc[-1], data['high'].iloc[-1]
#     close_prev = data['close'].iloc[-2]

#     # Volume data
#     volume_curr = data['volume'].iloc[-1]
#     avg_volume = data['volume'].iloc[-20:].mean()  # Average of last 20 periods

#     # EMA slopes
#     short_slope = short_curr - short_prev
#     long_slope = long_curr - long_prev

#     # Candle characteristics
#     is_red_candle = close_curr < open_curr
#     is_green_candle = close_curr > open_curr

#     # Support/Resistance threshold
#     support_threshold = 0.001 * close_curr  # 0.1% buffer

#     # EMA Compression threshold
#     ema_gap = abs(short_curr - long_curr)
#     compression_threshold = 0.0005 * close_curr  # 0.05% gap

#     # Wick sizes
#     upper_wick = high_curr - max(open_curr, close_curr)
#     lower_wick = min(open_curr, close_curr) - low_curr

#     # Candle body size for momentum check
#     body_size = abs(close_curr - open_curr)
#     avg_body_size = abs(data['close'].iloc[-5:] - data['open'].iloc[-5:]).mean()

#     # Volume conditions
#     is_high_volume = volume_curr > 1.1 * avg_volume  # 20% higher than average
#     is_low_volume = volume_curr < 0.9 * avg_volume   # 20% lower than average


#     wick_body_ratio = 1.5
#     # --- Enhanced Logic with Buffer and Momentum Analysis ---

#     # 1. High Volume Breakout Above EMA → Strong Buy (with Buffer and Momentum)
#     if (close_prev < short_prev and close_curr > short_curr * 1.001 and close_curr > long_curr * 1.001 
#              ):
#         logging.info("High volume breakout above EMA resistance with momentum. Strong BUY signal.")
#         return 'buy'

#     # 2. High Volume Breakdown Below EMA → Strong Sell (with Buffer and Momentum)
#     if (close_prev > short_prev and close_curr < short_curr * 0.999 and close_curr < long_curr * 0.999 
#              ):
#         logging.info("High volume breakdown below EMA support with momentum. Strong SELL signal.")
#         return 'sell'

#     # 3. Low Volume Breakout → Ignore Signal
#     if (close_prev < short_prev and close_curr > short_curr) and is_high_volume and is_green_candle:
#         logging.info("Green Candle after potential breakout")
#         return 'watch'
    
#     if (close_prev > short_prev and close_curr < short_curr)  and is_high_volume and is_red_candle:
#         logging.info("Red Candle after potential breakdown")
#         return 'watch'

#     # 4. EMA Compression (Squeeze) → Trend Reversal Alert
#     if ema_gap <= compression_threshold:
#         logging.info("EMA compression detected. Potential breakout or reversal ahead. Signal: WATCH")
#         return 'watch'

#     # 5. Long Lower Wick Near EMA + High Volume → Buy Signal
#     if (lower_wick > wick_body_ratio * body_size and  # Wick is 1.5x the body
#             abs(low_curr - short_curr) <= support_threshold and  # Close to EMA support
#               # Bullish candle
#             short_slope > 0 and long_slope > 0 and  # EMAs trending upward
#             is_high_volume):  # Confirmed by high volume
#         logging.info("Long lower wick near EMA with high volume and upward trend. Strong BUY signal.")
#         return 'buy'

#     # 6. Long Upper Wick Near EMA + High Volume → Sell Signal
#     if (upper_wick > wick_body_ratio * body_size and  # Wick is 1.5x the body
#             abs(high_curr - short_curr) <= support_threshold and  # Close to EMA resistance
#             short_slope < 0 and long_slope < 0 and  # EMAs trending downward
#             (is_high_volume or not is_low_volume)):  # Volume is high or average
#         logging.info("Long upper wick near EMA with volume confirmation and downward trend. Strong SELL signal.")
#         return 'sell'

#     return None


# new detect cross over logic
def detect_crossover(data):
    """
    Enhanced EMA Strategy with detailed debugging:
    - Double Moving Average Crossover
    - Golden Cross Strategy
    - Exponential Moving Average (EMA) Crossover
    - Triple Moving Average Crossover
    - Moving Average Ribbon
    - Pullback to EMA Strategy
    - EMA Dynamic Zone Strategy
    - EMA Breakout Strategy
    :param data: DataFrame containing EMA columns and price data.
    :return: 'buy', 'sell', or None.
    """
    try:
        # Extract EMA columns
        short_ema_1, long_ema_1 = 'EMA_10', 'EMA_50'  # For Double Moving Average Crossover
        short_ema_2, long_ema_2 = 'EMA_50', 'EMA_200'  # For Golden Cross Strategy
        triple_ema_short, triple_ema_mid, triple_ema_long = 'EMA_10', 'EMA_25', 'EMA_50'  # For Triple Crossover
        ema_columns = ['EMA_10', 'EMA_25', 'EMA_50', 'EMA_100', 'EMA_200']  # For Moving Average Ribbon

        # Extract latest values for all EMA columns
        short_ema_1_prev, short_ema_1_curr = data[short_ema_1].iloc[-2], data[short_ema_1].iloc[-1]
        long_ema_1_prev, long_ema_1_curr = data[long_ema_1].iloc[-2], data[long_ema_1].iloc[-1]

        short_ema_2_prev, short_ema_2_curr = data[short_ema_2].iloc[-2], data[short_ema_2].iloc[-1]
        long_ema_2_prev, long_ema_2_curr = data[long_ema_2].iloc[-2], data[long_ema_2].iloc[-1]

        triple_ema_short_prev, triple_ema_short_curr = data[triple_ema_short].iloc[-2], data[triple_ema_short].iloc[-1]
        triple_ema_mid_prev, triple_ema_mid_curr = data[triple_ema_mid].iloc[-2], data[triple_ema_mid].iloc[-1]
        triple_ema_long_prev, triple_ema_long_curr = data[triple_ema_long].iloc[-2], data[triple_ema_long].iloc[-1]

        close_prev, close_curr = data['close'].iloc[-2], data['close'].iloc[-1]

      

        # --- 1. Double Moving Average Crossover ---
        if short_ema_1_prev <= long_ema_1_prev and short_ema_1_curr > long_ema_1_curr:
            logging.info("Condition: Double Moving Average Crossover → BUY signal.")
            return 'buy'
        elif short_ema_1_prev >= long_ema_1_prev and short_ema_1_curr < long_ema_1_curr:
            logging.info("Condition: Double Moving Average Crossover → SELL signal.")
            return 'sell'

        # --- 2. Golden Cross Strategy ---
        elif short_ema_2_prev <= long_ema_2_prev and short_ema_2_curr > long_ema_2_curr:
            logging.info("Condition: Golden Cross Strategy → BUY signal.")
            return 'buy'
        elif short_ema_2_prev >= long_ema_2_prev and short_ema_2_curr < long_ema_2_curr:
            logging.info("Condition: Golden Cross Strategy → SELL signal.")
            return 'sell'

        # --- 3. Triple Moving Average Crossover ---
        elif (
            triple_ema_short_prev <= triple_ema_mid_prev <= triple_ema_long_prev and
            triple_ema_short_curr > triple_ema_mid_curr > triple_ema_long_curr
        ):
            logging.info("Condition: Triple EMA Crossover → BUY signal.")
            return 'buy'
        elif (
            triple_ema_short_prev >= triple_ema_mid_prev >= triple_ema_long_prev and
            triple_ema_short_curr < triple_ema_mid_curr < triple_ema_long_curr
        ):
            logging.info("Condition: Triple EMA Crossover → SELL signal.")
            return 'sell'

        # --- 4. Moving Average Ribbon ---
        ema_stack = data[ema_columns].iloc[-1]
        if ema_stack.is_monotonic_increasing:  # All EMAs are trending upward
            logging.info("Condition: Moving Average Ribbon → BUY signal.")
            return 'buy'
        elif ema_stack.is_monotonic_decreasing:  # All EMAs are trending downward
            logging.info("Condition: Moving Average Ribbon → SELL signal.")
            return 'sell'

        # --- 5. Pullback to EMA Strategy ---
        elif abs(close_curr - long_ema_1_curr) <= 0.002 * close_curr:  # Within 0.2% of EMA_50
            if close_curr > long_ema_1_curr and close_prev < long_ema_1_prev:  # Bullish bounce
                logging.info("Condition: Pullback to EMA_50 → BUY signal.")
                return 'buy'
            elif close_curr < long_ema_1_curr and close_prev > long_ema_1_prev:  # Bearish rejection
                logging.info("Condition: Pullback to EMA_50 → SELL signal.")
                return 'sell'

        # --- 6. EMA Dynamic Zone Strategy ---
        elif close_curr > max(ema_stack):  # Above all EMAs
            logging.info("Condition: EMA Dynamic Zone → BUY signal.")
            return 'buy'
        elif close_curr < min(ema_stack):  # Below all EMAs
            logging.info("Condition: EMA Dynamic Zone → SELL signal.")
            return 'sell'

        # --- 7. EMA Breakout Strategy ---
        elif close_prev < long_ema_2_prev and close_curr > long_ema_2_curr:  # Breakout above EMA_200
            logging.info("Condition: EMA Breakout → BUY signal.")
            return 'buy'
        elif close_prev > long_ema_2_prev and close_curr < long_ema_2_curr:  # Breakout below EMA_200
            logging.info("Condition: EMA Breakout → SELL signal.")
            return 'sell'

        # If no conditions met
        logging.info("No conditions met for BUY or SELL.")
        return None

    except Exception as e:
        logging.error(f"Error in enhanced EMA strategy: {e}")
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
            stop_price = current_price * 0.5  # Stop-loss price 2% below current price
        elif side == 'sell':
            stop_price = current_price * 1.5  # Stop-loss price 2% above current price
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

def confirm_trade_signal_with_atr(symbol, timeframe='5m', limit=14):
    """
    Confirms trade signal based on ATR buffer logic using the previous close and current open price.

    :param symbol: Trading pair symbol (e.g., 'BTC/USDT').
    :param timeframe: Timeframe for OHLCV data (default: '3m').
    :param limit: Number of candles to fetch for ATR calculation (default: 14).
    :return: 'buy', 'sell', or None (no signal).
    """
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        # ATR calculation (Average True Range)
        high_prices = [candle[2] for candle in ohlcv]
        low_prices = [candle[3] for candle in ohlcv]
        close_prices = [candle[4] for candle in ohlcv[:-1]]  # Exclude last candle for ATR

        tr_values = [
            max(high - low, abs(high - prev_close), abs(low - prev_close))
            for high, low, prev_close in zip(high_prices[1:], low_prices[1:], close_prices)
        ]
        atr = sum(tr_values) / len(tr_values)

        # Use the last two candles
        prev_candle = ohlcv[-2]  # Second-to-last candle (closed)
        curr_candle = ohlcv[-1]  # Most recent (forming) candle
        current_price = float(curr_candle[4])

        # Dynamic sensitivity factor based on price
        sensitivity_factor = 0.5 if current_price < 1 else 1.0
        dynamic_multiplier = (atr / current_price) * sensitivity_factor

        # Buffer based on ATR
        buffer = max(atr * dynamic_multiplier, 0.00007)

        # Signal logic for confirmation
        if prev_candle[4] > (current_price + buffer):  # Bearish confirmation
            return 'sell'
        elif prev_candle[4] < (current_price - buffer):  # Bullish confirmation
            return 'buy'

        return None  # No signal (neutral)

    except Exception as e:
        logging.error(f"Error in confirming trade signal with ATR buffer for {symbol}: {e}")
        return None


#new support resitance 
def support_resistance_signal(symbol, exchange=exchange, timeframe='15m', buffer=0.002, min_swing_distance=5):
    """
    Generates buy/sell signals based on support and resistance levels.
    
    :param symbol: The trading pair (e.g., 'DOGE/USDT').
    :param exchange: The exchange object (e.g., ccxt.binance instance).
    :param timeframe: The timeframe for the data (default: '5m').
    :param buffer: Buffer percentage to treat levels as zones (default: 0.2%).
    :param min_swing_distance: Minimum distance between swing points to avoid noise.
    :return: 'buy', 'sell', or None.
    """
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        # Identify recent swing highs and lows (with a minimum distance filter)
        data['Swing_High'] = data['high'][
            (data['high'] > data['high'].shift(1)) & 
            (data['high'] > data['high'].shift(-1))
        ]
        data['Swing_Low'] = data['low'][
            (data['low'] < data['low'].shift(1)) & 
            (data['low'] < data['low'].shift(-1))
        ]

        # Filter out swings that are too close (to avoid noise)
        recent_swing_highs = data['Swing_High'].dropna().iloc[-min_swing_distance:]
        recent_swing_lows = data['Swing_Low'].dropna().iloc[-min_swing_distance:]

        # Get the most recent swing high and low
        recent_resistance = recent_swing_highs.max() if not recent_swing_highs.empty else None
        recent_support = recent_swing_lows.min() if not recent_swing_lows.empty else None

        if recent_resistance is None or recent_support is None:
            logging.info(f"{symbol}: No valid support or resistance levels identified.")
            return None

        # Apply buffer zones around support and resistance
        resistance_zone = (
            recent_resistance - buffer * recent_resistance, 
            recent_resistance + buffer * recent_resistance
        )
        support_zone = (
            recent_support - buffer * recent_support, 
            recent_support + buffer * recent_support
        )
        
        
        # Fetch current price
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]


        # --- Buy Signal: Price near support and bouncing ---
        if support_zone[0] <= current_price <= support_zone[1]:
              # Bullish momentum
                logging.info(f"{symbol}: Buy signal detected. Price near support zone {support_zone}.")
                return 'buy'

        # --- Sell Signal: Price near resistance and rejecting ---
        if resistance_zone[0] <= current_price <= resistance_zone[1]:
             # Bearish momentum
                logging.info(f"{symbol}: Sell signal detected. Price near resistance zone {resistance_zone}.")
                return 'sell'

        # --- Breakout Buy Signal: Price breaks above resistance ---
        if current_price > resistance_zone[1] and previous_price <= resistance_zone[1]:
            logging.info(f"{symbol}: Breakout Buy signal detected. Price broke above resistance zone {resistance_zone}.")
            return 'buy'

        # --- Breakout Sell Signal: Price breaks below support ---
        if current_price < support_zone[0] and previous_price >= support_zone[0]:
            logging.info(f"{symbol}: Breakout Sell signal detected. Price broke below support zone {support_zone}.")
            return 'sell'

        # Default: No signal
        logging.info(f"{symbol}: No significant support-resistance signal generated.")
        return None

    except Exception as e:
        logging.error(f"Error in support_resistance_signal for {symbol}: {e}")
        return None


# Determine Trade Signal
def should_trade(symbol, model, scaler, data, balance):
    try:
        data = add_indicators(data)  # Add indicators before preparing input
        if data is None:
            logging.warning(f"Missing indicators for {symbol}. Skipping.")
            return None, 0

        current_price = data['close'].iloc[-1]
        #lstm_input =
        #prepare_lstm_input(data, scaler, n_steps=N_STEPS)
        # if lstm_input is None:
        #     return None, 0

        predicted_price = 1
        #model.predict(lstm_input)[0][0]
        dummy_row = np.zeros((1, 17)) 
        dummy_row[0, 3] = predicted_price
        predicted_price = 0
        #scaler.inverse_transform(dummy_row)[0][3]
        logging.info(f"LSTM Prediction for {symbol}: {predicted_price}, Current Price: {current_price}")

        position_size = ((POSITION_SIZE_PERCENT) * balance) / current_price
        position_size = position_size * LEVERAGE
        position_size = validate_position_size(symbol, position_size, current_price)
        atr = data['ATR'].iloc[-1]
        buy_threshold = 1
        #1.002 + (atr / current_price * 0.005)  # Adjust by 10% of ATR
        sell_threshold = 1
        #0.998 - (atr / current_price * 0.005)

        crossover_signal = detect_crossover(data)

        logging.info(f"Trade conditions for {symbol} - Predicted: {predicted_price}, Current: {current_price}, MA_10: {data['MA_10'].iloc[-1]}, MA_30: {data['MA_30'].iloc[-1]}, RSI: {data['RSI'].iloc[-1]} , MACD : {data['MACD'].iloc[-1]}, Signal {data['Signal'].iloc[-1]} , ATR Confimation {confirm_trade_signal_with_atr(symbol=symbol)}")
        logging.info(f"buy threshold {buy_threshold} - sell threshold {sell_threshold}")
        logging.info(f"cross over signal {crossover_signal}")
        # Remove the proximity condition for buy and sell
        # Buy Condition
        if ((crossover_signal == 'buy' 
            and confirm_trade_signal_with_atr(symbol=symbol) == 'buy'
            and (30 < data['RSI'].iloc[-1] < 50)
            )
            or (support_resistance_signal(symbol) == 'buy'
                and confirm_trade_signal_with_atr(symbol=symbol) == 'buy'
            )
           #   (predicted_price > (current_price * buy_threshold))
            #   and (crossover_signal == 'buy' ))
                
                #  or ((data['MA_10'].iloc[-1] > data['MA_30'].iloc[-1]) 
                #  and (data['MACD'].iloc[-1] > data['Signal'].iloc[-1]) 
                #and (30 < data['RSI'].iloc[-1] < 50) 
                and (data['MACD'].iloc[-1] > 0)
        ):
            return 'buy', position_size

        # Sell Condition
        elif ((crossover_signal == 'sell' 
              and confirm_trade_signal_with_atr(symbol=symbol) == 'sell'
              and (data['RSI'].iloc[-1] > 65)
              )
               or (support_resistance_signal(symbol) == 'sell'
                and confirm_trade_signal_with_atr(symbol=symbol) == 'sell'
            )
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
    Monitors open positions and manages them based on:
    1. ATR-based buffer for Previous Close + Current Open logic.
    2. Signal confirmation at -20% loss.
    3. A hard stop-loss at -30%.
    Cancels stop-loss orders only after closing positions.
    """
    try:
        positions = exchange.fetch_positions()
        for position in positions:
            if float(position['contracts']) > 0:  # Active positions only
                symbol = position['symbol']
                position_side = position['side']  # 'long' or 'short'

                # Fetch last 14 candles for ATR calculation
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=14)

                # ATR calculation (Average True Range)
                high_prices = [candle[2] for candle in ohlcv]
                low_prices = [candle[3] for candle in ohlcv]
                close_prices = [candle[4] for candle in ohlcv[:-1]]  # Exclude last candle for ATR

                tr_values = [
                    max(high - low, abs(high - prev_close), abs(low - prev_close))
                    for high, low, prev_close in zip(high_prices[1:], low_prices[1:], close_prices)
                ]
                atr = sum(tr_values) / len(tr_values)

                # Use the last two candles
                prev_candle = ohlcv[-2]  # Second-to-last candle (closed)
                curr_candle = ohlcv[-1]  # Most recent (forming) candle
                current_price = float(curr_candle[4])
                sensitivity_factor = 0.5 if current_price < 1 else 1.0
                dynamic_multiplier = (atr / current_price) * sensitivity_factor
                
                # Buffer based on ATR
                buffer = max(atr * dynamic_multiplier, 0.00007)
                
                #max(atr * dynamic_multiplier, 0.0001)  # Adjust multiplier as needed (e.g., 0.5x ATR)
                unrealized_profit = float(position['unrealizedPnl'])
                notional_value = float(position['initialMargin'])
                dynamic_profit_target = max(0.03, min(0.25, atr / notional_value * (LEVERAGE / 10)))
                logging.info(f"dynamic profit target for the coin {symbol} is {dynamic_profit_target}")

                # Function to close position and cancel stop-loss
                def close_position():
                    side = 'sell' if position_side == 'long' else 'buy'
                    size = abs(float(position['contracts']))
                    exchange.create_order(symbol, 'market', side, size)
                    logging.info(f"Position closed for {symbol}: {side} {size}")

                    # Cancel stop-loss orders after closing
                    open_orders = exchange.fetch_open_orders(symbol)
                    for order in open_orders:
                        if order['type'] == 'stop_market':
                            logging.info(f"Cancelling stop-loss order for {symbol}: {order['id']}")
                            exchange.cancel_order(order['id'], symbol)
                
                # 1️⃣ Previous Close + Current Open with ATR Buffer → Close Position
                if unrealized_profit >= notional_value * 0.025:  # Ensure the position is in profit
                    if position_side == 'long':
                        # 1️⃣ Check for trend reversal (bearish red candle for long)
                        if prev_candle[4] > (current_price + buffer):  # Previous close > Current open
                            logging.info(f"Reversal detected for {symbol} (long position). Closing position.")
                            close_position()
                            continue
                        #2️⃣ Fallback: Book profit at the dynamic profit target
                        elif  (unrealized_profit >= notional_value * dynamic_profit_target):
                            logging.info(f"Dynamic profit target ({dynamic_profit_target * 100}%) hit for {symbol}. Closing position as fallback.")
                            close_position()
                            continue
                        else:
                            # 3️⃣ Hold position if trend continues
                            logging.info(f"Trend continues for {symbol} (long position). Holding position.")

                    elif position_side == 'short':
                        # 1️⃣ Check for trend reversal (bullish green candle for short)
                        if prev_candle[4] < (current_price -buffer):  # Previous close < Current open
                            logging.info(f"Reversal detected for {symbol} (short position). Closing position.")
                            close_position()
                            continue
                        #2️⃣ Fallback: Book profit at the dynamic profit target
                        elif  unrealized_profit >= notional_value * dynamic_profit_target:
                            logging.info(f"Dynamic profit target ({dynamic_profit_target * 100}%) hit for {symbol}. Closing position as fallback.")
                            close_position()
                            continue
                        else:
                            # 3️⃣ Hold position if trend continues
                            logging.info(f"Trend continues for {symbol} (short position). Holding position.")


                # 3️⃣ Full Stop-Loss at -30% → Force Close
                if float(position['unrealizedPnl']) <= -float(position['initialMargin']) * 0.6:
                    logging.info(f"Hard stop-loss hit for {symbol}. Forcing close at -10%.")
                    close_position()
                    continue
                elif float(position['unrealizedPnl']) <= -float(position['initialMargin']) * 0.5:
                    logging.info(f"{symbol} hit -20% loss. Checking if we should close or hold.")

                    if position_side == 'long':
                        if (prev_candle[4] > (current_price + buffer )):  # Previous close > current open + ATR-based buffer
                            logging.info(f"Bearish reversal with ATR buffer detected for {symbol}. Closing long position.")
                            logging.info(f"Closing position for {symbol}: prev_candle[4]={prev_candle[4]}, curr_candle[1]={current_price}, buffer={buffer}")
                            close_position()
                            continue
                    elif position_side == 'short':
                        if (prev_candle[4] < (current_price - buffer)):  # Previous close < current open - ATR-based buffer
                            logging.info(f"Closing position for {symbol}: prev_candle[4]={prev_candle[4]}, curr_candle[1]={current_price}, buffer={buffer}")
                            logging.info(f"Bullish reversal with ATR buffer detected for {symbol}. Closing short position.")
                            close_position()
                            continue
                elif float(position['unrealizedPnl']) <= -float(position['initialMargin']) * 0.4:
                    logging.info(f"{symbol} hit -10% loss. Checking if we should close or hold.")
                    # Recheck signal on a shorter timeframe (5m)
                    new_data = fetch_data(symbol, '5m')
                    if new_data is not None and not new_data.empty:
                        new_action, _ = should_trade(symbol, None, 0, new_data, fetch_wallet_balance())

                        # Close if reversal is detected
                        if (position_side == 'long' and new_action == 'sell') or (position_side == 'short' and new_action == 'buy'):
                            logging.info(f"Rechecked signal suggests reversal for {symbol}. Closing position.")
                            close_position()
                            continue
                        else:
                            logging.info(f"Signal suggests holding {symbol}. Waiting for recovery.")

    except Exception as e:
        logging.error(f"Error monitoring positions: {e}")






# Main Trading Function
# def trade():
#     logging.info("Starting bot...")
#     while True:
#         try:
#             balance = fetch_wallet_balance()
#             if balance > 0:
#                 for symbol in TRADING_PAIRS:
#                     logging.info(f"Processing pair: {symbol}")
#                     data = fetch_data(symbol, TIMEFRAME)
#                     if data is not None:
#                         model_path = f"models_lstm/lstm_{symbol.replace('/', '_')}.h5"
#                         scaler_path = f"models_lstm/scaler_{symbol.replace('/', '_')}.pkl"
#                         #os.path.exists(model_path) and os.path.exists(scaler_path)
#                         if data is not None:
#                             model = None
#                             scaler = None
#                             action, size = should_trade(symbol, model, 0, data, balance)
#                             if action == 'buy':
#                                 place_order(symbol, 'buy', size)
#                             elif action == 'sell':
#                                 place_order(symbol, 'sell', size)
#                         else:
#                             logging.warning(f"No LSTM model or scaler found for {symbol}")

#                 # Monitor positions after trading
#                 # monitor_positions()
#             else:
#                 logging.info("Insufficient balance. Waiting for funds.")
#             time.sleep(25)  # Adjust as needed
#         except Exception as e:
#             logging.error(f"Error in main loop: {e}")
#             time.sleep(10)

#new Trade Logic
last_trade_time = {}
cooldown_period = 240  # 4-minute cooldown between trades on the same symbol
max_retries = 4
retry_counter = {}

# Main Trading Function
def trade():
    logging.info("Starting bot...")

    while True:
        try:
            balance = fetch_wallet_balance()

            if balance > 0:
                for symbol in TRADING_PAIRS:
                    logging.info(f"Processing pair: {symbol}")

                    # Check cooldown for the symbol
                    current_time = time.time()
                    if symbol in last_trade_time and current_time - last_trade_time[symbol] < cooldown_period:
                        logging.info(f"Cooldown active for {symbol}. Skipping this round.")
                        continue  # Skip this symbol due to cooldown

                    # Fetch market data
                    data = fetch_data(symbol, TIMEFRAME)
                    if data is None:
                        logging.warning(f"Failed to fetch data for {symbol}")
                        continue

                    # Load model and scaler
                    model_path = f"models_lstm/lstm_{symbol.replace('/', '_')}.h5"
                    scaler_path = f"models_lstm/scaler_{symbol.replace('/', '_')}.pkl"

                    if data is not None:
                        model = None  # Replace with model loading logic
                        scaler = None  # Replace with scaler loading logic

                        # Get trading action
                        action, size = should_trade(symbol, model, 0, data, balance)

                        # Check signal confirmation
                        confirmed_action = confirm_signal(symbol, action, data)

                        # Place the trade if confirmed
                        if confirmed_action == 'buy':
                            if place_order(symbol, 'buy', size):
                                last_trade_time[symbol] = current_time  # Reset cooldown
                                retry_counter[symbol] = 0  # Reset retries
                        elif confirmed_action == 'sell':
                            if place_order(symbol, 'sell', size):
                                last_trade_time[symbol] = current_time
                                retry_counter[symbol] = 0
                        else:
                            logging.info(f"No confirmed signal for {symbol}. Skipping trade.")
                    else:
                        logging.warning(f"No LSTM model or scaler found for {symbol}")

                # Monitor open positions after trading
                monitor_positions()
            else:
                logging.info("Insufficient balance. Waiting for funds.")
                time.sleep(60)  # Longer wait on low balance

            time.sleep(15)  # Regular wait before checking again

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)

# Signal confirmation function (checks signal consistency)
def confirm_signal(symbol, action, data):
    """
    Confirms if the trading signal is consistent over two consecutive checks.
    """
    time.sleep(10)  # Wait before re-checking the signal
    new_data = fetch_data(symbol, TIMEFRAME)
    if new_data is not None:
        new_action, _ = should_trade(symbol, None, 0, new_data, fetch_wallet_balance())
        if new_action == action:
            return action  # Confirmed signal
    return None  # Signal was not consistent


def monitor_thread():
    while True:
        try:
            monitor_positions()
            time.sleep(2)  # Check every 5 seconds
        except Exception as e:
            logging.error(f"Error in monitor thread: {e}")
            time.sleep(10)

threading.Thread(target=monitor_thread, daemon=True).start()


if __name__ == "__main__":
    trade()

import ccxt
import time
import pandas as pd
import numpy as np
import logging
import joblib
import os
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
LEVERAGE = 25
POSITION_SIZE_PERCENT = 0.25  # % of wallet balance to trade per coin
TIMEFRAME = '3m'
PROFIT_TARGET_PERCENT = 0.05  # 5% profit target
N_STEPS = 60  # For LSTM input sequence length

# Trading Pairs
TRADING_PAIRS = ["XRP/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT"]

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
        return data.dropna()
    except Exception as e:
        logging.error(f"Error adding indicators: {e}")
        return None

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

# Prepare LSTM Input Data
def prepare_lstm_input(data, scaler, n_steps=60):
    try:
        features = ['open', 'high', 'low', 'close', 'volume', 'MA_10', 'MA_30', 'RSI', 'ATR']
        scaled_data = scaler.transform(data[features].tail(n_steps))
        return scaled_data.reshape(1, n_steps, len(features))
    except Exception as e:
        logging.error(f"Error preparing LSTM input: {e}")
        return None

# Place Order
def place_order(symbol, side, size):
    try:
        exchange.set_leverage(LEVERAGE, symbol)
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

        predicted_price = model.predict(lstm_input)[0][0]
        dummy_row = np.zeros((1, 9)) 
        dummy_row[0, 3] = predicted_price
        predicted_price = scaler.inverse_transform(dummy_row)[0][3]
        logging.info(f"LSTM Prediction for {symbol}: {predicted_price}, Current Price: {current_price}")

        position_size = (POSITION_SIZE_PERCENT * balance) / current_price
        position_size = validate_position_size(symbol, position_size, current_price)

        logging.info(f"Trade conditions for {symbol} - Predicted: {predicted_price}, Current: {current_price}, MA_10: {data['MA_10'].iloc[-1]}, MA_30: {data['MA_30'].iloc[-1]}, RSI: {data['RSI'].iloc[-1]}")

        # Buy Condition
        if predicted_price > current_price and (data['MA_10'].iloc[-1] > data['MA_30'].iloc[-1]) and (data['RSI'].iloc[-1] < 35):
            return 'buy', position_size
        # Sell Condition
        elif predicted_price < current_price and data['RSI'].iloc[-1] > 68:
            return 'sell', position_size
        return None, 0
    except Exception as e:
        logging.error(f"Error determining trade signal for {symbol}: {e}")
        return None, 0

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
                            # print(f"{action} : {size}")
                            if action == 'buy':
                                place_order(symbol, 'buy', size)
                            elif action == 'sell':
                                place_order(symbol, 'sell', size)
                        else:
                            logging.warning(f"No LSTM model or scaler found for {symbol}")
            else:
                logging.info("Insufficient balance. Waiting for funds.")
            time.sleep(30)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    trade()

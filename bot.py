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
PROFIT_TARGET_PERCENT = 0.05  # 1% profit target
N_STEPS = 60  # For LSTM input sequence length

# Trading Pairs
TRADING_PAIRS = ["XRP/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT","COW/USDT"]

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

# Prepare LSTM Input Data
def prepare_lstm_input(data, scaler, n_steps=60):
    scaled_data = scaler.transform(data[['open', 'high', 'low', 'close', 'volume']])
    input_data = np.array([scaled_data[-n_steps:]]).reshape(1, n_steps, 5)  # Fixed shape
    return input_data

# Place Order
def place_order(symbol, side, size):
    try:
        exchange.set_leverage(LEVERAGE, symbol)
        order = exchange.create_order(symbol, 'market', side, size)
        logging.info(f"Order placed: {side} {size} {symbol}")
        return order
    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}")

# Monitor Open Positions
def monitor_positions():
    try:
        positions = exchange.fetch_positions()
        for position in positions:
            # print(position)
            if float(position['contracts']) > 0:  # Active positions only
                symbol = position['symbol']
                unrealized_profit = float(position['unrealizedPnl'])
                notional_value = float(position['initialMargin'])
                logging.info(f"Monitoring {symbol}: Unrealized PnL={unrealized_profit}, Notional={notional_value}")

                if unrealized_profit >= notional_value * PROFIT_TARGET_PERCENT:
                    logging.info(f"Profit target hit for {symbol}. Closing position.")
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    place_order(symbol, side, abs(float(position['contracts'])))
    except Exception as e:
        logging.error(f"Error monitoring positions: {e}")

# Determine Trade Signal
def should_trade(symbol, model, scaler, data, balance):
    current_price = data['close'].iloc[-1]
    lstm_input = prepare_lstm_input(data, scaler, n_steps=N_STEPS)
    predicted_price = model.predict(lstm_input)[0][0]  # Predict without retracing
    logging.info(f"LSTM Prediction for {symbol}: {predicted_price}, Current Price: {current_price}")

    # Indicators
    data['MA_10'] = data['close'].rolling(window=10).mean()
    data['MA_30'] = data['close'].rolling(window=30).mean()
    data['RSI'] = calculate_rsi(data['close'])

    position_size = (POSITION_SIZE_PERCENT * balance) / current_price
    logging.info(f"current price {current_price} , buy condition predicted_price:{predicted_price} , data1{data['MA_10'].iloc[-1]} , data2 {data['MA_30'].iloc[-1]} , RSI {data['RSI'].iloc[-1]} ")
    # Buy Condition
    if predicted_price > current_price and (data['MA_10'].iloc[-1] > data['MA_30'].iloc[-1]) and (data['RSI'].iloc[-1] < 35):
        return 'buy', position_size
    # Sell Condition
    elif predicted_price < current_price and data['RSI'].iloc[-1] > 68:
        return 'sell', position_size
    return None, 0

# Calculate RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
                monitor_positions()
            else:
                logging.info("Insufficient balance. Waiting for funds.")
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    trade()

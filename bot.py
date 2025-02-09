import ccxt
import time
import pandas as pd
import numpy as np
import logging
import os
import threading
from collections import defaultdict
from dotenv import load_dotenv
from scipy.signal import argrelextrema
from dotenv import load_dotenv
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import Huber
from keras.layers import Bidirectional, Attention, Conv1D, Flatten
from keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten, BatchNormalization, Bidirectional,Multiply
import tensorflow as tf

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import logging
import time

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
LEVERAGE = 10
POSITION_SIZE_PERCENT = 0.2  # % of wallet balance to trade per coin
TIMEFRAME = '15m'
PROFIT_TARGET_PERCENT = 0.1  # 10% profit target
N_STEPS = 60  # For LSTM input sequence length

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
import logging
import time

class CryptoTradingEnv(gym.Env):
    LEVERAGE = 10  # 20x Leverage
    TRADING_FEE_PERCENT = 0.04 / 100  # 0.04% Taker Fee (Binance)

    def __init__(self, exchange, symbol, timeframe='15m'):
        super(CryptoTradingEnv, self).__init__()

        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

        # Actions: Hold (0), Buy (1), Sell (2)
        self.action_space = spaces.Discrete(3)

        # Features: Closing price, volume, RSI, MACD, ATR, Bollinger, EMA, etc.
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        self.current_step = 0
        self.data = self.fetch_data()  # Load initial market data
        self.balance = 100  # Simulated starting balance
        self.position = 0  # 0: No position, 1: Long, -1: Short

    def fetch_data(self):
        """Fetch latest OHLCV and indicators for training."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1000)
            data = np.array(ohlcv)

            if len(data) < 125:
                logging.warning(f"Not enough historical data for {self.symbol}. Required: 125, Available: {len(data)}")
                return None

            # Extract OHLCV features
            close = data[:, 4]
            volume = data[:, 5]

            # Compute indicators
            rsi = self.calculate_rsi_rl(close)
            macd, signal = self.calculate_macd_rl(close)
            atr = self.calculate_atr_rl(data)
            ema_50 = self.calculate_ema_rl(close, 50)
            ema_200 = self.calculate_ema_rl(close, 200)
            upper_band, lower_band = self.calculate_bollinger_bands_rl(close)

            # Ensure all indicators have the same length
            min_length = min(len(close), len(volume), len(rsi), len(macd), len(signal), len(atr), len(ema_50), len(ema_200), len(upper_band), len(lower_band))
            close, volume = close[-min_length:], volume[-min_length:]
            rsi, macd, signal = rsi[-min_length:], macd[-min_length:], signal[-min_length:]
            atr, ema_50, ema_200 = atr[-min_length:], ema_50[-min_length:], ema_200[-min_length:]
            upper_band, lower_band = upper_band[-min_length:], lower_band[-min_length:]

            # Stack indicators
            indicators = np.column_stack((close, volume, rsi, macd, signal, atr, ema_50, ema_200, upper_band, lower_band))

            return indicators

        except Exception as e:
            logging.error(f"Error fetching data for {self.symbol}: {e}")
            return None

    def step(self, action):
        """Perform an action and return the new state, reward, and done flag."""
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        reward = self.get_reward(action)  # Calculate reward
        obs = self.get_observation()  # Get next observation

        return obs, reward, done, {}

    def reset(self):
        """Reset environment at the start of a new episode."""
        self.current_step = 0
        self.balance = 100
        self.position = 0
        self.data = self.fetch_data()
        return self.get_observation()

    def get_observation(self):
        """Return real-time indicators (normalized between -1 and 1)."""
        obs = self.data[self.current_step] if self.current_step < len(self.data) else np.zeros(10)
        return np.interp(obs, (np.min(obs), np.max(obs)), (-1, 1))

    def get_reward(self, action):
        """Reward function considering leverage and trading fees."""
        if self.current_step == 0:
            return 0  # No reward for the first step

        # **Price movement calculation**
        price_change = (self.data[self.current_step, 0] - self.data[self.current_step - 1, 0])

        # **ATR-based volatility adjustment**
        atr = self.calculate_atr_rl(self.data, period=14)[-1]
        atr = max(atr, 1e-6)  # Avoid division by zero

        # **Trading fee per trade**
        trading_fee = 0
        if action in [1, 2]:  # Buy or Sell
            trading_fee = abs(price_change) * self.LEVERAGE * self.TRADING_FEE_PERCENT  # Apply fee

        # **Leverage application only on buy/sell actions**
        price_change *= self.LEVERAGE if action in [1, 2] else 1

        # **Final adjusted reward**
        raw_reward = (price_change * 100) - trading_fee  # Fee deducted only for trades

        # **Risk-adjusted reward**
        risk_adjusted_reward = raw_reward / (atr * 10)

        # **Penalty for Large Losses**
        if risk_adjusted_reward < -1:
            risk_adjusted_reward *= 1.2  # Apply a larger penalty

        # **Encourage Holding in Strong Trends**
        if action == 0 and self.current_step >= 5:
            trend = self.data[self.current_step, 0] - self.data[self.current_step - 5, 0]
            if abs(trend) > atr * 1.5:
                return 0.3  # Increased reward for strong trend holding

        # **Final Reward Adjustment Based on Action**
        if action == 1:  # BUY
            self.position = 1
            return risk_adjusted_reward
        elif action == 2:  # SELL
            self.position = -1
            return -risk_adjusted_reward

        return 0


    def calculate_rsi_rl(self, prices, period=14):
        """Calculate Relative Strength Index (RSI)."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)  # Return array of NaNs if insufficient data

        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')

        rs = avg_gain / (avg_loss + 1e-9)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Pad the beginning with NaNs to maintain the same length as `prices`
        return np.concatenate([np.full(period-1, np.nan), rsi])


    def calculate_macd_rl(self, prices, short_window=12, long_window=26, signal_window=9):
        """Calculate MACD and Signal Line with proper length handling."""
        
        if len(prices) < long_window:
            return np.full(len(prices), np.nan), np.full(len(prices), np.nan)  # Handle short data case

        ema_short = self.calculate_ema_rl(prices, short_window)
        ema_long = self.calculate_ema_rl(prices, long_window)

        # **Ensure same length by padding the shorter EMA with NaNs**
        diff_length = len(ema_long) - len(ema_short)
        if diff_length > 0:
            ema_short = np.pad(ema_short, (diff_length, 0), mode='constant', constant_values=np.nan)
        elif diff_length < 0:
            ema_long = np.pad(ema_long, (-diff_length, 0), mode='constant', constant_values=np.nan)

        # Compute MACD and Signal
        macd = ema_short - ema_long
        signal = self.calculate_ema_rl(macd, signal_window)

        # **Ensure all arrays match `prices` length**
        min_length = min(len(prices), len(macd), len(signal))
        
        return macd[-min_length:], signal[-min_length:]



    def calculate_atr_rl(self, data, period=14):
        """Calculate Average True Range (ATR)."""
        if len(data) < period:
            return np.full(len(data), np.nan)  # Handle short data case

        high_low = data[:, 2] - data[:, 3]
        high_close = np.abs(data[:, 2] - np.roll(data[:, 4], shift=1))
        low_close = np.abs(data[:, 3] - np.roll(data[:, 4], shift=1))

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.convolve(true_range, np.ones(period) / period, mode='valid')

        return np.concatenate([np.full(period-1, np.nan), atr])


    def calculate_ema_rl(self, prices, period):
        """Calculate Exponential Moving Average (EMA)."""
        if len(prices) < period:
            return np.zeros(len(prices))
        return np.convolve(prices, np.ones(period) / period, mode='valid')

    def calculate_bollinger_bands_rl(self, prices, window=20):
        """Calculate Bollinger Bands with shape correction."""
        if len(prices) < window:
            logging.warning(f"Not enough data for Bollinger Bands. Required: {window}, Available: {len(prices)}")
            return np.zeros(len(prices)), np.zeros(len(prices))

        sma = np.convolve(prices, np.ones(window) / window, mode='valid')
        std = np.std(prices[-window:])

        # Ensure equal length
        min_len = min(len(sma), len(prices[-window:]))
        sma, std = sma[-min_len:], np.full(min_len, std)

        return sma + (2 * std), sma - (2 * std)

    




def fetch_top_movers(limit=6):
    """
    Fetch top high-volume coins based on 24h trading volume and filter coins priced below $10.

    :param limit: Number of coins to return.
    :return: List of selected trading pairs.
    """
    try:
        tickers = exchange.fetch_tickers()

        volume_list = []
        for symbol, data in tickers.items():
            if "USDT" in symbol and isinstance(data, dict) and 'quoteVolume' in data and 'last' in data:
                volume = float(data['quoteVolume'])  # 24h trading volume in quote currency
                price = float(data['last'])  # Current price of the asset
                
                if price < 10:  # Only select coins priced below $10
                    volume_list.append((symbol, volume))

        # Sort by highest volume
        volume_list = sorted(volume_list, key=lambda x: x[1], reverse=True)

        # Select top `limit` symbols
        top_symbols = [symbol for symbol, _ in volume_list[:limit]]

        logging.info(f"Top high-volume coins (below $10) selected: {top_symbols}")
        return top_symbols

    except Exception as e:
        logging.error(f"Error fetching high-volume coins: {e}")
        return []





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
print(fetch_top_movers())
TRADING_PAIRS = (fetch_top_movers())





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

        # Ensure we always have `N_STEPS` rows, otherwise pad with previous values
        if len(df) < N_STEPS:
            logging.warning(f"Not enough historical data for {symbol}. Required: {N_STEPS}, Available: {len(df)}")
            
            # Pad with last available row
            last_row = df.iloc[-1] if not df.empty else None
            while len(df) < N_STEPS:
                df = pd.concat([df, pd.DataFrame([last_row])], ignore_index=True)

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
def prepare_lstm_input(data, scaler, n_steps=N_STEPS):
    try:
        features = ['open', 'high', 'low', 'close', 'volume']
        scaled_data = scaler.fit_transform(data[features].tail(n_steps))
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


# AI Model Setup
def attention_layer(inputs):
    attention_probs = Dense(inputs.shape[-1], activation='softmax')(inputs)
    attention_output = Multiply()([inputs, attention_probs])
    return attention_output

scaler = MinMaxScaler()

# Input layer
input_layer = Input(shape=(N_STEPS, 5))

# Conv1D for short-term pattern detection
conv_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
conv_layer = BatchNormalization()(conv_layer)  # Normalization added

# First Bi-LSTM Layer
lstm_layer = Bidirectional(LSTM(100, return_sequences=True))(conv_layer)
lstm_layer = Dropout(0.2)(lstm_layer)

# Second Bi-LSTM Layer
lstm_layer = Bidirectional(LSTM(100, return_sequences=True))(lstm_layer)
lstm_layer = Dropout(0.2)(lstm_layer)

# ✅ Attention Layer with Dropout
attention = Attention()([lstm_layer, lstm_layer])
attention = Dropout(0.2)(attention)  # Added dropout

# Flatten to connect to Dense layers
flat = Flatten()(attention)

# Dense Layers with L2 Regularization
dense_layer = Dense(50, activation='relu', kernel_regularizer=l2(0.01))(flat)
dense_layer = Dropout(0.2)(dense_layer)  # Dropout added for stability

# Output Layer
output_layer = Dense(1)(dense_layer)

# Define & compile model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(delta=1.0))





# AI-Based Prediction with Proper Rescaling
def ai_predict(data):
    try:
        lstm_input = prepare_lstm_input(data, scaler)
        if lstm_input is not None:
            scaled_prediction = model.predict(lstm_input)[0][0]

            # Create a dummy row with same shape as input features
            dummy_row = np.zeros((1, 5))  # Adjust to match the number of input features
            dummy_row[0, 3] = scaled_prediction  # 'close' price is typically the 3rd index

            # Rescale back to actual price range
            predicted_price = scaler.inverse_transform(dummy_row)[0, 3]
            
            logging.info(f"AI Scaled Prediction: {scaled_prediction}, Rescaled Prediction: {predicted_price}")
            return predicted_price
    except Exception as e:
        logging.error(f"Error in AI prediction: {e}")
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
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=600)
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
        
        #--- Breakout Buy Signal: Price breaks above resistance ---
        # if current_price > resistance_zone[1] and previous_price <= resistance_zone[1]:
        #     logging.info(f"{symbol}:  Buy signal detected. Price broke above resistance zone {resistance_zone}.")
        #     return 'buy'

        # # --- Breakout Sell Signal: Price breaks below support ---
        # if current_price < support_zone[0] and previous_price >= support_zone[0]:
        #     logging.info(f"{symbol}:  Sell signal detected. Price broke below support zone {support_zone}.")
        #     return 'sell'

        # Default: No signal
        logging.info(f"{symbol}: No significant support-resistance signal generated.")
        return None

    except Exception as e:
        logging.error(f"Error in support_resistance_signal for {symbol}: {e}")
        return None


#new Function
def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def support_resistance_signal_new(
    symbol: str,
    exchange= exchange,
    timeframe: str = '15m',
    min_swing_distance: int = 5,
    atr_multiplier: float = 1,
    volume_threshold: float = 1.8
) :
    """
    Enhanced Support/Resistance Signal Generator with:
    - Dynamic ATR-based buffers
    - Volume confirmation
    - Trend filtering
    - Breakout detection
    
    Returns: 'buy', 'sell', or None
    """
    try:
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=600)
        if len(ohlcv) < 100:
            logging.warning(f"Insufficient data for {symbol}")
            return None

        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        # Calculate technical indicators
        data['ATR'] = calculate_atr(data)
        data['EMA_50'] = calculate_ema(data['close'], 50)
        
        # Identify swing points with minimum distance filter
        data['Swing_High'] = data['high'][
            (data['high'] > data['high'].shift(1)) & 
            (data['high'] > data['high'].shift(-1)) &
            (data['high'].rolling(3).max() == data['high'])
        ]
        
        data['Swing_Low'] = data['low'][
            (data['low'] < data['low'].shift(1)) & 
            (data['low'] < data['low'].shift(-1)) &
            (data['low'].rolling(3).min() == data['low'])
        ]

        # Get recent swing points
        swing_highs = data['Swing_High'].dropna().iloc[-min_swing_distance:]
        swing_lows = data['Swing_Low'].dropna().iloc[-min_swing_distance:]

        if swing_highs.empty or swing_lows.empty:
            logging.info(f"{symbol}: No clear swing points detected")
            return None

        recent_resistance = swing_highs.max()
        recent_support = swing_lows.min()
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]

        # Dynamic buffer calculation
        atr = data['ATR'].iloc[-1]
        buffer = atr * atr_multiplier

        # Define zones with ATR-based buffers
        resistance_zone = (
            recent_resistance - buffer,
            recent_resistance + buffer
        )
        support_zone = (
            recent_support - buffer,
            recent_support + buffer
        )

        # Trend determination
        trend = 'bullish' if current_price > data['EMA_50'].iloc[-1] else 'bearish'

        # Volume confirmation check
        volume_ok = current_volume >= avg_volume * volume_threshold

        # Signal logic
        signals = []
        
        # Support bounce with trend confirmation
        if (support_zone[0] <= current_price <= support_zone[1] and
            trend == 'bullish' and
            volume_ok):
            signals.append(('buy', 'support_bounce'))
            
        # Resistance rejection with trend confirmation
        if (resistance_zone[0] <= current_price <= resistance_zone[1] and
            trend == 'bearish' and
            volume_ok):
            signals.append(('sell', 'resistance_rejection'))
            
        # Breakout signals
        if current_price > resistance_zone[1] and volume_ok:
            signals.append(('buy', 'breakout'))
            
        if current_price < support_zone[0] and volume_ok:
            signals.append(('sell', 'breakdown'))

        # Prioritize signals (breakouts > bounces/rejections)
        signal_strength = {'breakout': 2, 'breakdown': 2, 
                          'support_bounce': 1, 'resistance_rejection': 1}
        
        if signals:
            # Select strongest signal
            signals.sort(key=lambda x: -signal_strength[x[1]])
            decision = signals[0][0]
            
            # Final confirmation with price action
            prev_candle = data.iloc[-2]
            curr_candle = data.iloc[-1]
            
            if decision == 'buy':
                if curr_candle['close'] > curr_candle['open']:  # Bullish candle
                    logging.info(f"BUY {symbol} | Reason: {signals[0][1]} | "
                                f"Price: {current_price:.5f} | Vol: {current_volume/avg_volume:.1f}x")
                    return 'buy'
                
            elif decision == 'sell':
                if curr_candle['close'] < curr_candle['open']:  # Bearish candle
                    logging.info(f"SELL {symbol} | Reason: {signals[0][1]} | "
                                f"Price: {current_price:.5f} | Vol: {current_volume/avg_volume:.1f}x")
                    return 'sell'

        logging.info(f"{symbol}: No qualified signals | "
                    f"Trend: {trend} | Vol Ratio: {current_volume/avg_volume:.1f}")
        return None

    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        return None




# all break out patterns

def detect_breakout_patterns(symbol: str, exchange: ccxt.Exchange = exchange, timeframe: str = '1h', 
                            confirmation_candles: int = 3, min_pattern_length: int = 10):
    """
    Enhanced breakout pattern detection with 12 major patterns:
    - Horizontal Breakout
    - Trendline Breakout
    - Ascending/Descending Triangle
    - Flag/Pennant
    - Order Blocks
    - Three White Soldiers/Black Crows
    - Volume Spike Breakouts
    - Cup & Handle (NEW)
    - Head & Shoulders (NEW)
    - Rectangle Patterns
    - Wedge Breakouts
    - Channel Breakouts
    
    Returns: 'buy', 'sell' with confidence score (1-5), or None
    """
    try:
        # Fetch and prepare data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=300)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Calculate technical indicators
        data['ATR'] = calculate_atr(data)
        data['EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()
        data['VWAP'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
        
        # Pattern weights dictionary
        PATTERN_WEIGHTS = {
            # Existing patterns
            'horizontal_breakout': 2.5,
            'trendline_breakout': 3.0,
            'ascending_triangle': 2.8,
            'descending_triangle': 2.8,
            'bull_flag': 3.2,
            'bear_flag': 3.2,
            'order_block': 2.0,
            'three_white_soldiers': 2.5,
            'three_black_crows': 2.5,
            'volume_spike': 3.5,
            # New patterns
            'cup_handle': 3.8,
            'head_shoulders': 4.0,
            'rectangle': 2.2,
            'wedge': 2.5,
            'channel': 2.3
        }

        # Initialize pattern storage with confidence scores
        patterns = {
            'bullish': defaultdict(float),
            'bearish': defaultdict(float)
        }

        # Helper functions
        def is_consecutive(prices, window, direction='up'):
            for i in range(len(prices)-window+1):
                subset = prices[i:i+window]
                if direction == 'up' and all(x < y for x, y in zip(subset, subset[1:])):
                    return True
                if direction == 'down' and all(x > y for x, y in zip(subset, subset[1:])):
                    return True
            return False

        # 1. Horizontal Breakout
        resistance = data['high'].rolling(20).max().iloc[-1]
        support = data['low'].rolling(20).min().iloc[-1]
        buffer = data['ATR'].iloc[-1] * 0.3
        
        if data['close'].iloc[-1] > resistance + buffer:
            patterns['bullish']['horizontal_breakout'] += PATTERN_WEIGHTS['horizontal_breakout']
        if data['close'].iloc[-1] < support - buffer:
            patterns['bearish']['horizontal_breakout'] += PATTERN_WEIGHTS['horizontal_breakout']

        # 2. Trendline Breakout
        highs = data['high'].values[-min_pattern_length:]
        lows = data['low'].values[-min_pattern_length:]
        
        for i in range(len(highs)-4):
            try:
                if highs[i+2] > highs[i] and highs[i+2] > highs[i+4]:
                    slope = (highs[i+4] - highs[i]) / 4
                    projected = highs[i] + slope * (len(highs)-i)
                    if data['close'].iloc[-1] > projected + buffer:
                        patterns['bullish']['trendline_breakout'] += PATTERN_WEIGHTS['trendline_breakout']
            except IndexError:
                continue

        # 3. Triangle Patterns
        max_high = data['high'].rolling(20).max()
        min_low = data['low'].rolling(20).min()
        volatility = (max_high - min_low).pct_change()
        
        if volatility.iloc[-1] < 0.5 * volatility.mean():
            if is_consecutive(data['high'].iloc[-5:], 3, 'up') and is_consecutive(data['low'].iloc[-5:], 3, 'up'):
                patterns['bullish']['ascending_triangle'] += PATTERN_WEIGHTS['ascending_triangle']
            
            if is_consecutive(data['high'].iloc[-5:], 3, 'down') and is_consecutive(data['low'].iloc[-5:], 3, 'down'):
                patterns['bearish']['descending_triangle'] += PATTERN_WEIGHTS['descending_triangle']

        # 4. Flag/Pennant Patterns
        if (data['volume'].iloc[-5:].mean() < 0.7 * data['volume'].iloc[-20:-5].mean() and
            data['high'].iloc[-5:].max() - data['low'].iloc[-5:].min() < 0.5 * data['ATR'].iloc[-1]):
            
            prev_trend = 'up' if data['close'].iloc[-6] < data['close'].iloc[-5] else 'down'
            if prev_trend == 'up' and data['close'].iloc[-1] > data['high'].iloc[-6]:
                patterns['bullish']['bull_flag'] += PATTERN_WEIGHTS['bull_flag']
            elif prev_trend == 'down' and data['close'].iloc[-1] < data['low'].iloc[-6]:
                patterns['bearish']['bear_flag'] += PATTERN_WEIGHTS['bear_flag']

        # 5. Order Blocks
        for i in range(3, len(data)-3):
            if (data['close'].iloc[i] > data['open'].iloc[i] and
                data['low'].iloc[i] < data['low'].iloc[i-1] and
                data['low'].iloc[i] < data['low'].iloc[i+1]):
                if data['close'].iloc[-1] > data['high'].iloc[i]:
                    patterns['bullish']['order_block'] += PATTERN_WEIGHTS['order_block']
            
            if (data['close'].iloc[i] < data['open'].iloc[i] and
                data['high'].iloc[i] > data['high'].iloc[i-1] and
                data['high'].iloc[i] > data['high'].iloc[i+1]):
                if data['close'].iloc[-1] < data['low'].iloc[i]:
                    patterns['bearish']['order_block'] += PATTERN_WEIGHTS['order_block']

        # 6. Candlestick Patterns
        if all(data['close'].iloc[-i] > data['open'].iloc[-i] and
               data['close'].iloc[-i] > data['close'].iloc[-(i+1)] for i in range(1,4)):
            patterns['bullish']['three_white_soldiers'] += PATTERN_WEIGHTS['three_white_soldiers']
            
        if all(data['close'].iloc[-i] < data['open'].iloc[-i] and
               data['close'].iloc[-i] < data['close'].iloc[-(i+1)] for i in range(1,4)):
            patterns['bearish']['three_black_crows'] += PATTERN_WEIGHTS['three_black_crows']

        # 7. Volume Spike Breakouts
        volume_avg = data['volume'].rolling(20).mean().iloc[-1]
        if data['volume'].iloc[-1] > 2.5 * volume_avg:
            if data['close'].iloc[-1] > resistance:
                patterns['bullish']['volume_spike'] += PATTERN_WEIGHTS['volume_spike']
            elif data['close'].iloc[-1] < support:
                patterns['bearish']['volume_spike'] += PATTERN_WEIGHTS['volume_spike']

        # 8. Cup & Handle (NEW)
        cup_period = 50
        handle_period = 10
        if len(data) > cup_period + handle_period:
            cup_high = data['high'].iloc[-cup_period-handle_period:-handle_period].max()
            cup_low = data['low'].iloc[-cup_period-handle_period:-handle_period].min()
            handle_high = data['high'].iloc[-handle_period:].max()
            
            # Check for U-shape and handle consolidation
            if (cup_high > handle_high and
                data['close'].iloc[-1] > handle_high + buffer and
                data['volume'].iloc[-handle_period:].mean() > data['volume'].iloc[-cup_period-handle_period:-handle_period].mean()):
                patterns['bullish']['cup_handle'] += PATTERN_WEIGHTS['cup_handle']

        # 9. Head & Shoulders (NEW)
        peaks = argrelextrema(data['high'].values, np.greater, order=3)[0]
        if len(peaks) >= 3:
            # Find the most recent three peaks
            recent_peaks = peaks[-3:]
            if (data['high'].iloc[recent_peaks[1]] > data['high'].iloc[recent_peaks[0]] and
                data['high'].iloc[recent_peaks[1]] > data['high'].iloc[recent_peaks[2]]):
                
                # Calculate neckline (lowest low between shoulders)
                left_trough = data['low'].iloc[recent_peaks[0]:recent_peaks[1]].min()
                right_trough = data['low'].iloc[recent_peaks[1]:recent_peaks[2]].min()
                neckline = min(left_trough, right_trough)
                
                if data['close'].iloc[-1] < neckline - data['ATR'].iloc[-1] * 0.5:
                    patterns['bearish']['head_shoulders'] += PATTERN_WEIGHTS['head_shoulders']

        # 10. Rectangle Pattern
        recent_high = data['high'].iloc[-10:].max()
        recent_low = data['low'].iloc[-10:].min()
        if (recent_high - recent_low) < data['ATR'].iloc[-1] * 0.5:
            if data['close'].iloc[-1] > recent_high:
                patterns['bullish']['rectangle'] += PATTERN_WEIGHTS['rectangle']
            elif data['close'].iloc[-1] < recent_low:
                patterns['bearish']['rectangle'] += PATTERN_WEIGHTS['rectangle']

        # 11. Wedge Patterns
        if is_consecutive(data['high'].iloc[-10:], 5, 'down') and is_consecutive(data['low'].iloc[-10:], 5, 'up'):
            patterns['bullish']['wedge'] += PATTERN_WEIGHTS['wedge']
        elif is_consecutive(data['high'].iloc[-10:], 5, 'up') and is_consecutive(data['low'].iloc[-10:], 5, 'down'):
            patterns['bearish']['wedge'] += PATTERN_WEIGHTS['wedge']

        # 12. Channel Breakouts
        upper_channel = data['high'].rolling(20).max()
        lower_channel = data['low'].rolling(20).min()
        if data['close'].iloc[-1] > upper_channel.iloc[-1] + buffer:
            patterns['bullish']['channel'] += PATTERN_WEIGHTS['channel']
        elif data['close'].iloc[-1] < lower_channel.iloc[-1] - buffer:
            patterns['bearish']['channel'] += PATTERN_WEIGHTS['channel']

        # Final scoring and validation
        bull_score = sum(patterns['bullish'].values())
        bear_score = sum(patterns['bearish'].values())
        
        confirmation_threshold = 5
        volume_confirmation = data['volume'].iloc[-1] > data['volume'].iloc[-2]
        trend_alignment = data['EMA_20'].iloc[-1] > data['EMA_50'].iloc[-1] if bull_score else data['EMA_20'].iloc[-1] < data['EMA_50'].iloc[-1]

        if bull_score >= confirmation_threshold and volume_confirmation and trend_alignment:
            confidence = min(5, int(bull_score))
            logging.info(f"Bullish Breakout ({confidence}/5) - {dict(patterns['bullish'])}")
            return 'buy', confidence
            
        if bear_score >= confirmation_threshold and volume_confirmation and trend_alignment:
            confidence = min(5, int(bear_score))
            logging.info(f"Bearish Breakout ({confidence}/5) - {dict(patterns['bearish'])}")
            return 'sell', confidence

        logging.info("No qualified breakout patterns detected")
        return None

    except Exception as e:
        logging.error(f"Breakout detection error: {str(e)}", exc_info=True)
        return None


# Determine Trade Signal
def should_trade(symbol, model, scaler, data, balance):
    try:
        data = add_indicators(data)  # Add indicators before preparing input
        if data is None:
            logging.warning(f"Missing indicators for {symbol}. Skipping.")
            return None, 0

        current_price = data['close'].iloc[-1]
        predicted_price = 0
        # if predicted_price is None:
        #     return None, 0

        # logging.info(f"LSTM Prediction for {symbol}: {predicted_price}, Current Price: {current_price}")

        position_size = ((POSITION_SIZE_PERCENT) * balance) / current_price
        position_size = position_size * LEVERAGE
        position_size = validate_position_size(symbol, position_size, current_price)

        atr = data['ATR'].iloc[-1]
        buy_threshold = 1
        sell_threshold = 1

        crossover_signal = detect_crossover(data)
        signal = detect_breakout_patterns(symbol=symbol)
        pattern_breakout = None
        if signal:
            direction, confidence = signal
            if confidence >= 4:
                pattern_breakout = direction

        # Define model path
        model_path = f"models/ppo_trading_{symbol.replace('/', '_')}.zip"

        # Create trading environment
        env = CryptoTradingEnv(symbol=symbol, exchange=exchange)

        # Check if model exists
        if os.path.exists(model_path):
            model_mtime = os.path.getmtime(model_path)  # Get last modified time
            age_hours = (time.time() - model_mtime) / 3600  # Convert age to hours

            # Retrain if model is older than 24 hours
            if age_hours < 6:
                logging.info(f"✅ Model found for {symbol} (last trained {age_hours:.2f} hours ago). Loading existing model...")
                model = PPO.load(model_path, env=env)
            else:
                logging.info(f"⚠️ Model for {symbol} is older than 6 hours. Retraining...")
                model = PPO("MlpPolicy", env, verbose=1)
                model.learn(total_timesteps=100000)
                model.save(model_path)
        else:
            logging.info(f"🚀 No model found for {symbol}. Training new model...")
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=100000)
            model.save(model_path)  # Save the

        obs = env.get_observation()  # Get real-time market data
        action, _ = model.predict(obs)
        trade_action = ["Hold", "buy", "sell"][action]

        logging.info(f"trade_action {trade_action}")
        logging.info(f"Trade conditions for {symbol} - Predicted: {predicted_price}, Current: {current_price}, MA_10: {data['MA_10'].iloc[-1]}, MA_30: {data['MA_30'].iloc[-1]}, RSI: {data['RSI'].iloc[-1]}, MACD: {data['MACD'].iloc[-1]}, Signal: {data['Signal'].iloc[-1]}, ATR Confirmation: {confirm_trade_signal_with_atr(symbol=symbol)}")
        logging.info(f"buy threshold {buy_threshold} - sell threshold {sell_threshold}")
        logging.info(f"cross over signal {crossover_signal}")

        # Buy Condition
        if (
            
            #  support_resistance_signal_new(symbol=symbol) == 'buy' 
            #  and (data['MACD'].iloc[-1] > 0))
             trade_action == 'buy'):
            return 'buy', position_size

        # Sell Condition
        elif (
            # ((
            #  support_resistance_signal_new(symbol=symbol) == 'sell')
            #  and (data['MACD'].iloc[-1] < 0))
         trade_action == 'sell'):
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
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=14)

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
                buffer = max(atr * dynamic_multiplier, current_price * 0.002)
                
                #max(atr * dynamic_multiplier, 0.0001)  # Adjust multiplier as needed (e.g., 0.5x ATR)
                unrealized_profit = float(position['unrealizedPnl'])
                notional_value = float(position['initialMargin'])
                dynamic_profit_target = max(0.1, min(0.3, atr / notional_value * (LEVERAGE / 10)))
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
                try:
                    model_path = f"models/ppo_trading_{symbol.replace('/', '_')}.zip"
                    env = CryptoTradingEnv(symbol=symbol, exchange=exchange)
                    model = PPO.load(model_path, env=env)
                    obs = env.get_observation() 
                    action, _ = model.predict(obs)
                    trade_action = ["Hold", "buy", "sell"][action]
                    logging.info(f"Model prediction for the  coin {symbol} is {trade_action}")
                
                except Exception as e:
                    logging.error(f"Monitor Trade Loading model exception {e} , setting trade_action to empty")
                    trade_action = ''
                

                # 1️⃣ Previous Close + Current Open with ATR Buffer → Close Position
                if unrealized_profit >= notional_value * 0.025:  # Ensure the position is in profit
                    if position_side == 'long' and trade_action != 'Hold':
                        # 1️⃣ Check for trend reversal (bearish red candle for long)
                        if trade_action == 'sell':  # Previous close > Current open
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

                    elif position_side == 'short' and trade_action != 'Hold' :
                        # 1️⃣ Check for trend reversal (bullish green candle for short)
                        if trade_action == 'buy':  # Previous close < Current open
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
                if float(position['unrealizedPnl']) <= -float(position['initialMargin']) * 0.35:
                    logging.info(f"Hard stop-loss hit for {symbol}. Forcing close at -30%.")
                    close_position()
                    continue
                elif float(position['unrealizedPnl']) <= -float(position['initialMargin']) * 0.25:
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


#new Trade Logic
last_trade_time = {}
cooldown_period = 240  # 4-minute cooldown between trades on the same symbol
max_retries = 2
retry_counter = {}

# Main Trading Function
def trade():
    logging.info("Starting bot...")

    while True:
        try:
            balance = fetch_wallet_balance()

            if balance > 0:
                TRADING_PAIRS = (fetch_top_movers())
                if not TRADING_PAIRS:
                    logging.info("No trading pairs available. Retrying...")
                    time.sleep(60)
                    continue
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
                time.sleep(120)  # Longer wait on low balance

            time.sleep(60)  # Regular wait before checking again

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

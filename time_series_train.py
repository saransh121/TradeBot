import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Initialize Binance API
exchange = ccxt.binance()

# Fetch Data
def fetch_data(symbol, timeframe, limit=1500):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Add Technical Indicators
def add_indicators(data):
    data['MA_10'] = data['close'].rolling(window=10).mean()
    data['MA_30'] = data['close'].rolling(window=30).mean()
    data['RSI'] = calculate_rsi(data['close'])
    data['ATR'] = calculate_atr(data)
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['EMA_7'] = data['close'].ewm(span=7, adjust=False).mean()
    data['EMA_25'] = data['close'].ewm(span=25, adjust=False).mean()
    data['EMA_99'] = data['close'].ewm(span=99, adjust=False).mean()
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data['close'])

    return data.dropna()

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

# Prepare Data
def prepare_data(data, n_steps=60):
    features = ['open', 'high', 'low', 'close', 'volume', 'MA_10', 'MA_30', 'RSI', 'ATR','EMA_12','EMA_26','MACD','EMA_7','EMA_25','EMA_99','Upper_Band','Lower_Band']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
        y.append(scaled_data[i, 3])  # Predicting 'close' price
    return np.array(X), np.array(y), scaler

# Train Model
def train_model_for_pair(symbol):
    print(f"Training LSTM model for {symbol}...")
    data = fetch_data(symbol, '3m', limit=1000)
    if data is None or data.empty:
        print(f"No data available for {symbol}. Skipping.")
        return

    # Add indicators and prepare data
    data = add_indicators(data)
    X, y, scaler = prepare_data(data)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Define LSTM Model
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])


    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Add Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, 
          verbose=1, callbacks=[early_stopping, lr_scheduler])

    # Save Model and Scaler
    model_path = f"models_lstm/lstm_{symbol.replace('/', '_')}.h5"
    scaler_path = f"models_lstm/scaler_{symbol.replace('/', '_')}.pkl"

    os.makedirs("models_lstm", exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path} and scaler saved to {scaler_path}.")

# Train for all pairs
if __name__ == "__main__":
#USUAL/USDT
#MOVE/USDT
#VELODROME/USDT
#TROY/USDT
#KOMA/USDT
#BIGTIME/USDT
#FLUX/USDT
#ETH/USDT
#["XRP/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT"]
    TRADING_PAIRS = ["XRP/USDT", 
                     "DOGE/USDT", "ADA/USDT","TRX/USDT"]
    for pair in TRADING_PAIRS:
        train_model_for_pair(pair)

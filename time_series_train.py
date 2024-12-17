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
def fetch_data(symbol, timeframe, limit=500):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Prepare Data
def prepare_data(data, n_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i])
        y.append(scaled_data[i, 3])  # Predicting 'close' price
    return np.array(X), np.array(y), scaler

# Train Model
def train_model_for_pair(symbol):
    print(f"Training LSTM model for {symbol}...")
    data = fetch_data(symbol, '15m', limit=500)
    if data is None or data.empty:
        print(f"No data available for {symbol}. Skipping.")
        return

    # Prepare LSTM Data
    X, y, scaler = prepare_data(data)
    X = X.reshape(X.shape[0], X.shape[1], 5)

    # Define LSTM Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Save Model and Scaler
    model_path = f"models_lstm/lstm_{symbol.replace('/', '_')}.h5"
    scaler_path = f"models_lstm/scaler_{symbol.replace('/', '_')}.pkl"

    os.makedirs("models_lstm", exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path} and scaler saved to {scaler_path}.")

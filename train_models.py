import ccxt
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize Binance API
exchange = ccxt.binance()

def fetch_data(symbol, timeframe, limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    fast_ema = series.ewm(span=fast).mean()
    slow_ema = series.ewm(span=slow).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def train_model_for_pair(symbol):
    print(f"Training model for {symbol}...")
    data = fetch_data(symbol, '15m', limit=500)
    data['MA_10'] = data['close'].rolling(window=10).mean()
    data['MA_30'] = data['close'].rolling(window=30).mean()
    data['RSI'] = calculate_rsi(data['close'])
    data['MACD'], data['Signal_Line'] = calculate_macd(data['close'])
    data = data.dropna()

    X = data[['MA_10', 'MA_30', 'RSI', 'MACD', 'Signal_Line']]
    y = (data['close'].shift(-1) > data['close']).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    model_path = f"models/{symbol.replace('/', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    TRADING_PAIRS = [
        "ETH/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "DOT/USDT", "COW/USDT" , "GOAT/USDT" , "TRX/USDT"
    ]
    for pair in TRADING_PAIRS:
        train_model_for_pair(pair)

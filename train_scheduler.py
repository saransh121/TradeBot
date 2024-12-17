import schedule
import time
import os
from time_series_train import train_model_for_pair

# Trading Pairs for Training
TRADING_PAIRS = ["XRP/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT","COW/USDT"]

# Function to trigger training
def run_training():
    print("Starting scheduled training...")
    for pair in TRADING_PAIRS:
        try:
            print(f"Training model for {pair}...")
            train_model_for_pair(pair)
        except Exception as e:
            print(f"Error training {pair}: {e}")

    print("Training completed for all pairs.")

# Schedule training every 1 hour
schedule.every(1).hours.do(run_training)

if __name__ == "__main__":
    print("Scheduler started. Training will run every 1 hour.")
    while True:
        schedule.run_pending()
        time.sleep(10)  # Check every 10 seconds

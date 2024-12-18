import schedule
import time
import os
import logging
from time_series_train import train_model_for_pair

# Configure logging
logging.basicConfig(level=logging.INFO, filename='training_scheduler.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Trading Pairs for Training
TRADING_PAIRS = ["XRP/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT", "COW/USDT"]

# Function to trigger training
def run_training():
    logging.info("Starting scheduled training...")
    for pair in TRADING_PAIRS:
        try:
            logging.info(f"Training model for {pair}...")
            train_model_for_pair(pair)
        except Exception as e:
            logging.error(f"Error training {pair}: {e}")
    logging.info("Training completed for all pairs.")

# Schedule training every 30 minutes
schedule.every(30).minutes.do(run_training)

if __name__ == "__main__":
    logging.info("Scheduler started. Training will run every 30 minutes.")
    run_training()  # Run immediately upon start
    while True:
        schedule.run_pending()
        time.sleep(10)  # Check every 10 seconds

import schedule
import time
import logging
from dotenv import load_dotenv
import os
from bot import monitor_positions, exchange, PROFIT_TARGET_PERCENT  # Import required components

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, filename='monitor_bot.log', format='%(asctime)s - %(levelname)s - %(message)s')

def run_monitoring():
    """
    Function to run the monitor_positions method every 30 seconds.
    """
    try:
        logging.info("Running monitoring...")
        monitor_positions()
    except Exception as e:
        logging.error(f"Error during monitoring: {e}")

# Schedule monitoring every 30 seconds
schedule.every(10).seconds.do(run_monitoring)

if __name__ == "__main__":
    logging.info("Monitor bot started. Running every 30 seconds.")
    while True:
        schedule.run_pending()
        time.sleep(1)

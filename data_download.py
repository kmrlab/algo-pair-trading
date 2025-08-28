"""
██╗  ██╗███╗   ███╗██████╗ ██╗      █████╗ ██████╗ 
██║ ██╔╝████╗ ████║██╔══██╗██║     ██╔══██╗██╔══██╗
█████╔╝ ██╔████╔██║██████╔╝██║     ███████║██████╔╝
██╔═██╗ ██║╚██╔╝██║██╔══██╗██║     ██╔══██║██╔══██╗
██║  ██╗██║ ╚═╝ ██║██║  ██║███████╗██║  ██║██████╔╝
╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ 

Crafted with ❤️ by Kristofer Meio-Renn

Found this useful? Star the repo to show your support! Thank you!
GitHub: https://github.com/kmrlab
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import os
from typing import List, Dict
import json

# ==================== CONFIGURATION ====================
# API Credentials
# Set your API credentials as environment variables or replace with your keys
API_KEY = os.getenv('BYBIT_API_KEY', 'your_bybit_api_key_here')
API_SECRET = os.getenv('BYBIT_API_SECRET', 'your_bybit_api_secret_here')

# Trading pairs
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "BCHUSDT",
    "AVAXUSDT",
    "HBARUSDT",
    "DOTUSDT",
    "AAVEUSDT",
    "ONDOUSDT",
    "APTUSDT"
]

# Timeframes
INTERVALS = {
    "5m": "5"
}

# Date range
START_DATE = "2024-01-01"
END_DATE = "2025-08-10"

# Output directory
OUTPUT_DIR = "bybit_data"

# Rate limit settings
RATE_LIMIT_DELAY = 0.1  # Delay between requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for failed requests

# ==================== MAIN CODE ====================

class BybitDataDownloader:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize Bybit client"""
        self.session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
    
    def timestamp_to_ms(self, date_str: str) -> int:
        """Convert date string to milliseconds timestamp"""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp() * 1000)
    
    def ms_to_datetime(self, ms: int) -> datetime:
        """Convert milliseconds timestamp to datetime"""
        return datetime.fromtimestamp(ms / 1000)
    
    def get_klines_batch(self, symbol: str, interval: str, start_time: int, end_time: int) -> List[Dict]:
        """Fetch a batch of klines data"""
        try:
            response = self.session.get_kline(
                category="linear",  # Perpetual futures
                symbol=symbol,
                interval=interval,
                start=start_time,
                end=end_time,
                limit=1000  # Maximum allowed by Bybit
            )
            
            if response["retCode"] == 0:
                return response["result"]["list"]
            else:
                print(f"Error fetching data: {response['retMsg']}")
                return []
                
        except Exception as e:
            print(f"Exception occurred: {e}")
            return []
    
    def download_symbol_data(self, symbol: str, interval: str, interval_name: str) -> pd.DataFrame:
        """Download all data for a specific symbol and interval"""
        print(f"Downloading {symbol} - {interval_name} timeframe...")
        
        all_data = []
        start_ms = self.timestamp_to_ms(START_DATE)
        end_ms = self.timestamp_to_ms(END_DATE)
        
        # Calculate batch size based on interval
        if interval == "1":
            batch_hours = 16  # ~1000 1-minute candles
        else:  # 5 minutes
            batch_hours = 80  # ~1000 5-minute candles
        
        batch_ms = batch_hours * 60 * 60 * 1000
        
        current_start = start_ms
        
        while current_start < end_ms:
            current_end = min(current_start + batch_ms, end_ms)
            
            # Retry logic
            for retry in range(MAX_RETRIES):
                klines = self.get_klines_batch(symbol, interval, current_start, current_end)
                
                if klines:
                    # Bybit returns data in reverse order (newest first)
                    klines.reverse()
                    all_data.extend(klines)
                    print(f"  Downloaded {len(klines)} candles from {self.ms_to_datetime(current_start).strftime('%Y-%m-%d %H:%M')}")
                    break
                else:
                    if retry < MAX_RETRIES - 1:
                        print(f"  Retry {retry + 1}/{MAX_RETRIES}...")
                        time.sleep(RATE_LIMIT_DELAY * 2)
            
            current_start = current_end
            time.sleep(RATE_LIMIT_DELAY)
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert price and volume columns to numeric
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])
            
            # Sort by timestamp
            df = df.sort_values('datetime')
            
            # Remove duplicates if any
            df = df.drop_duplicates(subset=['timestamp'])
            
            return df
        else:
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, symbol: str, interval_name: str):
        """Save DataFrame to CSV file"""
        if df.empty:
            print(f"  No data to save for {symbol} - {interval_name}")
            return
        
        filename = f"{OUTPUT_DIR}/{symbol}_{interval_name}_{START_DATE}_to_{END_DATE}.csv"
        df.to_csv(filename, index=False)
        print(f"  Saved {len(df)} candles to {filename}")
    
    def download_all_data(self):
        """Download data for all symbols and intervals"""
        total_combinations = len(SYMBOLS) * len(INTERVALS)
        current = 0
        
        print(f"Starting download of {total_combinations} symbol-interval combinations")
        print(f"Date range: {START_DATE} to {END_DATE}")
        print("=" * 50)
        
        for symbol in SYMBOLS:
            for interval_name, interval_value in INTERVALS.items():
                current += 1
                print(f"\n[{current}/{total_combinations}] Processing {symbol} - {interval_name}")
                
                # Download data
                df = self.download_symbol_data(symbol, interval_value, interval_name)
                
                # Save to file
                self.save_data(df, symbol, interval_name)
                
                # Small delay between different requests
                time.sleep(RATE_LIMIT_DELAY)
        
        print("\n" + "=" * 50)
        print("Download completed!")
        print(f"Data saved in '{OUTPUT_DIR}' directory")
    
    def get_data_summary(self):
        """Print summary of downloaded data"""
        print("\n" + "=" * 50)
        print("DATA SUMMARY")
        print("=" * 50)
        
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
        
        for file in sorted(files):
            filepath = os.path.join(OUTPUT_DIR, file)
            df = pd.read_csv(filepath)
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                start_date = df['datetime'].min()
                end_date = df['datetime'].max()
                
                print(f"\n{file}:")
                print(f"  Records: {len(df)}")
                print(f"  Date range: {start_date} to {end_date}")
                print(f"  File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")


def main():
    """Main function"""
    print("Bybit Historical Data Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = BybitDataDownloader(API_KEY, API_SECRET)
    
    # Download all data
    downloader.download_all_data()
    
    # Print summary
    downloader.get_data_summary()


if __name__ == "__main__":
    main()

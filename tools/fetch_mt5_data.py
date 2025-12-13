import MetaTrader5 as mt5
import pandas as pd
import os
import argparse
from datetime import datetime

# Import Interface
import sys
sys.path.append(os.getcwd())
from src.data.mt5_interface import MT5Interface

def fetch_data(interval_str):
    print(f"Fetching {interval_str} data from MT5...")
    
    # Map string to MT5 constant
    tf_map = {
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30
    }
    
    if interval_str not in tf_map:
        print(f"Interval {interval_str} not supported in specific fetcher.")
        return
        
    tf_const = tf_map[interval_str]
    
    # Init
    mt = MT5Interface()
    if not mt.connect():
        return
        
    # Fetch 5000 candles (enough for deep learning)
    df = mt.get_historical_data("XAUUSD", timeframe=tf_const, num_candles=5000)
    
    if df is not None and not df.empty:
        os.makedirs("data", exist_ok=True)
        out_path = f"data/XAUUSD_{interval_str}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")
    else:
        print("Failed to fetch data.")
        
    mt.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, required=True, help="15m or 30m")
    args = parser.parse_args()
    
    fetch_data(args.interval)

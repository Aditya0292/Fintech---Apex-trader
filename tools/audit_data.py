
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def check_file(filepath, timeframe_str):
    print(f"\n--- Auditing {filepath} ({timeframe_str}) ---")
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return
        
    df = pd.read_csv(filepath)
    print(f"Rows: {len(df)}")
    
    # Rename time to datetime if exists
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'datetime'})
        
    # Check Columns
    needed = ['datetime', 'open', 'high', 'low', 'close']
    if not all(c in df.columns for c in needed):
        print(f"❌ Missing columns! Found: {df.columns}")
        return
        
    # Check Times
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime')
    
    # 1. Duplicates
    dupes = df.duplicated('datetime').sum()
    if dupes > 0:
        print(f"❌ Found {dupes} duplicate timestamps.")
    else:
        print("✅ No duplicates found.")
        
    # 2. Logic (H >= L)
    bad_rows = df[df['high'] < df['low']]
    if len(bad_rows) > 0:
        print(f"❌ Found {len(bad_rows)} rows where High < Low.")
    else:
        print("✅ High/Low logic valid.")
        
    # 3. Zeros/NaNs
    zeros = (df[['open', 'high', 'low', 'close']] == 0).sum().sum()
    nans = df[['open', 'high', 'low', 'close']].isna().sum().sum()
    if zeros > 0 or nans > 0:
        print(f"❌ Found {zeros} Zeros and {nans} NaNs in price data.")
    else:
        print("✅ No Zero/NaN prices.")
        
    # 4. Gaps
    # Calculate expected delta
    if "h" in timeframe_str.lower():
        delta = timedelta(hours=int(timeframe_str.replace("h","").replace("H","")))
    elif "m" in timeframe_str.lower():
        delta = timedelta(minutes=int(timeframe_str.replace("m","")))
    elif "d" in timeframe_str.lower(): # Daily usually excludes weekends, harder to check strictly
        delta = timedelta(days=1)
    else:
        delta = None
        
    if delta and timeframe_str != "Daily":
        # Check diff
        time_diffs = df['datetime'].diff()
        # Allow some gaps (weekends), but flag major ones
        gap_threshold = delta * 5 # missed 5 candles
        gaps = time_diffs[time_diffs > gap_threshold]
        if len(gaps) > 0:
            print(f"⚠️ Found {len(gaps)} gaps larger than 5x interval.")
            print(f"   Largest Gap: {time_diffs.max()}")
        else:
            print("✅ Data continuity looks good (No massive gaps).")
    else:
        print("ℹ️ Skipping Gap check for Daily logic (Weekends).")

    print(f"✅ Data Integrity Score: {10 if (dupes+len(bad_rows)+zeros+nans)==0 else 5}/10")

def run_audit():
    check_file("data/XAUUSD_4h.csv", "4h")
    check_file("data/XAUUSD_1h.csv", "1h")
    # check_file("data/XAUUSD_1d.csv", "Daily")

if __name__ == "__main__":
    run_audit()

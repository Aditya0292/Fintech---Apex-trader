"""
APEX TRADE AI - DXY Data Fetcher
================================
Fetches Dollar Index (DXY) history from yfinance.
Symbol: DX-Y.NYB
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

def fetch_dxy(start_date="2000-01-01"):
    print("Fetching DXY data from yfinance...")
    try:
        # DX-Y.NYB is the ticker for US Dollar Index on Yahoo Finance
        dxy = yf.download("DX-Y.NYB", start=start_date, progress=False)
        
        if len(dxy) == 0:
            print("Error: No data fetched for DXY. Trying 'DX=F' (Futures)...")
            dxy = yf.download("DX=F", start=start_date, progress=False)
            
        if len(dxy) == 0:
            raise ValueError("Could not fetch DXY data.")
            
        # Clean up MultiIndex columns if present
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = dxy.columns.get_level_values(0)
            
        dxy.columns = [str(c).lower() for c in dxy.columns]
        dxy = dxy.reset_index()
        dxy = dxy.rename(columns={"date": "time", "Date": "time"})
        
        # Ensure time format matches XAUUSD (YYYY-MM-DD)
        dxy['time'] = pd.to_datetime(dxy['time']).dt.strftime('%Y-%m-%d')
        
        # Save
        dxy.to_csv("data/dxy_history.csv", index=False)
        print(f"DXY data saved: {len(dxy)} rows.")
        return dxy
        
    except Exception as e:
        print(f"Failed to fetch DXY: {e}")
        print("Switching to Synthetic DXY generation (Inverse of Gold) to allow pipeline to run...")
        # Fallback to reading gold from data/
        return generate_synthetic_dxy(gold_csv="data/XAUUSD_history.csv")

def generate_synthetic_dxy(gold_csv="data/XAUUSD_history.csv"):
    """
    Generates synthetic DXY data inversely correlated to Gold.
    Formula: DXY = Constant / Gold * Noise
    This is just to enable the pipeline to build structure without live data.
    """
    try:
        df = pd.read_csv(gold_csv)
        df['time'] = pd.to_datetime(df['time'])
        
        # Simple inverse + noise
        # Gold ~2000 -> DXY ~100. Constant ~200000
        # Add random walk
        np.random.seed(42)
        noise = np.random.normal(0, 0.5, len(df))
        
        # Create DXY structure
        dxy = pd.DataFrame()
        dxy['time'] = df['time']
        
        # Synthetic Close
        # Base inverse relationship with some drift
        dxy['close'] = (200000 / df['close']) + np.cumsum(noise)
        dxy['open'] = dxy['close'] # Simplify
        dxy['high'] = dxy['close'] * 1.002
        dxy['low'] = dxy['close'] * 0.998
        dxy['volume'] = 10000
        
        dxy['time'] = dxy['time'].dt.strftime('%Y-%m-%d')
        
        dxy.to_csv("data/dxy_history.csv", index=False)
        print("Synthetic DXY data generated.")
        return dxy
        
    except Exception as e:
        print(f"Failed to generate synthetic DXY: {e}")
        return None

def align_dxy_with_gold(gold_csv="data/XAUUSD_history.csv"):
    """
    Align DXY data to Gold data timestamps.
    Forward fill DXY data (if Gold trades on a holiday where DXY is closed, use last DXY).
    """
    if not Path("data/dxy_history.csv").exists():
        dxy = fetch_dxy()
    else:
        dxy = pd.read_csv("data/dxy_history.csv")
    
    if dxy is None:
        return None
        
    gold = pd.read_csv(gold_csv)
    
    # Merge
    # We want Gold dates as the master
    from src.utils.time_utils import normalize_ts
    gold['time'] = normalize_ts(gold['time'])
    dxy['time'] = normalize_ts(dxy['time'])
    
    # Rename DXY columns to avoid collision
    dxy = dxy[['time', 'open', 'high', 'low', 'close', 'volume']]
    dxy.columns = ['time', 'dxy_open', 'dxy_high', 'dxy_low', 'dxy_close', 'dxy_volume']
    
    # Merge (Left join on Gold)
    # Using merge (exact match) since we normalized both.
    # Note: align_datasets uses merge_asof, but here we might want exact alignment if data is daily?
    # Actually, align_dxy_with_gold is likely used for historical daily data. Exact merge is safer if timestamps align.
    # If timestamps are mismatched (different hours), merge_asof is better.
    # Let's stick to merge for now as originally intended, but normalized.
    merged = pd.merge(gold, dxy, on='time', how='left')
    
    # Forward fill DXY holes
    cols_to_fill = ['dxy_open', 'dxy_high', 'dxy_low', 'dxy_close', 'dxy_volume']
    merged[cols_to_fill] = merged[cols_to_fill].fillna(method='ffill')
    
    # Fill remaining (start of data) with backfill or first valid
    merged[cols_to_fill] = merged[cols_to_fill].fillna(method='bfill')
    
    return merged

if __name__ == "__main__":
    fetch_dxy()
    # merged = align_dxy_with_gold()
    # print(merged[['time', 'close', 'dxy_close']].tail())

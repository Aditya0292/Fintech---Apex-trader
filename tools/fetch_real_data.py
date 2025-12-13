
import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def fetch_and_process(symbol, yf_ticker, start_date="2004-01-01"):
    print(f"\n[{symbol}] Fetching from Yahoo Finance ({yf_ticker})...")
    time.sleep(2) # Initial nice delay
    
    if not os.path.exists("data"):
        os.makedirs("data")

    # 1. DAILY (From 2004)
    print("  > Downloading Daily data (Max History)...")
    try:
        # Create session to potentially bypass simple blocks
        import requests
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'})
        
        df_d = yf.download(yf_ticker, start=start_date, interval="1d", progress=False, threads=False) # session=session removed as recent yf might not support it directly in download efficiently or arguments changed.
        # Actually yf.Ticker(x, session=session).history(...) is better for control
        
        # Let's use Ticker object approach which is more robust
        ticker_obj = yf.Ticker(yf_ticker)
        # history() doesn't support 'start' with 'interval' well in all versions compared to download, but let's try download with threads=False first.
        
        time.sleep(2) 
        if not df_d.empty:
            # Flatten columns if MultiIndex
            if isinstance(df_d.columns, pd.MultiIndex):
                df_d.columns = [c[0].lower() for c in df_d.columns]
            else:
                df_d.columns = df_d.columns.str.lower()
                
            # Rename for consistency
            df_d.index.name = 'time'
            df_d = df_d.reset_index()
            # Ensure columns exist
            cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            df_d = df_d[[c for c in cols if c in df_d.columns]]
            
            # Save
            path = f"data/{symbol}_history.csv"
            df_d.to_csv(path, index=False)
            print(f"    Saved {path} ({len(df_d)} rows)")
            
            # Save explicit 1d
            path_1d = f"data/{symbol}_1d.csv"
            df_d.to_csv(path_1d, index=False)
            
    except Exception as e:
        print(f"    Error fetching Daily: {e}")

    # 2. HOURLY (Max 730 days)
    print("  > Downloading 1h data (Max 730d)...")
    try:
        df_1h = yf.download(yf_ticker, period="730d", interval="1h", progress=False)
        time.sleep(2)
        if not df_1h.empty:
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = [c[0].lower() for c in df_1h.columns]
            else:
                df_1h.columns = df_1h.columns.str.lower()
            
            df_1h.index = df_1h.index.tz_localize(None) # Remove timezone for simple CSV compatibility
            df_1h.index.name = 'time'
            df_1h = df_1h.reset_index()
            
            # Save 1h
            path = f"data/{symbol}_1h.csv"
            df_1h.to_csv(path, index=False)
            print(f"    Saved {path} ({len(df_1h)} rows)")
            
            # RESAMPLE TO 4H
            print("  > Resampling to 4h...")
            df_1h_idx = df_1h.set_index('time')
            df_4h = df_1h_idx.resample('4h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()
            
            path_4h = f"data/{symbol}_4h.csv"
            df_4h.to_csv(path_4h, index=False)
            print(f"    Saved {path_4h} ({len(df_4h)} rows)")
            
    except Exception as e:
        print(f"    Error fetching 1h: {e}")

    # 3. 15 MIN (Max 60 days)
    print("  > Downloading 15m data (Max 60d)...")
    try:
        df_15m = yf.download(yf_ticker, period="60d", interval="15m", progress=False)
        time.sleep(2)
        if not df_15m.empty:
            if isinstance(df_15m.columns, pd.MultiIndex):
                df_15m.columns = [c[0].lower() for c in df_15m.columns]
            else:
                df_15m.columns = df_15m.columns.str.lower()
                
            df_15m.index = df_15m.index.tz_localize(None)
            df_15m.index.name = 'time'
            df_15m = df_15m.reset_index()
            
            # Save 15m
            path_15m = f"data/{symbol}_15m.csv"
            df_15m.to_csv(path_15m, index=False)
            print(f"    Saved {path_15m} ({len(df_15m)} rows)")
            
            # RESAMPLE TO 30M
            print("  > Resampling to 30m...")
            df_15m_idx = df_15m.set_index('time')
            df_30m = df_15m_idx.resample('30min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna().reset_index()
            
            path_30m = f"data/{symbol}_30m.csv"
            df_30m.to_csv(path_30m, index=False)
            print(f"    Saved {path_30m} ({len(df_30m)} rows)")

    except Exception as e:
        print(f"    Error fetching 15m: {e}")

def main():
    # Map symbols to YFinance Tickers
    # EURUSD=X, GBPUSD=X, JPY=X (USDJPY is JPY=X in YF usually inverted? wait.)
    # YF Ticker for USD/JPY is "JPY=X" which is actually rate of JPY per USD? No. 
    # JPY=X is USD/JPY. 
    # EURUSD=X is EUR/USD.
    
    assets = [
        ("EURUSD", "EURUSD=X"),
        ("GBPUSD", "GBPUSD=X"),
        ("USDJPY", "JPY=X") # Verify if this is correct direction. Usually yes.
    ]
    
    for symbol, ticker in assets:
        fetch_and_process(symbol, ticker)

if __name__ == "__main__":
    main()

"""
APEX TRADE AI - XAUUSD Data Fetcher
===================================
Fetches Gold (XAUUSD) history from yfinance.
Symbol: GC=F (Gold Futures) as proxy for XAUUSD spot.
"""

import yfinance as yf
import pandas as pd
import os
import argparse

def fetch_gold(interval="1d"):
    """
    Fetches XAUUSD data.
    Args:
        interval: "1d", "4h", "1h", "15m", "5m"
    """
    print(f"Fetching XAUUSD (Gold Futures) data ({interval}) from yfinance...")
    ticker = "GC=F"
    
    # Intraday limits: 1m-15m = 60d, 1h = 730d
    period = "max"
    start_date = "2000-01-01"
    
    fetch_interval = interval
    resample_rule = None
    
    if interval == "4h":
        fetch_interval = "1h"
        resample_rule = "4h"
        period = "730d"
        start_date = None
    elif interval == "1h":
        period = "730d"
        start_date = None
    elif interval in ["5m", "15m", "30m"]:
        period = "60d"
        start_date = None
    
    try:
        # Fetch Logic
        if start_date:
            data = yf.download(ticker, start=start_date, interval=fetch_interval, progress=False, auto_adjust=True)
        else:
            data = yf.download(ticker, period=period, interval=fetch_interval, progress=False, auto_adjust=True)
        
        if len(data) == 0:
            print("Trying XAUUSD=X...")
            ticker = "XAUUSD=X"
            if start_date:
                data = yf.download(ticker, start=start_date, interval=fetch_interval, progress=False, auto_adjust=True)
            else:
                data = yf.download(ticker, period=period, interval=fetch_interval, progress=False, auto_adjust=True)

        if len(data) == 0:
            raise ValueError(f"No data found for {ticker} ({fetch_interval})")
            
        # Clean MultiIndex (YF behavior change in 0.2.x)
        if isinstance(data.columns, pd.MultiIndex):
            # Dropping the 'Ticker' level
            data.columns = data.columns.get_level_values(0)
            
        # Normalize columns
        data.columns = [str(c).lower() for c in data.columns]
        
        # Ensure standard OHLCV columns exist
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        for c in req_cols:
            if c not in data.columns:
                data[c] = 0.0
        
        # Resampling Logic
        if resample_rule:
             print(f"  Resampling {fetch_interval} to {resample_rule}...")
             agg_dict = {
                 'open': 'first',
                 'high': 'max',
                 'low': 'min',
                 'close': 'last',
                 'volume': 'sum'
             }
             valid_agg = {k: v for k, v in agg_dict.items() if k in data.columns}
             
             # Convert index to datetime if not already (YF usually returns index)
             data = data.resample(resample_rule).agg(valid_agg).dropna()
        
        data = data.reset_index()
        
        # Rename Date/Datetime -> time
        cols = {c: "time" for c in data.columns if "date" in c.lower()}
        data = data.rename(columns=cols)
        
        # Save to specific file
        suffix = "history" if interval == "1d" else interval
        os.makedirs("data", exist_ok=True)
        out_path = f"data/XAUUSD_{suffix}.csv"
        data.to_csv(out_path, index=False)
        print(f"XAUUSD data saved to {out_path}: {len(data)} rows.")
        return data
        
    except Exception as e:
        print(f"Error fetching XAUUSD: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, default="1d", help="1d, 4h, 1h, 15m")
    args = parser.parse_args()
    
    fetch_gold(args.interval)

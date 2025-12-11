
import yfinance as yf
import pandas as pd
import os

def fetch_1h_data():
    print("Fetching 1-Hour Gold Data (GC=F)...")
    
    # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # 1h max period is 730d
    
    try:
        # Ticker: GC=F (Gold Futures) or XAUUSD=X (Spot)
        # Using GC=F often has volume, XAUUSD=X often misses volume on Yahoo
        ticker = "GC=F" 
        df = yf.download(ticker, period="730d", interval="1h", progress=False)
        
        # Try cascading periods
        periods = ["730d", "60d", "1mo"]
        df = pd.DataFrame()
        
        for p in periods:
            print(f"Attempting period: {p}")
            try:
                df = yf.download(ticker, period=p, interval="1h", progress=False)
                if not df.empty:
                    print(f"Success with {p}!")
                    break
            except Exception as e:
                print(f"Failed {p}: {e}")
                
        if df.empty:
            print("Trying alternate ticker XAUUSD=X...")
            ticker = "XAUUSD=X"
            for p in periods:
                print(f"Attempting period: {p}")
                try:
                    df = yf.download(ticker, period=p, interval="1h", progress=False)
                    if not df.empty:
                        print(f"Success with {p}!")
                        break
                except Exception as e:
                    print(f"Failed {p}: {e}")
            
        if df.empty:
            print("Error: Could not fetch data from any source/period.")
            # Create DUMMY data if persistent failure, so we can at least write the code?
            # No, better to fail and tell user.
            return

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Rename standard columns
        # Yahoo cols: Open, High, Low, Close, Adj Close, Volume
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure 'time' column (Yahoo might explicitly name index 'Datetime')
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'time'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'time'})
            
        # Select required columns
        req_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        # Handle missing volume
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # Filter only existing columns
        existing_cols = [c for c in req_cols if c in df.columns]
        df = df[existing_cols]
        
        # Save
        if not os.path.exists("data"): os.makedirs("data")
        path = "data/XAUUSD_1h.csv"
        df.to_csv(path, index=False)
        print(f"Success! Saved {len(df)} 1H candles to {path}")
        print(df.tail())
        
    except Exception as e:
        print(f"Fetch Error: {e}")

if __name__ == "__main__":
    fetch_1h_data()

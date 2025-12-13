
import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_fx_data(symbol, days=2000, start_price=1.0, volatility=0.005):
    """
    Generates synthetic FX OHLCV data for base 5m and resamples for other TFs.
    """
    print(f"Generating synthetic data for {symbol}...")
    # Generate 5m base data
    freq = '5min'
    periods = days * 24 * 12 # 5m periods
    dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
    
    np.random.seed(hash(symbol) % 2**32)
    change = np.random.normal(0, volatility/10, len(dates)) # Lower vol for 5m
    prices = start_price * np.exp(np.cumsum(change))
    
    opens = prices
    closes = np.roll(prices, -1)
    closes[-1] = opens[-1]
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0005, len(dates))))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0005, len(dates))))
    volumes = np.random.randint(10, 500, len(dates))
    
    df_5m = pd.DataFrame({
        'time': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Save History (Base/Daily-ish use)
    df_5m.to_csv(f"data/{symbol}_history.csv", index=False)
    print(f"Saved data/{symbol}_history.csv")
    
    # Resample and Save Timeframes
    timeframes = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D"
    }
    
    validation_df = df_5m.set_index('time')
    
    for tf_name, rule in timeframes.items():
        resampled = validation_df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Reset index to get time column back
        resampled = resampled.reset_index()
        
        outfile = f"data/{symbol}_{tf_name}.csv"
        resampled.to_csv(outfile, index=False)
        print(f"  Saved {outfile} ({len(resampled)} rows)")

def main():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # XAUUSD is usually existing, but let's assume we might need to fill gaps if user asks "all assets"
    # But user specifically asked for USDJPY, EURUSD, GBPUSD.
    generate_fx_data("EURUSD", 1000, 1.08, 0.005)
    generate_fx_data("GBPUSD", 1000, 1.25, 0.006)
    generate_fx_data("USDJPY", 1000, 145.0, 0.007)
    
if __name__ == "__main__":
    main()

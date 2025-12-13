
import pandas as pd
import sys
import os

# Add root
sys.path.append(os.getcwd())

from src.analysis.smc_analyzer import SMCAnalyzer

def debug_smc():
    path = "data/XAUUSD_4h.csv"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    print(f"Loading {path}...")
    df = pd.read_csv(path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
    
    print(f"Data shape: {df.shape}")
    print(f"Time range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    print(f"Last Close: {df.iloc[-1]['close']}")

    smc = SMCAnalyzer(df, timeframe="4h")
    print(f"SMC Analyzer Initialized. Window Size applied internally: {len(smc.df)}")
    
    print("Running find_structure_and_blocks()...")
    obs = smc.find_structure_and_blocks()
    print(f"Total OBs found: {len(obs)}")
    
    bull_obs = [o for o in obs if o['type'] == 'bullish']
    bear_obs = [o for o in obs if o['type'] == 'bearish']
    print(f"  Bullish: {len(bull_obs)}")
    print(f"  Bearish: {len(bear_obs)}")
    
    if bull_obs:
        print("Last 3 Bullish OBs:")
        for o in bull_obs[-3:]:
            print(f"  {o['time']} | Top: {o['top']}, Bot: {o['bottom']}")
            
    # Check filtering
    current_price = float(df.iloc[-1]['close'])
    print(f"\nFiltering for Current Price: {current_price}")
    
    valid_bull = [o for o in bull_obs if o['bottom'] < current_price]
    print(f"Valid Bull OBs (< Price): {len(valid_bull)}")
    
    valid_bear = [o for o in bear_obs if o['top'] > current_price]
    print(f"Valid Bear OBs (> Price): {len(valid_bear)}")

if __name__ == "__main__":
    debug_smc()

import subprocess
import os
import sys
import argparse

def run_command(command_list):
    print(f"Running: {' '.join(command_list)}")
    res = subprocess.run(command_list, shell=False)
    if res.returncode != 0:
        print(f"Command failed: {command_list}")
        return False
    return True


def train_multiframe(timeframes, n_splits, symbols):
    for symbol in symbols:
        print(f"\n========================================")
        print(f"🚀 Starting Multi-Timeframe Training for {symbol}")
        print(f"========================================")
        print(f"Timeframes: {timeframes}")
        print(f"N_Splits: {n_splits}")
        
        # 1. Iterate Timeframes
        for tf in timeframes:
            print(f"\n[{tf.upper()}] Processing...")
            
            # Determine Data File
            filename = f"data/{symbol}_{tf}.csv"
            suffix = f"_{tf}"
            
            # Special Handling for 1d (history file)
            if tf == "1d":
                if not os.path.exists(filename):
                    hist_file = f"data/{symbol}_history.csv"
                    if os.path.exists(hist_file):
                        filename = hist_file
                        suffix = "_1d" 
                    else:
                        print(f"  Warning: No data found for 1d ({filename} or {hist_file}). Assuming synthetic generation needed or skipping.")
                        # Check availability
                        if not os.path.exists(filename) and not os.path.exists(hist_file):
                             print("  Skipping.")
                             continue
            
            if not os.path.exists(filename):
                 print(f"  Warning: Data file {filename} not found. Skipping.")
                 continue
                 
            # 1. Feature Engineering
            print(f"  > Generating Features for {tf}...")
            
            # Use unique suffix for features too so each symbol has its own numpy files
            unique_suffix = f"_{symbol}{suffix}"
            
            # feature_pipeline signature: --data DATA --suffix SUFFIX --symbol SYMBOL
            cmd_feat = [sys.executable, "src/features/feature_pipeline.py", "--data", filename, "--suffix", unique_suffix, "--symbol", symbol]
            if not run_command(cmd_feat): continue
            
            # 2. Train Model
            print(f"  > Training Ensemble for {tf}...")
            
            cmd_train = [sys.executable, "src/models/train_ensemble.py", "--suffix", unique_suffix, "--n_splits", str(n_splits)]
            if not run_command(cmd_train): continue
            
            print(f"✅ {symbol} {tf} Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframes", nargs='+', default=["1d", "4h", "1h", "30m", "15m", "5m"], help="List of timeframes")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for walk-forward validation")
    parser.add_argument("--assets", nargs='+', default=["XAUUSD"], help="List of assets or 'all'")
    
    args = parser.parse_args()
    
    symbols = args.assets
    if "all" in symbols:
        # Load from assets.yaml
        try:
            import yaml
            with open("src/config/assets.yaml", "r") as f:
                d = yaml.safe_load(f)
                symbols = list(d.keys())
        except:
            symbols = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
            
    train_multiframe(args.timeframes, args.n_splits, symbols)



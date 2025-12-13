import pandas as pd
import pickle
import os
import sys
import numpy as np
from sklearn.preprocessing import RobustScaler

# Add root to sys.path
sys.path.append(os.getcwd())

from src.features.feature_pipeline import FeatureEngineering, config as feat_config

def rebuild(symbol, tf_code):
    # Determine input path
    path = f"data/{symbol}_{tf_code}.csv"
    if tf_code == '1d' and not os.path.exists(path):
        path = f"data/{symbol}_history.csv"
    
    if not os.path.exists(path):
        print(f"Skipping {symbol} {tf_code} (File not found: {path})")
        return

    print(f"Building Scaler for {symbol} {tf_code}...")
    try:
        df = pd.read_csv(path)
        
        # Init FE
        fe = FeatureEngineering(symbol=symbol)
        
        # Build Features (Validation: clean column names logic is inside build_features)
        # Note: build_features returns (df_enhanced, features_only_df)
        _, features = fe.build_features(df)
        
        # Clean Features same as pipeline
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Fit Scaler
        scaler = RobustScaler()
        scaler.fit(features)
        
        # Construct Suffix
        # predict_all.py uses: run_suffix = f"_{symbol}_{tf_code}"
        suffix = f"_{symbol}_{tf_code}"
        
        out_path = f"data/scaler_features{suffix}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(scaler, f)
            
        print(f"  [OK] Saved {out_path}")
        
    except Exception as e:
        print(f"  [ERROR] Failed {symbol} {tf_code}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting Scaler Rebuild...")
    
    # Assets from config or hardcoded defaults
    assets = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
    
    # Timeframes (Must match keys in tools/predict_all.py tf_map values)
    # tf_map = { "Daily": "1d", "4 Hour": "4h", "1 Hour": "1h", "15 Min": "15m" }
    timeframes = ["1d", "4h", "1h", "15m"]
    
    for symbol in assets:
        for tf in timeframes:
            rebuild(symbol, tf)
            
    print("\nRebuild Complete.")

if __name__ == "__main__":
    main()

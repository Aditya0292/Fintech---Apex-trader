
import pandas as pd
import numpy as np
import sys
import os
import pickle

sys.path.append(os.getcwd())
from src.features.feature_pipeline import FeatureEngineering

def build_dataset_1h():
    print("Building 1H Dataset...")
    
    # Load 1H Data
    path = "data/XAUUSD_1h.csv"
    if not os.path.exists(path):
        print("Error: 1H data not found.")
        return
        
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows.")
    
    # Feature Engineering
    fe = FeatureEngineering()
    df_full, df_features = fe.build_features(df)
    
    # Create Targets (Same logic as 'X_scaled.npy' creation in pipeline)
    # We follow the convention: X_1h.npy, y_class_1h.npy
    
    # Define Target: Next candle Close > Open (Binary) or 3-Class
    # The default FeaturePipeline assumes 'target' column is created?
    # No, feature_pipeline.build_features usually DOES create 'target_class'. 
    # Let's check if 'target_class' is in df_full.
    
    if 'target_class' in df_full.columns:
        y = df_full['target_class'].values
    else:
        # Create default classification target (Next candle direction)
        print("Constructing target...")
        close = df_full['close']
        future_close = close.shift(-1)
        # 3-Class: 0=Bear, 1=Bull, 2=Neutral
        change = (future_close - close) / close
        
        y = np.zeros(len(df_full)) # Default Bear? No.
        # Let's verify config classes. usually: 0=Bear, 1=Bull, 2=Netural?
        # Or 0=Bear, 1=Bull.
        
        # Let's use simple Bull/Bear for now to match training script classes [0,1,2]
        # Bear < -0.001, Bull > 0.001, else Neutral?
        limit = 0.0005
        y = np.where(change > limit, 1, np.where(change < -limit, 0, 2))
        
        # Shift back because row[t] predicts t+1. 
        # But change is (t+1 - t)/t. So row[t] has target of movement of t+1?
        # Yes.
    
    # Align X and y
    # Feature DF aligned with Df Full.
    # Drop NaNs
    valid_idx = ~np.isnan(df_features).any(axis=1) & ~np.isnan(y)
    
    X = df_features[valid_idx].values
    y = y[valid_idx]
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM (Samples, Window, Features) - Window=30 or 50?
    # Training script expects 3D array?
    # src/models/train_ensemble.py Line 96: X = np.load(x_path)
    # Line 127: split(X).
    # Line 131: X_train_seq = X[...]
    # Line 154: input_shape=(X_train_seq.shape[1], ...) meaning (Timesteps, Features).
    # So X MUST be 3D (N, T, F).
    
    window_size = 30 # 1H is faster, maybe shorter window?
    # Create sequences
    X_seq = []
    y_class_seq = []
    y_reg_seq = []
    
    print(f"Creating Sequences (Window={window_size})...")
    close = df_full['close'].values # Need raw close values for reg calc
    
    for i in range(window_size, len(X_scaled)-1): # -1 because we need i+1
        # Feature i
        X_seq.append(X_scaled[i-window_size:i])
        
        # Target Class i (Predicts i+1 direction)
        y_class_seq.append(y[i]) 
        
        # Target Reg i (Actual Return of i+1)
        # return = (close[i+1] - close[i]) / close[i]
        # This matches how backtest calculates 'actual_ret'
        ret = (close[i+1] - close[i]) / close[i]
        y_reg_seq.append(ret)
        
    X_seq = np.array(X_seq)
    y_class_seq = np.array(y_class_seq) 
    y_reg_seq = np.array(y_reg_seq)
    
    print(f"Final Data Shape: {X_seq.shape}, Labels: {y_class_seq.shape}")
    
    # Save
    np.save("data/X_1h.npy", X_seq)
    np.save("data/y_class_1h.npy", y_class_seq)
    np.save("data/y_reg_1h.npy", y_reg_seq) # Save raw returns for Backtest
    joblib.dump(scaler, "data/scaler_features_1h.pkl") # Changed to joblib.dump
    
    print(f"Dataset Built 1H:")
    print(f"  X Shape: {X_seq.shape}")
    print(f"  y_class Shape: {y_class_seq.shape}")
    print(f"  y_reg Saved ({y_reg_seq.shape})") # Updated print statement
    
    # Original print statement was partially overwritten, reconstructing based on context
    print("Saved data/X_1h.npy, data/y_class_1h.npy, data/y_reg_1h.npy, data/scaler_features_1h.pkl")

if __name__ == "__main__":
    build_dataset_1h()

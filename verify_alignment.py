import pandas as pd
import numpy as np
import sys
# sys.path.append(r"c:\Users\adity\OneDrive\Documents\A&B")
from Feature_pipeline import make_windows, build_targets, WINDOW

def test_alignment():
    print("Testing Alignment...")
    N = 200
    # continuous price to make math easy
    # open=i, close=i+0.5
    # Next day i+1: open=i+1, close=i+1.5. Move = (1.5 - 1.0)/1.0 = 0.5 (50%) -> Bullish (>0.1%)
    # So all targets should be 1 if looking at next day.
    
    df = pd.DataFrame({
        'open': np.arange(N, dtype=float) + 100, # start at 100 to avoid div/0
        'close': np.arange(N, dtype=float) + 100.5,
        'high': np.arange(N, dtype=float) + 101,
        'low': np.arange(N, dtype=float) + 99,
        'time': pd.date_range('2024-01-01', periods=N)
    })
    
    # Run pipeline
    y_reg, y_cls = build_targets(df)
    features = df.copy() # simplify features to just df
    X = make_windows(features, window=WINDOW)
    
    # Check lengths
    # X len should be N - WINDOW
    # y_cls len should be N - WINDOW
    print(f"N={N}, WINDOW={WINDOW}")
    print(f"X shape: {X.shape}")
    print(f"y_cls shape: {y_cls.shape}")
    
    # Check last element of X[0]
    # X[0] is df[0..WINDOW-1]. Last element is df[WINDOW-1].
    # Index of last element in X[0] is WINDOW-1.
    last_idx_in_first_seq = WINDOW - 1
    
    # Comparison
    print(f"Sequence 0 ends at index {last_idx_in_first_seq} of DF.")
    
    # Target 0 (y_cls[0])
    # y_cls comes from targets[WINDOW:]
    # So y_cls[0] is targets[WINDOW]
    
    # targets[i] is based on shift(-1) -> prediction for i+1.
    # so targets[WINDOW] is prediction for WINDOW+1.
    
    # If X ends at WINDOW-1, we want prediction for WINDOW.
    # Prediction for WINDOW corresponds to targets[WINDOW-1].
    
    print(f"Target vector index 0 corresponds to raw targets index {WINDOW}")
    print(f"Raw targets index {WINDOW} predicts movement for day {WINDOW+1} (using data from {WINDOW+1})")
    print(f"We NEED prediction for day {WINDOW}.")
    
    if last_idx_in_first_seq == WINDOW - 1:
        desired_target_idx = WINDOW - 1
        actual_target_idx = WINDOW
        print(f"MISALIGNMENT DETECTED: Features end at {last_idx_in_first_seq}, Target predicts {actual_target_idx+1}.")
        print(f"Gap size: {actual_target_idx - desired_target_idx} days.")
    
if __name__ == "__main__":
    test_alignment()

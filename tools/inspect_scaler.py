import pickle
import numpy as np

try:
    with open("data/scaler_features_1h.pkl", "rb") as f:
        scaler = pickle.load(f)
        print(f"Scaler (1h) expects {scaler.n_features_in_} features.")
        # if hasattr(scaler, 'feature_names_in_'):
        #     print(scaler.feature_names_in_)
        # Standard scaler might not have feature names if trained on numpy array
        
    with open("data/scaler_features.pkl", "rb") as f:
        scaler_d = pickle.load(f)
        print(f"Scaler (Daily) expects {scaler_d.n_features_in_} features.")

except Exception as e:
    print(e)

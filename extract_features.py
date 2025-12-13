import pandas as pd
import json
from src.features.feature_pipeline import FeatureEngineering

df = pd.read_csv('data/XAUUSD_history.csv').tail(100)
fe = FeatureEngineering()
_, features = fe.build_features(df)
cols = list(features.columns)

print(f"Count: {len(cols)}")
with open("feature_list.json", "w") as f:
    json.dump(cols, f)

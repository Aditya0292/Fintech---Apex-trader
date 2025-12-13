import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Add root
sys.path.append(os.getcwd())

from src.utils.logger import get_logger
from tools.predict import Predictor

logger = get_logger("validator")

def validate_system():
    report_path = Path("reports")
    report_path.mkdir(exist_ok=True)
    report_file = report_path / "system_health.txt"
    
    with open(report_file, "w") as f:
        f.write("APEX TRADE AI - SYSTEM HEALTH REPORT\n")
        f.write("====================================\n\n")
        
        # 1. Feature Schema
        f.write("1. Feature Schema Validation\n")
        f.write("----------------------------\n")
        try:
            with open("src/config/feature_schema.json", "r") as sf:
                schema = json.load(sf)
                n_features = len(schema['features'])
                f.write(f"Schema loaded. defined features: {n_features}\n")
        except Exception as e:
            f.write(f"Schema Error: {e}\n")
            
        # 2. Model Artifacts
        f.write("\n2. Model Artifacts Check\n")
        f.write("------------------------\n")
        target_models = ["", "_4h", "_1h", "_30m", "_15m"]
        model_types = ["xgboost_model", "lightgbm_model.txt", "bilstm_model.h5", "transformer_model.h5", "meta_learner.pkl"]
        
        missing = []
        for suffix in target_models:
            suffix_clean = "_" + suffix.lstrip("_") if suffix else ""
            f.write(f"\n[Timeframe Suffix: {suffix_clean if suffix_clean else '(Daily)'}]\n")
            for m in model_types:
                # Handle suffix logic for filenames
                if m.endswith(".pkl"):
                     fname = f"meta_learner{suffix_clean}.pkl"
                elif "xgboost" in m:
                     fname = f"xgboost_model{suffix_clean}.json"
                elif "lightgbm" in m:
                     fname = f"lightgbm_model{suffix_clean}.txt"
                elif "bilstm" in m:
                     fname = f"bilstm_model{suffix_clean}.h5"
                elif "transformer" in m:
                     fname = f"transformer_model{suffix_clean}.h5"
                
                path = Path("saved_models") / fname
                if path.exists():
                    f.write(f"  OK: {fname} ({path.stat().st_size / 1024:.1f} KB)\n")
                else:
                    f.write(f"  MISSING: {fname}\n")
                    missing.append(fname)
                    
        if missing:
            f.write(f"\nCRITICAL: {len(missing)} model artifacts are missing!\n")
        else:
            f.write("\nAll model artifacts present.\n")
            
        # 3. Validation Run (Load & Predict)
        f.write("\n3. Functional Validation (Predictor Load)\n")
        f.write("-----------------------------------------\n")
        
        timeframes = ["Daily", "4 Hour", "1 Hour", "30 Min", "15 Min"]
        tf_map = {"Daily": "", "4 Hour": "4h", "1 Hour": "1h", "30 Min": "30m", "15 Min": "15m"}
        
        for tf in timeframes:
            code = tf_map[tf]
            f.write(f"\nTesting {tf} Predictor...\n")
            try:
                # Mock Data Load
                data_path = f"data/XAUUSD_{code}.csv" if code else "data/XAUUSD_history.csv"
                if not os.path.exists(data_path):
                    f.write(f"  Skip: Data {data_path} not found.\n")
                    continue
                    
                df = pd.read_csv(data_path)
                
                # Ensure integrity for test
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.sort_values('time')
                    
                # Take last 300 rows (Sufficient for SMC 4H window of 200)
                df_sample = df.tail(300)
                
                predictor = Predictor(timeframe=tf)
                res = predictor.predict(df_sample)
                
                if "error" in res:
                    f.write(f"  FAILURE: {res['error']}\n")
                else:
                    f.write(f"  SUCCESS. Signal: {res['prediction']} (Conf: {res['confidence']:.2f})\n")
                    bull_obs = res['smc'].get('bull_obs_found') or []
                    f.write(f"  SMC Zones: {len(bull_obs)} Bull OBs found.\n")
                    
            except Exception as e:
                f.write(f"  CRASH: {e}\n")

    print(f"Validation Report generated at {report_file}")
    with open(report_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    validate_system()

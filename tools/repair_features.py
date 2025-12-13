
import sys
import os
import argparse
import pandas as pd
import numpy as np

# Add root
sys.path.append(os.getcwd())

from src.features.feature_pipeline import run_pipeline

def repair_dataset(input_path, schema_path, out_path):
    print(f"Repairing features for {input_path} using {schema_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)
        
    try:
        # We assume run_pipeline handles loading, feature engineering, and schema validation.
        # run_pipeline returns (features_df, target_series) or saves internally.
        # We need to ensure we capture the output.
        
        # NOTE: run_pipeline signature is (data_path, suffix, ...).
        # It usually saves to data/{data_filename without ext}{suffix}_features.csv
        # We want to force output to `out_path`.
        # This requires some adaptation if feature_pipeline doesn't support custom out path.
        
        # Alternative: We use FeatureEngineering class directly.
        from src.features.feature_pipeline import FeatureEngineering
        from src.utils.schema_validator import validate_feature_schema
        import json
        
        fe = FeatureEngineering()
        
        # Load
        df = pd.read_csv(input_path)
        
        # Generate Features (This might include time alignment, indicators etc)
        # Assuming df has OHLCV
        _, df_processed = fe.build_features(df)
        
        # Enforce Schema
        df_validated = validate_feature_schema(df_processed)
        
        # Save output
        if out_path.endswith('.npy'):
            np.save(out_path, df_validated.values)
        else:
            df_validated.to_csv(out_path, index=True)
            
        print(f"✅ Saved repaired features to {out_path}")
        
    except Exception as e:
        print(f"❌ Repair failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--schema", required=True, help="Schema JSON path")
    parser.add_argument("--out", required=True, help="Output file path (.npy or .csv)")
    
    args = parser.parse_args()
    
    repair_dataset(args.input, args.schema, args.out)


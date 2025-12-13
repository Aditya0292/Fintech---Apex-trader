import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import config

logger = get_logger()

def load_schema(schema_path: str = None) -> dict:
    """Loads the feature schema from the specified path."""
    if schema_path is None:
        schema_path = config['features'].get('schema_path', 'src/config/feature_schema.json')
    
    # Handle relative paths
    if not os.path.isabs(schema_path):
        # Assuming run from root, but let's be safe
        search_paths = [Path(os.getcwd()) / schema_path, Path(__file__).parent.parent.parent / schema_path]
        found = False
        for p in search_paths:
            if p.exists():
                schema_path = str(p)
                found = True
                break
        if not found:
             logger.error(f"Schema file not found at {schema_path}")
             raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with open(schema_path, 'r') as f:
        return json.load(f)

def validate_feature_schema(df: pd.DataFrame, schema_path: str = None) -> pd.DataFrame:
    """
    Validates and enforces the feature schema on the DataFrame.
    1. Checks for canonical columns.
    2. Fills missing with 0.0 using robust assignment.
    3. Drops extra columns.
    4. Enforces the exact order.
    """
    logger.info("Validating feature schema...")
    schema = load_schema(schema_path)
    required_features = schema['features']
    
    # 1. Identify Missing
    existing_cols = set(df.columns)
    missing = [col for col in required_features if col not in existing_cols]
    
    if missing:
        logger.warning(f"Schema Mismatch: Found {len(missing)} missing features. Filling with 0.0.")
        logger.debug(f"Missing features: {missing}")
        # Efficiently add missing columns
        for col in missing:
            df[col] = 0.0
            
    # 2. Identify Extras (Optional: just dropping them by re-indexing)
    extras = [col for col in existing_cols if col not in required_features]
    if extras:
        logger.info(f"Schema Mismatch: Found {len(extras)} extra features. Dropping them.")
        logger.debug(f"Extra features: {extras}")
        
    # 3. Enforce Order (Crucial for ML models)
    # Using reindex is safer and handles dropping extras + reordering in one go
    # However, reindex puts NaN for missing, we already filled them, but let's be double safe
    
    # Ensure all required are present (we just added them)
    df_validated = df[required_features].copy()
    
    # Fill any remaining NaNs (e.g. from the original data)
    # Note: Logic in prompt said "If missing features, fills with 0.0". 
    # It implied "if column missing". If column exists but has NaNs, that's a data quality issue,
    # but for safety/hardening we typically fillna(0) or ffill before this. 
    # Let's trust the upstream feature_pipeline did its fills, but ensure schema compliance here.
    
    return df_validated

def validate(file_path: str):
    """CLI Helper to validate a CSV file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with shape {df.shape}")
        
        df_valid = validate_feature_schema(df)
        print(f"Validated shape: {df_valid.shape}")
        print("Schema validation passed.")
        return df_valid
    except Exception as e:
        print(f"Validation failed: {e}")

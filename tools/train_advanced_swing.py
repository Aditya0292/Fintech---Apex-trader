
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Add root project to path
sys.path.append(os.getcwd())
try:
    from src.features.feature_pipeline import FeatureEngineering
except ImportError:
    # Try slightly deeper path if running from root
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
    from src.features.feature_pipeline import FeatureEngineering

def train_advanced_swing():
    print("Initializing Standard Feature Pipeline (SMC + ICT + Pivots)...")
    fe = FeatureEngineering()
    
    print("Loading Daily Data...")
    path = "data/XAUUSD_history.csv"
    if not os.path.exists(path):
        print("Error: data/XAUUSD_history.csv not found.")
        return

    df = pd.read_csv(path)
    
    print("Building 67+ Features...")
    # This generates all the SMC, Order Blocks, FVGs attached to the dataframe
    # build_features returns (df_with_cols, df_numeric_features)
    df_full, df_features_numeric = fe.build_features(df)
    
    # Check for NaN (Feature engineering often creates NaNs at start)
    df_features = df_full.dropna()
    
    print(f"Features Generated: {df_features.shape[1]} columns")
    
    # --- CREATE SWING TARGET (5 Days) ---
    horizon = 5
    # Calculate Future Return
    df_features['future_close'] = df_features['close'].shift(-horizon)
    df_features['return_5d'] = (df_features['future_close'] - df_features['close']) / df_features['close']
    
    # Threshold: 0.5% move required to be "Actionable Swing"
    threshold = 0.005
    
    # Target Class: 1=Bull, 0=Bear, -1=Neutral (We will filter Neutrals for training to sharpen edge)
    conditions = [
        (df_features['return_5d'] > threshold),
        (df_features['return_5d'] < -threshold)
    ]
    choices = [1, 0]
    df_features['target'] = np.select(conditions, choices, default=-1)
    
    # Drop rows where target cannot be calculated (last 5 rows)
    df_features = df_features.dropna(subset=['target', 'return_5d'])
    
    # FILTER: Remove Neutrals (Optional, but helps learn strong signals)
    # Strategy: "I only want to know if it's DEFINITELY Up or Down"
    # df_model = df_features[df_features['target'] != -1]
    
    # Actually, for accuracy metric, let's keep it Binary (Up vs Down/Flat) or strict Up vs Down
    # Let's try Strict Up vs Down to see if we have an edge on clear moves.
    df_model = df_features[df_features['target'] != -1].copy()
    
    print(f"Training Samples (After filtering for >0.5% moves): {len(df_model)}")
    
    # Clean Inputs
    # Drop non-feature columns
    exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume', 
                    'target', 'return_5d', 'future_close', 
                    # Drop the report-only columns we added recently if they exist
                    'R2_traditional', 'S2_traditional', 'R2_fibonacci', 'S2_fibonacci',
                    'prediction_target', 'target_class'] # columns from fe pipeline
                    
    feature_cols = [c for c in df_model.columns if c not in exclude_cols]
    # Filter strictly numeric
    X = df_model[feature_cols].select_dtypes(include=[np.number])
    y = df_model['target']
    
    print(f"Input Features: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train XGBoost
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03, # Slower learning for better generalization
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("\n" + "="*40)
    print(f"ADVANCED SWING MODEL ACCURACY (SMC/ICT Features)")
    print("="*40)
    print(f"Horizon: 5 Days")
    print(f"Feature Count: {X.shape[1]}")
    print(f"Accuracy: {acc:.2%}")
    print("-" * 40)
    print(classification_report(y_test, preds, target_names=['Bearish Swing', 'Bullish Swing']))
    
    # Feature Importance
    print("\nTop 5 Important Features for Swing:")
    fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    print(fi.sort_values('importance', ascending=False).head(5))
    
    # Save
    if not os.path.exists("saved_models"): os.makedirs("saved_models")
    joblib.dump(model, "saved_models/swing_model_advanced_xgb.pkl")
    print("\nModel saved to saved_models/swing_model_advanced_xgb.pkl")

if __name__ == "__main__":
    train_advanced_swing()

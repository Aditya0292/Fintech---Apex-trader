"""
APEX TRADE AI - Evaluation & Confidence Analysis
================================================
Goal: Verify if Precision >= 80% at Confidence >= 65%

Steps:
1. Load Models & Data
2. Generate Probabilities
3. Apply Confidence Thresholds
4. Compute Metrics (Precision, Coverage)
5. Plot Results
"""

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve

import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================================
# CONFIGURATION
# ============================================================================

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Config:
    X_PATH = "data/X.npy"
    Y_PATH = "data/y_class.npy"
    MODEL_DIR = Path("saved_models")
    OUTPUT_DIR = Path("images") # Save plots to images folder
    THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

config = Config()
from src.utils.logger import get_logger
logger = get_logger()

# ============================================================================
# UTILS
# ============================================================================

def load_data(suffix=""):
    if suffix:
        suffix_clean = "_" + suffix.lstrip("_")
    else:
        suffix_clean = ""
        
    x_path = f"data/X{suffix_clean}.npy"
    y_path = f"data/y_class{suffix_clean}.npy"
    
    # Fallback to config if not found (or error)
    if not os.path.exists(x_path):
        if not suffix:
             x_path = config.X_PATH
             y_path = config.Y_PATH
        else:
             raise FileNotFoundError(f"{x_path} not found")

    X = np.load(x_path)
    y = np.load(y_path)
    
    # Feature Engineering for Trees
    X_last = X[:, -1, :]
    X_mean = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)
    X_tree = np.hstack([X_last, X_mean, X_std])
    
    return X, y, X_tree

from src.models.model_factory import ModelFactory # Updated location

def load_all_models(input_shape, suffix=""):
    logger.info(f"Loading models (Suffix: {suffix})...")
    
    if suffix:
        suffix_clean = "_" + suffix.lstrip("_")
    else:
        suffix_clean = ""
        
    try:
        # XGBoost
        try:
            model_xgb = xgb.XGBClassifier()
            model_xgb.load_model(config.MODEL_DIR / f"xgboost_model{suffix_clean}.json")
        except Exception as e:
            logger.warning(f"  Warning: XGBoost load failed: {e}")
            model_xgb = None

        # LightGBM
        try:
            model_lgb = lgb.Booster(model_file=str(config.MODEL_DIR / f"lightgbm_model{suffix_clean}.txt"))
        except Exception as e:
            print(f"  Warning: LightGBM load failed: {e}")
            model_lgb = None

        # Keras Models (Rebuild and Load Weights)
        print("  Rebuilding Keras models...")
        
        try:
            model_lstm = ModelFactory.get_bilstm_attention(input_shape)
            model_lstm.load_weights(config.MODEL_DIR / f"bilstm_model{suffix_clean}.h5")
        except Exception as e:
            print(f"  Warning: BiLSTM load/weights failed: {e}")
            model_lstm = None
            
        try:
            model_trans = ModelFactory.get_transformer(input_shape)
            model_trans.load_weights(config.MODEL_DIR / f"transformer_model{suffix_clean}.h5")
        except Exception as e:
             print(f"  Warning: Transformer load/weights failed: {e}")
             model_trans = None
        # Meta Learner
        try:
            with open(config.MODEL_DIR / f"meta_learner{suffix_clean}.pkl", "rb") as f:
                meta_learner = pickle.load(f)
        except Exception as e:
            logger.warning(f"  Warning: Meta Learner load failed: {e}")
            meta_learner = None
            
        return [model_xgb, model_lgb, model_lstm, model_trans, meta_learner]

    except Exception as e:
        logger.error(f"CRITICAL: Model loading crashed: {e}")
        return [None, None, None, None, None]

def evaluate(suffix=""):
    print(f"Evaluating (Suffix: {suffix})...")
    X, y, X_tree = load_data(suffix)
    
    # Use last 20% for evaluation (Consistent with last fold)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    X_tree_test = X_tree[split_idx:]
    y_test = y[split_idx:]
    
    print(f"Evaluation Set: {len(y_test)} samples")
    
    # Input Shape for Deep Learning Models
    input_shape = (X_test.shape[1], X_test.shape[2])
    
    models = load_all_models(input_shape, suffix)
    xgb_m, lgb_m, lstm_m, trans_m, meta_m = models
    
    if xgb_m is None:
         print("Error: Models not loaded.")
         return

    # Generate Base Predictions
    print("Generating base predictions...")
    p_xgb = xgb_m.predict_proba(X_tree_test)
    p_lgb = lgb_m.predict(X_tree_test) 
    p_lstm = lstm_m.predict(X_test, verbose=0)
    p_trans = trans_m.predict(X_test, verbose=0)
    
    # Stack
    stacked_input = np.hstack([p_xgb, p_lgb, p_lstm, p_trans])
    
    # Meta Prediction
    print("Generating calibrated meta predictions...")
    final_probs = meta_m.predict_proba(stacked_input)
    final_preds = np.argmax(final_probs, axis=1)
    
    # Overall Accuracy
    acc = accuracy_score(y_test, final_preds)
    print(f"Overall Accuracy: {acc:.4f}")
    
    # Metrics per Threshold (Simplified output for quick view)
    print("\nConfidence Analysis:")
    print(f"{'Threshold':<10} | {'Bull Prec':<10} | {'Bear Prec':<10} | {'Coverage':<10}")
    print("-" * 50)
    
    for thresh in config.THRESHOLDS:
        max_probs = np.max(final_probs, axis=1)
        mask = max_probs >= thresh
        if np.sum(mask) == 0: continue
            
        y_true_f = y_test[mask]
        y_pred_f = final_preds[mask]
        
        prec_bull = precision_score(y_true_f, y_pred_f, labels=[1], average=None, zero_division=0)[0] if 1 in y_pred_f else 0
        prec_bear = precision_score(y_true_f, y_pred_f, labels=[0], average=None, zero_division=0)[0] if 0 in y_pred_f else 0
        coverage = np.mean(mask) * 100
        
        print(f"{thresh:<10.2f} | {prec_bull:<10.4f} | {prec_bear:<10.4f} | {coverage:<10.1f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    
    evaluate(args.suffix)

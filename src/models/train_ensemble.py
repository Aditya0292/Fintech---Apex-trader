"""
APEX TRADE AI - Multi-Model Ensemble Training Pipeline
======================================================
Models:
1. XGBoost
2. LightGBM
3. BiLSTM + Attention
4. Transformer Encoder
5. Meta-Learner (Stacking)

Strategy:
- Walk-Forward Validation (5 splits)
- Class Weighting for Imbalance
- Probability Calibration
- Ensemble Stacking
"""

import sys
import os
# Add root directory to sys.path to allow running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.models.model_factory import ModelFactory

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    SEED = 42
    N_SPLITS = 5
    EPOCHS = 30 # Reduced for speed, increase for prod
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    CLASSES = [0, 1, 2] # Bear, Bull, Neutral
    
    # Paths
    X_PATH = "data/X.npy"
    Y_PATH = "data/y_class.npy"
    MODEL_DIR = Path("saved_models")

    def __init__(self):
        self.MODEL_DIR.mkdir(exist_ok=True)

config = Config()
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

# ... (Model Definitions omitted for brevity, they are unchanged) ...

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_ensemble(suffix=""):
    print(f"Loading data (Suffix: {suffix})...")
    
    # Update Config Paths dynamically
    if suffix:
        suffix_clean = "_" + suffix.lstrip("_")
        x_path = f"data/X{suffix_clean}.npy"
        y_path = f"data/y_class{suffix_clean}.npy"
    else:
        suffix_clean = ""
        x_path = config.X_PATH
        y_path = config.Y_PATH
        
    if not os.path.exists(x_path):
        print(f"Error: Data file {x_path} not found.")
        return

    X = np.load(x_path)
    y = np.load(y_path)
    
    print(f"Data shape: {X.shape}, Labels: {y.shape}")
    
    # Prepare flatten data for Tree models
    X_flat = X.reshape(X.shape[0], -1) 
    # Or better: Extract statistical features from sequence (Mean, Max, Min, Last)
    # Simple Flatten is (Samples, Time*Features) -> Very high dim.
    # Let's use Last Step Features + Mean/Std of window
    X_last = X[:, -1, :]
    X_mean = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)
    X_tree = np.hstack([X_last, X_mean, X_std])
    print(f"Tree Input Shape: {X_tree.shape}")
    
    # Store meta-features (predictions)
    meta_X = np.zeros((len(X), 3 * 4)) # 3 classes * 4 models
    meta_y = y
    
    # Walk-Forward Validation
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS)
    
    print(f"Starting Walk-Forward Validation ({config.N_SPLITS} splits)...")
    
    model_xgb = None
    model_lgb = None
    model_lstm = None
    model_trans = None
    
    fold = 0
    for train_index, test_index in tscv.split(X):
        fold += 1
        print(f"\n[Fold {fold}/{config.N_SPLITS}] Train: {len(train_index)}, Test: {len(test_index)}")
        
        X_train_seq, X_test_seq = X[train_index], X[test_index]
        X_train_tree, X_test_tree = X_tree[train_index], X_tree[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Class Weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        # --- 1. XGBoost ---
        print("  Training XGBoost...")
        model_xgb = ModelFactory.get_xgboost(X_train_tree.shape[1])
        model_xgb.fit(X_train_tree, y_train, eval_set=[(X_test_tree, y_test)], verbose=False)
        pred_xgb = model_xgb.predict_proba(X_test_tree)
        
        # --- 2. LightGBM ---
        print("  Training LightGBM...")
        model_lgb = ModelFactory.get_lightgbm(X_train_tree.shape[1])
        model_lgb.fit(X_train_tree, y_train, eval_set=[(X_test_tree, y_test)], eval_metric='logloss')
        pred_lgb = model_lgb.predict_proba(X_test_tree)
        
        # --- 3. BiLSTM ---
        print("  Training BiLSTM...")
        model_lstm = ModelFactory.get_bilstm_attention((X_train_seq.shape[1], X_train_seq.shape[2]))
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model_lstm.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test), 
                       epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, 
                       callbacks=[es], class_weight=class_weight_dict, verbose=0)
        pred_lstm = model_lstm.predict(X_test_seq, verbose=0)
        
        # --- 4. Transformer ---
        print("  Training Transformer...")
        model_trans = ModelFactory.get_transformer((X_train_seq.shape[1], X_train_seq.shape[2]))
        model_trans.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test), 
                        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, 
                        callbacks=[es], class_weight=class_weight_dict, verbose=0)
        pred_trans = model_trans.predict(X_test_seq, verbose=0)
        
        # Store predictions for Stacking
        stacked_preds = np.hstack([pred_xgb, pred_lgb, pred_lstm, pred_trans])
        meta_X[test_index] = stacked_preds
        
        # Fold Evaluation
        acc_xgb = accuracy_score(y_test, np.argmax(pred_xgb, axis=1))
        acc_lstm = accuracy_score(y_test, np.argmax(pred_lstm, axis=1))
        print(f"  Result -> XGB Acc: {acc_xgb:.4f}, LSTM Acc: {acc_lstm:.4f}")
        
    # ============================================================================
    # META LEARNER (STACKING)
    # ============================================================================
    print("\nTraining Meta-Learner (Stacking)...")
    
    valid_indices = np.where(np.sum(meta_X, axis=1) > 0)[0]
    
    if len(valid_indices) == 0:
        print("Error: No valid predictions for meta-learner.")
        return

    meta_X_train = meta_X[valid_indices]
    meta_y_train = meta_y[valid_indices]
    
    meta_learner = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    meta_learner.fit(meta_X_train, meta_y_train)
    
    calibrated_meta = CalibratedClassifierCV(meta_learner, cv=3, method='isotonic')
    calibrated_meta.fit(meta_X_train, meta_y_train)
    
    # Save Models with suffix
    print(f"Saving models (Suffix: {suffix_clean})...")
    model_xgb.save_model(config.MODEL_DIR / f"xgboost_model{suffix_clean}.json")
    model_lgb.booster_.save_model(config.MODEL_DIR / f"lightgbm_model{suffix_clean}.txt")
    model_lstm.save(config.MODEL_DIR / f"bilstm_model{suffix_clean}.h5")
    model_trans.save(config.MODEL_DIR / f"transformer_model{suffix_clean}.h5")
    
    with open(config.MODEL_DIR / f"meta_learner{suffix_clean}.pkl", "wb") as f:
        pickle.dump(calibrated_meta, f)
        
    print("Complete.")

    # ============================================================================
    # EVALUATION ON META SET
    # ============================================================================
    print("\nEvaluating Ensemble...")
    meta_probs = calibrated_meta.predict_proba(meta_X_train)
    meta_preds = np.argmax(meta_probs, axis=1)
    
    print(classification_report(meta_y_train, meta_preds, target_names=['Bear', 'Bull', 'Neutral']))
    
    # Confusion Matrix (Skip plot for batch mode)
    # cm = confusion_matrix(meta_y_train, meta_preds)
    # plt.figure(figsize=(6,5))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title("Ensemble Confusion Matrix (Validation)")
    # plt.savefig("ensemble_confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    
    train_ensemble(args.suffix)

"""
APEX TRADE AI - Production Inference
====================================
Usage:
    predictor = Predictor()
    prediction = predictor.predict(df_latest)
"""

import numpy as np
import pandas as pd
import pickle
import json
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model, Model
from pathlib import Path
import sys

# Ensure we can import from local modules
import sys
sys.path.append(".") 
from src.features.feature_pipeline import FeatureEngineering, config as feat_config
from src.models.train_ensemble import config as train_config
# from src.models.model_factory import ModelFactory # Not actually used explicitly as load_all_models handles it, 
# but load_all_models is in evaluate which now imports it correctly.
# Predictor uses load_all_models from evaluate.
from evaluate import load_all_models 

class Predictor:
    def __init__(self, model_dir="saved_models", timeframe="Daily"):
        self.model_dir = Path(model_dir)
        self.fe = FeatureEngineering()
        self.scaler = None
        self.models = None
        self.timeframe = timeframe
        self.suffix = self._get_suffix(timeframe)
        
        self._load_resources()
    
    def _get_suffix(self, timeframe):
        map_tf = {
            "Daily": "",
            "4 Hour": "_4h",
            "1 Hour": "_1h",
            "15 Min": "_15m",
            "5 Min": "_5m",
            "4h": "_4h",
            "1h": "_1h",
            "15m": "_15m",
            "5m": "_5m",
            "d1": ""
        }
        return map_tf.get(timeframe, "")

    def _load_resources(self):
        print(f"Loading resources (Timeframe: {self.timeframe}, Suffix: '{self.suffix}')...")
        # Scaler
        try:
            with open(f"data/scaler_features{self.suffix}.pkl", "rb") as f:
                self.scaler = pickle.load(f)
        except:
            # Fallback to daily scaler if specific not found (Risky but better than crash)
            print("  Warning: Specific scaler not found, trying default...")
            try:
                with open("data/scaler_features.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
            except:
                with open("../data/scaler_features.pkl", "rb") as f:
                     self.scaler = pickle.load(f)
            
        # Models
        input_shape = (50, 67) 
        self.models = load_all_models(input_shape, self.suffix)
        
    def predict(self, df: pd.DataFrame):
        """
        Predict on new data.
        df: DataFrame with OHLCV. Must contain at least 250 rows to calculate indicators.
        Returns: Dict with prediction details.
        """
        # 1. Feature Engineering
        print(f"Predicting ({self.timeframe}): Generating Features...")
        df_enhanced, features = self.fe.build_features(df)
        
        # 2. Scale
        X_scaled = self.scaler.transform(features)
        
        # 3. Create Sequence (Last window)
        win = feat_config.WINDOW # 50
        if len(X_scaled) < win:
            raise ValueError(f"Not enough data. Need {win} rows, got {len(X_scaled)}")
            
        # Take the last sequence
        X_seq = X_scaled[-win:].reshape(1, win, -1)
        
        # 4. Prepare inputs for ensemble
        # Tree input: Last Step + Mean + Std
        X_last = X_seq[:, -1, :]
        X_mean = np.mean(X_seq, axis=1)
        X_std = np.std(X_seq, axis=1)
        X_tree = np.hstack([X_last, X_mean, X_std])
        
        if self.models[0] is None:
             return {"error": "Models not loaded correctly."}

        xgb_m, lgb_m, lstm_m, trans_m, meta_m = self.models
        
        # 5. Base Predictions
        p_xgb = xgb_m.predict_proba(X_tree)
        p_lgb = lgb_m.predict(X_tree) 
        
        p_lstm = lstm_m.predict(X_seq, verbose=0) if lstm_m else np.zeros((1,3))
        p_trans = trans_m.predict(X_seq, verbose=0) if trans_m else np.zeros((1,3))
        
        # 6. Stacking
        stacked = np.hstack([p_xgb, p_lgb, p_lstm, p_trans])
        
        # 7. Meta Prediction
        if meta_m:
            final_probs = meta_m.predict_proba(stacked)[0]
        else:
            # Fallback average
            final_probs = np.mean([p_xgb[0], p_lgb[0], p_lstm[0], p_trans[0]], axis=0) # simplified
            
        pred_class = int(np.argmax(final_probs))
        confidence = float(final_probs[pred_class])
        
        classes = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
        
        # Result Logic (Trade Levels, Date)
        # Calculate Target Date
        last_date = pd.to_datetime(df.iloc[-1]['time']) if 'time' in df.columns else pd.Timestamp.now()
        
        # Timeframe specific delta
        if self.timeframe in ["4 Hour", "4h"]:
             target_date = last_date + pd.Timedelta(hours=4)
        elif self.timeframe in ["1 Hour", "1h"]:
             target_date = last_date + pd.Timedelta(hours=1)
        elif self.timeframe in ["15 Min", "15m"]:
             target_date = last_date + pd.Timedelta(minutes=15)
        elif self.timeframe in ["5 Min", "5m"]:
             target_date = last_date + pd.Timedelta(minutes=5)
        else: # Daily fallback
            dow = last_date.dayofweek
            if dow >= 4: 
                days_to_add = 7 - dow
            else:
                days_to_add = 1
            target_date = last_date + pd.Timedelta(days=days_to_add)
        
        # Extract Context features (from last row)
        dxy_trend = int(df_enhanced.iloc[-1].get('dxy_trend', 0))
        dxy_corr = float(df_enhanced.iloc[-1].get('gold_dxy_corr', 0.0))
        news_score = float(df_enhanced.iloc[-1].get('news_sentiment', 0.0))
        
        # Trade Levels Calculation (Modified for Timeframe)
        last_close = float(df.iloc[-1]['close'])
        last_atr = float(df_enhanced.iloc[-1]['atr'])
        
        entry_price = last_close
        
        # Adjust multipliers based on timeframe? 
        # ATR naturally scales with timeframe, so multipliers can stay similar for structure.
        # But scalping usually uses tighter multipliers.
        risk_per_share = last_atr * 1.5
        reward_per_share = last_atr * 2.0
        
        if pred_class == 1: # Bullish
            sl_price = entry_price - risk_per_share
            tp_price = entry_price + reward_per_share
        elif pred_class == 0: # Bearish
            sl_price = entry_price + risk_per_share
            tp_price = entry_price - reward_per_share
        else:
            sl_price, tp_price = 0, 0

        result = {
            "prediction": classes[pred_class],
            "class_id": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "bearish": round(float(final_probs[0]), 4),
                "bullish": round(float(final_probs[1]), 4),
                "neutral": round(float(final_probs[2]), 4)
            },
            "trade_levels": {
                "timeframe": self.timeframe,
                "entry": round(entry_price, 2),
                "sl": round(sl_price, 2),
                "tp": round(tp_price, 2),
                "atr": round(last_atr, 2)
            },
            "context": {
                "dxy_trend": "Bullish" if dxy_trend == 1 else "Bearish" if dxy_trend == -1 else "Neutral",
                "dxy_correlation": round(dxy_corr, 2),
                "news_sentiment": round(news_score, 2)
            },
            "input_time": last_date.strftime('%Y-%m-%d %H:%M:%S'),
            "forecast_time": target_date.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="Daily", help="Daily, 4h, 1h, 15m, 5m")
    args = parser.parse_args()
    
    # Map timeframe to file
    tf_file_map = {
        "Daily": "XAUUSD_history.csv",
        "4h": "XAUUSD_4h.csv",
        "4 Hour": "XAUUSD_4h.csv",
        "1h": "XAUUSD_1h.csv",
        "1 Hour": "XAUUSD_1h.csv",
        "15m": "XAUUSD_15m.csv",
        "15 Min": "XAUUSD_15m.csv",
        "5m": "XAUUSD_5m.csv",
        "5 Min": "XAUUSD_5m.csv"
    }
    
    filename = tf_file_map.get(args.timeframe, "XAUUSD_history.csv")
    path = f"data/{filename}"
    
    print(f"Running prediction for {args.timeframe} using {path}")
    
    try:
        try:
            df = pd.read_csv(path)
        except:
            # Fallback to root data if running from tools/
            df = pd.read_csv(f"../data/{filename}")
            
        pred = Predictor(timeframe=args.timeframe)
        res = pred.predict(df)
        print("\nResult:")
        print(json.dumps(res, indent=2))
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        # If file missing (e.g. 5m data not fetched yet), alert user
        if "No such file" in str(e):
             print(f"Suggestion: Run 'python tools/train_multiframe.py' to generate models and data for {args.timeframe}.")

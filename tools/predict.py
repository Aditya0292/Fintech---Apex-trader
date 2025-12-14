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
from src.utils.logger import get_logger
# from src.models.model_factory import ModelFactory # Not actually used explicitly as load_all_models handles it, 
# but load_all_models is in evaluate which now imports it correctly.
# Predictor uses load_all_models from evaluate.
try:
    from tools.evaluate import load_all_models
except ImportError:
    from evaluate import load_all_models 

class Predictor:
    def __init__(self, model_dir="saved_models", timeframe="Daily", run_id=None, symbol="XAUUSD"):
        self.model_dir = Path(model_dir)
        self.fe = FeatureEngineering(symbol=symbol)
        self.scaler = None
        self.models = None
        self.timeframe = timeframe
        self.symbol = symbol
        # If run_id is provided, use it directly as the suffix (it should include the leading underscore if needed)
        # e.g. run_id = "_XAUUSD_1h"
        self.suffix = run_id if run_id else self._get_suffix(timeframe)
        self.logger = get_logger()
        
        self._load_resources()
    
    def _get_suffix(self, timeframe):
        map_tf = {
            "Daily": "",
            "4 Hour": "_4h",
            "1 Hour": "_1h",
            "30 Min": "_30m",
            "15 Min": "_15m",
            "5 Min": "_5m",
            "4h": "_4h",
            "1h": "_1h",
            "30m": "_30m",
            "15m": "_15m",
            "5m": "_5m",
            "d1": ""
        }
        return map_tf.get(timeframe, "")

    def _load_resources(self):
        self.logger.info(f"Loading resources (Timeframe: {self.timeframe}, Suffix: '{self.suffix}')...")
        # Scaler
        try:
            with open(f"data/scaler_features{self.suffix}.pkl", "rb") as f:
                self.scaler = pickle.load(f)
        except:
            # Fallback to daily scaler if specific not found (Risky but better than crash)
            self.logger.warning("  Warning: Specific scaler not found, trying default...")
            try:
                with open("data/scaler_features.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
            except:
                with open("../data/scaler_features.pkl", "rb") as f:
                     self.scaler = pickle.load(f)
            
        # Models
        n_features = 67 # Default fallback
        if self.scaler and hasattr(self.scaler, 'n_features_in_'):
            n_features = self.scaler.n_features_in_
            
        input_shape = (50, n_features)
        self.logger.info(f"  Model Input Shape: {input_shape}")
        self.models = load_all_models(input_shape, self.suffix)
        
    def predict(self, df: pd.DataFrame, news_override: float = None):
        """
        Predict on new data.
        df: DataFrame with OHLCV. Must contain at least 250 rows to calculate indicators.
        news_override: Optional float (-1.0 to 1.0) to override news_sentiment feature.
        Returns: Dict with prediction details.
        """
        # 0. Enforce Data Integrity
        if 'time' in df.columns:
            # Check if string, convert
            if df['time'].dtype == object: 
                df['time'] = pd.to_datetime(df['time'])
            # Sort is critical for SMC and last_close
            df = df.sort_values('time')
            
        # 1. Feature Engineering
        self.logger.info(f"Predicting ({self.timeframe}): Generating Features...")
        df_enhanced, features = self.fe.build_features(df)
        
        # Override News Sentiment if provided
        if news_override is not None:
             self.logger.info(f"  [Info] Overriding News Sentiment: {news_override}")
             # We need to update both 'news_sentiment' and 'news_impact_score'
             # Assuming last row is the target
             if 'news_sentiment' in features.columns:
                 features.iloc[-1, features.columns.get_loc('news_sentiment')] = news_override
             
             if 'news_impact_score' in features.columns and 'adx' in features.columns:
                 # Re-calc impact: sentiment * adx / 100
                 adx = features.iloc[-1]['adx']
                 features.iloc[-1, features.columns.get_loc('news_impact_score')] = news_override * adx / 100.0
        
        
        # Clean Features (Defensive)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0) # Logic handled by fillna below as well, but Inf->NaN first is key.
        
        # 2. Scale
        # Robust Feature Alignment
        if hasattr(self.scaler, 'feature_names_in_'):
            expected_feats = self.scaler.feature_names_in_
            # 1. Add missing (fill 0)
            missing = [f for f in expected_feats if f not in features.columns]
            if missing:
                self.logger.info(f"  Note: Filling missing features: {missing[:5]}...")
                for m in missing:
                    features[m] = 0.0
            
            # 2. Drop extra / Reorder
            features = features[expected_feats]
        else:
            # Fallback legacy count check
            if features.shape[1] != self.scaler.n_features_in_:
                self.logger.warning(f"Warning: Feature mismatch (Got {features.shape[1]}, Expected {self.scaler.n_features_in_}). Slice fallback...")
                if features.shape[1] > self.scaler.n_features_in_:
                    features = features.iloc[:, :self.scaler.n_features_in_]

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
        
        # Check if at least one model is loaded (Relaxed from strict [0])
        if all(m is None for m in self.models[:4]):
             return {"error": "No models loaded."}

        xgb_m, lgb_m, lstm_m, trans_m, meta_m = self.models
        
        # 5. Base Predictions (Optimized)
        # Use predict_on_batch to avoid retracing validation overhead for single batch
        # Fallback to zeros if model missing
        if xgb_m:
            try:
                p_xgb = xgb_m.predict_proba(X_tree)
            except:
                p_xgb = np.zeros((1, 3))
        else:
            p_xgb = np.zeros((1, 3))
            
        if lgb_m:
            try:
                p_lgb = lgb_m.predict(X_tree)
                # Ensure 2D
                if len(p_lgb.shape) == 1:
                    p_lgb = p_lgb.reshape(1, -1)
            except:
                p_lgb = np.zeros((1, 3))
        else:
            p_lgb = np.zeros((1, 3))
        
        p_lstm = lstm_m.predict_on_batch(X_seq) if lstm_m else np.zeros((1,3))
        p_trans = trans_m.predict_on_batch(X_seq) if trans_m else np.zeros((1,3))

        
        # 6. Stacking
        stacked = np.hstack([p_xgb, p_lgb, p_lstm, p_trans])
        
        # 7. Meta Prediction
        if meta_m:
            final_probs = meta_m.predict_proba(stacked)[0]
        else:
            final_probs = np.mean([p_xgb[0], p_lgb[0], p_lstm[0], p_trans[0]], axis=0)
            
        pred_class = int(np.argmax(final_probs))
        confidence = float(final_probs[pred_class])
        
        classes = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
        
        # ... (Date logic remains) ...
        # Calculate Target Date
        last_date = pd.to_datetime(df.iloc[-1]['time']) if 'time' in df.columns else pd.Timestamp.now()
        # ... (Timeframe delta logic skipped for brevity, keeping original flow if not modifying) ...
        # (Assuming the original code for date calc is preserved in context or I re-insert it if I need to replace block)
        # I'll just skip specifically to the additions.
        
        # Re-inserting Date Logic to match context
        if self.timeframe in ["4 Hour", "4h"]:
             target_date = last_date + pd.Timedelta(hours=4)
        elif self.timeframe in ["1 Hour", "1h"]:
             target_date = last_date + pd.Timedelta(hours=1)
        elif self.timeframe in ["15 Min", "15m"]:
             target_date = last_date + pd.Timedelta(minutes=15)
        elif self.timeframe in ["5 Min", "5m"]:
             target_date = last_date + pd.Timedelta(minutes=5)
        else: # Daily
            target_date = last_date + pd.Timedelta(days=1)

        # 8. Run SMC & Microstructure Analysis
        from src.analysis.smc_analyzer import SMCAnalyzer
        from src.analysis.microstructure import MicrostructureAnalyzer
        
        # Define latest price/volatility
        last_close = float(df.iloc[-1]['close'])
        last_atr = float(df_enhanced.iloc[-1]['atr']) if 'atr' in df_enhanced.columns else last_close * 0.01

        # SMC

        smc = SMCAnalyzer(df, timeframe=self.timeframe, symbol=self.symbol)
        smc_levels = smc.get_nearest_structures(last_close)
        
        # Microstructure
        micro = MicrostructureAnalyzer(df)
        has_displacement = micro.detect_displacement().iloc[-1]
        killzone = micro.check_killzone(pd.to_datetime(last_date))
        
        # Trade Levels Calculation
        entry_price = last_close
        sl_buffer = last_atr * 0.1 
        risk_per_share = last_atr * 1.5
        reward_per_share = last_atr * 2.5 # Updated to 2.5R to match new aggressive targets
        
        sl_price = 0.0
        tp_price = 0.0
        
        if pred_class == 1: # Bullish
            supp_ob = smc_levels.get("support_ob")
            if supp_ob:
                proposed_sl = supp_ob['bottom'] - sl_buffer
                if proposed_sl < entry_price:
                    sl_price = proposed_sl
                else:
                    sl_price = entry_price - risk_per_share
            else:
                sl_price = entry_price - risk_per_share
                
            res_ob = smc_levels.get("resistance_ob")
            if res_ob:
                tp_price = res_ob['bottom'] if res_ob['bottom'] > entry_price else entry_price + reward_per_share
            else:
                tp_price = entry_price + reward_per_share
                
        elif pred_class == 0: # Bearish
            res_ob = smc_levels.get("resistance_ob")
            if res_ob:
                proposed_sl = res_ob['top'] + sl_buffer
                if proposed_sl > entry_price:
                    sl_price = proposed_sl
                else:
                    sl_price = entry_price + risk_per_share
            else:
                sl_price = entry_price + risk_per_share
                
            supp_ob = smc_levels.get("support_ob")
            if supp_ob:
                tp_price = supp_ob['top'] if supp_ob['top'] < entry_price else entry_price - reward_per_share
            else:
                tp_price = entry_price - reward_per_share
        
        # 9. Result Construction
        # Extract Context features
        dxy_trend = int(df_enhanced.iloc[-1].get('dxy_trend', 0))
        dxy_corr = float(df_enhanced.iloc[-1].get('gold_dxy_corr', 0.0))
        news_score = float(df_enhanced.iloc[-1].get('news_sentiment', 0.0))
        gold_trend = int(df_enhanced.iloc[-1].get('gold_trend', 0))
        ema_50 = float(df_enhanced.iloc[-1].get('gold_ema_50', 0.0))
        ema_200 = float(df_enhanced.iloc[-1].get('gold_ema_200', 0.0))
        fvg_density = float(df_enhanced.iloc[-1].get('fvg_density', 0.0))
        ob_density = float(df_enhanced.iloc[-1].get('ob_density', 0.0))

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
                "gold_trend": "Bullish" if gold_trend == 1 else "Bearish" if gold_trend == -1 else "Neutral",
                "ema_50": round(ema_50, 2),
                "ema_200": round(ema_200, 2),
                "trend_strength": "Strong" if abs(ema_50 - ema_200) > last_atr else "Weak",
                "dxy_trend": "Bullish" if dxy_trend == 1 else "Bearish" if dxy_trend == -1 else "Neutral",
                "dxy_correlation": round(dxy_corr, 2),
                "news_sentiment": round(news_score, 2),
                "market_density": {
                    "fvg": round(fvg_density, 2),
                    "ob": round(ob_density, 2)
                }
            },
            "smc": {
                 "support_ob": smc_levels.get("support_ob"),
                 "resistance_ob": smc_levels.get("resistance_ob"),
                 "bull_obs_found": smc_levels.get("all_bull_obs"),
                 "bear_obs_found": smc_levels.get("all_bear_obs"),
                 "fvgs": smc_levels.get("fvgs", []),
                 "liquidity": smc_levels.get("liquidity", {})
            },
            "microstructure": {
                "displacement": bool(has_displacement['is_displacement']),
                "displacement_dir": int(has_displacement['displacement_dir']),
                "killzone": killzone
            },
            "technicals": {
                "rsi": df_enhanced['rsi'].iloc[-1],
                "adx": df_enhanced['adx'].iloc[-1],
                "ema50": df_enhanced['ema50'].iloc[-1],
                "close": df_enhanced['close'].iloc[-1]
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
        # Try MT5 First
        df = None
        try:
            from src.data.mt5_interface import MT5Interface
            import MetaTrader5 as mt5
            
            # Map timeframe
            tf_map = {
                "1h": mt5.TIMEFRAME_H1, 
                "4h": mt5.TIMEFRAME_H4, 
                "15m": mt5.TIMEFRAME_M15,
                "5m": mt5.TIMEFRAME_M5,
                "Daily": mt5.TIMEFRAME_D1
            }
            # Handle long/short names
            key = args.timeframe.replace(" Hour", "h").replace(" Min", "m").strip()
            selected_tf = tf_map.get(key, mt5.TIMEFRAME_D1)
            
            mt = MT5Interface()
            if mt.connect():
                print(f"Connected to MT5. Fetching LIVE candles ({args.timeframe})...")
                df = mt.get_historical_data("XAUUSD", timeframe=selected_tf, num_candles=2000)
                mt.shutdown()
        except ImportError:
            pass
        except Exception as e:
            print(f"MT5 Live Fetch failed: {e}")

        if df is None:
            try:
                df = pd.read_csv(path)
            except:
                # Fallback to root data if running from tools/
                df = pd.read_csv(f"../data/{filename}")
            
        pred = Predictor(timeframe=args.timeframe)
        res = pred.predict(df)
        print("\nResult:")
        
        # Helper for numpy serialization
        def np_encoder(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, 
                                  np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return str(obj)
            
        print(json.dumps(res, indent=2, default=np_encoder))
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        # If file missing (e.g. 5m data not fetched yet), alert user
        if "No such file" in str(e):
             print(f"Suggestion: Run 'python tools/train_multiframe.py' to generate models and data for {args.timeframe}.")

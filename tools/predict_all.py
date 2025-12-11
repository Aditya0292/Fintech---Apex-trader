import sys
import os
import pandas as pd
from tabulate import tabulate
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Add root to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.getcwd())

from tools.predict import Predictor
try:
    from tools.fetch_todays_news import get_data, process_events, display_dashboard
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'tools'))
    from fetch_todays_news import get_data, process_events, display_dashboard

# --- CONFIG ---
TARGET_CONFIDENCE = 0.60
RISK_REWARD_RATIO = 1.2 # Target 1.2R for every 1R risk
MAX_RISK_PER_TRADE = 0.02 # Cap at 2% capital

def calculate_kelly(prob_win, risk_reward):
    """
    Calculates Kelly Criterion fraction.
    f = p - (1-p)/b where b = risk_reward
    """
    if prob_win < 0.5:
        return 0.0
    q = 1 - prob_win
    f = prob_win - (q / risk_reward)
    return max(f, 0.0)

def get_position_size(confidence, bankroll=1000):
    """
    Returns suggested risk % based on Fractional Kelly (Half Kelly).
    """
    k = calculate_kelly(confidence, RISK_REWARD_RATIO)
    fractional_k = k * 0.5 # Half Kelly for safety
    
    # Cap risk
    risk_pct = min(fractional_k, MAX_RISK_PER_TRADE)
    return risk_pct

def run_multi_timeframe_prediction():
    print("Generating Multi-Timeframe Analysis...")

    # 0. Auto-Fetch News & CSM
    print("\n[AUTO] Fetching Live Market Intelligence...")
    
    # News
    print("  > News (ForexFactory)...")
    try:
        raw = get_data()
        if raw:
            clean = process_events(raw)
            if clean:
                df_news = pd.DataFrame(clean)
                df_news = df_news.sort_values(by="DateTime")
                display_dashboard(df_news)
            else:
                print("    No events to display.")
        else:
             print("    News Fetch Failed (Check logs).")

    except Exception as e:
        print(f"    [Warning] News fetch failed: {e}")

    # CSM
    print("  > USD Currency Strength (CSM)...")
    usd_strength = 5.0 # Default
    try:
        from src.data.fetch_csm import CSMScraper
        scraper = CSMScraper()
        usd_strength = scraper.fetch_usd_strength()
    except Exception as e:
        print(f"    [Warning] CSM fetch failed: {e}")

    print("=" * 60)
    print(f"MARKET CONTEXT | USD Strength: {usd_strength}/10")
    print("=" * 60)
    
    # User Request: 4H and 1H
    timeframes = ["4 Hour", "1 Hour"]
    results = []
    
    # Model Stats (Hardcoded from verified backtests)
    stats_map = {
        "4 Hour": {"WR": "79%", "Ret": "+20%"}, # Confirmed
        "1 Hour": {"WR": "65%", "Ret": "TBD"},  # Estimated
    }
    
    tf_file_map = {
        "4 Hour": "XAUUSD_4h.csv",
        "1 Hour": "XAUUSD_1h.csv",
    }
    
    for tf in timeframes:
        try:
            print(f"Analyzing {tf}...")
            filename = tf_file_map.get(tf)
            path = f"data/{filename}"
            if not os.path.exists(path):
                path = f"../data/{filename}" 
            
            if os.path.exists(path):
                df = pd.read_csv(path)
                predictor = Predictor(timeframe=tf)
                result = predictor.predict(df)
                
                # Get Stats
                stat = stats_map.get(tf, {"WR": "-", "Ret": "-"})
                
                if result:
                    if "error" in result:
                         results.append({"Timeframe": tf, "Signal": "ERROR", "Conf": "-", "Kelly Risk": "-", "WR (Est)": "-", "Msg": result["error"]})
                    else:
                        conf_val = result['confidence']
                        signal_str = result['prediction'].upper()
                        entry_val = result['trade_levels']['entry']
                        tp_val = result['trade_levels']['tp']
                        sl_val = result['trade_levels']['sl']
                        
                        # Trend Context
                        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
                        price = df['close'].iloc[-1]
                        trend_str = "Uptrend" if price > ema_50 else "Downtrend"
                        
                        results.append({
                            "Timeframe": tf,
                            "Signal": signal_str,
                            "Conf": f"{result['confidence']:.1%}",
                            "WR (Est)": stat['WR'],
                            "Entry": entry_val,
                            "TP": tp_val,
                            "SL": sl_val,
                            "Context": f"DXY: {result['context']['dxy_trend']} | {trend_str}"
                        })
                else:
                     results.append({"Timeframe": tf, "Signal": "ERROR", "Conf": "-", "Msg": "No Result"})
            else:
                 results.append({"Timeframe": tf, "Signal": "NO DATA", "Conf": "-", "Msg": "File missing"})

        except Exception as e:
            print(f"Error analyzing {tf}: {e}")
            results.append({"Timeframe": tf, "Signal": "ERROR", "Confidence": "-", "Kelly Risk": "-", "WR (Est)": "-", "Action": "-", "TP": "-", "SL": "-", "Context": str(e)})

    print("\n" + "=" * 110) # Widen line
    print(f"\nAPEX TRADE AI - DUAL TIMEFRAME CONSENSUS ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("=" * 110)
    
    mapped_results = pd.DataFrame(results)
    print(tabulate(mapped_results, headers=["Timeframe", "Signal", "Confidence", "Kelly Risk", "WR (Est)", "Action", "Entry", "TP", "SL", "Context"], tablefmt="grid"))
    print(f"Risk Logic: Fractional Kelly (Max {MAX_RISK_PER_TRADE:.1%}) | Target Win Rate > 60%")
    
    print("\nAnalysis:")
    print(f"USD CSM is {usd_strength}. " + ("Strong USD -> Bearish Gold." if usd_strength > 6 else "Weak USD -> Bullish Gold." if usd_strength < 4 else "Neutral USD."))
    signals = mapped_results['Signal'].unique()
    if len(signals) > 1:
        print("[CHECK] DIVERGENCE DETECTED.")
    else:
        print("[INFO] CONVERGENCE: Strong Trend.")
        
    # --- SWING MODEL PREDICTION (Extra) ---
    print("\n" + "=" * 80)
    print("SWING TRADER (5-Day Hold) | Model: SMC+ICT (Acc: 57%)")
    print("=" * 80)
    try:
        import joblib
        import xgboost as xgb
        import numpy as np
        from src.features.feature_pipeline import FeatureEngineering
        
        swing_model_path = "saved_models/swing_model_advanced_xgb.pkl"
        if os.path.exists(swing_model_path):
            # Load Daily Data
            df_swing = pd.read_csv("data/XAUUSD_history.csv")
            fe = FeatureEngineering()
            # Suppress prints
            import contextlib
            with contextlib.redirect_stdout(None):
                df_full, _ = fe.build_features(df_swing)
            
            # Prepare Input
            last_row = df_full.iloc[[-1]] # DataFrame format
            
            # Select Numeric Features (Same logic as training)
            exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume', 
                    'target', 'return_5d', 'future_close', 
                    'R2_traditional', 'S2_traditional', 'R2_fibonacci', 'S2_fibonacci',
                    'prediction_target', 'target_class']
            
            feature_cols = [c for c in df_full.columns if c not in exclude_cols]
            X_swing = last_row[feature_cols].select_dtypes(include=[np.number])
            
            # Predict
         # --- CONFIG ---
            swing_model = joblib.load(swing_model_path)
            swing_pred = swing_model.predict(X_swing)[0]
            swing_prob = swing_model.predict_proba(X_swing)[0][swing_pred]
            dir_str = "BULLISH" if swing_pred == 1 else "BEARISH"
            
            # Position Size
            risk_pct = get_position_size(swing_prob, bankroll=1000)
            
            print(f"Signal: {dir_str}")
            print(f"Conf:   {swing_prob:.1%}")
            print(f"Risk:   {risk_pct:.1%} (Kelly)")
            
            # Explain with Top Features
            # We know from training: in_discount, inside_ob_bull
            is_discount = last_row['in_discount'].values[0]
            in_ob_bull = last_row['inside_ob_bull'].values[0]
            
            reasons = []
            if is_discount: reasons.append("Price in Discount Zone")
            if in_ob_bull: reasons.append("Inside Bullish OB")
            if not reasons: reasons.append("Price Action / Range Logic")
            
            print(f"Logic:  {', '.join(reasons)}")
            
        else:
            print("Swing Model not found (Train it first).")
            
    except Exception as e:
        print(f"Swing Prediction Error: {e}")

if __name__ == "__main__":
    run_multi_timeframe_prediction()

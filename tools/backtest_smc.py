import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import os

# Ensure src is in path
sys.path.append(".") 

from src.analysis.smc_analyzer import SMCAnalyzer
from src.features.feature_pipeline import FeatureEngineering # Import FE
from tools.evaluate import load_all_models, load_data

class SMCBacktester:
    def __init__(self, timeframe="1 Hour", initial_capital=10000):
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.commission = 0.001 # 0.1% per trade
        self.results = []
        
    def run(self, candles_to_test=0, threshold=0.50, confluence=False):
        print(f"--- Starting SMC Backtest ({self.timeframe}) ---")
        print(f"    Mode: {'Full History' if candles_to_test==0 else f'Last {candles_to_test}'} | Threshold: {threshold} | Confluence: {confluence}")
        
        # 1. Load Data
        print("  Loading Data & Models...")
        # Map timeframe to file suffix
        suffix_map = {
            "4 Hour": "_4h", "4h": "_4h",
            "1 Hour": "_1h", "1h": "_1h",
            "15 min": "_15m", "15m": "_15m",
            "30 min": "_30m", "30m": "_30m"
        }
        suffix = suffix_map.get(self.timeframe, "_1h")
        
        try:
            df_path = f"data/XAUUSD{suffix}.csv"
            df = pd.read_csv(df_path)
            # Load Pre-processed features for the model
            X, y, X_tree = load_data(suffix=suffix)
            
            # Load Feature DF for Confluence (Indicators)
            try:
                feat_path = f"data/features_enhanced{suffix}.csv"
                df_feats = pd.read_csv(feat_path)
                print("  Loaded cached features.")
            except:
                print("  Generating features on the fly (file not found)...")
                fe = FeatureEngineering()
                df_feats, _ = fe.build_features(df)
            
            # Load Models
            input_shape = (X.shape[1], X.shape[2])
            models = load_all_models(input_shape, suffix=suffix)
            xgb_m, lgb_m, lstm_m, trans_m, meta_m = models
            
        except Exception as e:
            print(f"Data Load Error: {e}")
            return
            
        # Align Data
        window_size = 50
        start_idx = len(df) - len(X)
        
        # Test Loop
        if candles_to_test == 0:
            test_start = 0
        else:
            test_start = len(X) - candles_to_test
            if test_start < 0: test_start = 0
        
        print(f"  Simulating {len(X) - test_start} trades...")
        
        equity = [self.initial_capital]
        trades = []
        
        # Pre-calculate Model Sigs
        print("  Pre-calculating Model Signals...", end="")
        p_xgb = xgb_m.predict_proba(X_tree[test_start:])
        p_lgb = lgb_m.predict(X_tree[test_start:])
        p_lstm = lstm_m.predict(X[test_start:], verbose=0)
        p_trans = trans_m.predict(X[test_start:], verbose=0)
        stacked = np.hstack([p_xgb, p_lgb, p_lstm, p_trans])
        probs = meta_m.predict_proba(stacked)
        preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        print(" Done.")
        
        
        for i in range(len(preds)):
            model_idx = test_start + i
            df_idx = start_idx + model_idx
            
            current_date = df.iloc[df_idx]['time']
            close_price = float(df.iloc[df_idx]['close'])
            
            pred_class = preds[i]
            confidence = max_probs[i]
            
            if confidence < threshold: continue
            if pred_class == 2: continue
            
            # --- Confluence Check (Optional) ---
            if confluence and not df_feats.empty:
                # Need to map to feature index
                # X was loaded from features_enhanced, so model_idx aligns with X, 
                # but X is a numpy array. df_feats corresponds to X rows (roughly).
                # Actually, load_data takes the LAST len(X) rows of df_feats.
                # So model_idx in X corresponds to (len(df_feats) - len(X)) + model_idx
                feat_idx = (len(df_feats) - len(X)) + model_idx
                
                if feat_idx < len(df_feats):
                    row = df_feats.iloc[feat_idx]
                    
                    # 1. Trend Filter (EMA 200)
                    # Assuming 'gold_ema_200' or similar exists. Standard is 'ema_200' or 'gold_ema_200'
                    # Let's try flexible lookup
                    ema_200 = row.get('gold_ema_200') or row.get('ema_200') or row.get('EMA_200')
                    
                    # 2. Momentum Filter (RSI)
                    rsi = row.get('rsi') or row.get('RSI')
                    
                    if ema_200 and rsi:
                        if pred_class == 1: # Buy
                            # Trend: Price > EMA 200
                            if close_price < ema_200: continue
                            # Momentum: RSI < 70 (Not overbought)
                            if rsi > 70: continue
                        else: # Sell
                            # Trend: Price < EMA 200
                            if close_price > ema_200: continue
                            # Momentum: RSI > 30 (Not oversold)
                            if rsi < 30: continue
            # -----------------------------------
            
            # 2. SMC Analysis
            lookback_smc = 500
            smc_start = max(0, df_idx - lookback_smc)
            df_slice = df.iloc[smc_start : df_idx + 1].copy()
            
            smc = SMCAnalyzer(df_slice, timeframe=self.timeframe)
            smc_levels = smc.get_nearest_structures(close_price)
            
            # 3. Levels
            atr = 2.0
            if 'atr' in df.columns: atr = float(df.iloc[df_idx]['atr'])
            elif 'high' in df.columns: atr = float(df.iloc[df_idx]['high']) - float(df.iloc[df_idx]['low'])
            
            entry = close_price
            sl_buffer = atr * 0.1
            
            sl = 0
            tp = 0
            
            if pred_class == 1: # Bullish
                type_str = "BUY"
                supp_ob = smc_levels.get("support_ob")
                if supp_ob:
                     proposed = supp_ob['bottom'] - sl_buffer
                     sl = proposed if proposed < entry else entry - (atr * 0.8)  # Tighter SL for better risk mgmt
                else:
                    sl = entry - (atr * 0.8)  # Reduced from 1.5 to 0.8
                
                res_ob = smc_levels.get("resistance_ob")
                if res_ob:
                    proposed = res_ob['bottom']
                    tp = proposed if proposed > entry else entry + (atr * 2.5)  # Wider TP for better R:R
                else:
                    tp = entry + (atr * 2.5)  # Increased from 2.0 to 2.5
                    
            else: # Bearish
                type_str = "SELL"
                res_ob = smc_levels.get("resistance_ob")
                if res_ob:
                    proposed = res_ob['top'] + sl_buffer
                    sl = proposed if proposed > entry else entry + (atr * 0.8)  # Tighter SL
                else:
                    sl = entry + (atr * 0.8)  # Reduced from 1.5 to 0.8
                    
                supp_ob = smc_levels.get("support_ob")
                if supp_ob:
                    proposed = supp_ob['top']
                    tp = proposed if proposed < entry else entry - (atr * 2.5)  # Wider TP
                else:
                    tp = entry - (atr * 2.0)

            # 4. Check Outcome
            outcome = "OPEN"
            exit_price = entry
            
            look_forward = 48 
            
            for f in range(1, look_forward):
                future_idx = df_idx + f
                if future_idx >= len(df): break
                
                high = float(df.iloc[future_idx]['high'])
                low = float(df.iloc[future_idx]['low'])
                
                if type_str == "BUY":
                    if low <= sl: 
                        outcome = "SL"
                        exit_price = sl
                        break
                    if high >= tp: 
                        outcome = "TP"
                        exit_price = tp
                        break
                else: 
                    if high >= sl: 
                        outcome = "SL"
                        exit_price = sl
                        break
                    if low <= tp:
                        outcome = "TP"
                        exit_price = tp
                        break
            
            if outcome == "OPEN":
                final_close = float(df.iloc[min(df_idx+look_forward, len(df)-1)]['close'])
                exit_price = final_close
                outcome = "TIMEOUT"
                
            if type_str == "BUY":
                raw_ret = (exit_price - entry) / entry
            else:
                raw_ret = (entry - exit_price) / entry
                
            trades.append({
                "date": current_date,
                "type": type_str,
                "confidence": confidence, # Added this
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit": exit_price,
                "outcome": outcome,
                "ret": raw_ret,
                "smc_sl": bool(smc_levels.get("support_ob" if type_str=="BUY" else "resistance_ob"))
            })
            
        if not trades:
            print("No trades found.")
            return

        # Build Compounded Equity Curve
        curr_cap = self.initial_capital
        equity_curve = [curr_cap]
        dates = [df.iloc[len(df)-candles_to_test if candles_to_test else 0]['time']] 
        
        # Sort trades by date just in case
        # (They are appended in order, so should be fine)
        
        for t in trades:
            # Simple compounding: New = Old * (1 + ret)
            # Modeling 0.1% commission per trade
            net_ret = t['ret'] - self.commission
            curr_cap = curr_cap * (1 + net_ret)
            equity_curve.append(curr_cap)
            dates.append(t['date'])
            
            t['net_ret'] = net_ret
            t['equity'] = curr_cap

        df_res = pd.DataFrame(trades)
        wins = df_res[df_res['ret'] > 0]
        losses = df_res[df_res['ret'] <= 0]
        
        win_rate = len(wins) / len(df_res)
        avg_win = wins['ret'].mean() if not wins.empty else 0
        avg_loss = losses['ret'].mean() if not losses.empty else 0
        # Fix potential zero division
        loss_abs_sum = abs(losses['ret'].sum())
        profit_factor = abs(wins['ret'].sum() / loss_abs_sum) if loss_abs_sum > 0 else 999.0
        
        smc_usage = df_res['smc_sl'].mean()
        
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        start_date = df_res['date'].iloc[0]
        end_date = df_res['date'].iloc[-1]
        
        # Calculate approximate days (basic str parsing or just printing)
        # Verify format (it's likely string or datetime)
        try:
             s_dt = pd.to_datetime(start_date)
             e_dt = pd.to_datetime(end_date)
             duration = (e_dt - s_dt).days
        except:
             duration = "N/A"

        print("\n" + "="*60)
        print(f"SMC BACKTEST SUMMARY ({self.timeframe}) | {len(df_res)} Trades")
        print("="*60)
        print(f"Period:          {start_date}  <--->  {end_date}")
        print(f"Duration:        {duration} Days")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"Final Equity:    ${final_equity:.2f} ({total_return:+.1%})")
        print(f"Win Rate:        {win_rate:.1%}")
        print(f"Profit Factor:   {profit_factor:.2f}")
        print(f"Avg Win:         {avg_win:.2%}")
        print(f"Avg Loss:        {avg_loss:.2%}")
        print(f"SMC Levels Used: {smc_usage:.1%} of trades")
        print("="*60)
        
        print(df_res[['date', 'type', 'confidence', 'outcome', 'ret', 'equity']].tail(10).to_string())
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label=f"Equity (Start ${self.initial_capital})")
        plt.title(f"SMC Strategy Performance ({self.timeframe}) | PF: {profit_factor:.2f}")
        plt.xlabel("Trades")
        plt.ylabel("Capital ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        out_path = "data/equity_curve_smc.png"
        plt.savefig(out_path)
        print(f"\n[INFO] Equity curve saved to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="1 Hour")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to test. 0 for full.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--capital", type=float, default=100.0, help="Initial Capital")
    parser.add_argument("--confluence", action="store_true", help="Enable strict technical confluence (EMA, RSI)")
    args = parser.parse_args()
    
    bt = SMCBacktester(timeframe=args.timeframe, initial_capital=args.capital)
    bt.run(candles_to_test=args.limit, threshold=args.threshold, confluence=args.confluence)

"""
APEX TRADE AI - Backtesting Engine
==================================
Simulates trading based on Ensemble Predictions.
- Strategy: Enter if Confidence >= Threshold
- Direction: Buy if Bullish, Sell if Bearish
- PnL: Uses 'y_reg' (Next Day Open-to-Close Return)
- Costs: Commission + Slippage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score

# Reuse evaluation loader
from evaluate import load_all_models, load_data, config

class Backtester:
    def __init__(self, initial_capital=10000, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission # 0.1%
        self.slippage = slippage # 0.05%
        self.results = {}
        
    def run_backtest(self, threshold=0.60):
        print(f"Running Backtest with Confidence Threshold: {threshold}")
        
        # Load Data & Models
        # Ideally we should cache this if running multiple times but for script it's fine
        X, y, X_tree = load_data()
        
        # Split (Same as evaluation)
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        X_tree_test = X_tree[split_idx:]
        y_test = y[split_idx:]
        
        # Load Returns (y_reg)
        y_reg = np.load("data/y_reg.npy")
        y_reg_test = y_reg[split_idx:]
        
        # Load Models
        input_shape = (X_test.shape[1], X_test.shape[2])
        xgb_m, lgb_m, lstm_m, trans_m, meta_m = load_all_models(input_shape)
        
        # Predict
        print("  Generating predictions...")
        p_xgb = xgb_m.predict_proba(X_tree_test)
        p_lgb = lgb_m.predict(X_tree_test)
        p_lstm = lstm_m.predict(X_test, verbose=0)
        p_trans = trans_m.predict(X_test, verbose=0)
        
        stacked = np.hstack([p_xgb, p_lgb, p_lstm, p_trans])
        probs = meta_m.predict_proba(stacked)
        preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        
        # Simulation
        equity = [self.initial_capital]
        trades = []
        
        for i in range(len(preds)):
            pred_class = preds[i]
            conf = max_probs[i]
            actual_ret = y_reg_test[i]
            
            # Position logic
            position = 0
            if conf >= threshold:
                if pred_class == 1: # Bull
                    position = 1
                elif pred_class == 0: # Bear
                    position = -1
            
            # PnL Calculation
            if position != 0:
                # Gross return
                gross_ret = position * actual_ret
                
                # Net return (Gross - Comm - Slippage)
                net_ret = gross_ret - self.commission - self.slippage
                
                # Update Capital
                current_cap = equity[-1]
                pnl = current_cap * net_ret 
                allocation = 1.0 
                
                trade_pnl = current_cap * allocation * net_ret
                new_cap = current_cap + trade_pnl
                
                equity.append(new_cap)
                trades.append({
                    "step": i,
                    "type": "Long" if position==1 else "Short",
                    "conf": conf,
                    "ret": actual_ret,
                    "net_ret": net_ret,
                    "pnl": trade_pnl
                })
            else:
                equity.append(equity[-1])
        
        # Metrics
        equity = np.array(equity)
        returns = pd.Series(equity).pct_change().dropna()
        
        total_return = (equity[-1] - equity[0]) / equity[0]
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
        max_dd = drawdown.min()
        
        win_trades = [t for t in trades if t["pnl"] > 0]
        try:
            win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
        except:
            win_rate = 0
        
        print(f"  Backtest Result (Thresh={threshold}):")
        print(f"    Trades: {len(trades)}")
        print(f"    Win Rate: {win_rate:.2%}")
        print(f"    Total Return: {total_return:.2%}")
        print(f"    Sharpe Ratio: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_dd:.2%}")
        
        # Save results
        self.results = {
            "equity": equity.tolist(),
            "trades": trades,
            "metrics": {
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "trade_count": len(trades)
            }
        }
        
        with open("data/backtest_results.json", "w") as f:
            json.dump(self.results, f)
            
        # Plot Equity
        plt.figure(figsize=(10, 6))
        plt.plot(equity)
        plt.title((f"Equity Curve (Thresh={threshold})"))
        plt.xlabel("Days")
        plt.ylabel("Capital")
        plt.grid(True)
        plt.savefig("images/equity_curve.png")
        print("  Equity curve saved.")

    def run_comparative_analysis(self, thresholds=[0.50, 0.55, 0.60, 0.65, 0.70], full_data=False, suffix=""):
        suffix_clean = "_" + suffix.lstrip("_") if suffix else ""
        mode_str = f"FULL DATASET ({suffix})" if full_data else f"TEST SET (20% {suffix})"
        print(f"\n--- Running Comparative Analysis on {mode_str} (Start Capital: ${self.initial_capital}) ---")
        if full_data:
            print("WARNING: This includes training data. Results will be optimistic (Overfitting check).")
        
        # Load Data Once
        X, y, X_tree = load_data(suffix=suffix_clean)
        y_reg = np.load(f"data/y_reg{suffix_clean}.npy")
        
        # Load Feature DataFrame to check News Context (if exists)
        try:
            df_feats = pd.read_csv(f"data/features_enhanced{suffix_clean}.csv")
        except:
            print(f"Warning: Could not load features_enhanced{suffix_clean}.csv. Skipping News Split Analysis.")
            df_feats = None

        win_size = X.shape[1] # 50

        if full_data:
            X_test = X
            X_tree_test = X_tree
            y_reg_test = y_reg
            start_row_idx = win_size - 1
        else:
            split_idx = int(len(X) * 0.8)
            X_test = X[split_idx:]
            X_tree_test = X_tree[split_idx:]
            y_reg_test = y_reg[split_idx:]
            start_row_idx = (win_size - 1) + split_idx
        
        # Load Models
        input_shape = (X_test.shape[1], X_test.shape[2])
        xgb_m, lgb_m, lstm_m, trans_m, meta_m = load_all_models(input_shape, suffix=suffix_clean)
        
        # Pre-calculate probabilities once
        print("  Generating predictions...")
        p_xgb = xgb_m.predict_proba(X_tree_test)
        p_lgb = lgb_m.predict(X_tree_test)
        p_lstm = lstm_m.predict(X_test, verbose=0)
        p_trans = trans_m.predict(X_test, verbose=0)
        stacked = np.hstack([p_xgb, p_lgb, p_lstm, p_trans])
        probs = meta_m.predict_proba(stacked)
        preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        
        results_summary = []
        
        # We will collect stats for split analysis on the recommendation threshold (0.60)
        split_stats = {
            "News": {"trades": 0, "wins": 0, "return": 0.0},
            "Technical": {"trades": 0, "wins": 0, "return": 0.0}
        }
        
        plt.figure(figsize=(12, 8))
        
        for thresh in thresholds:
            equity = [self.initial_capital]
            trades_count = 0
            wins = 0
            
            for i in range(len(preds)):
                pred_class = preds[i]
                conf = max_probs[i]
                actual_ret = y_reg_test[i]
                
                # Check News Context
                is_news_day = False
                if df_feats is not None:
                    df_idx = start_row_idx + i
                    if df_idx < len(df_feats):
                        news_val = df_feats.iloc[df_idx].get('news_impact', 0)
                        is_news_day = abs(news_val) > 0 # Any sentiment is news
                
                position = 0
                if conf >= thresh:
                    if pred_class == 1: position = 1
                    elif pred_class == 0: position = -1
                
                if position != 0:
                    trades_count += 1
                    net_ret = (position * actual_ret) - self.commission - self.slippage
                    # 100% Allocation
                    equity.append(equity[-1] * (1 + net_ret))
                    if net_ret > 0: wins += 1
                    
                    # Split Stats (Only for 0.60 thresh as reference)
                    if abs(thresh - 0.60) < 0.01:
                        cat = "News" if is_news_day else "Technical"
                        split_stats[cat]["trades"] += 1
                        if net_ret > 0: split_stats[cat]["wins"] += 1
                        split_stats[cat]["return"] += net_ret
                        
                else:
                    equity.append(equity[-1])
            
            final_pl = (equity[-1] - self.initial_capital) / self.initial_capital
            win_rate = wins / trades_count if trades_count > 0 else 0
            label = f"Thresh {thresh:.2f} (WR: {win_rate:.0%})"
            plt.plot(equity, label=label)
            
            results_summary.append({
                "Threshold": thresh,
                "Return": final_pl,
                "Trades": trades_count,
                "WinRate": win_rate,
                "Final Equity": equity[-1]
            })

        plt.title(f"Equity Curve - {mode_str}")
        plt.xlabel("Periods") # Days or Candles
        plt.ylabel("Capital ($)")
        plt.legend()
        plt.grid(True)
        fname = f"images/equity_comparison_full{suffix_clean}.png" if full_data else f"images/equity_comparison{suffix_clean}.png"
        plt.savefig(fname)
        print(f"  Comparison plot saved to '{fname}'.")
        
        # Print Table
        print(f"\n{mode_str} Summary:")
        print(f"{'Threshold':<10} | {'Return':<10} | {'WinRate':<10} | {'Trades':<8} | {'Final Equity':<15}")
        print("-" * 65)
        for r in results_summary:
            print(f"{r['Threshold']:<10.2f} | {r['Return']:<10.1%} | {r['WinRate']:<10.1%} | {r['Trades']:<8} | ${r['Final Equity']:<15.2f}")

        # Print Split Analysis (Threshold 0.60)
        print(f"\n--- Strategy Breakdown (Threshold 0.60) ---")
        print(f"{'Category':<15} | {'Trades':<8} | {'WinRate':<10} | {'AvgReturn':<10}")
        print("-" * 50)
        
        # News
        n_t = split_stats["News"]["trades"]
        n_wr = split_stats["News"]["wins"] / n_t if n_t > 0 else 0
        n_ret = split_stats["News"]["return"] 
        n_avg_ret = n_ret / n_t if n_t > 0 else 0
        print(f"{'News Days':<15} | {n_t:<8} | {n_wr:<10.1%} | {n_avg_ret:<10.2%}")
        
        # Technical
        t_t = split_stats["Technical"]["trades"]
        t_wr = split_stats["Technical"]["wins"] / t_t if t_t > 0 else 0
        t_ret = split_stats["Technical"]["return"]
        t_avg_ret = t_ret / t_t if t_t > 0 else 0
        print(f"{'Technical Only':<15} | {t_t:<8} | {t_wr:<10.1%} | {t_avg_ret:<10.2%}")
        print("(Note: 'News Days' = Days with significant News Sentiment != 0)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="Daily", help="Daily, 4h, 15m")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "full"], help="Backtest mode: 'test' (20% Out-of-Sample) or 'full' (Entire History)")
    args = parser.parse_args()
    
    suffix_map = {
        "Daily": "",
        "4 Hour": "_4h", "4h": "_4h",
        "1 Hour": "_1h", "1h": "_1h", "1H": "_1h",
        "15 Min": "_15m", "15m": "_15m"
    }
    suffix = suffix_map.get(args.timeframe, "")
    
    bt = Backtester(initial_capital=1000) 
    
    full_data = (args.mode == "full")
    
    print(f"Starting Multi-Timeframe Backtest for: {args.timeframe} (Suffix: '{suffix}') | Mode: {args.mode.upper()}")
    try:
        # Use lower thresholds for 15M investigation
        threshs = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65] if "15" in args.timeframe else [0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
        bt.run_comparative_analysis(thresholds=threshs, full_data=full_data, suffix=suffix)
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()

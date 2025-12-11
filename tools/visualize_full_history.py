"""
APEX TRADE AI - Full History Visualization
==========================================
Plots the entire history of Gold prices with AI predictions overlaid.
Includes News Sentiment track (if available).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from tools.evaluate import load_all_models, load_data

def visualize_full_history(threshold=0.60):
    print("Loading Data & Models for Visualization...")
    
    # 1. Load Feature Data (X) and Raw Price Data
    X, y, X_tree = load_data()
    
    # We need the original timestamps and prices to plot
    # features_enhanced.csv corresponds to the inputs
    df = pd.read_csv("data/features_enhanced.csv")
    
    # Alignment:
    # X[i] is the window ending at row i + window_size - 1
    # It predicts the target at row i + window_size
    # We want to plot the Price at the TARGET Time.
    
    window_size = 50
    # The first target is at index `window_size` of the dataframe
    # The length of X is len(df) - window_size
    
    # Slice DF to match X predictions (Target Times)
    # We want rows [window_size : window_size + len(X)]
    df_aligned = df.iloc[window_size : window_size + len(X)].reset_index(drop=True)
    
    # 2. Load Models
    input_shape = (X.shape[1], X.shape[2])
    xgb_m, lgb_m, lstm_m, trans_m, meta_m = load_all_models(input_shape)
    
    # 3. Predict Full History
    print("Generating predictions for full history (this may take a moment)...")
    p_xgb = xgb_m.predict_proba(X_tree)
    p_lgb = lgb_m.predict(X_tree)
    p_lstm = lstm_m.predict(X, verbose=0)
    p_trans = trans_m.predict(X, verbose=0)
    
    stacked = np.hstack([p_xgb, p_lgb, p_lstm, p_trans])
    probs = meta_m.predict_proba(stacked)
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    
    # 4. Prepare Plot Data
    dates = pd.to_datetime(df_aligned['time'])
    prices = df_aligned['close']
    sentiment = df_aligned.get('news_impact', np.zeros(len(df_aligned)))
    
    # Signals
    buy_signals = []
    sell_signals = []
    
    for i in range(len(preds)):
        if max_probs[i] >= threshold:
            if preds[i] == 1: # Bullish
                buy_signals.append((dates[i], prices[i]))
            elif preds[i] == 0: # Bearish
                sell_signals.append((dates[i], prices[i]))
                
    # 5. Plotting
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Top Panel: Price & Signals
    ax1.plot(dates, prices, label='Gold Price (Close)', color='black', alpha=0.6, linewidth=1)
    
    if buy_signals:
        bx, by = zip(*buy_signals)
        ax1.scatter(bx, by, marker='^', color='lime', s=30, label='AI Buy Signal', zorder=5)
        
    if sell_signals:
        sx, sy = zip(*sell_signals)
        ax1.scatter(sx, sy, marker='v', color='red', s=30, label='AI Sell Signal', zorder=5)
        
    ax1.set_title(f"XAUUSD Full History: Actual vs Predicted Signals (Thresh={threshold})")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom Panel: News Sentiment (and Confidence?)
    # Let's plot Confidence + Sentiment overlaid
    ax2.plot(dates, sentiment, label='News Sentiment', color='blue', alpha=0.8)
    
    # Also plot Model Confidence (faded)
    ax2.fill_between(dates, 0, max_probs, color='gray', alpha=0.2, label='AI Confidence')
    ax2.axhline(threshold, color='orange', linestyle='--', label='Trade Threshold')
    
    ax2.set_title("News Sentiment & AI Confidence")
    ax2.set_ylabel("Score")
    ax2.set_ylim(-1.0, 1.1) # Sentiment is -1 to 1, Conf is 0 to 1
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "images/full_history_prediction.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    visualize_full_history(threshold=0.60)

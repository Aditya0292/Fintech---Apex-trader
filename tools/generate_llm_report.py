
import sys
import os
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.getcwd())

from tools.predict import Predictor

def generate_report():
    timeframes = ["Daily", "4 Hour", "15 Min"]
    
    tf_file_map = {
        "Daily": "XAUUSD_history.csv",
        "4 Hour": "XAUUSD_4h.csv",
        "15 Min": "XAUUSD_15m.csv"
    }

    report = []
    report.append("# XAUUSD Market Analysis Report")
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("Prompt for ChatGPT: 'Analyze the following Gold market data and validate the AI prediction.'")
    report.append("-" * 60)
    
    for tf in timeframes:
        print(f"Processing {tf}...")
        try:
            filename = tf_file_map.get(tf)
            path = f"data/{filename}"
            
            if os.path.exists(path):
                df = pd.read_csv(path)
                
                # Initialize Predictor just to use its FE and Models
                predictor = Predictor(timeframe=tf)
                
                # 1. Get Prediction
                pred_result = predictor.predict(df)
                
                # 2. Get Technical Context (Last Row of Enhanced DF)
                # Predictor.predict creates df_enhanced internally but doesn't return it.
                # We interpret it again easily.
                df_enhanced, _ = predictor.fe.build_features(df)
                last_row = df_enhanced.iloc[-1]
                
                # Extract Key Metrics
                close_price = last_row.get('close', 0)
                rsi = last_row.get('rsi', 0)
                atr = last_row.get('atr', 0)
                adx = last_row.get('adx', 0)
                dxy_trend = last_row.get('dxy_trend', 0)
                news_score = last_row.get('news_sentiment', 0)
                
                # Pivot / Structure (Now with R2/S2)
                pivot_p = last_row.get('P_traditional', 0)
                r1 = last_row.get('R1_traditional', 0)
                r2 = last_row.get('R2_traditional', 0)
                s1 = last_row.get('S1_traditional', 0)
                s2 = last_row.get('S2_traditional', 0)
                
                # Report Block
                report.append(f"\n## Timeframe: {tf}")
                report.append(f"**Price**: {close_price:.2f}")
                report.append(f"**AI Prediction**: {pred_result['prediction']} ({pred_result['confidence']:.1%})")
                
                report.append("\n### Technical Indicators")
                report.append(f"- **RSI (14)**: {rsi:.2f}")
                report.append(f"- **ADX (14)**: {adx:.2f}")
                report.append(f"- **ATR (14)**: {atr:.4f}")
                
                 # Structure
                report.append("\n### Support & Resistance (Traditional)")
                report.append(f"- R2:    {r2:.2f}")
                report.append(f"- R1:    {r1:.2f}")
                report.append(f"- Pivot: {pivot_p:.2f}")
                report.append(f"- S1:    {s1:.2f}")
                report.append(f"- S2:    {s2:.2f}")
                
                report.append("\n### Context")
                report.append(f"- **DXY Trend**: {'Bullish' if dxy_trend==1 else 'Bearish' if dxy_trend==-1 else 'Neutral'}")
                report.append(f"- **News Impact**: {news_score:.2f} (Decayed Score)")
                
                # Recommendation (AI Trade Levels)
                report.append("\n### AI Trade Plan")
                levels = pred_result['trade_levels']
                report.append(f"- Entry: {levels['entry']}")
                report.append(f"- TP: {levels['tp']}")
                report.append(f"- SL: {levels['sl']}")
                
                report.append("-" * 30)
                
            else:
                 report.append(f"\n## Timeframe: {tf} - DATA MISSING ({path})")
                 
        except Exception as e:
            report.append(f"\n## Timeframe: {tf} - ERROR: {str(e)}")

    # Save to file
    out_path = "llm_market_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"Report saved to {out_path}")
    print("-" * 60)
    print("\n".join(report))

if __name__ == "__main__":
    generate_report()

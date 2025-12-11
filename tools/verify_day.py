
import pandas as pd
import sys
import os
from tabulate import tabulate

sys.path.append(os.getcwd())
from tools.predict import Predictor

def verify_day():
    print("Loading Data...")
    path = "data/XAUUSD_4h.csv"
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    
    # Identify rows for 2025-12-09
    target_date = "2025-12-09"
    day_rows = df[df['time'].dt.strftime('%Y-%m-%d') == target_date]
    
    if len(day_rows) == 0:
        print(f"No data found for {target_date}")
        return

    print(f"Found {len(day_rows)} candles for {target_date}")
    
    predictor = Predictor(timeframe="4 Hour")
    
    results = []
    
    # For each row in that day, we want to see if the model predicted it correctly
    # The model would have predicted it using data UP TO that row (exclusive)
    
    for idx in day_rows.index:
        target_row = df.loc[idx]
        target_time = target_row['time']
        
        # Prepare data input (Historical data available BEFORE this candle)
        # We assume we are standing at the close of the PREVIOUS candle
        input_data = df.loc[:idx-1] 
        
        # Skip if not enough data
        if len(input_data) < 100: 
            continue
            
        # Predict
        # Note: We silence the loud prints from predictor if possible, or just accept them
        # Predictor returns dictionary
        try:
            pred_res = predictor.predict(input_data)
            
            # Actual Outcome
            open_p = target_row['open']
            close_p = target_row['close']
            move = close_p - open_p
            pct = (move / open_p) * 100
            
            actual_dir = "BULLISH" if move > 0 else "BEARISH"
            if abs(pct) < 0.05: # Flat/Doji
                 actual_dir = "NEUTRAL"
            
            model_signal = pred_res['prediction'].upper()
            conf = pred_res['confidence']
            
            # Check correctness
            is_correct = (model_signal == actual_dir)
            if actual_dir == "NEUTRAL": is_correct = "N/A" # Hard to predict flat
            
            # Color
            status = "✅ WIN" if is_correct is True else "❌ LOSS" if is_correct is False else "➖ FLAT"
            
            results.append({
                "Time": target_time.strftime("%H:%M"),
                "Model": f"{model_signal} ({conf:.1%})",
                "Actual": f"{actual_dir} ({move:.2f})",
                "Status": status
            })
            
        except Exception as e:
            print(f"Error at {target_time}: {e}")

    print("\n" + "="*60)
    print(f"BACKTEST RESULTS FOR {target_date} (4H Timeframe)")
    print("="*60)
    print(tabulate(results, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    verify_day()

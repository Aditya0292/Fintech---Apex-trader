
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())
from tools.predict import Predictor

def verify_yesterday():
    path = "data/XAUUSD_4h.csv"
    df = pd.read_csv(path)
    
    # We want to check the prediction that happened BEFORE the 08:00 candle.
    # The 08:00 candle is the last row (index 3725).
    # So we simulate the state at 08:00 OPEN (by using data up to 04:00 CLOSE).
    # We slice off the last row.
    
    df_past = df.iloc[:-1] # Drop 08:00 row
    
    # Verify the last row is now 04:00
    last_time = df_past.iloc[-1]['time']
    print(f"Running prediction based on data ending: {last_time}")
    
    predictor = Predictor(timeframe="4 Hour")
    # Force load models? (Predictor does it)
    
    # Predict
    result = predictor.predict(df_past)
    
    print("\nPrediction for Next Candle (which was 08:00):")
    print(f"Signal: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    # Check Actual 08:00 outcome
    actual_row = df.iloc[-1]
    open_p = actual_row['open']
    close_p = actual_row['close']
    move = close_p - open_p
    direction = "BULLISH" if move > 0 else "BEARISH"
    pct = (move / open_p) * 100
    
    print(f"\nActual Outcome (08:00 Candle):")
    print(f"Time: {actual_row['time']}")
    print(f"Open: {open_p:.2f}")
    print(f"Close: {close_p:.2f}")
    print(f"Move: {move:.2f} ({pct:.2f}%) -> {direction}")
    
    if result['prediction'].upper() == direction:
        print("\n[SUCCESS] The model ACCURATELY predicted this move!")
    else:
        print("\n[FAIL] The prediction was incorrect.")

if __name__ == "__main__":
    verify_yesterday()

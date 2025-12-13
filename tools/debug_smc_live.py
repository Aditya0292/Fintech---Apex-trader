import pandas as pd
import numpy as np
from src.analysis.smc_analyzer import SMCAnalyzer

def create_mock_data():
    # Create a pattern: Uptrend -> Swing High -> Down (Bearish OB created) -> Price returns to OB
    data = {
        'open': [10, 11, 12, 13, 14, 13, 12, 11, 12, 13, 13.5],
        'high': [11, 12, 13, 14, 15, 14, 13, 12, 13, 14, 14.2], # OB High around 15?
        # Let's make it clearer.
        # Candle 1: Bullish (10-12)
        # Candle 2: Bullish (12-14)
        # Candle 3: Bullish (14-16) -> Pivot High created at 16
        # Candle 4: Bearish (16-14) -> BOS if break logic fits, or just OB creation
        # Candle 5..10: Down
        # Candle 11: Back up to 15 (Inside OB)
    }
    
    # We need enough data for lookback (50).
    # Let's generate 100 candles.
    dates = pd.date_range(start="2024-01-01", periods=100, freq="15min")
    df = pd.DataFrame(index=range(100))
    df['time'] = dates
    
    # Base price
    prices = [100.0] * 100
    
    # Create Swing High at index 50
    # 40-50: Up
    for i in range(40, 51): prices[i] = prices[i-1] + 1 
    # prices[50] = 111.0 (High)
    
    # 51: Down Candle (Bearish OB candidate is the up candle before the down move? 
    # The logic says "Highest Green Candle" in the move before a break.
    # So index 50 (Green) is the OB logic if 51 breaks structure.
    
    # 51-60: Down (Create BOS)
    for i in range(51, 61): prices[i] = prices[i-1] - 2
    
    # 90-99: Return to OB (111.0)
    # Current Price should be 110.5 (Inside OB 110-111)
    prices[99] = 110.5
    
    # OHLC
    df['close'] = prices
    df['open'] = [p - 0.5 for p in prices] # Mostly Bullish candles construction
    df['high'] = [p + 0.5 for p in prices]
    df['low'] = [p - 1.0 for p in prices]
    
    # Fix the Swing High (Index 50) - Make it Green
    df.loc[50, 'open'] = 110.0
    df.loc[50, 'close'] = 111.0
    df.loc[50, 'high'] = 111.5
    df.loc[50, 'low'] = 109.5
    
    # Fix the BOS trigger (Index 51) - Make it RED and huge drop to break structure
    # Structure needs to break previous Swing Low. 
    # Previous Swing Low might be index 40 (100.0).
    # prices[60] is 111 - 20 = 91. So it breaks.
    
    return df

def test_active_ob():
    df = create_mock_data()
    analyzer = SMCAnalyzer(df, timeframe="15m")
    
    # Current Price is ~110.5
    # Expected OB: The candle at index 50 (High ~111.5, Low ~109.5)
    # Range: 109.5 - 111.5
    # Price 110.5 is INSIDE.
    
    last_close = df.iloc[-1]['close']
    print(f"Current Price: {last_close}")
    
    res = analyzer.get_nearest_structures(last_close)
    
    print("\n[Result] Resistance OB (Expect Found):")
    if res['resistance_ob']:
        print(f"Found: {res['resistance_ob']}")
    else:
        print("None (Filtered out?)")

    print("\n[Result] All Bear Obs (Debugging):")
    for ob in res['all_bear_obs']:
        print(f"OB: {ob['bottom']} - {ob['top']} | Mitigated? {ob.get('mitigated')}")

if __name__ == "__main__":
    test_active_ob()

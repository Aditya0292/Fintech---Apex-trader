
import pandas as pd
import numpy as np

def verify_ema_strategy():
    df = pd.read_csv("data/XAUUSD_history.csv")
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Strategy: If Close > EMA 50, Buy. Hold 5 days.
    df['future_close'] = df['close'].shift(-5)
    df['return'] = (df['future_close'] - df['close']) / df['close']
    
    # Filter for signals
    bullish_signals = df[df['close'] > df['ema_50']]
    bearish_signals = df[df['close'] < df['ema_50']]
    
    # Win Rate
    bull_wins = len(bullish_signals[bullish_signals['return'] > 0])
    bull_total = len(bullish_signals)
    bull_wr = bull_wins / bull_total if bull_total > 0 else 0
    
    bear_wins = len(bearish_signals[bearish_signals['return'] < 0]) # Return should be negative for short
    bear_total = len(bearish_signals)
    bear_wr = bear_wins / bear_total if bear_total > 0 else 0
    
    print(f"EMA 50 Swing Strategy (5-Day Hold):")
    print(f"Bullish Win Rate: {bull_wr:.2%}")
    print(f"Bearish Win Rate: {bear_wr:.2%}")
    
    print(f"Total Signals: {bull_total + bear_total}")

if __name__ == "__main__":
    verify_ema_strategy()

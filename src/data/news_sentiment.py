import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_surprise(actual, forecast, impact_str):
    """
    Calculate surprise score.
    """
    try:
        # Clean strings
        def clean(x):
            if isinstance(x, str):
                x = x.replace('K', '000').replace('M', '000000').replace('%', '').replace('B', '000000000')
                return float(x)
            return float(x)
            
        a = clean(actual)
        f = clean(forecast)
        
        if f == 0: return 0
        
        diff = (a - f) / abs(f)
        
        # Invert logic for some events (Unemployment claims: High is bad for USD)
        # Simplified: We rely on impact color usually, but here we estimate magnitude
        
        return diff
    except:
        return 0

def decay_function(hours_elapsed, half_life=4):
    """
    Exponential decay.
    """
    return np.exp(-np.log(2) * hours_elapsed / half_life)

def simulate_news_impact(timestamps, events=None):
    """
    Generate Sentiment Score Series aligned to timestamps.
    Args:
        timestamps: Series of timestamps (normalized)
        events: List of event dicts (if None, return neutral)
    """
    # Logic: 
    # For each timestamp, find recent past events (within 8 hours).
    # Sum their (Impact * Decay).
    
    # Since this is "Simulate" for backtest (using data we don't fully have in simplified CSV),
    # we usually return zeros or use the 'news_sentiment' column if it existed.
    # But for Real-Time, we use `calculate_current_sentiment`.
    
    # This function is used by feature_pipeline for historical training.
    # Without a historical news database, we return zeros as per current design.
    
    return pd.DataFrame({'time': timestamps, 'news_sentiment': 0.0})

def calculate_live_sentiment(events, current_time):
    """
    Calculate net sentiment at current_time based on list of recent events.
    Returns score: Positive = Strong USD (Bearish Gold), Negative = Weak USD (Bullish Gold).
    """
    score = 0.0
    
    for ev in events:
        # Check if Released
        if ev.get('status') != 'released': continue
        
        ev_time = pd.to_datetime(ev['date']).tz_localize(None)
        elapsed = (current_time - ev_time).total_seconds() / 3600.0
        
        if 0 <= elapsed < 12: # Impact lasts 12 hours
            impact_map = {'High': 1.0, 'Medium': 0.5, 'Low': 0.1}
            base_impact = impact_map.get(ev.get('impact'), 0)
            
            # Direction?
            # We need to parse "Better Than Expected" etc.
            # Simplified: Use the 'actual' vs 'forecast' if available.
            # OR relies on scraping 'class' (green/red) which logic elsewhere handles.
            # Here let's assume 'sentiment' field is populated ('bullish'/'bearish' for USD)
            
            direction = 0
            s = ev.get('sentiment', 'neutral')
            if s == 'bullish': direction = 1
            elif s == 'bearish': direction = -1
            
            decay = decay_function(elapsed)
            score += (direction * base_impact * decay)
            
    return score

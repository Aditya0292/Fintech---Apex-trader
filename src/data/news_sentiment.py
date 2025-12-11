"""
APEX TRADE AI - News Sentiment Analysis
=======================================
Uses FinBERT to analyze economic news headlines.
Source: HuggingFace (ProsusAI/finbert)

Features:
- Sentiment Score (-1 to 1)
- Confidence Level
- Impact Weighting (High/Med/Low)
"""

import pandas as pd
import numpy as np
import os
from transformers import pipeline
import torch
from typing import List, Dict

class NewsManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsManager, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance
    
    def load_model(self):
        if self.model is None:
            print("Loading FinBERT model...")
            try:
                # Use CPU for inference to avoid VRAM issues during training
                device = -1 
                if torch.cuda.is_available():
                    # Optional: use GPU if needed
                    pass
                    
                self.model = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
                print("FinBERT loaded.")
            except Exception as e:
                print(f"Failed to load FinBERT: {e}")
                
    def analyze_news(self, headlines: List[str]) -> float:
        """
        Analyze a list of headlines and return aggregated sentiment score (-1 to 1).
        """
        if not headlines:
            return 0.0
            
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            return 0.0
            
        results = self.model(headlines)
        
        total_score = 0.0
        
        for res in results:
            # FinBERT labels: positive, negative, neutral
            label = res['label']
            score = res['score']
            
            val = 0.0
            if label == 'positive':
                val = score
            elif label == 'negative':
                val = -score
            # Neutral = 0
            
            total_score += val
            
        # Average sentiment
        return total_score / len(headlines)

class FundamentalEngine:
    """
    Specific logic engine for High-Impact USD News.
    Rule: USD Strength -> XAUUSD Down (Sell).
    """
    
    def __init__(self):
        self.rules = {
            "FOMC": "direct", # Rate Up -> USD Bull -> XAU Sell
            "Interest Rate": "direct",
            "CPI": "direct", # CPI Up -> USD Bull -> XAU Sell
            "PPI": "direct",
            "NFP": "direct", # NFP Up -> USD Bull -> XAU Sell
            "Non-Farm": "direct",
            "Claims": "inverse", # Claims Up -> USD Bear -> XAU Buy
            "Unemployment": "inverse", # Unemp Up -> USD Bear -> XAU Buy
            "GDP": "direct",
            "Retail Sales": "direct",
            "Consumer Confidence": "direct",
            "PMI": "direct",
            "Fed Speech": "speech"
        }

    def process_event(self, event_name: str, actual: float, forecast: float, previous: float = None) -> Dict:
        """
        Process a high-impact event and return XAUUSD direction.
        """
        event_lower = event_name.lower()
        
        # Identify Event Type
        rule_type = "direct" # Default assumption: Better data = Strong USD
        
        if "unemployment" in event_lower or "claims" in event_lower:
            rule_type = "inverse" # Bad data = Weak USD = Buy Gold
        elif "speech" in event_lower or "fed" in event_lower:
            return {"usd_impact": "NEUTRAL", "xau_signal": "WAIT", "reason": "Speech requires NLP analysis"}
            
        # Determine Deviation
        deviation = actual - forecast
        
        usd_impact = "NEUTRAL"
        
        if deviation > 0:
            if rule_type == "direct":
                usd_impact = "BULLISH" # Strong Data -> Strong USD
            else:
                usd_impact = "BEARISH" # High Claims -> Weak USD
        elif deviation < 0:
            if rule_type == "direct":
                usd_impact = "BEARISH" # Weak Data -> Weak USD
            else:
                usd_impact = "BULLISH" # Low Claims -> Strong USD
        
        # Final XAU Signal (Always Inverse to USD)
        xau_signal = "SELL" if usd_impact == "BULLISH" else "BUY" if usd_impact == "BEARISH" else "NEUTRAL"
        
        return {
            "event": event_name,
            "actual": actual,
            "forecast": forecast,
            "usd_reaction": usd_impact,
            "xau_signal": xau_signal,
            "reason": f"Actual ({actual}) vs Forecast ({forecast}) -> USD {usd_impact}"
        }

if __name__ == "__main__":
    # Test FinBERT
    print("--- 1. Testing FinBERT ---")
    nm = NewsManager()
    headlines = ["Gold rallies as inflation cools down"]
    score = nm.analyze_news(headlines)
    print(f"Sentiment Score: {score:.4f}")
    
    # Test Fundamental Engine
    print("\n--- 2. Testing Fundamental Engine ---")
    fe = FundamentalEngine()
    
    # Example 1: High NFP (Bullish USD -> Sell Gold)
    res1 = fe.process_event("Non-Farm Payrolls", actual=250, forecast=180)
    print(f"NFP (250 vs 180): USD {res1['usd_reaction']} -> XAU {res1['xau_signal']}")
    
    # Example 2: High Unemployment (Bearish USD -> Buy Gold)
    res2 = fe.process_event("Unemployment Rate", actual=4.2, forecast=3.9)
    print(f"Unemployment (4.2 vs 3.9): USD {res2['usd_reaction']} -> XAU {res2['xau_signal']}")

def simulate_news_impact(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load actual news from data/news_sentiment.csv if available.
    Otherwise fallback to simulation.
    """
    csv_path = "data/news_sentiment.csv"
    
    if os.path.exists(csv_path):
        try:
            print("Loading REAL news sentiment from CSV...")
            news_df = pd.read_csv(csv_path)
            # news_df columns: time,event,impact,sentiment,score
            
            # Convert 'time' to datetime
            # Format in CSV is YYYY-MM-DD HH:MM:SS
            news_df['time'] = pd.to_datetime(news_df['time'])
            
            # We need to map this to the input 'dates' line by line?
            # Or assume 'dates' is the target index.
            # We will merge on date (Day) or just use the latest sentiment.
            
            # Simple approach: Create a Series aligned to 'dates'
            # Fill with 0, then update where news matches date.
            
            sent_series = pd.Series(0.0, index=dates)
            
            # For each news event, propagate its impact forward for 24h?
            # Or just set the value on that day.
            
            news_df = news_df.sort_values('time')
            
            # Map news to input dates (Resampling)
            # This is tricky because input dates might be 15m or Daily.
            # Let's use asof merge or ffill.
            
            temp_df = pd.DataFrame(index=dates)
            temp_df['news_sentiment'] = 0.0
            
            # Iterate and apply
            for _, row in news_df.iterrows():
                t = row['time']
                s = row['score']
                
                # Apply to nearest date in index (forward looking or backward?)
                # News happens at T. It affects T and T+1.
                # Find indices >= T
                mask = temp_df.index >= t
                # Decay impact? For now, constant impact for rest of day?
                # Or just assign to nearest.
                
                # Assign to the exact timestamp or next available
                # If Daily, assign to that Day.
                # If 15m, assign to that 15m bar and follow.
                
                # Intelligent Decay: Exponential decay (Half-life ~2 hours)
                # News affects the market strongly at T=0, then fades.
                impact_window_hours = 24
                end_t = t + pd.Timedelta(hours=impact_window_hours)
                
                mask_valid = (temp_df.index >= t) & (temp_df.index <= end_t)
                valid_indices = temp_df.index[mask_valid]
                
                if len(valid_indices) > 0:
                    # Calculate hours elapsed for each point in window
                    # Ensure t is Timestamp for subtraction
                    time_diffs = (valid_indices - t).total_seconds() / 3600.0
                    
                    # Decay formula: Score * exp(-decay_rate * t)
                    # decay_rate 0.5 means effect halves every ~1.4 hours
                    decay_rate = 0.5 
                    
                    decay_factors = np.exp(-decay_rate * time_diffs)
                    
                    # Add to existing sentiment (Cumulative impact of multiple events)
                    # Use .loc with specific indices. 
                    # Note: We += to accumulate if multiple events overlap.
                    current_vals = temp_df.loc[valid_indices, 'news_sentiment']
                    new_vals = current_vals + (s * decay_factors)
                    
                    temp_df.loc[valid_indices, 'news_sentiment'] = new_vals
                
            print(f"Loaded {len(news_df)} news events.")
            return temp_df.reset_index().rename(columns={'index': 'time'})
            
            return temp_df.reset_index().rename(columns={'index': 'time'})
            
        except Exception as e:
            print(f"Failed to load news CSV: {e}. defaulting to NEUTRAL (0.0).")
    
    print("No real news found. Using NEUTRAL sentiment (0.0).")
    # Return 0.0s
    df = pd.DataFrame({'time': dates, 'news_sentiment': 0.0})
    # Ensure time is datetime? Input dates is DatetimeIndex.
    # The return format expects specific structure?
    # Original: df['time'] = df['time'].dt.strftime('%Y-%m-%d')
    # Actually caller (feature_pipeline) expects: df = pd.merge(df, news_df, on='time', how='left')
    # So 'time' column must match.
    return df

if __name__ == "__main__":
    nm = NewsManager()
    headlines = [
        "Gold rallies as inflation cools down",
        "Fed signals rate cuts next year",
        "Dollar strengthens on strong NFP data"
    ]
    score = nm.analyze_news(headlines)
    print(f"Sentiment Score: {score:.4f}")

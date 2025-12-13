import sys
import os
import pandas as pd
from datetime import datetime
import time

# Add root
sys.path.append(os.path.abspath(os.getcwd()))

from src.data.news_sentiment import simulate_news_impact
from src.data.news_sentiment_azure import AzureOpenAIEngine
from tools.fetch_todays_news import get_data, process_events

def test_historical_simulation():
    print("\n[1] Testing Historical News Simulation (Feature Pipeline)...")
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    df_impact = simulate_news_impact(dates)
    
    if 'news_sentiment' in df_impact.columns and len(df_impact) == 5:
        print("    ✅ Simulation returned valid DataFrame shape.")
    else:
        print("    ❌ Simulation returned invalid shape.")
        
    if df_impact['news_sentiment'].sum() == 0:
         print("    ℹ️  Simulation returned all zeros (Expected for mock).")

def test_azure_engine_logic():
    print("\n[2] Testing Azure AI Engine (Simulation Mode)...")
    ai = AzureOpenAIEngine()
    
    # Test Bullish for Gold (Weak USD)
    # Event: CPI Lower than forecast
    res = ai.analyze_event("CPI m/m", actual=0.1, forecast=0.3)
    score = res['sentiment_score']
    print(f"    CPI (0.1 vs 0.3): Score={score} Reason={res['reasoning'][:50]}...")
    
    if score > 0:
        print("    ✅ Logic Valid: Lower CPI -> Weak USD -> Bullish Gold (+)")
    else:
        print("    ❌ Logic Invalid: Expected Positive Score.")

    # Test Bearish for Gold (Strong USD)
    # Event: NFP Higher than forecast
    res = ai.analyze_event("Non-Farm Employment Change", actual=300, forecast=150)
    score = res['sentiment_score']
    print(f"    NFP (300 vs 150): Score={score} Reason={res['reasoning'][:50]}...")
    
    if score < 0:
        print("    ✅ Logic Valid: High NFP -> Strong USD -> Bearish Gold (-)")
    else:
        print("    ❌ Logic Invalid: Expected Negative Score.")

def test_fetch_live_logic():
    print("\n[3] Testing Live Fetch Logic (Mock/Cache)...")
    # We won't actually hit the web to avoid flakiness, but we check if get_data import works
    # and if process_events handles data correctly.
    
    mock_raw = [{
        "title": "Core CPI m/m",
        "country": "USD",
        "impact": "High",
        "raw_date": "Jan 1 2025 10:00am", # Future date
        "forecast": "0.3%",
        "actual": "0.5%", 
        "sentiment": "bearish" # Bearish for instrument? usually FF sentiment is green/red.
    }]
    
    processed = process_events(mock_raw)
    if processed:
        print(f"    ✅ Processed {len(processed)} event(s).")
        print(f"    Event: {processed[0]['event']} | Sentiment: {processed[0]['sentiment']}")
    else:
        print("    ❌ Processing failed.")

if __name__ == "__main__":
    test_historical_simulation()
    test_azure_engine_logic()
    test_fetch_live_logic()

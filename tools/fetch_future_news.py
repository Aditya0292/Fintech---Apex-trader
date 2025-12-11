"""
APEX TRADE AI - Long Term News Fetcher
======================================
Fetches High-Impact USD news for the next 6 MONTHS from Forex Factory using the Scraper.
Saves to data/future_news_6m.csv.

Usage:
    python tools/fetch_future_news.py
"""

import sys
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.fetch_calendar import NewsScraper

def fetch_six_months():
    print("--- Fetching Future News (Next 6 Months) ---")
    scraper = NewsScraper()
    
    current_date = datetime.now()
    all_events = []
    
    # ForexFactory uses 'mmm.yyyy' format (e.g. 'dec.2025', 'jan.2026')
    for i in range(6):
        target_date = current_date + relativedelta(months=i)
        month_str = target_date.strftime("%b.%Y").lower() # e.g. 'dec.2025'
        
        print(f"\n[{i+1}/6] Scraping {month_str}...")
        try:
            events = scraper.fetch_news(month_str)
            print(f"   Found {len(events)} High-Impact USD events.")
            all_events.extend(events)
        except Exception as e:
            print(f"   Error fetching {month_str}: {e}")
            
    if not all_events:
        print("\nNo events found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_events)
    
    # Ensure directory exists
    os.makedirs("data", exist_ok=True)
    out_path = "data/future_news_6m.csv"
    
    df.to_csv(out_path, index=False)
    print(f"\nDone! Saved {len(df)} events to {out_path}")
    print("Sample:")
    print(df.head())

if __name__ == "__main__":
    fetch_six_months()

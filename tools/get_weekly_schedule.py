"""
APEX TRADE AI - Weekly Schedule Viewer
======================================
Fetches and displays the schedule of High-Impact USD news for the upcoming week.
Uses the Selenium Scraper (Free) as the API is restricted.

Usage:
    python get_weekly_schedule.py --week next
    python get_weekly_schedule.py --week this
"""

import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.fetch_calendar import NewsScraper
import pandas as pd

def show_schedule(week="next"):
    print(f"\n--- Fetching Schedule for {week.upper()} WEEK ---")
    
    try:
        print("Please wait... (Scraping ForexFactory)")
        from src.data.fetch_calendar import NewsScraper
        scraper = NewsScraper()
        # map 'next' -> 'next_week'
        period = "next_week" if week == "next" else "this_week"
        events = scraper.fetch_news(period)
        
        if not events:
            print("No High-Impact USD events found for this period.")
            print("(Note: Check if today is weekend or if events are Medium impact only)")
            return

        print(f"\nFound {len(events)} High-Impact Events:")
        print("-" * 60)
        print(f"{'DATE':<12} | {'TIME':<8} | {'EVENT':<30} | {'FORECAST':<8}")
        print("-" * 60)
        
        for e in events:
            # Clean up text
            date = e['date']
            time = e['time']
            name = e['event']
            fcst = e['forecast']
            print(f"{date:<12} | {time:<8} | {name:<30} | {fcst:<8}")
            
        print("-" * 60)
        print("To trade these events, run:")
        print("  python live_news.py --auto")
        print("at the scheduled time.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=str, default="next", choices=["this", "next"], help="Which week to fetch")
    args = parser.parse_args()
    
    show_schedule(args.week)

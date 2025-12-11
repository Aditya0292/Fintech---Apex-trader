"""
APEX TRADE AI - News Monitor Service
====================================
Running this script starts a background process that:
1. Checks today's schedule on startup.
2. Monitors time.
3. Automatically fetches 'Actual' data when an event is released.
4. Updates the prediction.

Usage: python news_monitor.py
"""

import time
import schedule
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.getcwd())

# Import our robust new tools
from tools.fetch_todays_news import fetch_and_save_news
from tools.predict_all import run_multi_timeframe_prediction
import pandas as pd

# Global state to track changes
LAST_NEWS_STATE = ""

def check_and_update():
    """
    1. Fetch Latest News.
    2. Check if data changed (e.g. valid Actual appeared).
    3. If changed, trigger prediction.
    """
    global LAST_NEWS_STATE
    
    print(f"\n[MONITOR] Checking News Feed ({datetime.now().strftime('%H:%M:%S')})...")
    
    try:
        # 1. Fetch & Save
        fetch_and_save_news(save_path="data/news_sentiment.csv")
        
        # 2. Check for Change
        # We read the file we just saved to generate a checksum/state
        if os.path.exists("data/news_sentiment.csv"):
            df = pd.read_csv("data/news_sentiment.csv")
            # Create a string representation of Actuals
            current_state = "".join(df['actual'].astype(str).tolist())
            
            if current_state != LAST_NEWS_STATE:
                if LAST_NEWS_STATE == "":
                     print("[MONITOR] Initial State Loaded.")
                else:
                     print("🚨 [ALERT] NEW NEWS DATA DETECTED! 🚨")
                     print("Triggering Prediction Pipeline...")
                     run_multi_timeframe_prediction()
                
                LAST_NEWS_STATE = current_state
            else:
                 print("[MONITOR] No new data release yet.")
                 
    except Exception as e:
        print(f"[ERROR] Monitor Failed: {e}")

def start_monitor():
    print(f"APEX Real-Time News Monitor Started.")
    print("Mode: Polling every 60 seconds.")
    print("Press Ctrl+C to stop.")
    
    # Run once immediately
    check_and_update()
    
    while True:
        try:
            time.sleep(60)
            check_and_update()
        except KeyboardInterrupt:
            print("Monitor Stopped.")
            break
        except Exception as e:
            print(f"Monitor Error: {e}")

if __name__ == "__main__":
    start_monitor()

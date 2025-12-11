"""
APEX TRADE AI - Live News Update
================================
Run this tool when High-Impact USD news is released.
It will:
1. Accept Event Details (Actual check)
2. Calculate Fundamental Bias using Specific Rules covers
3. Combine with Technical Model Prediction
4. Output Final Trade Signal
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.news_sentiment import FundamentalEngine
from predict import Predictor
import pandas as pd
import json

def run_live_update(auto_mode=False):
    print("------------------------------------------------")
    print("   APEX Live News Update (High-Impact USD)      ")
    print("------------------------------------------------")
    
    event_name = ""
    forecast = 0.0
    actual = 0.0
    
    if auto_mode:
        print(f"[AUTO] Fetching news from Forex Factory (Selenium)...")
        try:
            from src.data.fetch_calendar import NewsScraper
            scraper = NewsScraper()
            news_list = scraper.fetch_news("today")
            
            if not news_list:
                print("No High-Impact USD news found today.")
                return

            print(f"Found {len(news_list)} events.")
            # Pick the most recent one with data
            target_event = None
            for n in news_list:
                if n['actual'] and n['actual'].strip():
                    target_event = n
                    # Keep looking for latest? List is usually chronological.
                    # We want the LAST one that has data (most recent release)
            
            if not target_event:
                print("Events found but NO Actual data released yet.")
                print("Upcoming:", news_list[0]['event'], "@", news_list[0]['time'])
                return
                
            print(f"Selected Event: {target_event['event']}")
            print(f"Time: {target_event['time']}")
            print(f"Data: Act={target_event['actual']} | Fcst={target_event['forecast']}")
            
            # FMP returns raw numbers, Selenium returns strings
            # We need to handle both
            try:
                # If using Selenium scraper, we need clean_value
                actual = scraper.clean_value(target_event['actual'])
                forecast = scraper.clean_value(target_event['forecast'])
            except:
                print(f"Error converting values: Act={target_event['actual']}, Fcst={target_event['forecast']}")
                return
            
            if actual is None or forecast is None:
                print("Error parsing values.")
                return
                
        except Exception as e:
            print(f"Auto-Fetch Failed: {e}")
            return
    else:
        event_name = input("Enter Event Name (e.g. CPI, NFP): ").strip()
        try:
            forecast = float(input("Forecast Value: "))
            actual = float(input("Actual Value  : "))
        except ValueError:
            print("Invalid number format.")
            return

    # 1. Fundamental Analysis
    fe = FundamentalEngine()
    fund_res = fe.process_event(event_name, actual, forecast)
    
    print("\n[FUNDAMENTAL ANALYSIS]")
    print(f"Event: {event_name}")
    print(f"Result: {fund_res['usd_reaction']} USD")
    print(f"XAUUSD Fundamental Bias: {fund_res['xau_signal']} ({fund_res['reason']})")
    
    # 2. Technical Analysis
    print("\n[TECHNICAL ANALYSIS]")
    try:
        # Load latest data
        try:
             df = pd.read_csv("data/XAUUSD_history.csv")
        except:
             df = pd.read_csv("../data/XAUUSD_history.csv")
        pred_engine = Predictor() # This loads models (slow)
        tech_res = pred_engine.predict(df)
        
        tech_dir = tech_res['prediction']
        tech_conf = tech_res['confidence']
        dxy_trend = tech_res['context']['dxy_trend']
        
        print(f"Model Prediction: {tech_dir} (Conf: {tech_conf:.2%})")
        print(f"Context: DXY Trend is {dxy_trend}")
        
        # Trade Levels
        levels = tech_res.get('trade_levels', {})
        if levels and tech_dir != "NEUTRAL":
            print(f"Setup: Entry {levels.get('entry')} | SL {levels.get('sl')} | TP {levels.get('tp')}")
            print(f"Risk/Reward: {levels.get('risk_reward')} (ATR: {levels.get('atr')})")
        
    except Exception as e:
        print(f"Technical Analysis failed: {e}")
        tech_dir = "NEUTRAL"
        tech_conf = 0.0
        levels = {}

    # 4. Integrate Currency Strength Meter
    try:
        from src.data.fetch_csm import CSMScraper
        csm = CSMScraper()
        usd_index = csm.fetch_usd_strength() # e.g. 4.0
    except Exception as e:
        print(f"CSM Error: {e}")
        usd_index = 5.0 # Neutral default

    csm_signal = "NEUTRAL"
    if usd_index < 4.5:
        csm_signal = "WEAK (Bullish Gold)"
    elif usd_index > 5.5:
        csm_signal = "STRONG (Bearish Gold)"
        
    print(f"USD Strength Index: {usd_index} | Signal: {csm_signal}")

    # 3. Final Synthesis
    print("\n------------------------------------------------")
    print("   FINAL LIVE PREDICTION")
    print("------------------------------------------------")
    
    # Logic: Fundamental Veto
    # If Fundamental is Strong (Deviation exists), it usually overrides technicals intraday.
    
    final_signal = "NEUTRAL"
    reasoning = ""
    
    fund_signal = fund_res['xau_signal'] # BUY / SELL
    
    # Normalize signals for comparison
    fund_dir = "NEUTRAL"
    if fund_signal == "BUY": fund_dir = "Bullish"
    elif fund_signal == "SELL": fund_dir = "Bearish"
    
    # Check Verification with CSM
    csm_confirmation = ""
    if (fund_dir == "Bullish" and usd_index < 4.5) or (fund_dir == "Bearish" and usd_index > 5.5):
        csm_confirmation = " + CSM Confirmed"
    elif (fund_dir == "Bullish" and usd_index > 6.0) or (fund_dir == "Bearish" and usd_index < 4.0):
        csm_confirmation = " (CSM Divergence Warning)"
    
    if fund_dir == tech_dir:
        final_signal = f"STRONG {fund_signal}{csm_confirmation}"
        reasoning = f"Fundamental ({fund_signal}) and Technicals ({tech_dir}) align perfectly."
    elif fund_dir == "NEUTRAL":
        final_signal = tech_dir
        reasoning = "News impact neutral, following technical trend."
    else:
        # Conflict
        final_signal = f"{fund_signal} (Short Term){csm_confirmation}"
        reasoning = f"News Trigger ({fund_signal}) overrides Technical Trend ({tech_dir}) for intraday scalps."
        # Levels might be invalid if direction flips
        if fund_dir != tech_dir:
             levels = {} 
        
    print(f"Overall USD Bias: {fund_res['usd_reaction']}")
    print(f"Final Direction for XAUUSD: {final_signal}")
    print(f"Why: {reasoning}")
    
    if levels and "STRONG" in final_signal or final_signal == tech_dir:
         print("-" * 30)
         print(f"🎯 SWING PLAN (Safe)")
         print(f"   Entry: {levels.get('entry')} (Market)")
         print(f"   SL   : {levels.get('sl')} (1.5x ATR)")
         print(f"   TP   : {levels.get('tp')} (2.0x ATR)")
         
         s_levels = tech_res.get('scalp_levels', {})
         if s_levels:
             print("-" * 30)
             print(f"⚡ SCALP PLAN (Fast)")
             print(f"   Entry: {s_levels.get('entry')} (Market)")
             print(f"   SL   : {s_levels.get('sl')} (0.5x ATR)")
             print(f"   TP   : {s_levels.get('tp')} (0.8x ATR)")
         print("-" * 30)
         
    print("------------------------------------------------")
    print("This is the updated best direction for XAUUSD based on High-Impact USD fundamentals.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Enable auto-fetch")
    args = parser.parse_args()
    
    run_live_update(auto_mode=args.auto)

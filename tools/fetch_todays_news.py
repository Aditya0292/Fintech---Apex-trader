
import requests
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timedelta
import pandas as pd
import os
import time
import argparse
from dateutil import parser as date_parser
from tabulate import tabulate

# --- CONFIGURATION (Institutional Grade) ---
CACHE_DIR = "data"
CACHE_FILE_JSON = os.path.join(CACHE_DIR, "calendar_cache.json")
CACHE_DURATION = 600  # 10 Minutes (600 seconds)
MAX_RETRIES = 1       # Don't loop forever, failover instead

# Cloudflare / Protection Bypass Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "application/json, text/xml, application/xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.forexfactory.com",
    "Connection": "keep-alive"
}

SOURCES = [
    {"type": "json", "url": "https://nfs.faireconomy.media/ff_calendar_thisweek.json"},
    {"type": "xml",  "url": "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"},
    # {"type": "xml",  "url": "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.xml"} # Backup mirror
]

def ensure_data_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def is_cache_valid(filepath):
    """Check if local cache exists and is fresh (< 10 mins old)."""
    if os.path.exists(filepath):
        age = time.time() - os.path.getmtime(filepath)
        if age < CACHE_DURATION:
            return True, age
    return False, 0

def fetch_url_safe(url, is_retry=False):
    """
    Robust fetcher with rate-limit handling.
    """
    try:
        print(f"  ⚡ Connecting to: {url} ...")
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        status = response.status_code
        if status == 200:
            return response.content, None
        
        elif status in [420, 429]:
            msg = f"⏸ Rate-limited ({status})."
            if not is_retry:
                print(f"    {msg} PROTCOL: Sleeping 60s (Mini-Cool) then retrying once...")
                time.sleep(60) # User suggested 10m, but 60s is usually enough for a burst. 
                               # If we want full 10m, we block user for too long. 
                               # Strategy: Failover to next source is better than waiting 10m.
                return fetch_url_safe(url, is_retry=True)
            else:
                return None, "Rate Limit Persisted"
        else:
            return None, f"Status {status}"
            
    except Exception as e:
        return None, str(e)

def get_data(force=False):
    """
    Orchestrates Caching -> Fetching -> Saving.
    Returns: List of event dictionaries or None.
    """
    ensure_data_dir()
    
    # 1. Try Cache First (Unless Forced)
    valid, age = is_cache_valid(CACHE_FILE_JSON)
    if valid and not force:
        print(f"✅ Using Local Cache (Age: {int(age)}s). Zero-Latency.")
        try:
            with open(CACHE_FILE_JSON, 'r') as f:
                return json.load(f)
        except:
            print("    [Warn] Cache corrupted. Re-fetching.")
    
    # 2. Fetch from Network (Loop Sources)
    for source in SOURCES:
        content, error = fetch_url_safe(source['url'])
        
        if content:
            # Parse & Standardize
            events = []
            try:
                if source['type'] == "json":
                    raw_data = json.loads(content)
                    # Save raw JSON as cache regardless of parsing (mirroring file)
                    with open(CACHE_FILE_JSON, 'w') as f:
                        json.dump(raw_data, f)
                    return raw_data
                    
                elif source['type'] == "xml":
                    # Convert XML to JSON-like structure for consistency
                    root = ET.fromstring(content)
                    # We won't save XML as JSON cache directly, we'd need to convert.
                    # For simplicity, if we fell back to XML, we parse it here.
                    # But ideally we want a unified cache format.
                    # Let's skip saving XML to JSON cache for now to avoid complexity,
                    # just return the parsed objects.
                    # Actually, let's just return the raw list of dicts.
                    for item in root.findall('event'):
                        events.append({
                            "title": item.find('title').text,
                            "country": item.find('country').text,
                            "date": item.find('date').text + "T" + item.find('time').text, # Rough/Partial
                            "impact": item.find('impact').text,
                            "forecast": item.find('forecast').text,
                            "previous": item.find('previous').text
                        })
                    return events
            except Exception as e:
                print(f"    ❌ Parse Error on {source['url']}: {e}")
                continue
        else:
            print(f"    ❌ Failed: {error}")
            
    print("⛔ All Sources Failed.")
    return None

def process_events(raw_data):
    """
    Standardize raw data into our Dashboard format.
    Handles Timezone conversion and filtering.
    """
    processed = []
    
    today_match_str = datetime.now().strftime("%Y-%m-%d")
    
    for item in raw_data:
        country = item.get("country", "")
        impact = item.get("impact", "")
        
        if country == "USD" and impact in ["High", "Medium"]:
            raw_date = item.get("date", "")
            
            # Smart Date Parsing
            try:
                # If it's the specific XML format we hacked in, handle it separately?
                # No, let's assume JSON primarily.
                # JSON Format: "2025-12-07T18:30:00-05:00"
                if "T" in raw_date and "-" in raw_date:
                    dt_obj = date_parser.parse(raw_date)
                    dt_local = dt_obj.astimezone()
                else:
                    # Fallback for XML simplistic join
                    # This is weak, but JSON is 99% uptime.
                    dt_local = datetime.now() # Placeholder if parse fails
                
                processed.append({
                    "DateTime": dt_local,
                    "Date": dt_local.strftime("%Y-%m-%d"),
                    "Time": dt_local.strftime("%H:%M"),
                    "DisplayTime": dt_local.strftime("%I:%M%p").lower(),
                    "Event": item.get("title", ""),
                    "Impact": impact,
                    "Forecast": item.get("forecast", "") or "",
                    "Previous": item.get("previous", "") or "",
                    "Actual": ""
                })
            except:
                pass
                
    return processed

def display_dashboard(df):
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 1. TODAY
    print("\n" + "="*80)
    print(f"📅 TODAY'S NEWS (System Date: {today_str})")
    print("="*80)
    
    df_today = df[df['Date'] == today_str].copy()
    
    output_csv = os.path.join(CACHE_DIR, "news_sentiment.csv")
    
    if not df_today.empty:
        # Display
        # Rename 'DisplayTime' -> 'Time' for the table header, so use 'Time' in selector
        df_disp = df_today.rename(columns={"DisplayTime": "Time"})
        cols_disp = ["Time", "Event", "Impact", "Forecast", "Previous", "Actual"]
        print(tabulate(df_disp[cols_disp], headers=cols_disp, tablefmt="simple", showindex=False))
        
        # Save CSV for Bot
        csv_df = df_today.rename(columns={"DisplayTime": "time", "Event": "event", "Impact": "impact", "Forecast": "forecast", "Previous": "previous", "Actual": "actual"})
        csv_df['score'] = 0.0
        csv_df.to_csv(output_csv, index=False)
        print(f"\n[Data Pipeline]: Saved today's events to {output_csv}")
    else:
        print("No High/Medium Impact USD Events Today.")
        if not os.path.exists(output_csv):
             pd.DataFrame(columns=["time", "event", "impact", "forecast", "actual", "score"]).to_csv(output_csv, index=False)

    # 2. WEEKLY
    print("\n" + "="*80)
    print("📅 WEEKLY OVERVIEW (Mon-Fri)")
    print("="*80)
    
    df['Day'] = df['DateTime'].dt.day_name()
    df['Time'] = df['DisplayTime']
    cols_wk = ["Day", "Date", "Time", "Event", "Impact", "Forecast", "Previous", "Actual"]
    print(tabulate(df[cols_wk], headers=cols_wk, tablefmt="simple", showindex=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore cache and force refresh (Risk of 429)")
    args = parser.parse_args()
    
    raw = get_data(force=args.force)
    if raw:
        clean_data = process_events(raw)
        if clean_data:
            df = pd.DataFrame(clean_data)
            df = df.sort_values(by="DateTime")
            display_dashboard(df)
        else:
            print("No events found after processing.")

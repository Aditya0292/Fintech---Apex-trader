
import requests
import xml.etree.ElementTree as ET
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
import os
import time
import argparse
from dateutil import parser as date_parser
from tabulate import tabulate
from bs4 import BeautifulSoup

# Selenium Imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# --- CONFIGURATION (Institutional Grade) ---
CACHE_DIR = "data"
CACHE_FILE_JSON = os.path.join(CACHE_DIR, "calendar_cache.json")
CACHE_DURATION = 300  # 5 Minutes
MAX_RETRIES = 2

def ensure_data_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def is_cache_valid(filepath):
    if os.path.exists(filepath):
        age = time.time() - os.path.getmtime(filepath)
        if age < CACHE_DURATION:
            return True, age
    return False, 0

def fetch_with_selenium(url):
    if not SELENIUM_AVAILABLE:
        print("    ⚠️ Selenium not installed. Cannot scrape actuals properly.")
        return None
        
    print(f"  ⚡ Scraping via Selenium: {url} ...")
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Suppress logging
        options.add_argument("--log-level=3")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        driver.get(url)
        time.sleep(5) # Allow CF/JS to load
        
        content = driver.page_source
        driver.quit()
        return content
    except Exception as e:
        print(f"    ❌ Selenium Error: {e}")
        return None

def parse_ff_html(html_content):
    """Parse ForexFactory HTML Table"""
    events = []
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table", class_="calendar__table")
        if not table:
            return []
            
        rows = table.find_all("tr", class_="calendar__row")
        
        current_date_str = ""
        
        for row in rows:
            # Date handling (Row with date usually precedes events or is part of first event)
            # FF structure: Date is in a separate column or row sometimes
            # Actually, valid rows usually have data-event-id
            
            # Extract basic data
            currency_div = row.find("td", class_="calendar__currency")
            currency = currency_div.get_text(strip=True) if currency_div else ""
            
            # Update date if new day row? 
            # Actually FF rows have date in the first column for the first event of the day
            date_cell = row.find("td", class_="calendar__date")
            if date_cell:
                d_text = date_cell.get_text(strip=True)
                if d_text:
                    # Format: "SunDec 7" or "Dec 7"
                    current_date_str = d_text
            
            if not currency: continue # Skip spacer rows
            
            # Title
            title_div = row.find("td", class_="calendar__event")
            title = title_div.find("span", class_="calendar__event-title").get_text(strip=True) if title_div else ""
            
            # Impact
            impact_div = row.find("td", class_="calendar__impact")
            impact_span = impact_div.find("span") if impact_div else None
            # Class name indicates impact: icon--ff-impact-red (High), ora (Medium), yel (Low)
            impact = "Low"
            if impact_span:
                cls = impact_span.get("class", [])
                # Check for "icon--ff-impact-red" or "icon--ff-impact-ora"
                # Sometimes it might be "orange" suffix depending on FF version, so check properly
                str_cls = " ".join(cls)
                if "icon--ff-impact-red" in str_cls: impact = "High"
                elif "icon--ff-impact-ora" in str_cls or "icon--ff-impact-orange" in str_cls: impact = "Medium"
            
            
            # Time
            time_div = row.find("td", class_="calendar__time")
            time_str = time_div.get_text(strip=True) if time_div else ""
            
            # Forecast
            forecast_div = row.find("td", class_="calendar__forecast")
            forecast = forecast_div.get_text(strip=True) if forecast_div else ""
            
            # Actual & Sentiment
            actual_div = row.find("td", class_="calendar__actual")
            actual = actual_div.get_text(strip=True) if actual_div else ""
            
            sentiment = "neutral"
            if actual_div:
                # Check spans for 'better' or 'worse'
                spans = actual_div.find_all("span")
                for span in spans:
                    cls = span.get("class", [])
                    if "better" in cls:
                        sentiment = "bullish" # Green = Good for USD
                    elif "worse" in cls:
                        sentiment = "bearish" # Red = Bad for USD
            
            # Previous
            prev_div = row.find("td", class_="calendar__previous")
            previous = prev_div.get_text(strip=True) if prev_div else ""

            # Construct Date
            # We need year. Assume current year or infer.
            # Simplified: Use Today's year.
            year = datetime.now().year
            # Parse "Dec 7" -> Date object
            # This is complex without proper year logic for year boundaries.
            # For "This Week", it's usually current year.
            
            full_date_str = f"{current_date_str} {year} {time_str}" 
            # This is messy. Instead rely on the JSON feed for structure if scraping fails?
            # Or assume standard parsing.
            
            events.append({
                "title": title,
                "country": currency,
                "impact": impact,
                "forecast": forecast,
                "previous": previous,
                "actual": actual,
                "sentiment": sentiment,
                "raw_date": full_date_str, # Store raw for processing later
                "time_str": time_str,
                "day_str": current_date_str
            })
            
    except Exception as e:
        print(f"    ❌ Parse HTML Error: {e}")
        return []
        
    return events

def get_data(force=False):
    ensure_data_dir()
    
    # 1. Try Cache
    valid, age = is_cache_valid(CACHE_FILE_JSON)
    if valid and not force:
        print(f"✅ Using Local Cache (Age: {int(age)}s).")
        try:
            with open(CACHE_FILE_JSON, 'r') as f:
                return json.load(f)
        except: pass
    
    # 2. Try Selenium Scraping (PRIMARY for Actuals)
    # URL: Calendar for this week
    url = "https://www.forexfactory.com/calendar?week=this"
    html = fetch_with_selenium(url)
    
    events = []
    if html:
        events = parse_ff_html(html)
        
    if not events:
        print("    ⚠️ Scraping failed/empty. Falling back to JSON feed (No Actuals).")
        # Fallback to JSON
        try:
            r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
            if r.status_code == 200:
                events = json.loads(r.content)
        except: pass
        
    if events:
        # Save cache
        with open(CACHE_FILE_JSON, 'w') as f:
            json.dump(events, f)
        return events
        
    print("⛔ All Sources Failed.")
    return None

def process_events(raw_data):
    processed = []
    now = datetime.now(timezone.utc)
    
    # Logic to parse dates from scraped data is tricky.
    # We will prioritize the known fields.
    
    for item in raw_data:
        # Check if it matches USD and High/Med
        country = item.get("country", "")
        impact = item.get("impact", "")
        
        # Scraper returns: impact="High"/"Medium"
        # JSON returns: impact="High"/"Medium" 
        
        if country == "USD" and impact in ["High", "Medium"]:
            
            # Date Parsing
            # Scraper output: raw_date="SunDec 7 2025 3:00pm" (Example)
            # JSON output: date="2025-12-07T15:00:00-05:00"
            
            dt_obj = None
            raw_date = item.get("raw_date") or item.get("date")
            
            try:
                # Try standard parser first ( works for JSON ISO and fuzzy English)
                # Note: "SunDec 7 2025" might confuse parser.
                # Clean up: "SunDec 7" -> "Dec 7"
                if "raw_date" in item: # From Scraper
                     # Custom clean
                     # "SunDec 7 2025 3:00pm"
                     # Remove day name prefix if concatenated?
                     # Actually FF date text is "SunDec 7".
                     # Just try dateutil
                     dt_obj = date_parser.parse(raw_date)
                else:
                     dt_obj = date_parser.parse(raw_date)
                     
                # Localize
                if dt_obj.tzinfo is None:
                     # Assume NY (ET)
                     # Start simple: UTC-5
                     dt_obj = dt_obj.replace(tzinfo=timezone(timedelta(hours=-5)))
                
                dt_obj = dt_obj.astimezone(timezone.utc)
                
            except:
                # If parse fails, use placeholder
                dt_obj = now
            
            actual = item.get("actual", "")
            forecast = item.get("forecast", "")
            
            # Logic
            status = "released" if actual else "scheduled"
            if dt_obj < now and not actual:
                status = "delayed"

            processed.append({
                "time": dt_obj.isoformat(),
                "timestamp": dt_obj.timestamp(),
                "display_time": dt_obj.strftime("%H:%M UTC"),
                "event": item.get("title", ""),
                "impact": impact,
                "forecast": forecast or "-",
                "previous": item.get("previous", "") or "-",
                "actual": actual or "-",
                "sentiment": item.get("sentiment", "neutral"),
                "status": status
            })
            
    # Sort
    processed.sort(key=lambda x: x['timestamp'])
    return processed

def display_dashboard(events):
    if isinstance(events, pd.DataFrame):
        df = events
    else:
        if not events:
            print("No USD High/Medium Impact Events Found.")
            return
        df = pd.DataFrame(events)
        
    if df.empty:
        print("No USD High/Medium Impact Events Found.")
        return

    cols = ["display_time", "event", "impact", "forecast", "previous", "actual", "status"]
    # Ensure cols exist
    existing_cols = [c for c in cols if c in df.columns]
    
    print("\n" + "="*80)
    print(f"📅 ECONOMIC CALENDAR (Weekly w/ Actuals)")
    print("="*80)
    print(tabulate(df[existing_cols], headers=existing_cols, tablefmt="simple", showindex=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force refresh")
    args = parser.parse_args()
    
    raw = get_data(force=args.force)
    if raw:
        clean = process_events(raw)
        display_dashboard(clean)


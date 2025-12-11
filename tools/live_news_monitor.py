
import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import Selenium setup from existing module logic
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def calculate_impact(event):
    """
    Calculate Immediate Directional Bias for Gold (XAUUSD).
    Rule: Strong USD Data -> Bearish Gold.
    """
    try:
        actual_str = event.get('actual', '')
        forecast_str = event.get('forecast', '')
        
        if not actual_str or not forecast_str:
            return None
            
        def clean(val):
            val = str(val).lower().replace('k','e3').replace('m','e6').replace('b','e9').replace('%','').replace(',','')
            return float(val)

        act = clean(actual_str)
        fcst = clean(forecast_str)
        
        if act == 0 and fcst == 0: return None
        
        deviation = act - fcst
        
        # Determine Logic
        # Most USD news: Higher = Strong USD = Bearish Gold
        # Unemployment/Claims: Higher = Weak USD = Bullish Gold
        
        is_inverse = "unemployment" in event['event'].lower() or "claims" in event['event'].lower()
        
        if is_inverse:
            # Bad Data (High Unemp) -> Weak USD -> BULLISH Gold
            if deviation > 0: direction = "BULLISH 🟢"
            elif deviation < 0: direction = "BEARISH 🔴"
            else: direction = "NEUTRAL ⚪"
        else:
            # Good Data (High CPI/NFP) -> Strong USD -> BEARISH Gold
            if deviation > 0: direction = "BEARISH 🔴"
            elif deviation < 0: direction = "BULLISH 🟢"
            else: direction = "NEUTRAL ⚪"
            
        return {
            "direction": direction,
            "deviation": deviation
        }
    except:
        return None

def monitor_live_news():
    print("Initializing Live News Monitor...")
    print("Keeping Browser Open for Speed...")
    
    options = Options()
    # options.add_argument("--headless") # Output is cleaner without, but user wants to 'see'? Maybe headless is fine if we print well.
    # Keep headless for speed, user reads terminal.
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_window_size(1920, 1080)
    
    try:
        # Load Today's Calendar
        url = "https://www.forexfactory.com/calendar?day=today"
        print(f"Navigating to: {url}")
        driver.get(url)
        
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("="*60)
            print(f"LIVE NEWS MONITOR | {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            # Refresh
            driver.refresh()
            
            # Scroll to trigger lazy load (Stepwise)
            for i in range(1, 4):
                driver.execute_script(f"window.scrollTo(0, {i*400});")
                time.sleep(1)
            
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2) # Wait
            
            # Locate Table
            try:
                table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "calendar__table")))
                rows = table.find_elements(By.TAG_NAME, "tr")
            except:
                print("Error finding calendar table. Retrying...")
                continue
                
            found_upcoming = False
            
            print(f"{'Time':<10} | {'Event':<40} | {'Actual':<10} | {'Forecast':<10} | {'Signal'}")
            print("-" * 110)
            
            for row in rows:
                try:
                    # Filter USD High Impact Only? Or just display USD.
                    # User wants "at the time of news".
                    txt = row.get_attribute("textContent")
                    if "USD" not in txt: continue
                    
                    # Impact
                    is_high = False
                    spans = row.find_elements(By.TAG_NAME, "span")
                    for s in spans:
                        cls = s.get_attribute("class")
                        if "red" in cls or "High" in cls: # Use simpler string check on class
                            is_high = True
                            break
                    if not is_high: continue
                    
                    # Parse Data
                    try:
                        time_ele = row.find_element(By.CLASS_NAME, "calendar__time")
                        event_time = time_ele.get_attribute("textContent").strip()
                        
                        event_ele = row.find_element(By.CLASS_NAME, "calendar__event-title")
                        event_name = event_ele.get_attribute("textContent").strip()
                        
                        actual_ele = row.find_element(By.CLASS_NAME, "calendar__actual")
                        actual_text = actual_ele.get_attribute("textContent").strip()
                        
                        try:
                            forecast_ele = row.find_element(By.CLASS_NAME, "calendar__forecast")
                            forecast_text = forecast_ele.get_attribute("textContent").strip()
                        except: forecast_text = ""
                        
                    except:
                        continue
                        
                    # Calculate Signal
                    signal = ""
                    if actual_text and forecast_text:
                        imp = calculate_impact({
                            "event": event_name,
                            "actual": actual_text,
                            "forecast": forecast_text
                        })
                        if imp:
                            signal = imp['direction']
                    elif not actual_text:
                        signal = "WAITING..."
                        found_upcoming = True
                        
                    print(f"{event_time:<10} | {event_name:<35} | {actual_text:<10} | {forecast_text:<10} | {signal}")
                    
                except:
                    continue
            
            print("="*60)
            if not found_upcoming:
                print("No pending High-Impact USD events found for remainder of day.")
                print("Monitoring for updates/revisions... (Press Ctrl+C to Stop)")
            else:
                print("Watching for releases... (Refreshing every 30s)")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nMonitor Stopped.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    monitor_live_news()

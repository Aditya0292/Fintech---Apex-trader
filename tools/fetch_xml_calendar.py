
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

def fetch_weekly_xml():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    print(f"Fetching XML Feed from: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        events = []
        for item in root.findall('event'):
            title = item.find('title').text
            country = item.find('country').text
            date_str = item.find('date').text # MM-DD-YYYY
            time_str = item.find('time').text
            impact = item.find('impact').text
            
            # Forecast/Previous might be None
            forecast = item.find('forecast').text if item.find('forecast') is not None else ""
            
            if country == "USD" and (impact == "High" or impact == "Medium"):
                events.append({
                    "Date": date_str,
                    "Time": time_str,
                    "Event": title,
                    "Impact": impact,
                    "Forecast": forecast
                })
                
        # To DataFrame
        df = pd.DataFrame(events)
        print(f"\nFound {len(df)} High/Medium Impact USD Events this week (XML Source):")
        print("="*80)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No events found.")
        print("="*80)
        
        return df
        
    except Exception as e:
        print(f"Error fetching XML: {e}")
        return None

if __name__ == "__main__":
    fetch_weekly_xml()

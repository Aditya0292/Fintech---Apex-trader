
import requests
from xml.etree import ElementTree as ET

url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
try:
    print(f"Fetching {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {resp.status_code}")
    
    root = ET.fromstring(resp.content)
    events = root.findall('event')
    print(f"Fetched {len(events)} events.")
    
    print("\nSample Event (First 3):")
    for i, e in enumerate(events[:3]):
        title = e.find('title').text
        country = e.find('country').text
        date = e.find('date').text
        actual_elem = e.find('actual')
        actual = actual_elem.text if actual_elem is not None else "None"
        print(f"[{i}] {date} | {country} | {title} | Actual: '{actual}'")
        
    print("\nSearching for ANY Actual values...")
    found = False
    for e in events:
        act = e.find('actual')
        if act is not None and act.text:
            print(f"FOUND: {e.find('title').text} -> {act.text}")
            found = True
            break
            
    if not found:
        print("WARNING: No actual values found in XML either.")

except Exception as e:
    print(f"Error: {e}")

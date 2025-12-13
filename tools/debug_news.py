
import requests
import json
from pprint import pprint

url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
try:
    print(f"Fetching {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()
    print(f"Fetched {len(data)} events.")
    
    if data:
        print("First event keys:", data[0].keys())
        print("\nSample Event (First 3):")
        for i in range(min(3, len(data))):
            pprint(data[i])
            
        # Find a past event with actual value potential
        print("\nChecking for 'actual' values in past events...")
        has_actual = False
        for e in data:
            if e.get('actual'):
                print(f"Found Actual: {e['title']} -> {e['actual']}")
                has_actual = True
                break
        
        if not has_actual:
            print("WARNING: No 'actual' values found in any event.")
            
except Exception as e:
    print(f"Error: {e}")

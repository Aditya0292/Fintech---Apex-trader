import time
import requests
import json
import os
import pandas as pd
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger()

class DataProvider:
    """
    Unified Data Provider for APEX Trade AI.
    Features:
    - Multi-tiered fetching (JSON -> XML -> Local Cache)
    - Exponential Backoff & Retry Logic
    - Centralized Caching
    """
    
    def __init__(self):
        self.config = ConfigLoader.get("data")
        self.cache_dir = "data"
        self.news_cache_file = os.path.join(self.cache_dir, "calendar_unified_cache.json")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Connection": "keep-alive"
        }
        
    def _fetch_with_retry(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """Fetch URL with Exponential Backoff"""
        for i in range(retries):
            try:
                logger.info(f"Fetching {url} (Attempt {i+1}/{retries})")
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    return response
                elif response.status_code in [429, 503]:
                    wait_time = (2 ** i) + 1 # 2, 3, 5 seconds...
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP {response.status_code} from {url}")
            
            except Exception as e:
                logger.error(f"Connection Error: {e}")
                
        return None

    def fetch_economic_calendar(self) -> List[Dict]:
        """
        Fetch calendar events with fallback strategy.
        Strategy:
        1. Try Selenium Scraper (Best - Has 'Actual' values)
        2. Try JSON API (Fast, but no 'Actual')
        3. Try XML Feed (Backup)
        4. Load Local Cache (Safety Net)
        """
        # 0. Try Scraper
        try:
             from src.data.news_scraper import NewsScraper
             scraper = NewsScraper()
             data = scraper.fetch_calendar()
             if data:
                 self._save_cache(data)
                 logger.info(f"Successfully scraped {len(data)} events from ForexFactory.")
                 return data
        except Exception as e:
             logger.warning(f"Scraper failed: {e}. Falling back to API.")

        # Sources
        sources = [
            {"type": "json", "url": "https://nfs.faireconomy.media/ff_calendar_thisweek.json"},
            {"type": "xml", "url": "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"}
        ]
        
        for source in sources:
            response = self._fetch_with_retry(source["url"])
            
            if response:
                try:
                    if source["type"] == "json":
                        data = response.json()
                        self._save_cache(data)
                        logger.info(f"Successfully fetched {len(data)} events from JSON source.")
                        return data
                    
                    elif source["type"] == "xml":
                        data = self._parse_xml(response.content)
                        self._save_cache(data)
                        logger.info(f"Successfully fetched {len(data)} events from XML source.")
                        return data
                        
                except Exception as e:
                    logger.error(f"Parsing error for {source['type']}: {e}")
                    continue
        
        # Fallback to Cache
        logger.warning("All network sources failed. Loading from local cache.")
        return self._load_cache()

    def _parse_xml(self, content: bytes) -> List[Dict]:
        """Convert ForexFactory XML to standard list of dicts"""
        events = []
        try:
            root = ET.fromstring(content)
            for item in root.findall('event'):
                # Combine Date and Time
                date_str = item.find('date').text
                time_str = item.find('time').text
                dt_str = f"{date_str} {time_str}"
                
                # Normalize using util (but here we construct dict first)
                
                events.append({
                    "title": item.find('title').text,
                    "country": item.find('country').text,
                    "date": dt_str, 
                    "impact": item.find('impact').text,
                    "forecast": item.find('forecast').text,
                    "previous": item.find('previous').text,
                    "actual": item.find('actual').text if item.find('actual') is not None else "",
                    "status": "released" if item.find('actual') is not None and item.find('actual').text else "scheduled"
                })
        except Exception as e:
            logger.error(f"XML Parse Failed: {e}")
        return events

    def _save_cache(self, data: List[Dict]):
        try:
            with open(self.news_cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Cache Save Failed: {e}")

    def _load_cache(self) -> List[Dict]:
        if os.path.exists(self.news_cache_file):
            try:
                with open(self.news_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Cache Read Failed: {e}")
        return []

    def get_high_impact_usd_news(self) -> pd.DataFrame:
        """
        Get filtered high-impact USD events, formatted for the dashboard.
        """
        raw_events = self.fetch_economic_calendar()
        if not raw_events:
            return pd.DataFrame()
            
        filtered = []
        for e in raw_events:
            if e.get("country") == "USD" and e.get("impact") in ["High", "Medium"]:
                filtered.append({
                    "DateTime": e.get("date"),
                    "Event": e.get("title"),
                    "Impact": e.get("impact"),
                    "Forecast": e.get("forecast") or "",
                    "Actual": e.get("actual", "") # Might be empty in calendar until released
                })
        
        return pd.DataFrame(filtered)

if __name__ == "__main__":
    dp = DataProvider()
    df = dp.get_high_impact_usd_news()
    print("Unified Data Provider Test:")
    print(df.head())

"""
APEX TRADE AI - Currency Strength Meter Scraper
===============================================
Source: https://currencystrengthmeter.org/
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By 
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import re

class CSMScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless") 
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        # Anti-detection
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    def fetch_usd_strength(self):
        """
        Fetches the current USD strength from currencystrengthmeter.org.
        Returns float (0-10 scale assumed, or 0-100).
        Based on CSS height %.
        """
        print("Scraping USD Strength from currencystrengthmeter.org...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
        
        strength = None
        
        try:
            driver.get("https://currencystrengthmeter.org/")
            time.sleep(5) # Wait for animation/render
            
            # Logic: Find 'USD' text -> Parent -> .level -> style height
            # 1. Find all potential USD labels
            # The one inside the bar chart usually has class 'str-container' or similar
            # From debug: Currency Found: USD | Class: str-container
            
            usd_labels = driver.find_elements(By.XPATH, "//*[contains(text(), 'USD')]")
            
            for label in usd_labels:
                # specific check for our target
                # We expect the parent to have a sibling/child with class 'level'
                try:
                    # Let's try 2 levels up to find the container 'bar-cont' usually
                    # Based on debug, the label is likely a sibling of the bar or inside the same container
                    parent = label.find_element(By.XPATH, "..")
                    
                    # Search valid bar in this parent
                    levels = parent.find_elements(By.CLASS_NAME, "level")
                    if not levels:
                        # Try grandparent
                        grandparent = parent.find_element(By.XPATH, "..")
                        levels = grandparent.find_elements(By.CLASS_NAME, "level")
                        
                    if levels:
                        # Found it
                        style = levels[0].get_attribute("style") 
                        # style="height: 40%;"
                        if "height" in style:
                            match = re.search(r"height:\s*([\d\.]+)%", style)
                            if match:
                                val = float(match.group(1))
                                # Convert 40% to 4.0 (standard 0-10 scale) or keep 40?
                                # Most meters are 0-10. 40% height = 4.0.
                                strength = val / 10.0 
                                print(f"  USD Bar Height: {val}% -> Index: {strength}")
                                break
                except:
                    continue
                    
        except Exception as e:
            print(f"CSM Scraping Error: {e}")
            
        finally:
            driver.quit()
            
        if strength is None:
            print("  Warning: Could not detect USD strength. Defaulting to Neutral (5.0).")
            return 5.0
            
        return strength

if __name__ == "__main__":
    csm = CSMScraper()
    val = csm.fetch_usd_strength()
    print(f"Final USD Strength: {val}")

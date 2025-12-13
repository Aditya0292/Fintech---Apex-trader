
import logging
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

logger = logging.getLogger("news_scraper")

class NewsScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
    def fetch_calendar(self):
        logger.info("Initializing Headless Chrome for News Scraping...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=self.options)
        
        events = []
        try:
            url = "https://www.forexfactory.com/calendar"
            logger.info(f"Navigating to {url}...")
            driver.get(url)
            
            # Wait for table
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "calendar__table")))
            
            # Get Rows
            rows = driver.find_elements(By.TAG_NAME, "tr")
            logger.info(f"Found {len(rows)} TR elements.")
            
            for i, row in enumerate(rows):
                # logger.info(f"Row {i} Classes: {row.get_attribute('class')}")
                pass
                
            rows = driver.find_elements(By.XPATH, "//tr[contains(@class, 'calendar__row')]")
            logger.info(f"Found {len(rows)} matching calendar__row.")
            
            if len(rows) == 0:
                 # Debug: Print body to see what we loaded
                 # logger.info(driver.page_source[:2000])
                 pass
            
            current_date = ""
            
            for row in rows:
                try:
                    # Skip blank rows
                    if "calendar__row--new-day" in row.get_attribute("class"):
                         # Date is here usually
                         pass
                    
                    # HTML Check showed: calendar__cell--blank for spacers
                    if "calendar__cell--blank" in row.get_attribute("innerHTML"):
                         continue

                    # Date Extraction
                    # Logic: If row is new-day, try to get date.
                    # Else keep existing.
                    
                    if "calendar__row--new-day" in row.get_attribute("class"):
                         try:
                             date_cell = row.find_element(By.CLASS_NAME, "calendar__date")
                             # .text might be empty if hidden? innerText usually works.
                             date_text = date_cell.get_attribute("innerText").strip()
                             # print(f"DEBUG DATE TEXT (Inner): '{date_text}'")
                             
                             if date_text:
                                 current_date = self._parse_date(date_text)
                                 # print(f"Set Current Date: {current_date}")
                         except Exception as ex:
                             pass
                    
                    if not current_date:
                        # Skip until we have a date state
                        # print(f"No Date State. Row class: {row.get_attribute('class')}")
                        continue               # Currency
                    try:
                        currency = row.find_element(By.CLASS_NAME, "calendar__currency").text.strip()
                    except:
                        currency = "" # Some rows might be detail rows?
                    
                    if not currency: 
                        # print("No Currency.")
                        continue 
                    
                    # print(f"Processing: {current_date} {currency}") 

                    # Time
                    try:
                        time_str = row.find_element(By.CLASS_NAME, "calendar__time").text.strip()
                    except:
                        time_str = "00:00"

                    # Event
                    event = row.find_element(By.CLASS_NAME, "calendar__event").text.strip()

                    # Impact
                    impact = "Low"
                    try:
                        impact_elem = row.find_element(By.CLASS_NAME, "calendar__impact").find_element(By.TAG_NAME, "span")
                        cls = impact_elem.get_attribute("class")
                        if "red" in cls: impact = "High"
                        elif "orange" in cls: impact = "Medium"
                    except:
                        pass

                    # Values
                    actual = row.find_element(By.CLASS_NAME, "calendar__actual").text.strip()
                    forecast = row.find_element(By.CLASS_NAME, "calendar__forecast").text.strip()
                    previous = row.find_element(By.CLASS_NAME, "calendar__previous").text.strip()
                    
                    full_date_str = f"{current_date} {time_str}"
                    
                    events.append({
                        "title": event,
                        "country": currency,
                        "date": full_date_str,
                        "impact": impact,
                        "forecast": forecast,
                        "actual": actual,
                        "previous": previous
                    })
               
                except Exception as e:
                    # Generic row skip
                    if "no such element" not in str(e):
                        print(f"FAILED ROW: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Scraping Failed: {e}")
        finally:
            driver.quit()
            
        logger.info(f"Scraped {len(events)} events.")
        return events

    def _parse_date(self, date_text):
        # Format: "Sun Dec 7" or "Dec 7"
        # We need YYYY-MM-DD
        try:
            # Remove Day Name (Mon, Tue...)
            clean_date = " ".join(date_text.split()[1:]) # "Dec 7"
            
            # Add year
            current_year = datetime.now().year
            dt = datetime.strptime(f"{clean_date} {current_year}", "%b %d %Y")
            
            # Simple rollover check: If Dec data in Jan run, subtract year. If Jan data in Dec run, add year.
            # Ignored for simple "This Week" view.
            
            return dt.strftime("%Y-%m-%d")
        except:
             return datetime.now().strftime("%Y-%m-%d") # Fallback

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    s = NewsScraper()
    data = s.fetch_calendar()
    import pandas as pd
    df = pd.DataFrame(data)
    print(df[df['country'] == 'USD'].head(10))

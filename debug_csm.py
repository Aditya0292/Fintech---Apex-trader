from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By 
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def check_bars_detailed():
    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get("https://currencystrengthmeter.org/")
        time.sleep(5)
        
        # Get the main wrap
        wrap = driver.find_element(By.CLASS_NAME, "bar-wrap")
        
        # Look for the internal bars.
        # Often they are in a structure like <div class="item"> <div class="title">AUD</div> <div class="bar"></div> </div>
        # Let's inspect all 'div' children
        divs = wrap.find_elements(By.TAG_NAME, "div")
        print(f"Total divs in wrap: {len(divs)}")
        
        for i, div in enumerate(divs):
             # check if it text contains a currency
             txt = div.text.strip()
             cls = div.get_attribute("class")
             
             if txt in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]:
                 print(f"Currency Found: {txt} | Class: {cls}")
                 # Look at siblings or checking outerHTML of parent to see context
                 parent = div.find_element(By.XPATH, "..")
                 print(f"  Parent HTML: {parent.get_attribute('outerHTML')[:200]}")
                 
             # Check for style attributes (height/width)
             style = div.get_attribute("style")
             if style and ("height" in style or "width" in style):
                 print(f"  Styled Div found nearby: Class='{cls}' Style='{style}'")

    finally:
        driver.quit()

if __name__ == "__main__":
    check_bars_detailed()

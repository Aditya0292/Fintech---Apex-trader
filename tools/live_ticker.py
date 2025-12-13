import yfinance as yf
import time
import sys
from colorama import Fore, Style, init

init(autoreset=True)

def live_ticker_stream(symbol="GC=F"):
    print(Fore.CYAN + f"Connecting to Live Data for {symbol}...")
    print(Fore.WHITE + "Press [Ctrl+C] to stop.")
    
    ticker = yf.Ticker(symbol)
    
    last_price = 0
    try:
        while True:
            # fast_info is faster/lighter than history
            try:
                # Note: fast_info keys vary by version, lazy loading
                price = ticker.fast_info['last_price']
                # Or regular market price
                # price = ticker.history(period='1d', interval='1m').iloc[-1]['Close']
            except:
                # Fallback
                 df = ticker.history(period='1d', interval='1m')
                 if not df.empty:
                     price = df.iloc[-1]['Close']
                 else:
                     price = 0

            if price != 0:
                color = Fore.GREEN if price > last_price else Fore.RED
                change = price - last_price
                sign = "+" if change > 0 else ""
                
                # Clear line and print
                sys.stdout.write(f"\r{Fore.WHITE}LIVE XAUUSD: {color}${price:,.2f} {Fore.WHITE}({sign}{change:.2f})   ")
                sys.stdout.flush()
                
                last_price = price
            
            time.sleep(2) # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    live_ticker_stream()

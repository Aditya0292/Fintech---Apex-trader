import MetaTrader5 as mt5
import time
from typing import Dict, Optional

class MT5Interface:
    """
    Connects to MetaTrader 5 (MT5) Terminal for Real-Time Data.
    Requires MT5 Terminal to be installed and running.
    """
    
    def __init__(self):
        self.connected = False
        
    def connect(self) -> bool:
        if not mt5.initialize():
            print(f"MT5 Init Failed: {mt5.last_error()}")
            return False
            
        self.connected = True
        return True
        
    def get_live_price(self, symbol: str = "XAUUSD") -> Optional[float]:
        """
        Get the exact live Tick Price (Mid Price)
        Mid = (Bid + Ask) / 2
        Matches TradingView.
        """
        if not self.connected:
            if not self.connect():
                return None
        
        # Check symbol
        # Sometimes it's XAUUSD, XAUUSD.a, GOLD, etc.
        # We try strict match first, then search
        tick = mt5.symbol_info_tick(symbol)
        
        if tick is None:
            # Try finding a matching symbol
            all_symbols = mt5.symbols_get()
            candidates = [s.name for s in all_symbols if "XAU" in s.name or "GOLD" in s.name]
            
            if candidates:
                # Use first candidate
                symbol = candidates[0]
                tick = mt5.symbol_info_tick(symbol)
            else:
                print(f"MT5 Warning: Symbol {symbol} not found.")
                return None
                
        if tick:
            mid_price = (tick.bid + tick.ask) / 2
            return mid_price
        
        return None

    def get_historical_data(self, symbol: str = "XAUUSD", timeframe: int = mt5.TIMEFRAME_H4, num_candles: int = 1000):
        """
        Fetch historical candles from MT5.
        Returns a DataFrame compatible with SMC Analyzer.
        """
        import pandas as pd
        if not self.connected:
            if not self.connect():
                return None
                
        # Copy rates
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
        
        if rates is None or len(rates) == 0:
            print(f"MT5 Error: No rates found for {symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to match our standard (open, high, low, close only)
        # MT5 returns: time, open, high, low, close, tick_volume, spread, real_volume
        # We need lower case
        return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

    def shutdown(self):
        mt5.shutdown()
        self.connected = False

if __name__ == "__main__":
    mt = MT5Interface()
    if mt.connect():
        print("Connected to MT5")
        price = mt.get_live_price("XAUUSD")
        if price:
            print(f"Live Mid Price: {price}")
        mt.shutdown()
    else:
        print("Could not connect to MT5. Ensure Terminal is installed.")

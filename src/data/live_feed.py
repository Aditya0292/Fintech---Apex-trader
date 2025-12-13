
import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime, timedelta
import threading

# Global Cache (Simple In-Memory)
# Structure: { key: (timestamp, dataframe) }
_DATA_CACHE = {}
_CACHE_LOCK = threading.Lock()
CACHE_TTL = 3 # Seconds (Aggressive for live charts)

class LiveFeed:
    """
    Unified Live Data Feed
    Primary: Yahoo Finance (yfinance) - Free, Stable
    Fallback: MetaTrader 5 (MT5) - If running locally
    """
    
    @staticmethod
    def _get_cache_key(symbol, timeframe):
        return f"{symbol}_{timeframe}"

    @staticmethod
    def fetch_live_ohlc(symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLC data with caching and fallback.
        Timeframe map: 1m, 5m, 15m, 1h, 4h, 1d
        """
        key = LiveFeed._get_cache_key(symbol, timeframe)
        now = time.time()
        
        # 1. Check Cache
        with _CACHE_LOCK:
            if key in _DATA_CACHE:
                ts, df = _DATA_CACHE[key]
                if now - ts < CACHE_TTL:
                    return df
        
        # 2. Fetch Data
        df = LiveFeed._fetch_from_yahoo(symbol, timeframe, limit)
        
        if df is None or df.empty:
            print(f"  ⚠️ Yahoo fetch failed for {symbol}. Trying MT5...")
            df = LiveFeed._fetch_from_mt5(symbol, timeframe, limit)
            
        if df is not None and not df.empty:
            # Update Cache
            with _CACHE_LOCK:
                _DATA_CACHE[key] = (now, df)
            return df
        else:
            # Return empty or last known if desperate?
            return pd.DataFrame()

    @staticmethod
    def _fetch_from_yahoo(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch from Yahoo Finance
        """
        try:
            # Map timeframe
            tf_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", 
                "1h": "1h", "4h": "1h", # 4h not supported by yf standard, need to resample (later) or just use 1h and build. 
                                        # YF only supports 1h, then 1d. 4h is tricky.
                                        # For now, let's request 1h and we can resample if needed, 
                                        # OR just let the chart aggregate. 
                "1d": "1d", "daily": "1d"
            }
            yf_tf = tf_map.get(timeframe, "1h")
            
            # Period calculation based on limit/tf?
            # YF period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            period = "1mo" # Safe default
            if timeframe == "1m": period = "5d"
            
            # Ticker adjustment (Yahoo uses specific fmt)
            # Gold: GC=F, XAUUSD=X, XAU-USD?
            ticker_map = {
                "XAUUSD": "GC=F", # Futures often better for volume
                "EURUSD": "EURUSD=X",
                "BTCUSD": "BTC-USD"
            }
            yf_sym = ticker_map.get(symbol, symbol)
            
            # Download
            df = yf.download(tickers=yf_sym, period=period, interval=yf_tf, progress=False, multi_level_index=False)
            
            if df.empty:
                # Try alternate XAU
                df = yf.download(tickers="XAU-USD", period=period, interval=yf_tf, progress=False, multi_level_index=False)
            
            if df.empty: return None
            
            # Normalize Columns
            # YF Cols: Open, High, Low, Close, Adj Close, Volume
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            
            # Rename 'date' or 'datetime' to 'time'
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'time'})
            elif 'datetime' in df.columns:
                df = df.rename(columns={'datetime': 'time'})
                
            # Filter cols
            req_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in req_cols if c in df.columns]]
            
            # Sort
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            
            # Limit
            if len(df) > limit:
                df = df.iloc[-limit:]
                
            return df
            
        except Exception as e:
            print(f"YF Error: {e}")
            return None

    @staticmethod
    def _fetch_from_mt5(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch from local MT5 (Windows Only)
        """
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                return None
            
            # Map TF
            tf_map = {
                "1h": mt5.TIMEFRAME_H1,
                "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1,
                "15m": mt5.TIMEFRAME_M15,
                "5m": mt5.TIMEFRAME_M5,
                "1m": mt5.TIMEFRAME_M1
            }
            mt_tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            rates = mt5.copy_rates_from_pos(symbol, mt_tf, 0, limit)
            if rates is None: return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename for compatibility
            # MT5 cols: time, open, high, low, close, tick_volume, spread, real_volume
            # We want volume -> real_volume or tick_volume?
            # Spot forex usually tick_volume
            df = df.rename(columns={'tick_volume': 'volume'})
            
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
        except ImportError:
             return None
        except Exception as e:
            print(f"MT5 Error: {e}")
            return None

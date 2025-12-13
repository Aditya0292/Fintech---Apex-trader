import pandas as pd
import numpy as np
from datetime import datetime, time

class MicrostructureAnalyzer:
    """
    ICT Microstructure Analyzer.
    Detects:
    - Displacement (High Momentum Candles)
    - Breaker Blocks (Failed OBs)
    - Session Killzones (LO, NYO, PM)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'time' not in self.df.columns:
            self.df['time'] = self.df.index
            
        # Ensure UTC-naive for session checks
        from src.utils.time_utils import normalize_ts
        self.df['time'] = normalize_ts(self.df['time'])

    def detect_displacement(self, threshold_factor: float = 2.0) -> pd.DataFrame:
        """
        Detect displacement candles (body > 2x ATR).
        """
        df = self.df.copy()
        
        # Calculate ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        # Body Size
        body_size = (df['close'] - df['open']).abs()
        
        df['is_displacement'] = (body_size > (atr * threshold_factor))
        df['displacement_dir'] = 0
        df.loc[df['is_displacement'] & (df['close'] > df['open']), 'displacement_dir'] = 1  # Bullish
        df.loc[df['is_displacement'] & (df['close'] < df['open']), 'displacement_dir'] = -1 # Bearish
        
        return df[['time', 'is_displacement', 'displacement_dir']]

    def detect_breaker_blocks(self, obs: list) -> list:
        """
        Identify Breaker Blocks from a list of Order Blocks.
        A Breaker is an OB that was broken and is now valid in the opposite direction.
        Bullish OB -> Broken Down -> Becomes Bearish Breaker (Resistance).
        Bearish OB -> Broken Up -> Becomes Bullish Breaker (Support).
        """
        breakers = []
        df = self.df
        
        for ob in obs:
            # Only check broken OBs
            # Note: smc_analyzer might filter out broken OBs, so we might need RAW OBs.
            # Assuming 'obs' contains broken ones or we re-detect.
            # If we only have active ones, we can't find breakers easily.
            # Ideally this should be integrated in SMC Analyzer or passed broken OBs.
            
            # Logic if we had broken OBs:
            # If Bullish OB broken: check if retested from below.
            pass
            
        return breakers # Placeholder until we update SMC to return broken OBs

    def check_killzone(self, current_time: datetime = None) -> str:
        """
        Check if current time is in a Killzone.
        LO: 02:00 - 05:00 NYC
        NYO: 07:00 - 10:00 NYC
        LC: 13:00 - 16:00 NYC (London Close)
        """
        if current_time is None:
            current_time = datetime.now()
            
        # UTC Times (Assuming input is roughly UTC based on our normalization)
        # London Open: 07:00 - 10:00 UTC (approx)
        # New York Open: 12:00 - 15:00 UTC (approx)
        
        h = current_time.hour
        
        if 7 <= h < 10:
            return "London Open"
        elif 12 <= h < 15:
            return "New York Open"
        elif 15 <= h < 19:
            return "London Close / NY PM"
        
        return "Asian Session / Dead Zone"

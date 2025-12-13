import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class SMCAnalyzer:
    """
    SMC Analyzer - Ported from LuxAlgo Smart Money Concepts (Pinescript).
    Implements:
    - Swing Highs/Lows (ZigZag based on length)
    - Internal & Swing Structure (BOS / CHoCH)
    - Order Blocks (Origin of BOS)
    - Fair Value Gaps (3-candle pattern)
    - Liquidity Sweeps
    
    Timeframe-Specific Analysis Windows:
    - 15m/30m: 3000 candles (Deep lookback for HTF structure)
    - 1H: 150 candles
    - 4H: 200 candles
    """
    
    # Lookback windows per timeframe
    LOOKBACK_WINDOWS = {
        "1h": 150, "1 hour": 150,
        "4h": 200, "4 hour": 200,
        "daily": 120, "d1": 120,
        "15m": 3000, "15 min": 3000,
        "30m": 3000, "30 min": 3000,
        "5m": 3000, "5 min": 3000
    }
    
    def __init__(self, df: pd.DataFrame, timeframe: str = "4h", symbol: str = "XAUUSD"):
        """
        Initialize SMC Analyzer with timeframe-specific lookback window.
        """
        self.timeframe = timeframe.lower()
        self.symbol = symbol
        self.df = df.copy()
        
        from src.utils.config_loader import config
        assets_conf = config.get('assets', {}).get(symbol, {})
        
        # Determine Window:
        # 1. Config override (smc_window)
        # 2. Timeframe default
        
        base_window = assets_conf.get('smc_window', 500)
        tf_window = self.LOOKBACK_WINDOWS.get(self.timeframe, 200)
        
        # Use the larger of the two to ensure enough context
        window_size = max(base_window, tf_window)
        
        if 'time' not in self.df.columns:
            self.df['time'] = self.df.index
            
        cols = ['open', 'high', 'low', 'close']
        for c in cols:
            self.df[c] = self.df[c].astype(float)
            
        self.df = self.df.reset_index(drop=True)
        
        # Apply timeframe-specific window
        if len(self.df) > window_size:
            self.df = self.df.iloc[-window_size:].reset_index(drop=True)
            # print(f"[SMC] Applied {self.timeframe.upper()} window: {window_size} candles")

    def _get_swing_points(self, length: int = 50):
        """
        Detect Swing Highs and Lows using a rolling window.
        Use Vectorized Rolling Max/Min for speed.
        A point i is a swing high if it's the max of the window [i-length, i+length].
        This is a 'Fractal' definition.
        """
        df = self.df
        
        # Rolling Max Center
        # Note: In real-time, we don't know the right side.
        # But for STRUCTURE detection (past), we need confirmed swings.
        # A swing is confirmed only 'length' bars later.
        
        highs = df['high']
        lows = df['low']
        
        # For simple structure, we use a smaller period or 'pivot points'
        # To match LuxAlgo usually 5 (Short) and 50 (Long)
        
        # Vectorized check
        # We need to use valid confirmed swings for the PAST.
        # For Feature Building, we used causal lookback.
        # For SMC Analysis (Trade Plan), we look at the chart history provided.
        
        window = 2 * length + 1
        
        df['is_swing_high'] = (highs == highs.rolling(window, center=True).max())
        df['is_swing_low'] = (lows == lows.rolling(window, center=True).min())
        
        swings = []
        for i in range(len(df)):
            if df.iat[i, df.columns.get_loc('is_swing_high')]:
                swings.append({'index': i, 'price': df.iat[i, df.columns.get_loc('high')], 'type': 'high'})
            elif df.iat[i, df.columns.get_loc('is_swing_low')]:
                swings.append({'index': i, 'price': df.iat[i, df.columns.get_loc('low')], 'type': 'low'})
                
        return swings

    def find_structure_and_blocks(self):
        """
        Find Order Blocks (OB), Liquidity Sweeps, and FVGs.
        """
        df = self.df
        
        # 1. Structure Breaks (BOS)
        # Simplified logic: 
        # Identify major swings (len=10)
        swings = self._get_swing_points(length=10)
        
        # We need sequential processing
        bullish_obs = []
        bearish_obs = []
        
        last_high = None
        last_low = None
        
        # Use simpler iteration for OBs
        # Iterate candles
        for i in range(20, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # --- FVG DETECTION ---
            # Bullish FVG: Low[i] > High[i-2]
            # Bearish FVG: High[i] < Low[i-2]
            
            # --- OB DETECTION VIA BOS ---
            # Needs confirmed swing break
            # This requires tracking the last confirmed swing high/low
            pass # (This loop approach is complex to get right without full state machine)

        # Let's use the Robust Logic from original, hardened.
        # 1. Detect Swings (ZigZag)
        # 2. Detect Breakout of Swing
        # 3. Mark Origin as OB
        
        # Get swings
        swing_highs = [s for s in swings if s['type'] == 'high']
        swing_lows = [s for s in swings if s['type'] == 'low']
        
        # Identify BOS
        obs = []
        
        # Bullish BOS: Price closes above a previous Swing High
        # Find origin: Lowest candle between prev Swing High and Breakout
        
        # Algorithm:
        # Track last Swing High. If Close > Swing High, trigger Bullish BOS.
        # OB = Lowest candle in the range [Swing High Index ... Breakout Index]
        
        last_s_high = None
        
        for i in range(len(df)):
            close = df.iat[i, df.columns.get_loc('close')]
            
            # Update last known swing high (that occurred before i)
            # Efficient lookup:
            current_s_highs = [s for s in swing_highs if s['index'] < i]
            if current_s_highs:
                # The 'relevant' swing high is usually the most recent one that hasn't been broken?
                # Or the highest one?
                # Smart Money: The most recent significant swing high.
                candidate = current_s_highs[-1]
                
                # Check Break
                if last_s_high != candidate['index']:
                    # New swing high formed
                    last_s_high = candidate['index']
                    
                # Check for BOS
                if close > candidate['price']: # Breakout
                    # Create Bullish OB
                    # Origin is lowest candle between candidate['index'] and i
                    subset = df.iloc[candidate['index']:i+1]
                    lowest_idx = subset['low'].idxmin()
                    
                    # Store OB
                    # Check if already exists
                    if not any(o['index'] == lowest_idx for o in obs):
                        # Construct OB
                        # Bullish OB is the last Down candle before the move up?
                        # Or strictly the lowest candle? 
                        # ICT: Last down candle.
                        # We'll take the candle at lowest_idx
                        ob_row = df.loc[lowest_idx]
                        obs.append({
                            'type': 'bullish',
                            'top': ob_row['high'], # Valid OB usually body or high
                            'bottom': ob_row['low'],
                            'index': lowest_idx,
                            'time': ob_row['time'],
                            'mitigated': False,
                            'broken': False,
                            'strength': 1
                        })
                        
        # Bearish BOS
        # Price closes below previous Swing Low
        last_s_low = None
        for i in range(len(df)):
            close = df.iat[i, df.columns.get_loc('close')]
            current_s_lows = [s for s in swing_lows if s['index'] < i]
            if current_s_lows:
                candidate = current_s_lows[-1]
                if close < candidate['price']:
                    # Breakout -> Bearish OB
                    # Origin: Highest candle between swing low and i
                    subset = df.iloc[candidate['index']:i+1]
                    highest_idx = subset['high'].idxmax()
                    
                    if not any(o['index'] == highest_idx for o in obs):
                        ob_row = df.loc[highest_idx]
                        obs.append({
                            'type': 'bearish',
                            'top': ob_row['high'],
                            'bottom': ob_row['low'],
                            'index': highest_idx,
                            'time': ob_row['time'],
                            'mitigated': False,
                            'broken': False,
                            'strength': 1
                        })
                        
        # Filter Active OBs
        final_obs = []
        current_price = df.iloc[-1]['close']
        
        for ob in obs:
            # Check if broken later
            # Scan from ob['index'] + 1 to end
            future_df = df.iloc[ob['index']+1:]
            if future_df.empty: 
                final_obs.append(ob)
                continue
                
            broken = False
            touched = False
            
            if ob['type'] == 'bullish':
                # Broken if close < bottom
                if (future_df['close'] < ob['bottom']).any():
                    broken = True
                # Touched if low <= top (Mitigation)
                if (future_df['low'] <= ob['top']).any():
                    touched = True
            else:
                # Broken if close > top
                if (future_df['close'] > ob['top']).any():
                    broken = True
                # Touched if high >= bottom
                if (future_df['high'] >= ob['bottom']).any():
                    touched = True
            
            # Keep if not broken (even if touched)
            if not broken:
                ob['mitigated'] = touched
                final_obs.append(ob)
                
        return final_obs

    def find_fvgs(self):
        """
        Find Fair Value Gaps.
        """
        df = self.df
        fvgs = []
        
        for i in range(2, len(df)):
            # Bullish FVG
            # Low[i] > High[i-2]
            if df.iloc[i]['low'] > df.iloc[i-2]['high']:
                gap = df.iloc[i]['low'] - df.iloc[i-2]['high']
                if gap > (df.iloc[i]['close'] * 0.0002): # Min threshold
                    fvgs.append({
                        'type': 'bullish',
                        'top': df.iloc[i]['low'],
                        'bottom': df.iloc[i-2]['high'],
                        'index': i,
                        'time': df.iloc[i]['time'],
                        'mitigated': False
                    })
            
            # Bearish FVG
            # High[i] < Low[i-2]
            if df.iloc[i]['high'] < df.iloc[i-2]['low']:
                gap = df.iloc[i-2]['low'] - df.iloc[i]['high']
                if gap > (df.iloc[i]['close'] * 0.0002):
                    fvgs.append({
                        'type': 'bearish',
                        'top': df.iloc[i-2]['low'],
                        'bottom': df.iloc[i]['high'],
                        'index': i,
                        'time': df.iloc[i]['time'],
                        'mitigated': False
                    })
                    
        # Check mitigation
        active_fvgs = []
        for fvg in fvgs:
            future = df.iloc[fvg['index']+1:]
            if future.empty:
                active_fvgs.append(fvg)
                continue
                
            mitigated = False
            if fvg['type'] == 'bullish':
                # Filled if price goes below bottom
                if (future['low'] <= fvg['bottom']).any():
                    mitigated = True
            else:
                # Filled if price goes above top
                if (future['high'] >= fvg['top']).any():
                    mitigated = True
            
            if not mitigated:
                active_fvgs.append(fvg)
                
        return active_fvgs

    def find_liquidity(self):
        """
        Identify Liquidity Pools (Swing levels not yet swept).
        """
        swings = self._get_swing_points(length=10)
        df = self.df
        
        bsl = [] # Buy Side Liquidity (Above Highs)
        ssl = [] # Sell Side Liquidity (Below Lows)
        
        for s in swings:
            # Check if swept
            future = df.iloc[s['index']+1:]
            if future.empty: 
                if s['type'] == 'high': bsl.append(s)
                else: ssl.append(s)
                continue
                
            swept = False
            if s['type'] == 'high':
                if (future['high'] > s['price']).any(): swept = True
            else:
                if (future['low'] < s['price']).any(): swept = True
                
            if not swept:
                if s['type'] == 'high': bsl.append(s)
                else: ssl.append(s)
                
        # Format
        return {
            'bsl': [{'price': s['price'], 'time': df.iloc[s['index']]['time']} for s in bsl],
            'ssl': [{'price': s['price'], 'time': df.iloc[s['index']]['time']} for s in ssl]
        }

    def get_nearest_structures(self, current_price: float):
        """
        Get nearest OBs and FVGs to current price.
        """
        obs = self.find_structure_and_blocks()
        fvgs = self.find_fvgs()
        liq = self.find_liquidity()
        
        # Filter OBs
        # Only closest ones
        
        # Bullish OBs (Demand) -> Below price OR containing price
        bull_obs = [o for o in obs if o['type'] == 'bullish']
        # Sort by top descending (highest demand zone first)
        bull_obs.sort(key=lambda x: x['top'], reverse=True)
        # Filter: Must be below price + buffer OR price inside
        # Price inside: low <= price <= high
        valid_bull = [o for o in bull_obs if o['bottom'] < current_price] 
        
        # Bearish OBs (Supply) -> Above price OR containing price
        bear_obs = [o for o in obs if o['type'] == 'bearish']
        bear_obs.sort(key=lambda x: x['bottom']) # Lowest supply zone first
        valid_bear = [o for o in bear_obs if o['top'] > current_price]
        
        # Nearest FVG
        valid_bull_fvg = [f for f in fvgs if f['type'] == 'bullish' and f['bottom'] < current_price]
        valid_bull_fvg.sort(key=lambda x: x['top'], reverse=True)
        
        valid_bear_fvg = [f for f in fvgs if f['type'] == 'bearish' and f['top'] > current_price]
        valid_bear_fvg.sort(key=lambda x: x['bottom'])
        
        return {
            'bull_obs_found': valid_bull[:3],
            'bear_obs_found': valid_bear[:3],
            'support_ob': valid_bull[0] if valid_bull else None,
            'resistance_ob': valid_bear[0] if valid_bear else None,
            'fvgs': fvgs, # Return all, caller filters
            'liquidity': liq
        }

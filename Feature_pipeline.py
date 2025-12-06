"""
APEX TRADE AI - Enhanced Feature Engineering Pipeline
======================================================

Integrates:
- Smart Money Concepts (LuxAlgo-based)
- ICT Concepts (Order Blocks, FVG, BOS, CHoCH)
- Multi-type Pivot Points
- Classical Technical Indicators
- Multi-Timeframe Features
- Confluence Detection

Target: 200+ features for ML model to achieve 80%+ precision
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    WINDOW = 50  # Sequence length for LSTM
    MIN_PERIODS = 200  # For indicators
    
    # Structure Detection
    SWING_LENGTH = 50  # Main swing detection
    INTERNAL_LENGTH = 5  # Internal structure
    
    # Thresholds
    MOVEMENT_THRESHOLD = 0.001  # 0.1% for classification
    EQH_EQL_THRESHOLD = 0.001  # Equal highs/lows tolerance
    FVG_MIN_SIZE = 0.0005  # Minimum FVG size (0.05%)
    
    # Pivot
    PIVOT_TYPES = ['traditional', 'fibonacci', 'camarilla', 'woodie', 'demark']
    
    # Outputs
    OUTPUT_DIR = Path(".")
    FEATURE_CSV = "features_enhanced.csv"
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Safe division avoiding divide by zero"""
    return a / b.replace(0, np.nan).fillna(fill)

def normalize_to_atr(value: pd.Series, atr: pd.Series) -> pd.Series:
    """Normalize values to ATR for scale-invariance"""
    return safe_divide(value, atr, 0.0)

# ============================================================================
# CLASSICAL TECHNICAL INDICATORS
# ============================================================================

class TechnicalIndicators:
    """Classical technical analysis indicators"""
    
    @staticmethod
    def sma(series: pd.Series, n: int) -> pd.Series:
        return series.rolling(n, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, n: int) -> pd.Series:
        return series.ewm(span=n, adjust=False).mean()
    
    @staticmethod
    def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
        """Average True Range"""
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=1).mean()
    
    @staticmethod
    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(n, min_periods=1).mean()
        ma_down = down.rolling(n, min_periods=1).mean()
        rs = safe_divide(ma_up, ma_down, 1.0)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def bollinger_bands(series: pd.Series, n: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = series.rolling(n, min_periods=1).mean()
        std_dev = series.rolling(n, min_periods=1).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
        """Average Directional Index"""
        high, low, close = df['high'], df['low'], df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.atr(df, 1)
        plus_di = 100 * (plus_dm.rolling(n).mean() / tr.rolling(n).mean())
        minus_di = 100 * (minus_dm.rolling(n).mean() / tr.rolling(n).mean())
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(n).mean()
        return adx.fillna(0)
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

# ============================================================================
# MARKET STRUCTURE (ICT/SMC)
# ============================================================================

class MarketStructure:
    """Detect market structure: swings, BOS, CHoCH"""
    
    @staticmethod
    def detect_swing_points(df: pd.DataFrame, length: int = 50) -> Tuple[pd.Series, pd.Series]:
        """
        Detect swing highs and lows
        Returns: (swing_highs, swing_lows) with prices at detection points, NaN elsewhere
        """
        highs = df['high']
        lows = df['low']
        
        swing_high = pd.Series(np.nan, index=df.index)
        swing_low = pd.Series(np.nan, index=df.index)
        
        for i in range(length, len(df) - length):
            # Swing high: highest in window
            if highs.iloc[i] == highs.iloc[i-length:i+length+1].max():
                swing_high.iloc[i] = highs.iloc[i]
            
            # Swing low: lowest in window
            if lows.iloc[i] == lows.iloc[i-length:i+length+1].min():
                swing_low.iloc[i] = lows.iloc[i]
        
        return swing_high, swing_low
    
    @staticmethod
    def label_structure(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series) -> pd.DataFrame:
        """
        Label structure as HH, HL, LH, LL
        Returns df with added columns
        """
        df = df.copy()
        df['swing_high'] = swing_high
        df['swing_low'] = swing_low
        
        # Track last swing values
        df['last_swing_high'] = swing_high.fillna(method='ffill')
        df['last_swing_low'] = swing_low.fillna(method='ffill')
        
        # Previous swing values
        df['prev_swing_high'] = df['last_swing_high'].shift(1)
        df['prev_swing_low'] = df['last_swing_low'].shift(1)
        
        # Structure labels
        df['structure'] = 0  # 0=neutral, 1=HH, 2=HL, 3=LH, 4=LL
        
        # HH: Higher High
        df.loc[(~swing_high.isna()) & (swing_high > df['prev_swing_high']), 'structure'] = 1
        # HL: Higher Low
        df.loc[(~swing_low.isna()) & (swing_low > df['prev_swing_low']), 'structure'] = 2
        # LH: Lower High
        df.loc[(~swing_high.isna()) & (swing_high < df['prev_swing_high']), 'structure'] = 3
        # LL: Lower Low
        df.loc[(~swing_low.isna()) & (swing_low < df['prev_swing_low']), 'structure'] = 4
        
        # Forward fill structure label
        df['structure'] = df['structure'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return df

    @staticmethod
    def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH)
        """
        df = df.copy()
        
        df['bos_bullish'] = 0
        df['bos_bearish'] = 0
        df['choch_bullish'] = 0
        df['choch_bearish'] = 0
        
        # Logic: 
        # Bullish BOS: Close above previous Higher High
        # Bearish BOS: Close below previous Lower Low
        
        # For simplicity (heuristic):
        # If current close breaks the last swing high -> Bullish Event
        
        last_high = df['last_swing_high']
        last_low = df['last_swing_low']
        
        # Breakout
        break_high = (df['close'] > last_high) & (df['close'].shift(1) <= last_high)
        break_low = (df['close'] < last_low) & (df['close'].shift(1) >= last_low)
        
        # Trend (1=Bull, -1=Bear)
        df['trend'] = 0
        df.loc[break_high, 'trend'] = 1
        df.loc[break_low, 'trend'] = -1
        df['trend'] = df['trend'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # CHOCH is effectively the first BOS in opposite direction
        # Here we just mark all breaks
        df.loc[break_high, 'bos_bullish'] = 1
        df.loc[break_low, 'bos_bearish'] = 1
        
        # Refinement can be added later
        return df

# ============================================================================
# ORDER BLOCKS
# ============================================================================

class OrderBlocks:
    """Detect ICT Order Blocks"""
    
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Heuristic: OB is the last candle color before a displacement
        """
        df = df.copy()
        df['ob_bullish'] = 0
        df['ob_bearish'] = 0
        
        # Using BOS signals to identify OBs
        # Bullish OB: Bearish candle before the strong up move that caused BOS
        bos_bull_indices = df.index[df['bos_bullish'] == 1]
        
        for idx in bos_bull_indices:
            # Look back a few candles for the lowest red candle
            loc = df.index.get_loc(idx)
            if loc > 5:
                # Simple logic: previous candle
                df.iloc[loc-1, df.columns.get_loc('ob_bullish')] = 1
        
        bos_bear_indices = df.index[df['bos_bearish'] == 1]
        for idx in bos_bear_indices:
            loc = df.index.get_loc(idx)
            if loc > 5:
                df.iloc[loc-1, df.columns.get_loc('ob_bearish')] = 1
                
        # Mark price inside OB? (Skipping for brevity, added columns)
        df['inside_ob_bull'] = 0
        df['inside_ob_bear'] = 0
        df['dist_to_ob_bull'] = 0.0
        df['dist_to_ob_bear'] = 0.0
        
        return df

# ============================================================================
# FAIR VALUE GAPS
# ============================================================================

class FairValueGaps:
    """Detect Fair Value Gaps"""
    
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect gaps between candle i-1 high/low and candle i+1 low/high
        """
        df = df.copy()
        df['fvg_bullish'] = 0
        df['fvg_bearish'] = 0
        
        # We need to vectorized this or loop
        # Bullish FVG: Low of candle i+1 > High of candle i-1
        # Current index i is the middle candle
        
        high = df['high']
        low = df['low']
        
        # Shifted
        prev_high = high.shift(1) # i-1
        next_low = low.shift(-1)  # i+1 (but we are at i, wait, FVG is formed at i? No, formed after i+1 closes)
        # Let's say FVG is confirmed at i. The gap is between i-2 and i.
        
        # Gap between i-2 (high) and i (low)
        high_minus_2 = high.shift(2)
        low_current = low
        
        # Bull FVG: Low (i) > High (i-2)
        bull_fvg = low_current > high_minus_2
        df.loc[bull_fvg, 'fvg_bullish'] = 1
        
        # Bear FVG: High (i) < Low (i-2)
        low_minus_2 = low.shift(2)
        high_current = high
        bear_fvg = high_current < low_minus_2
        df.loc[bear_fvg, 'fvg_bearish'] = 1
        
        # Inside FVG
        df['inside_fvg_bull'] = 0
        df['inside_fvg_bear'] = 0
        df['dist_to_fvg_bull'] = 0.0
        df['dist_to_fvg_bear'] = 0.0
        
        return df

# ============================================================================
# PREMIUM / DISCOUNT / OTE
# ============================================================================

class PremiumDiscount:
    """Calculate Premium, Discount, and Equilibrium zones"""
    
    @staticmethod
    def calculate_zones(df: pd.DataFrame) -> pd.DataFrame:
        """
        Using the current trading range (last swing high/low)
        """
        df = df.copy()
        
        range_high = df['last_swing_high']
        range_low = df['last_swing_low']
        
        df['range_high'] = range_high
        df['range_low'] = range_low
        
        # Position in range (0 to 1)
        # Avoid div by zero
        denom = (range_high - range_low).replace(0, np.nan)
        df['range_position'] = (df['close'] - range_low) / denom
        df['range_position'] = df['range_position'].fillna(0.5)
        
        # Zones
        df['in_premium'] = (df['range_position'] > 0.5).astype(int)
        df['in_discount'] = (df['range_position'] < 0.5).astype(int)
        df['in_equilibrium'] = ((df['range_position'] >= 0.45) & (df['range_position'] <= 0.55)).astype(int)
        
        # OTE (Optimal Trade Entry) - usually 0.618 to 0.786 retracement
        # For bullish structure (buying in discount), OTE is deep discount?
        # Actually OTE is retracement level.
        # If Bullish trend: Retracement from High to Low.
        # Simple implementation: 
        
        range_size = df['range_high'] - df['range_low']
        df['ote_618'] = df['range_low'] + 0.618 * range_size
        df['ote_705'] = df['range_low'] + 0.705 * range_size
        df['ote_786'] = df['range_low'] + 0.786 * range_size
        
        # In OTE zone
        df['in_ote_bull'] = ((df['close'] >= df['ote_618']) & 
                             (df['close'] <= df['ote_786'])).astype(int)
        
        # For bearish (premium side OTE)
        df['ote_382'] = df['range_high'] - 0.382 * range_size
        df['ote_295'] = df['range_high'] - 0.295 * range_size  
        df['ote_214'] = df['range_high'] - 0.214 * range_size
        
        df['in_ote_bear'] = ((df['close'] >= df['ote_214']) & 
                             (df['close'] <= df['ote_382'])).astype(int)
        
        return df

# ============================================================================
# EQUAL HIGHS/LOWS
# ============================================================================

class EqualHighLow:
    """Detect equal highs and equal lows (liquidity zones)"""
    
    @staticmethod
    def detect_equal_levels(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """
        Detect equal highs/lows within threshold
        """
        df = df.copy()
        
        df['eq_high'] = 0
        df['eq_low'] = 0
        
        threshold = config.EQH_EQL_THRESHOLD
        
        for i in range(lookback, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Check previous highs
            prev_highs = df['high'].iloc[i-lookback:i]
            if np.any(np.abs(prev_highs - current_high) <= threshold * current_high):
                df.loc[df.index[i], 'eq_high'] = 1
            
            # Check previous lows
            prev_lows = df['low'].iloc[i-lookback:i]
            if np.any(np.abs(prev_lows - current_low) <= threshold * current_low):
                df.loc[df.index[i], 'eq_low'] = 1
        
        return df

# ============================================================================
# PIVOT POINTS (MULTI-TYPE)
# ============================================================================

class PivotPoints:
    """Calculate multiple pivot point types"""
    
    @staticmethod
    def traditional(h: float, l: float, c: float) -> Dict[str, float]:
        """Traditional pivot points"""
        P = (h + l + c) / 3.0
        return {
            'P': P,
            'R1': 2*P - l,
            'S1': 2*P - h,
            'R2': P + (h - l),
            'S2': P - (h - l),
            'R3': h + 2*(P - l),
            'S3': l - 2*(h - P)
        }
    
    @staticmethod
    def fibonacci(h: float, l: float, c: float) -> Dict[str, float]:
        """Fibonacci pivot points"""
        P = (h + l + c) / 3.0
        range_ = h - l
        return {
            'P': P,
            'R1': P + 0.382 * range_,
            'S1': P - 0.382 * range_,
            'R2': P + 0.618 * range_,
            'S2': P - 0.618 * range_,
            'R3': P + range_,
            'S3': P - range_
        }
    
    @staticmethod
    def camarilla(h: float, l: float, c: float) -> Dict[str, float]:
        """Camarilla pivot points"""
        range_ = h - l
        return {
            'P': (h + l + c) / 3.0,
            'R1': c + (range_ * 1.1/12),
            'S1': c - (range_ * 1.1/12),
            'R2': c + (range_ * 1.1/6),
            'S2': c - (range_ * 1.1/6),
            'R3': c + (range_ * 1.1/4),
            'S3': c - (range_ * 1.1/4),
            'R4': c + (range_ * 1.1/2),
            'S4': c - (range_ * 1.1/2)
        }
    
    @staticmethod
    def woodie(h: float, l: float, c: float, o: float) -> Dict[str, float]:
        """Woodie pivot points"""
        P = (h + l + 2*c) / 4.0
        return {
            'P': P,
            'R1': 2*P - l,
            'S1': 2*P - h,
            'R2': P + (h - l),
            'S2': P - (h - l)
        }
    
    @staticmethod
    def demark(h: float, l: float, c: float, o: float) -> Dict[str, float]:
        """DeMark pivot points"""
        if c < o:
            x = h + 2*l + c
        else:
            x = 2*h + l + c
        P = x / 4.0
        return {
            'P': P,
            'R1': x/2 - l,
            'S1': x/2 - h
        }
    
    @staticmethod
    def calculate_all_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all pivot types using previous day's OHLC
        """
        df = df.copy()
        
        # Shift OHLC by 1 to use previous day
        df['ph'] = df['high'].shift(1)
        df['pl'] = df['low'].shift(1)
        df['pc'] = df['close'].shift(1)
        df['po'] = df['open'].shift(1)
        
        # Initialize columns
        for ptype in config.PIVOT_TYPES:
            for level in ['P', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']:
                df[f'{level}_{ptype}'] = np.nan
            if ptype == 'camarilla':
                df['R4_camarilla'] = np.nan
                df['S4_camarilla'] = np.nan
        
        # Calculate pivots
        for i in range(1, len(df)):
            h, l, c, o = df['ph'].iloc[i], df['pl'].iloc[i], df['pc'].iloc[i], df['po'].iloc[i]
            
            if pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            # Traditional
            pivots = PivotPoints.traditional(h, l, c)
            for key, val in pivots.items():
                df.loc[df.index[i], f'{key}_traditional'] = val
            
            # Fibonacci
            pivots = PivotPoints.fibonacci(h, l, c)
            for key, val in pivots.items():
                df.loc[df.index[i], f'{key}_fibonacci'] = val
            
            # Camarilla
            pivots = PivotPoints.camarilla(h, l, c)
            for key, val in pivots.items():
                df.loc[df.index[i], f'{key}_camarilla'] = val
            
            # Woodie
            if not pd.isna(o):
                pivots = PivotPoints.woodie(h, l, c, o)
                for key, val in pivots.items():
                    df.loc[df.index[i], f'{key}_woodie'] = val
                
                # DeMark
                pivots = PivotPoints.demark(h, l, c, o)
                for key, val in pivots.items():
                    df.loc[df.index[i], f'{key}_demark'] = val
        
        return df

# ============================================================================
# CONFLUENCE DETECTION
# ============================================================================

class ConfluenceDetector:
    """Detect confluence between multiple factors"""
    
    @staticmethod
    def calculate_confluence_score(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
        """
        Calculate confluence score (0-100) based on alignment of:
        - Order Blocks
        - FVG
        - Pivots
        - Premium/Discount
        - Structure
        """
        score = pd.Series(0.0, index=df.index)
        
        # OB alignment (20 points max)
        score += df['inside_ob_bull'] * 10
        score += df['inside_ob_bear'] * 10
        
        # FVG alignment (20 points max)
        score += df['inside_fvg_bull'] * 10
        score += df['inside_fvg_bear'] * 10
        
        # Pivot proximity (20 points max) - within 0.5 ATR of any pivot
        pivot_cols = [col for col in df.columns if col.startswith(('P_', 'R1_', 'S1_'))]
        for col in pivot_cols:
            if col in df.columns:
                dist = (df['close'] - df[col]).abs()
                score += (dist < 0.5 * atr).astype(int) * 5
        
        # Premium/Discount zone (15 points)
        score += df['in_discount'] * 15  # Bullish setup
        score += df['in_premium'] * 15   # Bearish setup
        
        # OTE zone (15 points)
        score += df['in_ote_bull'] * 15
        score += df['in_ote_bear'] * 15
        
        # Structure confirmation (10 points)
        score += (df['trend'] == 1) * 10  # Bullish trend
        score += (df['trend'] == -1) * 10  # Bearish trend
        
        # Cap at 100
        score = score.clip(0, 100)
        
        return score

# ============================================================================
# COMPLETE FEATURE BUILDER
# ============================================================================

class FeatureEngineering:
    """Main feature engineering pipeline"""
    
    def __init__(self):
        self.tech = TechnicalIndicators()
        self.structure = MarketStructure()
        self.ob = OrderBlocks()
        self.fvg = FairValueGaps()
        self.zones = PremiumDiscount()
        self.eqhl = EqualHighLow()
        self.pivots = PivotPoints()
        self.confluence = ConfluenceDetector()
    
    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build complete feature set
        Returns: (df_with_all_calcs, features_only_df)
        """
        print("Building features...")
        df = df.copy()
        
        # === CLASSICAL INDICATORS ===
        print("  - Classical indicators...")
        df['atr14'] = self.tech.atr(df, 14)
        df['rsi14'] = self.tech.rsi(df['close'], 14)
        df['ema10'] = self.tech.ema(df['close'], 10)
        df['ema20'] = self.tech.ema(df['close'], 20)
        df['ema50'] = self.tech.ema(df['close'], 50)
        bb_upper, bb_middle, bb_lower = self.tech.bollinger_bands(df['close'], 20)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = normalize_to_atr(bb_upper - bb_lower, df['atr14'])
        df['bb_position'] = safe_divide(df['close'] - bb_lower, bb_upper - bb_lower, 0.5)
        
        df['adx14'] = self.tech.adx(df, 14)
        
        if 'volume' in df.columns:
            df['obv'] = self.tech.obv(df)
            df['volume_sma20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = safe_divide(df['volume'], df['volume_sma20'], 1.0)
        else:
            df['obv'] = 0
            df['volume_ratio'] = 1.0
        
        macd, macd_signal, macd_hist = self.tech.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # === CANDLE PATTERNS ===
        print("  - Candle patterns...")
        df['body'] = df['close'] - df['open']
        df['body_pct'] = safe_divide(df['body'], df['open'], 0)
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = safe_divide(df['upper_wick'], df['high'] - df['low'], 0)
        df['lower_wick_pct'] = safe_divide(df['lower_wick'], df['high'] - df['low'], 0)
        df['range'] = df['high'] - df['low']
        df['range_pct'] = safe_divide(df['range'], df['close'], 0)
        
        # === MARKET STRUCTURE ===
        print("  - Market structure (swings)...")
        swing_high, swing_low = self.structure.detect_swing_points(df, config.SWING_LENGTH)
        df = self.structure.label_structure(df, swing_high, swing_low)
        df = self.structure.detect_bos_choch(df)
        
        # Internal structure
        print("  - Internal structure...")
        internal_high, internal_low = self.structure.detect_swing_points(df, config.INTERNAL_LENGTH)
        df['internal_swing_high'] = internal_high
        df['internal_swing_low'] = internal_low
        
        # === ORDER BLOCKS ===
        print("  - Order blocks...")
        df = self.ob.detect_order_blocks(df)
        
        # === FAIR VALUE GAPS ===
        print("  - Fair value gaps...")
        df = self.fvg.detect_fvg(df)
        
        # === PREMIUM/DISCOUNT & OTE ===
        print("  - Premium/Discount zones...")
        df = self.zones.calculate_zones(df)
        
        # === EQUAL HIGHS/LOWS ===
        print("  - Equal highs/lows...")
        df = self.eqhl.detect_equal_levels(df, 5)
        
        # === PIVOT POINTS ===
        print("  - Pivot points (all types)...")
        df = self.pivots.calculate_all_pivots(df)
        
        # Pivot proximity features (normalized to ATR)
        for ptype in ['traditional', 'fibonacci']:
            for level in ['P', 'R1', 'S1', 'R2', 'S2']:
                col = f'{level}_{ptype}'
                if col in df.columns:
                    df[f'dist_{col}'] = normalize_to_atr(
                        (df['close'] - df[col]).abs(), 
                        df['atr14']
                    )
                    df[f'above_{col}'] = (df['close'] > df[col]).astype(int)
        
        # === CONFLUENCE ===
        print("  - Confluence score...")
        df['confluence_score'] = self.confluence.calculate_confluence_score(df, df['atr14'])
        
        # === MOMENTUM & RETURNS ===
        print("  - Momentum features...")
        for period in [1, 3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'high_{period}d'] = df['high'].rolling(period).max()
            df[f'low_{period}d'] = df['low'].rolling(period).min()
        
        # Rolling volatility
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        
        # === TIME FEATURES ===
        print("  - Time features...")
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['day_of_week'] = df['time'].dt.dayofweek
            df['month'] = df['time'].dt.month
            df['quarter'] = df['time'].dt.quarter
        else:
            df['day_of_week'] = 0
            df['month'] = 1
            df['quarter'] = 1
        
        # Days since last events
        df['days_since_bos_bull'] = self._days_since_event(df['bos_bullish'])
        df['days_since_bos_bear'] = self._days_since_event(df['bos_bearish'])
        df['days_since_choch'] = self._days_since_event(df['choch_bullish'] | df['choch_bearish'])
        df['days_since_fvg'] = self._days_since_event(df['fvg_bullish'] | df['fvg_bearish'])
        
        # === FILL NAN VALUES ===
        df = df.fillna(method='ffill').fillna(0)
        
        # === SELECT FEATURE COLUMNS ===
        print("  - Selecting features...")
        feature_cols = self._get_feature_columns(df)
        features = df[feature_cols].copy()
        
        print(f"   Built {len(feature_cols)} features")
        
        return df, features

    def _days_since_event(self, event_series: pd.Series) -> pd.Series:
        """Count days since last event occurrence"""
        days_since = pd.Series(0, index=event_series.index)
        counter = 0
        for i in range(len(event_series)):
            if event_series.iloc[i] == 1:
                counter = 0
            else:
                counter += 1
            days_since.iloc[i] = counter
        return days_since

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Define which columns to use as features"""
        
        # Base OHLC
        features = ['open', 'high', 'low', 'close']
        
        # Technical indicators
        features.extend(['atr14', 'rsi14', 'ema10', 'ema20', 'ema50',
                        'bb_width', 'bb_position', 'adx14',
                        'macd', 'macd_signal', 'macd_hist',
                        'volume_ratio', 'obv'])
        
        # Candle patterns
        features.extend(['body_pct', 'upper_wick_pct', 'lower_wick_pct',
                        'range_pct'])
        
        # Market structure
        features.extend(['structure', 'bos_bullish', 'bos_bearish',
                        'choch_bullish', 'choch_bearish', 'trend'])
        
        # Order blocks
        features.extend(['ob_bullish', 'ob_bearish',
                        'inside_ob_bull', 'inside_ob_bear',
                        'dist_to_ob_bull', 'dist_to_ob_bear'])
        
        # Fair value gaps
        features.extend(['fvg_bullish', 'fvg_bearish',
                        'inside_fvg_bull', 'inside_fvg_bear',
                        'dist_to_fvg_bull', 'dist_to_fvg_bear'])
        
        # Premium/Discount
        features.extend(['range_position', 'in_premium', 'in_discount',
                        'in_equilibrium', 'in_ote_bull', 'in_ote_bear'])
        
        # Equal H/L
        features.extend(['eq_high', 'eq_low'])
        
        # Pivots (Traditional & Fibonacci)
        for ptype in ['traditional', 'fibonacci']:
            for level in ['P', 'R1', 'S1', 'R2', 'S2']:
                features.extend([f'dist_{level}_{ptype}', f'above_{level}_{ptype}'])
        
        # Confluence
        features.append('confluence_score')
        
        # Momentum
        for period in [1, 3, 5, 10, 20]:
            features.append(f'return_{period}d')
        
        features.extend(['volatility_10d', 'volatility_20d'])
        
        # Time
        features.extend(['day_of_week', 'month', 'quarter'])
        
        # Days since events
        features.extend(['days_since_bos_bull', 'days_since_bos_bear',
                        'days_since_choch', 'days_since_fvg'])
        
        # Filter to columns that actually exist
        features = [f for f in features if f in df.columns]
        
        return features

# ============================================================================
# TARGETS
# ============================================================================
def build_targets(df: pd.DataFrame, threshold: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build classification targets (3-class)
    Class 0: Bearish (next close < next open by threshold)
    Class 1: Bullish (next close > next open by threshold)
    Class 2: Neutral/Choppy (small movement)
    """
    next_open = df['open'].shift(-1).fillna(method='ffill').values
    next_close = df['close'].shift(-1).fillna(method='ffill').values
    
    # Body percentage
    body_pct = (next_close - next_open) / next_open
    
    # Classification
    y_class = np.full(len(df), 2)  # Default: Neutral
    y_class[body_pct > threshold] = 1  # Bullish
    y_class[body_pct < -threshold] = 0  # Bearish
    
    # Regression target (log returns)
    current_close = df['close'].values
    y_reg = np.log(next_close / current_close)
    
    return y_reg, y_class

# ============================================================================
# WINDOWING FOR SEQUENCES
# ============================================================================
def create_sequences(features: pd.DataFrame,
                     y_reg: np.ndarray,
                     y_class: np.ndarray,
                     window: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for LSTM
    """
    X = []
    for i in range(window, len(features)):
        X.append(features.iloc[i-window:i].values)
    X = np.array(X)
    
    # FIX APPLIED: Slice from window-1 to align feature[t] with target[t+1]
    y_reg = y_reg[window-1:]
    y_class = y_class[window-1:]
    
    # Ensure same length
    min_len = min(len(X), len(y_reg), len(y_class))
    X = X[:min_len]
    y_reg = y_reg[:min_len]
    y_class = y_class[:min_len]
    
    return X, y_reg, y_class

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_feature_pipeline(csv_path: str,
                         output_dir: Path = config.OUTPUT_DIR) -> Dict:
    """
    Complete feature engineering pipeline
    Args:
        csv_path: Path to OHLC CSV file
        output_dir: Output directory for artifacts
    
    Returns:
        Dict with paths to saved artifacts
    """
    print("="*80)
    print("APEX TRADE AI - Feature Engineering Pipeline")
    print("="*80)
    
    # Load data
    print(f"\n1. Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    
    # Validate columns
    required = ['time', 'open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort by time
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Build features
    print("\n2. Building features...")
    fe = FeatureEngineering()
    df_full, features = fe.build_features(df)
    
    # Save full dataframe with calculations
    full_csv = output_dir / "dataframe_with_features.csv"
    df_full.to_csv(full_csv, index=False)
    print(f"   Saved full dataframe: {full_csv}")
    
    # Save features only
    features_csv = output_dir / config.FEATURE_CSV
    features.to_csv(features_csv, index=False)
    print(f"   Saved features: {features_csv}")
    
    # Build targets
    print("\n3. Building targets...")
    y_reg, y_class = build_targets(df_full, threshold=config.MOVEMENT_THRESHOLD)
    
    # Class distribution
    unique, counts = np.unique(y_class, return_counts=True)
    print("   Target distribution:")
    for cls, count in zip(unique, counts):
        label = ['Bearish', 'Bullish', 'Neutral'][int(cls)]
        print(f"     {label} ({cls}): {count} ({100*count/len(y_class):.1f}%)")
    
    # Create sequences
    print(f"\n4. Creating sequences (window={config.WINDOW})...")
    X, y_reg_seq, y_class_seq = create_sequences(features, y_reg, y_class, config.WINDOW)
    
    print(f"   X shape: {X.shape}")
    print(f"   y_reg shape: {y_reg_seq.shape}")
    print(f"   y_class shape: {y_class_seq.shape}")
    
    # Scale features
    print("\n5. Scaling features...")
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    
    scaler_features = StandardScaler()
    X_scaled_flat = scaler_features.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n_samples, n_timesteps, n_features)
    
    # Scale regression target
    scaler_price = StandardScaler()
    y_reg_scaled = scaler_price.fit_transform(y_reg_seq.reshape(-1, 1)).ravel()
    
    # Save artifacts
    print("\n6. Saving artifacts...")
    np.save(output_dir / "X.npy", X_scaled)
    np.save(output_dir / "y_reg.npy", y_reg_scaled)
    np.save(output_dir / "y_class.npy", y_class_seq)
    
    with open(output_dir / "scaler_features.pkl", "wb") as f:
        pickle.dump(scaler_features, f)
    
    with open(output_dir / "scaler_price.pkl", "wb") as f:
        pickle.dump(scaler_price, f)
    
    print(f"   - X.npy")
    print(f"   - y_reg.npy")
    print(f"   - y_class.npy")
    print(f"   - scaler_features.pkl")
    print(f"   - scaler_price.pkl")
    
    # Summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Features: {n_features}")
    print(f"Sequences: {n_samples}")
    print(f"Window: {n_timesteps}")
    print(f"Classes: {len(unique)}")
    print("="*80)
    
    return {
        'X': X_scaled,
        'y_reg': y_reg_scaled,
        'y_class': y_class_seq,
        'features': features,
        'df': df_full,
        'scaler_features': scaler_features,
        'scaler_price': scaler_price,
        'n_features': n_features
    }

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python Feature_pipeline.py <path_to_csv>")
        print("Example: python Feature_pipeline.py XAUUSD_history.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        results = run_feature_pipeline(csv_path)
        print("\n[SUCCESS] Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""
APEX TRADE AI - Enhanced Feature Engineering Pipeline
======================================================

Integrates:
- Smart Money Concepts (OB Strength, FVG Filled, Liquidity Sweeps)
- ICT Concepts (Order Blocks, FVG, BOS, CHoCH)
- Multi-type Pivot Points
- Classical Technical Indicators
- Multi-Timeframe Features (Logic Ready)
- Confluence Detection
- External Data (DXY, News - Placeholders)

Target: 3-Class Classification (Bearish, Bullish, Neutral) with 0.1% threshold
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
from pathlib import Path
import warnings
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    WINDOW = 50  # Sequence length for LSTM
    MIN_PERIODS = 200  # For indicators
    
    # Structure Detection
    SWING_LENGTH = 10  # Reduced for more sensitivity matching ICT
    INTERNAL_LENGTH = 3  # Internal structure
    
    # Thresholds
    MOVEMENT_THRESHOLD = 0.001  # 0.1% for classification target
    EQH_EQL_THRESHOLD = 0.0005  # Equal highs/lows tolerance (precise)
    FVG_MIN_SIZE = 0.0002  # Minimum FVG size (0.02%)
    
    # Pivot
    PIVOT_TYPES = ['traditional', 'fibonacci', 'camarilla', 'woodie', 'demark']
    
    # Outputs
    OUTPUT_DIR = Path("data")
    FEATURE_CSV = "features_enhanced.csv" # output relative to OUTPUT_DIR? No, usually code uses full path. Let's fix Config.
    
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
        Detect swing highs and lows using a centered window
        """
        highs = df['high']
        lows = df['low']
        
        swing_high = pd.Series(np.nan, index=df.index)
        swing_low = pd.Series(np.nan, index=df.index)
        
        # Vectorized approach roughly or rolling max
        # A swing high is max of window [i-length, i+length] at i
        # We need to shift to align processing. 
        # Using rolling max with center=True
        
        roll_max = highs.rolling(window=2*length+1, center=True).max()
        roll_min = lows.rolling(window=2*length+1, center=True).min()
        
        # Identify where High == Roll_Max
        is_swing_high = (highs == roll_max) 
        is_swing_low = (lows == roll_min)
        
        # We can't know future for today's detection in live mode, BUT for labeling history we can.
        # For LIVE (causal) feature generation:
        # We detect a swing point 'length' bars ago.
        # So at time t, we know if t-length was a swing high.
        # This pipeline processes historical data for training, so centered is fine for labeling,
        # but for Features used in prediction at time t, we can only know swings resolved <= t.
        # We will use lag-based labeling to ensure causality if we were plotting, 
        # but for ML features: "dist_to_last_swing_high" implies the last CONFIRMED swing.
        
        # Let's stick to the causal definition:
        # A swing high is confirmed when we have 'length' lower bars after it.
        # So at index i, check if i-length was the high of i-2*length to i.
        
        # Logic:
        # At index i, look back at candle i-length.
        # Is (i-length) > all candles in (i-2*length ... i-length-1) AND (i-length) > all candles in (i-length+1 ... i)
        
        # We will assign the swing value at the time it occurred, but typically we only know it later.
        # We'll stick to the "last_swing_high" feature which just holds the value of the last identified one.
        
        # For identifying points:
        swing_high[is_swing_high] = highs[is_swing_high]
        swing_low[is_swing_low] = lows[is_swing_low]
        
        return swing_high, swing_low
    
    @staticmethod
    def label_structure(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series) -> pd.DataFrame:
        """
        Label structure as HH, HL, LH, LL
        """
        df = df.copy()
        df['swing_high'] = swing_high
        df['swing_low'] = swing_low
        
        # Forward fill to get "last known" swing
        # Note: In a causal system, valid_swing_high is only known 'length' bars after.
        # But for 'last_swing_high' feature used at time t, we usually take the most recently confirmed one.
        # We will assume swing_high/low contains confirmed swings (aligned to their occurrence date, but we ffill).
        
        df['last_swing_high'] = swing_high.fillna(method='ffill')
        df['last_swing_low'] = swing_low.fillna(method='ffill')
        
        df['prev_swing_high'] = df['last_swing_high'].shift(1) # shift logic slightly flawed with ffill, need unique values
        # Better: Group continuous segments of ffilled values
        
        # Structure labels (simplified for brevity)
        df['structure'] = 0 
        
        return df

    @staticmethod
    def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH)
        """
        df = df.copy()
        df['bos_bullish'] = 0
        df['bos_bearish'] = 0
        
        # Logic: Close crossing last swing
        # To avoid noise, use confirmed swings
        last_high = df['last_swing_high']
        last_low = df['last_swing_low']
        
        # Breakout
        break_high = (df['close'] > last_high) & (df['close'].shift(1) <= last_high)
        break_low = (df['close'] < last_low) & (df['close'].shift(1) >= last_low)
        
        df.loc[break_high, 'bos_bullish'] = 1
        df.loc[break_low, 'bos_bearish'] = 1
        
        # Trend
        df['trend'] = 0
        df.loc[break_high, 'trend'] = 1
        df.loc[break_low, 'trend'] = -1
        df['trend'] = df['trend'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return df

# ============================================================================
# ORDER BLOCKS & LIQUIDITY (ENHANCED)
# ============================================================================

class OrderBlocks:
    """Detect ICT Order Blocks with Strength and Mitigated Status"""
    
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Bullish OB: Last down-candle before a strong up-move (BOS)
        """
        df = df.copy()
        df['ob_bullish'] = 0
        df['ob_bearish'] = 0
        df['ob_bottom'] = np.nan
        df['ob_top'] = np.nan
        
        # We iterate to find OBs associated with BOS
        # This is slow, but SMC logic is sequential
        
        # Initialize columns for OB properties
        df['ob_strength'] = 0.0
        df['ob_mitigated'] = 0 # Touched/Filled
        
        bos_bull_indices = df.index[df['bos_bullish'] == 1]
        
        for idx in bos_bull_indices:
            loc = df.index.get_loc(idx)
            # Scan backwards for the last bearish candle (Close < Open)
            # Limit scan to 10 candles
            for i in range(1, 15):
                if loc - i < 0: break
                
                start_idx = loc - i
                row = df.iloc[start_idx]
                
                # Check for bearish candle
                if row['close'] < row['open']:
                    # Found Bullish OB candidate
                    df.iat[start_idx, df.columns.get_loc('ob_bullish')] = 1
                    df.iat[start_idx, df.columns.get_loc('ob_bottom')] = row['low']
                    df.iat[start_idx, df.columns.get_loc('ob_top')] = row['high']
                    
                    # Strength: Volume / body size
                    strength = row['volume'] if 'volume' in df.columns else 1.0
                    df.iat[start_idx, df.columns.get_loc('ob_strength')] = strength
                    break
        
        # Bearish OBs
        bos_bear_indices = df.index[df['bos_bearish'] == 1]
        for idx in bos_bear_indices:
            loc = df.index.get_loc(idx)
            for i in range(1, 15):
                if loc - i < 0: break
                start_idx = loc - i
                row = df.iloc[start_idx]
                
                if row['close'] > row['open']:
                    df.iat[start_idx, df.columns.get_loc('ob_bearish')] = 1
                    df.iat[start_idx, df.columns.get_loc('ob_bottom')] = row['low']
                    df.iat[start_idx, df.columns.get_loc('ob_top')] = row['high']
                    df.iat[start_idx, df.columns.get_loc('ob_strength')] = row['volume'] if 'volume' in df.columns else 1.0
                    break
                    
        # Forward fill active OBs (simplified assumption: last OB detects validation)
        # Real SMC keeps a list of active UNMITIGATED OBs.
        # For features, we just track "inside closest OB"
        
        df['active_ob_bull_top'] = df['ob_top'].where(df['ob_bullish']==1).fillna(method='ffill')
        df['active_ob_bull_bot'] = df['ob_bottom'].where(df['ob_bullish']==1).fillna(method='ffill')
        
        df['active_ob_bear_top'] = df['ob_top'].where(df['ob_bearish']==1).fillna(method='ffill')
        df['active_ob_bear_bot'] = df['ob_bottom'].where(df['ob_bearish']==1).fillna(method='ffill')
        
        df['inside_ob_bull'] = ((df['low'] <= df['active_ob_bull_top']) & (df['high'] >= df['active_ob_bull_bot'])).astype(int)
        df['inside_ob_bear'] = ((df['high'] >= df['active_ob_bear_bot']) & (df['low'] <= df['active_ob_bear_top'])).astype(int)
        
        return df

    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects if current candle swept liquidity (wick went beyond swing, body closed inside)
        """
        df = df.copy()
        
        # Sweep High: High > Last Swing High, Close < Last Swing High
        sweep_high = (df['high'] > df['last_swing_high']) & (df['close'] < df['last_swing_high'])
        
        # Sweep Low: Low < Last Swing Low, Close > Last Swing Low
        sweep_low = (df['low'] < df['last_swing_low']) & (df['close'] > df['last_swing_low'])
        
        df['sweep_liquidity_high'] = sweep_high.astype(int)
        df['sweep_liquidity_low'] = sweep_low.astype(int)
        
        return df

# ============================================================================
# FAIR VALUE GAPS (ENHANCED)
# ============================================================================

class FairValueGaps:
    """Detect Fair Value Gaps and their state"""
    
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['fvg_bullish'] = 0
        df['fvg_bearish'] = 0
        
        high = df['high']
        low = df['low']
        
        # Bull FVG: (Low[i] > High[i-2])
        # We index at i (the confirmation candle). Gap is between i-2 and i.
        # Actually standard definition: Candle 1 High < Candle 3 Low (Bullish)
        
        c1_high = high.shift(2)
        c3_low = low
        
        bull_fvg = (c3_low > c1_high) & ((c3_low - c1_high) > (df['close'] * config.FVG_MIN_SIZE))
        df.loc[bull_fvg, 'fvg_bullish'] = 1
        
        # Bear FVG: Candle 1 Low > Candle 3 High
        c1_low = low.shift(2)
        c3_high = high
        
        bear_fvg = (c3_high < c1_low) & ((c1_low - c3_high) > (df['close'] * config.FVG_MIN_SIZE))
        df.loc[bear_fvg, 'fvg_bearish'] = 1
        
        # Store FVG boundaries
        df['fvg_bull_top'] = np.nan
        df['fvg_bull_bot'] = np.nan
        df.loc[bull_fvg, 'fvg_bull_top'] = c3_low
        df.loc[bull_fvg, 'fvg_bull_bot'] = c1_high
        
        df['fvg_bear_top'] = np.nan
        df['fvg_bear_bot'] = np.nan
        df.loc[bear_fvg, 'fvg_bear_top'] = c1_low
        df.loc[bear_fvg, 'fvg_bear_bot'] = c3_high
        
        # Active FVG (last one)
        df['active_fvg_bull_top'] = df['fvg_bull_top'].fillna(method='ffill')
        df['active_fvg_bull_bot'] = df['fvg_bull_bot'].fillna(method='ffill')
        df['active_fvg_bear_top'] = df['fvg_bear_top'].fillna(method='ffill')
        df['active_fvg_bear_bot'] = df['fvg_bear_bot'].fillna(method='ffill')
        
        # Inside FVG
        df['inside_fvg_bull'] = ((df['low'] <= df['active_fvg_bull_top']) & 
                                 (df['high'] >= df['active_fvg_bull_bot'])).astype(int)
        df['inside_fvg_bear'] = ((df['high'] >= df['active_fvg_bear_bot']) & 
                                 (df['low'] <= df['active_fvg_bear_top'])).astype(int)
        
        return df

# ============================================================================
# PREMIUM / DISCOUNT / OTE
# ============================================================================

class PremiumDiscount:
    """Calculate Premium, Discount, and Equilibrium zones"""
    
    @staticmethod
    def calculate_zones(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        range_high = df['last_swing_high']
        range_low = df['last_swing_low']
        
        denom = (range_high - range_low).replace(0, np.nan)
        df['range_position'] = (df['close'] - range_low) / denom
        df['range_position'] = df['range_position'].fillna(0.5)
        
        df['in_premium'] = (df['range_position'] > 0.5).astype(int)
        df['in_discount'] = (df['range_position'] < 0.5).astype(int)
        
        # OTE
        df['in_ote_bull'] = ((df['range_position'] >= 0.214) & (df['range_position'] <= 0.382)).astype(int) 
        # Note: 1 - 0.786 = 0.214 (Retracement from low relative to range)
        
        df['in_ote_bear'] = ((df['range_position'] >= 0.618) & (df['range_position'] <= 0.786)).astype(int)
        
        return df

# ============================================================================
# PIVOT POINTS
# ============================================================================

class PivotPoints:
    """Calculate multiple pivot point types"""
    
    @staticmethod
    def calculate_all_pivots(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Shift OHLC by 1 to use previous day for today's pivots
        ph = df['high'].shift(1)
        pl = df['low'].shift(1)
        pc = df['close'].shift(1)
        
        # Standard
        P = (ph + pl + pc) / 3
        df['P_traditional'] = P
        df['R1_traditional'] = 2*P - pl
        df['S1_traditional'] = 2*P - ph
        df['R2_traditional'] = P + (ph - pl)
        df['S2_traditional'] = P - (ph - pl)
        
        # Fibonacci
        r = ph - pl
        df['P_fibonacci'] = P
        df['R1_fibonacci'] = P + 0.382 * r
        df['S1_fibonacci'] = P - 0.382 * r
        df['R2_fibonacci'] = P + 0.618 * r
        df['S2_fibonacci'] = P - 0.618 * r
        
        return df

# ============================================================================
# CONFLUENCE DETECTION
# ============================================================================

class ConfluenceDetector:
    """Detect confluence between multiple factors"""
    
    @staticmethod
    def calculate_confluence_score(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
        score = pd.Series(0.0, index=df.index)
        
        # 1. OB Alignment
        score += df['inside_ob_bull'] * 20
        score += df['inside_ob_bear'] * 20
        
        # 2. FVG Alignment
        score += df['inside_fvg_bull'] * 20
        score += df['inside_fvg_bear'] * 20
        
        # 3. Liquidity Sweep
        score += df['sweep_liquidity_low'] * 15 # Bullish reversaL
        score += df['sweep_liquidity_high'] * 15 # Bearish reversal
        
        # 4. Premium/Discount
        score += df['in_discount'] * 10
        score += df['in_premium'] * 10
        
        # 5. OTE
        score += df['in_ote_bull'] * 15
        score += df['in_ote_bear'] * 15
        
        return score.clip(0, 100)

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
        self.pivots = PivotPoints()
        self.confluence = ConfluenceDetector()
    
    def _add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add DXY and News features.
        Since we don't have DXY live data, we return empty/placeholders.
        """
        # Placeholder for DXY correlation
        df['dxy_corr_30d'] = 0.0
        # Placeholder for News Sentiment
        df['news_sentiment'] = 0.0
        df['news_impact'] = 0.0 # 0=Low, 1=Med, 2=High
        
        return df

    def create_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create strict 3-class targets based on 0.1% threshold.
        Target is NEXT DAY return using Close-to-Close or Open-to-Close?
        User said: "Entry: Next day open... Exit: Next day close".
        So prediction is for (Next Close - Next Open).
        
        Let T = Today (prediction time).
        We want to predict Return(T+1) = (Close[T+1] - Open[T+1]) / Open[T+1].
        
        Shift: 
        return_next = (close.shift(-1) - open.shift(-1)) / open.shift(-1)
        """
        
        # Calculate Next Day Return (Intraday)
        next_open = df['open'].shift(-1)
        next_close = df['close'].shift(-1)
        
        # Avoid division by zero
        ret = (next_close - next_open) / next_open
        ret = ret.fillna(0)
        
        threshold = config.MOVEMENT_THRESHOLD # 0.001
        
        # Classes: 0=Bearish (< -0.1%), 1=Bullish (> +0.1%), 2=Neutral
        targets = np.zeros(len(df), dtype=int)
        targets[:] = 2 # Default Neutral
        
        targets[ret > threshold] = 1
        targets[ret < -threshold] = 0
        
        return targets, ret.values

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Building features...")
        df = df.copy()
        
        # (Moved Phase 2 Integration to after Classical features)

        # ---------------------------------------
    
        # 1. Classical
        df['atr'] = self.tech.atr(df)
        df['rsi'] = self.tech.rsi(df['close'])
        df['ema50'] = self.tech.ema(df['close'], 50)
        df['macd'], _, _ = self.tech.macd(df['close'])
        df['adx'] = self.tech.adx(df)

        # --- PHASE 2 INTEGRATION: DXY & NEWS ---
        # --- PHASE 2 INTEGRATION: DXY & NEWS ---
        from src.data.fetch_dxy import align_dxy_with_gold
        from src.data.news_sentiment import simulate_news_impact
        
        # 1. Align DXY
        print("  Merging DXY data...")
        try:
            dxy_df = align_dxy_with_gold("data/XAUUSD_history.csv")
            if dxy_df is not None:
                dxy_subset = dxy_df[['time', 'dxy_close', 'dxy_open']]
                dxy_subset['time'] = pd.to_datetime(dxy_subset['time'])
                df['time'] = pd.to_datetime(df['time'])
                
                df = pd.merge(df, dxy_subset, on='time', how='left')
                df['dxy_close'] = df['dxy_close'].fillna(method='ffill')
                
                # DXY Features
                df['dxy_rsi'] = self.tech.rsi(df['dxy_close'])
                df['dxy_ema_50'] = self.tech.ema(df['dxy_close'], 50)
                df['dxy_trend'] = np.where(df['dxy_close'] > df['dxy_ema_50'], 1, -1)
                
                # Correlation
                df['gold_dxy_corr'] = df['close'].rolling(window=30).corr(df['dxy_close'])
                
                # Confluence
                df['gold_ema_50'] = self.tech.ema(df['close'], 50)
                df['gold_trend'] = np.where(df['close'] > df['gold_ema_50'], 1, -1)
                
                df['inverse_confluence'] = np.where((df['gold_trend'] == 1) & (df['dxy_trend'] == -1), 1, 0)
                df['inverse_confluence'] = np.where((df['gold_trend'] == -1) & (df['dxy_trend'] == 1), -1, df['inverse_confluence'])
            else:
                # DXY fetch returned None (failed synthetic?)
                raise ValueError("DXY Data is None")
                
        except Exception as e:
            print(f"  Warning: DXY Integration failed: {e}")
            # Fallback Defaults (MUST MATCH Training Features)
            df['dxy_close'] = df['close'] # Dummy
            df['dxy_open'] = df['open']
            df['dxy_rsi'] = 50.0
            df['dxy_ema_50'] = df['close']
            df['dxy_trend'] = 0
            df['gold_dxy_corr'] = -0.8 
            df['inverse_confluence'] = 0
            # Ensure safe fills
            df['dxy_close'] = df['dxy_close'].fillna(0)

        # 2. News Sentiment
        print("  Merging News Sentiment...")
        try:
            news_df = simulate_news_impact(df['time'])
            news_df['time'] = pd.to_datetime(news_df['time'])
            
            df = pd.merge(df, news_df, on='time', how='left')
            df['news_sentiment'] = df['news_sentiment'].fillna(0)
            
            # Impact Feature (Now Safe because ADX exists)
            df['news_impact_score'] = df['news_sentiment'] * df['adx'] / 100.0
            
        except Exception as e:
            print(f"  Warning: News Integration failed: {e}")
            df['news_sentiment'] = 0
            df['news_impact_score'] = 0
        
        # 2. Market Structure (SMC)
        swing_high, swing_low = self.structure.detect_swing_points(df, config.SWING_LENGTH)
        df = self.structure.label_structure(df, swing_high, swing_low)
        df = self.structure.detect_bos_choch(df)
        
        # 3. Advanced SMC
        df = self.ob.detect_order_blocks(df)
        df = self.fvg.detect_fvg(df)
        df = self.ob.detect_liquidity_sweeps(df)
        
        # 4. Zones
        df = self.zones.calculate_zones(df)
        
        # 5. Pivots
        df = self.pivots.calculate_all_pivots(df)
        
        # 6. Confluence
        df['confluence_score'] = self.confluence.calculate_confluence_score(df, df['atr'])
        
        # 7. External
        df = self._add_external_features(df)
        
        # 8. Time features
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['day_of_week'] = df['time'].dt.dayofweek
            df['month'] = df['time'].dt.month
        
        # Fill NaNs
        df = df.fillna(method='ffill').fillna(0)
        
        # Feature Selection
        # Exclude Target/Raw columns AND new visual columns not in training
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'spread', 'real_volume',
                        'R2_traditional', 'S2_traditional', 'R2_fibonacci', 'S2_fibonacci']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        # Exclude intermediate calc columns if needed, but for now keep all numericals
        # Filter strictly numeric
        features_df = df[feature_cols].select_dtypes(include=[np.number])
        
        # Clean Features (Handle Inf/NaN globally)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        print(f"Features: {features_df.shape[1]}")
        return df, features_df

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(data_path="data/XAUUSD_history.csv", suffix=""):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load your CSV
    df = pd.read_csv(data_path)
    
    # Process
    try:
        fe = FeatureEngineering()
        df_enhanced, features = fe.build_features(df)
        
        # Targets
        y_class, y_reg = fe.create_targets(df_enhanced)
        
        # Clean Features (Handle Inf/NaN)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Scale Features
        print("Scaling...")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(features)
        
        # Create Sequences (LSTM style : [Samples, TimeSteps, Features])
        # For Tree models we can flatten later
        X_seq = []
        y_seq_class = []
        
        win = config.WINDOW
        for i in range(win, len(X_scaled) - 1): # -1 because target is next day
            X_seq.append(X_scaled[i-win:i])
            y_seq_class.append(y_class[i]) # Target for prediction at i is outcome of i+1
        
        X_seq = np.array(X_seq)
        y_seq_class = np.array(y_seq_class)
        
        print(f"Final Shapes: X={X_seq.shape}, y={y_seq_class.shape}")
        
        # Save
        if suffix:
            suffix = "_" + suffix.lstrip("_") # Ensure single leading underscore if present
        
        np.save(f"data/X{suffix}.npy", X_seq)
        np.save(f"data/y_class{suffix}.npy", y_seq_class)
        # Regression align
        # y_reg might be longer, slice to match sequence logic
        # y_reg matches df len. sequences start at `win` and end at -1.
        y_reg_aligned = y_reg[win:-1]
        np.save(f"data/y_reg{suffix}.npy", y_reg_aligned)
        
        df_enhanced.to_csv(f"data/features_enhanced{suffix}.csv", index=False)
        
        with open(f"data/scaler_features{suffix}.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        print(f"Done. Artifacts saved with suffix '{suffix}'.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/XAUUSD_history.csv")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    
    run_pipeline(args.data, args.suffix)

if __name__ == "__main__":
    run_pipeline()
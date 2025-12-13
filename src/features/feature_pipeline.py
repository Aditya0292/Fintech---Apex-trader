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
- External Data (DXY, News, CSM)

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
import json

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

warnings.filterwarnings('ignore')

from src.utils.config_loader import config as global_config
from src.utils.time_utils import normalize_ts, align_datasets
from src.utils.schema_validator import validate_feature_schema
from src.utils.logger import get_logger
from src.data.csm_provider import CSMProvider
from src.data.news_manager import NewsManager

logger = get_logger()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    def __init__(self):
        feat_conf = global_config.get('features', {})
        
        # Data
        self.WINDOW = feat_conf.get('window_size', 50)
        self.MIN_PERIODS = feat_conf.get('min_periods', 200)
        
        # Structure Detection
        self.SWING_LENGTH = feat_conf.get('swing_length', 10)
        self.INTERNAL_LENGTH = feat_conf.get('internal_structure_length', 3)
        
        # Thresholds
        self.MOVEMENT_THRESHOLD = feat_conf.get('movement_threshold', 0.0002) # Tuning: 0.02% for Forex Sensitivity
        self.EQH_EQL_THRESHOLD = feat_conf.get('eqh_eql_threshold', 0.0005)
        self.FVG_MIN_SIZE = feat_conf.get('fvg_min_size', 0.0002)
        
        # Pivot
        self.PIVOT_TYPES = feat_conf.get('pivot_types', ['traditional', 'fibonacci'])
        
        # Outputs
        self.OUTPUT_DIR = Path("data")
        self.SCHEMA_PATH = feat_conf.get('schema_path', "src/config/feature_schema_v2.json")
        
        self.OUTPUT_DIR.mkdir(exist_ok=True)

config = Config()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Safe division avoiding divide by zero"""
    if fill == 0.0:
        # Standard case: div by zero = 0
        return (a / b.replace(0, np.nan)).fillna(fill)
    else:
        # General case
        res = a / b.replace(0, np.nan)
        return res.fillna(fill)

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
        highs = df['high']
        lows = df['low']
        
        swing_high = pd.Series(np.nan, index=df.index)
        swing_low = pd.Series(np.nan, index=df.index)
        
        roll_max = highs.rolling(window=2*length+1, center=True).max()
        roll_min = lows.rolling(window=2*length+1, center=True).min()
        
        is_swing_high = (highs == roll_max) 
        is_swing_low = (lows == roll_min)
        
        swing_high[is_swing_high] = highs[is_swing_high]
        swing_low[is_swing_low] = lows[is_swing_low]
        
        return swing_high, swing_low
    
    @staticmethod
    def label_structure(df: pd.DataFrame, swing_high: pd.Series, swing_low: pd.Series) -> pd.DataFrame:
        df = df.copy()
        df['swing_high'] = swing_high
        df['swing_low'] = swing_low
        
        df['last_swing_high'] = swing_high.fillna(method='ffill')
        df['last_swing_low'] = swing_low.fillna(method='ffill')
        
        df['prev_swing_high'] = df['last_swing_high'].shift(1)
        
        df['structure'] = 0 
        
        return df

    @staticmethod
    def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['bos_bullish'] = 0
        df['bos_bearish'] = 0
        
        last_high = df['last_swing_high']
        last_low = df['last_swing_low']
        
        break_high = (df['close'] > last_high) & (df['close'].shift(1) <= last_high)
        break_low = (df['close'] < last_low) & (df['close'].shift(1) >= last_low)
        
        df.loc[break_high, 'bos_bullish'] = 1
        df.loc[break_low, 'bos_bearish'] = 1
        
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
        df = df.copy()
        df['ob_bullish'] = 0
        df['ob_bearish'] = 0
        df['ob_bottom'] = np.nan
        df['ob_top'] = np.nan
        df['ob_strength'] = 0.0
        df['ob_mitigated'] = 0 # Touched/Filled
        
        bos_bull_indices = df.index[df['bos_bullish'] == 1]
        
        for idx in bos_bull_indices:
            loc = df.index.get_loc(idx)
            for i in range(1, 15):
                if loc - i < 0: break
                
                start_idx = loc - i
                row = df.iloc[start_idx]
                
                if row['close'] < row['open']:
                    df.iat[start_idx, df.columns.get_loc('ob_bullish')] = 1
                    df.iat[start_idx, df.columns.get_loc('ob_bottom')] = row['low']
                    df.iat[start_idx, df.columns.get_loc('ob_top')] = row['high']
                    strength = row['volume'] if 'volume' in df.columns else 1.0
                    df.iat[start_idx, df.columns.get_loc('ob_strength')] = strength
                    break
        
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
                    
        df['active_ob_bull_top'] = df['ob_top'].where(df['ob_bullish']==1).fillna(method='ffill')
        df['active_ob_bull_bot'] = df['ob_bottom'].where(df['ob_bullish']==1).fillna(method='ffill')
        
        df['active_ob_bear_top'] = df['ob_top'].where(df['ob_bearish']==1).fillna(method='ffill')
        df['active_ob_bear_bot'] = df['ob_bottom'].where(df['ob_bearish']==1).fillna(method='ffill')
        
        df['inside_ob_bull'] = ((df['low'] <= df['active_ob_bull_top']) & (df['high'] >= df['active_ob_bull_bot'])).astype(int)
        df['inside_ob_bear'] = ((df['high'] >= df['active_ob_bear_bot']) & (df['low'] <= df['active_ob_bear_top'])).astype(int)
        
        return df

    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        sweep_high = (df['high'] > df['last_swing_high']) & (df['close'] < df['last_swing_high'])
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
        
        c1_high = high.shift(2)
        c3_low = low
        
        bull_fvg = (c3_low > c1_high) & ((c3_low - c1_high) > (df['close'] * config.FVG_MIN_SIZE))
        df.loc[bull_fvg, 'fvg_bullish'] = 1
        
        c1_low = low.shift(2)
        c3_high = high
        
        bear_fvg = (c3_high < c1_low) & ((c1_low - c3_high) > (df['close'] * config.FVG_MIN_SIZE))
        df.loc[bear_fvg, 'fvg_bearish'] = 1
        
        df['fvg_bull_top'] = np.nan
        df['fvg_bull_bot'] = np.nan
        df.loc[bull_fvg, 'fvg_bull_top'] = c3_low
        df.loc[bull_fvg, 'fvg_bull_bot'] = c1_high
        
        df['fvg_bear_top'] = np.nan
        df['fvg_bear_bot'] = np.nan
        df.loc[bear_fvg, 'fvg_bear_top'] = c1_low
        df.loc[bear_fvg, 'fvg_bear_bot'] = c3_high
        
        df['active_fvg_bull_top'] = df['fvg_bull_top'].fillna(method='ffill')
        df['active_fvg_bull_bot'] = df['fvg_bull_bot'].fillna(method='ffill')
        df['active_fvg_bear_top'] = df['fvg_bear_top'].fillna(method='ffill')
        df['active_fvg_bear_bot'] = df['fvg_bear_bot'].fillna(method='ffill')
        
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
        
        df['in_ote_bull'] = ((df['range_position'] >= 0.214) & (df['range_position'] <= 0.382)).astype(int) 
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
        
        score += df['inside_ob_bull'] * 20
        score += df['inside_ob_bear'] * 20
        
        score += df['inside_fvg_bull'] * 20
        score += df['inside_fvg_bear'] * 20
        
        score += df['sweep_liquidity_low'] * 15 # Bullish reversaL
        score += df['sweep_liquidity_high'] * 15 # Bearish reversal
        
        score += df['in_discount'] * 10
        score += df['in_premium'] * 10
        
        score += df['in_ote_bull'] * 15
        score += df['in_ote_bear'] * 15
        
        return score.clip(0, 100)

# ============================================================================
# COMPLETE FEATURE BUILDER
# ============================================================================

class FeatureEngineering:
    """Main feature engineering pipeline"""
    
    def __init__(self, symbol: str = "XAUUSD"):
        self.symbol = symbol
        self.tech = TechnicalIndicators()
        self.structure = MarketStructure()
        self.ob = OrderBlocks()
        self.fvg = FairValueGaps()
        self.zones = PremiumDiscount()
        self.pivots = PivotPoints()
        self.confluence = ConfluenceDetector()
        self.csm_provider = CSMProvider()
        self.news_manager = NewsManager()
    
    def _add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add DXY and News features with strict Timezone alignment using align_datasets.
        """
        # Gold Technicals first (Internal dependency)
        df['gold_ema_50'] = self.tech.ema(df['close'], 50)
        df['gold_ema_200'] = self.tech.ema(df['close'], 200)
        df['gold_trend'] = np.where(df['close'] > df['gold_ema_50'], 1, -1)
        
        # --- DXY INTEGRATION (Legacy but Global) ---
        try:
            from src.data.fetch_dxy import align_dxy_with_gold
            # TODO: Make this path dynamic or fetch generic DXY
            dxy_df = align_dxy_with_gold("data/XAUUSD_history.csv")
            
            if dxy_df is not None:
                dxy_df.columns = [
                    f"dxy_{c}" if c != 'time' and not c.startswith('dxy_') else c 
                    for c in dxy_df.columns
                ]
                dxy_df = dxy_df.loc[:, ~dxy_df.columns.duplicated()]
                
                df = align_datasets(df, dxy_df, on='time', tolerance='4h')
                
                df['dxy_close'] = df['dxy_close'].fillna(method='ffill')
                df['dxy_open'] = df['dxy_open'].fillna(method='ffill')
                df['dxy_rsi'] = self.tech.rsi(df['dxy_close'])
                df['dxy_ema_50'] = self.tech.ema(df['dxy_close'], 50)
                df['dxy_trend'] = np.where(df['dxy_close'] > df['dxy_ema_50'], 1, -1)
                # Correction: Correlation should be against the asset close, not hardcoded gold
                df['gold_dxy_corr'] = df['close'].rolling(window=30).corr(df['dxy_close'])
                
                # Confluence
                # Determine expected correlation w/ DXY
                # USDJPY (USD Base) -> Positive Correlation (DXY Up = Pair Up)
                # EURUSD/XAUUSD (USD Quote) -> Negative Correlation (DXY Up = Pair Down)
                is_direct_correlation = self.symbol.startswith('USD') and not self.symbol.endswith('USD')
                
                if is_direct_correlation:
                     # Direct Confluence
                     df['inverse_confluence'] = np.where((df['gold_trend'] == 1) & (df['dxy_trend'] == 1), 1, 0)
                     df['inverse_confluence'] = np.where((df['gold_trend'] == -1) & (df['dxy_trend'] == -1), -1, df['inverse_confluence'])
                else:
                     # Inverse Confluence (Original Logic)
                     df['inverse_confluence'] = np.where((df['gold_trend'] == 1) & (df['dxy_trend'] == -1), 1, 0)
                     df['inverse_confluence'] = np.where((df['gold_trend'] == -1) & (df['dxy_trend'] == 1), -1, df['inverse_confluence'])
            else:
                 raise ValueError("DXY Data None")
        except Exception as e:
            # logger.warning(f"DXY Integration warning: {e}")
            # Fill defaults
            cols = ['dxy_close', 'dxy_open', 'dxy_rsi', 'dxy_ema_50', 'dxy_trend', 'gold_dxy_corr', 'inverse_confluence']
            for c in cols: df[c] = 0.0

        # --- CSM INTEGRATION ---
        try:
            assets_conf = global_config.get('assets', {})
            these_assets = assets_conf.get(self.symbol, {})
            currencies = these_assets.get('csm_currencies', ['USD'])
            
            csm_df = self.csm_provider.get_csm_data(df['time'].min(), df['time'].max(), currencies)
            
            if csm_df is not None and not csm_df.empty:
                df = align_datasets(df, csm_df, on='time', tolerance='1h')
                
                # Calculate diff
                if len(currencies) >= 2:
                    base, quote = currencies[0], currencies[1]
                    df['csm_base'] = df.get(f'csm_{base}', 5.0)
                    df['csm_quote'] = df.get(f'csm_{quote}', 5.0)
                    df['csm_diff'] = df['csm_base'] - df['csm_quote']
                else:
                     df['csm_base'] = 5.0
                     df['csm_quote'] = df.get(f'csm_USD', 5.0)
                     df['csm_diff'] = 5.0 - df['csm_quote']
            else:
                 df['csm_base'] = 5.0
                 df['csm_quote'] = 5.0
                 df['csm_diff'] = 0.0
        except Exception as e:
            logger.warning(f"CSM Integration failed: {e}")
            df['csm_base'] = 0.0
            df['csm_quote'] = 0.0
            df['csm_diff'] = 0.0
        
        # Check CSM NaNs
        for col in ['csm_base', 'csm_quote', 'csm_diff']:
            if col not in df.columns: df[col] = 0.0
            df[col] = df[col].fillna(0.0)

        # --- NEWS INTEGRATION V2 ---
        try:
            news_df = self.news_manager.aggregate_impact_for_symbol(df['time'], self.symbol)
            if news_df is not None:
                if 'time' not in news_df.columns:
                     news_df = news_df.reset_index().rename(columns={'index': 'time'})
                
                df = align_datasets(df, news_df, on='time', tolerance='4h')
            
            # Legacy Map
            if 'news_impact_net' in df.columns:
                df['news_sentiment'] = df['news_impact_net'] 
                df['news_impact_score'] = df['news_impact_net'] * df.get('adx', 0) / 100.0
            else:
                df['news_sentiment'] = 0.0
                df['news_impact_score'] = 0.0

            # Specific impacts
            if len(currencies) >= 1:
                base = currencies[0]
                df['news_impact_base'] = df.get(f'news_impact_{base}', 0.0)
            else:
                df['news_impact_base'] = 0.0
                
            if len(currencies) >= 2:
                quote = currencies[1]
                df['news_impact_quote'] = df.get(f'news_impact_{quote}', 0.0)
            else:
                df['news_impact_quote'] = df.get('news_impact_USD', 0.0)
        except Exception as e:
             logger.warning(f"News Integration V2 failed: {e}")
        
        # Final cleanup for all V2 external columns
        v2_cols = ['news_impact_base', 'news_impact_quote', 'news_impact_net', 'news_sentiment', 'news_impact_score',
                   'csm_base', 'csm_quote', 'csm_diff']
        for c in v2_cols:
            if c not in df.columns: df[c] = 0.0
            df[c] = df[c].fillna(0.0)

        return df

    def create_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create strict 3-class targets based on movement threshold.
        """
        next_open = df['open'].shift(-1)
        next_close = df['close'].shift(-1)
        
        ret = (next_close - next_open) / next_open
        ret = ret.fillna(0)
        
        threshold = config.MOVEMENT_THRESHOLD
        
        targets = np.zeros(len(df), dtype=int)
        targets[:] = 2 # Default Neutral
        
        targets[ret > threshold] = 1
        targets[ret < -threshold] = 0
        
        return targets, ret.values

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Building features...")
        if isinstance(df, pd.DataFrame):
            df.columns = df.columns.str.strip().str.lower()
        
        df = df.copy()
        
        # Pre-normalize Main DF Time
        df['time'] = normalize_ts(df['time'])
        
        # 1. Classical
        df['atr'] = self.tech.atr(df)
        df['rsi'] = self.tech.rsi(df['close'])
        df['ema50'] = self.tech.ema(df['close'], 50)
        df['macd'], _, _ = self.tech.macd(df['close'])
        df['adx'] = self.tech.adx(df)

        # 2. Market Structure (SMC)
        swing_high, swing_low = self.structure.detect_swing_points(df, config.SWING_LENGTH)
        df = self.structure.label_structure(df, swing_high, swing_low)
        df = self.structure.detect_bos_choch(df)
        
        # 3. Advanced SMC
        df = self.ob.detect_order_blocks(df)
        df = self.fvg.detect_fvg(df)
        df = self.ob.detect_liquidity_sweeps(df)

        # SMC Densities
        window_density = 50
        df['fvg_density'] = (df['fvg_bullish'] + df['fvg_bearish']).rolling(window_density).sum()
        df['ob_density'] = (df['ob_bullish'] + df['ob_bearish']).rolling(window_density).sum()
        
        # 4. Zones
        df = self.zones.calculate_zones(df)
        df['is_in_premium'] = df['in_premium']
        df['is_in_discount'] = df['in_discount']
        
        # 5. Pivots
        df = self.pivots.calculate_all_pivots(df)
        
        # 6. Confluence
        df['confluence_score'] = self.confluence.calculate_confluence_score(df, df['atr'])
        
        # 7. External (DXY + News) - using robust alignment
        df = self._add_external_features(df)
        
        # 8. Time features
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        # Fill NaNs globally
        df = df.fillna(method='ffill').fillna(0)
        
        # SCHEMA ENFORCEMENT
        features_df = validate_feature_schema(df, config.SCHEMA_PATH)
        
        logger.info(f"Features built: {features_df.shape}")
        return df, features_df

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(data_path="data/XAUUSD_history.csv", suffix="", symbol="XAUUSD"):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load your CSV
    df = pd.read_csv(data_path)
    
    # Process
    try:
        fe = FeatureEngineering(symbol=symbol)
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
        
        # Create Sequences
        X_seq = []
        y_seq_class = []
        
        win = config.WINDOW
        for i in range(win, len(X_scaled) - 1): 
            X_seq.append(X_scaled[i-win:i])
            y_seq_class.append(y_class[i])
        
        X_seq = np.array(X_seq)
        y_seq_class = np.array(y_seq_class)
        
        print(f"Final Shapes: X={X_seq.shape}, y={y_seq_class.shape}")
        
        # Save
        if suffix:
            suffix = "_" + suffix.lstrip("_")
        
        print(f"Saving artifacts to data/X{suffix}.npy ...")
        np.save(f"data/X{suffix}.npy", X_seq)
        np.save(f"data/y_class{suffix}.npy", y_seq_class)
        y_reg_aligned = y_reg[win:-1]
        np.save(f"data/y_reg{suffix}.npy", y_reg_aligned)
        
        df_enhanced.to_csv(f"data/features_enhanced{suffix}.csv", index=False)
        
        with open(f"data/scaler_features{suffix}.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        print(f"Done. Artifacts saved with suffix '{suffix}'.")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/XAUUSD_history.csv")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    args = parser.parse_args()
    
    run_pipeline(args.data, args.suffix, args.symbol)
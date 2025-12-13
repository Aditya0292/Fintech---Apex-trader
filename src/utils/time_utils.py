import pandas as pd
from datetime import datetime
import numpy as np
from typing import Union
from src.utils.logger import get_logger

logger = get_logger()

def normalize_ts(data: Union[pd.DataFrame, pd.Series, str, datetime]) -> Union[pd.DataFrame, pd.Series, datetime]:
    """
    Universal timestamp normalizer.
    Ensures all timestamps are:
    1. Converted to datetime64[ns]
    2. UTC-naive (tz-localize(None))
    3. Standardized
    
    Args:
        data: DataFrame, Series, string, or datetime object
        
    Returns:
        Normalized data of the same type
    """
    try:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            # Check standard time columns
            offset_aware = False
            for col in ['time', 'date', 'Date', 'Time', 'datetime', 'DateTime']:
                if col in df.columns:
                    try:
                        # Try mixed format ensuring UTC (Pandas 2.0+)
                        try:
                            s = pd.to_datetime(df[col], utc=True, format='mixed')
                        except (ValueError, TypeError):
                            # Fallback for older pandas or really weird formats
                            s = pd.to_datetime(df[col], utc=True)
                            
                        if pd.api.types.is_datetime64_any_dtype(s):
                             df[col] = s.dt.tz_localize(None)
                        else:
                             # Fallback: conversion failed to produce datetime dtype
                             df[col] = s
                        offset_aware = True 
                    except Exception as e:
                        logger.warning(f"Could not normalize column {col}: {e}")
            if not offset_aware:
                # Try index if it's DatetimeIndex
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index = df.index.tz_localize(None)
            return df
            
        elif isinstance(data, pd.Series):
             return pd.to_datetime(data, utc=True).dt.tz_localize(None)
            
        elif isinstance(data, (str, datetime)):
            ts = pd.to_datetime(data, utc=True)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts
            
        return data
    except Exception as e:
        logger.error(f"Timestamp normalization failed: {e}")
        return data

def validate_time_monotonicity(df: pd.DataFrame, time_col: str = 'time') -> bool:
    """Checks if the dataframe is sorted by time monotonically increasing."""
    if time_col not in df.columns:
        return False
    
    return df[time_col].is_monotonic_increasing

def align_datasets(df_main: pd.DataFrame, df_other: pd.DataFrame, on: str = 'time', tolerance: str = '4h') -> pd.DataFrame:
    """
    Safely aligns two datasets on a timestamp column after normalization.
    """
    df_main_norm = normalize_ts(df_main)
    df_other_norm = normalize_ts(df_other)
    
    # Ensure usage of exact column name
    if on not in df_main_norm.columns or on not in df_other_norm.columns:
        raise ValueError(f"Column '{on}' not found in one of the datasets.")
        
    # Sort for merge_asof requirements
    df_main_norm = df_main_norm.sort_values(on)
    df_other_norm = df_other_norm.sort_values(on)
    
    merged = pd.merge_asof(
        df_main_norm, 
        df_other_norm, 
        on=on, 
        direction='backward',
        tolerance=pd.Timedelta(tolerance) 
    )
    return merged

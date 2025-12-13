import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.time_utils import normalize_ts, align_datasets, validate_time_monotonicity

def test_normalize_ts_series():
    series = pd.Series(["2023-01-01 10:00:00", "2023-01-02 10:00:00"])
    norm = normalize_ts(series)
    assert norm.dt.tz is None
    assert pd.api.types.is_datetime64_ns_dtype(norm)

def test_normalize_ts_df():
    df = pd.DataFrame({
        "time": ["2023-01-01 10:00+00:00", "2023-01-02 10:00:00"],
        "val": [1, 2]
    })
    df_norm = normalize_ts(df)
    assert df_norm["time"].dt.tz is None
    assert pd.api.types.is_datetime64_ns_dtype(df_norm["time"])

def test_validate_time_monotonicity():
    df = pd.DataFrame({
        "time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    })
    assert validate_time_monotonicity(df)
    
    df_unsorted = pd.DataFrame({
        "time": pd.to_datetime(["2023-01-02", "2023-01-01"])
    })
    assert not validate_time_monotonicity(df_unsorted)

def test_align_datasets():
    df_main = pd.DataFrame({
        "time": pd.to_datetime(["2023-01-01 10:00", "2023-01-01 11:00"]),
        "main_val": [1, 2]
    })
    df_other = pd.DataFrame({
        "time": pd.to_datetime(["2023-01-01 09:50", "2023-01-01 10:55"]), # slightly before
        "other_val": [10, 20]
    })
    
    merged = align_datasets(df_main, df_other, tolerance='1h')
    
    # Expect 10:00 matches 09:50 -> 10
    # Expect 11:00 matches 10:55 -> 20
    assert merged.iloc[0]['other_val'] == 10
    assert merged.iloc[1]['other_val'] == 20

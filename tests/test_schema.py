import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import os
from src.utils.schema_validator import validate_feature_schema

@pytest.fixture
def mock_schema():
    schema = {
        "version": "1.0",
        "feature_count": 3,
        "features": ["col_a", "col_b", "col_c"]
    }
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(schema, f)
        path = f.name
    yield path
    os.remove(path)

def test_validate_exact_match(mock_schema):
    df = pd.DataFrame({
        "col_a": [1, 2],
        "col_b": [3, 4],
        "col_c": [5, 6]
    })
    df_valid = validate_feature_schema(df, mock_schema)
    assert list(df_valid.columns) == ["col_a", "col_b", "col_c"]
    assert df_valid.shape == (2, 3)

def test_validate_missing_col(mock_schema):
    df = pd.DataFrame({
        "col_a": [1, 2],
        "col_c": [5, 6]
    })
    df_valid = validate_feature_schema(df, mock_schema)
    assert "col_b" in df_valid.columns
    assert (df_valid["col_b"] == 0).all()
    assert list(df_valid.columns) == ["col_a", "col_b", "col_c"]

def test_validate_extra_col(mock_schema):
    df = pd.DataFrame({
        "col_a": [1, 2],
        "col_b": [3, 4],
        "col_c": [5, 6],
        "col_extra": [7, 8]
    })
    df_valid = validate_feature_schema(df, mock_schema)
    assert "col_extra" not in df_valid.columns
    assert df_valid.shape == (2, 3)

def test_validate_reordering(mock_schema):
    df = pd.DataFrame({
        "col_c": [5, 6],
        "col_a": [1, 2],
        "col_b": [3, 4]
    })
    df_valid = validate_feature_schema(df, mock_schema)
    assert list(df_valid.columns) == ["col_a", "col_b", "col_c"]

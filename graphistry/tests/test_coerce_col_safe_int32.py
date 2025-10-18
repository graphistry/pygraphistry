# -*- coding: utf-8 -*-
"""Test for coerce_col_safe int32 NaN handling bug fix

Bug report: Arrow optimization converts int64 → int32, then hypergraph operations
introduce NaN during pandas merges/reindexing. coerce_col_safe() tried to convert
float64 → int32 without filling NaN first, causing "Cannot convert non-finite values".
"""

import numpy as np
import pandas as pd
import pytest

from graphistry.hyper_dask import coerce_col_safe


def test_coerce_col_safe_int32_with_nan():
    """Test that coerce_col_safe handles int32 dtype with NaN values correctly.

    Regression test for bug where coerce_col_safe only handled int64 with NaN,
    but not int32, causing "Cannot convert non-finite values (NA or inf) to integer"
    error when Arrow optimizes int64 → int32.
    """
    # Create a float64 Series with NaN (simulates result of pandas merge/reindex)
    s = pd.Series([1.0, 2.0, np.nan, 4.0], name='test_col')
    assert s.dtype == 'float64'

    # Create int32 dtype (simulates Arrow optimization converting int64 → int32)
    target_dtype = pd.Series([1], dtype='int32').dtype
    assert target_dtype.name == 'int32'

    # Call coerce_col_safe - should NOT raise "Cannot convert non-finite values" error
    result = coerce_col_safe(s, target_dtype)

    # Verify NaN was filled with 0 and converted to int32
    assert result.dtype.name == 'int32'
    assert result.tolist() == [1, 2, 0, 4]  # NaN → 0


def test_coerce_col_safe_int64_with_nan():
    """Test that existing int64 NaN handling still works (regression test)."""
    s = pd.Series([1.0, 2.0, np.nan, 4.0], name='test_col')
    target_dtype = pd.Series([1], dtype='int64').dtype

    result = coerce_col_safe(s, target_dtype)

    assert result.dtype.name == 'int64'
    assert result.tolist() == [1, 2, 0, 4]


def test_coerce_col_safe_no_conversion_needed():
    """Test that coerce_col_safe returns original Series when dtypes match."""
    s = pd.Series([1, 2, 3, 4], dtype='int32', name='test_col')
    target_dtype = s.dtype

    result = coerce_col_safe(s, target_dtype)

    # Should return original Series when dtypes already match
    assert result is s


def test_coerce_col_safe_int32_without_nan():
    """Test that int32 conversion works when no NaN present."""
    s = pd.Series([1, 2, 3, 4], dtype='int64', name='test_col')
    target_dtype = pd.Series([1], dtype='int32').dtype

    result = coerce_col_safe(s, target_dtype)

    assert result.dtype.name == 'int32'
    assert result.tolist() == [1, 2, 3, 4]

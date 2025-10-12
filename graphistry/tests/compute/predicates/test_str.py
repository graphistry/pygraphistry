import pandas as pd
import pytest

from graphistry.compute.predicates.str import (
    contains,
    startswith,
    endswith,
    match,
    IsUpper, isupper
)


# Helper to check if cuDF is available
def has_cudf():
    try:
        import cudf
        return True
    except ImportError:
        return False


# Skip tests that require cuDF when it's not available
requires_cudf = pytest.mark.skipif(
    not has_cudf(),
    reason="cudf not installed"
)


def test_is_upper():
    d = isupper()
    assert isinstance(d, IsUpper)

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'IsUpper'

    d2 = IsUpper.from_json(o)
    assert isinstance(d2, IsUpper)


# ============= Contains Tests =============

def test_contains_pandas_basic():
    """Test basic contains functionality with pandas"""
    s = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
    predicate = contains('og')
    result = predicate(s)
    expected = pd.Series([False, True, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_regex():
    """Test regex patterns with pandas"""
    s = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
    predicate = contains('house|dog', regex=True)
    result = predicate(s)
    expected = pd.Series([False, True, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_case_insensitive():
    """Test case-insensitive matching with pandas"""
    s = pd.Series(['Mouse', 'dog', 'HOUSE', 'house'])
    predicate = contains('house', case=False)
    result = predicate(s)
    expected = pd.Series([False, False, True, True])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_na_default():
    """Test default NA handling with pandas"""
    s = pd.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og')
    result = predicate(s)
    assert result[0] is False
    assert result[1] is True
    assert pd.isna(result[2])
    assert result[3] is False


def test_contains_pandas_na_false():
    """Test NA=False handling with pandas"""
    s = pd.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og', na=False)
    result = predicate(s)
    expected = pd.Series([False, True, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_na_true():
    """Test NA=True handling with pandas"""
    s = pd.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og', na=True)
    result = predicate(s)
    expected = pd.Series([False, True, True, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_contains_cudf_basic():
    """Test basic contains functionality with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'house and parrot', '23'])
    predicate = contains('og')
    result = predicate(s)
    expected = cudf.Series([False, True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_contains_cudf_case_insensitive():
    """Test case-insensitive matching with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'HOUSE', 'house'])
    predicate = contains('house', case=False)
    result = predicate(s)
    expected = cudf.Series([False, False, True, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_contains_cudf_na_handling():
    """Test NA handling with cuDF"""
    import cudf
    
    # Test default NA behavior
    s = cudf.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og')
    result = predicate(s).to_pandas()
    assert result[0] is False
    assert result[1] is True
    assert pd.isna(result[2])
    assert result[3] is False
    
    # Test NA=False
    predicate = contains('og', na=False)
    result = predicate(s)
    expected = cudf.Series([False, True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())
    
    # Test NA=True
    predicate = contains('og', na=True)
    result = predicate(s)
    expected = cudf.Series([False, True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_contains_pandas_cudf_parity():
    """Verify identical behavior between pandas and cuDF"""
    import cudf
    
    # Create identical data
    data = ['Mouse', 'dog', None, 'HOUSE', 'house and parrot']
    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)
    
    # Test case-sensitive
    predicate = contains('house')
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)
    
    # Test case-insensitive
    predicate = contains('house', case=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)
    
    # Test with na=False
    predicate = contains('house', na=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)


# ============= Startswith Tests =============

def test_startswith_pandas_basic():
    """Test basic startswith functionality with pandas"""
    s = pd.Series(['Mouse', 'dog', 'house', 'Home'])
    predicate = startswith('ho')
    result = predicate(s)
    expected = pd.Series([False, False, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_startswith_pandas_na_handling():
    """Test NA handling with pandas"""
    s = pd.Series(['Mouse', None, 'house'])
    predicate = startswith('ho')
    result = predicate(s)
    assert result[0] is False
    assert pd.isna(result[1])
    assert result[2] is True

    # Test with na parameter
    predicate = startswith('ho', na=False)
    result = predicate(s)
    expected = pd.Series([False, False, True])
    pd.testing.assert_series_equal(result, expected)


def test_startswith_pandas_case_insensitive():
    """Test case-insensitive matching with pandas"""
    s = pd.Series(['John', 'john', 'JOHN', 'Jane'])
    predicate = startswith('john', case=False)
    result = predicate(s)
    expected = pd.Series([True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_startswith_cudf_basic():
    """Test basic startswith functionality with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'house', 'Home'])
    predicate = startswith('ho')
    result = predicate(s)
    expected = cudf.Series([False, False, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_startswith_cudf_na_handling():
    """Test NA handling with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', None, 'house'])

    # Default NA handling
    predicate = startswith('ho')
    result = predicate(s).to_pandas()
    assert result[0] is False
    assert pd.isna(result[1])
    assert result[2] is True

    # NA=False
    predicate = startswith('ho', na=False)
    result = predicate(s)
    expected = cudf.Series([False, False, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_startswith_cudf_case_insensitive():
    """Test case-insensitive matching with cuDF"""
    import cudf
    s = cudf.Series(['John', 'john', 'JOHN', 'Jane'])
    predicate = startswith('john', case=False)
    result = predicate(s)
    expected = cudf.Series([True, True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


# ============= Endswith Tests =============

def test_endswith_pandas_basic():
    """Test basic endswith functionality with pandas"""
    s = pd.Series(['Mouse', 'dog', 'house', 'Home'])
    predicate = endswith('se')
    result = predicate(s)
    expected = pd.Series([True, False, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_endswith_pandas_na_handling():
    """Test NA handling with pandas"""
    s = pd.Series(['Mouse', None, 'house'])
    predicate = endswith('se')
    result = predicate(s)
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is True

    # Test with na parameter
    predicate = endswith('se', na=False)
    result = predicate(s)
    expected = pd.Series([True, False, True])
    pd.testing.assert_series_equal(result, expected)


def test_endswith_pandas_case_insensitive():
    """Test case-insensitive matching with pandas"""
    s = pd.Series(['test.com', 'test.COM', 'test.Com', 'test.org'])
    predicate = endswith('.com', case=False)
    result = predicate(s)
    expected = pd.Series([True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_endswith_cudf_basic():
    """Test basic endswith functionality with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'house', 'Home'])
    predicate = endswith('se')
    result = predicate(s)
    expected = cudf.Series([True, False, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_endswith_cudf_na_handling():
    """Test NA handling with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', None, 'house'])

    # Default NA handling
    predicate = endswith('se')
    result = predicate(s).to_pandas()
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is True

    # NA=False
    predicate = endswith('se', na=False)
    result = predicate(s)
    expected = cudf.Series([True, False, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_endswith_cudf_case_insensitive():
    """Test case-insensitive matching with cuDF"""
    import cudf
    s = cudf.Series(['test.com', 'test.COM', 'test.Com', 'test.org'])
    predicate = endswith('.com', case=False)
    result = predicate(s)
    expected = cudf.Series([True, True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


# ============= Match Tests =============

def test_match_pandas_basic():
    """Test basic match functionality with pandas"""
    s = pd.Series(['Mouse', 'dog', 'house', '123'])
    predicate = match(r'\d+')
    result = predicate(s)
    expected = pd.Series([False, False, False, True])
    pd.testing.assert_series_equal(result, expected)


def test_match_pandas_case_insensitive():
    """Test case-insensitive matching with pandas"""
    s = pd.Series(['Mouse', 'mouse', 'MOUSE', 'dog'])
    predicate = match(r'mouse', case=False)
    result = predicate(s)
    expected = pd.Series([True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_match_pandas_na_handling():
    """Test NA handling with pandas"""
    s = pd.Series(['123', None, 'abc'])
    predicate = match(r'\d+')
    result = predicate(s)
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is False
    
    # Test with na=False
    predicate = match(r'\d+', na=False)
    result = predicate(s)
    expected = pd.Series([True, False, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_match_cudf_basic():
    """Test basic match functionality with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'house', '123'])
    predicate = match(r'\d+')
    result = predicate(s)
    expected = cudf.Series([False, False, False, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_match_cudf_case_insensitive():
    """Test case-insensitive matching with cuDF"""
    import cudf
    s = cudf.Series(['Mouse', 'mouse', 'MOUSE', 'dog'])
    predicate = match(r'mouse', case=False)
    result = predicate(s)
    expected = cudf.Series([True, True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_match_cudf_na_handling():
    """Test NA handling with cuDF"""
    import cudf
    s = cudf.Series(['123', None, 'abc'])
    
    # Default NA handling
    predicate = match(r'\d+')
    result = predicate(s).to_pandas()
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is False
    
    # NA=False
    predicate = match(r'\d+', na=False)
    result = predicate(s)
    expected = cudf.Series([True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_match_pandas_cudf_parity():
    """Verify identical behavior between pandas and cuDF for match"""
    import cudf
    
    # Create identical data
    data = ['Mouse', '123', None, 'MOUSE', 'dog123']
    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)
    
    # Test case-sensitive
    predicate = match(r'\d+')
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)
    
    # Test case-insensitive
    predicate = match(r'mouse', case=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)
    
    # Test with na=False
    predicate = match(r'\d+', na=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)


# ============= Edge Case Tests =============

def test_edge_cases_pandas():
    """Test edge cases with pandas"""
    # Empty strings
    s = pd.Series(['', 'test', ''])
    predicate = contains('')
    result = predicate(s)
    expected = pd.Series([True, True, True])  # Empty pattern matches everything
    pd.testing.assert_series_equal(result, expected)
    
    # Special regex characters
    s = pd.Series(['test.txt', 'test_txt', 'test?txt'])
    predicate = contains(r'test\.txt', regex=True)
    result = predicate(s)
    expected = pd.Series([True, False, False])
    pd.testing.assert_series_equal(result, expected)
    
    # Unicode
    s = pd.Series(['café', 'naïve', 'test'])
    predicate = contains('café')
    result = predicate(s)
    expected = pd.Series([True, False, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_edge_cases_cudf():
    """Test edge cases with cuDF"""
    import cudf
    
    # Empty strings
    s = cudf.Series(['', 'test', ''])
    predicate = contains('')
    result = predicate(s)
    expected = cudf.Series([True, True, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())
    
    # Special regex characters
    s = cudf.Series(['test.txt', 'test_txt', 'test?txt'])
    predicate = contains(r'test\.txt', regex=True)
    result = predicate(s)
    expected = cudf.Series([True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_all_predicates_pandas_cudf_parity():
    """Comprehensive test ensuring all predicates have identical behavior"""
    import cudf
    
    # Test data with various edge cases
    data = ['Test', 'test', None, '', '123', 'Test123', 'END']
    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)
    
    # Test all predicates with various parameters
    predicates = [
        contains('est'),
        contains('EST', case=False),
        contains('est', na=False),
        startswith('Test'),
        startswith('Test', na=False),
        endswith('END'),
        endswith('end', na=True),
        match(r'\d+'),
        match(r'test', case=False),
        match(r'\d+', na=False)
    ]
    
    for predicate in predicates:
        result_pandas = predicate(s_pandas)
        result_cudf = predicate(s_cudf).to_pandas()
        
        # Check if results are identical
        try:
            pd.testing.assert_series_equal(result_pandas, result_cudf)
        except AssertionError as e:
            pytest.fail(f"Parity check failed for {predicate.__class__.__name__}: {e}")

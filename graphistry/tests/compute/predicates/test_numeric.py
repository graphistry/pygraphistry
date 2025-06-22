import pandas as pd
import pytest
from datetime import datetime, date, time
from graphistry.compute.predicates.comparison import GT, gt, LT, lt, GE, ge, LE, le, EQ, eq, NE, ne, Between, between
from graphistry.compute.predicates.temporal_values import DateTimeValue, DateValue, TimeValue

def test_gt():
    # Test numeric GT (existing test)
    d = gt(1)
    assert isinstance(d, GT)
    assert d.val == 1

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'GT'
    assert o['val'] == 1

    d2 = GT.from_json(o)
    assert isinstance(d2, GT)
    assert d2.val == 1

def test_gt_temporal():
    """Test GT with temporal values"""
    # Test with pd.Timestamp
    ts = pd.Timestamp('2024-01-01', tz='UTC')
    d = gt(ts)
    assert isinstance(d, GT)
    assert isinstance(d.val, DateTimeValue)
    
    # Test serialization
    o = d.to_json()
    assert o['type'] == 'GT'
    assert isinstance(o['val'], dict)
    assert o['val']['type'] == 'datetime'
    assert o['val']['value'] == '2024-01-01T00:00:00+00:00'
    assert o['val']['timezone'] == 'UTC'
    
    # Test deserialization
    d2 = GT.from_json(o)
    assert isinstance(d2, GT)
    assert isinstance(d2.val, DateTimeValue)

def test_lt_temporal():
    """Test LT with temporal values"""
    # Test with datetime
    dt = datetime(2024, 1, 1, 12, 0, 0)
    d = lt(dt)
    assert isinstance(d, LT)
    assert isinstance(d.val, DateTimeValue)
    
    # Test with date
    date_val = date(2024, 1, 1)
    d = lt(date_val)
    assert isinstance(d, LT)
    assert isinstance(d.val, DateValue)
    
    # Test with time
    time_val = time(12, 0, 0)
    d = lt(time_val)
    assert isinstance(d, LT)
    assert isinstance(d.val, TimeValue)

def test_temporal_with_series():
    """Test temporal predicates with pandas Series"""
    # Create datetime series
    dates = pd.date_range('2024-01-01', periods=5, freq='D', tz='UTC')
    s = pd.Series(dates)
    
    # Test GT
    cutoff = pd.Timestamp('2024-01-03', tz='UTC')
    gt_pred = gt(cutoff)
    result = gt_pred(s)
    expected = pd.Series([False, False, False, True, True])
    assert result.equals(expected)
    
    # Test LT
    lt_pred = lt(cutoff)
    result = lt_pred(s)
    expected = pd.Series([True, True, False, False, False])
    assert result.equals(expected)
    
    # Test GE
    ge_pred = ge(cutoff)
    result = ge_pred(s)
    expected = pd.Series([False, False, True, True, True])
    assert result.equals(expected)
    
    # Test LE
    le_pred = le(cutoff)
    result = le_pred(s)
    expected = pd.Series([True, True, True, False, False])
    assert result.equals(expected)
    
    # Test EQ
    eq_pred = eq(cutoff)
    result = eq_pred(s)
    expected = pd.Series([False, False, True, False, False])
    assert result.equals(expected)
    
    # Test NE
    ne_pred = ne(cutoff)
    result = ne_pred(s)
    expected = pd.Series([True, True, False, True, True])
    assert result.equals(expected)

def test_between_temporal():
    """Test Between with temporal values"""
    # Test with timestamps
    start = pd.Timestamp('2024-01-02', tz='UTC')
    end = pd.Timestamp('2024-01-04', tz='UTC')
    
    # Create predicate
    between_pred = between(start, end, inclusive=True)
    assert isinstance(between_pred, Between)
    assert isinstance(between_pred.lower, DateTimeValue)
    assert isinstance(between_pred.upper, DateTimeValue)
    
    # Test serialization
    o = between_pred.to_json()
    assert o['type'] == 'Between'
    assert o['inclusive'] is True
    assert isinstance(o['lower'], dict)
    assert o['lower']['type'] == 'datetime'
    assert isinstance(o['upper'], dict)
    assert o['upper']['type'] == 'datetime'
    
    # Test with series
    dates = pd.date_range('2024-01-01', periods=5, freq='D', tz='UTC')
    s = pd.Series(dates)
    result = between_pred(s)
    expected = pd.Series([False, True, True, True, False])
    assert result.equals(expected)
    
    # Test exclusive
    between_pred_excl = between(start, end, inclusive=False)
    result = between_pred_excl(s)
    expected = pd.Series([False, False, True, False, False])
    assert result.equals(expected)

def test_temporal_with_tagged_dict():
    """Test predicates with tagged dict input"""
    # Test GT with tagged dict
    d = gt({
        "type": "datetime",
        "value": "2024-01-01T00:00:00Z",
        "timezone": "UTC"
    })
    assert isinstance(d, GT)
    assert isinstance(d.val, DateTimeValue)
    assert d.val.value == "2024-01-01T00:00:00Z"
    assert d.val.timezone == "UTC"
    
    # Test date dict
    d = lt({
        "type": "date",
        "value": "2024-01-01"
    })
    assert isinstance(d, LT)
    assert isinstance(d.val, DateValue)
    
    # Test time dict
    d = eq({
        "type": "time",
        "value": "12:00:00"
    })
    assert isinstance(d, EQ)
    assert isinstance(d.val, TimeValue)

def test_temporal_error_cases():
    """Test error handling for temporal predicates"""
    # Test raw string rejection
    with pytest.raises(ValueError, match="ambiguous"):
        gt("2024-01-01")
    
    with pytest.raises(ValueError, match="ambiguous"):
        between("2024-01-01", "2024-12-31")
    
    # Test type mismatch in Between - error happens during validation or call
    between_pred = between(100, pd.Timestamp('2024-01-01'))
    with pytest.raises(TypeError, match="same type"):
        between_pred.validate()
    
    # Test unknown temporal type
    with pytest.raises(ValueError, match="Unknown temporal type"):
        gt({"type": "duration", "value": "P1D"})

def test_timezone_handling():
    """Test timezone normalization in comparisons"""
    # Create series with different timezone
    dates = pd.date_range('2024-01-01', periods=3, freq='D', tz='US/Eastern')
    s = pd.Series(dates)
    
    # Compare with UTC timestamp
    cutoff = pd.Timestamp('2024-01-02T05:00:00', tz='UTC')  # Equivalent to 2024-01-02 00:00:00 EST
    gt_pred = gt(cutoff)
    result = gt_pred(s)
    expected = pd.Series([False, False, True])
    assert result.equals(expected)

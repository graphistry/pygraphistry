import pandas as pd
import pytest
from datetime import datetime, date, time
import pytz

from graphistry.compute.predicates.is_in import IsIn, is_in


def test_is_in():

    d = is_in([1, 2, 3])
    assert isinstance(d, IsIn)
    assert d.options == [1, 2, 3]

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'IsIn'

    d2 = IsIn.from_json(o)
    assert isinstance(d2, IsIn)
    assert d2.options == [1, 2, 3]


def test_is_in_with_datetime():
    """Test IsIn with datetime values"""
    dt1 = datetime(2023, 1, 1, 12, 0, 0)
    dt2 = datetime(2023, 1, 2, 12, 0, 0)
    dt3 = datetime(2023, 1, 3, 12, 0, 0)
    
    # Test with datetime objects
    pred = is_in([dt1, dt2])
    assert isinstance(pred, IsIn)
    # Should be converted to pd.Timestamp
    assert all(isinstance(opt, pd.Timestamp) for opt in pred.options)
    
    # Test with pandas series
    s = pd.Series([dt1, dt2, dt3, datetime(2023, 1, 4)])
    result = pred(s)
    assert result.tolist() == [True, True, False, False]


def test_is_in_with_pandas_timestamp():
    """Test IsIn with pandas Timestamp values"""
    ts1 = pd.Timestamp('2023-01-01 12:00:00')
    ts2 = pd.Timestamp('2023-01-02 12:00:00')
    ts3 = pd.Timestamp('2023-01-03 12:00:00')
    
    pred = is_in([ts1, ts2])
    assert isinstance(pred, IsIn)
    assert pred.options == [ts1, ts2]
    
    # Test with series
    s = pd.Series([ts1, ts2, ts3])
    result = pred(s)
    assert result.tolist() == [True, True, False]


def test_is_in_with_date():
    """Test IsIn with date values"""
    d1 = date(2023, 1, 1)
    d2 = date(2023, 1, 2)
    
    pred = is_in([d1, d2])
    assert isinstance(pred, IsIn)
    # Should be converted to pd.Timestamp
    assert all(isinstance(opt, pd.Timestamp) for opt in pred.options)
    
    # Test with datetime series (will compare dates)
    s = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    result = pred(s)
    assert result.tolist() == [True, True, False]


def test_is_in_with_time():
    """Test IsIn with time values"""
    t1 = time(12, 0, 0)
    t2 = time(13, 0, 0)
    
    pred = is_in([t1, t2])
    assert isinstance(pred, IsIn)
    # Time values should remain as time objects
    assert all(isinstance(opt, time) for opt in pred.options)
    
    # Test with datetime series (will extract time component)
    dt_series = pd.Series(pd.to_datetime([
        '2023-01-01 12:00:00',
        '2023-01-01 13:00:00', 
        '2023-01-01 14:00:00'
    ]))
    result = pred(dt_series)
    assert result.tolist() == [True, True, False]


def test_is_in_with_timezone():
    """Test IsIn with timezone-aware datetime values"""
    utc = pytz.UTC
    
    # Create timezone-aware timestamps
    ts1_utc = pd.Timestamp('2023-01-01 12:00:00', tz=utc)
    ts2_utc = pd.Timestamp('2023-01-01 17:00:00', tz=utc)  # Same as 12:00 EST
    
    pred = is_in([ts1_utc])
    
    # Test with timezone-aware series
    s = pd.Series([ts1_utc, ts2_utc, pd.Timestamp('2023-01-01 13:00:00', tz=utc)])
    result = pred(s)
    assert result.tolist() == [True, False, False]


def test_is_in_with_tagged_dict():
    """Test IsIn with tagged dictionary values (JSON deserialization)"""
    # Test datetime tagged dict
    dt_dict = {
        "type": "datetime",
        "value": "2023-01-01T12:00:00",
        "timezone": "UTC"
    }
    
    pred = is_in([dt_dict, {"type": "datetime", "value": "2023-01-02T12:00:00"}])
    assert isinstance(pred, IsIn)
    assert all(isinstance(opt, pd.Timestamp) for opt in pred.options)
    
    # Test with series
    s = pd.Series(pd.to_datetime(['2023-01-01 12:00:00', '2023-01-02 12:00:00', '2023-01-03 12:00:00']))
    s = s.dt.tz_localize('UTC')
    result = pred(s)
    assert result.tolist() == [True, True, False]


def test_is_in_mixed_types():
    """Test IsIn rejects mixed temporal and numeric values"""
    dt = datetime(2023, 1, 1)
    
    # Should raise error for temporal + numeric mix
    with pytest.raises(ValueError, match="Cannot mix temporal and numeric"):
        is_in([42, dt])
    
    # But temporal + strings should work
    pred = is_in(["hello", dt, None])
    s = pd.Series(["hello", pd.Timestamp(dt), None, "world"])
    result = pred(s)
    assert result.tolist() == [True, True, True, False]


def test_is_in_validation():
    """Test IsIn validation with temporal values"""
    dt = datetime(2023, 1, 1)
    
    # Test with temporal + strings (should work)
    pred = is_in([dt, "test", None])
    pred.validate()
    
    # Test with just numeric (should work)
    pred2 = is_in([123, 456.7])
    pred2.validate()


def test_is_in_json_serialization_with_temporal():
    """Test JSON serialization/deserialization with temporal values"""
    dt = datetime(2023, 1, 1, 12, 30, 45)
    pred = is_in([dt, "test"])
    
    # Serialize to JSON
    json_dict = pred.to_json()
    assert json_dict['type'] == 'IsIn'
    assert 'options' in json_dict
    
    # Deserialize from JSON
    pred2 = IsIn.from_json(json_dict)
    assert isinstance(pred2, IsIn)
    # The datetime should be preserved (as pd.Timestamp)
    assert len(pred2.options) == 2
    assert isinstance(pred2.options[0], pd.Timestamp)
    assert pred2.options[1] == "test"

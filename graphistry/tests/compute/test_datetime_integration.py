"""Integration tests for GFQL datetime predicates end-to-end usage"""

import pandas as pd
import pytest
from datetime import datetime, date, time
import pytz

from graphistry import n
from graphistry.compute import (
    DateTimeValue, DateValue, TimeValue,
    gt, lt, between, eq, is_in
)


def test_datetime_predicates_in_chain():
    """Test datetime predicates in a chain operation"""
    # Create sample data with datetime columns
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-02 14:30:00',
            '2023-01-03 09:00:00',
            '2023-01-04 16:45:00'
        ]),
        'value': [10, 20, 30, 40]
    })
    
    # Test GT with datetime
    dt_threshold = datetime(2023, 1, 2, 12, 0, 0)
    result = df[gt(pd.Timestamp(dt_threshold))(df['timestamp'])]
    assert len(result) == 3  # All timestamps after 2023-01-02 12:00:00
    assert result['value'].tolist() == [20, 30, 40]
    
    # Test Between with dates (converts to midnight timestamps)
    start_date = date(2023, 1, 2)  # 2023-01-02 00:00:00
    end_date = date(2023, 1, 3)    # 2023-01-03 00:00:00
    result = df[between(pd.Timestamp(start_date), pd.Timestamp(end_date))(df['timestamp'])]
    # Only 2023-01-02 14:30:00 falls between these dates (inclusive)
    assert len(result) == 1
    assert result['value'].tolist() == [20]


def test_datetime_with_timezone():
    """Test timezone-aware datetime comparisons"""
    utc = pytz.UTC
    
    # Create timezone-aware data
    df = pd.DataFrame({
        'timestamp': [
            pd.Timestamp('2023-01-01 10:00:00', tz=utc),
            pd.Timestamp('2023-01-01 15:00:00', tz=utc),  # 10:00 EST
            pd.Timestamp('2023-01-01 18:00:00', tz=utc),  # 13:00 EST
        ],
        'event': ['A', 'B', 'C']
    })
    
    # Test with UTC timestamp
    threshold = pd.Timestamp('2023-01-01 14:00:00', tz=utc)
    result = df[gt(threshold)(df['timestamp'])]
    assert result['event'].tolist() == ['B', 'C']


def test_time_predicates():
    """Test time-only predicates"""
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2023-01-01 09:00:00',
            '2023-01-01 12:30:00',
            '2023-01-01 15:45:00',
            '2023-01-02 09:00:00',
            '2023-01-02 18:00:00'
        ]),
        'event': ['morning1', 'lunch', 'afternoon', 'morning2', 'evening']
    })
    
    # Test IsIn with time values
    morning_times = [time(9, 0, 0), time(9, 30, 0)]
    pred = is_in(morning_times)
    result = df[pred(df['timestamp'])]
    assert result['event'].tolist() == ['morning1', 'morning2']


def test_mixed_temporal_types():
    """Test predicates with mixed temporal types"""
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']).date,
        'datetime': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-02 14:00:00', 
            '2023-01-03 16:00:00'
        ]),
        'id': [1, 2, 3]
    })
    
    # Test date comparison
    target_date = date(2023, 1, 2)
    result = df[eq(target_date)(pd.to_datetime(df['date']))]
    assert result['id'].tolist() == [2]


def test_json_serialization_roundtrip():
    """Test JSON serialization/deserialization with temporal predicates"""
    # Create predicate with temporal value
    dt = datetime(2023, 1, 1, 12, 0, 0)
    pred = gt(pd.Timestamp(dt))
    
    # Serialize to JSON
    json_data = pred.to_json()
    assert json_data['type'] == 'GT'
    
    # Deserialize and verify it still works
    from graphistry.compute.predicates.comparison import GT
    pred2 = GT.from_json(json_data)
    
    # Test on data
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 14:00:00']),
        'value': [1, 2]
    })
    result = pred2(df['timestamp'])
    assert result.tolist() == [False, True]


def test_temporal_values_api():
    """Test the temporal value classes directly"""
    # DateTimeValue
    dt_val = DateTimeValue("2023-01-01T12:00:00", "UTC")
    assert dt_val.to_json() == {
        "type": "datetime",
        "value": "2023-01-01T12:00:00",
        "timezone": "UTC"
    }
    
    # DateValue
    date_val = DateValue("2023-01-01")
    assert date_val.to_json() == {
        "type": "date",
        "value": "2023-01-01"
    }
    
    # TimeValue
    time_val = TimeValue("14:30:00")
    assert time_val.to_json() == {
        "type": "time",
        "value": "14:30:00"
    }
    
    # Factory method
    from graphistry.compute.predicates.temporal_values import temporal_value_from_json
    dt_val2 = temporal_value_from_json({
        "type": "datetime",
        "value": "2023-01-01T12:00:00",
        "timezone": "US/Eastern"
    })
    assert isinstance(dt_val2, DateTimeValue)
    assert dt_val2.timezone == "US/Eastern"

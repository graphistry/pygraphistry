"""Test that wire protocol dictionaries work directly in temporal predicates"""

import pandas as pd
from datetime import time

from graphistry.compute.predicates import gt, ge, between, is_in


class TestWireProtocolDicts:
    """Test using wire protocol dictionaries directly in predicates"""

    def test_datetime_wire_dict(self):
        """Test datetime wire protocol dicts in comparison predicates"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5,
                                       freq='D', tz='UTC')
        })

        # Test with gt
        wire_dict = {"type": "datetime", "value": "2023-01-03T00:00:00",
                     "timezone": "UTC"}
        pred = gt(wire_dict)
        result = pred(df['timestamp'])
        assert result.tolist() == [False, False, False, True, True]

        # Test with between
        start_dict = {"type": "datetime", "value": "2023-01-02T00:00:00",
                      "timezone": "UTC"}
        end_dict = {"type": "datetime", "value": "2023-01-04T00:00:00",
                    "timezone": "UTC"}
        pred2 = between(start_dict, end_dict)
        result2 = pred2(df['timestamp'])
        assert result2.tolist() == [False, True, True, True, False]

    def test_date_wire_dict(self):
        """Test date wire protocol dicts"""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D').date
        })

        wire_dict = {"type": "date", "value": "2023-01-03"}
        pred = ge(wire_dict)
        result = pred(df['date'])
        assert result.tolist() == [False, False, True, True, True]

    def test_time_wire_dict(self):
        """Test time wire protocol dicts"""
        times = [time(9, 0), time(12, 0), time(15, 0), time(18, 0)]
        df = pd.DataFrame({'time': times})

        wire_dict = {"type": "time", "value": "12:00:00"}
        pred = gt(wire_dict)
        result = pred(df['time'])
        assert result.tolist() == [False, False, True, True]

    def test_is_in_wire_dicts(self):
        """Test is_in with wire protocol dicts"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5,
                                       freq='D', tz='UTC')
        })

        wire_dicts = [
            {"type": "datetime", "value": "2023-01-02T00:00:00",
             "timezone": "UTC"},
            {"type": "datetime", "value": "2023-01-04T00:00:00",
             "timezone": "UTC"}
        ]
        pred = is_in(wire_dicts)
        result = pred(df['timestamp'])
        assert result.tolist() == [False, True, False, True, False]

    def test_wire_dict_serialization(self):
        """Test that wire dicts serialize correctly in to_json()"""
        wire_dict = {"type": "datetime", "value": "2023-01-01T00:00:00",
                     "timezone": "UTC"}
        pred = gt(wire_dict)

        json_obj = pred.to_json()
        assert json_obj["type"] == "GT"
        assert json_obj["val"]["type"] == "datetime"
        assert json_obj["val"]["value"] == "2023-01-01T00:00:00"
        assert json_obj["val"]["timezone"] == "UTC"

    def test_mixed_types_with_wire_dicts(self):
        """Test mixing wire dicts with regular values"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5,
                                       freq='D', tz='UTC'),
            'value': [1, 2, 3, 4, 5]
        })

        # Numeric comparison still works normally
        pred_num = gt(3)
        result_num = pred_num(df['value'])
        assert result_num.tolist() == [False, False, False, True, True]

        # Temporal comparison with wire dict
        wire_dict = {"type": "datetime", "value": "2023-01-03T00:00:00",
                     "timezone": "UTC"}
        pred_temporal = gt(wire_dict)
        result_temporal = pred_temporal(df['timestamp'])
        assert result_temporal.tolist() == [False, False, False, True, True]

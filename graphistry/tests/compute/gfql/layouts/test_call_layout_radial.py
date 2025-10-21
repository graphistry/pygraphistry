import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.gfql.call_executor import execute_call
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
from graphistry.tests.test_compute import CGFull


class TestGFQLRingLayoutsCPU:
    """Dedicated tests for ring_* GFQL calls across modes."""

    def _graph_with_numeric(self):
        edges = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        nodes = pd.DataFrame({
            'node': [0, 1, 2],
            'score': [0.1, 0.6, 0.9]
        })
        return CGFull()\
            .edges(edges)\
            .nodes(nodes)\
            .bind(source='source', destination='target', node='node')

    def _graph_with_categories(self):
        edges = pd.DataFrame({
            'source': [0, 1, 2, 3],
            'target': [1, 2, 3, 0]
        })
        nodes = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'segment': ['a', 'b', 'a', 'c']
        })
        return CGFull()\
            .edges(edges)\
            .nodes(nodes)\
            .bind(source='source', destination='target', node='node')

    def _graph_with_time(self):
        edges = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        nodes = pd.DataFrame({
            'node': [0, 1, 2],
            'ts': pd.to_datetime([
                '2024-01-01T00:00:00',
                '2024-01-01T00:01:00',
                '2024-01-01T00:02:00'
            ])
        })
        return CGFull()\
            .edges(edges)\
            .nodes(nodes)\
            .bind(source='source', destination='target', node='node')

    def test_continuous_mode_basic(self):
        """Continuous mode should bind coordinates from a numeric column."""
        g = self._graph_with_numeric()

        result = execute_call(
            g,
            'ring_continuous_layout',
            {'ring_col': 'score'},
            Engine.PANDAS
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)
        assert result._point_x == 'x'
        assert result._point_y == 'y'

    def test_default_mode_is_continuous(self):
        """Omitting mode should default to continuous layout."""
        g = self._graph_with_numeric()

        result = execute_call(
            g,
            'ring_continuous_layout',
            {'ring_col': 'score'},
            Engine.PANDAS
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)

    def test_categorical_mode_basic(self):
        """Categorical mode should position nodes by category without numeric params."""
        g = self._graph_with_categories()

        result = execute_call(
            g,
            'ring_categorical_layout',
            {
                'ring_col': 'segment',
                'order': ['a', 'b', 'c']
            },
            Engine.PANDAS
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)

    def test_categorical_rejects_continuous_params(self):
        """Passing continuous-only params to categorical mode should fail validation."""
        g = self._graph_with_categories()

        with pytest.raises(GFQLTypeError) as exc_info:
            execute_call(
                g,
                'ring_categorical_layout',
                {'ring_col': 'segment', 'num_rings': 4},
                Engine.PANDAS
            )
        assert exc_info.value.code == ErrorCode.E303

    def test_time_mode_iso_strings(self):
        """Time mode should accept ISO strings for range and format label shim."""
        g = self._graph_with_time()

        result = execute_call(
            g,
            'time_ring_layout',
            {
                'time_col': 'ts',
                'time_start': '2024-01-01T00:00:00',
                'time_end': '2024-01-01T00:02:00',
                'num_rings': 3,
                'format_label': lambda ts, *_: pd.Timestamp(ts).strftime('%H:%M')
            },
            Engine.PANDAS
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)

    def test_time_mode_invalid_strings_raise(self):
        """Time mode should raise when time bounds are invalid strings."""
        g = self._graph_with_time()

        with pytest.raises(GFQLTypeError):
            execute_call(
                g,
                'time_ring_layout',
                {
                    'time_col': 'ts',
                    'time_start': 'not-a-date',
                    'time_end': '2024-01-01T00:02:00'
                },
                Engine.PANDAS
            )

    def test_categorical_requires_ring_col(self):
        """Categorical layout requires specifying ring_col."""
        g = self._graph_with_numeric()

        with pytest.raises(GFQLTypeError) as exc_info:
            execute_call(
                g,
                'ring_categorical_layout',
                {},
                Engine.PANDAS
            )
        assert exc_info.value.code == ErrorCode.E105

"""GPU smoke tests for GFQL radial layout call."""

import os
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.gfql.call_executor import execute_call
from graphistry.tests.test_compute import CGFull


skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestGFQLRingLayoutsGPU:
    """Ensure ring_* layouts execute on cudf-backed graphs."""

    @skip_gpu
    def test_continuous_mode_cudf(self):
        import cudf

        edges = cudf.from_pandas(pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        }))
        nodes = cudf.from_pandas(pd.DataFrame({
            'node': [0, 1, 2],
            'score': [0.2, 0.5, 0.8]
        }))

        g = CGFull()\
            .edges(edges)\
            .nodes(nodes)\
            .bind(source='source', destination='target', node='node')

        result = execute_call(
            g,
            'ring_continuous_layout',
            {'ring_col': 'score'},
            Engine.CUDF
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)
        assert len(result._nodes) == 3

    @skip_gpu
    def test_categorical_mode_cudf(self):
        import cudf

        edges = cudf.from_pandas(pd.DataFrame({
            'source': [0, 1, 2, 3],
            'target': [1, 2, 3, 0]
        }))
        nodes = cudf.from_pandas(pd.DataFrame({
            'node': [0, 1, 2, 3],
            'segment': ['a', 'b', 'a', 'c']
        }))

        g = CGFull()\
            .edges(edges)\
            .nodes(nodes)\
            .bind(source='source', destination='target', node='node')

        result = execute_call(
            g,
            'ring_categorical_layout',
            {'ring_col': 'segment'},
            Engine.CUDF
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)
        assert len(result._nodes) == 4

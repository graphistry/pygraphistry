"""Timestamp scalar and membership encodings retain exact units and admission."""
from datetime import datetime, timezone

import numpy as np
import pytest

from graphistry.Engine import Engine
from graphistry.compute.gfql.index.property_keys import property_query_values
from graphistry.compute.gfql.index.registry import NodePropIndex


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("predicate", ["1970-01-01T00:00:00.001", datetime(1970, 1, 1, 0, 0, 0, 1000), [],
                                        ["1970-01-01T00:00:00.002", "1970-01-01T00:00:00.001", "1970-01-01T00:00:00.001"]])
def test_timestamp_scalar_empty_and_membership_preserve_sorted_ticks(backend, unit, predicate):
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    index = NodePropIndex(key_col="v", keys_sorted=xp.asarray([1], dtype=xp.int64),
                          group_offsets=xp.asarray([0, 1]), row_positions=xp.asarray([0]),
                          backend=backend, engine=Engine.PANDAS if backend == "numpy" else Engine.CUDF,
                          timestamp_dtype=np.dtype("datetime64[" + unit + "]"))
    result = property_query_values(index, predicate, xp)
    scale = {"ms": 1, "us": 1000, "ns": 1000000}[unit]
    expected = [] if predicate == [] else ([scale, 2 * scale] if isinstance(predicate, list) else [scale])
    host = np.asarray if backend == "numpy" else xp.asnumpy
    np.testing.assert_array_equal(host(result), expected)
    assert result.dtype == xp.int64


@pytest.mark.parametrize("predicate", ["not-a-timestamp", datetime(1970, 1, 1, tzinfo=timezone.utc), np.datetime64("NaT"), 1])
def test_timestamp_ambiguous_or_invalid_literals_decline_to_canonical_filter(predicate):
    index = NodePropIndex(key_col="v", keys_sorted=np.asarray([1]), group_offsets=np.asarray([0, 1]),
                          row_positions=np.asarray([0]), backend="numpy", engine=Engine.PANDAS,
                          timestamp_dtype=np.dtype("datetime64[ns]"))
    assert property_query_values(index, predicate, np) is None

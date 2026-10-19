"""Adjacency probes preserve promoted identifiers and independent CSR outputs."""
import numpy as np
import pytest

from graphistry.Engine import Engine
from graphistry.compute.gfql.index.lookup import lookup_edge_rows
from graphistry.compute.gfql.index.registry import AdjacencyIndex


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("key_dtype", ["int32", "uint32", "int64"])
@pytest.mark.parametrize("frontier", [[1], [8], [9], [2**32 + 1], [], [1, 8]])
def test_adjacency_probe_rows_promoted_ids_and_owned_outputs(backend, key_dtype, frontier):
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    keys = xp.asarray([0, 1, 8], dtype=key_dtype)
    positions = xp.asarray([2, 0, 3, 1], dtype=xp.int64)
    seeds = xp.asarray(frontier, dtype=xp.int64)
    index = AdjacencyIndex(
        kind="edge_out_adj", key_col="src", other_col="dst", edge_id_col=None,
        keys_sorted=keys, group_offsets=xp.asarray([0, 1, 3, 4], dtype=xp.int64),
        row_positions=positions, other_values=xp.asarray([5, 6, 7, 8]),
        backend=backend, engine=Engine.PANDAS if backend == "numpy" else Engine.CUDF,
    )
    rows, matched = lookup_edge_rows(index, seeds, xp)
    host = np.asarray if backend == "numpy" else xp.asnumpy
    expected_ids = [value for value in frontier if value in [0, 1, 8]]
    expected_rows = [row for value in expected_ids for row in {0: [2], 1: [0, 3], 8: [1]}[value]]
    np.testing.assert_array_equal(host(rows), expected_rows)
    np.testing.assert_array_equal(host(matched), expected_ids)
    assert matched.dtype == xp.promote_types(seeds.dtype, keys.dtype)
    if rows.size:
        rows[0] = 99
    if matched.size:
        matched[0] = 99
    np.testing.assert_array_equal(host(positions), [2, 0, 3, 1])
    np.testing.assert_array_equal(host(seeds), frontier)

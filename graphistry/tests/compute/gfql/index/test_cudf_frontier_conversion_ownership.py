"""Frontier conversion preserves nullable IDs, promotion and owned buffers."""
import numpy as np
import pytest

from graphistry.compute.chain_fast_paths import _ids_to_key_array


@pytest.mark.parametrize("values,dtype,key_dtype,expected", [
    ([], "int64", "int64", []),
    ([1], "int64", "int64", [1]),
    ([None], "int64", "int64", []),
    ([None, 1, 1, 2], "int64", "int64", [1, 2]),
    ([3, 1, 3, 2], "int64", "int64", [1, 2, 3]),
    ([1.0, float("nan"), 2.0], "float64", "float64", [1.0, 2.0]),
    ([float("nan")], "float64", "float64", []),
    ([2**53 + 1], "int64", "int64", [2**53 + 1]),
    ([2**63 + 1], "uint64", "uint64", [2**63 + 1]),
    ([2**63 + 1], "uint64", "int64", None),
    (["a", None], "str", "int64", None),
    ([1], "int32", "int64", [1]),
    ([0.0, -0.0], "float64", "float64", [0.0]),
])
@pytest.mark.parametrize("nan_as_null", [True, False])
def test_cudf_frontier_values_dtype_and_independent_buffers(
    values, dtype, key_dtype, expected, nan_as_null,
):
    cudf = pytest.importorskip("cudf")
    cp = pytest.importorskip("cupy")
    from cudf.testing import assert_series_equal

    series = cudf.Series(values, dtype=dtype, nan_as_null=nan_as_null)
    saved = series.copy(deep=True)
    keys = cp.asarray([0, 1], dtype=key_dtype)
    saved_keys = keys.copy()
    result = _ids_to_key_array(series, keys, cp)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result.dtype == np.promote_types(np.dtype(dtype), np.dtype(key_dtype))
        cp.testing.assert_array_equal(result, cp.asarray(expected, dtype=result.dtype))
        if result.size:
            result[0] = 0
    assert_series_equal(series, saved)
    cp.testing.assert_array_equal(keys, saved_keys)


@pytest.mark.parametrize("seed", [0, 1, 8])
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_cudf_seeded_hop_matches_scan_and_owns_frames(seed, direction):
    import graphistry
    from graphistry import n, e_forward, e_reverse

    edge = e_forward() if direction == "forward" else e_reverse()
    query = [n({"v": seed}), edge, n()]
    cudf = pytest.importorskip("cudf")
    from cudf.testing import assert_frame_equal

    nodes = cudf.DataFrame({"id": [0, 1, 2, 3], "v": [0, 1, 2, 3]})
    edges = cudf.DataFrame({"src": [0, 1, 1, 2], "dst": [1, 2, 3, 0], "label": [0, 1, 2, 3]})
    saved_nodes, saved_edges = nodes.copy(deep=True), edges.copy(deep=True)
    graph = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    indexed = graph.create_index("node_id", engine="cudf")
    indexed = indexed.create_index("node_prop", column="v", engine="cudf")
    indexed = indexed.create_index("edge_out_adj", engine="cudf")
    indexed = indexed.create_index("edge_in_adj", engine="cudf")
    expected = graph.gfql(query, engine="cudf", index_policy="off")
    result = indexed.gfql(query, engine="cudf", index_policy="use")
    for name in ("nodes", "edges"):
        actual_frame = getattr(result, "_" + name)
        expected_frame = getattr(expected, "_" + name)
        keys = ["id"] if name == "nodes" else ["label"]
        assert_frame_equal(actual_frame.sort_values(keys).reset_index(drop=True), expected_frame.sort_values(keys).reset_index(drop=True))
        if len(actual_frame):
            actual_frame[keys[0]] = -1
    assert_frame_equal(graph._nodes, saved_nodes)
    assert_frame_equal(indexed._nodes, saved_nodes)
    assert_frame_equal(graph._edges, saved_edges)
    assert_frame_equal(indexed._edges, saved_edges)

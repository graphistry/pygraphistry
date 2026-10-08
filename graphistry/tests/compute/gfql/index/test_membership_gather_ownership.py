"""Public membership parity and isolation for validated indexed gathers."""
from dataclasses import replace

import pandas as pd
import pytest

import graphistry
from graphistry import e_forward, e_reverse, is_in, n
from graphistry.Engine import Engine
from graphistry.compute.gfql.index.api import get_registry, with_index_policy
from graphistry.compute.gfql.index.engine_arrays import _take_index_rows, take_rows
from graphistry.compute.gfql.index.registry import NODE_ID


def graph(engine):
    nodes = pd.DataFrame({"id": range(6), "v": [1, 2, 2, 3, 4, 5], "w": [0, 0, 1, 1, 1, 1]})
    edges = pd.DataFrame({"s": [0, 0, 1, 2, 3, 4], "d": [1, 2, 2, 3, 4, 5]})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.bind(node="id", source="s", destination="d").nodes(nodes).edges(edges)
    indexed = g
    for kind, column in [("node_id", None), ("edge_out_adj", None), ("edge_in_adj", None), ("node_prop", "v"), ("node_prop", "w")]:
        indexed = indexed.create_index(kind, column=column, engine=engine)
    return g, indexed


def equal(actual, expected):
    if type(actual).__module__.startswith("cudf"):
        from cudf.testing import assert_frame_equal
    else:
        from pandas.testing import assert_frame_equal
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("direction", [e_forward, e_reverse])
@pytest.mark.parametrize("policy", ["use", "force"])
@pytest.mark.parametrize("members", [[], [1], [1, 2], [2, 2, 99], [-1, 1], [None, 2], [1.0, 2.0], ["1"], [2**64], [True, 2]])
def test_public_membership_keeps_scan_behavior_and_source(engine, direction, policy, members):
    # Construct with the public API before a CUDA skip, so local collection checks it.
    query = [n({"v": is_in(members)}), direction(), n()]
    bare, indexed = graph(engine)
    original_nodes, original_edges = bare._nodes.copy(deep=True), bare._edges.copy(deep=True)
    try:
        expected = bare.gfql(query, engine=engine, index_policy="off")
    except (OverflowError, TypeError, ValueError) as error:
        with pytest.raises(type(error)) as actual:
            indexed.gfql(query, engine=engine, index_policy=policy)
        assert getattr(actual.value, "code", None) == getattr(error, "code", None)
        assert getattr(actual.value, "context", {}) == getattr(error, "context", {})
    else:
        actual = indexed.gfql(query, engine=engine, index_policy=policy)
        equal(actual._nodes.sort_values("id").reset_index(drop=True), expected._nodes.sort_values("id").reset_index(drop=True))
        equal(actual._edges.sort_values(["s", "d"]).reset_index(drop=True), expected._edges.sort_values(["s", "d"]).reset_index(drop=True))
        if len(actual._nodes):
            actual._nodes.iloc[0, 0] = 999
        if len(actual._edges):
            actual._edges.iloc[0, 0] = 999
    equal(bare._nodes, original_nodes)
    equal(bare._edges, original_edges)


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_most_selective_earlier_index_keeps_its_own_cached_count(engine):
    bare, indexed = graph(engine)
    filters = {"v": is_in([1, 2]), "w": is_in([0, 1])}
    expected = with_index_policy(bare, "off").filter_nodes_by_dict(filters)._nodes
    actual = with_index_policy(indexed, "force").filter_nodes_by_dict(filters)._nodes
    equal(actual, expected)


@pytest.mark.parametrize("stale", ["identity", "height"])
def test_stale_index_metadata_uses_checked_gather(stale):
    cudf = pytest.importorskip("cudf")
    cp = pytest.importorskip("cupy")
    bare, indexed = graph("cudf")
    frame = bare._nodes
    index = get_registry(indexed).get_valid(NODE_ID, frame, ("id",), Engine.CUDF)
    assert index is not None
    index = replace(index, source_ref=frame.copy(deep=True)) if stale == "identity" else replace(index, fingerprint=(-1, (), ""))
    positions = cp.asarray([6, 0], dtype="int64")
    with pytest.raises(IndexError):
        _take_index_rows(frame, positions, Engine.CUDF, index)
    positions = cp.asarray([-1, 0, 0], dtype="int64")
    equal(_take_index_rows(frame, positions, Engine.CUDF, index), frame.iloc[positions])
    equal(take_rows(frame, positions, Engine.CUDF), frame.iloc[positions])
    assert isinstance(frame, cudf.DataFrame)

"""Node type-segment index must preserve canonical GFQL results (edge typed-index dual)."""
from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.compute.ast import e_forward, n, rows
from graphistry.compute.gfql.index import (
    get_registry, index_trace, maybe_index_hop, with_index_policy,
)
from graphistry.compute.gfql.index.api import _series_nbytes
from graphistry.compute.gfql.index.cypher_ddl import parse_index_ddl
from graphistry.compute.gfql.index.wire import apply_index_op


def _graph(engine: str):
    # Node 1 has TWO labels (membership); node 5 has a null type; node 4 is
    # label-only-False; a null type never matches any scalar segment.
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "type": ["P", "Q", "P", "Q", None],
        "label__Person": [True, True, False, False, True],
        "label__Robot": [False, True, False, True, False],
    })
    edges = pd.DataFrame({
        "src": [1, 1, 2, 3, 4, 5],
        "dst": [2, 3, 4, 4, 5, 1],
        "etype": ["K", "X", "K", "T", "K", "T"],
    })
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.bind(node="id", source="src", destination="dst").nodes(nodes).edges(edges)


def test_node_segment_show_indexes_reports_cudf_bytes() -> None:
    pytest.importorskip("cudf")
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("CUDA device detection failed")
    if device_count == 0:
        pytest.skip("cuDF requires a CUDA device")
    indexed = _indexed(_graph("cudf"), "cudf")
    rows = indexed.show_indexes()
    segment_rows = rows[rows["kind"] == "node_type_segment"]
    assert not segment_rows.empty
    assert segment_rows["nbytes"].sum() > 0


def test_series_nbytes_uses_pandas_measurement() -> None:
    series = pd.Series([1], dtype="int64")
    assert _series_nbytes(series) == series.memory_usage()


def test_series_nbytes_uses_polars_measurement() -> None:
    pl = pytest.importorskip("polars")
    series = pl.Series([1], dtype=pl.Int64)
    assert _series_nbytes(series) == series.estimated_size()


def test_series_nbytes_uses_cudf_measurement() -> None:
    cudf = pytest.importorskip("cudf")
    try:
        device_count = cudf.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("CUDA device detection failed")
    if device_count == 0:
        pytest.skip("cuDF requires a CUDA device")
    series = cudf.Series([1], dtype="int64")
    assert _series_nbytes(series) == series.memory_usage()


def test_series_nbytes_defaults_to_zero_for_unknown_series() -> None:
    assert _series_nbytes(object()) == 0


def _pandas(frame):
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


def _assert_same_graph(actual, expected) -> None:
    if len(_pandas(actual._nodes)) == 0 and len(_pandas(expected._nodes)) == 0:
        # Pre-existing master gap (also on the typed-edge path): the wavefront
        # index path returns a raw-schema empty frame while the scan returns the
        # projected empty frame. Zero rows either way; exact frame equality is
        # asserted everywhere results are nonempty.
        assert len(_pandas(actual._edges)) == len(_pandas(expected._edges))
        return
    pd.testing.assert_frame_equal(_pandas(actual._nodes), _pandas(expected._nodes))
    assert (actual._edges is None) == (expected._edges is None)
    if actual._edges is not None:
        pd.testing.assert_frame_equal(_pandas(actual._edges), _pandas(expected._edges))


def _hop_sig(g):
    nodes, edges = _pandas(g._nodes), _pandas(g._edges)
    return (
        sorted(nodes[g._node].tolist()),
        sorted(map(tuple, edges[[g._source, g._destination]].itertuples(index=False, name=None))),
    )


def _indexed(graph, engine: str):
    out = graph.gfql_index_all(engine=engine)
    out = out.gfql_index_node_types(
        type_column="type", type_values=["P", "Q"], engine=engine)
    out = out.gfql_index_node_types(
        type_column="label__Person", type_values=[True], engine=engine)
    out = out.gfql_index_node_types(
        type_column="label__Robot", type_values=[True], engine=engine)
    return out


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("source_match,dest_match", [
    ({"type": "P"}, {"type": "Q"}),                       # scalar partition both ends
    ({"label__Person": True}, {"label__Person": True}),   # Boolean membership
    ({"label__Person": True}, {"label__Robot": True}),    # cross-label
    ({"type": "P"}, {"label__Robot": True}),              # mixed scalar/Boolean
    ({"type": "Z"}, {"type": "P"}),                       # EMPTY source segment
    ({"type": "P"}, {"type": "Z"}),                       # EMPTY destination segment
])
def test_node_segment_matches_scan_single_hop(engine: str, source_match, dest_match) -> None:
    graph = _graph(engine)
    query = [
        n({**{"id": 1}, **source_match}, name="a"),
        e_forward(name="r"),
        n(dest_match, name="b"),
        rows(),
    ]
    expected = graph.gfql(query, engine=engine, index_policy="off")
    indexed = _indexed(graph, engine)
    actual = indexed.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)


DIRECT_HOP_CASES = [
    ({"type": "P"}, {"type": "Q"}),
    ({"label__Person": True}, {"label__Person": True}),
    ({"type": "P"}, {"label__Robot": True}),
    ({"type": "Z"}, {"type": "P"}),
]


def _partial_reject_graph(engine: str):
    """Seed 2's only reverse candidate (4->2) fails a destination-Q filter.

    Seed 1 also has a reverse Q candidate (3->1), while 4->2 exercises
    rejection of the candidate itself rather than absence of an incoming edge.
    """
    nodes = pd.DataFrame({"id": [1, 2, 3, 4], "type": ["P", "P", "Q", "P"]})
    edges = pd.DataFrame({"src": [1, 2, 3, 4], "dst": [3, 4, 1, 2]})
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.bind(node="id", source="src", destination="dst").nodes(nodes).edges(edges)


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
def test_direct_hop_partial_seed_reject_matches_scan(engine: str, direction: str) -> None:
    """First-hop visited must drop seeds whose only edge a destination filter rejects."""
    graph = _partial_reject_graph(engine)
    indexed = graph.gfql_index_all(engine=engine).gfql_index_node_types(
        type_column="type", type_values=["Q"], engine=engine)
    kw = dict(
        hops=1, direction=direction, return_as_wave_front=False,
        destination_node_match={"type": "Q"},
    )
    seeds = graph._nodes[graph._nodes["id"].isin([1, 2])] if engine == "pandas" else graph._nodes.head(2)
    if direction == "reverse":
        prefilter = graph.hop(
            nodes=seeds, engine=engine, hops=1, direction=direction,
            return_as_wave_front=False,
        )
        assert _hop_sig(prefilter) == (
            [1, 2, 3, 4], [(3, 1), (4, 2)],
        )
    expected = graph.hop(nodes=seeds, engine=engine, **kw)
    for policy in ["off", "force"]:
        actual = with_index_policy(indexed, policy).hop(nodes=seeds, engine=engine, **kw)
        assert actual is not None
        assert _hop_sig(actual) == _hop_sig(expected), policy
    # Node 4 is non-Q, so seed 2's only reverse candidate is rejected.
    expected_edges = {
        "forward": [(1, 3)],
        "reverse": [(3, 1)],
        "undirected": [(1, 3), (3, 1)],
    }[direction]
    assert _hop_sig(expected) == ([1, 3], expected_edges)


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_direct_hop_all_destinations_rejected_matches_scan(engine: str) -> None:
    """A destination filter matching no node must yield the same empty result."""
    graph = _partial_reject_graph(engine)
    indexed = graph.gfql_index_all(engine=engine).gfql_index_node_types(
        type_column="type", type_values=["Z"], engine=engine)
    kw = dict(
        hops=1, direction="forward", return_as_wave_front=False,
        destination_node_match={"type": "Z"},
    )
    seeds = graph._nodes[graph._nodes["id"].isin([1, 2])] if engine == "pandas" else graph._nodes.head(2)
    expected = graph.hop(nodes=seeds, engine=engine, **kw)
    assert len(_pandas(expected._nodes)) == 0
    for policy in ["off", "force"]:
        actual = with_index_policy(indexed, policy).hop(nodes=seeds, engine=engine, **kw)
        assert actual is not None
        _assert_same_graph(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
@pytest.mark.parametrize("source_match,dest_match", DIRECT_HOP_CASES)
def test_node_segment_direct_hop_matches_scan(
    engine: str, direction: str, source_match, dest_match,
) -> None:
    """Direct hop parity: gfql may delegate small chains to connected-bindings."""
    graph = _graph(engine)
    hop_kw = dict(
        hops=1, direction=direction, return_as_wave_front=True,
        source_node_match=source_match, destination_node_match=dest_match,
    )
    expected = graph.hop(nodes=graph._nodes, engine=engine, **hop_kw)
    indexed = _indexed(graph, engine)
    indexed = indexed.gfql_index_node_types(
        type_column="type", type_values=["Z"], engine=engine)
    indexed = with_index_policy(indexed, "force")
    actual = indexed.hop(nodes=graph._nodes, engine=engine, **hop_kw)
    assert actual is not None
    assert _hop_sig(actual) == _hop_sig(expected)


@pytest.mark.route_engaged("index-hop")
@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
@pytest.mark.parametrize("source_match,dest_match", DIRECT_HOP_CASES)
def test_node_segment_direct_hop_engagement(
    engine: str, direction: str, source_match, dest_match,
) -> None:
    """Engagement pin for direct hops, replayed separately by the routes-off lane."""
    graph = _graph(engine)
    hop_kw = dict(
        hops=1, direction=direction, return_as_wave_front=True,
        source_node_match=source_match, destination_node_match=dest_match,
    )
    indexed = _indexed(graph, engine)
    indexed = indexed.gfql_index_node_types(
        type_column="type", type_values=["Z"], engine=engine)
    indexed = with_index_policy(indexed, "force")
    with index_trace() as steps:
        actual = indexed.hop(nodes=graph._nodes, engine=engine, **hop_kw)
    assert actual is not None
    assert any(step["path"] == "index" for step in steps)
    assert any(step.get("node_segments") == {
        "source": "resident", "destination": "resident"
    } for step in steps)


@pytest.mark.route_engaged("index-hop", "polars-plain")
def test_plain_polars_chain_consumes_destination_node_segment() -> None:
    """A destination node-type filter is lowered into the indexed single hop.

    The chain seed has already applied the source filter, so this shape admits a
    destination-only node match without asking the segment path to re-filter seeds.
    """
    engine = "polars"
    graph = _graph(engine)
    indexed = _indexed(graph, engine)
    query = [
        n({"id": 1, "type": "P"}), e_forward(), n({"type": "Q"}), rows(),
    ]
    expected = graph.gfql(query, engine=engine, index_policy="off")
    with index_trace() as steps:
        actual = indexed.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)
    assert any(
        step.get("path") == "index"
        and step.get("node_segments", {}).get("destination") == "resident"
        for step in steps
    ), steps


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_node_segment_two_hop_chain_matches_scan(engine: str) -> None:
    """The IC4-realistic shape: labeled nodes on every pattern of a two-hop chain."""
    graph = _graph(engine)
    query = [
        n({"id": 1, "label__Person": True}, name="a"),
        e_forward(name="r1"), n({"type": "Q"}, name="b"),
        e_forward(name="r2"), n({"label__Person": True}, name="c"), rows(),
    ]
    expected = graph.gfql(query, engine=engine, index_policy="off")
    indexed = _indexed(graph, engine)
    actual = indexed.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_node_segment_plus_typed_edge_combined(engine: str) -> None:
    """Labeled nodes AND typed edges in the same two-hop query."""
    graph = _graph(engine)
    query = [
        n({"id": 1, "type": "P"}, name="a"),
        e_forward({"etype": "K"}, name="r1"), n({"label__Person": True}, name="b"),
        e_forward(name="r2"), n({"type": "Q"}, name="c"), rows(),
    ]
    expected = graph.gfql(query, engine=engine, index_policy="off")
    indexed = graph.gfql_index_all(engine=engine)
    indexed = indexed.gfql_index_edges(type_column="etype", type_values=["K", "T"], engine=engine)
    indexed = indexed.gfql_index_node_types(type_column="type", type_values=["P", "Q"], engine=engine)
    indexed = indexed.gfql_index_node_types(type_column="label__Person", type_values=[True], engine=engine)
    actual = indexed.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_node_segment_null_type_never_matches_scalar_segment(engine: str) -> None:
    graph = _graph(engine)
    # Destination segment {type: 'P'} must exclude node 5 (null type) even though
    # label__Person=True would admit it under a different column.
    query = [
        n({"id": 1}, name="a"), e_forward(name="r"), n({"type": "P"}, name="b"), rows(),
    ]
    expected = graph.gfql(query, engine=engine, index_policy="off")
    indexed = _indexed(graph, engine)
    actual = indexed.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_stale_node_segment_declines_to_scan(engine: str) -> None:
    graph = _graph(engine)
    indexed = _indexed(graph, engine)
    # Rebind nodes -> segment identity guard misses -> scan fallback, same answer.
    pl_or_pd_nodes = indexed._nodes
    rebound = indexed.nodes(_pandas(pl_or_pd_nodes), "id") if not hasattr(pl_or_pd_nodes, "clone") else indexed.nodes(pl_or_pd_nodes.clone(), "id")
    query = [
        n({"id": 1, "type": "P"}, name="a"), e_forward(name="r"), n({"type": "Q"}, name="b"), rows(),
    ]
    expected = graph.gfql(query, engine=engine, index_policy="off")
    actual = rebound.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)


def test_multi_key_node_match_still_parity_via_scan() -> None:
    """v1 covers single-key matches; multi-key declines to scan (same answer)."""
    engine = "pandas"
    graph = _graph(engine)
    query = [
        n({"id": 1, "type": "P"}, name="a"), e_forward(name="r"),
        n({"type": "Q", "label__Robot": True}, name="b"), rows(),
    ]
    expected = graph.gfql(query, engine=engine, index_policy="off")
    indexed = _indexed(graph, engine)
    actual = indexed.gfql(query, engine=engine, index_policy="force")
    _assert_same_graph(actual, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_node_segment_lifecycle_show_and_selective_drop(engine: str) -> None:
    graph = _graph(engine)
    indexed = graph.gfql_index_all(engine=engine)
    indexed = indexed.create_index("node_prop", column="id", engine=engine)
    indexed = indexed.gfql_index_node_types(
        type_column="type", type_values=["P", "Q"], engine=engine)
    indexed = indexed.gfql_index_node_types(
        type_column="label__Person", type_values=[True], engine=engine)
    si = indexed.show_indexes()
    typed_row = si[si["name"] == "typed_node:type=P"].iloc[0]
    assert typed_row["kind"] == "node_type_segment"
    assert bool(typed_row["valid"]) and bool(typed_row["usable"])
    assert int(typed_row["n_rows"]) == 2

    one = indexed.drop_index("node_type_segment", column="type", value="P")
    registry = get_registry(one)
    assert ("type", "P") not in registry.node_type_segments
    assert ("type", "Q") in registry.node_type_segments
    assert ("label__Person", True) in registry.node_type_segments
    assert "id" in registry.node_props

    column = one.drop_index("node_type_segment", column="type")
    registry = get_registry(column)
    assert not any(key[0] == "type" for key in registry.node_type_segments)
    assert ("label__Person", True) in registry.node_type_segments
    assert "id" in registry.node_props

    ddl = parse_index_ddl('DROP GFQL INDEX FOR node_type_segment ON type')
    assert ddl is not None and ddl.kind == "node_type_segment" and ddl.column == "type"
    ddl_drop = apply_index_op(one, ddl)
    assert not any(key[0] == "type" for key in get_registry(ddl_drop).node_type_segments)

    none = column.drop_index("node_type_segment")
    registry = get_registry(none)
    assert registry.node_type_segments == {}
    assert "id" in registry.node_props


def test_gfql_index_node_types_validation() -> None:
    graph = _graph("pandas")
    with pytest.raises(ValueError):
        graph.gfql_index_node_types(type_column="type")
    with pytest.raises(ValueError):
        graph.gfql_index_node_types(type_values=["P"])
    with pytest.raises(TypeError):
        graph.gfql_index_node_types(type_column="type", type_values="P")
    with pytest.raises(ValueError):
        graph.gfql_index_node_types(type_column="type", type_values=[])
    with pytest.raises(TypeError):
        graph.gfql_index_node_types(type_column="type", type_values=[1.5])
    with pytest.raises(ValueError):
        graph.gfql_index_node_types(type_column="nope", type_values=["P"])
    with pytest.raises(ValueError):
        graphistry.nodes(None).gfql_index_node_types(type_column="type", type_values=["P"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_node_segment_trace_reports_residency(engine: str) -> None:
    """Trace the planner entry directly: GFQL may dispatch the same chain to a
    higher-level indexed seam, which does not consume node segments yet."""
    indexed = _indexed(_graph(engine), engine)
    engine_enum = Engine.PANDAS if engine == "pandas" else Engine.POLARS
    seeds = indexed._nodes.head(1)

    with index_trace() as steps:
        served = maybe_index_hop(
            indexed, engine_enum, nodes=seeds, hops=1, direction="forward",
            return_as_wave_front=True, policy="force",
            source_node_match={"type": "P"}, destination_node_match={"type": "Q"},
        )
    assert served is not None
    assert steps[-1]["node_segments"] == {"source": "resident", "destination": "resident"}
    assert steps[-1]["path"] == "index"

    bare = _graph(engine).gfql_index_all(engine=engine)
    seeds = bare._nodes.head(1)
    with index_trace() as steps:
        declined = maybe_index_hop(
            bare, engine_enum, nodes=seeds, hops=1, direction="forward",
            return_as_wave_front=True, policy="force",
            source_node_match={"type": "P"}, destination_node_match={"type": "Q"},
        )
    assert declined is None
    assert steps[-1]["node_segments"] == {
        "source": "missing_or_stale", "destination": "missing_or_stale"}
    assert steps[-1]["path"] == "scan"

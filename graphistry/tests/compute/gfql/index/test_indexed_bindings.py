"""Generic indexed fixed-hop GFQL differential contract.

The shapes are reduced from existing LDBC-derived IS1/IS3/IS5/IS7 tests. Product
selection must remain operator/index/cost based; benchmark names live only in test
ids and comments. Every expected result comes from the same-engine canonical path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTEdge, ASTObject, e_forward, e_undirected, n


ENGINES = ["pandas", "polars", "cudf"]
ENGINE_ENUM = {
    "pandas": Engine.PANDAS,
    "polars": Engine.POLARS,
    "cudf": Engine.CUDF,
}
TRACE_KEYS = {
    "operation",
    "seam",
    "engine",
    "served",
    "reason",
    "hop_count",
    "public_seed_scan",
    "hop_details",
    "path",
    "decision_reason",
}


def _base_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(
        {
            "id": np.arange(1, 13, dtype=np.int64),
            "public": np.arange(100, 112, dtype=np.int64),
            "kind": [
                "seed", "mid", "mid", "end", "end", "tail",
                "tail", "reverse", "seed", "noise", "noise", "noise",
            ],
            "rank": np.arange(12, dtype=np.int64),
            "maybe": [
                0.0, 1.0, 2.0, None, 4.0, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
        }
    )
    edges = pd.DataFrame(
        {
            "src": [1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 4, 5, 1, 1, 1, 2, 2],
            "dst": [2, 2, 3, 4, 5, 5, 6, 6, 7, 1, 2, 3, 1, 2, 2, 1, 3],
            "type": [
                "A", "A", "A", "B", "B", "B", "C", "C", "D",
                "REV", "B", "B", "U", "U", "U", "U", "U",
            ],
            "weight": np.arange(10, 27, dtype=np.int64),
        }
    )
    return nodes, edges


def _native_frames(
    engine: str, nodes: pd.DataFrame, edges: pd.DataFrame
) -> Tuple[Any, Any]:
    if engine in ("polars", "polars-gpu"):
        pl = pytest.importorskip("polars")
        return pl.from_pandas(nodes), pl.from_pandas(edges)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return nodes, edges


def _graph_from_frames(
    engine: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    node_col: str = "id",
    source_col: str = "src",
    destination_col: str = "dst",
    indexed: bool = True,
) -> Any:
    nodes_native, edges_native = _native_frames(engine, nodes, edges)
    g = graphistry.nodes(nodes_native, node_col).edges(
        edges_native, source_col, destination_col
    )
    return g.gfql_index_all(engine=engine) if indexed else g


def _graph(engine: str, *, indexed: bool = True) -> Any:
    return _graph_from_frames(engine, *_base_frames(), indexed=indexed)


def _native_copy(frame: Any) -> Any:
    return frame.clone() if "polars" in type(frame).__module__ else frame.copy()


def _to_pandas(frame: Any) -> pd.DataFrame:
    if "polars" in type(frame).__module__ or "cudf" in type(frame).__module__:
        return frame.to_pandas()
    return frame


def _assert_result_exact(actual: Any, expected: Any, engine: str) -> None:
    assert actual._nodes is not None and expected._nodes is not None
    if engine in ("polars", "polars-gpu"):
        assert "polars" in type(actual._nodes).__module__
        assert actual._nodes.schema == expected._nodes.schema
    elif engine == "cudf":
        assert "cudf" in type(actual._nodes).__module__
    else:
        assert isinstance(actual._nodes, pd.DataFrame)
    pd.testing.assert_frame_equal(
        _to_pandas(actual._nodes),
        _to_pandas(expected._nodes),
        check_dtype=True,
        check_index_type=True,
    )
    assert (actual._edges is None) == (expected._edges is None)
    if actual._edges is not None and expected._edges is not None:
        pd.testing.assert_frame_equal(
            _to_pandas(actual._edges),
            _to_pandas(expected._edges),
            check_dtype=True,
            check_index_type=True,
        )


def _run(
    g: Any,
    query: str,
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generic: bool,
    index_policy: str = "force",
    policy: Any = None,
) -> Any:
    if generic:
        import graphistry.compute.gfql.index.bindings as indexed_bindings
        import graphistry.compute.gfql_unified as gfql_unified
        import graphistry.compute.chain as chain_mod

        monkeypatch.setattr(
            indexed_bindings,
            "try_indexed_connected_bindings_state",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            gfql_unified,
            "_execute_seeded_typed_hop_fast_path",
            lambda *args, **kwargs: None,
        )
        # `generic` enumerates the acceleration seams to disable. The chain fast path now
        # serves NAMED patterns (it previously rejected any alias, which is why it was
        # absent here), so it has to be listed — otherwise the comparison stops being
        # generic-vs-accelerated and becomes accelerated-vs-a-DIFFERENT-accelerated.
        #
        # But disable ONLY the NEW capability, not the seam. This path has always served
        # UNNAMED middles, and `test_unnamed_middle_rows_call_matches_canonical` compares
        # exactly that shape — blanket-disabling turns that case from fast-vs-fast into
        # general-vs-fast and it fails for a reason that has nothing to do with what it
        # tests. Verified: master's product code plus a blanket patch reproduces that
        # failure on its own.
        _real_fast_path = chain_mod._try_chain_fast_path

        def _fast_path_without_named(
            g_in: Plottable,
            ops: List[ASTObject],
            engine_concrete: Engine,
            start_nodes: Optional[pd.DataFrame] = None,
        ) -> Optional[Plottable]:
            if any(op._name is not None for op in ops):
                return None
            return _real_fast_path(g_in, ops, engine_concrete, start_nodes)

        monkeypatch.setattr(chain_mod, "_try_chain_fast_path", _fast_path_without_named)
    return g.gfql(
        query,
        engine=engine,
        index_policy=index_policy,
        policy=policy,
    )


def _trace_run(
    g: Any,
    query: str,
    engine: str,
    *,
    index_policy: str = "force",
    policy: Any = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    from graphistry.compute.gfql.index import index_trace

    with index_trace() as captured:
        out = g.gfql(
            query,
            engine=engine,
            index_policy=index_policy,
            policy=policy,
        )
    return out, [
        dict(step)
        for step in captured
        if step.get("operation") == "indexed_traversal"
    ]


def _assert_decision(step: Dict[str, Any], *, seam: str, served: bool) -> None:
    assert TRACE_KEYS <= set(step)
    assert step["seam"] == seam
    assert step["served"] is served
    assert step["path"] == ("index" if served else "scan")
    assert step["decision_reason"] == step["reason"]
    assert isinstance(step["hop_details"], list)
    if not served:
        assert step["hop_details"] == []


def _assert_parity(
    g: Any,
    query: str,
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seam: str,
    index_policy: str = "force",
    expect_served: bool = True,
) -> Tuple[Any, List[Dict[str, Any]]]:
    actual, steps = _trace_run(
        g, query, engine, index_policy=index_policy
    )
    with monkeypatch.context() as m:
        expected = _run(
            g,
            query,
            engine,
            m,
            generic=True,
            index_policy=index_policy,
        )
    _assert_result_exact(actual, expected, engine)
    decisions = [step for step in steps if step.get("seam") == seam]
    assert decisions
    assert any(step.get("served") is True for step in decisions) is expect_served
    for step in decisions:
        _assert_decision(step, seam=seam, served=bool(step["served"]))
    return actual, decisions


DESTINATION_QUERY = (
    "MATCH (source {public:100})-[:A]->(destination) "
    "RETURN destination.public AS destination ORDER BY destination"
)
CONNECTED_QUERY = (
    "MATCH (a {public:100})-[:A]->(b)-[:B]->(c) "
    "RETURN a.public AS a, b.public AS b, c.public AS c ORDER BY b, c"
)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "case,reason,served",
    [
        pytest.param("resident", "served", True),
        pytest.param("missing", "index_missing", False),
        pytest.param("stale", "index_stale", False),
        pytest.param("off", "index_policy_off", False),
    ],
)
def test_destination_unique_trace_and_lifecycle(
    engine: str,
    case: str,
    reason: str,
    served: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _graph(engine, indexed=case != "missing")
    if case == "stale":
        assert g._edges is not None
        g = g.edges(_native_copy(g._edges), "src", "dst")
    elif case == "off":
        setattr(g, "_gfql_index_policy", "off")
    policy = "off" if case == "off" else "force"
    actual, steps = _trace_run(
        g, DESTINATION_QUERY, engine, index_policy=policy
    )
    with monkeypatch.context() as m:
        expected = _run(
            g,
            DESTINATION_QUERY,
            engine,
            m,
            generic=True,
            index_policy=policy,
        )
    _assert_result_exact(actual, expected, engine)
    decisions = [
        step for step in steps if step.get("seam") == "destination_return"
    ]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="destination_return", served=served)
    assert decisions[0]["reason"] == reason
    assert decisions[0]["hop_count"] == 1
    assert decisions[0]["public_seed_scan"] is True


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "case,reason,served",
    [
        pytest.param("resident", "served", True),
        pytest.param("missing", "index_missing", False),
        pytest.param("stale", "index_stale", False),
        pytest.param("off", "index_policy_off", False),
    ],
)
def test_connected_path_bag_trace_and_lifecycle(
    engine: str,
    case: str,
    reason: str,
    served: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _graph(engine, indexed=case != "missing")
    if case == "stale":
        assert g._nodes is not None
        g = g.nodes(_native_copy(g._nodes), "id")
    elif case == "off":
        setattr(g, "_gfql_index_policy", "off")
    policy = "off" if case == "off" else "force"
    actual, steps = _trace_run(g, CONNECTED_QUERY, engine, index_policy=policy)
    with monkeypatch.context() as m:
        expected = _run(
            g,
            CONNECTED_QUERY,
            engine,
            m,
            generic=True,
            index_policy=policy,
        )
    _assert_result_exact(actual, expected, engine)
    decisions = [
        step for step in steps if step.get("seam") == "connected_bindings"
    ]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="connected_bindings", served=served)
    assert decisions[0]["reason"] == reason
    assert decisions[0]["hop_count"] == 2
    assert decisions[0]["public_seed_scan"] is True


STANDARD_DERIVED_POSITIVES = [
    pytest.param(
        "MATCH (a {public:100})-[r:A]->(b {kind:'mid'}) "
        "RETURN a.public AS a, b.public AS b, r.weight AS w ORDER BY w",
        id="is1-directed-projection",
    ),
    pytest.param(
        "MATCH (a {public:100})-[r:U]-(b) "
        "RETURN a.public AS a, b.public AS b, r.weight AS w ORDER BY w, b",
        id="is3-undirected-multiplicity",
    ),
    pytest.param(CONNECTED_QUERY, id="is7-connected-two-hop-bag"),
    pytest.param(
        "MATCH (a {public:100})-[:A]->(b) "
        "OPTIONAL MATCH (b)-[:B]->(c) "
        "RETURN a.public AS a, b.public AS b, c.public AS c ORDER BY b, c",
        id="is7-optional-continuation",
    ),
    pytest.param(
        "MATCH (a {public:100})-[:A]->(b)<-[:B]-(c) "
        "RETURN a.public AS a, b.public AS b, c.public AS c ORDER BY b, c",
        id="fixed-hop-reverse-composition",
    ),
    pytest.param(
        "MATCH (a {public:100})-[:A]->(b)-[:B]->(c) "
        "RETURN DISTINCT b.public AS b, c.public AS c "
        "ORDER BY b DESC, c DESC LIMIT 1",
        id="canonical-distinct-order-limit-suffix",
    ),
    pytest.param(
        "MATCH (a {public:999999})-[:A]->(b)-[:B]->(c) "
        "RETURN a.public AS a, b.public AS b, c.public AS c",
        id="official-no-match-stratum",
    ),
    pytest.param(
        "MATCH (a {public:100})-[:A]->(b)-[:B]->(c) "
        "RETURN b.public AS b, c.public AS c, c.maybe AS maybe ORDER BY b, c",
        id="null-and-dtype-parity",
    ),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", STANDARD_DERIVED_POSITIVES)
def test_standard_derived_connected_parity(
    engine: str,
    query: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_parity(
        _graph(engine),
        query,
        engine,
        monkeypatch,
        seam="connected_bindings",
    )


def test_pandas_connected_boundary_bypasses_canonical_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _graph("pandas")
    with monkeypatch.context() as m:
        expected = _run(
            g,
            CONNECTED_QUERY,
            "pandas",
            m,
            generic=True,
        )

    def unexpected_traversal(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("indexed boundary entered canonical AST traversal")

    monkeypatch.setattr(ASTEdge, "execute", unexpected_traversal)
    actual, steps = _trace_run(g, CONNECTED_QUERY, "pandas")
    _assert_result_exact(actual, expected, "pandas")
    decisions = [
        step for step in steps if step.get("seam") == "connected_bindings"
    ]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="connected_bindings", served=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_destination_property_projection_dtype_parity(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lean property RETURN must match its OWN engine's canonical dtypes.

    The pandas rows-pivot upcasts int->float64 and bool->object; cuDF's keeps the
    source dtypes. A single pandas-shaped cast rule silently diverges on cuDF.
    """
    nodes, edges = _base_frames()
    nodes = nodes.assign(flag=[True, False] * 6)
    g = _graph_from_frames(engine, nodes, edges)
    query = (
        "MATCH (source {public:100})-[:A]->(destination) "
        "RETURN destination.rank AS r, destination.maybe AS m, "
        "destination.flag AS f, destination.kind AS k, destination.id AS i "
        "ORDER BY i"
    )
    _assert_parity(g, query, engine, monkeypatch, seam="destination_return")


def test_polars_connected_boundary_bypasses_canonical_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("polars")
    from graphistry.compute.gfql.lazy.engine.polars import chain as polars_chain

    g = _graph("polars")
    with monkeypatch.context() as m:
        expected = _run(
            g,
            CONNECTED_QUERY,
            "polars",
            m,
            generic=True,
        )

    def unexpected_traversal(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("indexed boundary entered canonical polars traversal")

    monkeypatch.setattr(
        polars_chain, "_chain_traversal_polars", unexpected_traversal
    )
    actual, steps = _trace_run(g, CONNECTED_QUERY, "polars")
    _assert_result_exact(actual, expected, "polars")
    decisions = [
        step for step in steps if step.get("seam") == "connected_bindings"
    ]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="connected_bindings", served=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_unnamed_middle_rows_call_matches_canonical(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rows() call with no binding ops reads the TRAVERSAL result, not the
    whole graph: an indexed bypass must not engage for an unnamed middle."""
    from graphistry.compute.ast import rows as rows_call

    if engine == "polars":
        pytest.importorskip("polars")
    g = _graph(engine)
    ops = [n({"id": 1}), e_forward(), n(), rows_call()]
    with monkeypatch.context() as m:
        expected = _run(g, ops, engine, m, generic=True)
    actual = g.gfql(ops, engine=engine, index_policy="force")
    _assert_result_exact(actual, expected, engine)


@pytest.mark.parametrize("seed_kind", ["seed", "noise"])
def test_pandas_internal_id_plus_constraints_gathers_seed_before_filter(
    seed_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import graphistry.compute.gfql.index.bindings as indexed_bindings

    g = _graph("pandas")
    query = (
        f"MATCH (a {{id:1, kind:'{seed_kind}'}})-[:A]->(b)-[:B]->(c) "
        "RETURN a.id AS a, b.id AS b, c.id AS c ORDER BY b, c"
    )
    with monkeypatch.context() as m:
        expected = _run(g, query, "pandas", m, generic=True)

    filtered_lengths: List[int] = []
    original_filter = indexed_bindings._filter_frame

    def record_filter(frame: Any, *args: Any, **kwargs: Any) -> Any:
        filtered_lengths.append(len(frame))
        return original_filter(frame, *args, **kwargs)

    monkeypatch.setattr(indexed_bindings, "_filter_frame", record_filter)
    actual, steps = _trace_run(g, query, "pandas")

    _assert_result_exact(actual, expected, "pandas")
    decisions = [
        step for step in steps if step.get("seam") == "connected_bindings"
    ]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="connected_bindings", served=True)
    assert filtered_lengths
    assert filtered_lengths[0] == 1


# --- secondary (node property) index -----------------------------------------
# CONNECTED_QUERY seeds on ``public``, which is NOT the graph's node-id binding
# (``id``), so without a property index the seed costs a full node scan.


def _prop_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes, edges = _base_frames()
    # ``grp`` repeats (4 rows per value): the duplicate-key case a node-id index
    # cannot express but a CSR property index can.
    nodes = nodes.assign(grp=(nodes["rank"] % 3).astype("int64"))
    return nodes, edges


def _prop_graph(engine: str, columns: Sequence[str] = ("public",)) -> Any:
    nodes, edges = _prop_frames()
    return _graph_from_frames(engine, nodes, edges).gfql_index_node_props(
        list(columns), engine=engine
    )


def _seed_filter_widths(
    g: Any, query: Any, engine: str, monkeypatch: pytest.MonkeyPatch
) -> Tuple[Any, List[Dict[str, Any]], List[int]]:
    """Run traced, recording how many rows each helper filter had to look at."""
    import graphistry.compute.gfql.index.bindings as indexed_bindings

    widths: List[int] = []
    original = indexed_bindings._filter_frame

    def record(frame: Any, *args: Any, **kwargs: Any) -> Any:
        widths.append(int(frame.shape[0]))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(indexed_bindings, "_filter_frame", record)
    actual, steps = _trace_run(g, query, engine)
    return actual, steps, widths


@pytest.mark.parametrize("engine", ENGINES)
def test_node_property_index_seeds_without_scanning(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _prop_graph(engine)
    with monkeypatch.context() as m:
        expected = _run(g, CONNECTED_QUERY, engine, m, generic=True)
    actual, steps, widths = _seed_filter_widths(
        g, CONNECTED_QUERY, engine, monkeypatch
    )
    _assert_result_exact(actual, expected, engine)
    decisions = [s for s in steps if s.get("seam") == "connected_bindings"]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="connected_bindings", served=True)
    assert widths and widths[0] == 1  # one indexed candidate, not the node table


@pytest.mark.parametrize("engine", ENGINES)
def test_node_property_index_absent_matches_indexed(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same query, same answer, with and without the secondary index."""
    nodes, edges = _prop_frames()
    plain = _graph_from_frames(engine, nodes, edges)
    indexed = plain.gfql_index_node_props(["public"], engine=engine)
    with monkeypatch.context() as m:
        expected = _run(plain, CONNECTED_QUERY, engine, m, generic=True)
    for g in (plain, indexed):
        actual, _ = _trace_run(g, CONNECTED_QUERY, engine)
        _assert_result_exact(actual, expected, engine)


def test_node_property_index_duplicate_values_match_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-unique property still gathers EVERY matching row (CSR, not first-hit)."""
    from graphistry.compute.ast import rows as rows_call

    g = _prop_graph("pandas", columns=("grp",))
    query = [n({"grp": 0}, name="a"), e_forward({"type": "A"}, name="r"), n(name="b"), rows_call()]
    with monkeypatch.context() as m:
        expected = _run(g, query, "pandas", m, generic=True)
    actual, steps, widths = _seed_filter_widths(g, query, "pandas", monkeypatch)
    _assert_result_exact(actual, expected, "pandas")
    assert widths and widths[0] == 4  # every row with grp == 0, none of the others
    assert [s for s in steps if s.get("seam") == "connected_bindings"]


@pytest.mark.parametrize("case", ["stale", "policy_off"])
def test_node_property_index_lifecycle_falls_back(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _prop_graph("pandas")
    index_policy = "off" if case == "policy_off" else "force"
    if case == "stale":
        g = g.nodes(_native_copy(g._nodes), "id")  # rebind -> index treated as absent
    with monkeypatch.context() as m:
        expected = _run(
            g, CONNECTED_QUERY, "pandas", m, generic=True, index_policy=index_policy
        )
    actual, _ = _trace_run(g, CONNECTED_QUERY, "pandas", index_policy=index_policy)
    _assert_result_exact(actual, expected, "pandas")


def test_node_property_index_declines_unindexable_columns() -> None:
    """Unindexable DTYPE is skippable; a caller mistake is not.

    The convenience builder suppresses exactly one condition — a column whose dtype
    this index cannot serve. A missing column, a missing argument, or any other
    failure must propagate, or a typo would silently leave the query unindexed.
    """
    from graphistry.compute.gfql.index import (
        NODE_PROP, GfqlIndexUnsupportedError, create_index, get_registry,
    )

    g = _prop_graph("pandas", columns=())
    for column in ("kind", "maybe"):  # object dtype, float-with-null
        with pytest.raises(GfqlIndexUnsupportedError):
            create_index(g, NODE_PROP, column=column)
    # caller mistakes are NOT the skippable kind
    for bad in ({"column": "nosuch"}, {}):
        with pytest.raises(ValueError) as excinfo:
            create_index(g, NODE_PROP, **bad)
        assert not isinstance(excinfo.value, GfqlIndexUnsupportedError)

    # the convenience wrapper skips the unindexable dtypes and indexes what it can
    g2 = g.gfql_index_node_props(["kind", "maybe", "public"])
    assert get_registry(g2).node_prop_cols() == ("public",)
    # ...but does NOT swallow a real failure
    with pytest.raises(ValueError):
        g.gfql_index_node_props(["nosuch"])


def test_node_property_index_prefers_the_most_selective_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from graphistry.compute.ast import rows as rows_call

    g = _prop_graph("pandas", columns=("public", "grp"))
    query = [
        n({"public": 100, "grp": 0}, name="a"),
        e_forward({"type": "A"}, name="r"),
        n(name="b"),
        rows_call(),
    ]
    with monkeypatch.context() as m:
        expected = _run(g, query, "pandas", m, generic=True)
    actual, _, widths = _seed_filter_widths(g, query, "pandas", monkeypatch)
    _assert_result_exact(actual, expected, "pandas")
    assert widths and widths[0] == 1  # 'public' (1 match) beats 'grp' (4 matches)


@pytest.mark.parametrize(
    "seed,indexed_column,expect_gathered",
    [
        pytest.param({"public": 100}, "public", True, id="selective-uses-index"),
        pytest.param({"grp": 0}, "grp", False, id="unselective-keeps-scan"),
    ],
)
def test_node_property_index_cost_gate_under_policy_use(
    seed: Dict[str, Any],
    indexed_column: str,
    expect_gathered: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the default `use` policy the gate must reject a non-selective predicate.

    `force` skips the gate, so the other property-index cases cannot exercise it.
    The crossover is pinned here via the public knob rather than relying on the
    engine default, so the test states the threshold it is testing: at 25%,
    `public` (1 of 12 rows) gathers through the index and `grp` (4 of 12) does not.
    Either way the answer is the canonical one.
    """
    from graphistry.compute.ast import rows as rows_call
    from graphistry.compute.gfql.index import reset_cost_gate_frac, set_cost_gate_frac

    set_cost_gate_frac(Engine.PANDAS, 0.25)
    try:
        g = _prop_graph("pandas", columns=(indexed_column,))
        query = [
            n(seed, name="a"), e_forward({"type": "A"}, name="r"), n(name="b"), rows_call(),
        ]
        with monkeypatch.context() as m:
            expected = _run(g, query, "pandas", m, generic=True, index_policy="use")

        import graphistry.compute.gfql.index.bindings as indexed_bindings

        widths: List[int] = []
        original = indexed_bindings._filter_frame

        def record(frame: Any, *args: Any, **kwargs: Any) -> Any:
            widths.append(int(frame.shape[0]))
            return original(frame, *args, **kwargs)

        monkeypatch.setattr(indexed_bindings, "_filter_frame", record)
        actual, _ = _trace_run(g, query, "pandas", index_policy="use")
    finally:
        reset_cost_gate_frac(Engine.PANDAS)

    _assert_result_exact(actual, expected, "pandas")
    assert widths
    n_nodes = int(g._nodes.shape[0])
    if expect_gathered:
        assert widths[0] < n_nodes  # indexed candidates only
    else:
        assert widths[0] == n_nodes  # gate declined -> canonical scan


def test_node_property_index_shows_and_drops() -> None:
    from graphistry.compute.gfql.index import NODE_PROP, get_registry

    g = _prop_graph("pandas", columns=("public", "grp"))
    shown = g.show_indexes()
    props = shown[shown["kind"] == NODE_PROP]
    assert sorted(props["key_col"]) == ["grp", "public"]
    assert bool(props["valid"].all())
    assert sorted(props["n_keys"]) == [3, 12]
    assert get_registry(g.drop_index(NODE_PROP, column="grp")).node_prop_cols() == ("public",)
    assert get_registry(g.drop_index(NODE_PROP)).node_prop_cols() == ()
    assert get_registry(g.drop_index()).is_empty()


@pytest.mark.parametrize(
    "suffix_params,seeded,reason",
    [
        pytest.param({"source": "a"}, False, "rows-with-source", id="rows-source"),
        pytest.param({"alias_endpoints": {"a": "src"}}, False, "alias-endpoints", id="alias-endpoints"),
        pytest.param({"alias_prefilters": {"b": {"rank": 1}}}, False, "prefiltered", id="alias-prefilters"),
        pytest.param({}, True, "seeded re-entry", id="carried-seed"),
    ],
)
def test_polars_early_gate_refuses_unsupported_boundaries(
    suffix_params: Dict[str, Any],
    seeded: bool,
    reason: str,
) -> None:
    """The polars bypass must not engage on shapes its rows call does not consume.

    Engaging on any of these would hand the materializer a compact state for a plan
    it is not executing, so each refusal condition is asserted on the gate directly.
    """
    pytest.importorskip("polars")
    from graphistry.compute.ast import rows as rows_call
    from graphistry.compute.gfql.lazy.engine.polars.chain import _try_indexed_middle_polars

    g = _graph("polars")
    middle = [n({"public": 100}, name="a"), e_forward({"type": "A"}, name="r"), n(name="b")]
    start_nodes = g._nodes.head(1) if seeded else None

    state, attempted = _try_indexed_middle_polars(
        g, middle, [rows_call(**suffix_params)], start_nodes
    )
    assert state is None, f"polars gate engaged on {reason}"
    assert attempted is False, f"polars gate recorded an attempt on {reason}"


def test_polars_early_gate_requires_the_whole_middle() -> None:
    """binding_ops that do not cover the middle must not take the bypass."""
    pytest.importorskip("polars")
    from graphistry.compute.ast import rows as rows_call
    from graphistry.compute.chain import serialize_binding_ops
    from graphistry.compute.gfql.lazy.engine.polars.chain import _try_indexed_middle_polars

    g = _graph("polars")
    middle = [n({"public": 100}, name="a"), e_forward({"type": "A"}, name="r"), n(name="b")]

    other = serialize_binding_ops([n({"public": 101}, name="a"), e_forward(), n(name="b")])
    state, attempted = _try_indexed_middle_polars(
        g, middle, [rows_call(binding_ops=other)], None
    )
    assert state is None and attempted is False  # a different plan

    unnamed = [n({"public": 100}), e_forward({"type": "A"}), n()]
    state, attempted = _try_indexed_middle_polars(g, unnamed, [rows_call()], None)
    assert state is None and attempted is False  # rewrite would not install them

    # the supported shape DOES engage (guards against a vacuous negative suite);
    # `force` skips the cost gate, which a 12-row fixture would otherwise trip
    from graphistry.compute.gfql.index import with_index_policy

    state, attempted = _try_indexed_middle_polars(
        with_index_policy(g, "force"), middle, [rows_call()], None
    )
    assert attempted is True and state is not None


@pytest.mark.parametrize("engine", ENGINES)
def test_indexed_execution_is_pure(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _graph(engine)
    assert g._nodes is not None and g._edges is not None
    nodes_before = _to_pandas(g._nodes).copy(deep=True)
    edges_before = _to_pandas(g._edges).copy(deep=True)
    _assert_parity(
        g,
        CONNECTED_QUERY,
        engine,
        monkeypatch,
        seam="connected_bindings",
    )
    pd.testing.assert_frame_equal(_to_pandas(g._nodes), nodes_before)
    pd.testing.assert_frame_equal(_to_pandas(g._edges), edges_before)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "ops,kwargs",
    [
        pytest.param(
            [
                n({"public": 100}, name="a"),
                e_forward(hops=None, min_hops=2, max_hops=2),
                n(name="b"),
            ],
            {},
            id="exact-bounded-varpath",
        ),
        pytest.param(
            [
                n({"public": 100}, name="a"),
                e_forward(include_zero_hop_seed=True),
                n(name="b"),
            ],
            {},
            id="zero-hop",
        ),
        pytest.param(
            [n({"public": 100}, name="a"), e_undirected(hops=2), n(name="b")],
            {},
            id="multi-hop-undirected",
        ),
        pytest.param(
            [n({"public": 100}, name="a"), e_forward(), n(name="a")],
            {},
            id="alias-reentry-cycle",
        ),
        pytest.param(
            [
                n({"public": 100}, name="a", query="rank >= 0"),
                e_forward(),
                n(name="b"),
            ],
            {},
            id="node-query",
        ),
        pytest.param(
            [n({"public": 100}, name="a"), e_forward(edge_query="weight > 0"), n(name="b")],
            {},
            id="edge-query",
        ),
        pytest.param(
            [n({"public": 100}, name="a"), e_forward(), n(name="b")],
            {"start_nodes": pd.DataFrame({"id": [1]})},
            id="carried-seed",
        ),
        pytest.param(
            [n({"public": 100}, name="a"), e_forward(), n(name="b")],
            {"alias_prefilters": {"b": {"rank": 1}}},
            id="alias-prefilter",
        ),
    ],
)
def test_unsupported_shapes_decline_before_work(
    engine: str,
    ops: Sequence[Any],
    kwargs: Dict[str, Any],
) -> None:
    import graphistry.compute.gfql.index.bindings as indexed_bindings
    from graphistry.compute.gfql.index import index_trace

    call_kwargs = dict(kwargs)
    if engine != "pandas" and "start_nodes" in call_kwargs:
        native_seed, _ = _native_frames(
            engine, call_kwargs["start_nodes"], _base_frames()[1]
        )
        call_kwargs["start_nodes"] = native_seed
    with index_trace() as captured:
        out = indexed_bindings.try_indexed_connected_bindings_state(
            _graph(engine),
            ops,
            engine=ENGINE_ENUM[engine],
            **call_kwargs,
        )
    assert out is None
    decisions = [
        dict(step)
        for step in captured
        if step.get("operation") == "indexed_traversal"
    ]
    assert len(decisions) == 1
    _assert_decision(
        decisions[0], seam="connected_bindings", served=False
    )
    assert decisions[0]["reason"] == "unsupported_shape"


@pytest.mark.parametrize(
    "node_ids,edge_src,edge_dst,seed",
    [
        pytest.param(["a", "b"], ["a"], ["b"], "a", id="string-ids"),
        pytest.param([1.0, np.nan, 2.0], [1.0], [2.0], 1.0, id="null-float-ids"),
    ],
)
def test_unsafe_id_domains_decline_structurally(
    node_ids: List[Any],
    edge_src: List[Any],
    edge_dst: List[Any],
    seed: Any,
) -> None:
    import graphistry.compute.gfql.index.bindings as indexed_bindings
    from graphistry.compute.gfql.index import index_trace

    g = graphistry.nodes(pd.DataFrame({"id": node_ids}), "id").edges(
        pd.DataFrame({"src": edge_src, "dst": edge_dst}), "src", "dst"
    ).gfql_index_all(engine="pandas")
    ops = [n({"id": seed}, name="a"), e_forward(), n(name="b")]
    with index_trace() as captured:
        out = indexed_bindings.try_indexed_connected_bindings_state(
            g, ops, engine=Engine.PANDAS
        )
    assert out is None
    decisions = [
        dict(step)
        for step in captured
        if step.get("operation") == "indexed_traversal"
    ]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="connected_bindings", served=False)
    assert decisions[0]["reason"] == "unsupported_dtype"


def test_shortest_path_never_enters_fixed_hop_helper() -> None:
    from graphistry.compute.gfql.index import index_trace

    query = (
        "MATCH (a {id:1}), (b {id:5}), "
        "path = shortestPath((a)-[*]-(b)) RETURN length(path) AS hops"
    )
    with index_trace() as captured:
        _graph("pandas").gfql(query, engine="pandas", index_policy="force")
    assert not any(
        step.get("operation") == "indexed_traversal" for step in captured
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_policy_declines_without_skipping_hooks(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fired: List[str] = []

    def hook(ctx: Dict[str, Any]) -> None:
        fired.append(str(ctx.get("phase", "?")))

    policy = {"prechain": hook, "postchain": hook}
    g = _graph(engine)
    actual, steps = _trace_run(
        g, CONNECTED_QUERY, engine, policy=policy
    )
    actual_fired = list(fired)
    fired.clear()
    with monkeypatch.context() as m:
        expected = _run(
            g,
            CONNECTED_QUERY,
            engine,
            m,
            generic=True,
            policy=policy,
        )
    _assert_result_exact(actual, expected, engine)
    assert actual_fired == fired
    decisions = [
        step for step in steps if step.get("seam") == "connected_bindings"
    ]
    assert len(decisions) == 1
    _assert_decision(decisions[0], seam="connected_bindings", served=False)
    assert decisions[0]["reason"] == "policy_active"


@pytest.mark.parametrize("engine", ENGINES)
def test_renamed_and_permuted_shape_remains_generic(
    engine: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes, edges = _base_frames()
    nodes = (
        nodes.rename(columns={"id": "vertex", "public": "external"})
        .iloc[::-1]
        .reset_index(drop=True)
    )
    edges = (
        edges.rename(columns={"src": "source", "dst": "target"})
        .assign(
            type=lambda frame: frame["type"].replace({"A": "X", "B": "Y"})
        )
        .iloc[np.roll(np.arange(len(edges)), 5)]
        .reset_index(drop=True)
    )
    g = _graph_from_frames(
        engine,
        nodes,
        edges,
        node_col="vertex",
        source_col="source",
        destination_col="target",
    )
    query = (
        "MATCH (origin {external:100})-[first:X]->(middle)-[:Y]->(target) "
        "RETURN origin.external AS origin, middle.external AS middle, "
        "target.external AS target, first.weight AS weight "
        "ORDER BY middle, target, weight"
    )
    _assert_parity(
        g,
        query,
        engine,
        monkeypatch,
        seam="connected_bindings",
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_unexpected_and_memory_errors_propagate(
    engine: str,
    error_type: Type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import graphistry.compute.gfql.index.bindings as indexed_bindings

    def fail_filter(*args: Any, **kwargs: Any) -> Any:
        raise error_type("sentinel")

    monkeypatch.setattr(indexed_bindings, "_filter_frame", fail_filter)
    ops = [
        n({"public": 100}, name="source"),
        e_forward({"type": "A"}),
        n(name="destination"),
    ]
    with pytest.raises(error_type, match="sentinel"):
        indexed_bindings.try_indexed_connected_bindings_state(
            _graph(engine),
            ops,
            engine=ENGINE_ENUM[engine],
        )


def test_use_policy_sparse_serves_dense_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_nodes = 512
    src = np.repeat(np.arange(n_nodes, dtype=np.int64), 2)
    edges = pd.DataFrame(
        {
            "src": src,
            "dst": (src + np.tile(np.array([1, 2]), n_nodes)) % n_nodes,
            "type": ["X"] * len(src),
        }
    )
    # two hops: the seeded typed-hop fast path serves a one-hop two-alias RETURN first
    # (no cardinality gate of its own), so the connected-bindings gate needs a longer path
    query = (
        "MATCH (a {kind:'seed'})-[:X]->(b)-[:X]->(c) "
        "RETURN a.id AS a, c.id AS c ORDER BY a, c"
    )
    for dense, expected_reason in [(False, "served"), (True, "cost_frontier")]:
        kinds = ["seed"] * n_nodes if dense else ["seed"] + ["noise"] * (n_nodes - 1)
        nodes = pd.DataFrame({"id": np.arange(n_nodes), "kind": kinds})
        g = _graph_from_frames("pandas", nodes, edges)
        _, decisions = _assert_parity(
            g,
            query,
            "pandas",
            monkeypatch,
            seam="connected_bindings",
            index_policy="use",
            expect_served=not dense,
        )
        assert decisions[0]["reason"] == expected_reason


def _cudf_polars_available() -> bool:
    """Mirrors chain.py's own gate: engine='polars-gpu' EXECUTION raises ImportError without
    the RAPIDS cudf_polars stack. The helper-declines half needs no GPU at all."""
    import importlib.util

    return importlib.util.find_spec("cudf_polars") is not None


def test_explicit_polars_gpu_declines_indexed_helper() -> None:
    """CPU-only half: the indexed helper must decline for Engine.POLARS_GPU. Split out of the
    fall-back test so it runs wherever polars does, instead of being lost behind a GPU gate."""
    import graphistry.compute.gfql.index.bindings as indexed_bindings

    g = _graph("polars-gpu")
    ops = [
        n({"public": 100}, name="a"),
        e_forward({"type": "A"}),
        n(name="b"),
    ]
    assert indexed_bindings.try_indexed_connected_bindings_state(
        g, ops, engine=Engine.POLARS_GPU
    ) is None


@pytest.mark.skipif(
    not _cudf_polars_available(),
    reason="engine='polars-gpu' execution requires the RAPIDS cudf_polars stack",
)
def test_explicit_polars_gpu_declines_indexed_helper_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = _graph("polars-gpu")
    _assert_parity(
        g,
        CONNECTED_QUERY,
        "polars-gpu",
        monkeypatch,
        seam="connected_bindings",
        expect_served=False,
    )

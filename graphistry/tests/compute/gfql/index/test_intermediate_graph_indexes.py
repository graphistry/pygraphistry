"""Named intermediate graphs retain native indexes for later non-empty stages."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry import call, e_forward, is_in, let, n, ref
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ast import ASTRef
from graphistry.compute.exceptions import GFQLTypeError, GFQLValidationError
from graphistry.compute.gfql.index import get_registry
from graphistry.compute.gfql.index.wire import CreateIndex, DropIndex, ShowIndexes
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine):
    rng = np.random.default_rng(4)
    nodes = pd.DataFrame({"id": np.arange(4000), "region": rng.integers(0, 4, 4000), "score": np.where(np.arange(4000) % 5, 1.5, np.nan)})
    edges = pd.DataFrame({"s": rng.integers(0, 4000, 24000), "d": rng.integers(0, 4000, 24000), "eid": np.arange(24000), "weight": np.where(np.arange(24000) % 5, 2.5, np.nan)})
    region = nodes.loc[nodes.region == 3, "id"]
    induced = edges.loc[edges.s.isin(region) & edges.d.isin(region)]
    seeds = induced.s.drop_duplicates().head(5).tolist()
    return graphistry.edges(df_to_engine(edges, Engine(engine)), "s", "d", "eid").nodes(df_to_engine(nodes, Engine(engine)), "id"), seeds


def native(seeds, ops):
    return let({
        "sub": [n({"region": 3}), e_forward(), n({"region": 3})],
        "indexed": ref("sub", ops),
        "out": ref("indexed", [n({"id": is_in(seeds)}), e_forward(), n()]),
    })


def cypher(seeds, ops):
    query = "GRAPH sub = GRAPH { MATCH (a {region:3})-[e]->(b {region:3}) } "
    previous = "sub"
    for i, operation in enumerate(ops):
        current = f"indexed{i}"
        query += f"GRAPH {current} = GRAPH {{ USE {previous} CALL graphistry.{operation} }} "
        previous = current
    return query + f"USE {previous} MATCH (u)-[e]->(v) WHERE u.id IN {seeds} RETURN v ORDER BY v.id"


def assert_same_nodes(actual, expected, engine, form):
    if engine == "cudf" and form != "cypher":
        # General cuDF joins do not promise row order for native graph results.
        assert_same_frame(actual.sort_values("id").reset_index(drop=True), expected.sort_values("id").reset_index(drop=True), engine)
    else:
        assert_same_frame(actual, expected, engine)


def assert_same_edges(actual, expected, engine):
    if engine.startswith("polars"):
        assert_same_frame(actual, expected, engine)
    elif engine == "cudf" and len(expected):
        assert_same_frame(actual[expected.columns].sort_values("eid").reset_index(drop=True), expected.sort_values("eid").reset_index(drop=True), engine)
    else:
        # Existing indexed hops group edges by seed; empty Cypher edges may reorder columns.
        assert_same_frame(actual[expected.columns].sort_index(), expected.sort_index(), engine)


@pytest.mark.parametrize("form", ["call", "wire", "cypher"])
@pytest.mark.parametrize("engagement", [
    False,
    pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-seeded", "index-hop", "cypher-fast")),
])
def test_intermediate_index_matches_scan_and_serves_later_hop(engine, form, engagement, monkeypatch):
    g, seeds = graph(engine)
    original_nodes, original_edges = g._nodes, g._edges
    if form == "cypher":
        query = cypher(seeds, ["create_index.write({kind:'edge_out_adj'})", "create_index.write({kind:'node_id'})"])
        baseline = cypher(seeds, [])
    else:
        ops = ([CreateIndex("edge_out_adj"), CreateIndex("node_id")] if form == "wire" else
               [call("create_index", {"kind": "edge_out_adj"}), call("create_index", {"kind": "node_id"})])
        query, baseline = native(seeds, ops), native(seeds, [])
    scan = g.gfql(baseline, engine=engine, index_policy="off")
    assert len(scan._nodes) > 0
    for policy in ["off", "use", "force"]:
        out = g.gfql(query, engine=engine, index_policy=policy)
        assert_same_nodes(out._nodes, scan._nodes, engine, form)
        assert_same_edges(out._edges, scan._edges, engine)
        if form == "wire":
            existing = native(seeds, [call("create_index", {"kind": "edge_out_adj"}), call("create_index", {"kind": "node_id"})])
            existing_edges = g.gfql(existing, engine=engine, index_policy=policy)._edges
            if engine == "cudf":
                assert_same_edges(out._edges, existing_edges, engine)
            else:
                assert_same_frame(out._edges, existing_edges, engine)
    assert g._nodes is original_nodes and g._edges is original_edges
    assert get_registry(g).is_empty()
    if engagement:
        import graphistry.compute.gfql.index.traverse as traverse
        import graphistry.compute.gfql.index.bindings as bindings
        import graphistry.compute.gfql.index.lookup as lookup
        original_lookup = traverse.lookup_edge_rows
        gathered = []

        def tracked_lookup(index, values, xp):
            rows, matched = original_lookup(index, values, xp)
            gathered.append(int(rows.shape[0]))
            return rows, matched

        monkeypatch.setattr(traverse, "lookup_edge_rows", tracked_lookup)
        monkeypatch.setattr(bindings, "lookup_edge_rows", tracked_lookup)
        monkeypatch.setattr(lookup, "lookup_edge_rows", tracked_lookup)
        report = g.gfql_explain(query, engine=engine, index_policy="use")
        assert report["error"] is None and report["used_index"]
        assert any(step.get("decision_code") == "index_selected" for step in report["steps"])
        assert any(count > 0 for count in gathered)


@pytest.mark.parametrize("form", ["wire", "cypher"])
def test_dropping_intermediate_adjacency_returns_scan_rows(engine, form):
    g, seeds = graph(engine)
    if form == "wire":
        query = native(seeds, [CreateIndex("edge_out_adj"), DropIndex(kind="edge_out_adj")])
        baseline = native(seeds, [])
    else:
        query = cypher(seeds, ["create_index.write({kind:'edge_out_adj'})", "drop_index.write({kind:'edge_out_adj'})"])
        baseline = cypher(seeds, [])
    actual = g.gfql(query, engine=engine, index_policy="use")
    expected = g.gfql(baseline, engine=engine, index_policy="off")
    assert len(actual._nodes) > 0
    assert_same_nodes(actual._nodes, expected._nodes, engine, form)
    assert_same_edges(actual._edges, expected._edges, engine)
    assert not g.gfql_explain(query, engine=engine, index_policy="use")["used_index"]


@pytest.mark.parametrize("form", ["wire", "call", "cypher"])
def test_index_stage_preserves_frames_and_drops_without_mutating_input(engine, form, monkeypatch):
    g, _ = graph(engine)
    from graphistry.compute.gfql.call import executor

    def forbidden_bridge(*args, **kwargs):
        raise AssertionError("Index stages must stay on native frames")

    monkeypatch.setattr(executor, "_bridge_graph_for_offengine_call", forbidden_bridge)
    monkeypatch.setenv("GFQL_POLARS_CALL_MODE", "strict")
    if form == "cypher":
        query = "GRAPH { CALL graphistry.create_index.write({kind:'edge_out_adj', name:'my_edges'}) }"
        drop = "GRAPH { CALL graphistry.drop_index.write({kind:'edge_out_adj'}) }"
    else:
        ops = [CreateIndex("edge_out_adj", name="my_edges")] if form == "wire" else [call("create_index", {"kind": "edge_out_adj", "name": "my_edges"})]
        query = let({"sub": [], "out": ref("sub", ops)})
        drop = let({"sub": [], "out": ref("sub", [DropIndex(kind="edge_out_adj")])})
    indexed = g.gfql(query, engine=engine)
    assert indexed._nodes is g._nodes
    assert_same_frame(indexed._edges, g._edges, engine)
    assert get_registry(g).is_empty()
    index = get_registry(indexed).get("edge_out_adj")
    assert index is not None and index.name == "my_edges"
    assert index.engine.value == engine
    assert index.source_ref is indexed._edges
    dropped = indexed.gfql(drop, engine=engine)
    assert dropped._nodes is indexed._nodes
    assert_same_frame(dropped._edges, indexed._edges, engine)
    assert get_registry(dropped).is_empty()
    assert get_registry(indexed).get("edge_out_adj") is index


def test_ref_wire_ops_round_trip_and_match_chain_constraints():
    ops = [CreateIndex("node_prop", column="region"), DropIndex(kind="node_prop", column="region")]
    operation = ref("sub", ops)
    operation.validate()
    assert ASTRef.from_json(operation.to_json()).to_json() == operation.to_json()
    for unsupported in [ShowIndexes(), DropIndex(name="custom")]:
        with pytest.raises(GFQLTypeError) as error:
            ref("sub", [unsupported])
        assert error.value.code is not None
    with pytest.raises(GFQLTypeError) as error:
        ref("sub", "bad").validate()
    assert error.value.context["field"] == "chain"


@pytest.mark.parametrize("procedure", [
    "create_index.write()",
    "create_index.write({kind:'unknown'})",
    "create_index.write({kind:'edge_out_adj', typo:1})",
    "create_index.write({kind:'edge_out_adj'}) YIELD value",
    "create_index({kind:'edge_out_adj'})",
    "drop_index.write({name:'custom'})",
])
def test_cypher_index_stage_rejects_invalid_calls(procedure):
    from graphistry.compute.gfql.cypher import compile_cypher
    with pytest.raises(GFQLValidationError) as error:
        compile_cypher(f"GRAPH {{ CALL graphistry.{procedure} }}")
    assert error.value.code is not None


@pytest.mark.parametrize("form", ["wire", "cypher"])
def test_each_invocation_builds_new_indexes_and_drop_all_preserves_other_graphs(engine, form):
    g, _ = graph(engine)
    if form == "wire":
        query = let({"sub": [], "out": ref("sub", [CreateIndex("edge_out_adj"), CreateIndex("node_id")])})
        drop = let({"sub": [], "out": ref("sub", [DropIndex()])})
    else:
        query = "GRAPH { CALL graphistry.create_index.write({kind:'edge_out_adj'}) }"
        drop = "GRAPH { CALL graphistry.drop_index.write() }"
    first = g.gfql(query, engine=engine)
    second = g.gfql(query, engine=engine)
    assert get_registry(first).get("edge_out_adj") is not get_registry(second).get("edge_out_adj")
    assert get_registry(first.gfql(drop, engine=engine)).is_empty()
    assert get_registry(first).get("edge_out_adj") is not None
    assert get_registry(second).get("edge_out_adj") is not None
    assert get_registry(g).is_empty()


@pytest.mark.parametrize("kind,column", [("node_prop", "region"), ("edge_prop", "s")])
def test_cypher_index_stage_accepts_parameterized_property_options(engine, kind, column):
    g, _ = graph(engine)
    indexed = g.gfql(
        "GRAPH { CALL graphistry.create_index.write($options) }", engine=engine,
        params={"options": {"kind": kind, "column": column, "name": "my_property"}},
    )
    metadata = indexed.show_indexes(engine=engine)
    assert metadata["name"].tolist() == ["my_property"]
    assert metadata["valid"].tolist() == [True]
    assert metadata["usable"].tolist() == [True]
    assert get_registry(g).is_empty()
    dropped = indexed.gfql(
        "GRAPH { CALL graphistry.drop_index.write($options) }", engine=engine,
        params={"options": {"kind": kind, "column": column}},
    )
    assert get_registry(dropped).is_empty()


def test_cypher_index_stage_preserves_schema_and_call_policy(engine):
    from graphistry.schema import EdgeType, GraphSchema, NodeType
    g, _ = graph(engine)
    g = g.bind(schema=GraphSchema(
        node_types=[NodeType("Node", {"id": int, "region": int})],
        edge_types=[EdgeType("REL", "Node", "Node", {})],
        node_id_column="id", edge_source_column="s", edge_destination_column="d",
    ))
    events = []

    def hook(context):
        events.append((context["phase"], context["call_op"], context["call_params"]["kind"]))

    query = "GRAPH { CALL graphistry.create_index.write({kind:'edge_out_adj'}) }"
    result = g.gfql(query, engine=engine, policy={"precall": hook, "postcall": hook})
    assert result._gfql_schema is g._gfql_schema
    assert result._node == g._node and result._source == g._source and result._destination == g._destination
    assert events == [("precall", "create_index", "edge_out_adj"), ("postcall", "create_index", "edge_out_adj")]


def test_ref_drops_one_property_and_keeps_other_resident_indexes(engine):
    g, _ = graph(engine)
    indexed = g.create_index("node_id", engine=engine).create_index("node_prop", column="region", engine=engine)
    query = let({"base": [], "out": ref("base", [DropIndex(kind="node_prop", column="region")])})
    result = indexed.gfql(query, engine=engine)
    assert not get_registry(result).node_props
    assert get_registry(result).get("node_id") is get_registry(indexed).get("node_id")
    assert get_registry(indexed).node_props


def test_cypher_index_stage_obeys_policy_denial_before_building(engine):
    from graphistry.compute.gfql.policy import PolicyException
    g, _ = graph(engine)

    def deny(context):
        raise PolicyException("precall", "index creation denied")

    with pytest.raises(PolicyException) as error:
        g.gfql("GRAPH { CALL graphistry.create_index.write({kind:'edge_out_adj'}) }", engine=engine, policy={"precall": deny})
    assert error.value.phase == "precall"
    assert get_registry(g).is_empty()


@pytest.mark.route_engaged("native-fast", "polars-seeded", "index-hop", "cypher-fast")
def test_two_later_refs_reuse_intermediate_indexes(engine, monkeypatch):
    import graphistry.compute.gfql.index as indexes
    g, seeds = graph(engine)
    build = indexes.create_index
    builds = []

    def tracked_build(graph, kind, **kwargs):
        builds.append(kind)
        return build(graph, kind, **kwargs)

    monkeypatch.setattr(indexes, "create_index", tracked_build)
    hop = [n({"id": is_in(seeds)}), e_forward(), n()]
    query = let({
        "sub": [n({"region": 3}), e_forward(), n({"region": 3})],
        "indexed": ref("sub", [CreateIndex("edge_out_adj"), CreateIndex("node_id")]),
        "first": ref("indexed", hop),
        "second": ref("indexed", hop),
    })
    result = g.gfql(query, engine=engine, index_policy="use")
    assert len(result._nodes) > 0
    assert builds == ["edge_out_adj", "node_id"]
    report = g.gfql_explain(query, engine=engine, index_policy="use")
    assert sum(step.get("decision_code") == "index_selected" for step in report["steps"]) >= 2

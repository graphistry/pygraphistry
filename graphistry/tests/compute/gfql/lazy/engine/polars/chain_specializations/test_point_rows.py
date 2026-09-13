"""Native indexed source rows preserve the existing Polars result contract."""
import pytest
import graphistry
from graphistry.compute.ast import e_forward, e_reverse, n, rows, select
from graphistry.compute.gfql.lazy.engine.polars.chain_specializations.point_rows import _try_point_rows_polars
from graphistry.tests.compute.chain_specializations.test_point_rows import graph
from graphistry.tests.compute.gfql.routes.switch import routes_off

pl = pytest.importorskip("polars")


def make_graph(variant="base"):
    base = graph("pandas", variant)
    nodes = pl.from_pandas(base._nodes).reverse()
    edges = pl.from_pandas(base._edges)
    return graphistry.nodes(nodes, "key").edges(edges, "s", "d", "eid").gfql_index_all(engine="polars").gfql_index_node_props(["id"], engine="polars")


@pytest.mark.parametrize("variant", ["base", "loop", "dangling", "nonleading", "empty", "nullable", "timestamp", "float-null"])
@pytest.mark.parametrize("source", ["a", "b"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("project", [False, True])
def test_source_rows_exact_parity(variant, source, reverse, project):
    g = make_graph(variant)
    edge = e_reverse if reverse else e_forward
    ops = [n({"id": 1000}, name="a"), edge({"type": "X"}, name="e"), n({"kind": "Person"}, name="b"), rows(source=source)]
    if project:
        ops.append(select([("id", source + ".id"), ("value", source + ".value")]))
    direct = _try_point_rows_polars(g, ops)
    assert direct is not None
    with routes_off(["polars-point-rows"]):
        expected = g.gfql(ops, engine="polars")
    actual = g.gfql(ops, engine="polars")
    from polars.testing import assert_frame_equal
    for result in (direct, actual):
        assert_frame_equal(result._nodes, expected._nodes)
        assert_frame_equal(result._edges, expected._edges)
        for attr in ("_node", "_edge", "_source", "_destination", "_gfql_rows_base_graph", "_gfql_start_nodes"):
            assert getattr(result, attr) == getattr(expected, attr)


@pytest.mark.parametrize("reason", ["unindexed", "collision", "start", "gpu", "extra-params"])
def test_source_rows_decline_boundaries(reason):
    from graphistry.compute.gfql.lazy import target_mode, ExecutionTarget
    g = make_graph()
    ops = [n({"id": 1000}, name="a"), rows(source="a")]
    start = None
    if reason == "unindexed":
        g = graphistry.nodes(g._nodes.clone(), "key").edges(g._edges.clone(), "s", "d")
    elif reason == "collision":
        g = g.nodes(g._nodes.with_columns(pl.lit("user").alias("a")))
    elif reason == "start":
        start = g._nodes.head(1)
    elif reason == "extra-params":
        ops[-1].params["attach_prop_aliases"] = ["a"]
    with target_mode(ExecutionTarget.GPU if reason == "gpu" else ExecutionTarget.CPU):
        assert _try_point_rows_polars(g, ops, start) is None


@pytest.mark.parametrize("project", [False, True])
def test_single_node_property_multihit_order(project):
    from polars.testing import assert_frame_equal
    g = make_graph()
    g = graphistry.nodes(g._nodes.with_columns((pl.col("id") % 2).alias("bucket")), "key").edges(g._edges, "s", "d", "eid")
    g = g.gfql_index_all(engine="polars").gfql_index_node_props(["bucket"], engine="polars")
    ops = [n({"bucket": 0}, name="a"), rows(source="a")]
    if project:
        ops.append(select([("id", "a.id")]))
    with routes_off(["polars-point-rows"]):
        expected = g.gfql(ops, engine="polars", index_policy="force")
    actual = g.gfql(ops, engine="polars", index_policy="force")
    assert_frame_equal(actual._nodes, expected._nodes)


@pytest.mark.parametrize("variant", ["base", "loop", "dangling", "nonleading", "empty", "nullable", "timestamp", "float-null", "categorical"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("multi_seed", [False, True])
def test_joined_projection_exact_path_bag(variant, reverse, multi_seed):
    from polars.testing import assert_frame_equal
    g = make_graph(variant)
    if multi_seed:
        g = graphistry.nodes(g._nodes.with_columns((pl.col("id") % 2).alias("bucket")), "key").edges(g._edges, "s", "d", "eid")
        g = g.gfql_index_all(engine="polars").gfql_index_node_props(["bucket"], engine="polars")
    edge = e_reverse if reverse else e_forward
    ops = [n({"bucket": 0} if multi_seed else {"id": 1000}, name="a"), edge({"type": "X"}, name="e"),
           n({"kind": "Person"}, name="b"), rows(),
           select([("source", "a.id"), ("target", "b.id"), ("value", "b.value"), ("weight", "e.weight")])]
    from graphistry.compute.gfql.index.api import with_index_policy
    direct = _try_point_rows_polars(with_index_policy(g, "force"), ops)
    assert direct is not None
    with routes_off(["polars-point-rows"]):
        expected = g.gfql(ops, engine="polars", index_policy="force")
    assert_frame_equal(direct._nodes, expected._nodes)
    assert_frame_equal(direct._edges, expected._edges)
    for attr in ("_node", "_edge", "_source", "_destination", "_gfql_rows_base_graph", "_gfql_start_nodes"):
        assert getattr(direct, attr) == getattr(expected, attr)


@pytest.mark.parametrize("items", [
    [("out", "a.id + 1")], [("out", "coalesce(a.content, b.content)")],
    [("out", "missing.id")], [("out", "a.missing")],
    [("out", "a.id"), ("out", "b.id")],
])
def test_joined_projection_declines_unsupported_items(items):
    g = make_graph()
    ops = [n({"id": 1000}, name="a"), e_forward({"type": "X"}), n(name="b"), rows(), select(items)]
    assert _try_point_rows_polars(g, ops) is None


@pytest.mark.parametrize("tail_filter", [{}, {"id": 1001}])
def test_joined_projection_temporal_constructor_text_declines(tail_filter):
    g = make_graph()
    g = graphistry.nodes(g._nodes.with_columns(pl.lit("date({year: 2020})").alias("text")), "key").edges(g._edges, "s", "d", "eid").gfql_index_all(engine="polars")
    ops = [n({"key": 0}, name="a"), e_forward({"type": "X"}), n(tail_filter, name="b"), rows(), select([("text", "b.text")])]
    assert _try_point_rows_polars(g, ops) is None


@pytest.mark.parametrize("variant", ["base", "nullable", "categorical", "timestamp", "float-null", "empty"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("include_edge", [False, True])
def test_singleton_joined_projection_preserves_edge_bag(variant, reverse, include_edge):
    from polars.testing import assert_frame_equal
    g = make_graph(variant)
    edge = e_reverse if reverse else e_forward
    items = [("source", "a.id"), ("target", "b.id"), ("value", "b.value")]
    if include_edge:
        items.append(("weight", "e.weight"))
    ops = [n({"id": 1000}, name="a"), edge({"type": "X"}, name="e"),
           n({"id": 1001}, name="b"), rows(), select(items)]
    direct = _try_point_rows_polars(g, ops)
    assert direct is not None
    with routes_off(["polars-point-rows"]):
        expected = g.gfql(ops, engine="polars")
    assert_frame_equal(direct._nodes, expected._nodes)
    assert_frame_equal(direct._edges, expected._edges)
    assert direct._nodes.height == (0 if variant == "empty" else 1 if reverse else 2)


@pytest.mark.parametrize("variant", ["single", "parallel", "loop", "dangling", "null", "filtered", "mixed", "empty", "large-ids", "two-seeds"])
@pytest.mark.parametrize("source", ["a", "b"])
@pytest.mark.parametrize("reverse", [False, True])
def test_singleton_bookkeeping_preserves_indexed_endpoint_semantics(variant, source, reverse):
    from polars.testing import assert_frame_equal
    offset = 2**63 + 2 if variant == "large-ids" else 0
    dtype = pl.UInt64 if variant == "large-ids" else pl.Int64
    nodes = pl.DataFrame({
        "key": pl.Series([offset + i for i in range(100)], dtype=dtype),
        "id": [1000 + i for i in range(100)],
        "kind": ["Other" if i == 2 else "Person" for i in range(100)],
    })
    targets = {"parallel": [1, 1, 1], "loop": [0], "dangling": [9999], "null": [None],
               "filtered": [2], "mixed": [1, 2, 9999, None, 1], "empty": []}.get(variant, [1])
    starts = [0] * len(targets)
    if variant == "two-seeds":
        nodes = nodes.with_columns(pl.when(pl.col("key") == 2).then(1000).otherwise(pl.col("id")).alias("id"))
        starts, targets = [0, 2], [1, 1]
    columns = {"s": starts, "d": targets} if not reverse else {"s": targets, "d": starts}
    edges = pl.DataFrame({
        **{column: pl.Series([None if value is None else value + offset for value in values], dtype=dtype)
           for column, values in columns.items()},
        "eid": pl.Series(range(len(targets)), dtype=pl.Int64),
        "type": pl.Series(["X"] * len(targets), dtype=pl.String),
    })
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d", "eid").gfql_index_all(engine="polars").gfql_index_node_props(["id"], engine="polars")
    before_nodes, before_edges = g._nodes.clone(), g._edges.clone()
    edge = e_reverse if reverse else e_forward
    ops = [n({"id": 1000}, name="a"), edge({"type": "X"}, name="e"), n({"kind": "Person"}, name="b"), rows(source=source)]
    actual = g.gfql(ops, engine="polars", index_policy="use")
    with routes_off(["polars-point-rows"]):
        expected = g.gfql(ops, engine="polars", index_policy="use")
    assert_frame_equal(actual._nodes, expected._nodes)
    assert_frame_equal(actual._edges, expected._edges)
    assert_frame_equal(g._nodes, before_nodes)
    assert_frame_equal(g._edges, before_edges)

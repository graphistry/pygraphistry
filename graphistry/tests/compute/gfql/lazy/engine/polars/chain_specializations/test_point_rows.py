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

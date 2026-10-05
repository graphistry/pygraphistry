"""Native Polars membership hops use resident seed and adjacency gathers."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry import n, e_forward, e_reverse, is_in
from graphistry.Engine import Engine, df_to_engine
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine, reverse=False):
    if engine != "pandas":
        pytest.importorskip("cudf_polars" if engine == "polars-gpu" else engine)
    ids = np.arange(5000)
    nodes = pd.DataFrame({"key": ids, "id": ids + 10000, "label__Message": ids < 1000,
                          "label__Person": ids >= 1000, "score": np.where(ids % 5, 1.5, np.nan)})
    source = np.arange(1000).repeat(2)
    destination = 1000 + source % 1000
    edges = pd.DataFrame({"s": destination if reverse else source, "d": source if reverse else destination,
                          "eid": np.arange(2000), "type": "HAS_CREATOR", "weight": np.where(source % 5, 2.5, np.nan)})
    g = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "key").edges(df_to_engine(edges, Engine(engine)), "s", "d", "eid")
    return g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine)


@pytest.mark.parametrize("column", ["key", "id"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("representation", ["predicate", "list"])
@pytest.mark.parametrize("engagement", [False, pytest.param(True, marks=pytest.mark.route_engaged("polars-seeded", "native-fast", "index-hop"))])
def test_membership_seed_keeps_typed_nonempty_graph_and_engages(engine, column, reverse, representation, engagement, monkeypatch):
    g = graph(engine, reverse)
    original_nodes, original_edges = g._nodes, g._edges
    values = list(range(50)) if column == "key" else list(range(10000, 10050))
    values += [values[0], 999999]
    value = is_in(values) if representation == "predicate" else values
    query = [n({column: value, "label__Message": True}, name="m"),
             (e_reverse if reverse else e_forward)({"type": "HAS_CREATOR"}, name="e"),
             n({"label__Person": True}, name="p")]
    scan = g.gfql(query, engine=engine, index_policy="off")
    assert len(scan._nodes) == 100 and len(scan._edges) == 100
    for policy in ["off", "use", "force"]:
        out = g.gfql(query, engine=engine, index_policy=policy)
        if engine == "cudf":
            assert_same_frame(out._nodes.sort_values("key").reset_index(drop=True), scan._nodes.sort_values("key").reset_index(drop=True), engine)
            assert_same_frame(out._edges.sort_values("eid").reset_index(drop=True), scan._edges.sort_values("eid").reset_index(drop=True), engine)
        else:
            assert_same_frame(out._nodes, scan._nodes, engine)
            if engine == "pandas":
                assert_same_frame(out._edges.sort_values("eid").reset_index(drop=True), scan._edges.sort_values("eid").reset_index(drop=True), engine)
            else:
                assert_same_frame(out._edges, scan._edges, engine)
    assert g._nodes is original_nodes and g._edges is original_edges
    if engagement:
        import graphistry.compute.gfql.index.lookup as lookup
        import graphistry.compute.gfql.index.property_lookup as property_lookup
        original = lookup.lookup_edge_rows
        gathers = []
        seed_gathers = []
        seed_lookup = property_lookup.lookup_prop_rows if column == "id" else lookup.lookup_node_rows

        def spy(index, values, xp):
            rows, matched = original(index, values, xp)
            gathers.append(int(rows.shape[0]))
            return rows, matched

        def seed_spy(index, values, xp):
            rows = seed_lookup(index, values, xp)
            seed_gathers.append(int(rows.shape[0]))
            return rows

        monkeypatch.setattr(lookup, "lookup_edge_rows", spy)
        monkeypatch.setattr(property_lookup if column == "id" else lookup,
                            "lookup_prop_rows" if column == "id" else "lookup_node_rows", seed_spy)
        report = g.gfql_explain(query, engine=engine, index_policy="use")
        assert report["error"] is None
        assert any(step.get("seam") == "native_seeded_hop" and step.get("path") == "index" for step in report["steps"])
        assert any(count == 100 for count in gathers)
        assert any(count == 50 for count in seed_gathers)


@pytest.mark.parametrize("values", [[], [999999], [0, 0, 1]])
def test_empty_missing_and_duplicate_members_preserve_polars_frames(values):
    g = graph("polars")
    query = [n({"key": is_in(values)}, name="m"), e_forward(name="e"), n(name="p")]
    if not values:
        for policy in ["off", "use", "force"]:
            with pytest.raises(NotImplementedError):
                g.gfql(query, engine="polars", index_policy=policy)
        return
    scan = g.gfql(query, engine="polars", index_policy="off")
    for policy in ["use", "force"]:
        out = g.gfql(query, engine="polars", index_policy=policy)
        assert_same_frame(out._nodes, scan._nodes, "polars")
        assert_same_frame(out._edges, scan._edges, "polars")


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("change", ["off", "no-index", "no-node-index", "rebound-nodes", "rebound-edges"])
def test_membership_lane_respects_policy_and_stale_or_missing_indexes(engine, change, monkeypatch):
    if engine == "polars-gpu":
        pytest.importorskip("cudf_polars")
    pl = pytest.importorskip("polars")
    import graphistry.compute.gfql.lazy.engine.polars.chain as chain
    g = graph(engine)
    if change == "no-index":
        g = g.drop_index()
    elif change == "no-node-index":
        g = g.drop_index("node_id")
    elif change == "rebound-nodes":
        g = g.nodes(g._nodes.with_columns((pl.col("id") + 20000).alias("id")))
    elif change == "rebound-edges":
        g = g.edges(g._edges.reverse())
    query = [n({"id": is_in(list(range(10000, 10050))), "label__Message": True}, name="m"),
             e_forward({"type": "HAS_CREATOR"}, name="e"), n({"label__Person": True}, name="p")]
    real_lane = chain._try_seeded_chain_polars
    served = []

    def spy(*args, **kwargs):
        result = real_lane(*args, **kwargs)
        served.append(result is not None)
        return result

    monkeypatch.setattr(chain, "_try_seeded_chain_polars", spy)
    policy = "off" if change == "off" else "use"
    actual = g.gfql(query, engine=engine, index_policy=policy)
    assert not any(served)
    monkeypatch.setattr(chain, "_try_seeded_chain_polars", lambda *args, **kwargs: None)
    expected = g.gfql(query, engine=engine, index_policy=policy)
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)
    assert len(actual._nodes) == (0 if change == "rebound-nodes" else 100)


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("column", ["key", "id"])
@pytest.mark.parametrize("offset", [2**53 + 1, 2**63 + 1])
def test_membership_seed_preserves_large_unsigned_integer_ids(engine, column, offset):
    if engine == "polars-gpu":
        pytest.importorskip("cudf_polars")
    pl = pytest.importorskip("polars")
    g = graph(engine)
    g = g.nodes(g._nodes.with_columns((pl.col(column).cast(pl.UInt64) + pl.lit(offset, dtype=pl.UInt64)).alias(column)))
    if column == "key":
        g = g.edges(g._edges.with_columns([(pl.col(c).cast(pl.UInt64) + pl.lit(offset, dtype=pl.UInt64)).alias(c) for c in ["s", "d"]]))
    g = g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine)
    first = offset + (10000 if column == "id" else 0)
    query = [n({column: is_in(list(range(first, first + 50))), "label__Message": True}, name="m"),
             e_forward({"type": "HAS_CREATOR"}), n({"label__Person": True}, name="p")]
    scan = g.gfql(query, engine=engine, index_policy="off")
    assert len(scan._nodes) == 100 and len(scan._edges) == 100
    for policy in ["use", "force"]:
        actual = g.gfql(query, engine=engine, index_policy=policy)
        assert_same_frame(actual._nodes, scan._nodes, engine)
        assert_same_frame(actual._edges, scan._edges, engine)


@pytest.mark.parametrize("values", [[True], [0.0, 1.0], [0, "1"], [None], [0, None], [np.nan], [2**65]])
def test_membership_query_boundaries_keep_canonical_polars_results_or_errors(values, monkeypatch):
    pl = pytest.importorskip("polars")
    import graphistry.compute.gfql.lazy.engine.polars.chain as chain
    from graphistry.compute.exceptions import GFQLValidationError
    g = graph("polars")
    query = [n({"key": is_in(values)}, name="m"), e_forward(name="e"), n(name="p")]

    def outcome(policy):
        try:
            return g.gfql(query, engine="polars", index_policy=policy)
        except (GFQLValidationError, pl.exceptions.PolarsError, TypeError, ValueError, OverflowError, NotImplementedError) as error:
            return error

    for policy in ["off", "use", "force"]:
        actual = outcome(policy)
        with monkeypatch.context() as patch:
            patch.setattr(chain, "_try_seeded_chain_polars", lambda *args, **kwargs: None)
            expected = outcome(policy)
        if isinstance(expected, Exception):
            assert type(actual) is type(expected)
            assert getattr(actual, "code", None) == getattr(expected, "code", None)
            assert getattr(actual, "context", None) == getattr(expected, "context", None)
        else:
            assert_same_frame(actual._nodes, expected._nodes, "polars")
            assert_same_frame(actual._edges, expected._edges, "polars")

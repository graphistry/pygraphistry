"""Seeded polars point rows resolve seed, edges and tail by one scan each when no index can serve."""
import pytest

import graphistry
from graphistry import e_forward, e_reverse, e_undirected, n
from graphistry.compute.ast import order_by, rows, select
from graphistry.compute.gfql.index.api import index_trace
from graphistry.tests.compute.gfql.routes.switch import routes_off

pl = pytest.importorskip("polars")
from polars.testing import assert_frame_equal  # noqa: E402

SEAM = "point_rows_scan"


def graph(duplicate_tail=False):
    nodes = pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7] + ([5] if duplicate_tail else []),
        "kind": ["P", "P", "P", "P", "C", "C", "P"] + (["C"] if duplicate_tail else []),
        "name": ["a", "b", None, "d", "x", "y", "z"] + (["x2"] if duplicate_tail else []),
        "score": [1.5, None, 3.0, 4.0, None, None, 7.0] + ([None] if duplicate_tail else []),
    })
    edges = pl.DataFrame({
        "s": [1, 2, 3, 1, 2, 4, 4, 3, 7, 1],
        "d": [5, 5, 6, 2, 3, 1, 2, 4, 7, 2],
        "type": ["LOC", "LOC", "LOC", "KNOWS", "KNOWS", "KNOWS", "KNOWS", "KNOWS", "KNOWS", "KNOWS"],
        "since": [10, 20, 30, 40, 50, 60, 70, 80, 90, None],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


SELECT_R = select([("fid", "f.id"), ("fname", "f.name"), ("since", "r.since")])
SHAPES = {
    "forward_joined": ([n({"id": 1}, name="p"), e_forward({"type": "LOC"}), n({"kind": "C"}, name="c"),
                        rows(), select([("pname", "p.name"), ("city", "c.id")])], 1),
    "reverse_edge_alias": ([n({"id": 2}, name="p"), e_reverse({"type": "KNOWS"}, name="r"), n(name="f"),
                            rows(), SELECT_R], 3),
    "undirected_ordered": ([n({"id": 4, "kind": "P"}, name="p"), e_undirected({"type": "KNOWS"}, name="r"),
                            n({"kind": "P"}, name="f"), rows(), SELECT_R,
                            order_by([("since", "desc"), ("fid", "asc")])], 3),
    "undirected_self_loop": ([n({"id": 7}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(name="f"),
                              rows(), SELECT_R], 1),
    "undirected_parallel_edges": ([n({"id": 1}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(name="f"),
                                   rows(), SELECT_R], 3),
    "undirected_two_seeds": ([n({"kind": "P"}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(name="f"),
                              rows(), select([("pid", "p.id"), ("fid", "f.id"), ("since", "r.since")])], 13),
    "seed_absent": ([n({"id": 99}, name="p"), e_forward(), n(name="f"), rows(), select([("fid", "f.id")])], 0),
    "source_rows": ([n({"id": 1}, name="p"), e_forward({"type": "KNOWS"}), n(name="f"), rows(source="f")], 1),
    "source_rows_ordered": ([n({"id": 4}, name="p"), e_forward(), n(name="f"), rows(source="f"),
                             select([("fid", "f.id"), ("score", "f.score")]), order_by([("fid", "desc")])], 2),
}


def _run(g, query, index_policy="off"):
    return g.gfql(query, engine="polars", index_policy=index_policy)


def _served(g, query, index_policy="off"):
    with index_trace() as steps:
        _run(g, query, index_policy)
    return {bool(s["served"]) for s in steps if s.get("op") == "fast_path" and s.get("seam") == SEAM}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_scan_point_rows_match_the_general_path(shape):
    query, expected_rows = SHAPES[shape]
    g = graph()
    with routes_off(["polars-point-rows"]):
        expected = _run(g, query)
    assert len(expected._nodes) == expected_rows
    actual = _run(g, query)
    assert_frame_equal(actual._nodes, expected._nodes)
    assert_frame_equal(actual._edges, expected._edges)


@pytest.mark.route_engaged("polars-point-rows")
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_scan_point_rows_serve(shape):
    assert _served(graph(), SHAPES[shape][0]) == {True}


@pytest.mark.parametrize("index_policy", ["auto", "force"])
def test_index_policy_that_may_serve_declines_the_scan(index_policy):
    query, expected_rows = SHAPES["undirected_ordered"]
    g = graph()
    assert _served(g, query, index_policy) == set()
    assert len(_run(g, query, index_policy)._nodes) == expected_rows


def test_resident_index_declines_the_scan():
    g = graph().create_index("node_prop", column="name", engine="polars")
    query = SHAPES["forward_joined"][0]
    assert _served(g, query, "use") == set()
    assert len(_run(g, query, "use")._nodes) == 1


def test_undirected_source_rows_decline():
    query = [n({"id": 4}, name="p"), e_undirected(), n(name="f"), rows(source="f")]
    g = graph()
    assert _served(g, query) == set()
    with routes_off(["polars-point-rows"]):
        expected = _run(g, query)
    assert_frame_equal(_run(g, query)._nodes, expected._nodes)


def test_duplicate_tail_ids_decline_to_the_general_path():
    query = SHAPES["forward_joined"][0]
    g = graph(duplicate_tail=True)
    assert _served(g, query) == set()
    with routes_off(["polars-point-rows"]):
        expected = _run(g, query)
    assert len(expected._nodes) == 2
    assert_frame_equal(_run(g, query)._nodes, expected._nodes)


def test_pandas_answers_match():
    g = graph()
    for shape, (query, expected_rows) in SHAPES.items():
        if shape == "source_rows":
            continue  # bare rows(source=) column order differs by engine on the general path too
        expected = g.nodes(g._nodes.to_pandas(), "id").edges(g._edges.to_pandas(), "s", "d").gfql(
            query, engine="pandas", index_policy="off")._nodes
        actual = _run(g, query)._nodes.to_pandas()
        assert len(actual) == expected_rows, shape
        assert actual.columns.tolist() == expected.columns.tolist(), shape
        def bag(frame):
            def cell(value):
                return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            return sorted(repr([cell(v) for v in row]) for row in frame.astype(object).where(frame.notna(), None).values.tolist())
        assert bag(actual) == bag(expected), shape


@pytest.mark.parametrize("edge", [e_forward(name="r"), e_undirected(name="r")])
def test_many_seeds_use_the_join_branch_and_match(edge):
    size = 40
    nodes = pl.DataFrame({"id": list(range(size)), "kind": ["P"] * size, "name": [f"n{i}" for i in range(size)]})
    edges = pl.DataFrame({"s": list(range(size)) + [0], "d": [(i + 1) % size for i in range(size)] + [0],
                          "since": list(range(size + 1))})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = [n({"kind": "P"}, name="p"), edge, n(name="f"), rows(),
             select([("pid", "p.id"), ("fid", "f.id"), ("since", "r.since")])]
    assert _served(g, query) == {True}
    with routes_off(["polars-point-rows"]):
        expected = _run(g, query)
    assert len(expected._nodes) == (size + 1 if edge.direction == "forward" else 2 * size + 1)
    assert_frame_equal(_run(g, query)._nodes, expected._nodes)


def _outcome(g, query):
    try:
        out = _run(g, query)
    except Exception as error:  # the outcome under test is the error class and message
        return type(error).__name__, str(error)
    return "rows", out._nodes.to_dicts(), [str(dtype) for dtype in out._nodes.dtypes], out._edges.columns


def _same_as_general_path(g, query):
    actual = _outcome(g, query)
    with routes_off(["polars-point-rows"]):
        expected = _outcome(g, query)
    assert actual == expected
    return actual


def _precision_graph(kind):
    """Five nodes whose ids differ by single nanoseconds, so a Python value (microseconds) cannot tell them apart."""
    dtype = {"ns_datetime": pl.Datetime("ns"), "ns_duration": pl.Duration("ns"), "time": pl.Time}[kind]
    ids = pl.Series([1, 2, 3, 4, 5]).cast(pl.Int64).cast(dtype)
    nodes = pl.DataFrame({"id": ids, "name": ["a", "b", "c", "d", "e"]})
    edges = pl.DataFrame({"s": ids[[0, 0]], "d": ids[[1, 2]]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("kind", ["ns_datetime", "ns_duration", "time"])
@pytest.mark.parametrize("shape", ["source_rows", "joined"])
def test_ids_finer_than_a_python_value_keep_exact_rows(kind, shape):
    """Up to eight ids compare by equality; a Python round-trip drops nanoseconds, so only exact dtypes take that branch."""
    query = {
        "source_rows": [n({"name": "a"}, name="p"), e_forward(), n(name="f"), rows(source="f")],
        "joined": [n({"name": "a"}, name="p"), e_forward(name="r"), n(name="f"), rows(),
                   select([("fname", "f.name")])],
    }[shape]
    assert _same_as_general_path(_precision_graph(kind), query)[0] == "rows"


def test_duplicate_node_ids_decline_the_source_rows_form():
    g = graph(duplicate_tail=True)
    query = [n({"id": 1}, name="p"), e_forward({"type": "LOC"}), n({"kind": "C"}, name="c"), rows(source="c")]
    _same_as_general_path(g, query)
    assert _served(g, query) == set()


def _twelve_seed_graph(extra_edge_column):
    nodes = pl.DataFrame({"id": list(range(1, 13)), "kind": ["P"] * 12})
    edges = pl.DataFrame({"s": list(range(1, 12)), "d": list(range(2, 13)), "type": ["KNOWS"] * 11,
                          extra_edge_column: [1] * 11})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("name", ["__gfql_point_order__", "__gfql_point_side__", "__gfql_point_rows_order__"])
def test_user_columns_named_like_generated_columns_do_not_collide(name):
    few = graph()
    few = few.edges(few._edges.with_columns(pl.lit(1).alias(name)))
    seeded = [n({"id": 4}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(name="f"), rows(), SELECT_R]
    assert _same_as_general_path(few, seeded)[0] == "rows"
    many = [n({"kind": "P"}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(name="f"), rows(),
            select([("pid", "p.id"), ("fid", "f.id")])]
    assert _same_as_general_path(_twelve_seed_graph(name), many)[0] == "rows"


@pytest.mark.parametrize("column", ["s", "d"])
def test_selecting_an_edge_endpoint_column_keeps_the_general_path_outcome(column):
    query = [n({"id": 1}, name="p"), e_forward({"type": "KNOWS"}, name="r"), n(name="f"), rows(),
             select([("fid", "f.id"), ("end", f"r.{column}")])]
    g = graph()
    _same_as_general_path(g, query)
    with index_trace() as steps:
        _outcome(g, query)
    assert not [s for s in steps if s.get("seam") == SEAM and s.get("served")]


@pytest.mark.parametrize("source", ["f", "p"])
def test_an_id_repeated_across_the_seed_and_tail_filters_keeps_the_general_path(source):
    nodes = pl.DataFrame({"id": [1, 1, 2, 3], "kind": ["S", "T", "T", "S"], "name": ["a", "b", "c", "d"]})
    edges = pl.DataFrame({"s": [3, 1], "d": [1, 2], "type": ["K", "K"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = [n({"kind": "S"}, name="p"), e_forward(), n({"kind": "T"}, name="f"), rows(source=source)]
    assert _same_as_general_path(g, query)[0] == "rows"
    assert _served(g, query) == set()


def test_generated_helper_columns_are_not_selectable_from_the_edge_alias():
    from graphistry.compute.util import generate_safe_column_name_from

    g = graph()
    helper = generate_safe_column_name_from("__gfql_point_from__", list(g._edges.columns) + list(g._nodes.columns))
    query = [n({"id": 4}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(name="f"), rows(),
             select([("fid", "f.id"), ("x", f"r.{helper}")])]
    _same_as_general_path(g, query)


def test_selecting_an_edge_endpoint_column_matches_the_general_path_with_resident_indexes():
    g = graph().create_index("node_id", engine="polars").create_index("edge_out_adj", engine="polars")
    query = [n({"id": 1}, name="p"), e_forward({"type": "LOC"}, name="r"), n(name="f"), rows(),
             select([("fid", "f.id"), ("end", "r.d")])]

    def outcome():
        try:
            return "rows", _run(g, query, "use")._nodes.to_dicts()
        except Exception as error:  # the outcome under test is the error class
            return type(error).__name__, None

    actual = outcome()
    with routes_off(["polars-point-rows"]):
        expected = outcome()
    assert actual == expected

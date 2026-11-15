"""A boundary rows() call that rebuilds bindings from the base graph skips the unread middle traversal."""
import inspect
import os

import pytest

import graphistry
from graphistry import e_forward, e_reverse, e_undirected, n
from graphistry.Engine import Engine, df_cons
from graphistry.compute.ast import rows, select
from graphistry.compute.gfql.index.api import index_trace, with_index_policy
from graphistry.compute.predicates.numeric import gt
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame
from graphistry.tests.compute.gfql.routes.switch import routes_off

SEAM = "rows_bindings_skip_traversal"
ROUTE = "bindings-skip-traversal"


@pytest.fixture(params=[
    "pandas",
    "polars",
    pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1")),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(
        os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu")),
])
def engine(request):
    if request.param != "pandas":
        pytest.importorskip("cudf_polars" if request.param == "polars-gpu" else request.param)
    return request.param


def graph(engine):
    cons = df_cons(Engine(engine))
    nodes = cons({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "label__Person": [True, True, True, True, False, False, True],
        "label__City": [False, False, False, False, True, True, False],
        "name": ["a", "b", None, "d", "x", "y", "lonely"],
        "score": [1.5, None, 3.0, 4.0, None, None, 7.0],
    })
    nodes = nodes.assign(blank=nodes["score"] * float("nan")) if engine in ("pandas", "cudf") else (
        nodes.with_columns(blank=nodes["score"] * None))
    edges = cons({
        "s": [1, 2, 3, 1, 2, 4, 4, 3],
        "d": [5, 5, 6, 2, 3, 1, 2, 4],
        "type": ["LOC", "LOC", "LOC", "KNOWS", "KNOWS", "KNOWS", "KNOWS", "KNOWS"],
        "since": [10, 20, 30, 40, 50, 60, 70, 80],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


PERSON = {"label__Person": True}
SHAPES = {
    "seeded_forward_select": (
        [n({"id": 1, **PERSON}, name="p"), e_forward({"type": "LOC"}), n({"label__City": True}, name="c"),
         rows(), select([("pid", "p.id"), ("pname", "p.name"), ("city", "c.name")])],
        1,
    ),
    "seeded_undirected_edge_alias": (
        [n({"id": 4, **PERSON}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(PERSON, name="f"),
         rows(), select([("fid", "f.id"), ("since", "r.since"), ("score", "f.score")])],
        3,
    ),
    "unseeded_two_hop": (
        [n(PERSON, name="a"), e_forward({"type": "KNOWS"}), n(PERSON, name="b"), e_forward({"type": "LOC"}),
         n(name="c"), rows(), select([("a", "a.id"), ("b", "b.id"), ("c", "c.id")])],
        4,
    ),
    "reverse_all_null_property": (
        [n({"label__City": True}, name="c"), e_reverse({"type": "LOC"}), n(name="p"),
         rows(), select([("c", "c.id"), ("p", "p.id"), ("blank", "p.blank")])],
        3,
    ),
    "seed_absent_empty": (
        [n({"id": 99}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])],
        0,
    ),
    "explicit_binding_ops": (
        [n({"id": 1}, name="p"), e_forward({"type": "KNOWS"}, name="r"), n(name="q"),
         rows(binding_ops=[n({"id": 1}, name="p").to_json(), e_forward({"type": "KNOWS"}, name="r").to_json(),
                           n(name="q").to_json()]),
         select([("q", "q.id"), ("since", "r.since")])],
        1,
    ),
}
CYPHER = {
    "multi_alias_return": ("MATCH (p {id: 4})-[r]-(f) RETURN f.id AS fid, r.since AS since ORDER BY fid", 3),
    "single_alias_prefilter": (
        "MATCH (p)-[r]->(q) WHERE p.score > 2 RETURN p.id AS p, q.id AS q ORDER BY p, q", 4),
}


def _run(g, query, engine):
    return g.gfql(query, engine=engine, index_policy="off")


def _consulted(g, query, engine, index_policy="off"):
    with index_trace() as steps:
        g.gfql(query, engine=engine, index_policy=index_policy)
    return {bool(s["served"]) for s in steps if s.get("op") == "fast_path" and s.get("seam") == SEAM}


def _assert_matches_traversal(g, query, engine, expected_rows):
    with routes_off([ROUTE]):
        expected = _run(g, query, engine)
    assert len(expected._nodes) == expected_rows
    actual = _run(g, query, engine)
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_chain_rows_match_the_traversal_path(engine, shape):
    query, expected_rows = SHAPES[shape]
    _assert_matches_traversal(graph(engine), query, engine, expected_rows)


@pytest.mark.parametrize("shape", sorted(CYPHER))
def test_cypher_rows_match_the_traversal_path(engine, shape):
    query, expected_rows = CYPHER[shape]
    _assert_matches_traversal(graph(engine), query, engine, expected_rows)


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_chain_rows_skip_the_traversal(engine, shape):
    assert _consulted(graph(engine), SHAPES[shape][0], engine) == {True}


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("shape", sorted(CYPHER))
def test_cypher_rows_skip_the_traversal(engine, shape):
    assert _consulted(graph(engine), CYPHER[shape][0], engine) == {True}


DECLINED = {
    "unnamed_middle_reads_traversal_nodes": (
        [n({"id": 1}), e_forward(), n(), rows()], 3),
    "edge_table_reads_traversal_edges": (
        [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(table="edges")], 2),
    "source_alias_reads_traversal_nodes": (
        [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(source="q")], 2),
}


@pytest.mark.parametrize("shape", sorted(DECLINED))
def test_rows_that_read_the_traversal_keep_it(engine, shape):
    query, expected_rows = DECLINED[shape]
    g = graph(engine)
    assert _consulted(g, query, engine) == set()
    assert len(_run(g, query, engine)._nodes) == expected_rows


UNBOUND_QUERY = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]


@pytest.mark.route_engaged(ROUTE)
def test_unbound_node_frame_declines_the_skip(engine):
    assert _consulted(graphistry.edges(graph(engine)._edges, "s", "d"), UNBOUND_QUERY, engine) == {False}


def test_unbound_node_frame_keeps_the_traversal(engine):
    edges_only = graphistry.edges(graph(engine)._edges, "s", "d")
    query = UNBOUND_QUERY
    with routes_off([ROUTE]):
        expected = _run(edges_only, query, engine)
    assert len(expected._nodes) == 2
    assert_same_frame(_run(edges_only, query, engine)._nodes, expected._nodes, engine)


POLICIES = [("off", True), ("use", False), ("auto", False), ("force", False)]
NAMED_SEED = [n({"name": "a"}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("index_policy,served", POLICIES)
def test_index_policy_that_may_serve_keeps_the_traversal(engine, index_policy, served):
    g = graph(engine).create_index("node_prop", column="name", engine=engine)
    assert _consulted(g, NAMED_SEED, engine, index_policy) == {served}


@pytest.mark.parametrize("index_policy", [policy for policy, _ in POLICIES])
def test_every_index_policy_returns_the_same_rows(engine, index_policy):
    g = graph(engine).create_index("node_prop", column="name", engine=engine)
    expected = g.gfql(NAMED_SEED, engine=engine, index_policy="off")
    assert len(expected._nodes) == 2
    assert_same_frame(g.gfql(NAMED_SEED, engine=engine, index_policy=index_policy)._nodes, expected._nodes, engine)


@pytest.mark.route_engaged(ROUTE)
def test_default_policy_without_indexes_skips(engine):
    query = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]
    assert _consulted(graph(engine), query, engine, "use") == {True}


def test_policy_keeps_the_traversal():
    g = graph("pandas")
    seen = []
    query = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]
    with index_trace() as steps:
        out = g.gfql(query, engine="pandas", policy={"prechain": lambda ctx: seen.append(ctx.get("phase"))})
    assert not any(s.get("seam") == SEAM for s in steps)
    assert seen and len(out._nodes) == 2


@pytest.mark.parametrize("validate_schema", [True, False])
@pytest.mark.parametrize("bad_filter", [
    {"name": 5}, {"score": "high"}, {"missing": 1}, {"name": gt(1)},
])
@pytest.mark.parametrize("seed", [1, 99, None])
@pytest.mark.parametrize("string_ids", [False, True])
def test_filter_errors_match_the_traversal(engine, bad_filter, validate_schema, seed, string_ids):
    g = graph(engine)
    if string_ids:
        g = g.nodes(g._nodes.with_columns(g._nodes["id"].cast(str)) if engine.startswith("polars")
                    else g._nodes.assign(id=g._nodes["id"].astype(str)))
    g = with_index_policy(g, "off")
    first = n(name="p") if seed is None else n({"id": seed}, name="p")
    query = [first, e_forward(), n(bad_filter, name="q"), rows(), select([("q", "q.id")])]

    def outcome():
        try:
            out = g.chain(query, engine=engine, validate_schema=validate_schema)
        except Exception as error:  # the outcome under test is the error class and code
            return type(error), getattr(error, "code", None)
        return "rows", len(out._nodes)

    from graphistry.compute.chain import chain as chain_impl
    assert "validate_schema" in inspect.signature(inspect.unwrap(chain_impl)).parameters
    with routes_off([ROUTE]):
        expected = outcome()
    assert outcome() == expected


VAR_LENGTH = [
    "MATCH (p)-[*0..2]->(q) WHERE p.id = 7 RETURN p.id AS p, q.id AS q",
    "MATCH (p)-[*1..3]->(q) WHERE p.id = 1 RETURN p.id AS p, q.id AS q",
]


@pytest.mark.parametrize("query", VAR_LENGTH)
def test_variable_length_keeps_its_traversal_schema(engine, query):
    _assert_matches_traversal(graph(engine), query, engine, len(_run(graph(engine), query, engine)._nodes))


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("query", VAR_LENGTH)
def test_variable_length_declines_the_skip(engine, query):
    assert _consulted(graph(engine), query, engine) == {False}


def _widened(g, engine):
    since = g._edges["since"] * 2
    return g.edges(g._edges.with_columns(weight=since) if engine.startswith("polars") else g._edges.assign(weight=since))


def test_zero_row_twin_follows_input_schema(engine):
    query = SHAPES["seeded_undirected_edge_alias"][0]
    g = graph(engine)
    _assert_matches_traversal(g, query, engine, 3)
    _assert_matches_traversal(_widened(g, engine), query, engine, 3)
    assert "weight" in _run(_widened(g, engine), query, engine)._edges.columns


@pytest.mark.route_engaged(ROUTE)
def test_zero_row_twin_cache_fills_and_clears(engine):
    from graphistry.compute.gfql import rows_skip_traversal
    from graphistry.compute.gfql_unified import gfql_clear_caches

    gfql_clear_caches()
    assert not rows_skip_traversal._TWINS
    _run(graph(engine), SHAPES["seeded_undirected_edge_alias"][0], engine)
    assert len(rows_skip_traversal._TWINS) == 1
    _run(graph(engine), SHAPES["seeded_undirected_edge_alias"][0], engine)
    assert len(rows_skip_traversal._TWINS) == 1
    gfql_clear_caches()
    assert not rows_skip_traversal._TWINS


def test_cached_twin_keeps_each_graphs_categories():
    import pandas as pd

    query = SHAPES["seeded_undirected_edge_alias"][0]
    results = []
    for categories in (["x", "y"], ["p", "q", "z"]):
        g = graph("pandas")
        kinds = pd.Categorical([categories[i % 2] for i in range(len(g._edges))], categories=categories)
        g = g.edges(g._edges.assign(kind=kinds))
        _assert_matches_traversal(g, query, "pandas", 3)
        results.append(list(_run(g, query, "pandas")._edges["kind"].cat.categories))
    assert results == [["x", "y"], ["p", "q", "z"]]


def test_categorical_columns_match_the_traversal_on_every_engine(engine):
    import pandas as pd

    query = SHAPES["seeded_undirected_edge_alias"][0]
    g = graph(engine)
    kinds = ["x", "y"] * 4
    if engine == "polars":
        import polars as pl
        g = g.edges(g._edges.with_columns(kind=pl.Series(kinds).cast(pl.Categorical)))
    elif engine == "polars-gpu":
        pytest.skip("polars-gpu does not execute categorical columns; the decline is pinned elsewhere")
    else:
        frame = graph("pandas")._edges.assign(kind=pd.Categorical(kinds, categories=["y", "x"]))
        if engine == "cudf":
            import cudf
            frame = cudf.from_pandas(frame)
        g = g.edges(frame)
    _assert_matches_traversal(g, query, engine, 3)


ABSENT_FILTERS = {
    "absent_node_column": [n({"nope_col": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])],
    "absent_edge_column": [n({"id": 1}, name="p"), e_forward({"nope_col": 1}), n(name="q"), rows(), select([("q", "q.id")])],
    "absent_label": [n({"label__Nope": True}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])],
}


@pytest.mark.parametrize("shape", sorted(ABSENT_FILTERS))
def test_warnings_match_the_traversal_on_every_call(engine, shape):
    import warnings

    def observed(g):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rows_out = len(_run(g, ABSENT_FILTERS[shape], engine)._nodes)
        return rows_out, sorted((w.category.__name__, str(w.message)) for w in caught if "GFQL" in str(w.message))

    g = graph(engine)
    with routes_off([ROUTE]):
        expected = observed(g)
    assert expected[1], "the shape must warn under the traversal for this check to mean anything"
    for _ in range(3):  # the first call fills the zero-row twin cache, later calls reuse it
        assert observed(g) == expected


def _categorical_edges(engine):
    import pandas as pd

    g = graph(engine)
    kinds = ["x", "y"] * 4
    if engine == "polars":
        import polars as pl
        return g.edges(g._edges.with_columns(kind=pl.Series(kinds).cast(pl.Categorical)))
    frame = graph("pandas")._edges.assign(kind=pd.Categorical(kinds, categories=["y", "x"]))
    if engine == "cudf":
        import cudf
        frame = cudf.from_pandas(frame)
    return g.edges(frame)


@pytest.mark.route_engaged(ROUTE)
def test_categorical_columns_decline_the_skip(engine):
    if engine == "polars-gpu":
        pytest.skip("polars-gpu does not execute categorical columns")
    assert _consulted(_categorical_edges(engine), SHAPES["seeded_undirected_edge_alias"][0], engine) == {False}


@pytest.mark.route_engaged(ROUTE)
def test_polars_enum_columns_decline_the_skip():
    pl = pytest.importorskip("polars")
    g = graph("polars")
    g = g.edges(g._edges.with_columns(kind=pl.Series(["x", "y"] * 4).cast(pl.Enum(["y", "x"]))))
    assert _consulted(g, SHAPES["seeded_undirected_edge_alias"][0], "polars") == {False}


def _cells(frame):
    if hasattr(frame, "to_pandas") and not hasattr(frame, "to_dicts"):
        frame = frame.to_pandas()
    rows = frame.to_dicts() if hasattr(frame, "to_dicts") else frame.to_dict("records")
    return repr(rows)


def _outcome(g, query, engine, validate_schema=True):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = ("rows", _cells(with_index_policy(g, "off").chain(query, engine=engine, validate_schema=validate_schema)._nodes))
        except Exception as error:  # the outcome under test is the error class, code and message
            result = (type(error).__name__, getattr(error, "code", None), str(error))
    return result, sorted((w.category.__name__, str(w.message)) for w in caught if "GFQL" in str(w.message))


def _contains(pattern):
    from graphistry.compute.predicates.str import contains
    return contains(pattern, regex=True)


def _is_in(options):
    from graphistry.compute.predicates.is_in import is_in
    return is_in(options)


VALUE_PAIRS = {
    "valid_then_invalid_regex": ({"name": lambda: _contains("a")}, {"name": lambda: _contains("(")}),
    "empty_then_nonempty_is_in": ({"name": lambda: _is_in([])}, {"name": lambda: _is_in(["a"])}),
    "in_range_then_out_of_range_int": ({"id": lambda: 5}, {"id": lambda: 2 ** 70}),
    "number_then_nan": ({"score": lambda: 1.5}, {"score": lambda: float("nan")}),
    "value_then_none": ({"name": lambda: "a"}, {"name": lambda: None}),
    "number_then_string_on_numeric": ({"score": lambda: 1.5}, {"score": lambda: "high"}),
    "date_string_parses_then_not": ({"score": lambda: gt("2020-01-01")}, {"score": lambda: gt("not a date")}),
}


@pytest.mark.parametrize("validate_schema", [True, False])
@pytest.mark.parametrize("order", ["forward", "reverse"])
@pytest.mark.parametrize("pair", sorted(VALUE_PAIRS))
def test_filter_values_of_one_type_never_change_the_outcome(engine, pair, order, validate_schema):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    filters = [{column: make() for column, make in spec.items()} for spec in VALUE_PAIRS[pair]]
    filters = filters if order == "forward" else filters[::-1]
    g = graph(engine)
    queries = [
        [n(flt, name="p"), e_undirected(), n(name="f"), rows(), select([("fid", "f.id")])] for flt in filters
    ]
    gfql_clear_caches()
    actual = [_outcome(g, query, engine, validate_schema) for query in queries * 2]  # cold, then warm
    with routes_off([ROUTE]):
        expected = [_outcome(g, query, engine, validate_schema) for query in queries * 2]
    assert actual == expected


def _frame_for(engine, column, pandas_values, polars_values):
    import pandas as pd

    g = graph(engine)
    if engine.startswith("polars"):
        import polars as pl
        return g.nodes(g._nodes.with_columns(**{column: polars_values(pl)}))
    frame = g._nodes.to_pandas() if engine == "cudf" else g._nodes
    frame = frame.assign(**{column: pandas_values(pd)})
    if engine == "cudf":
        import cudf
        try:
            frame = cudf.from_pandas(frame)
        except Exception as error:  # an unsupported cuDF dtype is a construction limit, not a route outcome
            pytest.skip(f"cuDF cannot hold this column: {type(error).__name__}")
    return g.nodes(frame)


N = 7
DTYPE_MATRIX = {
    "string": (lambda pd: pd.Series(list("abcdefg")), lambda pl: pl.Series(list("abcdefg"))),
    "nullable_int": (lambda pd: pd.array([1, None, 3, 4, None, 6, 7], dtype="Int64"), lambda pl: pl.Series([1, None, 3, 4, None, 6, 7])),
    "nullable_bool": (lambda pd: pd.array([True, None, False] * 2 + [None], dtype="boolean"), lambda pl: pl.Series([True, None, False] * 2 + [None])),
    "naive_datetime": (lambda pd: pd.date_range("2020-01-01", periods=N), lambda pl: pl.Series(range(N)).cast(pl.Datetime("us"))),
    "tz_datetime": (
        lambda pd: pd.date_range("2020-01-01", periods=N, tz="America/New_York"),
        lambda pl: pl.Series(range(N)).cast(pl.Datetime("us", "America/New_York"))),
    "timedelta": (lambda pd: pd.to_timedelta(range(N), unit="s"), lambda pl: pl.Series(range(N)).cast(pl.Duration("us"))),
    "all_null": (lambda pd: pd.Series([None] * N, dtype="float64"), lambda pl: pl.Series([None] * N, dtype=pl.Float64)),
    "all_null_object": (lambda pd: pd.Series([None] * N, dtype="object"), lambda pl: pl.Series([None] * N, dtype=pl.Null)),
    "list": (lambda pd: pd.Series([[1, 2]] * N), lambda pl: pl.Series([[1, 2]] * N)),
    "struct": (lambda pd: pd.Series([{"a": 1}] * N), lambda pl: pl.Series([{"a": 1}] * N)),
    "decimal": (
        lambda pd: pd.Series([__import__("decimal").Decimal("1.50")] * N),
        lambda pl: pl.Series([__import__("decimal").Decimal("1.50")] * N, dtype=pl.Decimal(10, 2))),
    "large_string": (
        lambda pd: pd.Series(list("abcdefg"), dtype=pd.ArrowDtype(__import__("pyarrow").large_string())),
        lambda pl: pl.Series(list("abcdefg"))),
    "unordered_categorical": (lambda pd: pd.Categorical(list("abcabca")), lambda pl: pl.Series(list("abcabca")).cast(pl.Categorical)),
    "ordered_categorical": (
        lambda pd: pd.Categorical(list("abcabca"), categories=list("cba"), ordered=True),
        lambda pl: pl.Series(list("abcabca")).cast(pl.Enum(list("cba")))),
}
ROWS_EXPECTED = frozenset(DTYPE_MATRIX) - {"struct", "list", "decimal", "all_null_object"}  # these may be declined by an engine's canonical path too
MATRIX_QUERY = [n({"id": 4, **PERSON}, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(PERSON, name="f"), rows(), select([("fid", "f.id"), ("val", "f.val")])]


@pytest.mark.parametrize("dtype", sorted(DTYPE_MATRIX))
def test_dtype_matrix_matches_the_traversal(engine, dtype):
    if engine == "polars-gpu" and dtype not in ("string", "nullable_int", "all_null", "naive_datetime", "timedelta"):
        pytest.skip("polars-gpu declines this column type in the canonical path too")
    pandas_values, polars_values = DTYPE_MATRIX[dtype]
    g = _frame_for(engine, "val", pandas_values, polars_values)
    on = _outcome(g, MATRIX_QUERY, engine)
    with routes_off([ROUTE]):
        off = _outcome(g, MATRIX_QUERY, engine)
    assert on == off
    assert on[0][0] == "rows" or dtype not in ROWS_EXPECTED, on[0]


def _structural_graph(engine, case):
    g = graph(engine)
    if case == "empty_edges":
        return g.edges(g._edges[:0])
    if case == "empty_nodes":
        return g.nodes(g._nodes[:0])
    if case == "duplicate_node_ids":
        nodes = g._nodes
        both = nodes.vstack(nodes[:2]) if engine.startswith("polars") else type(nodes)(
            __import__("pandas").concat([nodes.to_pandas() if engine == "cudf" else nodes, nodes.to_pandas()[:2] if engine == "cudf" else nodes[:2]]))
        return g.nodes(both)
    return graphistry.edges(g._edges, "s", "d")  # edges only


@pytest.mark.parametrize("case", ["empty_edges", "empty_nodes", "duplicate_node_ids", "edges_only"])
def test_structural_cases_match_the_traversal(engine, case):
    query = [n({"id": 4}, name="p"), e_undirected(), n(name="f"), rows(), select([("fid", "f.id")])]
    g = _structural_graph(engine, case)
    on = _outcome(g, query, engine)
    with routes_off([ROUTE]):
        off = _outcome(g, query, engine)
    assert on == off
    assert on[0][0] == "rows" or case == "edges_only", on[0]


def test_concurrent_queries_fill_the_cache_consistently():
    import concurrent.futures as futures

    from graphistry.compute.gfql_unified import gfql_clear_caches

    gfql_clear_caches()
    graphs = [graph("pandas"), _widened(graph("pandas"), "pandas")]
    query = SHAPES["seeded_undirected_edge_alias"][0]
    with routes_off([ROUTE]):
        expected = [_outcome(g, query, "pandas") for g in graphs]
    gfql_clear_caches()
    with futures.ThreadPoolExecutor(8) as pool:
        actual = list(pool.map(lambda i: _outcome(graphs[i % 2], query, "pandas"), range(32)))
    assert [actual[i] for i in range(2)] == expected
    assert all(actual[i] == expected[i % 2] for i in range(32))


def test_column_named_schema_keys_on_dtypes():
    query = SHAPES["seeded_undirected_edge_alias"][0]
    for values in ([1] * 8, ["a"] * 8):
        g = graph("pandas")
        g = g.edges(g._edges.assign(schema=values))
        _assert_matches_traversal(g, query, "pandas", 3)


def test_explain_trace_does_not_depend_on_the_twin_cache(engine):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    query = SHAPES["seeded_forward_select"][0]
    g = graph(engine)
    gfql_clear_caches()
    cold = g.gfql_explain(query, engine=engine, index_policy="off")["steps"]
    warm = g.gfql_explain(query, engine=engine, index_policy="off")["steps"]
    assert cold == warm


def test_prefix_call_keeps_the_traversal():
    from graphistry.compute.ast import call

    g = graph("pandas")
    query = [call("get_degrees"), n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]
    with routes_off(["polars-point-rows"]), index_trace() as steps:
        out = g.gfql(query, engine="pandas", index_policy="off")
    assert not any(s.get("seam") == SEAM for s in steps)
    assert len(out._nodes) == 2


@pytest.mark.parametrize("unbind", ["node_column", "edges"])
def test_partially_bound_graph_matches_the_traversal(engine, unbind):
    g = graph(engine)
    g = g.bind(node=None) if unbind == "node_column" else graphistry.nodes(g._nodes, "id")
    query = [n({"id": 1}, name="p"), e_forward(), n(name="q"), rows(), select([("q", "q.id")])]

    def outcome():
        try:
            return "rows", _run(g, query, engine)._nodes.shape
        except Exception as error:  # the outcome under test is the error class and code
            return type(error), getattr(error, "code", None)

    with routes_off([ROUTE]):
        expected = outcome()
    assert outcome() == expected


def test_default_engine_twins_stay_per_frame_type(engine):
    query = SHAPES["seeded_undirected_edge_alias"][0]
    graph("pandas").gfql(query, index_policy="off")
    g = graph(engine)
    with routes_off([ROUTE]):
        expected = g.gfql(query, index_policy="off")
    actual = g.gfql(query, index_policy="off")
    assert type(actual._edges) is type(g._edges)
    assert_same_frame(actual._edges, expected._edges, engine)
    assert_same_frame(actual._nodes, expected._nodes, engine)


def test_cached_twin_keeps_category_order():
    import pandas as pd

    query = SHAPES["seeded_undirected_edge_alias"][0]
    for categories in (["x", "y"], ["y", "x"]):
        g = graph("pandas")
        g = g.edges(g._edges.assign(kind=pd.Categorical(["x", "y"] * 4, categories=categories)))
        _assert_matches_traversal(g, query, "pandas", 3)
        assert list(_run(g, query, "pandas")._edges["kind"].cat.categories) == categories


def _graph_with_attrs(tenant):
    g = graph("pandas")
    edges = g._edges.copy()
    edges.attrs = {"tenant": tenant}
    return g.edges(edges)


@pytest.mark.route_engaged(ROUTE)
def test_frames_with_attrs_decline_the_skip():
    query = SHAPES["seeded_undirected_edge_alias"][0]
    assert _consulted(_graph_with_attrs("A"), query, "pandas") == {False}


def test_frames_with_attrs_match_the_traversal_whatever_the_pandas_version():
    query = SHAPES["seeded_undirected_edge_alias"][0]
    for tenant in ("A", "B"):
        g = _graph_with_attrs(tenant)
        _assert_matches_traversal(g, query, "pandas", 3)
        with routes_off([ROUTE]):
            expected = _run(g, query, "pandas")._edges
        assert _run(g, query, "pandas")._edges.attrs == expected.attrs


@pytest.mark.parametrize("query_field", ["edge_query", "destination_node_query"])
def test_query_strings_keep_the_traversal(query_field):
    g = graph("pandas")
    g = g.edges(g._edges.assign(mixed=[1, "a", 2, "b", 3, "c", 4, "d"]))
    query = [n({"id": 99}, name="p"), e_forward(**{query_field: "mixed > 0" if query_field == "edge_query" else "id > 0"}),
             n(name="q"), rows(), select([("q", "q.id")])]

    def outcome():
        try:
            return "rows", len(_run(g, query, "pandas")._nodes)
        except Exception as error:  # the outcome under test is the error class
            return type(error), None

    with routes_off([ROUTE]):
        expected = outcome()
    assert outcome() == expected

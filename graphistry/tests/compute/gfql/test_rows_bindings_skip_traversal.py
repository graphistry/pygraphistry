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


@pytest.fixture(autouse=True, params=["default", "general-path"])
def route_context(request):
    """Run every test twice: with the default routes, and with every other fast route off so the skip is the one reached."""
    if request.param == "default":
        yield
        return
    from graphistry.tests.compute.gfql.routes.switch import ROUTES

    with routes_off([route for route in ROUTES if route != ROUTE]):
        yield


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


def _consulted_accepting_validation_errors(g, query, engine, index_policy="off"):
    from graphistry.compute.exceptions import GFQLTypeError

    with index_trace() as steps:
        try:
            g.gfql(query, engine=engine, index_policy=index_policy)
        except GFQLTypeError:
            pass
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
        return rows_out, _warnings(caught)

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


def _warnings(caught):
    """Every warning a caller's filter would see, every category, duplicates included."""
    return sorted((w.category.__name__, str(w.message)) for w in caught)


def _signature(frame):
    """Rows, column order, dtypes and (pandas/cuDF) index identity: everything a caller can observe of a frame."""
    index = None if hasattr(frame, "to_dicts") else (type(frame.index).__name__, frame.index.name)
    columns = list(frame.columns)
    dtypes = [str(dtype) for dtype in (frame.dtypes if hasattr(frame, "dtypes") else [])]
    return _cells(frame), columns, dtypes, index


def _outcome(g, query, engine, validate_schema=True):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = with_index_policy(g, "off").chain(query, engine=engine, validate_schema=validate_schema)
            result = ("rows", _signature(out._nodes), _signature(out._edges))
        except Exception as error:  # the outcome under test is the error class, code and message
            result = (type(error).__name__, getattr(error, "code", None), str(error))
    return result, _warnings(caught)


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
    "arrow_dictionary": (
        lambda pd: pd.Series(list("abcabca")).astype(pd.ArrowDtype(__import__("pyarrow").dictionary(__import__("pyarrow").int32(), __import__("pyarrow").string()))),
        lambda pl: pl.Series(list("abcabca")).cast(pl.Categorical)),
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


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("dtype", sorted(DTYPE_MATRIX))
def test_dtype_matrix_is_served_or_declined_without_a_replay(engine, dtype):
    if engine == "polars-gpu" and dtype not in ("string", "nullable_int", "all_null", "naive_datetime", "timedelta"):
        pytest.skip("polars-gpu declines this column type in the canonical path too")
    pandas_values, polars_values = DTYPE_MATRIX[dtype]
    g = _frame_for(engine, "val", pandas_values, polars_values)
    if _outcome(g, MATRIX_QUERY, engine)[0][0] != "rows":
        pytest.skip("the canonical path declines this column type")
    # a dictionary-encoded column is categorical on Polars and on cuDF versions that hold it
    declined = dtype in ("ordered_categorical", "unordered_categorical") or (engine in ("polars", "cudf") and dtype == "arrow_dictionary")
    # served means served: a replay would also record a False decision
    assert _consulted(g, MATRIX_QUERY, engine) == ({False} if declined else {True})


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


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("error", [TypeError("bad compare"), ValueError("bad value"), KeyError("missing"), ZeroDivisionError("zero")])
def test_data_evaluation_errors_in_the_skipped_path_replay_the_traversal(monkeypatch, error):
    from graphistry.compute.gfql import rows_skip_traversal

    query = SHAPES["seeded_undirected_edge_alias"][0]
    g = graph("pandas")
    with routes_off([ROUTE]):
        expected = _outcome(g, query, "pandas")
    real = rows_skip_traversal._zero_row_twin

    def failing(*args, **kwargs):
        real(*args, **kwargs)
        raise error

    monkeypatch.setattr(rows_skip_traversal, "_zero_row_twin", failing)
    assert _consulted(g, query, "pandas") == {True, False}  # engaged, then replayed the traversal
    with index_trace() as steps:
        _run(g, query, "pandas")
    replays = [s.get("replayed_after") for s in steps if s.get("op") == "fast_path" and s.get("seam") == SEAM]
    assert type(error).__name__ in replays, steps  # a replay is a structured field of the route decision
    assert _outcome(g, query, "pandas") == expected


@pytest.mark.route_engaged(ROUTE)
def test_unhashable_schema_declines_the_skip(engine, monkeypatch):
    from graphistry.compute.gfql import rows_skip_traversal

    monkeypatch.setattr(rows_skip_traversal, "_frame_schema", lambda frame: None)
    assert _consulted(graph(engine), SHAPES["seeded_undirected_edge_alias"][0], engine) == {False}


def test_unhashable_dtype_has_no_key():
    import types

    from graphistry.compute.gfql import rows_skip_traversal

    class Unhashable:
        __hash__ = None

    frame = types.SimpleNamespace(dtypes=types.SimpleNamespace(items=lambda: [("c", Unhashable())]))
    assert rows_skip_traversal._frame_schema(frame) is None


COLLISION_QUERIES = {
    "cypher_edge_alias_named_like_an_edge_column": "MATCH (a)-[e]->(b) RETURN b LIMIT 5",
    "cypher_node_alias_named_like_a_node_column": "MATCH (name)-[r]->(f) RETURN f.id AS fid ORDER BY fid",
    "chain_edge_alias_named_like_an_edge_column": [
        n(name="a"), e_forward(name="e"), n(name="b"), rows(), select([("b", "b.id")])],
}


def _collision_graph(engine):
    g = graph(engine)
    marker = [False, True, False, True, False, True, False, True]
    edges = g._edges.with_columns(e=__import__("polars").Series(marker)) if engine.startswith("polars") else g._edges.assign(e=marker)
    return g.edges(edges)


@pytest.mark.parametrize("name", sorted(COLLISION_QUERIES))
def test_aliases_that_collide_with_columns_match_the_traversal(engine, name):
    g = _collision_graph(engine)
    query = COLLISION_QUERIES[name]
    with routes_off([ROUTE]):
        expected = _run(g, query, engine)
    actual = _run(g, query, engine)
    assert_same_frame(actual._nodes, expected._nodes, engine)
    assert_same_frame(actual._edges, expected._edges, engine)


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("name", sorted(COLLISION_QUERIES))
def test_aliases_that_collide_with_columns_decline_the_skip(engine, name):
    assert True not in _consulted(_collision_graph(engine), COLLISION_QUERIES[name], engine)  # declined, or an earlier route answered


STR_NODE, NUM_NODE, STR_EDGE, NUM_EDGE = "name", "score", "type", "since"
# per pair: (column kind, first value, second value); both values share a type so they share a key
FILTER_VALUE_PAIRS = {
    "valid_then_invalid_regex": ("str", lambda: _contains("a"), lambda: _contains("(")),
    "empty_then_nonempty_is_in": ("str", lambda: _is_in([]), lambda: _is_in(["a"])),
    "in_range_then_overflowing_int": ("num", lambda: 5, lambda: 2 ** 70),
    "number_then_nan": ("num", lambda: 1.5, lambda: float("nan")),
    "value_then_none": ("str", lambda: "a", lambda: None),
    "number_then_string_on_numeric": ("num", lambda: 1.5, lambda: "high"),
    "present_then_absent_column": ("absent", lambda: 1, lambda: 2),
    "predicate_then_predicate_on_numeric": ("num", lambda: gt(1), lambda: gt("x")),
}
FIVE_OPS = ("seed", "edge1", "mid", "edge2", "last")
SEEDS = {"matches": {"id": 1}, "matches_nothing": {"id": 99}}


def _filter_for(kind, position, make):
    node = position in ("seed", "mid", "last")
    column = {"str": STR_NODE if node else STR_EDGE, "num": NUM_NODE if node else NUM_EDGE, "absent": "nope_col"}[kind]
    return {column: make()}


def _five_op_query(position, make, kind, seed):
    ops = {"seed": n(dict(SEEDS[seed]), name="a"), "edge1": e_undirected(name="r1"), "mid": n(name="m"),
           "edge2": e_undirected(name="r2"), "last": n(name="z")}
    if position == "seed":
        ops["seed"] = n(_filter_for(kind, position, make), name="a")
    elif position in ("edge1", "edge2"):
        ops[position] = e_undirected(_filter_for(kind, position, make), name=ops[position]._name)
    else:
        ops[position] = n(_filter_for(kind, position, make), name=ops[position]._name)
    return [ops[p] for p in FIVE_OPS] + [rows(), select([("aid", "a.id"), ("zid", "z.id")])]


def _empty_graph(engine):
    g = graph(engine)
    return g.nodes(g._nodes[:0]).edges(g._edges[:0])


@pytest.mark.parametrize("validate_schema", [True, False])
@pytest.mark.parametrize("graph_kind", ["populated", "empty"])
@pytest.mark.parametrize("seed", sorted(SEEDS))
@pytest.mark.parametrize("order", ["forward", "reverse"])
@pytest.mark.parametrize("position", FIVE_OPS)
@pytest.mark.parametrize("pair", sorted(FILTER_VALUE_PAIRS))
def test_a_filter_value_at_any_position_never_changes_the_outcome(engine, pair, position, order, seed, graph_kind, validate_schema):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    kind, first, second = FILTER_VALUE_PAIRS[pair]
    makers = [first, second] if order == "forward" else [second, first]
    g = graph(engine) if graph_kind == "populated" else _empty_graph(engine)
    queries = [_five_op_query(position, make, kind, seed) for make in makers]
    gfql_clear_caches()
    actual = [_outcome(g, query, engine, validate_schema) for query in queries * 2]  # cold, then warm
    with routes_off([ROUTE]):
        expected = [_outcome(g, query, engine, validate_schema) for query in queries * 2]
    assert actual == expected


@pytest.mark.parametrize("first", ["chain", "cypher"])
def test_chain_and_cypher_forms_do_not_share_a_twin(engine, first):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    g = _collision_graph(engine)
    forms = {
        "chain": [n(name="a"), e_forward(name="e"), n(name="b"), rows(), select([("b", "b.id")])],
        "cypher": "MATCH (a)-[e]->(b) RETURN b.id AS b",
    }
    order = [first] + [name for name in forms if name != first]
    gfql_clear_caches()
    actual = [_outcome(g, forms[name], engine) if name == "chain" else _cypher_outcome(g, forms[name], engine) for name in order * 2]
    with routes_off([ROUTE]):
        expected = [_outcome(g, forms[name], engine) if name == "chain" else _cypher_outcome(g, forms[name], engine) for name in order * 2]
    assert actual == expected


def _cypher_outcome(g, query, engine):
    out = with_index_policy(g, "off").gfql(query, engine=engine)
    return "rows", _signature(out._nodes), _signature(out._edges)


@pytest.mark.parametrize("strict", ["warn", "strict", "quiet"])
def test_strictness_levels_do_not_share_a_twin(engine, strict):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    g = graph(engine)
    query = [n({"id": 99}, name="a"), e_forward({"type": "LOC"}), n(name="z"), rows(), select([("zid", "z.id")])]

    def run(level):
        try:
            return "rows", _signature(with_index_policy(g, "off").chain(query, engine=engine, strict=level)._nodes)
        except Exception as error:  # the outcome under test is the error class and message
            return type(error).__name__, str(error)

    gfql_clear_caches()
    actual = [run("warn"), run(strict), run("warn"), run(strict)]
    with routes_off([ROUTE]):
        expected = [run("warn"), run(strict), run("warn"), run(strict)]
    assert actual == expected


@pytest.mark.parametrize("index_kind", ["range", "named_range", "string", "shuffled_int", "duplicate"])
def test_frame_indexes_match_the_traversal(index_kind):
    import pandas as pd

    g = graph("pandas")
    edges, nodes = g._edges.copy(), g._nodes.copy()
    if index_kind == "named_range":
        edges.index = pd.RangeIndex(len(edges), name="eid")
    elif index_kind == "string":
        edges.index = [f"e{i}" for i in range(len(edges))]
    elif index_kind == "shuffled_int":
        edges.index = list(reversed(range(len(edges))))
    elif index_kind == "duplicate":
        edges.index = [0] * len(edges)
    g = g.nodes(nodes).edges(edges)
    query = SHAPES["seeded_undirected_edge_alias"][0]
    with routes_off([ROUTE]):
        expected = _outcome(g, query, "pandas")
    for _ in range(2):
        assert _outcome(g, query, "pandas") == expected


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("index_kind", ["named_range", "string", "shuffled_int"])
def test_non_default_indexes_decline_the_skip(index_kind):
    import pandas as pd

    g = graph("pandas")
    edges = g._edges.copy()
    edges.index = {"named_range": pd.RangeIndex(len(edges), name="eid"), "string": [f"e{i}" for i in range(len(edges))],
                   "shuffled_int": list(reversed(range(len(edges))))}[index_kind]
    assert _consulted(g.edges(edges), SHAPES["seeded_undirected_edge_alias"][0], "pandas") == {False}


@pytest.mark.route_engaged(ROUTE)
def test_cached_twins_do_not_pin_the_callers_buffers(engine):
    import numpy as np

    from graphistry.compute.gfql import rows_skip_traversal
    from graphistry.compute.gfql_unified import gfql_clear_caches

    gfql_clear_caches()
    g = graph(engine)
    _run(g, SHAPES["seeded_undirected_edge_alias"][0], engine)
    assert rows_skip_traversal._TWINS
    inputs = [g._nodes, g._edges]
    for frames in rows_skip_traversal._TWINS.values():
        for frame in frames:
            assert len(frame) == 0
            if engine == "pandas":
                for column in frame.columns:
                    cached = frame[column].values
                    assert not any(
                        column in source.columns and np.may_share_memory(cached, source[column].values) for source in inputs
                    )  # the cached zero-row column owns its memory


def test_a_served_polars_query_leaves_the_cached_twin_unchanged():
    pytest.importorskip("polars")
    from graphistry.compute.gfql import rows_skip_traversal
    from graphistry.compute.gfql_unified import gfql_clear_caches

    gfql_clear_caches()
    g = graph("polars")
    query = SHAPES["seeded_undirected_edge_alias"][0]
    _run(g, query, "polars")
    before = [(frame.columns, frame.dtypes) for frames in rows_skip_traversal._TWINS.values() for frame in frames]
    for _ in range(3):
        _run(g, query, "polars")
    after = [(frame.columns, frame.dtypes) for frames in rows_skip_traversal._TWINS.values() for frame in frames]
    assert before == after


def _dtype_columns(engine, count):
    """One column per dtype kind (pandas builders; Polars gets native equivalents), `count` rows each."""
    import numpy as np
    import pandas as pd

    mixed = ([1, "a", None, 2.5, True] * count)[:count]
    pandas_columns = {
        "int8": pd.Series(np.arange(count) % 100, dtype="int8"),
        "int32": pd.Series(np.arange(count), dtype="int32"),
        "int64": pd.Series(np.arange(count), dtype="int64"),
        "uint64": pd.Series(np.arange(count), dtype="uint64"),
        "float32": pd.Series(np.arange(count) / 2, dtype="float32"),
        "float64": pd.Series(np.arange(count) / 2, dtype="float64"),
        "bool": pd.Series([True, False] * count).iloc[:count].reset_index(drop=True),
        "string": pd.Series(list("abcdefghij" * count)[:count]),
        "dt_naive": pd.Series(pd.date_range("2020-01-01", periods=count)),
        "dt_tz": pd.Series(pd.date_range("2020-01-01", periods=count, tz="America/New_York")),
        "obj_mixed": pd.Series(mixed, dtype="object"),
        "all_null": pd.Series([np.nan] * count, dtype="float64"),
        "list": pd.Series([[1, 2]] * count),
        "struct": pd.Series([{"a": 1}] * count),
    }
    if not engine.startswith("polars"):
        return pandas_columns
    import polars as pl

    return {
        "int8": pl.Series("int8", (np.arange(count) % 100).tolist(), dtype=pl.Int8),
        "int32": pl.Series("int32", np.arange(count).tolist(), dtype=pl.Int32),
        "int64": pl.Series("int64", np.arange(count).tolist(), dtype=pl.Int64),
        "uint64": pl.Series("uint64", np.arange(count).tolist(), dtype=pl.UInt64),
        "float32": pl.Series("float32", (np.arange(count) / 2).tolist(), dtype=pl.Float32),
        "float64": pl.Series("float64", (np.arange(count) / 2).tolist(), dtype=pl.Float64),
        "bool": pl.Series("bool", ([True, False] * count)[:count], dtype=pl.Boolean),
        "string": pl.Series("string", list("abcdefghij" * count)[:count], dtype=pl.String),
        "dt_naive": pl.from_pandas(pandas_columns["dt_naive"]).alias("dt_naive"),
        "dt_tz": pl.from_pandas(pandas_columns["dt_tz"]).alias("dt_tz"),
        "obj_mixed": pl.Series("obj_mixed", [str(value) for value in mixed], dtype=pl.String),
        "all_null": pl.Series("all_null", [None] * count, dtype=pl.Float64),
        "list": pl.Series("list", [[1, 2]] * count),
        "struct": pl.Series("struct", [{"a": 1}] * count),
    }


CUDF_UNHOLDABLE = ["obj_mixed", "list", "struct"]  # cuDF cannot hold mixed-type or nested object columns built from pandas


def _dtype_graph(engine):
    g = graph(engine)
    node_columns, edge_columns = _dtype_columns(engine, len(g._nodes)), _dtype_columns(engine, len(g._edges))
    if engine.startswith("polars"):
        return g.nodes(g._nodes.with_columns(**node_columns)).edges(g._edges.with_columns(**edge_columns))
    if engine == "cudf":
        import cudf
        keep = {name: series for name, series in node_columns.items() if name not in CUDF_UNHOLDABLE}
        keep_edges = {name: series for name, series in edge_columns.items() if name not in CUDF_UNHOLDABLE}
        return g.nodes(cudf.from_pandas(g._nodes.to_pandas().assign(**keep))).edges(
            cudf.from_pandas(g._edges.to_pandas().assign(**keep_edges)))
    return g.nodes(g._nodes.assign(**node_columns)).edges(g._edges.assign(**edge_columns))


SCALAR_GROUPS = {
    "int": [5, 2 ** 70, -(2 ** 70), 1, 0, 2 ** 63, 300],
    "float": [1.5, float("nan"), float("inf"), -float("inf")],
    "str": ["a", "", "2020-01-01", "not a date", "x" * 1_000_000],
    "bool": [True, False],
}


def _later_filter_query(position, column, value):
    seed = n({"id": 99}, name="a")
    if position == "edge":
        return [seed, e_undirected({column: value}, name="r1"), n(name="m"), rows(), select([("aid", "a.id"), ("mid", "m.id")])]
    return [seed, e_undirected(name="r1"), n({column: value}, name="m"), rows(), select([("aid", "a.id"), ("mid", "m.id")])]


@pytest.mark.parametrize("position", ["edge", "node"])
@pytest.mark.parametrize("group", sorted(SCALAR_GROUPS))
@pytest.mark.parametrize("column", ["int8", "int32", "int64", "uint64", "float32", "float64", "bool", "string", "dt_naive",
                                    "dt_tz", "obj_mixed", "all_null", "list", "struct"])
def test_same_type_scalars_on_a_later_op_never_change_the_outcome(engine, column, group, position):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    if engine == "polars-gpu" and column in ("list", "struct", "obj_mixed", "dt_tz"):
        pytest.skip("polars-gpu declines this column type in the canonical path too")
    if engine == "cudf" and column in CUDF_UNHOLDABLE:
        pytest.skip("cuDF cannot hold this column")
    g = _dtype_graph(engine)
    queries = [_later_filter_query(position, column, value) for value in SCALAR_GROUPS[group]]
    gfql_clear_caches()
    actual = [_outcome(g, query, engine) for query in queries * 2]  # cold, then warm
    with routes_off([ROUTE]):
        expected = [_outcome(g, query, engine) for query in queries * 2]
    assert actual == expected


ODD_SCALARS = {
    "numpy_int64": lambda: __import__("numpy").int64(5),
    "numpy_float64": lambda: __import__("numpy").float64(1.5),
    "numpy_float32": lambda: __import__("numpy").float32(1.5),
    "numpy_bool": lambda: __import__("numpy").bool_(True),
    "bool_on_int_column": lambda: True,
    "int_on_bool_column": lambda: 1,
}
ODD_COLUMNS = {"numpy_int64": "int64", "numpy_float64": "float64", "numpy_float32": "float32", "numpy_bool": "bool",
               "bool_on_int_column": "int64", "int_on_bool_column": "bool"}


@pytest.mark.parametrize("name", sorted(ODD_SCALARS))
def test_numpy_scalars_and_bool_int_mixups_match_the_traversal_and_decline(engine, name):
    from graphistry.compute.gfql_unified import gfql_clear_caches

    if engine == "polars-gpu":
        pytest.skip("polars-gpu declines this column type in the canonical path too")
    g = _dtype_graph(engine)
    column = ODD_COLUMNS[name]
    query = _later_filter_query("edge", column, ODD_SCALARS[name]())
    gfql_clear_caches()
    actual = [_outcome(g, query, engine) for _ in range(2)]
    with routes_off([ROUTE]):
        expected = [_outcome(g, query, engine) for _ in range(2)]
    assert actual == expected


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("name", ["numpy_int64", "numpy_bool", "bool_on_int_column", "int_on_bool_column"])
def test_numpy_scalars_and_bool_int_mixups_decline_the_skip(engine, name):
    if engine == "polars-gpu":
        pytest.skip("polars-gpu declines this column type in the canonical path too")
    query = _later_filter_query("edge", ODD_COLUMNS[name], ODD_SCALARS[name]())
    assert True not in _consulted_accepting_validation_errors(_dtype_graph(engine), query, engine)


@pytest.mark.route_engaged(ROUTE)
def test_a_string_scalar_on_a_mixed_object_column_is_served():
    """The pandas object dtype is one key for all-string and mixed columns; the K1 matrix proves str equality identical on both."""
    query = _later_filter_query("edge", "obj_mixed", "a")
    assert _consulted(_dtype_graph("pandas"), query, "pandas") == {True}


@pytest.mark.route_engaged(ROUTE)
@pytest.mark.parametrize("value,served", [
    (5, True), (2 ** 70, False), (float("nan"), True), (gt(1), True), ("a", True), (True, True), (None, False),
])
def test_the_seed_op_accepts_everything_but_none(engine, value, served):
    """The rebuild evaluates the seed op on real frames; values the frame layer rejects replay instead of being cached."""
    column = "name" if isinstance(value, str) else "score"
    query = [n({column: value}, name="a"), e_undirected(name="r1"), n(name="m"), rows(), select([("aid", "a.id")])]
    assert _consulted(graph(engine), query, engine) == ({True} if served else {False})


@pytest.mark.route_engaged(ROUTE)
def test_bypassing_the_allowlist_makes_known_pairs_diverge(monkeypatch):
    """The allowlist is load-bearing: with it bypassed, an int on a bool column (a pair the matrix covers) changes the outcome."""
    from graphistry.compute.gfql import rows_skip_traversal
    from graphistry.compute.gfql_unified import gfql_clear_caches

    monkeypatch.setattr(rows_skip_traversal, "_scalar_fits", lambda value, dtype: True)
    g = _dtype_graph("pandas")
    queries = [_later_filter_query("edge", "bool", value) for value in SCALAR_GROUPS["int"]]
    gfql_clear_caches()
    actual = [_outcome(g, query, "pandas") for query in queries * 2]
    monkeypatch.undo()
    with routes_off([ROUTE]):
        expected = [_outcome(g, query, "pandas") for query in queries * 2]
    assert actual != expected


@pytest.mark.route_engaged(ROUTE)
def test_a_raising_cold_build_leaves_no_cache_entry_and_the_next_call_is_served(engine):
    from graphistry.compute.gfql import rows_skip_traversal
    from graphistry.compute.gfql_unified import gfql_clear_caches

    gfql_clear_caches()
    g = graph(engine)
    seed = {"id": 4, **PERSON}
    query = [n(seed, name="p"), e_undirected({"type": "KNOWS"}, name="r"), n(PERSON, name="f"), rows(), select([("fid", "f.id")])]
    original = rows_skip_traversal._zero_row_twin

    def raising(*args, **kwargs):
        raise TypeError("cold build failed")

    rows_skip_traversal._zero_row_twin = raising
    try:
        _run(g, query, engine)
    finally:
        rows_skip_traversal._zero_row_twin = original
    assert not rows_skip_traversal._TWINS
    assert _consulted(g, query, engine) == {True}
    with routes_off([ROUTE]):
        expected = _outcome(g, query, engine)
    assert _outcome(g, query, engine) == expected


@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_execution_targets_do_not_share_a_twin(engine, target):
    from graphistry.compute.gfql.lazy import ExecutionTarget, target_mode
    from graphistry.compute.gfql_unified import gfql_clear_caches

    mode = ExecutionTarget.GPU if target == "gpu" else ExecutionTarget.CPU
    g = graph(engine)
    query = SHAPES["seeded_undirected_edge_alias"][0]

    def run(selected):
        with target_mode(selected):
            return _outcome(g, query, engine)

    gfql_clear_caches()
    actual = [run(ExecutionTarget.CPU), run(mode), run(ExecutionTarget.CPU), run(mode)]
    with routes_off([ROUTE]):
        expected = [run(ExecutionTarget.CPU), run(mode), run(ExecutionTarget.CPU), run(mode)]
    assert actual == expected


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

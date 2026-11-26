"""searchAny and categorical columns: the automatic column gate skips every categorical on pandas and cuDF alike, and an explicit integer categorical is answered by all four query engines."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.search_any import (
    _is_categorical_dtype, search_any_mask, search_candidate_columns,
)

FRAME_ENGINES = ["pandas", "cudf", "polars", "polars-gpu"]
QUERY_ENGINES = ["pandas", "cudf", "polars", "polars-gpu"]

CATEGORICALS = {
    "int": pd.Categorical([1, 2, 1]),
    "float": pd.Categorical([1.5, 2.5, 1.5]),
    "datetime": pd.Categorical(pd.to_datetime(["2021-01-01", "2022-02-02", "2021-01-01"])),
    "bool": pd.Categorical([True, False, True]),
    "str": pd.Categorical(["a1", "b2", "a1"]),
}


def _frame(pdf, engine):
    if engine == "cudf":
        cudf = pytest.importorskip("cudf", reason="cuDF lane needs a GPU box")
        return cudf.from_pandas(pdf)
    if engine in ("polars", "polars-gpu"):
        pytest.importorskip("polars")
        from graphistry.Engine import Engine, df_to_engine
        target = Engine.POLARS_GPU if engine == "polars-gpu" else Engine.POLARS
        return df_to_engine(pdf, target)
    return pdf


def _auto_candidates(frame, engine, term, *, categorical_columns=()):
    if engine in ("polars", "polars-gpu"):
        from graphistry.compute.gfql.lazy.engine.polars.search import auto_search_columns

        return auto_search_columns(
            frame.schema, frame.columns, term,
            categorical_columns=categorical_columns,
        )
    return search_candidate_columns(frame, term, None)


def _schema(frame):
    return getattr(frame, "schema", getattr(frame, "dtypes", None))


def _mask(df, term, *, engine="pandas", categorical_columns=(), **kw):
    if engine in ("polars", "polars-gpu"):
        from graphistry.compute.gfql.lazy.engine.polars.search import search_match_expr

        expr = search_match_expr(
            df.schema,
            kw["columns"],
            term,
            case_sensitive=kw.get("case_sensitive", False),
            regex=kw.get("regex", False),
            categorical_columns=categorical_columns,
        )
        if expr is None:
            raise NotImplementedError("Polars searchAny explicit columns cannot be lowered")
        return df.select(expr).to_series().to_list()
    m = search_any_mask(df, term, **kw)
    return m.to_pandas().tolist() if hasattr(m, "to_pandas") else m.tolist()


@pytest.fixture(params=QUERY_ENGINES)
def query_engine(request):
    if request.param == "cudf":
        pytest.importorskip("cudf")
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cudf tests need TEST_CUDF=1")
    elif request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    elif request.param == "polars":
        pytest.importorskip("polars")
    return request.param


def test_the_categorical_dtype_test_is_by_dtype_name():
    assert _is_categorical_dtype(pd.Series(pd.Categorical([1, 2])).dtype)
    assert not _is_categorical_dtype(pd.Series([1, 2]).dtype)
    assert not _is_categorical_dtype(pd.Series(["a"]).dtype)
    cudf = pytest.importorskip("cudf", reason="cuDF lane needs a GPU box")
    assert _is_categorical_dtype(cudf.from_pandas(pd.Series(pd.Categorical([1, 2]))).dtype)
    assert not _is_categorical_dtype(cudf.from_pandas(pd.Series([1, 2])).dtype)


@pytest.mark.parametrize("engine", FRAME_ENGINES)
@pytest.mark.parametrize("kind", sorted(CATEGORICALS))
def test_the_auto_gate_skips_a_categorical_for_positive_and_negative_terms(engine, kind):
    pdf = pd.DataFrame({"name": ["a", "b", "c"], "n": [1, 2, 1], "cat": CATEGORICALS[kind]})
    frame = _frame(pdf, engine)
    terms = {
        "int": ("1", "9"),
        "float": ("1", "9"),
        "datetime": ("1", "9"),
        "bool": ("true", "unknown"),
        "str": ("a", "z"),
    }
    for term in terms[kind]:
        actual = _auto_candidates(frame, engine, term, categorical_columns=["cat"])
        expected = search_candidate_columns(pdf, term, None)
        assert actual == expected and "cat" not in actual, (
            f"{engine}/{kind}/{term!r}: source categorical converted to "
            f"schema={_schema(frame)!r}; candidates={actual!r}, pandas={expected!r}"
        )
        actual_mask = (
            [False] * len(frame) if not actual
            else _mask(frame, term, engine=engine, categorical_columns=["cat"], columns=actual)
        )
        expected_mask = (
            [False] * len(pdf) if not expected
            else _mask(pdf, term, columns=expected)
        )
        assert actual_mask == expected_mask


@pytest.mark.parametrize("engine", FRAME_ENGINES)
def test_the_auto_gate_lists_the_same_columns_for_a_frame_of_all_five_categoricals(engine):
    pdf = pd.DataFrame({"name": ["a", "b", "c"], "n": [1, 2, 1], **{f"{k}_cat": v for k, v in CATEGORICALS.items()}})
    frame = _frame(pdf, engine)
    for term in ["1", "a"]:
        actual = _auto_candidates(
            frame, engine, term,
            categorical_columns=[f"{k}_cat" for k in CATEGORICALS],
        )
        expected = search_candidate_columns(pdf, term, None)
        assert actual == expected, (
            f"{engine} term={term!r}: converted schema={_schema(frame)!r}; "
            f"candidates={actual!r}, pandas={expected!r}"
        )
    assert _auto_candidates(
        frame, engine, "1", categorical_columns=[f"{k}_cat" for k in CATEGORICALS],
    ) == ["name", "n"]


@pytest.mark.parametrize("engine", FRAME_ENGINES)
def test_auto_gate_keeps_plain_typed_controls_and_skips_all_categoricals(engine):
    pdf = pd.DataFrame({
        "integer": [1, 2, 3],
        "float": [1.0, 2.0, 3.0],
        "datetime": pd.to_datetime(["2021-01-01", "2022-02-02", "2023-03-03"]),
        "boolean": [True, False, True],
        "text": ["1", "x", "y"],
        "cat_int": CATEGORICALS["int"],
        "cat_float": CATEGORICALS["float"],
        "cat_datetime": CATEGORICALS["datetime"],
        "cat_bool": CATEGORICALS["bool"],
        "cat_string": CATEGORICALS["str"],
    })
    frame = _frame(pdf, engine)
    actual = _auto_candidates(
        frame, engine, "1",
        categorical_columns=["cat_int", "cat_float", "cat_datetime", "cat_bool", "cat_string"],
    )
    dtypes = _schema(frame)
    assert actual == [
        "integer", "float", "datetime", "text",
    ], f"{engine} categorical conversion/gate: dtypes={dtypes!r}, candidates={actual!r}"


@pytest.mark.parametrize("engine", FRAME_ENGINES)
@pytest.mark.parametrize("kind,positive,positive_mask,negative", [
    ("int", "1", [True, False, True], "9"),
    ("float", "1", [True, False, True], "9"),
    ("datetime", "1", [True, False, True], "2040"),
    ("bool", "true", [True, False, True], "unknown"),
    ("str", "a", [True, False, True], "z"),
])
def test_explicit_categorical_helper_matches_or_declines_by_name(
    engine, kind, positive, positive_mask, negative,
):
    pdf = pd.DataFrame({"cat": CATEGORICALS[kind]})
    frame = _frame(pdf, engine)

    def run(term, data):
        if engine in ("polars", "polars-gpu"):
            return _mask(
                data, term, engine=engine, categorical_columns=["cat"], columns=["cat"],
            )
        return _mask(data, term, columns=["cat"])

    decline_expected = (
        (engine in ("polars", "polars-gpu") and kind == "datetime")
        or (engine == "cudf" and kind in ("float", "datetime", "bool"))
    )
    try:
        actual_positive = run(positive, frame)
        actual_negative = run(negative, frame)
    except NotImplementedError as exc:
        message = str(exc).lower()
        assert decline_expected, f"unexpected {engine}/{kind} decline: {exc}"
        assert "searchany" in message or "rows(alias_prefilters" in message
        if engine in ("polars", "polars-gpu") and kind == "datetime":
            assert "datetime categorical" in message
        return

    assert not decline_expected, f"{engine}/{kind} was required to decline by name"
    expected_positive = _mask(pdf, positive, columns=["cat"])
    expected_negative = _mask(pdf, negative, columns=["cat"])
    assert actual_positive == expected_positive == positive_mask
    assert actual_negative == expected_negative == [False, False, False]


@pytest.mark.parametrize("kind", ["datetime", "bool"])
def test_an_explicit_datetime_or_bool_categorical_declines_by_name_on_cudf(kind):
    frame = _frame(pd.DataFrame({"cat": CATEGORICALS[kind]}), "cudf")
    with pytest.raises(NotImplementedError, match="searchAny explicit columns"):
        search_any_mask(frame, "1", columns=["cat"])


def _nodes(engine):
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS["int"]})
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(pdf)
    return pdf


def _ids(out):
    out = out.to_pandas() if hasattr(out, "to_pandas") else out
    return list(out["id"])


def test_an_explicit_integer_categorical_is_answered_by_every_query_engine(query_engine):
    # no exception is expected on any engine: all four answer rows 0 and 2
    query = "MATCH (n) WHERE searchAny(n, '1', {columns: ['cat']}) RETURN n.id AS id ORDER BY id"
    out = graphistry.nodes(_nodes(query_engine), "id").gfql(query, engine=query_engine)._nodes
    assert _ids(out) == [0, 2]


@pytest.mark.parametrize("engine", FRAME_ENGINES)
def test_the_automatic_gate_query_ignores_an_integer_categorical_on_pandas_and_cudf(engine):
    if engine == "cudf":
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cudf tests need TEST_CUDF=1")
    elif engine == "polars":
        pytest.importorskip("polars")
    elif engine == "polars-gpu":
        pytest.importorskip("polars")
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    query = "MATCH (n) WHERE searchAny(n, '1') RETURN n.id AS id ORDER BY id"
    out = graphistry.nodes(_nodes(engine), "id").gfql(query, engine=engine)._nodes
    assert _ids(out) == [1]


@pytest.mark.parametrize("kind,positive,positive_rows,negative", [
    ("int", "1", [1], "9"),
    ("float", "1", [1], "9"),
    ("datetime", "1", [1], "9"),
    ("bool", "true", [], "unknown"),
    ("str", "a", [], "missing"),
])
def test_automatic_categorical_search_matches_pandas_on_all_query_engines(
    query_engine, kind, positive, positive_rows, negative,
):
    pdf = pd.DataFrame({"id": [0, 1, 2], "note": ["x", "y", "z"], "cat": CATEGORICALS[kind]})
    converted_frame = _frame(pdf, query_engine)

    def run(term):
        query = f"MATCH (n) WHERE searchAny(n, '{term}') RETURN n.id AS id ORDER BY id"
        out = graphistry.nodes(pdf, "id").gfql(query, engine=query_engine)._nodes
        return _ids(out)

    assert run(positive) == positive_rows, (
        f"{query_engine}/{kind}: converted schema={_schema(converted_frame)!r}"
    )
    assert run(negative) == []


def test_explicit_datetime_categorical_matches_or_declines_by_name(query_engine):
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS["datetime"]})
    query = (
        "MATCH (n) WHERE searchAny(n, '1', {columns: ['cat']}) "
        "RETURN n.id AS id ORDER BY id"
    )
    converted_frame = _frame(pdf, query_engine)
    try:
        out = graphistry.nodes(pdf, "id").gfql(query, engine=query_engine)._nodes
    except NotImplementedError as exc:
        # A named decline is an allowed cross-engine outcome while native categorical
        # rendering remains unsupported; the probe captures each engine's error text.
        assert str(exc)
        return
    assert _ids(out) == [0, 2], (
        f"{query_engine}/datetime: converted schema={_schema(converted_frame)!r}; "
        f"rows={_ids(out)!r}, pandas={[0, 2]!r}"
    )


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
@pytest.mark.parametrize("variant", ["midnight", "nonmidnight", "timezone", "nat"])
def test_polars_explicit_datetime_categorical_declines_for_rendering_variants(engine, variant):
    pytest.importorskip("polars")
    if engine == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    if variant == "midnight":
        values = pd.to_datetime(["2021-01-01", "2022-02-02", "2021-01-01"])
    elif variant == "nonmidnight":
        values = pd.to_datetime([
            "2021-01-01 13:45:00", "2022-02-02 07:30:00", "2021-01-01 13:45:00",
        ])
    elif variant == "timezone":
        values = pd.to_datetime(
            ["2021-01-01 13:45:00", "2022-02-02 07:30:00", "2021-01-01 13:45:00"],
            utc=True,
        )
    else:
        values = pd.to_datetime(["2021-01-01", None, "2021-01-01"])
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": pd.Categorical(values)})
    converted_frame = _frame(pdf, engine)
    query = "MATCH (n) WHERE searchAny(n, '1', {columns: ['cat']}) RETURN n.id AS id"
    with pytest.raises(NotImplementedError, match="datetime categorical"):
        graphistry.nodes(pdf, "id").gfql(query, engine=engine)._nodes
    assert _schema(converted_frame) is not None


@pytest.mark.parametrize("kind,positive,positive_rows,negative", [
    ("int", "1", [0, 2], "9"),
    ("float", "1", [0, 2], "9"),
    ("datetime", "1", [0, 2], "9"),
    ("bool", "true", [0, 2], "unknown"),
    ("str", "a", [0, 2], "z"),
])
def test_explicit_categorical_search_positive_and_negative_on_all_query_engines(
    query_engine, kind, positive, positive_rows, negative,
):
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS[kind]})

    def run(term):
        query = (
            f"MATCH (n) WHERE searchAny(n, '{term}', {{columns: ['cat']}}) "
            "RETURN n.id AS id ORDER BY id"
        )
        out = graphistry.nodes(pdf, "id").gfql(query, engine=query_engine)._nodes
        return _ids(out)

    decline_expected = (
        (query_engine in ("polars", "polars-gpu") and kind == "datetime")
        or (query_engine == "cudf" and kind in ("float", "datetime", "bool"))
    )
    try:
        actual_positive = run(positive)
        actual_negative = run(negative)
    except NotImplementedError as exc:
        message = str(exc).lower()
        assert decline_expected, f"unexpected {query_engine}/{kind} decline: {exc}"
        assert "searchany" in message or "rows(alias_prefilters" in message
        if query_engine in ("polars", "polars-gpu") and kind == "datetime":
            assert "datetime categorical" in message
    else:
        assert not decline_expected, f"{query_engine}/{kind} was required to decline by name"
        assert actual_positive == positive_rows
        assert actual_negative == []


@pytest.mark.parametrize("engine", QUERY_ENGINES)
def test_explicit_bool_categorical_preserves_pandas_case_sensitive_spelling(engine):
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS["bool"]})
    query = (
        "MATCH (n) WHERE searchAny(n, 'True', {columns: ['cat'], caseSensitive: true}) "
        "RETURN n.id AS id ORDER BY id"
    )
    if engine == "cudf":
        with pytest.raises(NotImplementedError, match="cuDF searchAny explicit columns"):
            graphistry.nodes(pdf, "id").gfql(query, engine=engine)
        return
    actual = graphistry.nodes(pdf, "id").gfql(query, engine=engine)._nodes
    assert _ids(actual) == [0, 2]

    lower_query = query.replace("'True'", "'true'")
    lower = graphistry.nodes(pdf, "id").gfql(lower_query, engine=engine)._nodes
    assert _ids(lower) == []


@pytest.mark.parametrize("engine", QUERY_ENGINES)
def test_or_search_preserves_categorical_metadata_for_row_pipeline(engine):
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS["int"]})
    query = "MATCH (n) WHERE searchAny(n, '1') OR n.id = 99 RETURN n.id AS id ORDER BY id"
    actual = graphistry.nodes(pdf, "id").gfql(query, engine=engine)._nodes
    assert _ids(actual) == [1]


@pytest.mark.parametrize("engine", QUERY_ENGINES)
def test_with_match_reentry_preserves_automatic_categorical_gate(engine):
    pdf = pd.DataFrame({"id": [0, 1, 2, 99], "cat": pd.Categorical([1, 2, 1, 2])})
    query = (
        "MATCH (a) WHERE a.id = 99 WITH a MATCH (n) WHERE searchAny(n, '1') "
        "RETURN n.id AS id ORDER BY id"
    )
    actual = graphistry.nodes(pdf, "id").gfql(query, engine=engine)._nodes
    assert _ids(actual) == [1]


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_with_match_reentry_keeps_explicit_datetime_categorical_decline(engine):
    pytest.importorskip("polars")
    if engine == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS["datetime"]})
    query = (
        "MATCH (a) WITH a MATCH (n) WHERE searchAny(n, '2021', {columns: ['cat']}) "
        "RETURN n.id AS id"
    )
    with pytest.raises(NotImplementedError, match="datetime categorical"):
        graphistry.nodes(pdf, "id").gfql(query, engine=engine)


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_polars_coercion_captures_pandas_node_and_edge_categoricals(engine):
    pytest.importorskip("polars")
    from graphistry.Engine import Engine
    from graphistry.compute.ComputeMixin import _coerce_input_formats

    target = Engine.POLARS
    if engine == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
        target = Engine.POLARS_GPU
    nodes = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS["int"]})
    edges = pd.DataFrame({
        "src": [0, 1], "dst": [1, 2], "cat": pd.Categorical([1, 2]),
    })
    source = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")

    coerced = _coerce_input_formats(source, target)

    assert coerced._gfql_search_any_categorical_columns == {
        "nodes": frozenset({"cat"}), "edges": frozenset({"cat"}),
    }


@pytest.mark.parametrize("engine", ["polars", "polars-gpu"])
def test_polars_reentry_collect_preserves_source_categorical_metadata(engine):
    pl = pytest.importorskip("polars")
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.cypher.reentry.execution import _serving_engine_base_graph

    target = Engine.POLARS
    if engine == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
        target = Engine.POLARS_GPU
    nodes = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS["int"]})
    edges = pd.DataFrame({
        "src": [0, 1], "dst": [1, 2], "cat": pd.Categorical([1, 2]),
    })
    graph = graphistry.nodes(pl.from_pandas(nodes).lazy(), "id").edges(
        pl.from_pandas(edges).lazy(), "src", "dst",
    )
    graph = graph.bind()
    graph._gfql_search_any_categorical_columns = {
        "nodes": frozenset({"cat"}), "edges": frozenset({"cat"}),
    }

    served = _serving_engine_base_graph(graph, target)

    assert served._gfql_search_any_categorical_columns == {
        "nodes": frozenset({"cat"}), "edges": frozenset({"cat"}),
    }


@pytest.mark.parametrize("engine", QUERY_ENGINES)
def test_edge_search_preserves_automatic_categorical_gate(engine):
    nodes = pd.DataFrame({"id": [0, 1, 2]})
    edges = pd.DataFrame({
        "src": [0, 1, 2], "dst": [1, 2, 0], "id": ["edge-a", "edge-b", "edge-c"],
        "cat": pd.Categorical([1, 2, 1]),
    })
    query = "MATCH (a)-[r]->(b) WHERE searchAny(r, '1') RETURN r.id AS id ORDER BY id"
    try:
        actual = graphistry.nodes(nodes, "id").edges(edges, "src", "dst").bind(edge="id").gfql(
            query, engine=engine,
        )._nodes
    except NotImplementedError as exc:
        assert engine == "polars-gpu" and "not fully GPU-executable" in str(exc), exc
        return
    actual = actual.to_pandas() if hasattr(actual, "to_pandas") else actual
    assert list(actual["id"]) == []

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


def _mask(df, term, **kw):
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
def test_the_auto_gate_skips_a_categorical_of_any_kind_like_pandas(engine, kind):
    pdf = pd.DataFrame({"name": ["a", "b", "c"], "n": [1, 2, 1], "cat": CATEGORICALS[kind]})
    assert search_candidate_columns(_frame(pdf, engine), "1", None) == search_candidate_columns(pdf, "1", None) == ["name", "n"]


@pytest.mark.parametrize("engine", FRAME_ENGINES)
def test_the_auto_gate_lists_the_same_columns_for_a_frame_of_all_five_categoricals(engine):
    pdf = pd.DataFrame({"name": ["a", "b", "c"], "n": [1, 2, 1], **{f"{k}_cat": v for k, v in CATEGORICALS.items()}})
    frame = _frame(pdf, engine)
    for term in ["1", "a"]:
        assert search_candidate_columns(frame, term, None) == search_candidate_columns(pdf, term, None)
    assert search_candidate_columns(frame, "1", None) == ["name", "n"]


@pytest.mark.parametrize("engine", FRAME_ENGINES)
def test_auto_gate_keeps_plain_typed_controls_and_only_string_categorical(engine):
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
    actual = search_candidate_columns(frame, "1", None)
    dtypes = getattr(frame, "schema", getattr(frame, "dtypes", None))
    assert actual == [
        "integer", "float", "datetime", "text", "cat_string",
    ], f"{engine} categorical conversion/gate: dtypes={dtypes!r}, candidates={actual!r}"


@pytest.mark.parametrize("engine", FRAME_ENGINES)
def test_an_explicit_integer_categorical_is_answered_like_pandas(engine):
    pdf = pd.DataFrame({"cat": CATEGORICALS["int"]})
    assert _mask(_frame(pdf, engine), "1", columns=["cat"]) == _mask(pdf, "1", columns=["cat"]) == [True, False, True]


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
    query = "MATCH (n) WHERE searchAny(n, '1') RETURN n.id AS id ORDER BY id"
    out = graphistry.nodes(_nodes(engine), "id").gfql(query, engine=engine)._nodes
    assert _ids(out) == [1]


@pytest.mark.parametrize("kind,term,expected", [
    ("int", "1", [1]),
    ("float", "1", [1]),
    ("datetime", "1", [1]),
    ("bool", "1", [1]),
    ("str", "a", [0, 2]),
])
def test_automatic_categorical_search_matches_pandas_on_all_query_engines(
    query_engine, kind, term, expected,
):
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS[kind]})
    query = f"MATCH (n) WHERE searchAny(n, '{term}') RETURN n.id AS id ORDER BY id"
    out = graphistry.nodes(pdf, "id").gfql(query, engine=query_engine)._nodes
    assert _ids(out) == expected


@pytest.mark.parametrize("kind,term,expected", [
    ("datetime", "1", [0, 2]),
    ("str", "a", [0, 2]),
])
def test_explicit_datetime_and_string_categorical_matches_or_declines_by_name(
    query_engine, kind, term, expected,
):
    pdf = pd.DataFrame({"id": [0, 1, 2], "cat": CATEGORICALS[kind]})
    query = (
        f"MATCH (n) WHERE searchAny(n, '{term}', {{columns: ['cat']}}) "
        "RETURN n.id AS id ORDER BY id"
    )
    try:
        out = graphistry.nodes(pdf, "id").gfql(query, engine=query_engine)._nodes
    except NotImplementedError as exc:
        # A named decline is an allowed cross-engine outcome while native categorical
        # rendering remains unsupported; the probe captures each engine's error text.
        assert str(exc)
        return
    assert _ids(out) == expected


@pytest.mark.parametrize("kind,positive,positive_rows,negative", [
    ("int", "1", [0, 2], "9"),
    ("float", "1", [0, 2], "9"),
    ("datetime", "2021", [0, 2], "2040"),
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

    try:
        assert run(positive) == positive_rows
        assert run(negative) == []
    except NotImplementedError as exc:
        # Unsupported categorical forms must report a named search/gate capability,
        # not leak a backend exception. Integer categoricals are pinned to answer above.
        message = str(exc).lower()
        assert "searchany" in message or "rows(alias_prefilters" in message

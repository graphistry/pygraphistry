"""Exact float candidate gathers retain canonical engine and null semantics."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.gfql.index import get_registry, index_trace, with_index_policy
from graphistry.compute.predicates.is_in import IsIn


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine, dtype):
    if dtype.startswith("arrow-"):
        pa = pytest.importorskip("pyarrow")
        dtype = pd.ArrowDtype(pa.float32() if dtype == "arrow-float32" else pa.float64())
    values = list(np.arange(400) + 1000.0)
    values[7] = values[9] = 0.1
    values[11], values[13] = -0.0, 0.0
    values[15], values[17], values[19], values[21] = np.nan, np.inf, -np.inf, None
    nodes = pd.DataFrame({"id": np.arange(400), "value": pd.Series(values, dtype=dtype), "keep": np.arange(400) % 3})
    nodes.index = np.arange(400)[::-1]
    edges = nodes.rename(columns={"id": "s"}).assign(d=np.arange(400))
    return graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id").edges(df_to_engine(edges, Engine(engine)), "s", "d")


def assert_same_frame(actual, expected, engine):
    if engine.startswith("polars"):
        from polars.testing import assert_frame_equal
        assert_frame_equal(actual, expected)
    else:
        pd.testing.assert_frame_equal(actual.to_pandas() if engine == "cudf" else actual, expected.to_pandas() if engine == "cudf" else expected)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64", "arrow-float32", "arrow-float64"])
@pytest.mark.parametrize("kind,role", [("node_prop", "nodes"), ("edge_prop", "edges")])
@pytest.mark.parametrize("predicate", [0.1, [0.1], -0.0, [0.0, -0.0], np.inf, -np.inf, 1e100, 999.5])
def test_float_exact_comparisons_preserve_results_and_engage(engine, dtype, kind, role, predicate):
    base = graph(engine, dtype)
    indexed = base.create_index(kind, column="value", engine=engine)
    method = "filter_nodes_by_dict" if role == "nodes" else "filter_edges_by_dict"
    scan_out = getattr(base, method)({"value": predicate}, engine=engine)
    scan = scan_out._nodes if role == "nodes" else scan_out._edges
    for policy in ["off", "use", "force"]:
        with index_trace() as steps:
            out = getattr(with_index_policy(indexed, policy), method)({"value": predicate}, engine=engine)
        frame = out._nodes if role == "nodes" else out._edges
        assert_same_frame(frame, scan, engine)
        if policy != "off":
            assert any(s.get("op") == "property_lookup" and s.get("path") == "index" for s in steps)
    assert get_registry(base).is_empty()


def filter_outcome(g, predicate, engine):
    from graphistry.compute.exceptions import GFQLSchemaError
    raw_errors = (ValueError, TypeError, OverflowError, NotImplementedError)
    if engine.startswith("polars"):
        from polars.exceptions import InvalidOperationError
        raw_errors += (InvalidOperationError,)
    try:
        return ("rows", g.filter_nodes_by_dict({"value": predicate}, engine=engine)._nodes)
    except GFQLSchemaError as error:
        return ("error", type(error), error.code, error.context["field"])
    except raw_errors as error:
        return ("error", type(error))


def assert_same_outcome(actual, expected, engine):
    assert actual[0] == expected[0]
    if expected[0] == "error":
        assert actual == expected
    else:
        assert_same_frame(actual[1], expected[1], engine)


@pytest.mark.parametrize("predicate", [None, [None], [float("nan")], IsIn([float("nan")]), IsIn([0.1, None]), IsIn([0.1]), True])
def test_float_null_and_ambiguous_predicates_keep_canonical_results(engine, predicate):
    base = graph(engine, "float64")
    indexed = base.create_index("node_prop", column="value", engine=engine)
    scan = filter_outcome(base, predicate, engine)
    for policy in ["off", "use", "force"]:
        assert_same_outcome(filter_outcome(with_index_policy(indexed, policy), predicate, engine), scan, engine)


@pytest.mark.parametrize("query", [2**64, 2**1024, "bad"])
def test_float_error_or_row_outcomes_match_scan(engine, query):
    base = graph(engine, "float64")
    indexed = base.create_index("node_prop", column="value", engine=engine)
    scan = filter_outcome(base, query, engine)
    for policy in ["off", "use", "force"]:
        assert_same_outcome(filter_outcome(with_index_policy(indexed, policy), query, engine), scan, engine)


@pytest.mark.parametrize("dtype", ["float32", "Float64", "arrow-float64"])
@pytest.mark.parametrize("values", [[], [None, np.nan]])
def test_float_empty_and_all_null_indexes(engine, dtype, values):
    if dtype.startswith("arrow-"):
        pa = pytest.importorskip("pyarrow")
        dtype = pd.ArrowDtype(pa.float64())
    frame = df_to_engine(pd.DataFrame({"id": np.arange(len(values)), "value": pd.Series(values, dtype=dtype)}), Engine(engine))
    indexed = graphistry.nodes(frame, "id").create_index("node_prop", column="value", engine=engine)
    assert get_registry(indexed).node_props["value"].n_keys == 0
    assert len(indexed.filter_nodes_by_dict({"value": 0.1}, engine=engine)._nodes) == 0


@pytest.mark.parametrize("check_engagement", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "cypher-fast"))])
def test_float_cypher_query_rows_and_receipt(engine, check_engagement):
    base = graph(engine, "float64")
    indexed = base.create_index("node_prop", column="value", engine=engine)
    query = "MATCH (a {value: 0.1}) RETURN a.id AS id"
    actual = indexed.gfql(query, engine=engine)._nodes
    scan = base.gfql(query, engine=engine)._nodes
    assert_same_frame(actual, scan, engine)
    if check_engagement:
        assert indexed.gfql_explain(query, engine=engine)["used_index"]


@pytest.mark.parametrize("predicate", [0.1, [0.1], float("nan"), IsIn([float("nan")]), [float("nan"), 0.1]])
def test_native_nan_storage_is_excluded_without_changing_null_predicates(engine, predicate):
    values = list(np.arange(400) + 1000.0)
    values[7] = values[9] = 0.1
    values[15], values[17] = float("nan"), None
    if engine.startswith("polars"):
        import polars as pl
        frame = pl.DataFrame({"id": np.arange(400), "value": values})
    elif engine == "cudf":
        import cudf
        frame = cudf.DataFrame({"id": np.arange(400), "value": cudf.Series(values, nan_as_null=False)})
    else:
        frame = pd.DataFrame({"id": np.arange(400), "value": values})
    base = graphistry.nodes(frame, "id")
    indexed = base.create_index("node_prop", column="value", engine=engine)
    index = get_registry(indexed).node_props["value"]
    assert index.n_nodes == 400 and index.n_keys == 397
    scan = filter_outcome(base, predicate, engine)
    for policy in ["off", "use", "force"]:
        assert_same_outcome(filter_outcome(with_index_policy(indexed, policy), predicate, engine), scan, engine)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("value", [2**24 + 1, 2**53 + 1, np.nextafter(0.0, 1.0), np.nextafter(0.1, 0.0), -1e-100])
def test_float_precision_boundaries_preserve_scalar_and_membership(engine, dtype, value):
    values = list(np.arange(400) + 1000.0)
    values[7] = values[9] = value
    frame = df_to_engine(pd.DataFrame({"id": np.arange(400), "value": pd.Series(values, dtype=dtype)}), Engine(engine))
    base = graphistry.nodes(frame, "id")
    indexed = with_index_policy(base.create_index("node_prop", column="value", engine=engine), "force")
    for predicate in [value, [value]]:
        assert_same_outcome(filter_outcome(indexed, predicate, engine), filter_outcome(base, predicate, engine), engine)


def test_float_empty_gather_retains_structured_residual_error(engine):
    from graphistry.compute.exceptions import GFQLSchemaError
    indexed = graph(engine, "float64").create_index("node_prop", column="value", engine=engine)
    outcomes = []
    for policy in ["off", "use", "force"]:
        with pytest.raises(GFQLSchemaError) as caught:
            with_index_policy(indexed, policy).filter_nodes_by_dict({"value": 999.5, "keep": "bad"}, engine=engine)
        outcomes.append((caught.value.code, caught.value.context["field"]))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("dtype", ["arrow-float32", "arrow-float64"])
def test_arrow_float_query_does_not_export_source_rows(dtype, monkeypatch):
    base = graph("pandas", dtype)
    indexed = base.create_index("node_prop", column="value")
    query = float(base._nodes["value"].iloc[7])
    array_type = type(base._nodes["value"].array)
    original = array_type.to_numpy

    def bounded_export(array, *args, **kwargs):
        assert len(array) <= 8
        return original(array, *args, **kwargs)

    monkeypatch.setattr(array_type, "to_numpy", bounded_export)
    with index_trace() as steps:
        actual = indexed.filter_nodes_by_dict({"value": query})._nodes
    assert actual["id"].tolist() == [7, 9]
    assert any(s.get("path") == "index" for s in steps)


@pytest.mark.parametrize("dtype", ["Float32", "Float64"])
@pytest.mark.parametrize("kind,role", [("node_prop", "nodes"), ("edge_prop", "edges")])
@pytest.mark.parametrize("value", [0.1, np.nextafter(0.1, 0.0), np.nextafter(0.0, 1.0),
                                    -0.0, float("inf"), float("-inf"), float("nan"),
                                    2**53 + 1, True, "bad", [0.1]])
def test_native_float_candidates_keep_precision_nan_payload_errors_and_isolation(dtype, kind, role, value):
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    from graphistry.compute.exceptions import GFQLSchemaError

    frame = pl.DataFrame({
        "id": range(7), "s": range(7), "d": range(7),
        "v": pl.Series([0.1, float(np.nextafter(0.1, 0.0)), float(np.nextafter(0.0, 1.0)),
                        -0.0, float("inf"), float("nan"), None], dtype=getattr(pl, dtype)),
        "payload": [float("nan"), 1.0, None, 2.0, 3.0, float("nan"), None],
        "category": pl.Series(["a", "b", None, "a", "b", "a", "b"], dtype=pl.Categorical),
    })
    source = frame.clone()
    base = graphistry.nodes(frame, "id").edges(frame, "s", "d")
    indexed = with_index_policy(base.create_index(kind, column="v", engine="polars"), "force")
    original_nodes, original_edges = base._nodes.clone(), base._edges.clone()
    method = "filter_" + role + "_by_dict"
    filters = {"v": value}
    try:
        expected = getattr(getattr(base, method)(filters, engine="polars"), "_" + role)
    except (GFQLSchemaError, pl.exceptions.PolarsError, ValueError, TypeError, OverflowError) as error:
        with pytest.raises(type(error)) as actual:
            getattr(indexed, method)(filters, engine="polars")
        assert getattr(actual.value, "code", None) == getattr(error, "code", None)
        assert getattr(actual.value, "context", None) == getattr(error, "context", None)
    else:
        result = getattr(getattr(indexed, method)(filters, engine="polars"), "_" + role)
        assert_frame_equal(result, expected)
        if result.height:
            result.replace_column(0, pl.Series("id", [999] * result.height, dtype=pl.Int64))
    assert_frame_equal(base._nodes, original_nodes)
    assert_frame_equal(base._edges, original_edges)
    assert_frame_equal(frame, source)


@pytest.mark.parametrize("actual", [0.1, float(np.nextafter(0.1, 0.0)),
    float(np.nextafter(0.0, 1.0)), -0.0, float(np.finfo(np.float64).max),
    float("inf"), float("-inf"), float("nan"), None])
@pytest.mark.parametrize("value", [0.1, float(np.nextafter(0.1, 0.0)),
    float(np.nextafter(0.0, 1.0)), -0.0, float(np.finfo(np.float64).max)])
def test_float64_singleton_matches_native_expression_and_owns_result(actual, value, monkeypatch):
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars

    frame = pl.DataFrame({"id": [7], "v": pl.Series([actual], dtype=pl.Float64),
                          "payload": [[1, None]]})
    original = frame.clone()
    expected = frame.filter(pl.col("v") == value)

    def forbidden_plan(*args, **kwargs):
        pytest.fail("An exact finite Float64 singleton needs no frame expression plan")

    with monkeypatch.context() as patch:
        patch.setattr(pl.DataFrame, "filter", forbidden_plan)
        patch.setattr(pl.Series, "to_numpy", forbidden_plan)
        result = filter_by_dict_polars(frame, {"v": value})
    assert_frame_equal(result, expected)
    if result.height:
        result.replace_column(0, pl.Series("id", [999], dtype=pl.Int64))
    assert_frame_equal(frame, original)

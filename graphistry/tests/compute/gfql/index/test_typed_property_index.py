"""Typed property indexes preserve canonical rows, storage, and policy behavior."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.gfql.index import get_registry, index_trace, with_index_policy
from graphistry.tests.compute.gfql.index.test_edge_property_index import frame_records


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def typed_graph(engine, dtype):
    if dtype == "category":
        values = [f"label{i}" for i in range(400)]
        values[7] = values[9] = "alice"
        values[15] = None
        column = pd.Series(pd.Categorical(values, categories=["unused"] + sorted(set(v for v in values if v is not None)), ordered=True))
        query = "alice"
    else:
        values = list(pd.date_range("2025-01-01", periods=400, freq="h"))
        values[9] = values[7]
        values[15] = None
        column = pd.Series(np.asarray(values, dtype=dtype))
        query = pd.Timestamp("2025-01-01 07:00:00")
    nodes = pd.DataFrame({"id": np.arange(400), "value": column, "keep": np.arange(400) % 3})
    nodes.index = np.arange(400)[::-1]
    edges = nodes.rename(columns={"id": "s"}).assign(d=np.arange(400) + 1)
    g = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id").edges(df_to_engine(edges, Engine(engine)), "s", "d")
    return g, query


@pytest.mark.parametrize("dtype", ["category", "datetime64[ns]", "datetime64[us]", "datetime64[ms]"])
@pytest.mark.parametrize("kind,role", [("node_prop", "nodes"), ("edge_prop", "edges")])
def test_typed_scalar_engages_and_preserves_rows(engine, dtype, kind, role):
    base, value = typed_graph(engine, dtype)
    indexed = base.create_index(kind, column="value", engine=engine)
    method_name = "filter_nodes_by_dict" if role == "nodes" else "filter_edges_by_dict"
    expected = getattr(base, method_name)({"value": value}, engine=engine)
    expected_frame = expected._nodes if role == "nodes" else expected._edges
    for policy in ["off", "use", "force"]:
        with index_trace() as steps:
            actual = getattr(with_index_policy(indexed, policy), method_name)({"value": value}, engine=engine)
        actual_frame = actual._nodes if role == "nodes" else actual._edges
        assert frame_records(actual_frame) == frame_records(expected_frame)
        assert [r["id" if role == "nodes" else "s"] for r in frame_records(actual_frame)] == [7, 9]
        assert actual_frame.schema == expected_frame.schema if engine.startswith("polars") else actual_frame.dtypes.equals(expected_frame.dtypes)
        if policy != "off":
            assert any(s.get("op") == "property_lookup" and s.get("path") == "index" for s in steps)
    assert get_registry(base).is_empty()
    assert indexed.show_indexes(engine=engine)["n_rows"].tolist() == [400]


@pytest.mark.parametrize("values,query", [([1, 2, 1, None], 1), ([1.5, 2.5, 1.5, None], 1.5), ([True, False, True, None], True)])
def test_numeric_categories_keep_label_semantics(engine, values, query):
    if engine.startswith("polars"):
        pytest.skip("Polars categories hold text labels; numeric storage remains numeric")
    frame = df_to_engine(pd.DataFrame({"id": [0, 1, 2, 3], "value": pd.Series(values, dtype="category")}), Engine(engine))
    base = graphistry.nodes(frame, "id")
    indexed = with_index_policy(base.create_index("node_prop", column="value", engine=engine), "force")
    for member in [query, [query]]:
        expected = base.filter_nodes_by_dict({"value": member}, engine=engine)._nodes
        with index_trace() as steps:
            actual = indexed.filter_nodes_by_dict({"value": member}, engine=engine)._nodes
        assert frame_records(actual) == frame_records(expected)
        assert any(s.get("path") == "index" for s in steps)


@pytest.mark.parametrize("dtype", ["category", "datetime64[ns]"])
def test_typed_public_query_residual_and_missing_value(engine, dtype):
    base, value = typed_graph(engine, dtype)
    indexed = base.create_index("node_prop", column="value", engine=engine)
    for predicate in [{"value": value, "keep": 1}, {"value": "missing" if dtype == "category" else pd.Timestamp("2000-01-01")}]:
        if engine == "cudf" and dtype == "category" and predicate["value"] == "missing":
            for g in [base, indexed, with_index_policy(indexed, "force")]:
                with pytest.raises(ValueError):
                    g.filter_nodes_by_dict(predicate, engine=engine)
            continue
        actual = indexed.filter_nodes_by_dict(predicate, engine=engine)._nodes
        scan = base.filter_nodes_by_dict(predicate, engine=engine)._nodes
        assert frame_records(actual) == frame_records(scan)
    if dtype == "category":
        query = "MATCH (a {value: 'alice'}) RETURN a.id AS id"
        assert frame_records(indexed.gfql(query, engine=engine)._nodes) == frame_records(base.gfql(query, engine=engine)._nodes)


@pytest.mark.parametrize("values", [[], [None, None]])
@pytest.mark.parametrize("dtype", ["category", "datetime64[ns]"])
def test_typed_empty_and_all_null(engine, values, dtype):
    column = pd.Series(pd.Categorical(values, categories=["unused"])) if dtype == "category" else pd.Series(values, dtype=dtype)
    frame = df_to_engine(pd.DataFrame({"id": np.arange(len(values)), "value": column}), Engine(engine))
    indexed = graphistry.nodes(frame, "id").create_index("node_prop", column="value", engine=engine)
    assert get_registry(indexed).node_props["value"].n_keys == 0
    member = "unused" if dtype == "category" else datetime(2025, 1, 1)
    assert len(indexed.filter_nodes_by_dict({"value": member}, engine=engine)._nodes) == 0


@pytest.mark.parametrize("epoch", [1735689600000000000, -1000])
@pytest.mark.parametrize("query_kind", ["timestamp", "numpy", "text"])
def test_native_polars_nanoseconds_match_canonical_cast(epoch, query_kind):
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({
        "id": np.arange(400),
        "value": pl.Series(np.arange(400) + epoch).cast(pl.Datetime("ns")),
    })
    stamp = pd.Timestamp(epoch, unit="ns")
    member = {"timestamp": stamp, "numpy": np.datetime64(epoch, "ns"), "text": stamp.isoformat()}[query_kind]
    base = graphistry.nodes(frame, "id")
    indexed = base.create_index("node_prop", column="value", engine="polars")
    for policy in ["use", "force"]:
        actual = with_index_policy(indexed, policy).filter_nodes_by_dict({"value": member}, engine="polars")._nodes
        scan = base.filter_nodes_by_dict({"value": member}, engine="polars")._nodes
        assert actual.equals(scan)


@pytest.mark.parametrize("timezone", ["UTC", "America/New_York"])
def test_pandas_timezone_timestamp_matches_instant(timezone):
    column = pd.Series(pd.date_range("2025-03-08", periods=400, freq="h", tz=timezone))
    base = graphistry.nodes(pd.DataFrame({"id": np.arange(400), "value": column}), "id")
    indexed = base.create_index("node_prop", column="value")
    members = [column.iloc[7], column.iloc[7].tz_convert("Asia/Tokyo"), column.iloc[7].tz_localize(None)]
    for member in members:
        for policy in ["off", "use", "force"]:
            actual = with_index_policy(indexed, policy).filter_nodes_by_dict({"value": member})._nodes
            pd.testing.assert_frame_equal(actual, base.filter_nodes_by_dict({"value": member})._nodes)


@pytest.mark.parametrize("labels", [[1, 2, 1, None], ["a", "b", "a", None], [2**63 + 1, 2**63 + 2, 2**63 + 1, None]])
@pytest.mark.parametrize("other", ["missing", 1.5, True, None])
def test_category_mixed_membership_keeps_canonical_inference(engine, labels, other):
    if engine.startswith("polars"):
        pytest.skip("Mixed numeric/text categories are pandas/cuDF storage")
    frame = df_to_engine(pd.DataFrame({"id": range(4), "value": pd.Series(labels, dtype="category")}), Engine(engine))
    base = graphistry.nodes(frame, "id")
    indexed = with_index_policy(base.create_index("node_prop", column="value", engine=engine), "force")
    query = {"value": [labels[0], other]}
    actual = indexed.filter_nodes_by_dict(query, engine=engine)._nodes
    expected = base.filter_nodes_by_dict(query, engine=engine)._nodes
    assert frame_records(actual) == frame_records(expected)


@pytest.mark.route_engaged("native-fast", "polars-single-node", "cypher-fast")
def test_categorical_cypher_lookup_reports_gather(engine):
    base, _ = typed_graph(engine, "category")
    indexed = base.create_index("node_prop", column="value", engine=engine)
    assert indexed.gfql_explain("MATCH (a {value: 'alice'}) RETURN a.id AS id", engine=engine)["used_index"]


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_arrow_timestamps_preserve_units_and_rows(unit):
    pa = pytest.importorskip("pyarrow")
    values = list(pd.date_range("2025-01-01", periods=400, freq="h"))
    values[9] = values[7]
    values[15] = None
    column = pd.Series(values, dtype=pd.ArrowDtype(pa.timestamp(unit)))
    base = graphistry.nodes(pd.DataFrame({"id": np.arange(400), "value": column}), "id")
    indexed = base.create_index("node_prop", column="value")
    for member in [values[7], [values[7], values[11]], "2025-01-01 07:00:00"]:
        actual = with_index_policy(indexed, "force").filter_nodes_by_dict({"value": member})._nodes
        pd.testing.assert_frame_equal(actual, base.filter_nodes_by_dict({"value": member})._nodes)


@pytest.mark.parametrize("dtype_kind", ["categorical", "enum"])
def test_native_polars_category_dictionary_is_independent_of_label_order(dtype_kind):
    pl = pytest.importorskip("polars")
    dtype = pl.Categorical if dtype_kind == "categorical" else pl.Enum(["z", "b", "a", "unused"])
    values = ["z"] * 400
    values[7] = values[9] = "b"
    values[15] = None
    frame = pl.DataFrame({"id": np.arange(400), "value": pl.Series(values, dtype=dtype)})
    base = graphistry.nodes(frame, "id")
    indexed = base.create_index("node_prop", column="value", engine="polars")
    for predicate in ["b", ["b", "a", "b"]]:
        with index_trace() as steps:
            actual = indexed.filter_nodes_by_dict({"value": predicate}, engine="polars")._nodes
        assert actual.equals(base.filter_nodes_by_dict({"value": predicate}, engine="polars")._nodes)
        assert actual["id"].to_list() == [7, 9]
        assert any(s.get("path") == "index" for s in steps)


@pytest.mark.parametrize("dtype", ["category", "datetime64[ns]"])
def test_typed_gather_preserves_missing_key_residual_error(engine, dtype):
    from graphistry.compute.exceptions import GFQLSchemaError
    base, _ = typed_graph(engine, dtype)
    indexed = base.create_index("node_prop", column="value", engine=engine)
    missing = "unused" if dtype == "category" else pd.Timestamp("2000-01-01")
    outcomes = []
    for policy in ["off", "use", "force"]:
        with pytest.raises(GFQLSchemaError) as caught:
            with_index_policy(indexed, policy).filter_nodes_by_dict({"value": missing, "keep": "bad"}, engine=engine)
        outcomes.append((caught.value.code, caught.value.context["field"]))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_arrow_timestamp_query_stays_bounded_after_index_build(unit, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    column = pd.Series(pd.date_range("2025-01-01", periods=400, freq="h"), dtype=pd.ArrowDtype(pa.timestamp(unit)))
    indexed = graphistry.nodes(pd.DataFrame({"id": np.arange(400), "value": column}), "id").create_index("node_prop", column="value")
    array_type = type(column.array)
    original = array_type.to_numpy

    def bounded_export(array, *args, **kwargs):
        assert len(array) <= 8
        return original(array, *args, **kwargs)

    monkeypatch.setattr(array_type, "to_numpy", bounded_export)
    with index_trace() as steps:
        actual = indexed.filter_nodes_by_dict({"value": pd.Timestamp("2025-01-01 07:00:00")})._nodes
    assert actual["id"].tolist() == [7]
    assert any(s.get("path") == "index" for s in steps)


@pytest.mark.parametrize("as_ast", [False, True])
def test_category_nan_membership_preserves_distinct_literal_and_ast_null_semantics(engine, as_ast):
    from graphistry.compute.predicates.is_in import IsIn
    if engine.startswith("polars"):
        pytest.skip("Polars categorical labels are text")
    frame = df_to_engine(pd.DataFrame({"id": [0, 1, 2], "value": pd.Series([1.5, 2.5, None], dtype="category")}), Engine(engine))
    base = graphistry.nodes(frame, "id")
    indexed = with_index_policy(base.create_index("node_prop", column="value", engine=engine), "force")
    predicate = IsIn([float("nan")]) if as_ast else [float("nan")]
    expected = base.filter_nodes_by_dict({"value": predicate}, engine=engine)._nodes
    actual = indexed.filter_nodes_by_dict({"value": predicate}, engine=engine)._nodes
    if engine == "cudf":
        pd.testing.assert_frame_equal(actual.to_pandas(), expected.to_pandas())
    else:
        pd.testing.assert_frame_equal(actual, expected)
    assert len(actual) == (1 if as_ast else 0)


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("value", [
    datetime(1969, 12, 31, 23, 59, 59, 999999),
    datetime(1970, 1, 1, 0, 0, 0, 1),
    datetime(2026, 1, 1, 12, 30, 1, 123456),
])
def test_native_temporal_literal_metadata_matches_polars_without_series_plan(unit, value, monkeypatch):
    from dataclasses import replace
    from graphistry.compute.gfql.index.property_keys import property_query_values

    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": [0], "value": pl.Series([value], dtype=pl.Datetime(unit))})
    indexed = graphistry.nodes(frame, "id").create_index("node_prop", column="value", engine="polars")
    # A metadata-only oracle also covers native ns literals independently of
    # the builder's conservative microsecond candidate buckets.
    index = replace(get_registry(indexed).node_props["value"], timestamp_dtype=pl.Datetime(unit))
    expected = pl.Series([value], dtype=pl.Datetime(unit)).cast(pl.Int64).to_numpy()

    def forbidden_series(*args, **kwargs):
        raise AssertionError("Naive scalar encoding must use bounded integer literal metadata")

    monkeypatch.setattr(pl.Series, "__init__", forbidden_series)
    actual = property_query_values(index, value, np)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype_name", ["String", "Categorical", "Enum"])
@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
def test_dense_text_dictionary_native_scan_matches_canonical(dtype_name, role, kind):
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    dtype = pl.Enum(["a", "b", "c", "d"]) if dtype_name == "Enum" else getattr(pl, dtype_name)
    frame = pl.DataFrame({"id": range(400), "s": range(400), "d": range(400), "v": [None if i % 11 == 0 else "abcd"[i % 4] for i in range(400)]}).with_columns(pl.col("v").cast(dtype))
    base = graphistry.nodes(frame, "id").edges(frame, "s", "d")
    indexed = base.create_index(kind, column="v", engine="polars")
    method = "filter_" + role + "_by_dict"
    for value in ["b", "missing"]:
        assert_frame_equal(getattr(getattr(indexed, method)({"v": value}, engine="polars"), "_" + role),
                           getattr(getattr(base, method)({"v": value}, engine="polars"), "_" + role))


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
def test_dense_temporal_native_scan_preserves_values_schema_and_source(unit, nullable, role, kind, monkeypatch):
    from datetime import datetime, timedelta
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal

    epoch = datetime(2026, 1, 1)
    values = [None if nullable and i % 11 == 0 else epoch + timedelta(seconds=i % 4) for i in range(400)]
    frame = pl.DataFrame({"id": range(400), "s": range(400), "d": range(400),
                          "value": pl.Series(values, dtype=pl.Datetime(unit)), "payload": [[i, None] for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "s", "d")
    indexed = base.create_index(kind, column="value", engine="polars")
    method = "filter_" + role + "_by_dict"
    filters = {"value": (epoch + timedelta(seconds=1)).isoformat()}
    expected = getattr(getattr(base, method)(filters, engine="polars"), "_" + role)
    original = frame.clone()
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            pytest.fail("Dense temporal scans must retain native column filtering")
        patch.setattr(pl.Series, "to_numpy", forbidden)
        patch.setattr(pl.DataFrame, "filter", forbidden)
        actual = getattr(getattr(indexed, method)(filters, engine="polars"), "_" + role)
    assert_frame_equal(actual, expected)
    actual.replace_column(0, pl.Series("id", [999] * actual.height))
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("storage", ["int64", "uint64"])
@pytest.mark.parametrize("boundary", ["min", "max"])
def test_integer_category_scalar_and_missing_key_match_scan_without_mutation(storage, boundary):
    bounds = np.iinfo(storage)
    value = int(getattr(bounds, boundary))
    other = int(bounds.max if boundary == "min" else bounds.min)
    categories = pd.Index(np.array([value, other], dtype=storage))
    codes = np.ones(400, dtype=np.int64)
    codes[[7, 9]], codes[15] = 0, -1
    frame = pd.DataFrame({"id": np.arange(400), "value": pd.Categorical.from_codes(codes, categories)})
    frame.index = np.arange(400)[::-1]
    original = frame.copy(deep=True)
    base = graphistry.nodes(frame, "id")
    indexed = base.create_index("node_prop", column="value", engine="pandas")
    for member in [value, [value], 17, [17]]:
        expected = base.filter_nodes_by_dict({"value": member}, engine="pandas")._nodes
        for policy in ["off", "use", "force"]:
            actual = with_index_policy(indexed, policy).filter_nodes_by_dict({"value": member}, engine="pandas")._nodes
            pd.testing.assert_frame_equal(actual, expected)
            if len(actual):
                actual.iloc[0, 0] = -1
            pd.testing.assert_frame_equal(frame, original)

"""Business-key strings use native indexed gathers with canonical scan semantics."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ast import n
from graphistry.compute.gfql.index import get_registry, index_trace
from graphistry.tests.compute.gfql.index.test_edge_property_index import frame_records


@pytest.mark.parametrize("kind,role", [("node_prop", "nodes"), ("edge_prop", "edges")])
@pytest.mark.parametrize("value", ["key1", "missing", "\ud800"])
def test_dense_native_text_scan_retains_outputs_errors_and_source_without_encoding(kind, role, value, monkeypatch):
    from graphistry.compute.exceptions import GFQLSchemaError
    from graphistry.compute.gfql.index import property_keys
    pl = pytest.importorskip("polars")
    from polars.testing import assert_frame_equal
    frame = pl.DataFrame({"id": range(400), "s": range(400), "d": range(400),
                          "v": ["key" + str(i % 4) for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "s", "d")
    indexed = base.create_index(kind, column="v", engine="polars")
    nodes, edges = base._nodes.clone(), base._edges.clone()
    method = "filter_" + role + "_by_dict"
    try:
        expected = getattr(getattr(base, method)({"v": value}, engine="polars"), "_" + role)
    except (GFQLSchemaError, pl.exceptions.PolarsError, TypeError, ValueError, UnicodeError) as error:
        expected = error

    def redundant_encoding(*args, **kwargs):
        pytest.fail("A proven dense native text scan needs no dictionary probe")

    with monkeypatch.context() as patch:
        patch.setattr(property_keys, "_string_query_codes", redundant_encoding)
        if isinstance(expected, Exception):
            with pytest.raises(type(expected)) as actual:
                getattr(indexed, method)({"v": value}, engine="polars")
            assert getattr(actual.value, "code", None) == getattr(expected, "code", None)
            assert getattr(actual.value, "context", None) == getattr(expected, "context", None)
        else:
            actual = getattr(getattr(indexed, method)({"v": value}, engine="polars"), "_" + role)
            assert_frame_equal(actual, expected)
            if actual.height:
                actual.replace_column(0, pl.Series("id", [999] * actual.height, dtype=pl.Int64))
    assert_frame_equal(base._nodes, nodes)
    assert_frame_equal(base._edges, edges)


@pytest.mark.parametrize("value", [True, 7, 1.0, ["key1"]])
def test_dense_native_text_preserves_unsupported_admission_before_invalid_cost(value, monkeypatch):
    from graphistry.compute.exceptions import GFQLSchemaError

    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": range(400), "v": ["key" + str(i % 4) for i in range(400)]})
    base = graphistry.nodes(frame, "id")
    indexed = base.create_index("node_prop", column="v", engine="polars")
    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC_POLARS", "invalid")
    if isinstance(value, list):
        # Supported membership still reaches the same cost configuration.
        with pytest.raises(ValueError):
            indexed.filter_nodes_by_dict({"v": value}, engine="polars")
        return
    try:
        expected = base.filter_nodes_by_dict({"v": value}, engine="polars")._nodes
    except (GFQLSchemaError, pl.exceptions.PolarsError, TypeError, ValueError) as error:
        with pytest.raises(type(error)) as actual:
            indexed.filter_nodes_by_dict({"v": value}, engine="polars")
        assert getattr(actual.value, "code", None) == getattr(error, "code", None)
        assert getattr(actual.value, "context", None) == getattr(error, "context", None)
    else:
        from polars.testing import assert_frame_equal
        assert_frame_equal(indexed.filter_nodes_by_dict({"v": value}, engine="polars")._nodes, expected)


def test_dense_native_text_force_trace_and_override_still_encode(monkeypatch):
    from graphistry.compute.gfql.index import property_keys, set_cost_gate_frac, reset_cost_gate_frac, with_index_policy

    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": range(400), "v": ["key" + str(i % 4) for i in range(400)]})
    indexed = graphistry.nodes(frame, "id").create_index("node_prop", column="v", engine="polars")
    original = property_keys._string_query_codes
    calls = []

    def encode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(property_keys, "_string_query_codes", encode)
    with index_trace() as steps:
        assert len(indexed.filter_nodes_by_dict({"v": "missing"}, engine="polars")._nodes) == 0
    assert calls and any(s.get("path") == "index" and s.get("est_result_rows") == 0 for s in steps)
    calls.clear()
    try:
        set_cost_gate_frac(Engine.POLARS, 0.5)
        assert len(indexed.filter_nodes_by_dict({"v": "key1"}, engine="polars")._nodes) == 100
        assert calls
    finally:
        reset_cost_gate_frac(Engine.POLARS)
    calls.clear()
    monkeypatch.setenv("GFQL_INDEX_COST_GATE_FRAC_POLARS", "invalid")
    assert len(with_index_policy(indexed, "force").filter_nodes_by_dict({"v": "key1"}, engine="polars")._nodes) == 100
    assert calls


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine, dtype="object"):
    if dtype in ("arrow-string", "arrow-large-string"):
        arrow = pytest.importorskip("pyarrow")
        dtype = pd.ArrowDtype(arrow.string() if dtype == "arrow-string" else arrow.large_string())
    elif dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    values = [f"user{i}@example.test" for i in range(400)]
    values[7] = values[9] = "alice@example.test"
    values[11] = ""
    values[13] = "é用户🙂"
    values[15] = None
    emails = pd.Series(values, dtype=dtype)
    nodes = pd.DataFrame({"id": np.arange(400), "email": emails, "keep": np.arange(400) % 3})
    nodes.index = np.arange(400)[::-1]
    edges = pd.DataFrame({"s": np.arange(399), "d": np.arange(399) + 1, "external_id": emails.iloc[:399].array})
    concrete = Engine(engine)
    return graphistry.nodes(df_to_engine(nodes, concrete), "id").edges(
        df_to_engine(edges, concrete), "s", "d",
    )


@pytest.mark.parametrize("check_engagement", [
    False,
    pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "cypher-fast")),
])
@pytest.mark.parametrize("dtype", ["object", "string", "string[pyarrow]", "arrow-string", "arrow-large-string"])
@pytest.mark.parametrize("query,expected", [
    ("MATCH (a {email: 'alice@example.test'}) RETURN a.id AS id", [7, 9]),
    ("MATCH (a {email: 'alice@example.test', keep: 1}) RETURN a.id AS id", [7]),
    ("MATCH (a {email: ''}) RETURN a.id AS id", [11]),
    ("MATCH (a {email: 'é用户🙂'}) RETURN a.id AS id", [13]),
    ("MATCH (a {email: 'missing'}) RETURN a.id AS id", []),
    ("MATCH (a {email: 'alice@example.test'})-[e]->(b) RETURN b.id AS id", [8, 10]),
    ("MATCH (a) WHERE a.email IN ['alice@example.test', 'alice@example.test', 'é用户🙂'] RETURN a.id AS id", [7, 9, 13]),
])
def test_string_business_key_query_parity_and_engagement(engine, dtype, query, expected, check_engagement):
    base = graph(engine, dtype)
    indexed = base.gfql("CREATE GFQL INDEX FOR node_prop ON (email)", engine=engine)
    assert get_registry(base).is_empty()
    actual = frame_records(indexed.gfql(query, engine=engine)._nodes)
    scan = frame_records(indexed.gfql(query, engine=engine, index_policy="off")._nodes)
    assert actual == scan
    assert [r["id"] for r in actual] == expected
    report = indexed.gfql_explain(query, engine=engine)
    assert report["error"] is None
    if check_engagement:
        assert report["used_index"]


@pytest.mark.parametrize("kind,role,column", [("node_prop", "nodes", "email"), ("edge_prop", "edges", "external_id")])
def test_direct_string_membership_gathers_in_input_order(engine, kind, role, column):
    indexed = graph(engine).create_index(kind, column=column, engine=engine)
    method = indexed.filter_nodes_by_dict if role == "nodes" else indexed.filter_edges_by_dict
    with index_trace() as steps:
        out = method({column: ["é用户🙂", "alice@example.test", "alice@example.test"]}, engine=engine)
    frame = out._nodes if role == "nodes" else out._edges
    id_col = "id" if role == "nodes" else "s"
    assert [r[id_col] for r in frame_records(frame)] == [7, 9, 13]
    assert any(s.get("op") == "property_lookup" and s.get("path") == "index" for s in steps)
    shown = indexed.show_indexes(engine=engine)
    assert shown["valid"].all() and shown["usable"].all()
    assert shown["n_rows"].tolist() == [400 if role == "nodes" else 399]
    assert shown["nbytes"].iloc[0] > 0


def test_string_index_stale_rebinding_declines(engine):
    base = graph(engine)
    indexed = base.create_index("node_prop", column="email", engine=engine)
    rebound = indexed.nodes(graph(engine)._nodes)
    with index_trace() as steps:
        out = rebound.filter_nodes_by_dict({"email": "alice@example.test"}, engine=engine)
    assert [r["id"] for r in frame_records(out._nodes)] == [7, 9]
    assert not any(s.get("path") == "index" for s in steps)


@pytest.mark.parametrize("dtype", ["string", "arrow-string", "arrow-large-string"])
@pytest.mark.parametrize("values", [[], [None, None]])
def test_empty_and_all_null_text_indexes(engine, values, dtype):
    if dtype.startswith("arrow-"):
        pa = pytest.importorskip("pyarrow")
        dtype = pd.ArrowDtype(pa.string() if dtype == "arrow-string" else pa.large_string())
    nodes = pd.DataFrame({"id": np.arange(len(values)), "email": pd.Series(values, dtype=dtype)})
    base = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id")
    indexed = base.create_index("node_prop", column="email", engine=engine)
    assert get_registry(indexed).node_props["email"].n_keys == 0
    assert frame_records(indexed.filter_nodes_by_dict({"email": "missing"}, engine=engine)._nodes) == []


def test_mixed_object_column_is_not_coerced_to_text():
    base = graphistry.nodes(pd.DataFrame({"id": [0, 1], "mixed": ["1", 1]}), "id")
    with pytest.raises(NotImplementedError):
        base.create_index("node_prop", column="mixed")


def test_missing_business_key_preserves_invalid_residual_error(engine):
    from graphistry.compute.exceptions import GFQLSchemaError

    indexed = graph(engine).create_index("node_prop", column="email", engine=engine)
    query = "MATCH (a {email: 'missing', keep: 'bad'}) RETURN a.id AS id"
    outcomes = []
    for policy in ["off", "use", "force"]:
        with pytest.raises(GFQLSchemaError) as caught:
            indexed.gfql(query, engine=engine, index_policy=policy)
        outcomes.append((caught.value.code, caught.value.context["field"]))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("surface", ["direct", "native", "cypher"])
def test_missing_business_key_preserves_temporal_residual(engine, surface):
    from graphistry.compute.gfql.index.api import with_index_policy

    base = graph(engine)
    nodes = pd.DataFrame({"id": np.arange(400), "email": [f"user{i}@example.test" for i in range(400)],
                          "time": pd.date_range("2026-01-01", periods=400)})
    indexed = base.nodes(df_to_engine(nodes, Engine(engine))).create_index("node_prop", column="email", engine=engine)
    filters = {"email": "missing", "time": "2026-01-01T00:00:00"}
    query = ([n(filters)] if surface == "native" else
             "MATCH (a {email: 'missing', time: '2026-01-01T00:00:00'}) RETURN a.id AS id")

    def run(policy):
        if surface == "direct":
            return with_index_policy(indexed, policy).filter_nodes_by_dict(filters, engine=engine)
        return indexed.gfql(query, engine=engine, index_policy=policy)

    reference = run("off")
    for policy in ("use", "force"):
        actual = run(policy)
        assert len(actual._nodes) == 0
        for name in ("_nodes", "_edges"):
            expected_frame, actual_frame = getattr(reference, name), getattr(actual, name)
            assert frame_records(actual_frame) == frame_records(expected_frame)
            assert list(actual_frame.columns) == list(expected_frame.columns)
            assert list(actual_frame.dtypes) == list(expected_frame.dtypes)


@pytest.mark.parametrize("email", ["missing", "alice@example.test"])
def test_unfiltered_temporal_column_keeps_string_index_engaged(engine, email):
    from datetime import datetime
    from graphistry.compute.gfql.index.api import with_index_policy

    base = graph(engine)
    if engine in ("polars", "polars-gpu"):
        import polars as pl
        nodes = base._nodes.with_columns(pl.lit(datetime(2026, 1, 1)).alias("unrelated_time"))
    else:
        nodes = base._nodes.assign(unrelated_time=datetime(2026, 1, 1))
    indexed = base.nodes(nodes).create_index("node_prop", column="email", engine=engine)
    with index_trace() as steps:
        actual = indexed.filter_nodes_by_dict({"email": email}, engine=engine)
    reference = with_index_policy(indexed, "off").filter_nodes_by_dict({"email": email}, engine=engine)
    assert frame_records(actual._nodes) == frame_records(reference._nodes)
    assert any(s.get("decision_code") == "index_selected" for s in steps)


@pytest.mark.parametrize("dtype", ["string[pyarrow]", "arrow-string", "arrow-large-string"])
def test_arrow_key_queries_do_not_export_whole_columns(dtype, monkeypatch):
    indexed = graph("pandas", dtype).create_index("node_prop", column="email")
    array_type = type(indexed._nodes["email"].array)
    original = array_type.to_numpy

    def bounded_export(array, *args, **kwargs):
        assert len(array) <= 8, "query exported a whole Arrow column"
        return original(array, *args, **kwargs)

    monkeypatch.setattr(array_type, "to_numpy", bounded_export)
    out = indexed.filter_nodes_by_dict({"email": "alice@example.test"})
    assert out._nodes["id"].tolist() == [7, 9]
    assert out._nodes["email"].dtype == indexed._nodes["email"].dtype


@pytest.mark.parametrize("check_engagement", [
    False,
    pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node")),
])
def test_native_business_key_lookup_preserves_consumer_trace(engine, check_engagement):
    indexed = graph(engine).gfql_index_all(engine=engine).create_index("node_prop", column="email", engine=engine)
    query = [n({"email": "alice@example.test"})]
    actual = indexed.gfql(query, engine=engine)._nodes
    assert [r["id"] for r in frame_records(actual)] == [7, 9]
    assert frame_records(actual) == frame_records(indexed.gfql(query, engine=engine, index_policy="off")._nodes)
    report = indexed.gfql_explain(query, engine=engine)
    assert report["error"] is None
    if not check_engagement:
        return
    assert report["used_index"]
    assert [(s["seam"], s["reason"], s["hops"]) for s in report["steps"]] == [
        ("native_seed_lookup", "property_index", 0),
    ]


@pytest.mark.parametrize("large", [False, True])
def test_native_arrow_binary_is_not_coerced_to_string(large):
    pa = pytest.importorskip("pyarrow")
    dtype = pd.ArrowDtype(pa.large_binary() if large else pa.binary())
    nodes = pd.DataFrame({"id": [0, 1], "value": pd.Series([b"alice", b"bob"], dtype=dtype)})
    with pytest.raises(NotImplementedError):
        graphistry.nodes(nodes, "id").create_index("node_prop", column="value")


@pytest.mark.parametrize("member", ["key1", "雪", "🦀", "", "absent"])
def test_native_polars_scalar_dictionary_lookup_avoids_query_plans_and_text_export(member, monkeypatch):
    from graphistry.compute.gfql.index.property_keys import property_query_values

    pl = pytest.importorskip("polars")
    values = [f"key{i}" for i in range(10000)] + ["雪", "🦀", "", "雪"]
    frame = pl.DataFrame({"id": range(len(values)), "email": values})
    original = frame.clone()
    base = graphistry.nodes(frame, "id")
    indexed = base.create_index("node_prop", column="email", engine="polars")
    expected = base.filter_nodes_by_dict({"email": member}, engine="polars")._nodes
    original_export = pl.Series.to_numpy

    def bounded_export(series, *args, **kwargs):
        assert series.dtype != pl.String or len(series) <= 8
        return original_export(series, *args, **kwargs)

    def forbidden_plan(*args, **kwargs):
        raise AssertionError("Scalar dictionary metadata must not construct a query plan")

    monkeypatch.setattr(pl.Series, "to_numpy", bounded_export)
    with monkeypatch.context() as patch:
        patch.setattr(pl.LazyFrame, "collect", forbidden_plan)
        if hasattr(pl.LazyFrame, "_collect_eager"):
            patch.setattr(pl.LazyFrame, "_collect_eager", forbidden_plan)
        property_query_values(get_registry(indexed).node_props["email"], member, np)
    result = indexed.filter_nodes_by_dict({"email": member}, engine="polars")._nodes
    assert result.equals(expected)
    result.replace_column(0, pl.Series("id", [999] * result.height, dtype=pl.Int64))
    assert base._nodes.equals(original)

"""Nullable integer keys preserve exact values, original rows, and scan semantics."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.ast import is_in
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.gfql.index import get_registry, index_trace
from graphistry.compute.gfql.index.api import with_index_policy
from graphistry.tests.compute.gfql.index.test_edge_property_index import frame_records

@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine, dtype="Int64", offset=0):
    if dtype.startswith("arrow-"):
        pa = pytest.importorskip("pyarrow")
        dtype = pd.ArrowDtype(pa.uint64() if dtype == "arrow-uint64" else pa.int64())
    primitive = np.uint64 if pd.api.types.is_unsigned_integer_dtype(dtype) else np.int64
    values = pd.Series(np.arange(400, dtype=primitive) % 100 + offset, dtype=dtype)
    values.iloc[9] = values.iloc[7]
    values.iloc[[0, 15, 399]] = None
    nodes = pd.DataFrame({"id": np.arange(400), "account": values, "keep": np.arange(400) % 3})
    nodes.index = np.arange(400)[::-1]
    edges = pd.DataFrame({"s": np.arange(400), "d": np.arange(400), "account": values, "keep": np.arange(400) % 3})
    edges.index = np.arange(400)[::-1]
    return graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id").edges(
        df_to_engine(edges, Engine(engine)), "s", "d",
    )


@pytest.mark.parametrize("dtype", ["Int8", "Int64", "UInt64", "arrow-int64", "arrow-uint64"])
@pytest.mark.parametrize("kind,role", [("node_prop", "nodes"), ("edge_prop", "edges")])
def test_non_null_lookup_rows_policies_and_engagement(engine, dtype, kind, role):
    base = graph(engine, dtype)
    indexed = base.create_index(kind, column="account", engine=engine)
    assert get_registry(base).is_empty()
    method_name = "filter_nodes_by_dict" if role == "nodes" else "filter_edges_by_dict"
    id_col = "id" if role == "nodes" else "s"
    for policy in ["off", "use", "force"]:
        with index_trace() as steps:
            out = getattr(with_index_policy(indexed, policy), method_name)({"account": 7}, engine=engine)
        frame = out._nodes if role == "nodes" else out._edges
        assert [r[id_col] for r in frame_records(frame)] == [7, 9, 107, 207, 307]
        assert any(s.get("path") == "index" for s in steps) == (policy != "off")
        original = indexed._nodes if role == "nodes" else indexed._edges
        assert frame["account"].dtype == original["account"].dtype
    assert indexed.show_indexes(engine=engine)["n_rows"].tolist() == [400]


@pytest.mark.parametrize("predicate", [is_in([7, 7, 15]), is_in([7, None]), None, 999])
def test_nullable_membership_null_and_absent_match_canonical_scan(engine, predicate):
    indexed = graph(engine).create_index("node_prop", column="account", engine=engine)
    if engine in ("polars", "polars-gpu") and isinstance(predicate, IsIn) and None in predicate.options:
        for policy in ["off", "use", "force"]:
            with pytest.raises(NotImplementedError):
                with_index_policy(indexed, policy).filter_nodes_by_dict({"account": predicate}, engine=engine)
        return
    expected = with_index_policy(indexed, "off").filter_nodes_by_dict({"account": predicate}, engine=engine)._nodes
    for policy in ["use", "force"]:
        actual = with_index_policy(indexed, policy).filter_nodes_by_dict({"account": predicate}, engine=engine)._nodes
        if engine in ("polars", "polars-gpu"):
            from polars.testing import assert_frame_equal
            assert_frame_equal(actual, expected)
        else:
            actual_pd = actual.to_pandas() if engine == "cudf" else actual
            expected_pd = expected.to_pandas() if engine == "cudf" else expected
            pd.testing.assert_frame_equal(actual_pd, expected_pd)


@pytest.mark.parametrize("check_engagement", [
    False,
    pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "cypher-fast")),
])
@pytest.mark.parametrize("pattern,expected", [
    ("MATCH (a {account: 7}) RETURN a.id AS id", [7, 9, 107, 207, 307]),
    ("MATCH (a {account: 7, keep: 1}) RETURN a.id AS id", [7, 307]),
    ("MATCH (a {account: 7})-[e]->(b) RETURN b.id AS id", [7, 9, 107, 207, 307]),
    ("MATCH (a)-[e {account: 7}]->(b) RETURN a.id AS id", [7, 9, 107, 207, 307]),
])
def test_nullable_public_cypher_seeds(engine, pattern, expected, check_engagement):
    indexed = graph(engine).create_index("node_prop", column="account", engine=engine).create_index(
        "edge_prop", column="account", engine=engine,
    )
    scan = frame_records(indexed.gfql(pattern, engine=engine, index_policy="off")._nodes)
    assert [r["id"] for r in scan] == expected
    assert frame_records(indexed.gfql(pattern, engine=engine)._nodes) == scan
    if check_engagement:
        assert indexed.gfql_explain(pattern, engine=engine)["used_index"]


@pytest.mark.parametrize("dtype", ["Int8", "Int64", "UInt64"])
@pytest.mark.parametrize("values", [[], [None, None]])
def test_empty_and_all_null_integer_columns(engine, values, dtype):
    nodes = pd.DataFrame({"id": np.arange(len(values)), "account": pd.Series(values, dtype=dtype)})
    indexed = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id").create_index(
        "node_prop", column="account", engine=engine,
    )
    assert get_registry(indexed).node_props["account"].n_keys == 0
    assert len(indexed.filter_nodes_by_dict({"account": 7}, engine=engine)._nodes) == 0


@pytest.mark.parametrize("value,expected", [(2**63 + 7, [7, 9, 107, 207, 307]), (-1, []), (2**64, []), (2**53 + 1, [])])
def test_nullable_unsigned_values_keep_integer_precision(engine, value, expected):
    indexed = graph(engine, "UInt64", 2**63).create_index("node_prop", column="account", engine=engine)
    scan = with_index_policy(indexed, "off").filter_nodes_by_dict({"account": value}, engine=engine)._nodes
    assert [r["id"] for r in frame_records(scan)] == expected
    actual = indexed.filter_nodes_by_dict({"account": value}, engine=engine)._nodes
    assert frame_records(actual) == frame_records(scan)


def test_absent_nullable_key_retains_residual_error(engine):
    from graphistry.compute.exceptions import GFQLSchemaError
    indexed = graph(engine).create_index("node_prop", column="account", engine=engine)
    outcomes = []
    for policy in ["off", "use", "force"]:
        with pytest.raises(GFQLSchemaError) as caught:
            indexed.gfql("MATCH (a {account: 999, keep: 'bad'}) RETURN a.id", engine=engine, index_policy=policy)
        outcomes.append((caught.value.code, caught.value.context["field"]))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize("offset", [-(2**63), 2**53])
def test_nullable_signed_extremes_are_not_converted_to_float(engine, offset):
    indexed = graph(engine, "Int64", offset).create_index("node_prop", column="account", engine=engine)
    filters = {"account": offset + 7}
    for policy in ["off", "use", "force"]:
        out = with_index_policy(indexed, policy).filter_nodes_by_dict(filters, engine=engine)
        assert [r["id"] for r in frame_records(out._nodes)] == [7, 9, 107, 207, 307]


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("value", [1, 9])
def test_dense_nullable_scan_preserves_native_order_schema_and_ownership(role, kind, value, monkeypatch):
    from graphistry.tests.compute.gfql.index.test_native_dense_property_scan import test_dense_integer_scan_preserves_schema_order_nulls_and_native_ownership
    test_dense_integer_scan_preserves_schema_order_nulls_and_native_ownership(role, kind, True, value, monkeypatch)


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint64])
@pytest.mark.parametrize("kind,role", [("node_prop", "nodes"), ("edge_prop", "edges")])
def test_non_null_integer_build_retains_direct_storage_path(engine, dtype, kind, role, monkeypatch):
    import importlib
    build_module = importlib.import_module("graphistry.compute.gfql.index.build")
    if engine == "cudf":
        pytest.importorskip("cudf")
        from cudf.testing import assert_frame_equal
    else:
        assert_frame_equal = pd.testing.assert_frame_equal
    base = graph(engine)
    frame = getattr(base, "_" + role).assign(account=np.arange(400, dtype=dtype) % 100)
    base = base.nodes(frame) if role == "nodes" else base.edges(frame)
    before = frame.copy(deep=True)

    def reject_null_filter(*args, **kwargs):
        pytest.fail("primitive integer storage must not allocate a null mask")

    monkeypatch.setattr(build_module, "_non_null_id_rows", reject_null_filter)
    indexed = base.create_index(kind, column="account", engine=engine)
    method = "filter_nodes_by_dict" if role == "nodes" else "filter_edges_by_dict"
    actual = getattr(indexed, method)({"account": 7}, engine=engine)
    assert_frame_equal(getattr(actual, "_" + role), frame[frame.account == 7])
    assert_frame_equal(frame, before)
    assert getattr(actual, "_" + role).account.dtype == np.dtype(dtype)


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("gpu_target", [False, True])
def test_polars_gpu_dense_cost_decline_skips_candidate_work(role, kind, gpu_target, monkeypatch):
    pytest.importorskip("cudf_polars")
    import importlib
    from polars.testing import assert_frame_equal
    from graphistry.compute.gfql.lazy import ExecutionTarget, target_mode

    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": np.arange(400), "s": np.arange(400), "d": np.arange(400),
                          "account": [None if i % 11 == 0 else i % 4 for i in range(400)]})
    base = graphistry.nodes(frame, "id").edges(frame, "s", "d")
    indexed = base.create_index(kind, column="account", engine="polars-gpu")
    method = "filter_" + role + "_by_dict"
    original = frame.clone()
    lookup = importlib.import_module("graphistry.compute.gfql.index.property_lookup")
    filters = importlib.import_module("graphistry.compute.filter_by_dict")

    def forbidden(*args, **kwargs):
        pytest.fail("dense GPU cost decline must keep the canonical scan without candidate probes or CPU scalar execution")

    monkeypatch.setattr(lookup, "property_candidate_positions", forbidden)
    monkeypatch.setattr(filters, "_filter_native_property_scalar", forbidden)
    target = ExecutionTarget.GPU if gpu_target else ExecutionTarget.CPU
    with target_mode(target):
        expected = getattr(with_index_policy(indexed, "off"), method)({"account": 1}, engine="polars-gpu")
        actual = getattr(indexed, method)({"account": 1}, engine="polars-gpu")
    assert_frame_equal(getattr(actual, "_" + role), getattr(expected, "_" + role))
    assert_frame_equal(frame, original)

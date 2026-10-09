"""Explain distinguishes unsupported encodings from actual resident-index costs."""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry import n, gt
from graphistry.Engine import Engine, df_to_engine
from graphistry.compute.gfql.index import index_trace, with_index_policy
from graphistry.compute.predicates.is_in import IsIn
from graphistry.tests.compute.gfql.index.test_float_property_index import assert_same_frame
from graphistry.tests.compute.gfql.index.test_edge_property_index import frame_records


@pytest.fixture(params=["pandas", "polars", "cudf", "polars-gpu"])
def engine(request):
    if request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
    elif request.param != "pandas":
        pytest.importorskip(request.param)
    return request.param


def graph(engine):
    ids = np.arange(400)
    nodes = pd.DataFrame({"id": ids, "value": ids, "group": ids % 2})
    edges = pd.DataFrame({"s": ids, "d": (ids + 1) % 400, "txn": ids})
    return graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id").edges(df_to_engine(edges, Engine(engine)), "s", "d")


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("query", ["MATCH (a) WHERE a.value > 7 RETURN a.id AS id", [n({"value": gt(7)})], [n({"value": IsIn([float("nan")])})]])
def test_uncovered_property_predicates_report_stable_reason_and_preserve_rows(engine, query, check_receipt):
    base = graph(engine)
    indexed = base.create_index("node_prop", column="value", engine=engine)
    scan = base.gfql(query, engine=engine, index_policy="off")._nodes
    for policy in ["use", "force"]:
        assert_same_frame(indexed.gfql(query, engine=engine, index_policy=policy)._nodes, scan, engine)
        report = indexed.gfql_explain(query, engine=engine, index_policy=policy)
        assert report["error"] is None
        assert not report["used_index"]
        if check_receipt:
            assert report["decision_code"] == "not_index_coverable"
            decline = [s for s in report["steps"] if s.get("op") == "property_lookup"]
            assert decline and all(s["decision_code"] == "not_index_coverable" for s in decline)
            assert all(s["index_kind"] == "node_prop" and s["column"] == "value" for s in decline)
            assert all("est_result_rows" not in s for s in decline)
    assert indexed.gfql_explain(query, engine=engine, index_policy="off")["decision_code"] == "policy_off"


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("resident", [False, True])
def test_structurally_uncovered_predicate_is_independent_of_index_residency(engine, resident, check_receipt):
    g = graph(engine)
    if resident:
        g = g.create_index("node_prop", column="value", engine=engine)
    report = g.gfql_explain([n({"value": gt(7)})], engine=engine)
    assert report["error"] is None and not report["used_index"]
    assert_same_frame(g.gfql([n({"value": gt(7)})], engine=engine)._nodes, g.gfql([n({"value": gt(7)})], engine=engine, index_policy="off")._nodes, engine)
    if check_receipt:
        assert report["decision_code"] == "not_index_coverable"


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
def test_covered_predicate_with_uncovered_residual_still_reports_real_gather(engine, check_receipt):
    g = graph(engine).create_index("node_prop", column="value", engine=engine)
    query = "MATCH (a) WHERE a.value = 7 AND a.group > 0 RETURN a.id AS id"
    report = g.gfql_explain(query, engine=engine)
    if check_receipt:
        assert report["used_index"] and report["decision_code"] == "index_selected"
    assert [r["id"] for r in frame_records(g.gfql(query, engine=engine)._nodes)] == [7]


def test_property_scan_cost_identifies_actual_costed_index(engine):
    g = graph(engine).create_index("node_prop", column="group", engine=engine)
    with index_trace() as steps:
        actual = g.filter_nodes_by_dict({"group": 1}, engine=engine)._nodes
    scan = with_index_policy(g, "off").filter_nodes_by_dict({"group": 1}, engine=engine)._nodes
    assert_same_frame(actual, scan, engine)
    costs = [s for s in steps if s.get("decision_code") == "scan_cost"]
    assert costs and all(s["index_kind"] == "node_prop" and s["est_result_rows"] == 200 for s in costs)
    assert not any(s.get("path") == "index" for s in steps)


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("index-hop"))])
def test_adjacency_scan_cost_identifies_actual_costed_index(engine, check_receipt):
    g = graph(engine).create_index("edge_out_adj", engine=engine).create_index("node_id", engine=engine)
    report = g.gfql_explain("MATCH (a)-[e {txn: 7}]->(b) RETURN a, b", engine=engine)
    query = "MATCH (a)-[e {txn: 7}]->(b) RETURN a, b"
    assert_same_frame(g.gfql(query, engine=engine)._nodes, g.gfql(query, engine=engine, index_policy="off")._nodes, engine)
    if check_receipt:
        costs = [s for s in report["steps"] if s.get("decision_code") == "scan_cost"]
        assert costs and all(s["index_kind"] == "edge_out_adj" and s["n_keys"] > 0 for s in costs)
        assert all("seed_deg_sum" in s and "threshold_frac" in s for s in costs)


def test_decline_trace_does_not_change_structured_residual_errors(engine):
    from graphistry.compute.exceptions import GFQLSchemaError
    g = graph(engine).create_index("node_prop", column="value", engine=engine)
    errors = []
    for policy in ["off", "use", "force"]:
        with pytest.raises(GFQLSchemaError) as caught:
            with index_trace():
                with_index_policy(g, policy).filter_nodes_by_dict({"value": gt(7), "group": "bad"}, engine=engine)
        errors.append((caught.value.code, caught.value.context["field"]))
    assert errors[0] == errors[1] == errors[2]


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("cypher", [False, True])
def test_unindexable_boolean_storage_uses_existing_owner_receipt(engine, cypher, check_receipt):
    from graphistry.compute.gfql.index.errors import GfqlIndexNotImplementedError
    ids = np.arange(400)
    nodes = df_to_engine(pd.DataFrame({"id": ids, "active": ids % 2 == 0}), Engine(engine))
    g = graph(engine).nodes(nodes, "id")
    with pytest.raises(GfqlIndexNotImplementedError):
        g.create_index("node_prop", column="active", engine=engine)
    query = "MATCH (a {active: true}) RETURN a.id AS id" if cypher else [n({"active": True})]
    scan = g.gfql(query, engine=engine, index_policy="off")._nodes
    assert len(scan) == 200
    for policy in ["use", "force"]:
        report = g.gfql_explain(query, engine=engine, index_policy=policy)
        assert report["error"] is None and not report["used_index"]
        assert_same_frame(g.gfql(query, engine=engine, index_policy=policy)._nodes, scan, engine)
        if check_receipt:
            assert report["decision_code"] == "not_index_coverable"
            owners = [s for s in report["steps"] if s.get("op") != "fast_path"]
            assert len(owners) == 1
            assert owners[0]["op"] == "indexed_traversal"
            assert owners[0]["column"] == "active" and owners[0]["index_kind"] == "node_prop"
            assert owners[0]["reason"] == ("index_missing" if cypher else "no_valid_resident_index")
            assert owners[0]["hop_count"] == 0
            assert "est_result_rows" not in owners[0]
    assert g.gfql_explain(query, engine=engine, index_policy="off")["decision_code"] == "policy_off"
    assert g._nodes is nodes


@pytest.mark.parametrize("storage", ["boolean", "binary", "date", "mixed"])
@pytest.mark.parametrize("role", ["nodes", "edges"])
def test_direct_filter_reports_unsupported_storage_without_fictitious_cost(engine, storage, role):
    from datetime import date
    from graphistry.compute.gfql.index.errors import GfqlIndexNotImplementedError
    if storage == "mixed" and engine != "pandas":
        pytest.skip("Heterogeneous object columns are pandas storage")
    if storage in ("binary", "date") and engine == "cudf":
        pytest.skip("cuDF has no binary/date column distinct from supported string/timestamp storage")
    values, value = {
        "boolean": ([True, False] * 200, True),
        "binary": ([b"a", b"b"] * 200, b"a"),
        "date": ([date(2025, 1, 1), date(2025, 1, 2)] * 200, date(2025, 1, 1)),
        "mixed": ([1, "x"] * 200, "x"),
    }[storage]
    ids = np.arange(400)
    source = pd.DataFrame({"id": ids, "s": ids, "d": (ids + 1) % 400, "key": values})
    frame = df_to_engine(source, Engine(engine))
    g = graph(engine)
    g = g.nodes(frame, "id") if role == "nodes" else g.edges(frame, "s", "d")
    kind = "node_prop" if role == "nodes" else "edge_prop"
    with pytest.raises(GfqlIndexNotImplementedError):
        g.create_index(kind, column="key", engine=engine)
    method = "filter_nodes_by_dict" if role == "nodes" else "filter_edges_by_dict"
    scan = getattr(with_index_policy(g, "off"), method)({"key": value}, engine=engine)
    for policy in ["use", "force"]:
        with index_trace() as steps:
            out = getattr(with_index_policy(g, policy), method)({"key": value}, engine=engine)
        actual, expected = (out._nodes, scan._nodes) if role == "nodes" else (out._edges, scan._edges)
        assert len(actual) == 200
        assert_same_frame(actual, expected, engine)
        assert len(steps) == 1 and steps[0]["decision_code"] == "not_index_coverable"
        assert steps[0]["column"] == "key" and steps[0]["index_kind"] == kind
        assert "est_result_rows" not in steps[0]


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("cypher", [False, True])
def test_supported_missing_or_stale_indexes_do_not_report_unsupported_storage(engine, stale, cypher, check_receipt):
    g = graph(engine)
    if stale:
        g = g.create_index("node_prop", column="value", engine=engine)
        rebound = df_to_engine(pd.DataFrame({"id": np.arange(400), "value": np.arange(400), "group": np.arange(400) % 2}), Engine(engine))
        g = g.nodes(rebound, "id")
    query = "MATCH (a {value: 7}) RETURN a.id AS id" if cypher else [n({"value": 7})]
    report = g.gfql_explain(query, engine=engine)
    assert report["error"] is None and not report["used_index"]
    assert [r["id"] for r in frame_records(g.gfql(query, engine=engine)._nodes)] == [7]
    if check_receipt:
        assert report["decision_code"] == "index_path_unavailable"
        assert all(s.get("decision_code") != "not_index_coverable" for s in report["steps"])


def test_unsupported_storage_diagnostics_do_not_run_on_untraced_queries(engine, monkeypatch):
    from graphistry.compute.gfql.index import property_keys
    def unexpected_classifier(*args, **kwargs):
        pytest.fail("Coverage classification must only run while tracing")
    monkeypatch.setattr(property_keys, "uncovered_property_column", unexpected_classifier)
    monkeypatch.setattr("graphistry.compute.gfql_fast_paths._node_lookup_scan_reason", unexpected_classifier)
    monkeypatch.setattr("graphistry.compute.gfql.index.property_lookup.uncovered_property_column", unexpected_classifier)
    ids = np.arange(400)
    g = graph(engine).nodes(df_to_engine(pd.DataFrame({"id": ids, "active": ids % 2 == 0}), Engine(engine)), "id")
    assert len(g.gfql([n({"active": True})], engine=engine)._nodes) == 200
    assert len(g.gfql("MATCH (a {active: true}) RETURN a.id AS id", engine=engine)._nodes) == 200
    assert len(g.filter_nodes_by_dict({"active": True}, engine=engine)._nodes) == 200


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("resident", [False, True])
def test_supported_conjunct_prevents_false_unsupported_storage_reason(engine, resident, check_receipt):
    ids = np.arange(400)
    g = graph(engine).nodes(df_to_engine(pd.DataFrame({"id": ids, "value": ids, "active": ids % 2 == 0}), Engine(engine)), "id")
    if resident:
        g = g.create_index("node_prop", column="value", engine=engine)
    query = [n({"value": 7, "active": False})]
    report = g.gfql_explain(query, engine=engine)
    assert report["error"] is None
    assert [r["id"] for r in frame_records(g.gfql(query, engine=engine)._nodes)] == [7]
    if check_receipt:
        assert report["decision_code"] == ("index_selected" if resident else "index_path_unavailable")
        assert report["used_index"] == resident
        assert all(s.get("decision_code") != "not_index_coverable" for s in report["steps"])


def test_mixed_object_decline_preserves_canonical_type_error():
    from graphistry.compute.exceptions import GFQLSchemaError
    g = graphistry.nodes(pd.DataFrame({"id": [0, 1], "key": [1, "x"]}), "id")
    errors = []
    for policy in ["off", "use", "force"]:
        with pytest.raises(GFQLSchemaError) as caught:
            with index_trace():
                with_index_policy(g, policy).filter_nodes_by_dict({"key": 1})
        errors.append((caught.value.code, caught.value.context["field"], caught.value.context["value"]))
    assert errors[0] == errors[1] == errors[2]


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("cypher", [False, True])
def test_resident_encoding_decline_is_explained_without_inventing_a_cost(engine, cypher, check_receipt):
    g = graph(engine).create_index("node_prop", column="value", engine=engine)
    query = "MATCH (a {value: 7.0}) RETURN a.id AS id" if cypher else [n({"value": 7.0})]
    for policy in ["use", "force"]:
        report = g.gfql_explain(query, engine=engine, index_policy=policy)
        assert report["error"] is None and not report["used_index"]
        actual = g.gfql(query, engine=engine, index_policy=policy)._nodes
        assert len(actual) == 1
        assert_same_frame(actual, g.gfql(query, engine=engine, index_policy="off")._nodes, engine)
        if check_receipt:
            assert report["decision_code"] == "not_index_coverable"
            assert not any(s.get("decision_code") == "scan_cost" or s.get("reason") == "cost_gate" for s in report["steps"])


@pytest.mark.parametrize("covered_resident", [False, True])
def test_one_encoding_decline_does_not_obscure_another_coverable_predicate(engine, covered_resident):
    g = graph(engine).create_index("node_prop", column="value", engine=engine)
    if covered_resident:
        g = g.create_index("node_prop", column="group", engine=engine)
    with index_trace() as steps:
        out = with_index_policy(g, "force").filter_nodes_by_dict({"value": 7.0, "group": 1}, engine=engine)
    assert len(out._nodes) == 1
    assert_same_frame(out._nodes, with_index_policy(g, "off").filter_nodes_by_dict({"value": 7.0, "group": 1}, engine=engine)._nodes, engine)
    assert all(s.get("decision_code") != "not_index_coverable" for s in steps)
    if covered_resident:
        assert len(steps) == 1 and steps[0]["decision_code"] == "index_selected"
        assert steps[0]["column"] == "group" and steps[0]["est_result_rows"] == 200


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("resident_node_id", [False, True])
def test_alternate_node_id_kind_prevents_false_unsupported_verdict(engine, resident_node_id, check_receipt):
    g = graph(engine).create_index("node_prop", column="id", engine=engine)
    if resident_node_id:
        g = g.create_index("node_id", engine=engine)
    query = [n({"id": 7.0})]
    report = g.gfql_explain(query, engine=engine)
    assert report["error"] is None
    actual = g.gfql(query, engine=engine)._nodes
    assert len(actual) == 1
    assert_same_frame(actual, g.gfql(query, engine=engine, index_policy="off")._nodes, engine)
    if check_receipt:
        assert report["used_index"] == resident_node_id
        assert report["decision_code"] == ("index_selected" if resident_node_id else "index_path_unavailable")
        assert all(s.get("decision_code") != "not_index_coverable" for s in report["steps"])
    with index_trace() as steps:
        assert len(g.filter_nodes_by_dict({"id": 7.0}, engine=engine)._nodes) == 1
    assert all(s.get("decision_code") != "not_index_coverable" for s in steps)


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("null_member", [None, float("nan")])
def test_null_membership_capability_gap_does_not_require_resident_index(engine, null_member, check_receipt):
    ids = np.arange(400)
    values = pd.Series(np.arange(400), dtype="Int64").mask(ids % 2 == 0)
    g = graph(engine).nodes(df_to_engine(pd.DataFrame({"id": ids, "value": values}), Engine(engine)), "id")
    query = [n({"value": IsIn([7, null_member])})]
    try:
        scan = g.gfql(query, engine=engine, index_policy="off")._nodes
    except (NotImplementedError, TypeError) as error:
        for policy in ["use", "force"]:
            with pytest.raises(type(error)):
                g.gfql(query, engine=engine, index_policy=policy)
        report = g.gfql_explain(query, engine=engine)
        assert report["error"] is not None and not report["used_index"]
    else:
        for policy in ["use", "force"]:
            actual = g.gfql(query, engine=engine, index_policy=policy)._nodes
            assert 7 in [r["id"] for r in frame_records(actual)]
            assert_same_frame(actual, scan, engine)
        report = g.gfql_explain(query, engine=engine)
        assert report["error"] is None and not report["used_index"]
    if check_receipt:
        assert report["decision_code"] == "not_index_coverable"
        assert all("est_result_rows" not in s for s in report["steps"])


@pytest.mark.parametrize("actual_engine", ["pandas", "cudf"])
@pytest.mark.parametrize("requested_engine", ["pandas", "cudf"])
def test_direct_trace_preserves_mismatched_requested_engine_filter(actual_engine, requested_engine):
    if "cudf" in (actual_engine, requested_engine):
        pytest.importorskip("cudf")
    source = df_to_engine(pd.DataFrame({"id": np.arange(400), "value": ["x", "y"] * 200}), Engine(actual_engine))
    g = graphistry.nodes(source, "id")
    ordinary = g.filter_nodes_by_dict({"value": "x"}, engine=requested_engine)._nodes
    with index_trace() as steps:
        traced = g.filter_nodes_by_dict({"value": "x"}, engine=requested_engine)._nodes
    assert len(traced) == 200
    assert_same_frame(traced, ordinary, requested_engine)
    assert not steps and g._nodes is source


@pytest.mark.parametrize("check_receipt", [False, pytest.param(True, marks=pytest.mark.route_engaged("native-fast", "polars-single-node", "polars-plain", "cypher-fast"))])
@pytest.mark.parametrize("cypher", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_shared_polars_gpu_sidecar_decline_matches_real_lookup_validity(cypher, stale, check_receipt):
    pl = pytest.importorskip("polars")
    g = graph("polars").create_index("node_prop", column="value", engine="polars-gpu")
    if stale:
        g = g.nodes(pl.DataFrame({"id": np.arange(400), "value": np.arange(400)}), "id")
    query = "MATCH (a {value: 7.0}) RETURN a.id AS id" if cypher else [n({"value": 7.0})]
    for policy in ["use", "force"]:
        actual = g.gfql(query, engine="polars", index_policy=policy)._nodes
        assert len(actual) == 1
        assert_same_frame(actual, g.gfql(query, engine="polars", index_policy="off")._nodes, "polars")
        report = g.gfql_explain(query, engine="polars", index_policy=policy)
        assert report["error"] is None and not report["used_index"]
        if check_receipt:
            assert report["decision_code"] == ("index_path_unavailable" if stale else "not_index_coverable")
            if stale and cypher:
                assert report["decision_reason"] == "index_stale"
            else:
                assert report["decision_reason"] != "index_stale"

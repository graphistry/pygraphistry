"""String dictionary probes preserve public filtering, traces and owned results."""
import pytest

import graphistry
from graphistry.compute.gfql.index import index_trace, with_index_policy
from graphistry.compute.predicates.is_in import IsIn

cudf = pytest.importorskip("cudf")
cp = pytest.importorskip("cupy")
from cudf.testing import assert_frame_equal


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("values", [
    ["a", "a", "é", "é", "🙂", "", None, "z"],
    ["a"] * 8,
    [None] * 8,
    [],
])
@pytest.mark.parametrize("predicate", [
    "a", "missing", "", "é", "é", "🙂", ["a", "z", "a"], IsIn(["é", "missing"]), [],
])
@pytest.mark.parametrize("policy", ["off", "use", "force"])
def test_native_string_dictionary_public_filters_preserve_scan_and_ownership(role, kind, values, predicate, policy):
    frame = cudf.DataFrame({"id": cp.arange(len(values)), "v": cudf.Series(values, dtype="str")})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    original = frame.copy(deep=True)
    indexed = with_index_policy(base.create_index(kind, column="v", engine="cudf"), policy)
    method = "filter_" + role + "_by_dict"
    expected = getattr(getattr(base, method)({"v": predicate}, engine="cudf"), "_" + role)
    actual = getattr(getattr(indexed, method)({"v": predicate}, engine="cudf"), "_" + role)
    assert_frame_equal(actual, expected)
    if len(actual):
        actual["id"].values[0] = -99
        actual["v"] = "changed"
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("predicate,count", [("a", 2), ("missing", 0), ("é", 1), ("", 1)])
def test_native_string_dictionary_force_trace_reports_actual_gather(role, kind, predicate, count):
    frame = cudf.DataFrame({"id": cp.arange(8), "v": ["a", "a", "é", "é", "🙂", "", None, "z"]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = with_index_policy(base.create_index(kind, column="v", engine="cudf"), "force")
    with index_trace() as steps:
        actual = getattr(getattr(indexed, "filter_" + role + "_by_dict")(
            {"v": predicate}, engine="cudf"), "_" + role)
    decisions = [step for step in steps if step.get("op") == "property_lookup" and step.get("role") == role]
    assert len(decisions) == 1
    assert decisions[0]["path"] == "index"
    assert decisions[0]["est_result_rows"] == count
    assert len(actual) == count


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("predicate", ["a", "é", "missing"])
def test_native_string_dictionary_query_needs_no_host_string_export(role, kind, predicate, monkeypatch):
    frame = cudf.DataFrame({"id": cp.arange(8), "v": ["a", "a", "é", "é", "🙂", "", None, "z"]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = with_index_policy(base.create_index(kind, column="v", engine="cudf"), "force")
    method = "filter_" + role + "_by_dict"
    expected = getattr(getattr(base, method)({"v": predicate}, engine="cudf"), "_" + role)

    def unavailable_host_export(*args, **kwargs):
        pytest.fail("A native GPU dictionary query must not export string keys")

    with monkeypatch.context() as patch:
        patch.setattr(cudf.Series, "to_arrow", unavailable_host_export)
        actual = getattr(getattr(indexed, method)({"v": predicate}, engine="cudf"), "_" + role)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("predicate", ["a", "missing"])
def test_large_string_dictionary_preserves_scan_rows_and_force_trace(role, kind, predicate):
    frame = cudf.DataFrame({"id": [0, 1, 2], "v": ["z" * (9 * 1024 * 1024), "a", None]})
    base = graphistry.nodes(frame, "id").edges(frame, "id", "id")
    indexed = with_index_policy(base.create_index(kind, column="v", engine="cudf"), "force")
    method = "filter_" + role + "_by_dict"
    expected = getattr(getattr(base, method)({"v": predicate}, engine="cudf"), "_" + role)
    with index_trace() as steps:
        actual = getattr(getattr(indexed, method)({"v": predicate}, engine="cudf"), "_" + role)
    assert_frame_equal(actual, expected)
    assert any(step.get("op") == "property_lookup" and step.get("role") == role
               and step.get("path") == "index" and step.get("est_result_rows") == len(expected)
               for step in steps)

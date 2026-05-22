import pandas as pd
import pytest

from graphistry.compute.ast import ASTLet, n
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLValidationError
from graphistry.tests.test_compute import CGFull


def _mk_graph():
    nodes_df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label__Person": [True, True, False],
            "name": ["Alice", "Bob", "Corp"],
            "score": [3, 1, 2],
        }
    )
    edges_df = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"], "type": ["KNOWS", "WORKS_AT"]})
    return CGFull().nodes(nodes_df, "id").edges(edges_df, "s", "d")


def test_gfql_validate_exists_on_public_api():
    g = CGFull()
    assert hasattr(g, "gfql_validate")
    assert callable(g.gfql_validate)


def test_gfql_validate_chain_success():
    g = _mk_graph()
    report = g.gfql_validate([n({"name": "Alice"})])
    assert report["ok"] is True
    assert report["language"] == "gfql"
    assert report["query_type"] == "chain"
    assert report["diagnostics"] == []


def test_gfql_validate_chain_failure_collect_all():
    g = _mk_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate([n({"missing_col": "x"})], collect_all=True)
    assert exc_info.value.code == ErrorCode.E301
    diagnostics = exc_info.value.context.get("diagnostics")
    assert isinstance(diagnostics, list) and diagnostics
    assert diagnostics[0]["code"] == ErrorCode.E301


def test_gfql_validate_cypher_success():
    g = _mk_graph()
    report = g.gfql_validate(
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name DESC LIMIT $top_n",
        params={"top_n": 2},
    )
    assert report["ok"] is True
    assert report["language"] == "cypher"
    assert report["query_type"] == "chain"
    assert report["compiled_kind"] == "query"
    assert report["diagnostics"] == []


def test_gfql_validate_cypher_default_reports_schema_errors():
    g = _mk_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate("MATCH (p:Employee) RETURN p.name AS name")
    assert exc_info.value.code == ErrorCode.E301


def test_gfql_validate_cypher_can_disable_strict_schema_checks():
    g = _mk_graph()
    report = g.gfql_validate("MATCH (p:Employee) RETURN p.name AS name", strict=False)
    assert report["ok"] is True
    assert report["language"] == "cypher"
    assert report["diagnostics"] == []


def test_gfql_validate_treats_all_strings_as_cypher():
    g = _mk_graph()
    with pytest.raises(GFQLSyntaxError) as exc_info:
        g.gfql_validate("hello world not cypher")
    assert exc_info.value.code == ErrorCode.E107
    assert "Got str" not in str(exc_info.value)


def test_gfql_validate_does_not_execute_query_operators(monkeypatch):
    g = _mk_graph()

    def _should_not_run(*args, **kwargs):
        raise AssertionError("execution path should not be called by gfql_validate")

    monkeypatch.setattr("graphistry.compute.chain.chain", _should_not_run)
    report = g.gfql_validate([n({"name": "Alice"})])
    assert report["ok"] is True


def test_gfql_validate_let_success():
    g = _mk_graph()
    query = ASTLet({"people": Chain([n({"name": "Alice"})])})
    report = g.gfql_validate(query)
    assert report["ok"] is True
    assert report["language"] == "gfql"
    assert report["query_type"] == "dag"
    assert report["diagnostics"] == []


def test_gfql_validate_let_schema_failure():
    g = _mk_graph()
    query = ASTLet({"people": Chain([n({"missing_col": "x"})])})
    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate(query, collect_all=True)
    assert exc_info.value.code == ErrorCode.E301
    assert exc_info.value.context.get("query_type") == "dag"


def test_gfql_validate_exception_payload_is_llm_friendly():
    g = _mk_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql_validate([n({"missing_col": "x"})], collect_all=True)
    payload = exc_info.value.to_dict()
    assert payload["code"] == ErrorCode.E301
    assert payload["query_type"] == "chain"
    assert payload["language"] == "gfql"
    diagnostics = payload.get("diagnostics")
    assert isinstance(diagnostics, list) and diagnostics
    assert diagnostics[0]["code"] == ErrorCode.E301


def test_gfql_validate_chain_without_bound_tables_is_structural_only():
    g = CGFull()
    report = g.gfql_validate([n({"missing_col": "x"})])
    assert report["ok"] is True
    assert report["language"] == "gfql"
    assert report["query_type"] == "chain"
    assert report["diagnostics"] == []

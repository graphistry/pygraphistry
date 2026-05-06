import pandas as pd

from graphistry.compute.ast import ASTLet, n
from graphistry.compute.chain import Chain
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
    report = g.gfql_validate([n({"missing_col": "x"})], collect_all=True)
    assert report["ok"] is False
    assert report["language"] == "gfql"
    assert report["diagnostics"]
    assert report["diagnostics"][0]["code"] == "column-not-found"


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
    report = g.gfql_validate("MATCH (p:Employee) RETURN p.name AS name")
    assert report["ok"] is False
    assert report["language"] == "cypher"
    assert report["diagnostics"]
    assert report["diagnostics"][0]["code"] == "column-not-found"


def test_gfql_validate_cypher_can_disable_strict_schema_checks():
    g = _mk_graph()
    report = g.gfql_validate("MATCH (p:Employee) RETURN p.name AS name", strict=False)
    assert report["ok"] is True
    assert report["language"] == "cypher"
    assert report["diagnostics"] == []


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
    report = g.gfql_validate(query, collect_all=True)
    assert report["ok"] is False
    assert report["language"] == "gfql"
    assert report["query_type"] == "dag"
    assert report["diagnostics"]
    assert report["diagnostics"][0]["code"] == "column-not-found"

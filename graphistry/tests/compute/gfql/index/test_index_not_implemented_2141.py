"""Index support GFQL does not have yet raises NotImplementedError naming #2141.

Unsupported property column types are tracked work, not caller mistakes and not
malformed DDL. Every entry point (Cypher DDL, fused DDL, the wire op, ``create_index``) says so the same way,
while staying a ``ValueError`` for existing callers and staying skippable for the convenience builders.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.index import (
    NODE_PROP, GfqlIndexNotImplementedError, GfqlIndexUnsupportedError, create_index, get_registry,
)
from graphistry.compute.gfql.index.cypher_ddl import parse_index_ddl
from graphistry.compute.gfql.index.errors import INDEX_SUPPORT_ISSUE_URL
from graphistry.compute.gfql.index.wire import CreateIndex

ISSUE = "issues/2141"


def _graph() -> graphistry.Plottable:
    nodes = pd.DataFrame({
        "id": [0, 1, 2],
        "account_number": [48211, 48213, 48215],
        "email": ["a@x", "b@x", "c@x"],
        "score": [0.5, 1.5, 2.5],
        "active": [True, False, True],
        "binary": [b"a", b"b", b"c"],
        "maybe": pd.Series([1, None, 3], dtype="Int64"),
    })
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 2], "txn_id": [9001, 9002]})
    return graphistry.edges(edges, "s", "d").nodes(nodes, "id")


def _assert_tracked(excinfo: pytest.ExceptionInfo) -> None:
    assert isinstance(excinfo.value, NotImplementedError)
    assert isinstance(excinfo.value, ValueError)
    assert ISSUE in str(excinfo.value)


@pytest.mark.parametrize("ddl", [
    "CREATE GFQL INDEX FOR edge_prop ON (txn_id)",
    "CREATE GFQL INDEX IF NOT EXISTS FOR edge_prop ON txn_id",
    "DROP GFQL INDEX IF EXISTS FOR edge_prop ON (txn_id)",
])
def test_edge_prop_ddl_is_now_implemented(ddl):
    op = parse_index_ddl(ddl)
    assert op is not None
    assert op.kind == "edge_prop" and op.column == "txn_id"


def test_edge_prop_in_a_fused_query_builds_and_runs_the_match():
    out = _graph().gfql("CREATE GFQL INDEX FOR edge_prop ON (txn_id); MATCH (a)-[e {txn_id: 9001}]->(b) RETURN b.id AS id")
    assert out._nodes["id"].tolist() == [1]


def test_edge_prop_through_the_wire_and_the_python_api():
    op = CreateIndex.from_json({"type": "CreateIndex", "kind": "edge_prop", "column": "txn_id"})
    assert op.kind == "edge_prop"
    indexed = create_index(_graph(), "edge_prop", column="txn_id")
    assert set(get_registry(indexed).edge_props) == {"txn_id"}


@pytest.mark.parametrize("ddl", ["CREATE GFQL INDEX FOR bogus", "CREATE GFQL INDEX FOR node_prop ON ("])
def test_a_genuinely_unknown_or_broken_statement_stays_malformed(ddl):
    with pytest.raises(ValueError, match="Malformed GFQL INDEX DDL") as excinfo:
        parse_index_ddl(ddl)
    assert not isinstance(excinfo.value, NotImplementedError)


def test_unknown_kind_through_the_python_api_stays_a_plain_value_error():
    with pytest.raises(ValueError, match="Unknown GFQL index kind") as excinfo:
        create_index(_graph(), "bogus", column="x")  # type: ignore[arg-type]
    assert not isinstance(excinfo.value, NotImplementedError)


@pytest.mark.parametrize("column", ["active", "binary"])
def test_unsupported_property_columns_are_not_implemented(column):
    with pytest.raises(GfqlIndexNotImplementedError) as excinfo:
        create_index(_graph(), NODE_PROP, column=column)
    _assert_tracked(excinfo)
    assert isinstance(excinfo.value, GfqlIndexUnsupportedError)
    assert "dtype" in str(excinfo.value)


def test_supported_property_columns_build_and_the_builder_skips_the_rest():
    g = create_index(_graph(), NODE_PROP, column="account_number")
    assert get_registry(g).node_prop_cols() == ("account_number",)
    g2 = _graph().gfql_index_node_props(["email", "score", "maybe", "account_number", "active", "binary"])
    assert get_registry(g2).node_prop_cols() == ("account_number", "email", "maybe", "score")


def test_the_tracking_url_is_the_issue():
    assert INDEX_SUPPORT_ISSUE_URL.endswith(ISSUE)

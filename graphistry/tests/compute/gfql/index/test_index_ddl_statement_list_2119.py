"""``CREATE GFQL INDEX ...; <query>`` builds the indexes and runs the query in one gfql() call (#2119).

Index DDL used to be matched whole-string, so building and querying were two ``gfql()``
calls. Leading DDL statements separated by ``;`` are now applied in order and the remaining
text runs on the indexed graph; the caller's ``g`` is unchanged. A lone DDL statement, and
``SHOW GFQL INDEXES`` on its own, keep their existing paths.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.index.cypher_ddl import parse_index_ddl_prefix, split_top_level_statements
from graphistry.compute.gfql.index.wire import CreateIndex, DropIndex


def _graph():
    rng = np.random.default_rng(7)
    n_nodes, n_edges = 2000, 10000
    edges = pd.DataFrame({"src": rng.integers(0, n_nodes, n_edges), "dst": rng.integers(0, n_nodes, n_edges)})
    nodes = pd.DataFrame({"id": np.arange(n_nodes), "name": [f"n;{i}" for i in range(n_nodes)]})
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id"), edges


_DDL = "CREATE GFQL INDEX FOR edge_out_adj; CREATE GFQL INDEX FOR node_id; "


_FUSED = _DDL + "MATCH (a {id: 5})-[e]->(b) RETURN b.id AS id"


def test_leading_ddl_then_query_returns_the_scan_rows_and_leaves_the_caller_untouched():
    g, edges = _graph()
    out = g.gfql(_FUSED, engine="pandas")
    assert sorted(out._nodes["id"].tolist()) == sorted(edges[edges["src"] == 5]["dst"].tolist())
    assert g.show_indexes().empty  # the caller's graph is untouched


@pytest.mark.route_engaged("native-fast", "index-hop")
def test_leading_ddl_then_query_takes_the_indexes_it_built():
    g, _ = _graph()
    assert g.gfql_explain(_FUSED, engine="pandas")["used_index"] is True


def test_ddl_only_list_returns_the_indexed_graph():
    g, _ = _graph()
    built = g.gfql(_DDL.rstrip("; "))
    kinds = sorted(built.show_indexes()["kind"].tolist())
    assert kinds == ["edge_out_adj", "node_id"]
    assert g.show_indexes().empty


def test_semicolon_inside_a_string_literal_is_not_a_statement_break():
    g, _ = _graph()
    out = g.gfql("CREATE GFQL INDEX FOR node_id; MATCH (n) WHERE n.name = 'n;5' RETURN n.id AS id", engine="pandas")
    assert out._nodes["id"].tolist() == [5]


@pytest.mark.parametrize("query,needle", [
    ("MATCH (n) RETURN n.id AS id; CREATE GFQL INDEX FOR node_id", "must lead the statement list"),
    ("CREATE GFQL INDEX FOR node_id; MATCH (n) RETURN n.id AS id; DROP GFQL INDEX FOR node_id", "must lead the statement list"),
    ("SHOW GFQL INDEXES; MATCH (n) RETURN n.id AS id", "cannot be part of a statement list"),
    ("CREATE GFQL INDEX FOR bogus; MATCH (n) RETURN n.id AS id", "Malformed GFQL INDEX DDL"),
])
def test_misplaced_or_malformed_ddl_is_a_typed_error(query, needle):
    g, _ = _graph()
    with pytest.raises(ValueError, match=needle):
        g.gfql(query, engine="pandas")


def test_lone_statements_keep_their_paths():
    g, _ = _graph()
    assert len(g.gfql("CREATE GFQL INDEX FOR node_id").show_indexes()) == 1
    assert isinstance(g.gfql("SHOW GFQL INDEXES"), pd.DataFrame)
    # a query that merely mentions the DDL words inside a literal is not split or rejected
    assert g.gfql("MATCH (n) WHERE n.name = 'CREATE GFQL INDEX FOR x; y' RETURN n.id AS id", engine="pandas")._nodes.shape[0] == 0


def test_statement_splitter_and_prefix_parser_contract():
    assert split_top_level_statements("a; b ; ; c") == ["a", "b", "c"]
    assert split_top_level_statements("a; 'x;y'; [1;2]; {k: 'v;w'}") == ["a", "'x;y'", "[1;2]", "{k: 'v;w'}"]
    assert split_top_level_statements("a; \"q;r\"") == ["a", '"q;r"']
    ops, rest = parse_index_ddl_prefix("CREATE GFQL INDEX FOR node_id; DROP GFQL INDEX IF EXISTS FOR edge_out_adj; MATCH (n) RETURN n")
    assert isinstance(ops[0], CreateIndex) and ops[0].kind == "node_id"
    assert isinstance(ops[1], DropIndex) and ops[1].kind == "edge_out_adj" and ops[1].missing_ok
    assert rest == "MATCH (n) RETURN n"
    assert parse_index_ddl_prefix("CREATE GFQL INDEX FOR node_id") is None          # lone DDL: whole-string path
    assert parse_index_ddl_prefix("MATCH (n) RETURN n") is None                       # no DDL at all
    assert parse_index_ddl_prefix("CREATE GFQL INDEX FOR node_id;") is None          # trailing ; is still one statement

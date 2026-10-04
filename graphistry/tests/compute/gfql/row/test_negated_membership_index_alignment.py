"""Row-evaluator results stay row-aligned on a frame whose labels are not 0..n-1.

#2020 made the positive-IN pushdown apply its mask by position. ``NOT (x IN [...])`` goes
through the tri-valued NOT builder instead, which broadcasts on the table's labels and then
``.where``-aligns against positionally built masks; on a frame the index path hands back
(labels with gaps), the two disagreed and whole destination ids vanished. Every series the
expression evaluator returns now leaves on the table's own index, so each producer -- row
values, tri-valued masks, list comparison, quantifiers, list comprehensions, range, subscript,
slice, concatenation -- agrees with the label-aligned builders on any frame.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.row.pipeline import RowPipelineMixin, _gfql_expr_runtime_parser_bundle

Q_NOT = "MATCH (a)-[e]->(t) WHERE NOT (t.type IN ['person', 'company']) RETURN t.id AS id"
Q_EQ = "MATCH (a)-[e]->(t) WHERE t.type <> 'person' AND t.type <> 'company' RETURN t.id AS id"


def _graph(engine):
    nodes = pd.DataFrame({"id": ["a", "b", "c", "tx1", "tx2"],
                          "type": ["person", "person", "company", "transaction", "transaction"]})
    edges = pd.DataFrame({"src": ["a", "b", "a", "tx1", "tx2"], "dst": ["b", "c", "tx1", "tx2", "c"]})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id")


def _ids(res):
    nodes = res._nodes
    df = nodes.to_pandas() if hasattr(nodes, "to_pandas") else pd.DataFrame(nodes)
    return sorted(df["id"].tolist())


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_not_in_on_a_non_range_indexed_node_frame_matches_the_scalar_form(engine):
    g = _graph("pandas")
    nodes = g._nodes.iloc[[4, 2, 0, 3, 1]]  # labels out of order, no RangeIndex
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes = cudf.from_pandas(nodes)
        g = _graph("cudf")
    g = g.nodes(nodes)
    assert _ids(g.gfql(Q_NOT, engine=engine)) == _ids(g.gfql(Q_EQ, engine=engine)) == ["tx1", "tx2"]


def test_not_in_agrees_across_index_policies_on_the_index_path():
    # 30 nodes / 120 edges, seed 3001: the index path's hop returns a nodes frame whose
    # labels skip 5, and the forced lane dropped destinations 9, 28 and 29 before the fix.
    rng = np.random.default_rng(3001)
    n_nodes, n_edges = 30, 120
    edges = pd.DataFrame({"src": rng.integers(0, n_nodes, n_edges), "dst": rng.integers(0, n_nodes, n_edges)})
    g = graphistry.edges(edges, "src", "dst").nodes(pd.DataFrame({"id": np.arange(n_nodes)}), "id").gfql_index_all()
    q = "MATCH (a)-[e]->(b) WHERE NOT (b.id IN [3, 5, 10]) RETURN b.id AS id"
    truth = int((~edges["dst"].isin([3, 5, 10])).sum())
    assert truth == 111
    for policy in ("off", "use", "force"):
        assert len(g.gfql(q, engine="pandas", index_policy=policy)._nodes) == truth, policy


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_row_values_series_carries_the_table_index(engine):
    table = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 2, 7])  # gaps, like an index-path hop result
    if engine == "cudf":
        table = pytest.importorskip("cudf").from_pandas(table)
    out = RowPipelineMixin._gfql_series_from_row_values(RowPipelineMixin(), table, [True, False, None], "__t__")
    out = out.to_pandas() if hasattr(out, "to_pandas") else out
    assert list(out.index) == [0, 2, 7]
    assert list(out) == [True, False, None]


@pytest.mark.route_engaged("index-hop", "indexed-kernel")
def test_forced_policy_takes_the_index_on_the_gappy_graph():
    _, g = _gappy_graph()
    report = g.gfql_explain("MATCH (a)-[e]->(b) WHERE NOT (b.id IN [3, 5, 10]) RETURN b.id AS id",
                            engine="pandas", index_policy="force")
    assert report["used_index"] is True


def _gappy_graph():
    rng = np.random.default_rng(3001)
    n_nodes, n_edges = 30, 120
    edges = pd.DataFrame({"src": rng.integers(0, n_nodes, n_edges), "dst": rng.integers(0, n_nodes, n_edges)})
    nodes = pd.DataFrame({"id": np.arange(n_nodes),
                          "ts": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n_nodes), unit="D")})
    return edges, graphistry.edges(edges, "src", "dst").nodes(nodes, "id").gfql_index_all()


@pytest.mark.parametrize("label,query,truth_mask", [
    ("temporal IN", "MATCH (a)-[e]->(b) WHERE b.ts IN [datetime('2024-01-10T00:00:00')] RETURN b.id AS id", lambda d: d == 9),
    ("NOT temporal IN", "MATCH (a)-[e]->(b) WHERE NOT (b.ts IN [datetime('2024-01-10T00:00:00')]) RETURN b.id AS id", lambda d: d != 9),
    ("list =", "MATCH (a)-[e]->(b) WHERE [b.id] = [9] RETURN b.id AS id", lambda d: d == 9),
    ("list <>", "MATCH (a)-[e]->(b) WHERE [b.id] <> [9] RETURN b.id AS id", lambda d: d != 9),
    ("list <", "MATCH (a)-[e]->(b) WHERE [b.id] < [10] RETURN b.id AS id", lambda d: d < 10),
])
def test_evaluator_families_agree_across_index_policies_on_the_index_path(label, query, truth_mask):
    # Temporal IN and list comparison build a reset work frame and insert evaluator series into
    # it. On master the temporal cases returned 0 rows under 'force' (index engaged, every row
    # shifted past the gap); the list cases take the index too and pin the positional contract.
    edges, g = _gappy_graph()
    truth = int(truth_mask(edges["dst"]).sum())
    for policy in ("off", "use", "force"):
        assert len(g.gfql(query, engine="pandas", index_policy=policy)._nodes) == truth, (label, policy)


def test_positional_assign_keeps_rows_in_place_on_a_gappy_series():
    frame = pd.DataFrame({"x": range(5)})                       # RangeIndex 0..4 (a reset work frame)
    gappy = pd.Series(["v0", "v1", "v2", "v3", "v4"], index=[0, 1, 2, 4, 7])  # labels with gaps
    out = RowPipelineMixin._gfql_assign_positional(frame, c=gappy, k=range(5))
    assert list(out["c"]) == ["v0", "v1", "v2", "v3", "v4"]     # no NaN at the gap, no tail lost
    assert list(out["k"]) == [0, 1, 2, 3, 4]
    # a length mismatch is left to pandas' own alignment rather than silently truncated
    short = pd.Series([1, 2], index=[0, 1])
    assert out.shape[0] == RowPipelineMixin._gfql_assign_positional(frame, s=short).shape[0]


def test_on_table_index_leaves_unalignable_values_alone():
    table = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 2, 7])
    assert RowPipelineMixin._gfql_assign_positional(table, c=5)["c"].tolist() == [5, 5, 5]  # scalar: assigned as-is
    short = pd.Series([True, False])
    assert list(RowPipelineMixin._gfql_on_table_index(table, short).index) == [0, 1]  # length mismatch: untouched
    full = pd.Series([True, False, True])
    assert list(RowPipelineMixin._gfql_on_table_index(table, full).index) == [0, 2, 7]


@pytest.mark.parametrize("expr,truth", [
    ("NOT ([b.id] = [9])", lambda d: d != 9),                      # list compare under NOT
    ("NOT ([b.id, 1][0] = 9)", lambda d: d != 9),                  # subscript
    ("NOT any(x IN [b.id] WHERE x = 9)", lambda d: d != 9),        # quantifier
    ("NOT (size(range(0, b.id)) = 10)", lambda d: d != 9),         # range
    ("NOT ([x IN [b.id] | x][0] = 9)", lambda d: d != 9),          # list comprehension
    ("NOT (([b.id] + [1])[0] = 9)", lambda d: d != 9),             # concatenation
    ("[b.id] = [9] OR b.id = 3", lambda d: (d == 9) | (d == 3)),   # label-aligned OR builder
])
def test_every_producer_agrees_across_index_policies(expr, truth):
    # each shape returned 103 of 116 rows (or 5 of 9) under 'force' while the scan returned the
    # truth: the producer reset its index, the NOT/OR builder aligned by label.
    edges, g = _gappy_graph()
    q = f"MATCH (a)-[e]->(b) WHERE {expr} RETURN b.id AS id"
    expected = int(truth(edges["dst"]).sum())
    for policy in ("off", "use", "force"):
        assert len(g.gfql(q, engine="pandas", index_policy=policy)._nodes) == expected, (expr, policy)


def test_slice_compare_agrees_across_index_policies():
    _, g = _gappy_graph()
    q = "MATCH (a)-[e]->(b) WHERE [b.id, 2][0..1] = [9] RETURN b.id AS id"
    rows = {p: len(g.gfql(q, engine="pandas", index_policy=p)._nodes) for p in ("off", "use", "force")}
    assert rows["off"] == rows["use"] == rows["force"] == 4  # 'force' returned 0 before


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_the_evaluator_returns_series_on_the_table_index(engine):
    table = pd.DataFrame({"v": [9, 3, 9, 1]}, index=[0, 1, 3, 4])
    if engine == "cudf":
        table = pytest.importorskip("cudf").from_pandas(table)
    parser, _checker, _mod = _gfql_expr_runtime_parser_bundle()

    class _M(RowPipelineMixin):
        pass

    for expr in ("[v] = [9]", "NOT ([v] = [9])", "v IN [9]"):
        ok, value = _M()._gfql_eval_expr_ast(table, parser(expr))
        assert ok is True, expr
        index = value.index.to_pandas() if hasattr(value.index, "to_pandas") else value.index
        assert list(index) == [0, 1, 3, 4], expr

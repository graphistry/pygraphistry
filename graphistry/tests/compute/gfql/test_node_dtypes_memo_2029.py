"""#2029: node dtype reads (the object-column string-content gate) are memoized per frame.

The compile cache key asks for node dtypes on every string query; reading them scans every
object column's values, which on a wide node table cost more than the query. The memo keys on
frame identity plus a length/columns fingerprint (the resident indexes' contract).
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute import filter_by_dict as fbd
from graphistry.compute.gfql import node_dtypes_memo as memo
from graphistry.compute.gfql_unified import gfql_clear_caches


@pytest.fixture(autouse=True)
def _clean():
    memo.clear_node_dtypes_memo()
    yield
    memo.clear_node_dtypes_memo()


def _graph(rows=200):
    nodes = pd.DataFrame({"id": range(rows), "name": [f"n{i}" for i in range(rows)],
                          "mixed": [i if i % 2 else f"s{i}" for i in range(rows)], "type": "T"})
    edges = pd.DataFrame({"s": range(rows - 1), "d": range(1, rows)})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _count_scans(monkeypatch):
    calls = {"n": 0}
    real = fbd._object_column_holds_non_strings

    def spy(frame, column, dtype):
        calls["n"] += 1
        return real(frame, column, dtype)

    monkeypatch.setattr(fbd, "_object_column_holds_non_strings", spy)
    return calls


def test_second_read_of_the_same_frame_does_not_rescan(monkeypatch):
    g = _graph()
    calls = _count_scans(monkeypatch)
    first = fbd._read_node_dtypes(g, "pandas")
    scans = calls["n"]
    assert scans > 0 and "mixed" not in first and "name" in first  # the gate still decides
    again = fbd._read_node_dtypes(g, "pandas")
    assert again == first and calls["n"] == scans, "memo hit must not rescan columns"


def test_a_rebound_or_grown_frame_is_read_again(monkeypatch):
    g = _graph()
    calls = _count_scans(monkeypatch)
    fbd._read_node_dtypes(g, "pandas")
    scans = calls["n"]
    g2 = g.nodes(g._nodes.copy(), "id")  # new frame object, same shape
    fbd._read_node_dtypes(g2, "pandas")
    assert calls["n"] > scans, "a different frame object is a memo miss"
    scans = calls["n"]
    g._nodes.drop(g._nodes.index[-1], inplace=True)  # same object, length changed
    fbd._read_node_dtypes(g, "pandas")
    assert calls["n"] > scans, "a changed fingerprint is a memo miss"


def test_engines_have_separate_entries_and_clear_caches_empties_the_memo(monkeypatch):
    pytest.importorskip("polars")
    g = _graph()
    calls = _count_scans(monkeypatch)
    fbd._read_node_dtypes(g, "pandas")
    fbd._read_node_dtypes(g, "polars")
    assert len(memo._NODE_DTYPES_MEMO) == 2
    gfql_clear_caches()
    assert len(memo._NODE_DTYPES_MEMO) == 0
    before = calls["n"]
    fbd._read_node_dtypes(g, "pandas")
    assert calls["n"] > before


def test_string_queries_hit_the_memo_end_to_end(monkeypatch):
    g = _graph()
    calls = _count_scans(monkeypatch)
    q = "MATCH (a {id: 3}) RETURN a.name AS name"
    first = g.gfql(q, engine="pandas")
    scans = calls["n"]
    assert scans > 0
    second = g.gfql(q, engine="pandas")
    assert calls["n"] == scans
    assert first._nodes["name"].tolist() == second._nodes["name"].tolist() == ["n3"]

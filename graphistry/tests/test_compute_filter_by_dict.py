import pandas as pd
import pytest
from functools import lru_cache

from graphistry.compute.ast import is_in, IsIn
from graphistry.compute.filter_by_dict import filter_by_dict
from graphistry.tests.test_compute import CGFull

@lru_cache(maxsize=1)
def hops_graph():

    #| node, type, i, v | 
    nodes_df = (pd.DataFrame([
        {'node': 'a'},
        {'node': 'b'},
        {'node': 'c'},
        {'node': 'd'},
        {'node': 'e'},
        {'node': 'f'},
        {'node': 'g'},
        {'node': 'h'},
        {'node': 'i'},
        {'node': 'j'},
        {'node': 'k'},
        {'node': 'l'},
        {'node': 'm'},
        {'node': 'n'},
        {'node': 'o'},
        {'node': 'p'}
    ])
    .assign(type='n')
    .reset_index().rename(columns={'index': 'i'})
    .pipe(lambda df: df.assign(v=df.index * 2)))

    #| s, d, type, i, v |
    edges_df = (pd.DataFrame([
        {'s': 'e', 'd': 'l'},
        {'s': 'l', 'd': 'b'},
        {'s': 'k', 'd': 'a'},
        {'s': 'e', 'd': 'g'},
        {'s': 'g', 'd': 'a'},
        {'s': 'd', 'd': 'f'},
        {'s': 'd', 'd': 'c'},
        {'s': 'd', 'd': 'j'},
        {'s': 'd', 'd': 'i'},
        {'s': 'd', 'd': 'h'},
        {'s': 'j', 'd': 'p'},
        {'s': 'i', 'd': 'n'},
        {'s': 'h', 'd': 'm'},
        {'s': 'j', 'd': 'o'},
        {'s': 'o', 'd': 'b'},
        {'s': 'm', 'd': 'a'},
        {'s': 'n', 'd': 'a'},
        {'s': 'p', 'd': 'b'},
    ]).assign(type='e')
    .reset_index().rename(columns={'index': 'i'})
    .pipe(lambda df: df.assign(v=df.index * 2)))

    return CGFull().nodes(nodes_df, 'node').edges(edges_df, 's', 'd')

class TestFilterByDict(object):

    def test_none(self):
        g = hops_graph()
        assert filter_by_dict(g._nodes).equals(g._nodes)

    def test_kv_single_good_str(self):
        g = hops_graph()
        assert filter_by_dict(g._nodes, {'node': 'a'}).equals(g._nodes[:1])

    def test_kv_single_good_int(self):
        g = hops_graph()
        assert filter_by_dict(g._nodes, {'v': 0}).equals(g._nodes[:1])

    def test_kv_single_miss(self):
        g = hops_graph()
        assert filter_by_dict(g._nodes, {'node': 'bad'}).equals(g._nodes[:0])

    def test_kv_single_multiple(self):
        g = hops_graph()
        assert filter_by_dict(g._nodes, {'type': 'n'}).equals(g._nodes)

    def test_kv_single_membership_list(self):
        g = hops_graph()
        expected = g._nodes[g._nodes['node'].isin(['a', 'b'])].reset_index(drop=True)
        actual = filter_by_dict(g._nodes, {'node': ['a', 'b']}).reset_index(drop=True)
        assert actual.equals(expected)

    def test_label_filter_uses_labels_column_membership(self):
        nodes = pd.DataFrame(
            {
                "id": ["a", "b"],
                "labels": [["Person"], ["Animal", "Pet"]],
                "name": ["alice", "bear"],
            }
        )
        actual = filter_by_dict(nodes, {"label__Person": True}).reset_index(drop=True)
        expected = nodes.iloc[[0]].reset_index(drop=True)
        assert actual.equals(expected)

    def test_label_filter_falls_back_to_node_type_column(self):
        nodes = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "type": ["Person", "Animal", "Person"],
                "name": ["alice", "bear", "bob"],
            }
        )
        actual = filter_by_dict(nodes, {"label__Person": True}).reset_index(drop=True)
        expected = nodes.iloc[[0, 2]].reset_index(drop=True)
        assert actual.equals(expected)

    def test_kv_multiple_good(self):
        g = hops_graph()
        assert filter_by_dict(g._nodes, {'node': 'a', 'type': 'n'}).equals(g._nodes[:1])

    def test_kv_multiple_bad(self):
        g = hops_graph()
        assert filter_by_dict(g._nodes, {'node': 'bad', 'type': 'n'}).equals(g._nodes[:0])


class TestNodeFilterByDict(object):

    def test_kv_multiple_good(self):
        g = hops_graph()
        assert g.filter_nodes_by_dict({'node': 'a', 'type': 'n'})._nodes.equals(g._nodes[:1])

    def test_kv_multiple_bad(self):
        g = hops_graph()
        assert g.filter_nodes_by_dict({'node': 'bad', 'type': 'n'})._nodes.equals(g._nodes[:0])

class TestEdgeFilterByDict(object):

    def test_kv_multiple_good(self):
        g = hops_graph()
        assert g.filter_edges_by_dict({'i': 0, 'type': 'e'})._edges.equals(g._edges[:1])

    def test_kv_multiple_bad(self):
        g = hops_graph()
        assert g.filter_edges_by_dict({'i': -100, 'type': 'e'})._edges.equals(g._edges[:0])

class TestPredicateIsIn(object):

    def test_standalone(self):
        g = hops_graph()
        assert g.filter_nodes_by_dict({'node': is_in(['a'])})._nodes.equals(g._nodes[:1])
        assert g.filter_nodes_by_dict({'node': is_in(['a', 'b'])})._nodes.equals(g._nodes[:2])
    
    def test_combined(self):
        g = hops_graph()
        assert g.filter_nodes_by_dict({'node': is_in(['a', 'b']), 'type': 'n'})._nodes.equals(g._nodes[:2])
        assert g.filter_nodes_by_dict({'node': is_in(['a', 'b']), 'type': 'bad'})._nodes.equals(g._nodes[:0])
        assert g.filter_nodes_by_dict({'node': is_in(['a', 'bad']), 'type': 'n'})._nodes.equals(g._nodes[:1])
        assert g.filter_nodes_by_dict({'node': is_in(['a', 'bad']), 'type': 'bad'})._nodes.equals(g._nodes[:0])


class TestThreeValuedLogicNull(object):
    """openCypher/SQL 3-valued logic: a NULL cell is NEVER a match for ne()/<> or membership
    (`null <> x` and `null IN [...]` are both NULL -> excluded). pandas `NaN != x` is True and
    cuDF `isin` matches a null against a None list element, so both used to KEEP the null; these
    assert the fix in filter_by_dict / the NE predicate. Generic (pandas oracle here; the polars
    engine + 4-engine parity are asserted in the gfql polars suite)."""

    def test_ne_excludes_null(self):
        # numeric column so the comparison-on-string guard (a separate fix higher in the stack) is
        # not in play — this isolates the NE 3VL null-masking. score=None -> NaN; `NaN != 20` is
        # True on pandas (would KEEP), so the row must be excluded by the `& notna()` fix.
        from graphistry.compute.ast import ne
        df = pd.DataFrame({"id": ["n0", "n1", "n2", "n3"], "score": [10.0, 20.0, None, 40.0]})
        out = filter_by_dict(df, {"score": ne(20)})
        assert sorted(out["id"].tolist()) == ["n0", "n3"]   # n1 fails ne(20); n2 (null) excluded by 3VL

    def test_membership_excludes_null(self):
        df = pd.DataFrame({"id": ["n0", "n1", "n2", "n3"], "kind": ["x", "y", None, "z"]})
        out = filter_by_dict(df, {"kind": ["x", "z", None]})
        assert sorted(out["id"].tolist()) == ["n0", "n3"]   # x,z match; null excluded despite None in list


class _WarnsOnEq(object):
    """An object cell whose ``__eq__`` warns: comparing it at all changes the warning count."""

    def __eq__(self, other):
        import warnings
        warnings.warn("object cell __eq__ called", UserWarning)
        return False

    __hash__ = object.__hash__


class TestObjectColumnNonStringCells(object):
    """#2365: a string equality on a pandas ``object`` column holding non-string cells (numpy
    arrays, as read from list-typed parquet/Arrow columns) must treat those cells as non-matches
    on every pandas filter path -- no data-dependent ValueError, no per-cell ``__eq__`` (so no
    warnings), and a 1-element array cell must not match by truthiness. Oracle: the same graph
    with every non-string cell replaced by a string that never matches."""

    @staticmethod
    def _cells():
        import numpy as np
        return [np.array([1, 2]), "a", "b", np.array(["a"]), _WarnsOnEq(), "a", None]

    @classmethod
    def _graph(cls, str_ids, oracle=False):
        ids = [f"n{i}" for i in range(1, 8)] if str_ids else list(range(1, 8))
        cells = [c if c is None or isinstance(c, str) or not oracle else "zz" for c in cls._cells()]
        nodes = pd.DataFrame({"id": ids, "tag": pd.Series(cells, dtype=object)})
        pairs = [(1, 2), (2, 3), (3, 4), (1, 4), (1, 5), (1, 6), (6, 3), (1, 7)]
        import numpy as np
        ecells = [np.array([1, 2]), "x", "y", np.array(["x"]), _WarnsOnEq(), "x", None, "y"]
        ecells = [c if c is None or isinstance(c, str) or not oracle else "zz" for c in ecells]
        edges = pd.DataFrame({
            "s": [ids[a - 1] for a, _ in pairs],
            "d": [ids[b - 1] for _, b in pairs],
            "etag": pd.Series(ecells, dtype=object),
        })
        return CGFull().nodes(nodes, "id").edges(edges, "s", "d")

    @staticmethod
    def _run(g, query, force_full, monkeypatch):
        import graphistry.compute.chain as chain_mod
        import graphistry.compute.gfql_unified as gfql_unified
        if force_full:
            monkeypatch.setattr(chain_mod, "_try_chain_fast_path", lambda *a, **k: None)
            monkeypatch.setattr(gfql_unified, "_execute_seeded_typed_hop_fast_path", lambda *a, **k: None)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="object cell __eq__ called")
            return g.gfql(query, engine="pandas")

    @staticmethod
    def _shape(res):
        nodes = sorted(res._nodes["id"].tolist()) if res._nodes is not None else None
        edges = sorted(zip(res._edges["s"], res._edges["d"])) if res._edges is not None else None
        return nodes, edges

    def test_filter_by_dict(self):
        import warnings
        df = pd.DataFrame({"id": range(7), "tag": pd.Series(self._cells(), dtype=object)})
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="object cell __eq__ called")
            assert filter_by_dict(df, {"tag": "a"})["id"].tolist() == [1, 5]
            assert filter_by_dict(df, {"tag": "missing"})["id"].tolist() == []
            assert filter_by_dict(df, {"id": 1, "tag": "a"})["id"].tolist() == [1]

    @pytest.mark.parametrize("force_full", [False, True])
    @pytest.mark.parametrize("str_ids", [False, True])
    @pytest.mark.parametrize("query_name", [
        "issue_repro", "single_node", "seed_on_object_col", "edge_match", "two_hops", "reverse",
    ])
    def test_gfql_paths(self, query_name, str_ids, force_full, monkeypatch):
        from graphistry.compute.ast import n, e_forward, e_reverse
        seed = "n1" if str_ids else 1
        query = {
            "issue_repro": [n({"id": seed}), e_forward(), n({"tag": "a"})],
            "single_node": [n({"tag": "a"})],
            "seed_on_object_col": [n({"tag": "a"}), e_forward(), n()],
            "edge_match": [n({"id": seed}), e_forward(edge_match={"etag": "x"}), n()],
            "two_hops": [n({"id": seed}), e_forward(hops=2), n({"tag": "b"})],
            "reverse": [n({"tag": "b"}), e_reverse(), n({"id": seed})],
        }[query_name]
        actual = self._run(self._graph(str_ids), query, force_full, monkeypatch)
        expected = self._run(self._graph(str_ids, oracle=True), query, True, monkeypatch)
        assert self._shape(actual) == self._shape(expected)

    @pytest.mark.parametrize("force_full", [False, True])
    def test_cypher_seeded_return(self, force_full, monkeypatch):
        query = "MATCH (s {id: 1})-->(b {tag: 'a'}) RETURN b.id AS bid ORDER BY bid"
        actual = self._run(self._graph(False), query, force_full, monkeypatch)
        expected = self._run(self._graph(False, oracle=True), query, True, monkeypatch)
        assert actual._nodes["bid"].tolist() == expected._nodes["bid"].tolist() == [2, 6]

    def test_property_candidates_residual(self):
        from graphistry.Engine import Engine
        from graphistry.compute.filter_by_dict import _filter_property_candidates
        nodes = self._graph(False)._nodes
        candidates = nodes.iloc[[0, 1, 3, 4, 5]]
        for validated in (False, True):
            out = _filter_property_candidates(nodes, candidates, {"tag": "a"}, Engine.PANDAS, filter_validated=validated)
            assert out["id"].tolist() == [2, 6]

    def test_verify_scalar_filters_on_hit(self):
        from graphistry.Engine import Engine
        from graphistry.compute.chain_fast_paths import _verify_scalar_filters_on_hit
        nodes = self._graph(False)._nodes
        assert _verify_scalar_filters_on_hit(nodes, {"tag": "a"}, Engine.PANDAS)["id"].tolist() == [2, 6]
        for row in (0, 3, 4):  # single-row hits whose cell is not a string
            assert _verify_scalar_filters_on_hit(nodes.iloc[[row]], {"tag": "a"}, Engine.PANDAS).empty

    def test_index_edge_match_masks(self):
        import numpy as np
        from graphistry.Engine import Engine
        from graphistry.compute.gfql.index.traverse import _EdgeMatchRowFilter
        edges = self._graph(False)._edges
        row_filter = _EdgeMatchRowFilter({"etag": edges["etag"]}, [("etag", "x")], Engine.PANDAS)
        expected = [False, True, False, False, False, True, False, False]
        assert row_filter.full_mask().tolist() == expected
        rows = np.array([1, 3, 5])
        assert row_filter.mask_for(rows).tolist() == [True, False, True]

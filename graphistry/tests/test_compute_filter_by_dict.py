import pandas as pd
from functools import lru_cache

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

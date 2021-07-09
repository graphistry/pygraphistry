# -*- coding: utf-8 -*- 
from typing import Iterable
import os, numpy as np, pandas as pd, pyarrow as pa, pytest, queue
from common import NoAuthTestCase
from concurrent.futures import Future
from mock import patch
from gremlin_python.driver.resultset import ResultSet
from gremlin_python.structure.graph import Vertex, Edge, Path

from graphistry.gremlin import CosmosMixin, GremlinMixin, DROP_QUERY, nodes_to_queries, edges_to_queries
from graphistry.plotter import PlotterBase


# ### Helpers ### #


def fake_client(query_to_result = {}):

    class FakeCallback:

        def __init__(self, query: str):
            self.query = query

        def result(self):
            return query_to_result[self.query]

    class FakeClient:

        def submitAsync(self, query: str):
            cb = FakeCallback(query)
            return cb

    return FakeClient()


class TG(GremlinMixin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        GremlinMixin.__init__(self, *args, **kwargs)

class TGFull(GremlinMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        print('TGFull init')
        super(TGFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        super(GremlinMixin, self).__init__(*args, **kwargs)

class CFull(CosmosMixin, GremlinMixin, PlotterBase):
    def __init__(self, *args, **kwargs):
        print('CFull init')
        #super(CFull, self).__init__(*args, **kwargs)
        PlotterBase.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)



def make_resultset(items = []) -> Iterable:
    q = queue.Queue()
    for item in items:
        q.put(item)

    f = Future()
    f.set_result([])

    rs = ResultSet(q, 'x')

    rs.done = f

    return rs  # [x for x in rs]


# ### Gremlin ### #


class TestGremlinMixin(NoAuthTestCase):

    def test_connect_default_off(self):
        tg = TG()
        with self.assertRaises(ValueError):
            tg.connect()

    def test_drop(self):
        tg = TG(gremlin_client=fake_client({DROP_QUERY: 1}))
        assert tg.drop_graph() is tg

    def test_run_none(self):
        tg = TG(gremlin_client=fake_client({}))
        assert len([x for x in tg.gremlin_run([])]) == 0

    def test_run_one(self):
        tg = TG(gremlin_client=fake_client({'a': 'b'}))
        g = tg.gremlin_run(['a'])
        assert next(g) == 'b'
        for rest in g:
            raise ValueError('Unexpected additional elements')

    def test_run_mult(self):
        tg = TG(gremlin_client=fake_client({'a': 'b', 'c': 'd'}))
        g = tg.gremlin_run(['a', 'c'])
        assert next(g) == 'b'
        assert next(g) == 'd'
        for rest in g:
            raise ValueError('Unexpected additional elements')

    def test_resultset_to_g_empty(self):
        rs = make_resultset([])
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._nodes is None or len(g._nodes) == 0
        assert g._edges is None or len(g._edges) == 0

    def test_resultset_to_g_empty2(self):
        rs = make_resultset([[], []])
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._nodes is None or len(g._nodes) == 0
        assert g._edges is None or len(g._edges) == 0

    def test_resultset_to_g_single_edge(self):
        rs = make_resultset([{'type': 'edge', 'inV': 'a', 'outV': 'b'}])
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._nodes is None
        assert len(g._edges) == 1
        assert g._source == 'src'
        assert g._destination == 'dst'

    def test_resultset_to_g_edges_attributed(self):
        edges = [
            {'type': 'edge', 'inV': 'a', 'outV': 'b', 'label': 'l1',
             'properties': {'x': 'y', 'f': 'g', 'inV': 'ignoreme', 'src': 'ignoreme', 'label': 'ignoreme'}},
            {'type': 'edge', 'inV': 'm', 'outV': 'n', 'label': 'l2',
             'properties': {'x': 'yy', 'f': 'gg', 'outV': 'ignoreme', 'dst': 'ignoreme', 'label': 'ignoreme'}},
        ]
        rs = make_resultset(edges)
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._nodes is None
        assert len(g._edges) == 2
        assert g._source == 'src'
        assert g._destination == 'dst'
        assert g._edges.to_dict(orient='records') == [
            {'src': 'a', 'dst': 'b', 'x': 'y', 'f': 'g', 'label': 'l1', 'inV': 'ignoreme', 'outV': np.nan},
            {'src': 'm', 'dst': 'n', 'x': 'yy', 'f': 'gg', 'label': 'l2', 'inV': np.nan, 'outV': 'ignoreme', }
        ]

    def test_resultset_to_g_single_node(self):
        rs = make_resultset([{'type': 'vertex', 'id': 'a', 'label': 'b'}])
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'label': 'b'}
        ]
        assert g._edges.to_dict(orient='records') == []
        assert g._node == 'id'
        assert g._source == 'src'
        assert g._destination == 'dst'

    def test_resultset_to_g_multi_node_attributed(self):
        nodes = [
            {'type': 'vertex', 'id': 'a', 'label': 'b', 'properties': { 'a': 'b', 'c': 'd', 'id': 'ignoreme'}},
            {'type': 'vertex', 'id': 'b', 'label': 'bb', 'properties': { 'a': 'bb', 'c': 'dd', 'label': 'ignoreme'}}
        ]
        rs = make_resultset(nodes)
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'label': 'b', 'a': 'b', 'c': 'd'},
            {'id': 'b', 'label': 'bb', 'a': 'bb', 'c': 'dd'}
        ]
        assert g._edges.to_dict(orient='records') == []
        assert g._node == 'id'
        assert g._source == 'src'
        assert g._destination == 'dst'

    def test_resultset_to_g_vertex_stucture(self):
        rs = make_resultset([Vertex(id='a', label='b')])
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._nodes.to_dict(orient='records') == [
            {'id': 'a', 'label': 'b'}
        ]
        assert g._edges.to_dict(orient='records') == []
        assert g._node == 'id'
        assert g._source == 'src'
        assert g._destination == 'dst'

    def test_resultset_to_g_edge_stucture(self):
        inV = Vertex(id='a', label='b')
        outV = Vertex(id='c', label='d')
        e = Edge(id='a', outV=outV, label='e', inV=inV)
        rs = make_resultset([e])
        tg = TGFull()
        g = tg.resultset_to_g(rs)
        assert g._edges.to_dict(orient='records') == [ {'src': 'a', 'dst': 'c', 'id': 'a', 'label': 'e'} ]

    def test_gremlin_none(self):
        tg = TGFull(gremlin_client=fake_client())
        g = tg.gremlin([])
        assert g._edges.to_dict(orient='records') == []

    def test_gremlin_one_edge(self):
        tg = TGFull(gremlin_client=fake_client({'g.E()': [
            [ {'type': 'edge', 'inV': 'a', 'outV': 'b', 'properties': {'x': 'y', 'f': 'g'}} ]
        ]}))
        g = tg.gremlin(['g.E()'])
        assert g._nodes is None
        assert g._edges.to_dict(orient='records') == [ {'src': 'a', 'dst': 'b', 'x': 'y', 'f': 'g'} ]

    def test_nodes_to_queries_mt(self):
        df = pd.DataFrame({'n': [], 'v1': []})
        g = PlotterBase()
        assert len([ x for x in nodes_to_queries(g.nodes(df, 'n'), untyped=True)]) == 0

    def test_nodes_to_queries_single_untyped(self):
        df = pd.DataFrame({'n': ['i'], 'v1': [2]})
        g = PlotterBase()
        assert [ x for x in nodes_to_queries(g.nodes(df, 'n'), untyped=True)][0] == "g.addV().property('n', 'i').property('v1', '2')"

    def test_nodes_to_queries_single_typed(self):
        df = pd.DataFrame({'n': ['i'], 'v1': [2]})
        g = PlotterBase()
        assert [ x for x in nodes_to_queries(g.nodes(df, 'n'), type_col='n')][0] == "g.addV('i').property('v1', '2')"
        
    def test_nodes_to_queries_single_typed_inferred_type(self):
        df = pd.DataFrame({'type': ['i'], 'v1': [2]})
        g = PlotterBase()
        assert [ x for x in nodes_to_queries(g.nodes(df, 'n'))][0] == "g.addV('i').property('v1', '2')"

    def test_nodes_to_queries_single_typed_inferred_category(self):
        df = pd.DataFrame({'category': ['i'], 'v1': [2]})
        g = PlotterBase()
        assert [ x for x in nodes_to_queries(g.nodes(df, 'n'))][0] == "g.addV('i').property('v1', '2')"

    def test_nodes_to_queries_multi(self):
        df = pd.DataFrame({'n': ['i', 'i2'], 'v1': [2, 3]})
        g = PlotterBase()
        assert len([ x for x in nodes_to_queries(g.nodes(df, 'n'), untyped=True)]) == 2

    def test_edge_to_queries_mt(self):
        df = pd.DataFrame({'s': [], 'd': []})
        g = PlotterBase()
        assert len([ x for x in edges_to_queries(g.edges(df, 's', 'd'), untyped=True)]) == 0

    def test_edge_to_queries_single_untyped(self):
        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = PlotterBase()
        assert [ x for x in edges_to_queries(g.edges(df, 's', 'd'), untyped=True)][0] == "g.v('a').addE().to(g.v('b'))"

    def test_edge_to_queries_single_untyped_attributed(self):
        df = pd.DataFrame({'s': ['a'], 'd': ['b'], 'v1': [2]})
        g = PlotterBase()
        assert [ x for x in edges_to_queries(g.edges(df, 's', 'd'), untyped=True)][0] == "g.v('a').addE().to(g.v('b')).property('v1', '2')"

    def test_edge_to_queries_single_typed_attributed(self):
        df = pd.DataFrame({'s': ['a'], 'd': ['b'], 'v1': [2], 't': ['x']})
        g = PlotterBase()
        assert [ x for x in edges_to_queries(g.edges(df, 's', 'd'), type_col='t')][0] == "g.v('a').addE('x').to(g.v('b')).property('v1', '2')"

    def test_edge_to_queries_single_typed_inferred_type(self):
        df = pd.DataFrame({'s': ['a'], 'd': ['b'], 'v1': [2], 'type': ['x']})
        g = PlotterBase()
        assert [ x for x in edges_to_queries(g.edges(df, 's', 'd'))][0] == "g.v('a').addE('x').to(g.v('b')).property('v1', '2')"

    def test_edge_to_queries_single_typed_inferred_edgeType(self):
        df = pd.DataFrame({'s': ['a'], 'd': ['b'], 'v1': [2], 'edgeType': ['x']})
        g = PlotterBase()
        assert [ x for x in edges_to_queries(g.edges(df, 's', 'd'))][0] == "g.v('a').addE('x').to(g.v('b')).property('v1', '2')"

    def test_edge_to_queries_single_typed_inferred_category(self):
        df = pd.DataFrame({'s': ['a'], 'd': ['b'], 'v1': [2], 'category': ['x']})
        g = PlotterBase()
        assert [ x for x in edges_to_queries(g.edges(df, 's', 'd'))][0] == "g.v('a').addE('x').to(g.v('b')).property('v1', '2')"


class TestCosmosMixin(NoAuthTestCase):

    def test_cosmos_init(self):
        cg = CFull(gremlin_client=fake_client({'g.E()': [
            [ {'type': 'edge', 'inV': 'a', 'outV': 'b', 'properties': {'x': 'y', 'f': 'g'}} ]
        ]}))
        g = cg.gremlin(['g.E()'])
        assert g._nodes is None
        assert g._edges.to_dict(orient='records') == [ {'src': 'a', 'dst': 'b', 'x': 'y', 'f': 'g'} ]

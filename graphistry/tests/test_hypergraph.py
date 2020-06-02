# -*- coding: utf-8 -*-

import datetime as dt, numpy, pandas as pd, pyarrow as pa, unittest

import graphistry, graphistry.plotter
from common import NoAuthTestCase

nid = graphistry.plotter.Plotter._defaultNodeId

triangleNodesDict = {
    'id': ['a', 'b', 'c'], 
    'a1': [1, 2, 3], 
    'a2': ['red', 'blue', 'green'],
    '🙈': ['æski ēˈmōjē', '😋', 's']
}
triangleNodes = pd.DataFrame(triangleNodesDict)

hyper_df = pd.DataFrame({'aa': [0, 1, 2], 'bb': ['a', 'b', 'c'], 'cc': ['b', 0, 1]})

squareEvil = pd.DataFrame({
    'src': [0,1,2,3],
    'dst': [1,2,3,0],
    'colors': [1, 1, 2, 2],
    'list_int': [ [1], [2, 3], [4], []],
    'list_str': [ ['x'], ['1', '2'], ['y'], []],
    'list_bool': [ [True], [True, False], [False], []],
    'list_date_str': [ ['2018-01-01 00:00:00'], ['2018-01-02 00:00:00', '2018-01-03 00:00:00'], ['2018-01-05 00:00:00'], []],
    'list_date': [ [pd.Timestamp('2018-01-05')], [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')], [], []],
    'list_mixed': [ [1], ['1', '2'], [False, None], []],
    'bool': [True, False, True, True],
    'char': ['a', 'b', 'c', 'd'],
    'str': ['a', 'b', 'c', 'd'],
    'ustr': [u'a', u'b', u'c', u'd'],
    'emoji': ['😋', '😋😋', '😋', '😋'],
    'int': [0, 1, 2, 3],
    'num': [0.5, 1.5, 2.5, 3.5],
    'date_str': ['2018-01-01 00:00:00', '2018-01-02 00:00:00', '2018-01-03 00:00:00', '2018-01-05 00:00:00'],
    
    ## API 1 BUG: Try with https://github.com/graphistry/pygraphistry/pull/126
    'date': [dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1)],
    'time': [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')],
    
    ## API 2 BUG: Need timedelta in https://github.com/graphistry/pygraphistry/blob/master/graphistry/vgraph.py#L108
    'delta': [pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day')]
})
for c in squareEvil.columns:
    try:
        squareEvil[c + '_cat'] = squareEvil[c].astype('category')
    except:
        # lists aren't categorical
        #print('could not make categorical', c)
        1


def assertFrameEqual(df1, df2, **kwds ):
    """ Assert that two dataframes are equal, ignoring ordering of columns"""

    from pandas.util.testing import assert_frame_equal
    return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True, **kwds)


class TestHypergraphPlain(NoAuthTestCase):

  
    def test_hyperedges(self):

        h = graphistry.hypergraph(triangleNodes, verbose=False)
        
        self.assertEqual(len(h.keys()), len(['entities', 'nodes', 'edges', 'events', 'graph']))

        edges = pd.DataFrame({
            'a1': [1, 2, 3] * 4,
            'a2': ['red', 'blue', 'green'] * 4,
            'id': ['a', 'b', 'c'] * 4,
            '🙈': ['æski ēˈmōjē', '😋', 's'] * 4,
            'edgeType': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
            'attribID': [
                'a1::1', 'a1::2', 'a1::3', 
                'a2::red', 'a2::blue', 'a2::green',                 
                'id::a', 'id::b', 'id::c',
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
            'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})

        assertFrameEqual(h['edges'], edges)
        for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
            self.assertEqual(len(h[k]), v)

    def test_hyperedges_direct(self):

        h = graphistry.hypergraph(hyper_df, verbose=False, direct=True)
        
        self.assertEqual(len(h['edges']), 9)
        self.assertEqual(len(h['nodes']), 9)

    def test_hyperedges_direct_categories(self):

        h = graphistry.hypergraph(hyper_df, verbose=False, direct=True, opts={'CATEGORIES': {'n': ['aa', 'bb', 'cc']}})
        
        self.assertEqual(len(h['edges']), 9)
        self.assertEqual(len(h['nodes']), 6)

    def test_hyperedges_direct_manual_shaping(self):

        h1 = graphistry.hypergraph(hyper_df, verbose=False, direct=True, opts={'EDGES': {'aa': ['cc'], 'cc': ['cc']}})
        self.assertEqual(len(h1['edges']), 6)

        h2 = graphistry.hypergraph(hyper_df, verbose=False, direct=True, opts={'EDGES': {'aa': ['cc', 'bb', 'aa'], 'cc': ['cc']}})
        self.assertEqual(len(h2['edges']), 12)


    def test_drop_edge_attrs(self):
    
        h = graphistry.hypergraph(triangleNodes, verbose=False, drop_edge_attrs=True)

        self.assertEqual(len(h.keys()), len(['entities', 'nodes', 'edges', 'events', 'graph']))

        edges = pd.DataFrame({
            'edgeType': ['a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'id', 'id', 'id', '🙈', '🙈', '🙈'],
            'attribID': [
                'a1::1', 'a1::2', 'a1::3', 
                'a2::red', 'a2::blue', 'a2::green',                 
                'id::a', 'id::b', 'id::c',
                '🙈::æski ēˈmōjē', '🙈::😋', '🙈::s'],
            'EventID': ['EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2', 'EventID::0', 'EventID::1', 'EventID::2']})


        assertFrameEqual(h['edges'], edges)
        for (k, v) in [('entities', 12), ('nodes', 15), ('edges', 12), ('events', 3)]:
            self.assertEqual(len(h[k]), v)


    def test_drop_na_hyper(self):

        df = pd.DataFrame({
            'a': ['a', None, 'c'],
            'i': [1, 2, None]
        })

        hg = graphistry.hypergraph(df, drop_na=True)

        assert len(hg['graph']._nodes) == 7
        assert len(hg['graph']._edges) == 4

    def test_drop_na_direct(self):

        df = pd.DataFrame({
            'a': ['a', None, 'a'],
            'i': [1, 1, None]
        })

        hg = graphistry.hypergraph(df, drop_na=True, direct=True)

        assert len(hg['graph']._nodes) == 2
        assert len(hg['graph']._edges) == 1

    def test_skip_na_hyperedge(self):
    
        nans_df = pd.DataFrame({
          'x': ['a', 'b', 'c'],
          'y': ['aa', None, 'cc']
        })
        expected_hits = ['a', 'b', 'c', 'aa', 'cc']

        skip_attr_h_edges = graphistry.hypergraph(nans_df, drop_edge_attrs=True)['edges']
        self.assertEqual(len(skip_attr_h_edges), len(expected_hits))

        default_h_edges = graphistry.hypergraph(nans_df)['edges']
        self.assertEqual(len(default_h_edges), len(expected_hits))

    def test_hyper_evil(self):
        graphistry.hypergraph(squareEvil)

    def test_hyper_to_pa_vanilla(self):

        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': ['d', 'e', 'f']
        })

        hg = graphistry.hypergraph(df)
        nodes_arr = pa.Table.from_pandas(hg['graph']._nodes)
        assert len(nodes_arr) == 9
        edges_err = pa.Table.from_pandas(hg['graph']._edges)
        assert len(edges_err) == 6

    def test_hyper_to_pa_mixed(self):

        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],
            'y': [1, 2, 3]
        })

        hg = graphistry.hypergraph(df)
        nodes_arr = pa.Table.from_pandas(hg['graph']._nodes)
        assert len(nodes_arr) == 9
        edges_err = pa.Table.from_pandas(hg['graph']._edges)
        assert len(edges_err) == 6

    def test_hyper_to_pa_na(self):

        df = pd.DataFrame({
            'x': ['a', None, 'c'],
            'y': [1, 2, None]
        })

        hg = graphistry.hypergraph(df, drop_na=False)
        nodes_arr = pa.Table.from_pandas(hg['graph']._nodes)
        assert len(hg['graph']._nodes) == 9
        assert len(nodes_arr) == 9
        edges_err = pa.Table.from_pandas(hg['graph']._edges)
        assert len(hg['graph']._edges) == 6
        assert len(edges_err) == 6
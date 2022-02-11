import unittest
import copy, datetime as dt, graphistry, os, pandas as pd

from graphistry.ai.feature_utils import *
from data import
# python -m unittest

bad_df = pd.DataFrame({
    'src': [0, 1, 2, 3],
    'dst': [1, 2, 3, 0],
    'colors': [1, 1, 2, 2],
    'list_int': [[1], [2, 3], [4], []],
    'list_str': [['x'], ['1', '2'], ['y'], []],
    'list_bool': [[True], [True, False], [False], []],
    'list_date_str': [['2018-01-01 00:00:00'], ['2018-01-02 00:00:00', '2018-01-03 00:00:00'], ['2018-01-05 00:00:00'],
                      []],
    'list_date': [[pd.Timestamp('2018-01-05')], [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05')], [], []],
    'list_mixed': [[1], ['1', '2'], [False, None], []],
    'bool': [True, False, True, True],
    'char': ['a', 'b', 'c', 'd'],
    'str': ['a', 'b', 'c', 'd'],
    'ustr': [u'a', u'b', u'c', u'd'],
    'emoji': ['😋', '😋😋', '😋', '😋'],
    'int': [0, 1, 2, 3],
    'num': [0.5, 1.5, 2.5, 3.5],
    'date_str': ['2018-01-01 00:00:00', '2018-01-02 00:00:00', '2018-01-03 00:00:00', '2018-01-05 00:00:00'],
    
    # API 1 BUG: Try with https://github.com/graphistry/pygraphistry/pull/126
    'date': [dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 1)],
    'time': [pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'), pd.Timestamp('2018-01-05'),
             pd.Timestamp('2018-01-05')],
    
    # API 2 BUG: Need timedelta in https://github.com/graphistry/pygraphistry/blob/master/graphistry/vgraph.py#L108
    'delta': [pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day'), pd.Timedelta('1 day')],
    
    'textual': ['here we have a sentence. And here is another sentence. Graphistry is an amazing tool']*4
})

edge_df = bad_df.astype(str)
node_df = pd.DataFrame({})



double_target = pd.DataFrame({'label': ndf.label.values, 'type': ndf['type'].values})
single_target = pd.DataFrame({'label': ndf.label.values})

class TestFeatureProcessors(unittest.TestCase):
    
    def setUp(self):
        resses = process_dirty_dataframes(ndf, y=double_target, z_scale=False)
    
    def


class TestFeaturizerMethods(unittest.TestCase):

    def setUp(self):
        self.fm = FeatureMixin()
        
    def test_adding_edges_dataframe(self):
        fm = self.fm
        g = fm.edges(edge_df, 'src', 'dst')
        self.assertEqual(g._edges, edge_df) # pretty vanilla
        
    def test_edge_featurizer(self):
        process_edge_dataframes()

    def test_featurize_edges(self):
        fm = self.fm
        g = fm.edges(edge_df, 'src', 'dst')
        g2 = g.featurize(kind='edges')

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)



if __name__ == '__main__':
    unittest.main()
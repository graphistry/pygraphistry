import pandas as pd
import unittest
import pytest
import graphistry
from graphistry.Plottable import Plottable
from graphistry.constants import DBSCAN
from graphistry.models.ModelDict import ModelDict
from graphistry.utils.lazy_import import (
    lazy_dbscan_import,
    lazy_umap_import
)

has_dbscan, _, has_gpu_dbscan, _ = lazy_dbscan_import()
has_umap, _, _ = lazy_umap_import()


ndf = edf = pd.DataFrame({'src': [1, 2, 1, 4], 'dst': [4, 5, 6, 1], 'label': ['a', 'b', 'b', 'c']})

class TestComputeCluster(unittest.TestCase):
    
    def _condition(self, g: Plottable, kind):
        if kind == 'nodes':
            self.assertTrue(g._dbscan_nodes is not None, 'instance has no `_dbscan_nodes` method')
            self.assertTrue(DBSCAN in g._nodes, 'node df has no `_dbscan` attribute')
            #self.assertTrue(g._point_color is not None, 'instance has no `_point_color` method')
        else:
            self.assertTrue(g._dbscan_edges is not None, 'instance has no `_dbscan_edges` method')
            self.assertTrue(DBSCAN in g._edges, 'edge df has no `_dbscan` attribute')
    
    @pytest.mark.skipif(not has_dbscan or not has_umap, reason="requires ai dependencies")
    def test_umap_cluster(self):
        g = graphistry.nodes(ndf).edges(edf, 'src', 'dst')
        for kind in ['nodes', 'edges']:
            g2 = g.umap(kind=kind, n_topics=2, dbscan=False).dbscan(kind=kind, verbose=True)
            self._condition(g2, kind)
            g3 = g.umap(kind=kind, n_topics=2, dbscan=True)
            self._condition(g3, kind)
            if kind == 'nodes':
                self.assertEqual(g2._nodes[DBSCAN].tolist(), g3._nodes[DBSCAN].tolist())
            else:
                self.assertEqual(g2._edges[DBSCAN].tolist(), g3._edges[DBSCAN].tolist())

    @pytest.mark.skipif(not has_dbscan, reason="requires ai dependencies")
    def test_featurize_cluster(self):
        g = graphistry.nodes(ndf).edges(edf, 'src', 'dst')
        for kind in ['nodes', 'edges']:
            g = g.featurize(kind=kind, n_topics=2).dbscan(kind=kind, verbose=True)
            self._condition(g, kind)
            
    @pytest.mark.skipif(not has_dbscan or not has_umap, reason="requires ai dependencies")
    def test_dbscan_params(self):
        dbscan_params = [ModelDict('Testing UMAP', kind='nodes', min_dist=0.2, min_samples=1, cols=None, target=False, 
                                   fit_umap_embedding=False, verbose=True, engine_dbscan='sklearn'), 
                         ModelDict('Testing UMAP target', kind='nodes', min_dist=0.1, min_samples=1, cols=None, 
                                   fit_umap_embedding=True, target=True, verbose=True, engine_dbscan='sklearn'),

        ]
        for params in dbscan_params:
            g = graphistry.nodes(ndf).edges(edf, 'src', 'dst').umap(y='label', n_topics=2)
            g2 = g.dbscan(**params)
            self.assertTrue(g2._dbscan_params == params, f'dbscan params not set correctly, found {g2._dbscan_params} but expected {params}')
        
    @pytest.mark.skipif(not has_gpu_dbscan or not has_umap, reason="requires ai dependencies")
    def test_transform_dbscan(self):
        kind = 'nodes'
        g = graphistry.nodes(ndf).edges(edf, 'src', 'dst')
        g2 = g.umap(y='label', n_topics=2, kind=kind).dbscan(fit_umap_embedding=True)
        
        _, _, _, df = g2.transform_dbscan(ndf, kind=kind, verbose=True, return_graph=False)
        self.assertTrue(DBSCAN in df, f'transformed df has no `{DBSCAN}` attribute')
                
        g3 = g2.transform_dbscan(ndf, ndf, verbose=True)
        self._condition(g3, kind)
        
            
if __name__ == '__main__':
    unittest.main()
    

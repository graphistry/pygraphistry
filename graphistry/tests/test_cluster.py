import pandas as pd
import unittest
import pytest
import graphistry


from graphistry.compute.cluster import lazy_dbscan_import_has_dependency

has_dbscan, _, has_gpu_dbscan, _ = lazy_dbscan_import_has_dependency()


ndf = edf = pd.DataFrame({'src': [1, 2, 3, 4], 'dst': [4, 5, 6, 1]})

class TestComputeCluster(unittest.TestCase):
    
    def _condition(self, g, kind):
        if kind == 'nodes':
            self.assertTrue(g._node_dbscan is not None)
            self.assertTrue('_cluster' in g._nodes)
        else:
            self.assertTrue(g._edge_dbscan is not None)
            self.assertTrue('_cluster' in g._edges)
    
    @pytest.mark.skipif(not has_dbscan, reason="requires ai dependencies")
    def test_umap_cluster(self):
        for kind in ['nodes', 'edges']:
            g = graphistry.nodes(ndf).edges(edf, 'src', 'dst')
            g = g.umap(kind=kind, n_topics=2).dbscan(kind=kind)
            self._condition(g, kind)    


    @pytest.mark.skipif(not has_dbscan, reason="requires ai dependencies")
    def test_featurize_edge_cluster(self):
        g = graphistry.edges(edf, 'src', 'dst').nodes(ndf)
        for kind in ['nodes', 'edges']:
            g = g.featurize(kind=kind, n_topics=2).dbscan(kind=kind)
            self._condition(g, kind)
        
        
if __name__ == '__main__':
    unittest.main()
    
1

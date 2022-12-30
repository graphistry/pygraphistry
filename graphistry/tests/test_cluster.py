import pandas as pd
import unittest
import pytest
import graphistry


from graphistry.compute.cluster import lazy_dbscan_import_has_dependency

has_dbscan, _, has_gpu_dbscan, _ = lazy_dbscan_import_has_dependency()


ndf = edf = pd.DataFrame({'src': [1, 2, 3], 'dst': [4, 5, 6]})
edf_umap = pd.DataFrame({'src': [1, 2, 3], 'dst': [4, 5, 6], 'x': [1, 2, 3], 'y': [4, 5, 6]})

node_embedding = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
edge_embedding = node_embedding

class TestComputeCluster(unittest.TestCase):
    
    @pytest.mark.skipif(not has_dbscan, reason="requires DGL dependencies")
    def test_umap_node_cluster(self):
        g = graphistry.nodes(ndf)
        g = g.umap(kind='nodes').dbscan(kind='nodes')
        self.assertTrue('_cluster' in g._nodes)
        self.assertTrue(g._node_dbscan is not None)

    @pytest.mark.skipif(not has_dbscan, reason="requires DGL dependencies")
    def test_umap_edge_cluster(self):
        g = graphistry.bind(source='src', destination='dst').edges(edf)
        g = g.umap(kind='edges').dbscan(kind='edges')        
        self.assertTrue('_cluster' in g._edges)
        self.assertTrue(g._edge_dbscan is not None)

    @pytest.mark.skipif(not has_dbscan, reason="requires DGL dependencies")
    def test_featurize_edge_cluster(self):
        g = graphistry.bind(source='src', destination='dst').edges(edf).nodes(ndf)
        for kind in ['nodes', 'edges']:
            g = g.featurize(kind=kind).dbscan(kind=kind)        
            if kind == 'nodes':
                self.assertTrue(g._node_dbscan is not None)
                self.assertTrue('_cluster' in g._nodes)
            else:
                self.assertTrue(g._edge_dbscan is not None)
                self.assertTrue('_cluster' in g._edges)
        
        
        
if __name__ == '__main__':
    unittest.main()

    
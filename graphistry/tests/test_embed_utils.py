import os
import pytest
import pandas as pd
import unittest
import graphistry
import numpy as np

from graphistry.embed_utils import check_cudf
from graphistry.utils.lazy_import import lazy_embed_import

import logging
logger = logging.getLogger(__name__)

dep_flag, _, _, _, _, _, _, _ = lazy_embed_import()
has_cudf, cudf = check_cudf()

# enable tests if has cudf and env didn't explicitly disable
is_test_cudf = has_cudf and os.environ.get("TEST_CUDF", "0") == "1"

class TestEmbed(unittest.TestCase):

    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    def setUp(self):
        self.edf = pd.DataFrame([[0, 1, 0], [1, 2, 0], [2, 0, 1]],
            columns=['src', 'dst', 'rel']
        )
        ndf_no_ids = pd.DataFrame([['a'], ['a'], ['b']], columns=['feat'])
        ndf_with_ids = pd.DataFrame([[0, 'a'], [1, 'a'], [2, 'b']],
            columns = ['id', 'feat1']
        )

        self.graph_no_feat = graphistry.edges(self.edf, 'src', 'dst')
        self.graph_with_feat_no_ids = self.graph_no_feat.nodes(ndf_no_ids)
        self.graph_with_feat_with_ids = self.graph_no_feat.nodes(ndf_with_ids, 'id')
        self.graphs = [
            ('no_feat', self.graph_no_feat),
            ('with_feat_no_ids', self.graph_with_feat_no_ids),
            ('with_feat_with_ids', self.graph_with_feat_with_ids)
        ]
        self.d = 4

        self.kwargs = {'n_topics': 6, 'cardinality_threshold':10, 'epochs': 1, 'sample_size':10, 'num_steps':10}
        

    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    def test_embed_out_basic(self):
        for name, g in self.graphs:
            g = g.embed('rel', embedding_dim=self.d, **self.kwargs)
            num_nodes = len(set(g._edges['src'] + g._edges['dst']))
            logging.debug('name: %s basic tests', name)
            self.assertEqual(g._edges.shape, self.edf.shape)
            self.assertEqual(set(g._edges[g._relation]), set(g._edges['rel']))
            self.assertEqual(g._kg_embeddings.shape,(num_nodes, self.d))


    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    def test_predict_links(self):
        source = pd.Series([0,2])
        relation = None
        destination = pd.Series([1])
        g = self.graph_no_feat.embed('rel', embedding_dim=self.d, **self.kwargs)

        g_new = g.predict_links(source, relation, destination, threshold=0, anomalous=False)
        self.assertTrue( g_new._edges.shape[0] > 0)
        self.assertIn("score", g_new._edges.columns)
        
        g_new = g.predict_links(source, relation, destination, threshold=1, anomalous=True)
        self.assertTrue( g_new._edges.shape[0] > 0)
        self.assertIn("score", g_new._edges.columns)
    
    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    def test_predict_links_all(self):
        g = self.graph_no_feat.embed('rel', embedding_dim=self.d, **self.kwargs)
        g_new = g.predict_links_all(threshold=0)
        self.assertTrue( g_new._edges.shape[0] > 0)
        self.assertIn("score", g_new._edges.columns)

        
    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    def test_chaining(self):
        for name, g in self.graphs:
            logging.debug('name: %s test changing embedding dim with feats' % name)
            g = g.embed('rel', use_feat=True, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed('rel', use_feat=True, embedding_dim=2 * self.d, **self.kwargs)
            self.assertNotEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)

        [g.reset_caches() for _, g in self.graphs]
        for name, g in self.graphs:
            logging.debug('name: %s test changing embedding dim without feats', name)
            g = g.embed('rel', use_feat=False, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed('rel', use_feat=False, embedding_dim=2 * self.d, **self.kwargs)
            self.assertNotEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)

        [g.reset_caches() for _, g in self.graphs]
        for name, g in self.graphs:
            logging.debug('name: %s test relationship change', name)
            g = g.embed(relation='rel', use_feat=False, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed(relation='src', use_feat=False, embedding_dim=self.d, **self.kwargs)
            self.assertEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)
            self.assertNotEqual(np.linalg.norm(g._kg_embeddings - g2._kg_embeddings), 0)

        [g.reset_caches() for _, g in self.graphs]
        for name, g in self.graphs:
            logging.debug('name: %s test relationship change', name)
            g = g.embed(relation='rel', use_feat=False, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed(relation='rel', use_feat=True, embedding_dim=self.d, **self.kwargs)
            self.assertEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)
            self.assertNotEqual(np.linalg.norm(g._kg_embeddings - g2._kg_embeddings), 0)


class TestEmbedCUDF(unittest.TestCase):

    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def setUp(self):
        self.edf = cudf.DataFrame([[0, 1, 0], [1, 2, 0], [2, 0, 1]],
            columns=['src', 'dst', 'rel']
        )
        ndf_no_ids = cudf.DataFrame([['a'], ['a'], ['b']], columns=['feat'])
        ndf_with_ids = cudf.DataFrame([[0, 'a'], [1, 'a'], [2, 'b']],
            columns = ['id', 'feat1']
        )

        self.graph_no_feat = graphistry.edges(self.edf, 'src', 'dst')
        self.graph_with_feat_no_ids = self.graph_no_feat.nodes(ndf_no_ids)
        self.graph_with_feat_with_ids = self.graph_no_feat.nodes(ndf_with_ids, 'id')
        self.graphs = [
            ('no_feat', self.graph_no_feat),
            ('with_feat_no_ids', self.graph_with_feat_no_ids),
            ('with_feat_with_ids', self.graph_with_feat_with_ids)
        ]
        self.d = 4

        self.kwargs = {'n_topics': 6, 'cardinality_threshold':10, 'epochs': 1, 'sample_size':10, 'num_steps':10}
        

    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def test_embed_out_basic(self):
        for name, g in self.graphs:
            g = g.embed('rel', embedding_dim=self.d, **self.kwargs)
            num_nodes = len(set(g._edges['src'] + g._edges['dst']))
            logging.debug('name: %s basic tests', name)
            self.assertEqual(g._edges.shape, self.edf.shape)
            self.assertEqual(set(g._edges[g._relation]), set(g._edges['rel']))
            self.assertEqual(g._kg_embeddings.shape,(num_nodes, self.d))


    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def test_predict_links(self):
        source = pd.Series([0,2])
        relation = None
        destination = pd.Series([1])
        g = self.graph_no_feat.embed('rel', embedding_dim=self.d, **self.kwargs)

        g_new = g.predict_links(source, relation, destination, threshold=0, anomalous=False)
        self.assertTrue( g_new._edges.shape[0] > 0)
        self.assertIn("score", g_new._edges.columns)
        
        g_new = g.predict_links(source, relation, destination, threshold=1, anomalous=True)
        self.assertTrue( g_new._edges.shape[0] > 0)
        self.assertIn("score", g_new._edges.columns)
    
    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def test_predict_links_all(self):
        g = self.graph_no_feat.embed('rel', embedding_dim=self.d, **self.kwargs)
        g_new = g.predict_links_all(threshold=0)
        self.assertTrue( g_new._edges.shape[0] > 0)
        self.assertIn("score", g_new._edges.columns)

        
    @pytest.mark.skipif(not dep_flag, reason="requires ai feature dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def test_chaining(self):
        for name, g in self.graphs:
            logging.debug('name: %s test changing embedding dim with feats' % name)
            g = g.embed('rel', use_feat=True, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed('rel', use_feat=True, embedding_dim=2 * self.d, **self.kwargs)
            self.assertNotEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)

        [g.reset_caches() for _, g in self.graphs]
        for name, g in self.graphs:
            logging.debug('name: %s test changing embedding dim without feats', name)
            g = g.embed('rel', use_feat=False, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed('rel', use_feat=False, embedding_dim=2 * self.d, **self.kwargs)
            self.assertNotEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)

        [g.reset_caches() for _, g in self.graphs]
        for name, g in self.graphs:
            logging.debug('name: %s test relationship change', name)
            g = g.embed(relation='rel', use_feat=False, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed(relation='src', use_feat=False, embedding_dim=self.d, **self.kwargs)
            self.assertEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)
            self.assertNotEqual(np.linalg.norm(g._kg_embeddings - g2._kg_embeddings), 0)

        [g.reset_caches() for _, g in self.graphs]
        for name, g in self.graphs:
            logging.debug('name: %s test relationship change', name)
            g = g.embed(relation='rel', use_feat=False, embedding_dim=self.d, **self.kwargs)
            g2 = g.embed(relation='rel', use_feat=True, embedding_dim=self.d, **self.kwargs)
            self.assertEqual(g._kg_embeddings.shape, g2._kg_embeddings.shape)
            self.assertNotEqual(np.linalg.norm(g._kg_embeddings - g2._kg_embeddings), 0)



if __name__ == "__main__":
    unittest.main()

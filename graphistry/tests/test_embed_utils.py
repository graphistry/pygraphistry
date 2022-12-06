import pytest
import pandas as pd
import unittest
import graphistry

edf = pd.DataFrame([[0, 1, 0], [1, 2, 0], [2, 0, 1]],
        columns=['src', 'dst', 'rel']
)
ndf_no_ids = pd.DataFrame([['a'], ['a'], ['b']], columns=['feat'])
ndf_with_ids = pd.DataFrame([[0, 'a'], [1, 'a'], [2, 'b']],
        columns = ['id', 'feat1']
)

graph_no_feat = graphistry.edges(edf, 'src', 'dst')
graph_with_feat_no_ids = graph_no_feat.nodes(ndf_no_ids)
graph_with_feat_with_ids = graph_no_feat.nodes(ndf_with_ids, 'id')
       
class TestEmbed(unittest.TestCase):

    def test_embed_out_basic(self):
        d = 4
        g = graph_no_feat.embed('rel', embedding_dim=d, epochs=1)
        num_nodes = len(set(g._edges['src'] + g._edges['dst']))

        self.assertEqual(g._edges.shape, edf.shape)
        self.assertEqual(set(g._edges[g._relation]), set(graph_no_feat._edges['rel']))
        self.assertEqual(g._embeddings.shape,(num_nodes, d))

        g = graph_with_feat_no_ids.embed('rel', use_feat=True, embedding_dim=d, epochs=1)
        num_nodes = len(set(g._edges['src'] + g._edges['dst']))

        self.assertEqual(g._edges.shape, edf.shape)
        self.assertEqual(set(g._edges[g._relation]), set(graph_with_feat_no_ids._edges['rel']))
        self.assertEqual(g._embeddings.shape, (num_nodes, d))
        self.assertEqual(len(g._node_features), num_nodes)


        g = graph_with_feat_with_ids.embed('rel', use_feat=True, embedding_dim=d, epochs=1)
        num_nodes = len(set(g._edges['src'] + g._edges['dst']))

        self.assertEqual(g._edges.shape, edf.shape)
        self.assertEqual(set(g._edges[g._relation]), set(graph_with_feat_with_ids._edges['rel']))
        self.assertEqual(g._embeddings.shape, (num_nodes, d))
        self.assertEqual(len(g._node_features), num_nodes)


if __name__ == "__main__":
    unittest.main()

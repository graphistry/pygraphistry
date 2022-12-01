import pytest
import pandas as pd
import unittest
import graphistry

edf = pd.DataFrame([[0, 1, 0], [1, 2, 0], [2, 0, 1]],
        columns=['src', 'dst', 'rel']
)

ndf = pd.DataFrame([[0, 'a'], [1, 'a'], [2, 'b']],
        columns = ['id', 'feat1']
)

graph_no_feat = graphistry.edges(edf, 'src', 'dst')
graph_with_feat = graph_no_feat.nodes(ndf, 'id')
       
class TestEmbed(unittest.TestCase):
    def test_embed_out_basic(self):
        d = 4
        g = graph_no_feat.embed('rel', embedding_dim=d, epochs=1)
        num_nodes = len(set(g._edges['src'] + g._edges['dst']))

        # test num_edges
        self.assertEqual(g._edges.shape, edf.shape)
        # test num_unique_relations
        self.assertEqual(
                set(g._edges[g._relation]), 
                set(graph_no_feat._edges['rel'])
        )
        # test embedding shape
        self.assertEqual(
                g._embeddings.shape,
                (num_nodes, d)
        )

        g = graph_with_feat.embed(
                'rel', 
                use_feat=True, 
                X=['feat1'],
                embedding_dim=d,
                epochs=1
        )
        num_nodes = len(set(g._edges['src'] + g._edges['dst']))

        # test num_edges
        self.assertEqual(g._edges.shape, edf.shape)
        # test num_unique_relations
        self.assertEqual(
                set(g._edges[g._relation]), 
                set(graph_no_feat._edges['rel'])
        )
        # test embedding shape
        self.assertEqual(
                g._embeddings.shape,
                (num_nodes, d)
        )
        # test feature dim
        self.assertEqual(len(g._node_features), num_nodes)


if __name__ == "__main__":
    unittest.main()

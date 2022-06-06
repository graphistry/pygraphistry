import graphistry, logging, os, pandas as pd, pytest
from common import NoAuthTestCase
from graphistry.constants import SRC, DST, NODE
from graphistry.plugins.cugraph import (
    SRC_CUGRAPH, DST_CUGRAPH, NODE_CUGRAPH,
    from_cugraph, to_cugraph,
    compute_cugraph, layout_cugraph,
    compute_algs,
    node_compute_algs_to_attr,  # edge_compute_algs_to_attr, graph_compute_algs_to_attr
    layout_algs,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


test_cugraph = "TEST_CUGRAPH" in os.environ and os.environ["TEST_CUGRAPH"] == "1"

####################


edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4)]
names = ["my", "list", "of", "five", "edges"]
edges_df = pd.DataFrame({
    'a': [x for x, y in edges],
    'b': [y for x, y in edges]
})

nodes = [0, 1, 2, 3, 4]
names_v = ["eggs", "spam", "ham", "bacon", "yello"]

edges2_df = pd.DataFrame({
    'a': ['c', 'd', 'a', 'b', 'b'],
    'b': ['d', 'a', 'b', 'c', 'c'],
    'v1': ['cc', 'dd', 'aa', 'bb', 'bb2'],
    'i': [2, 4, 6, 8, 10]
})
nodes2_df = pd.DataFrame({
    'n': ['a', 'c', 'b', 'd'],
    'v': ['aa', 'cc', 'bb', 'dd'],
    'i': [2, 4, 6, 8]
})

edges3_df = pd.DataFrame({
    'a': ['c', 'd', 'a'],
    'b': ['d', 'a', 'b'],
    'v1': ['cc', 'dd', 'aa'],
    'i': [2, 4, 6]
})
nodes3_df = pd.DataFrame({
    'n': ['a', 'b', 'c', 'd'],
    't': [0, 1, 0, 1]
})

@pytest.mark.skipif(not test_cugraph, reason="Requires TEST_CUGRAPH=1")
class Test_from_cugraph(NoAuthTestCase):

    def test_minimal_edges(self):
        import cudf, cugraph
        G = cugraph.Graph()
        G.from_pandas_edgelist(edges_df, 'a', 'b')
        g = graphistry.from_cugraph(G, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert g._source is not None and g._destination is not None
        assert g._source == 'a'
        assert g._destination == 'b'
        assert g._edges is not None
        assert isinstance(g._edges, cudf.DataFrame)
        assert len(g._edges) == len(edges)
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)
        assert g._edges[g._source].dtype == edges_df['a'].dtype
        assert g._edges[g._destination].dtype == edges_df['b'].dtype

    def test_minimal_attributed_edges(self):
        import cudf, cugraph
        G = cugraph.Graph()
        edges_w = edges_df.assign(w=1)
        G.from_pandas_edgelist(edges_w, 'a', 'b', 'w')
        g = graphistry.from_cugraph(G, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert g._source == 'a'
        assert g._destination == 'b'
        assert g._edges is not None
        assert isinstance(g._edges, cudf.DataFrame)
        assert len(g._edges) == len(edges)
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)
        assert (g._edges['w'].to_pandas() == edges_w['w']).all()

    def test_merge_existing_edges_pandas(self):

        import cudf, cugraph
        G = cugraph.Graph()
        edges_w = edges_df.assign(name=names, idx=['aa', 'bb', 'cc', 'dd', 'ee'])
        G.from_pandas_edgelist(edges_w, 'a', 'b', 'w')

        g = (graphistry
            .nodes(pd.DataFrame({
                'i': nodes,
                'v1': [f'1_{x}' for x in names_v]
            }), 'i')
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
                'i': ['aaa','bbb','ccc','ddd','eee']
            }), 's', 'd', 'i')
        )
        g2 = g.from_cugraph(G)
        assert len(g._nodes) == len(g2._nodes)
        assert len(g._edges) == len(g2._edges)
        assert sorted(g2._nodes.columns) == sorted(g._nodes.columns)
        assert sorted(g2._edges.columns) == sorted(['s', 'd', 'i', 'idx', 'name'])

    def test_nodes_str_ids(self):
        g = (graphistry
            .nodes(
                pd.DataFrame({
                    'n': ['a', 'b', 'c']
                }), 'n')
            .edges(
                pd.DataFrame({
                    's': ['a', 'b', 'c'],
                    'd': ['b', 'c', 'a']
                }), 's', 'd')
        )
        G = g.to_cugraph()
        g2 = g.from_cugraph(G)

        assert len(g2._nodes) == len(g._nodes)
        assert g2._node == g._node
        assert sorted(g2._nodes.columns) == sorted(['n'])

        assert len(g2._edges) == len(g._edges)
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert sorted(g2._edges.columns) == sorted(['s', 'd'])

    def test_edges_named(self):
        g = graphistry.edges(edges2_df, 'a', 'b').nodes(nodes2_df, 'n')
        G = g.to_cugraph()
        g2 = g.from_cugraph(G)
        assert len(g2._nodes) == len(g._nodes)
        assert len(g2._edges) == len(g._edges)
        g2n = g2._nodes.sort_values(by='n').reset_index(drop=True)
        assert g2n.equals(pd.DataFrame({
            'n': ['a', 'b', 'c', 'd'],
            'v': ['aa', 'bb', 'cc', 'dd'],
            'i': [2, 6, 4, 8]
        }))

    def test_edges_named_without_nodes(self):
        g = graphistry.edges(edges2_df, 'a', 'b')
        G = g.to_cugraph()
        g2 = g.from_cugraph(G)
        assert len(g2._nodes) == len(nodes2_df)
        assert len(g2._edges) == len(g._edges)
        g2n = g2._nodes.sort_values(by=g2._node).reset_index(drop=True)
        assert g2n.equals(pd.DataFrame({
            g2._node: ['a', 'b', 'c', 'd'],
        }))

@pytest.mark.skipif(not test_cugraph, reason="Requires TEST_CUGRAPH=1")
class Test_to_cugraph(NoAuthTestCase):

    def test_minimal_edges(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
        )
        G = g.to_cugraph()
        logger.debug('G: %s', G)
        g2 = graphistry.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_CUGRAPH
        assert g2._destination == DST_CUGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes
        }))
        assert g2._node == NODE

    def test_minimal_edges_renamed(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
        )
        G = g.to_cugraph()
        logger.debug('G: %s', G)
        g2 = g.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == 's'
        assert g2._destination == 'd'
        assert g2._edge is None
        assert g2._edges.equals(g._edges)
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes
        }))
        assert g2._node == NODE

    def test_minimal_edges_str(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }).astype(str), 's', 'd')
        )
        G = g.to_cugraph()
        logger.debug('G: %s', G)
        g2 = graphistry.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == 'source'
        assert g2._destination == 'target'
        assert g2._edge is None
        assert g2._edges.rename(columns={'source': 's', 'target': 'd'}).equals(g._edges)
        logger.debug('g2._nodes: %s', g2._nodes)
        logger.debug('g2._nodes dtypes: %s', g2._nodes.dtypes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: pd.Series(nodes).astype(str)
        }))
        assert g2._node == NODE

    def test_nodes(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        G = g.to_cugraph()
        logger.debug('ig: %s', G)
        g2 = graphistry.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_CUGRAPH
        assert g2._destination == DST_CUGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes,
            'names': names_v
        }))
        assert g2._node == NODE

    def test_nodes_renamed(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        G = g.to_cugraph()
        logger.debug('ig: %s', G)
        g2 = g.from_cugraph(G)
        logger.debug('g2 edges: %s', g2._edges)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == 's'
        assert g2._destination == 'd'
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': nodes,
            'names': names_v
        }))
        assert g2._node == 'n'

    def test_drop_nodes(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        G = g.to_cugraph(include_nodes=False)
        logger.debug('G: %s', G)
        g2 = graphistry.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_CUGRAPH
        assert g2._destination == DST_CUGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({NODE: nodes}))
        assert g2._node == NODE

    def test_nodes_undirected(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': nodes,
                'names': names_v
            }), 'n')
        )
        G = g.to_cugraph(directed=False)
        logger.debug('G: %s', G)
        g2 = g.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': nodes,
            'names': names_v
        }))
        assert g2._node == 'n'

    def test_nodes_undirected_str(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': ['a', 'b', 'c'],
                'd': ['b', 'c', 'a'],
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': ['a', 'b', 'c'],
                'names': ['x', 'y', 'z']
            }), 'n')
        )
        G = g.to_cugraph(directed=False)
        logger.debug('G: %s', G)
        #ig.vs['cluster'] = ig.community_infomap().membership
        g2 = g.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': ['a', 'b', 'c'],
            'names': ['x', 'y', 'z'],
            #'cluster': [0, 0, 0]
        }))
        assert g2._node == 'n'

    def test_nodes_undirected_str_attributed(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': ['a', 'b', 'c'],
                'd': ['b', 'c', 'a'],
                'v': ['x', 'y', 'z']
            }), 's', 'd')
            .nodes(pd.DataFrame({
                'n': ['a', 'b', 'c'],
                'names': ['x', 'y', 'z']
            }), 'n')
        )
        G = g.to_cugraph(directed=False)
        logger.debug('G: %s', G)
        #ig.vs['cluster'] = ig.community_infomap().membership
        g2 = g.from_cugraph(G)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            'n': ['a', 'b', 'c'],
            'names': ['x', 'y', 'z'],
            #'cluster': [0, 0, 0]
        }))
        assert g2._node == 'n'


@pytest.mark.skipif(not test_cugraph, reason="Requires TEST_CUGRAPH=1")
class Test_curaph_usage(NoAuthTestCase):

    def test_enrich_with_stat(self):
        import cugraph
        g = graphistry.edges(pd.DataFrame({
            's': [x[0] for x in edges],
            'd': [x[1] for x in edges]
        }), 's', 'd')
        G = g.to_cugraph()
        #ig.vs['spinglass'] = ig.community_spinglass(spins=3).membership
        nodes_gdf = cugraph.pagerank(G)
        assert len(nodes_gdf) == len(g.materialize_nodes()._nodes)
        assert 'pagerank' in nodes_gdf
        assert 'vertex' in nodes_gdf
        
    def test_enrich_with_stat_direct(self):
        import cudf, cugraph
        g = (
            graphistry.edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges]
            }), 's', 'd')
            .materialize_nodes()
        )
        g2 = g.nodes(cudf.from_pandas(g._nodes).assign(
            #TODO this seems unsafe: no guarantee ._nodes order is ig vertices order?
            pagerank=cugraph.pagrank(g.to_cugraph())['pagerank']
        ))
        logger.debug('g2 nodes: %s', g2._nodes)
        logger.debug('g2 edges: %s', g2._edges)
        assert g2._edges.shape == g2._edges.shape
        assert len(g2._nodes) == len(nodes)
        assert sorted(g2._nodes.columns) == sorted([g2._node, 'pagerank'])


@pytest.mark.skipif(not test_cugraph, reason="Requires TEST_CUGRAPH=1")
class Test_cugraph_compute(NoAuthTestCase):

    def test_node_calls(self):

        import cudf, cugraph

        overrides = {
            #'bipartite_projection': {
            #    'params': {'which': 0}
            #},
            #'community_leading_eigenvector': {
            #    'directed': False
            #},
            #'community_leiden': {
            #    'directed': False
            #},
            #'community_multilevel': {
            #    'directed': False
            #},
            #'gomory_hu_tree': {
            #    'directed': False
            #}
        }

        skiplist = [
            #'eigenvector_centrality'
        ]

        edges3_gdf = cudf.from_cudf(edges3_df)

        g = graphistry.edges(edges3_gdf, 'a', 'b').materialize_nodes()
        for alg in [x for x in node_compute_algs_to_attr.keys()]:
            if alg not in skiplist:
                opts = overrides[alg] if alg in overrides else {}
                logger.debug('alg "%s", opts=(%s)', alg, opts)
                g2 = compute_cugraph(g, alg, **opts)
                assert g2 is not None
                assert len(g2._nodes) == len(g._nodes)
                assert alg in g2._nodes
                assert len(g2._nodes.columns) == len(g._nodes.columns) + 1
                assert g2._edges.shape == g._edges.shape

    def test_all_calls(self):

        import cudf, cugraph

        overrides = {
            #'bipartite_projection': {
            #    'params': {'which': 0}
            #},
            #'community_leading_eigenvector': {
            #    'directed': False
            #},
            #'community_leiden': {
            #    'directed': False
            #},
            #'community_multilevel': {
            #    'directed': False
            #},
            #'gomory_hu_tree': {
            #    'directed': False
            #}
        }

        skiplist = [
            #'eigenvector_centrality'
        ]

        edges3_gdf = cudf.from_cudf(edges3_df)

        g = graphistry.edges(edges3_gdf, 'a', 'b').materialize_nodes()
        for alg in [x for x in compute_algs]:
            if alg not in skiplist:
                opts = overrides[alg] if alg in overrides else {}
                logger.debug('alg "%s", opts=(%s)', alg, opts)
                assert compute_cugraph(g, alg, **opts) is not None



@pytest.mark.skipif(not test_cugraph, reason="Requires TEST_CUGRAPH=1")
class Test_cugraph_layouts(NoAuthTestCase):

    def test_all_calls(self):

        import cudf, cugraph

        overrides = {
            #'bipartite': {
            #    'params': {'types': 't'}
            #}
        }

        g = graphistry.edges(cudf.DataFrame(edges3_df), 'a', 'b').nodes(cudf.DataFrame(nodes3_df), 'n')
        for alg in layout_algs:
            opts = overrides[alg] if alg in overrides else {}
            logger.debug('alg "%s", opts=(%s)', alg, opts)
            g2 = layout_cugraph(g, alg, **opts)
            logger.debug('g._edges: %s', g._edges)
            logger.debug('2._edges: %s', g2._edges)
            assert len(g2._nodes) == len(g._nodes)
            assert g2._edges.equals(g._edges)

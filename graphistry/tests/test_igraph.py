import graphistry, logging, pandas as pd, pytest
from common import NoAuthTestCase
from graphistry.constants import SRC, DST, NODE
from graphistry.plugins.igraph import SRC_IGRAPH, DST_IGRAPH

try:
    import igraph
    has_igraph = True
except:
    has_igraph = False

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


####################


edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4)]
names = ["my", "list", "of", "five", "edges"]

nodes = [0, 1, 2, 3, 4]
names_v = ["eggs", "spam", "ham", "bacon", "yello"]


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_from_igraph(NoAuthTestCase):

    def test_minimal_edges(self):
        ig = igraph.Graph(edges)
        g = graphistry.from_igraph(ig, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)

    def test_minimal_attributed_edges(self):
        ig = igraph.Graph(edges)
        ig.es["name"] = names
        g = graphistry.from_igraph(ig, load_nodes=False)
        assert g._nodes is None and g._node is None
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)
        assert (g._edges['name'] == pd.Series(names)).all()

    def test_minimal_nodes(self):
        ig = igraph.Graph(edges)
        g = graphistry.from_igraph(ig)
        assert g._node is not None and g._nodes is not None
        assert len(g._nodes) == len(nodes)
        assert (g._nodes[g._node].sort_values() == pd.Series(nodes)).all()
        assert g._nodes.columns == [ g._node ]
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)

    def test_minimal_nodes_attributed(self):
        ig = igraph.Graph(edges)
        ig.vs["name"] = names_v
        g = graphistry.from_igraph(ig)
        assert g._node is not None and g._nodes is not None
        assert len(g._nodes) == len(nodes)
        assert (g._nodes[g._node].sort_values() == pd.Series(nodes)).all()
        assert (g._nodes['name'] == pd.Series(names_v)).all()
        assert sorted(g._nodes.columns) == sorted([ g._node, 'name' ])
        assert len(g._edges) == len(edges)
        assert g._source is not None and g._destination is not None
        assert len(g._edges[g._source].dropna()) == len(edges)
        assert len(g._edges[g._destination].dropna()) == len(edges)

    def test_merge_existing_nodes(self):
        ig = igraph.Graph(edges)
        ig.vs["idx"] = ['a', 'b', 'c', 'd', 'e']
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
        g2 = g.from_igraph(ig)
        assert len(g._nodes) == len(g2._nodes)
        assert len(g._edges) == len(g2._edges)
        assert sorted(g2._nodes.columns) == sorted(['idx', 'i', 'v1'])
        assert sorted(g2._edges.columns) == sorted(g._edges.columns)

    def test_merge_existing_nodes_attributed(self):
        ig = igraph.Graph(edges)
        ig.vs["name"] = names_v
        ig.vs["idx"] = ['a', 'b', 'c', 'd', 'e']
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
        g2 = g.from_igraph(ig)
        assert len(g._nodes) == len(g2._nodes)
        assert len(g._edges) == len(g2._edges)
        assert sorted(g2._nodes.columns) == sorted(['idx', 'i', 'v1', 'name'])
        assert sorted(g2._edges.columns) == sorted(g._edges.columns)

    def test_merge_existing_edges(self):
        ig = igraph.Graph(edges)
        ig.es["name"] = names
        ig.es["idx"] = ['aa', 'bb', 'cc', 'dd', 'ee']
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
        g2 = g.from_igraph(ig)
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
        ig = g.to_igraph()
        ig.vs['spinglass'] = ig.community_spinglass(spins=3).membership
        g2 = g.from_igraph(ig)

        assert len(g2._nodes) == len(g._nodes)
        assert g2._node == g._node
        assert sorted(g2._nodes.columns) == sorted(['n', 'spinglass'])

        assert len(g2._edges) == len(g._edges)
        assert g2._source == g._source
        assert g2._destination == g._destination
        assert sorted(g2._edges.columns) == sorted(['s', 'd'])


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_to_igraph(NoAuthTestCase):

    def test_minimal_edges(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
        )
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes,
            'name': nodes
        }))
        assert g2._node == NODE

    def test_minimal_edges_renamed(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }), 's', 'd')
        )
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = g.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == 's'
        assert g2._destination == 'd'
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes,
            'name': nodes
        }))
        assert g2._node == NODE

    def test_minimal_edges_str(self):
        g = (graphistry
            .edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges],
            }).astype(str), 's', 'd')
        )
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        logger.debug('g2._nodes dtypes: %s', g2._nodes.dtypes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes,
            'name': pd.Series(nodes).astype(str)
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
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes,
            'name': nodes,
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
        ig = g.to_igraph()
        logger.debug('ig: %s', ig)
        g2 = g.from_igraph(ig)
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
            }))
        )
        ig = g.to_igraph(include_nodes=False)
        logger.debug('ig: %s', ig)
        g2 = graphistry.from_igraph(ig)
        assert g2._edges.shape == g._edges.shape
        assert g2._source == SRC_IGRAPH
        assert g2._destination == DST_IGRAPH
        assert g2._edge is None
        logger.debug('g2._nodes: %s', g2._nodes)
        assert g2._nodes.equals(pd.DataFrame({
            NODE: nodes,
            'name': nodes
        }))
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
        ig = g.to_igraph(directed=False)
        logger.debug('ig: %s', ig)
        g2 = g.from_igraph(ig)
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


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_igraph_usage(NoAuthTestCase):

    def test_enrich_with_stat(self):
        g = graphistry.edges(pd.DataFrame({
            's': [x[0] for x in edges],
            'd': [x[1] for x in edges]
        }), 's', 'd')
        ig = g.to_igraph()
        ig.vs['spinglass'] = ig.community_spinglass(spins=3).membership
        g2 = g.from_igraph(ig)
        logger.debug('g2 nodes: %s', g2._nodes)
        logger.debug('g2 edges: %s', g2._edges)
        assert g2._edges.shape == g2._edges.shape
        assert len(g2._nodes) == len(nodes)
        assert sorted(g2._nodes.columns) == sorted([g2._node, 'name', 'spinglass'])

    def test_enrich_with_stat_direct(self):
        g = (
            graphistry.edges(pd.DataFrame({
                's': [x[0] for x in edges],
                'd': [x[1] for x in edges]
            }), 's', 'd')
            .materialize_nodes()
        )
        g2 = g.nodes(g._nodes.assign(
            #TODO this seems unsafe: no guarantee ._nodes order is ig vertices order?
            spinglass=g.to_igraph().community_spinglass(spins=3).membership
        ))
        logger.debug('g2 nodes: %s', g2._nodes)
        logger.debug('g2 edges: %s', g2._edges)
        assert g2._edges.shape == g2._edges.shape
        assert len(g2._nodes) == len(nodes)
        assert sorted(g2._nodes.columns) == sorted([g2._node, 'spinglass'])

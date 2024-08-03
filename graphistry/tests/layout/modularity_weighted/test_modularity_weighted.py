import os

import graphistry, pandas as pd, pytest
from graphistry.tests.common import NoAuthTestCase


try:
    import igraph
    has_igraph = True
except:
    has_igraph = False


test_cugraph = "TEST_CUGRAPH" in os.environ and os.environ["TEST_CUGRAPH"] == "1"


chain = [
    {'s': 'a', 'd': 'b'},
    {'s': 'b', 'd': 'c'},
    {'s': 'c', 'd': 'd'},
    {'s': 'd', 'd': 'e'}
]


@pytest.mark.skipif(not has_igraph, reason="Requires igraph")
class Test_from_igraph(NoAuthTestCase):

    def test_minimal_edges(self):

        g = graphistry.edges(pd.DataFrame(chain))
        g2 = g.modularity_weighted_layout()

        assert 'community_multilevel' in g2._nodes
        assert 'weight' in g2._edges
        assert len(g2._edges.dropna()) == len(chain)


@pytest.mark.skipif(not test_cugraph, reason="Requires cugraph")
class Test_from_cudf(NoAuthTestCase):


    def test_minimal_edges(self):

        import cudf
        g = graphistry.edges(cudf.DataFrame(chain))
        g2 = g.modularity_weighted_layout()

        assert 'louvain' in g2._nodes
        assert 'weight' in g2._edges
        assert len(g2._edges.dropna()) == len(chain)

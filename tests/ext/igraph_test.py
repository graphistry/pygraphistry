import pytest
import igraph

from graphistry.ext.igraph_to_arrow import to_arrow

def test_to_arrow():
    graph = igraph.Graph.Tree(2, 10)
    (edges, nodes) = to_arrow(graph)
    assert len(graph.vs) == len(nodes)
    assert len(graph.es) == len(edges)
    pass

if __name__ == '__main__':
    pytest.main()

import pytest
import networkx

from graphistry.ext.networkx_to_arrow import to_arrow

def test_to_arrow():
    graph = networkx.random_lobster(100, 0.9, 0.9)
    (edges, nodes) = to_arrow(graph)
    assert len(graph.nodes()) == len(nodes)
    assert len(graph.edges()) == len(edges)
    pass

if __name__ == '__main__':
    pytest.main()

"""Tests to understand undirected edge semantics in chain."""
import pandas as pd
import pytest
import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected
from graphistry.compute.chain import Chain
from tests.gfql.ref.conftest import to_set


@pytest.fixture
def linear_graph():
    """Linear graph: a -> b -> c -> d"""
    nodes = pd.DataFrame({'id': ['a', 'b', 'c', 'd']})
    edges = pd.DataFrame({
        'src': ['a', 'b', 'c'],
        'dst': ['b', 'c', 'd'],
        'eid': [0, 1, 2],
    })
    return graphistry.nodes(nodes, 'id').edges(edges, 'src', 'dst')


class TestSingleHopUndirected:
    """Single hop undirected edge tests."""

    def test_undirected_from_b_reaches_a_and_c(self, linear_graph):
        """Undirected from b should reach both a and c."""
        chain = Chain([n({'id': 'b'}), e_undirected(), n()])
        result = linear_graph.gfql(chain)
        node_ids = to_set(result._nodes['id'])
        # b is connected to a (via e0) and c (via e1)
        assert node_ids == {'a', 'b', 'c'}, f"Got {node_ids}"

    def test_undirected_from_b_uses_both_edges(self, linear_graph):
        """Undirected from b should include edges e0 and e1."""
        chain = Chain([n({'id': 'b'}), e_undirected(), n()])
        result = linear_graph.gfql(chain)
        edge_ids = to_set(result._edges['eid'])
        # e0 (a->b) and e1 (b->c) both touch b
        assert edge_ids == {0, 1}, f"Got {edge_ids}"

    def test_forward_from_b_only_reaches_c(self, linear_graph):
        """Forward from b should only reach c (via e1)."""
        chain = Chain([n({'id': 'b'}), e_forward(), n()])
        result = linear_graph.gfql(chain)
        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'b', 'c'}, f"Got {node_ids}"

    def test_forward_from_b_only_uses_e1(self, linear_graph):
        """Forward from b should only use edge e1."""
        chain = Chain([n({'id': 'b'}), e_forward(), n()])
        result = linear_graph.gfql(chain)
        edge_ids = to_set(result._edges['eid'])
        assert edge_ids == {1}, f"Got {edge_ids}"

    def test_reverse_from_b_only_reaches_a(self, linear_graph):
        """Reverse from b should only reach a (via e0)."""
        chain = Chain([n({'id': 'b'}), e_reverse(), n()])
        result = linear_graph.gfql(chain)
        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b'}, f"Got {node_ids}"

    def test_reverse_from_b_only_uses_e0(self, linear_graph):
        """Reverse from b should only use edge e0."""
        chain = Chain([n({'id': 'b'}), e_reverse(), n()])
        result = linear_graph.gfql(chain)
        edge_ids = to_set(result._edges['eid'])
        assert edge_ids == {0}, f"Got {edge_ids}"


class TestTwoHopUndirected:
    """Two hop undirected tests."""

    def test_two_hop_undirected_from_b_nodes(self, linear_graph):
        """Two hops undirected from b - nodes reached."""
        chain = Chain([n({'id': 'b'}), e_undirected(), n(), e_undirected(), n()])
        result = linear_graph.gfql(chain)
        node_ids = to_set(result._nodes['id'])
        # Step 1: b -> {a, c}
        # Step 2: from a -> b (via e0), from c -> {b, d} (via e1, e2)
        # Without edge uniqueness: all nodes {a, b, c, d}
        assert node_ids == {'a', 'b', 'c', 'd'}, f"Got {node_ids}"

    def test_two_hop_undirected_from_b_edges(self, linear_graph):
        """Two hops undirected from b - edges used."""
        chain = Chain([n({'id': 'b'}), e_undirected(), n(), e_undirected(), n()])
        result = linear_graph.gfql(chain)
        edge_ids = to_set(result._edges['eid'])
        # Step 1: e0 (a-b), e1 (b-c)
        # Step 2: from {a,c}, edges touching them: e0, e1, e2
        # All edges should be used
        assert edge_ids == {0, 1, 2}, f"Got {edge_ids}"

    def test_two_hop_forward_from_b_nodes(self, linear_graph):
        """Two hops forward from b - nodes reached."""
        chain = Chain([n({'id': 'b'}), e_forward(), n(), e_forward(), n()])
        result = linear_graph.gfql(chain)
        node_ids = to_set(result._nodes['id'])
        # Step 1: b -> c (via e1)
        # Step 2: c -> d (via e2)
        assert node_ids == {'b', 'c', 'd'}, f"Got {node_ids}"

    def test_two_hop_forward_from_b_edges(self, linear_graph):
        """Two hops forward from b - edges used."""
        chain = Chain([n({'id': 'b'}), e_forward(), n(), e_forward(), n()])
        result = linear_graph.gfql(chain)
        edge_ids = to_set(result._edges['eid'])
        assert edge_ids == {1, 2}, f"Got {edge_ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

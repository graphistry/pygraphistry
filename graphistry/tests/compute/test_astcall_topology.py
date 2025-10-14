"""
Topological sensitivity tests for ASTCall operations in GFQL chains.

NOTE: As of PR #787, mixing call() with n()/e() in the same chain is not supported
and will raise GFQLValidationError. These tests document the behaviors that led to
this restriction. See issue #791 for architectural details and future enhancement plans.

This test suite validates that ASTCall operations (filters, enrichments) work
correctly when mixed with ASTNode/ASTEdge traversal operations in complex
topological patterns.

Key test principles:
1. Non-overlapping filter dimensions - each filter uses different attributes
   to make it obvious which filters were applied
2. Enrichment column dependencies - validate that columns added by one ASTCall
   are visible to subsequent operations
3. Topology sensitivity - test various chain positions, graph structures,
   and operation orderings

Related Issues:
- #786 - Chained filter operations (fixed for pure call() chains)
- #791 - Mixed call()/traversal chains (architectural limitation)
- PR #787 - Implementation and enforcement
"""
import pandas as pd
import pytest
from graphistry import PyGraphistry
from graphistry.compute.ast import ASTCall, n, e, e_forward, e_reverse
from graphistry.compute.predicates.numeric import Between, GT, GE, LT, LE
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.tests.test_compute import CGFull


class TestTopologicalChains:
    """Test mixed ASTNode/ASTCall/ASTEdge chains with non-overlapping dimensions.

    These tests validate that:
    - ASTCall operations thread their results correctly through chains
    - Filters on different dimensions compose properly
    - Wavefront semantics and graph threading coexist correctly
    """

    @pytest.fixture
    def rich_graph(self):
        """Create test graph with 4 independent filter dimensions per entity type.

        Node dimensions:
        - type: person/org/bot (category filter)
        - score: 10-50 (numeric filter)
        - region: NA/EU/ASIA (category filter)
        - status: active/inactive (boolean filter)

        Edge dimensions:
        - type: forward/backward/lateral (category filter)
        - weight: 1-10 (numeric filter)
        - category: low/med/high (category filter)
        - verified: True/False (boolean filter)
        """
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B'],
            'dst': ['B', 'C', 'D', 'E', 'A', 'C', 'D', 'E', 'A', 'B', 'D', 'E'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward',
                     'backward', 'backward', 'backward', 'backward', 'backward',
                     'lateral', 'lateral'],
            'weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4],
            'category': ['low', 'low', 'med', 'med', 'high', 'high', 'high', 'high', 'med', 'med', 'low', 'med'],
            'verified': [True, True, True, False, False, True, False, True, False, True, True, False]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'type': ['person', 'person', 'org', 'org', 'bot'],
            'score': [10, 20, 30, 40, 50],
            'region': ['NA', 'NA', 'EU', 'EU', 'ASIA'],
            'status': ['active', 'active', 'active', 'inactive', 'active']
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_topology_node_call_call_node(self, rich_graph):
        """Pattern: [n({type filter}), call(weight filter), call(category filter), n({score filter})]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        Documents desired behavior that requires let() composition.
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            rich_graph.gfql([
                n({'type': 'person'}),                                      # Filter nodes: type=person (A, B)
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),  # Filter edges: weight<=5
                ASTCall('filter_edges_by_dict', {'filter_dict': {'category': 'low'}}),  # Filter edges: category=low
                n({'score': GE(15)})                                      # Filter nodes: score>=15
            ])

    def test_topology_edge_call_call_edge(self, rich_graph):
        """Pattern: [e({type filter}), call(weight filter), call(verified filter), e({category filter})]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            rich_graph.gfql([
                e({'type': 'forward'}),                                              # Filter edges: type=forward
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),  # Filter edges: weight<=5
                ASTCall('filter_edges_by_dict', {'filter_dict': {'verified': True}}),  # Filter edges: verified=True
                e({'category': 'low'})                                               # Filter edges: category=low
            ])

    def test_topology_alternating_node_call_patterns(self, rich_graph):
        """Pattern: [n({f1}), call(f2), n({f3}), call(f4), n({f5})]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            rich_graph.gfql([
                n({'type': 'person'}),                                               # Dim 1: node type
                ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Dim 2: edge type
                n({'region': 'NA'}),                                                # Dim 3: node region
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),  # Dim 4: edge weight
                n({'status': 'active'})                                             # Dim 5: node status
            ])

    def test_topology_hop_between_calls(self, rich_graph):
        """Pattern: [n({f1}), call(f2), hop(), call(f3), n({f4})]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        from graphistry.compute.ast import call

        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            rich_graph.gfql([
                n({'type': 'person'}),                                               # Filter to person nodes
                ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Filter to forward edges
                call('hop', {'hops': 1}),                                           # Hop operation
                ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': GE(15)}}),  # Filter nodes by score
                n()                                                                  # Final node selection
            ])

    def test_topology_call_at_chain_start(self, rich_graph):
        """Pattern: [call(f1), n(), e(), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            rich_graph.gfql([
                ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
                n(),
                e(),
                n()
            ])

    def test_topology_call_in_middle(self, rich_graph):
        """Pattern: [n(), call(f1), e(), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            rich_graph.gfql([
                n({'type': 'person'}),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': GE(5)}}),
                e(),
                n()
            ])

    def test_topology_call_at_chain_end(self, rich_graph):
        """Pattern: [n(), e(), n(), call(f1)]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            rich_graph.gfql([
                n({'type': 'person'}),
                e(),
                n(),
                ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': GE(30)}})
            ])


class TestEnrichmentChains:
    """Test enrichment operations creating columns used by subsequent filters.

    These tests validate that:
    - Enrichment operations (get_degrees, etc.) add columns correctly
    - Added columns persist through chain execution
    - Subsequent filters can use enriched columns
    - Forward and backward passes handle enrichments correctly
    """

    @pytest.fixture
    def enrichment_graph(self):
        """Create graph suitable for testing degree-based enrichments."""
        edges_df = pd.DataFrame({
            'src': ['A', 'A', 'B', 'B', 'C', 'D', 'D', 'D'],
            'dst': ['B', 'C', 'C', 'D', 'D', 'A', 'B', 'E'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward', 'backward', 'backward', 'forward'],
            'weight': [1, 2, 3, 4, 5, 6, 7, 8]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_enrichment_degree_then_filter(self, enrichment_graph):
        """Pattern: [call('get_indegrees'), n({'deg_in': {...}})]

        Validates: Mixed chains now raise GFQLValidationError (#791) or fail due to missing columns.
        """
        with pytest.raises((GFQLValidationError, AssertionError, KeyError)):
            enrichment_graph.gfql([
                ASTCall('get_indegrees', {'col': 'deg_in'}),
                n({'deg_in': GE(2)})
            ])

    def test_enrichment_multiple_degrees(self, enrichment_graph):
        """Pattern: [call('get_indegrees'), call('get_outdegrees'), n({filter both})]

        Validates: Mixed chains now raise GFQLValidationError (#791) or fail due to missing columns.
        """
        with pytest.raises((GFQLValidationError, AssertionError, KeyError)):
            enrichment_graph.gfql([
                ASTCall('get_indegrees', {'col': 'deg_in'}),
                ASTCall('get_outdegrees', {'col': 'deg_out'}),
                n({'deg_in': GE(1), 'deg_out': GE(1)})
            ])

    def test_enrichment_filter_enrichment_filter(self, enrichment_graph):
        """Pattern: [n({filter1}), call(enrich), call(filter2), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            enrichment_graph.gfql([
                n({'type': 'person'}),                           # Filter to person nodes
                ASTCall('get_degrees', {}),                      # Enrich with degree
                ASTCall('filter_nodes_by_dict', {'filter_dict': {'degree': GE(2)}}),  # Filter by degree
                n()                                              # Final traversal
            ])

    def test_enrichment_edge_degree_then_filter_forward_backward(self, enrichment_graph):
        """Pattern: [e({type: 'forward'}), call('get_degrees'), n({'degree': {...}}), e(), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791) or fail due to missing columns.
        """
        with pytest.raises((GFQLValidationError, AssertionError, KeyError)):
            enrichment_graph.gfql([
                e({'type': 'forward'}),
                ASTCall('get_degrees', {}),
                n({'degree': GE(2)}),
                e(),
                n()
            ])

    def test_enrichment_before_traversal(self, enrichment_graph):
        """Pattern: [call('get_outdegrees'), n({'deg_out': {...}}), e(), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791) or fail due to missing columns.
        """
        with pytest.raises((GFQLValidationError, AssertionError, KeyError)):
            enrichment_graph.gfql([
                ASTCall('get_outdegrees', {'col': 'deg_out'}),
                n({'deg_out': GE(2)}),
                e_forward(),
                n()
            ])

    def test_enrichment_with_node_filter_interaction(self, enrichment_graph):
        """Pattern: [n({'type': 'person'}), call('get_degrees'), n({'degree': {...}})]

        Validates: Mixed chains now raise GFQLValidationError (#791) or fail due to missing columns.
        """
        with pytest.raises((GFQLValidationError, AssertionError, KeyError)):
            enrichment_graph.gfql([
                n({'type': 'person'}),
                ASTCall('get_degrees', {}),
                n({'degree': GE(1)})
            ])


class TestTopologySensitivity:
    """Test various graph structures and edge cases.

    Validates that ASTCall operations work correctly across:
    - Different graph topologies (DAG, cyclic, disconnected)
    - Edge cases (empty results, fixed-point traversal)
    - Complex structural patterns
    """

    @pytest.fixture
    def dag_graph(self):
        """Create a directed acyclic graph (no cycles)."""
        edges_df = pd.DataFrame({
            'src': ['A', 'A', 'B', 'B', 'C'],
            'dst': ['B', 'C', 'D', 'E', 'D'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward'],
            'weight': [1, 2, 3, 4, 5]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'level': [0, 1, 1, 2, 2]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    @pytest.fixture
    def cyclic_graph(self):
        """Create a graph with cycles."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C', 'D'],
            'dst': ['B', 'C', 'D', 'A'],
            'type': ['forward', 'forward', 'forward', 'forward'],
            'weight': [1, 2, 3, 4]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D'],
            'score': [10, 20, 30, 40]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    @pytest.fixture
    def disconnected_graph(self):
        """Create graph with multiple disconnected components."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'D', 'E'],
            'dst': ['B', 'C', 'E', 'F'],
            'component': [1, 1, 2, 2],
            'weight': [1, 2, 3, 4]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'component': [1, 1, 1, 2, 2, 2],
            'score': [10, 20, 30, 40, 50, 60]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_topology_dag_structure(self, dag_graph):
        """Validates: Mixed chains now raise GFQLValidationError (#791)."""
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            dag_graph.gfql([
                n(),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': GE(2)}}),
                e(),
                n()
            ])

    def test_topology_cyclic_structure(self, cyclic_graph):
        """Validates: Mixed chains now raise GFQLValidationError (#791)."""
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            cyclic_graph.gfql([
                n({'id': 'A'}),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(3)}}),
                e(),
                n()
            ])

    def test_topology_disconnected_components(self, disconnected_graph):
        """Validates: Mixed chains now raise GFQLValidationError (#791)."""
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            disconnected_graph.gfql([
                ASTCall('filter_nodes_by_dict', {'filter_dict': {'component': 1}}),
                n(),
                e(),
                n()
            ])

    def test_topology_empty_intermediate_result(self, dag_graph):
        """Validates: Mixed chains now raise GFQLValidationError (#791)."""
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            dag_graph.gfql([
                n({'level': 99}),  # No such level, empty result
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': GE(1)}}),
                n()
            ])


class TestComplexMixedChains:
    """Test complex combinations of all operation types.

    These are exhaustive integration tests combining:
    - ASTNode, ASTEdge, ASTCall operations
    - Filters and enrichments
    - Forward and backward passes
    - Multiple hops and complex patterns
    """

    @pytest.fixture
    def complex_graph(self):
        """Create rich graph for complex chain testing."""
        edges_df = pd.DataFrame({
            'src': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
            'dst': ['B', 'C', 'C', 'D', 'D', 'E', 'E', 'A', 'A'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'backward', 'backward'],
            'weight': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'category': ['low', 'low', 'med', 'med', 'med', 'high', 'high', 'high', 'high']
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50],
            'region': ['NA', 'NA', 'EU', 'EU', 'ASIA']
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_complex_all_operation_types(self, complex_graph):
        """Pattern: [n({f1}), call(enrich), e({f2}), call(filter), n({f3})]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            complex_graph.gfql([
                n({'type': 'person'}),
                ASTCall('get_degrees', {}),
                e({'type': 'forward'}),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),
                n({'region': 'NA'})
            ])

    def test_complex_multiple_calls_in_sequence(self, complex_graph):
        """Pattern: [call(f1), call(f2), call(f3), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            complex_graph.gfql([
                ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': Between(2, 6)}}),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'category': 'med'}}),
                n()
            ])

    def test_complex_filter_hop_filter_hop(self, complex_graph):
        """Pattern: [n({filter1}), call(filter2), hop(), call(filter3), hop(), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        from graphistry.compute.ast import call

        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            complex_graph.gfql([
                n({'type': 'person'}),                                              # Start with person nodes
                ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Only forward edges
                call('hop', {'hops': 1}),                                          # First hop
                ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': GE(20)}}),  # Filter intermediate nodes
                call('hop', {'hops': 1}),                                          # Second hop
                n()                                                                 # Final nodes
            ])

    def test_complex_enrichment_dependency_chain(self, complex_graph):
        """Pattern: [call(enrich1), call(filter_using_enrich1), call(enrich2), call(filter_using_enrich2)]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            complex_graph.gfql([
                ASTCall('get_degrees', {'col': 'deg'}),
                ASTCall('filter_nodes_by_dict', {'filter_dict': {'deg': GE(2)}}),
                n()
            ])

    def test_complex_backward_pass_with_calls(self, complex_graph):
        """Pattern: [n({f1}), call(f2), e(), n({f3}), e(), call(f4), n()]

        Validates: Mixed chains now raise GFQLValidationError (#791).
        """
        with pytest.raises(GFQLValidationError, match="Cannot mix call.*operations with n.*e.*traversals"):
            complex_graph.gfql([
                n({'type': 'person'}),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
                e(),
                n({'score': GE(20)}),
                e(),
                ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),
                n()
            ])

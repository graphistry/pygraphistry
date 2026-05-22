"""Tests for GFQL Call operations."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from graphistry.tests.test_compute import CGFull
from graphistry.Engine import Engine, EngineAbstract
from graphistry.compute.ast import ASTCall, ASTLet, n
from graphistry.compute.chain import Chain
from graphistry.compute.chain_let import chain_let_impl
from graphistry.compute.gfql.call.executor import execute_call
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError


class TestGroupInABoxExecution:
    @pytest.fixture
    def simple_graph(self):
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 3, 4, 5],
            'target': [1, 2, 0, 4, 5, 3]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3, 4, 5],
            'type': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        return CGFull()\
            .edges(edges_df)\
            .nodes(nodes_df)\
            .bind(source='source', destination='target', node='node')
    
    def test_group_in_a_box_basic(self, simple_graph):
        if not hasattr(simple_graph, 'group_in_a_box_layout'):
            pytest.skip("group_in_a_box_layout not available on test object")
        pytest.importorskip("igraph", reason="group_in_a_box_layout requires python-igraph")
            
        result = execute_call(
            simple_graph,
            'group_in_a_box_layout',
            {'engine': 'pandas'},
            Engine.PANDAS
        )
        
        assert len(result._nodes) == len(simple_graph._nodes)
        assert len(result._edges) == len(simple_graph._edges)
        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns
    
    def test_group_in_a_box_with_partition_key(self, simple_graph):
        if not hasattr(simple_graph, 'group_in_a_box_layout'):
            pytest.skip("group_in_a_box_layout not available on test object")
        pytest.importorskip("igraph", reason="group_in_a_box_layout requires python-igraph")
            
        result = execute_call(
            simple_graph,
            'group_in_a_box_layout',
            {
                'partition_key': 'type',
                'engine': 'pandas',
                'encode_colors': True
            },
            Engine.PANDAS
        )
        
        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns


class TestCallExecution:
    @pytest.fixture
    def sample_graph(self):
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2],
            'target': [1, 2, 0, 3]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user']
        })
        
        return CGFull()\
            .edges(edges_df)\
            .nodes(nodes_df)\
            .bind(source='source', destination='target', node='node')
    
    def test_execute_get_degrees(self, sample_graph):
        result = execute_call(
            sample_graph,
            'get_degrees',
            {'col': 'degree'},
            Engine.PANDAS
        )
        
        assert hasattr(result, '_nodes')
        assert 'degree' in result._nodes.columns
        assert len(result._nodes) == 4

    def test_execute_return_alias(self, sample_graph):
        result = execute_call(
            sample_graph,
            'return_',
            {'items': [('node', 'node')]},
            Engine.PANDAS
        )

        assert list(result._nodes.columns) == ['node']
        assert len(result._nodes) == len(sample_graph._nodes)
    
    def test_execute_filter_nodes(self, sample_graph):
        result = execute_call(
            sample_graph,
            'filter_nodes_by_dict',
            {'filter_dict': {'type': 'user'}},
            Engine.PANDAS
        )
        
        assert len(result._nodes) == 3
        assert all(result._nodes['type'] == 'user')
    
    def test_execute_materialize_nodes(self):
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        g = CGFull().edges(edges_df).bind(source='source', destination='target')
        
        assert g._nodes is None
        result = execute_call(
            g,
            'materialize_nodes',
            {},
            Engine.PANDAS
        )
        
        assert result._nodes is not None
        assert len(result._nodes) == 3

    def test_execute_circle_layout(self, sample_graph):
        result = execute_call(
            sample_graph,
            'circle_layout',
            {'bounding_box': [0, 0, 100, 100], 'engine': 'pandas'},
            Engine.PANDAS
        )

        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns

    def test_execute_tree_layout(self):
        edges_df = pd.DataFrame({
            'source': ['a', 'b'],
            'target': ['b', 'c'],
        })
        nodes_df = pd.DataFrame({
            'node': ['a', 'b', 'c'],
            'rank': [1, 2, 3],
        })
        g = CGFull().edges(edges_df).nodes(nodes_df).bind(source='source', destination='target', node='node')

        result = execute_call(
            g,
            'tree_layout',
            {'level_sort_values_by': 'rank'},
            Engine.PANDAS
        )

        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns
        assert 'level' in result._nodes.columns

    def test_execute_mercator_layout(self):
        nodes_df = pd.DataFrame({
            'node': ['nyc', 'sf'],
            'latitude': [40.7128, 37.7749],
            'longitude': [-74.0060, -122.4194],
        })
        g = CGFull().nodes(nodes_df).bind(node='node')

        result = execute_call(
            g,
            'mercator_layout',
            {},
            Engine.PANDAS
        )

        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns
        assert result._point_x == 'x'
        assert result._point_y == 'y'

    def test_execute_modularity_weighted_layout_with_existing_community(self, sample_graph):
        result = execute_call(
            sample_graph,
            'modularity_weighted_layout',
            {'community_col': 'type', 'engine': 'pandas'},
            Engine.PANDAS
        )

        assert 'weight' in result._edges.columns
        assert 'same_community' in result._edges.columns
        assert result._edge_weight == 'weight'

    def test_execute_encode_edge_icon(self, sample_graph):
        result = execute_call(
            sample_graph,
            'encode_edge_icon',
            {'column': 'edge_kind', 'categorical_mapping': {'email': 'envelope'}},
            Engine.PANDAS
        )

        assert (
            result._complex_encodings['edge_encodings']['default']['edgeIconEncoding']
            ['mapping']['categorical']['fixed']['email']
            == 'envelope'
        )

    def test_execute_encode_axis(self, sample_graph):
        rows = [{'r': 10, 'external': True, 'label': 'outer'}]

        result = execute_call(
            sample_graph,
            'encode_axis',
            {'rows': rows},
            Engine.PANDAS
        )

        assert (
            result._complex_encodings['node_encodings']['default']['pointAxisEncoding']
            ['rows']
            == rows
        )

    def test_execute_with_validation_error(self, sample_graph):
        with pytest.raises(GFQLTypeError) as exc_info:
            execute_call(
                sample_graph,
                'hop',
                {'hops': 'invalid'},  # Should be int
                Engine.PANDAS
            )
        assert exc_info.value.code == ErrorCode.E201
    
    def test_execute_unknown_method(self, sample_graph):
        with pytest.raises(GFQLTypeError) as exc_info:
            execute_call(
                sample_graph,
                'unknown_method',
                {},
                Engine.PANDAS
            )
        assert exc_info.value.code == ErrorCode.E303


class TestCallInDAG:
    @pytest.fixture
    def sample_graph(self):
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2, 3],
            'target': [1, 2, 0, 3, 0],
            'weight': [1.0, 2.0, 1.5, 3.0, 0.5]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user']
        })
        
        return CGFull()\
            .edges(edges_df)\
            .nodes(nodes_df)\
            .bind(source='source', destination='target', node='node')
    
    def test_call_in_dag(self, sample_graph):
        dag = ASTLet({
            'filtered': Chain([n({'type': 'user'})]),
            'with_degrees': ASTCall('get_degrees', {'col': 'degree'})
        })

        result = chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)

        assert 'degree' in result._nodes.columns
        assert len(result._nodes) == 3  # Only 'user' nodes
    
    def test_multiple_calls(self, sample_graph):
        dag1 = ASTLet({
            'with_degrees': ASTCall('get_degrees', {'col': 'deg'})
        })
        result1 = chain_let_impl(sample_graph, dag1, EngineAbstract.PANDAS)
        assert 'deg' in result1._nodes.columns
        
        dag2 = ASTLet({
            'filtered': ASTCall('filter_nodes_by_dict', {'filter_dict': {'deg': 2}})
        })
        result2 = chain_let_impl(result1, dag2, EngineAbstract.PANDAS)
        
        assert len(result2._nodes) > 0
        assert all(result2._nodes['deg'] == 2)
    
    @patch('graphistry.compute.gfql.call.executor.getattr')
    def test_call_execution_error(self, mock_getattr, sample_graph):
        mock_method = Mock(side_effect=RuntimeError("Method failed"))
        mock_getattr.return_value = mock_method
        
        dag = ASTLet({
            'failing': ASTCall('get_degrees', {})
        })
        
        with pytest.raises(RuntimeError) as exc_info:
            chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)
        assert "Failed to execute node 'failing'" in str(exc_info.value)


class TestGraphAlgorithmCalls:
    @pytest.fixture
    def sample_graph(self):
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2],
            'target': [1, 2, 0, 3]
        })
        
        return CGFull()\
            .edges(edges_df)\
            .bind(source='source', destination='target')

    def test_fa2_layout_cpu_requires_gpu(self, sample_graph):
        """fa2_layout should raise on CPU engines instead of silently falling back."""
        with pytest.raises(GFQLTypeError) as exc_info:
            execute_call(
                sample_graph,
                'fa2_layout',
                {},
                Engine.PANDAS
            )
        assert exc_info.value.code == ErrorCode.E303
        assert 'requires a GPU-enabled engine' in str(exc_info.value)

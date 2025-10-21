"""Tests for GFQL Call operations."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from graphistry.tests.test_compute import CGFull
from graphistry.Engine import Engine, EngineAbstract
from graphistry.compute.ast import ASTCall, ASTLet, ASTRef, n, e_forward
from graphistry.compute.chain import Chain
from graphistry.compute.chain_let import chain_let_impl
from graphistry.compute.gfql.call_safelist import validate_call_params
from graphistry.compute.gfql.call_executor import execute_call
from graphistry.compute.exceptions import (
    ErrorCode,
    GFQLTypeError,
    GFQLSyntaxError,
    GFQLSchemaError,
)


class TestCallSafelist:
    """Test method safelist validation."""
    
    def test_allowed_method(self):
        """Test validation of allowed methods."""
        params = validate_call_params('get_degrees', {
            'col': 'degree'
        })
        assert params == {'col': 'degree'}
    
    def test_unknown_method(self):
        """Test rejection of unknown methods."""
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('unknown_method', {})
        assert exc_info.value.code == ErrorCode.E303
        assert 'not in the safelist' in str(exc_info.value)
    
    def test_required_params(self):
        """Test validation of required parameters."""
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('filter_nodes_by_dict', {})
        assert exc_info.value.code == ErrorCode.E105
        assert 'Missing required parameters' in str(exc_info.value)
    
    def test_unknown_params(self):
        """Test rejection of unknown parameters."""
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('get_degrees', {
                'col': 'degree',
                'unknown_param': 'value'
            })
        assert exc_info.value.code == ErrorCode.E303
        assert 'Unknown parameters' in str(exc_info.value)
    
    def test_param_type_validation(self):
        """Test parameter type validation."""
        # Valid types
        params = validate_call_params('hop', {
            'hops': 2,
            'direction': 'forward',
            'to_fixed_point': True
        })
        assert params['hops'] == 2
        
        # Invalid type
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('hop', {
                'hops': 'two'  # Should be int
            })
        assert exc_info.value.code == ErrorCode.E201
        assert 'Invalid type' in str(exc_info.value)
    
    def test_enum_validation(self):
        """Test enum parameter validation."""
        # Valid enum value
        params = validate_call_params('hop', {
            'direction': 'forward'
        })
        assert params['direction'] == 'forward'
        
        # Invalid enum value
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('hop', {
                'direction': 'sideways'
            })
        assert exc_info.value.code == ErrorCode.E201


class TestASTCall:
    """Test ASTCall node validation and serialization."""
    
    def test_basic_creation(self):
        """Test creating a basic ASTCall."""
        call = ASTCall('get_degrees', {'col': 'degree'})
        assert call.function == 'get_degrees'
        assert call.params == {'col': 'degree'}
    
    def test_empty_params(self):
        """Test ASTCall with no parameters."""
        call = ASTCall('prune_self_edges')
        assert call.function == 'prune_self_edges'
        assert call.params == {}
    
    def test_validation(self):
        """Test ASTCall field validation."""
        # Valid call
        call = ASTCall('get_degrees', {'col': 'degree'})
        call.validate()  # Should not raise
        
        # Invalid function type
        with pytest.raises(GFQLTypeError) as exc_info:
            call = ASTCall(123, {})
            call.validate()
        assert exc_info.value.code == ErrorCode.E201
        assert 'function must be a string' in str(exc_info.value)
        
        # Invalid params type
        with pytest.raises(GFQLTypeError) as exc_info:
            call = ASTCall('get_degrees', 'not_a_dict')
            call.validate()
        assert exc_info.value.code == ErrorCode.E201
        assert 'params must be a dict' in str(exc_info.value)
    
    def test_to_json(self):
        """Test ASTCall JSON serialization."""
        call = ASTCall('get_degrees', {'col': 'degree', 'engine': 'pandas'})
        json_data = call.to_json()
        
        assert json_data == {
            'type': 'Call',
            'function': 'get_degrees',
            'params': {'col': 'degree', 'engine': 'pandas'}
        }
    
    def test_from_json(self):
        """Test ASTCall JSON deserialization."""
        json_data = {
            'type': 'Call',
            'function': 'filter_nodes_by_dict',
            'params': {'filter_dict': {'type': 'user'}}
        }
        
        call = ASTCall.from_json(json_data)
        assert isinstance(call, ASTCall)
        assert call.function == 'filter_nodes_by_dict'
        assert call.params == {'filter_dict': {'type': 'user'}}
    
    def test_from_json_invalid(self):
        """Test ASTCall from_json with invalid data."""
        # Missing function
        with pytest.raises(AssertionError) as exc_info:
            ASTCall.from_json({'type': 'Call'})
        assert 'Call missing function' in str(exc_info.value)
        
        # Wrong type - this would be caught earlier in the AST dispatch
        # so we don't test it here


class TestGroupInABoxExecution:
    """Test actual execution of group_in_a_box_layout."""
    
    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph for testing."""
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
        """Test basic group_in_a_box_layout execution."""
        # Skip if method not available on test object
        if not hasattr(simple_graph, 'group_in_a_box_layout'):
            pytest.skip("group_in_a_box_layout not available on test object")
        pytest.importorskip("igraph", reason="group_in_a_box_layout requires python-igraph")
            
        result = execute_call(
            simple_graph,
            'group_in_a_box_layout',
            {'engine': 'pandas'},
            Engine.PANDAS
        )
        
        # Should still have same number of nodes/edges
        assert len(result._nodes) == len(simple_graph._nodes)
        assert len(result._edges) == len(simple_graph._edges)
        
        # Should have position columns
        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns
    
    def test_group_in_a_box_with_partition_key(self, simple_graph):
        """Test group_in_a_box_layout with existing partition key."""
        # Skip if method not available on test object
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
        
        # Should have positions
        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns


class TestCallExecution:
    """Test call execution functionality."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
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
        """Test executing get_degrees method."""
        result = execute_call(
            sample_graph,
            'get_degrees',
            {'col': 'degree'},
            Engine.PANDAS
        )
        
        # Result should be a Plottable
        assert hasattr(result, '_nodes')
        assert 'degree' in result._nodes.columns
        assert len(result._nodes) == 4
    
    def test_execute_filter_nodes(self, sample_graph):
        """Test executing filter_nodes_by_dict method."""
        result = execute_call(
            sample_graph,
            'filter_nodes_by_dict',
            {'filter_dict': {'type': 'user'}},
            Engine.PANDAS
        )
        
        # Should filter to only user nodes
        assert len(result._nodes) == 3
        assert all(result._nodes['type'] == 'user')
    
    def test_execute_materialize_nodes(self):
        """Test executing materialize_nodes method."""
        # Start with edges only
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        g = CGFull().edges(edges_df).bind(source='source', destination='target')
        
        # No nodes initially
        assert g._nodes is None
        
        # Execute materialize_nodes
        result = execute_call(
            g,
            'materialize_nodes',
            {},
            Engine.PANDAS
        )
        
        # Should have nodes now
        assert result._nodes is not None
        assert len(result._nodes) == 3

    def test_execute_ring_continuous_layout(self):
        """Test executing ring_continuous_layout in continuous mode."""
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2],
            'type': ['user', 'user', 'admin'],
            'score': [0.1, 0.4, 0.8]
        })

        g = CGFull()\
            .edges(edges_df)\
            .nodes(nodes_df)\
            .bind(source='source', destination='target', node='node')

        result = execute_call(
            g,
            'ring_continuous_layout',
            {'ring_col': 'score'},
            Engine.PANDAS
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)
        assert result._point_x == 'x'
        assert result._point_y == 'y'

    def test_execute_time_ring_layout(self):
        """Test executing time_ring_layout in time mode with ISO strings."""
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2],
            'ts': pd.to_datetime([
                '2024-01-01T00:00:00',
                '2024-01-01T00:01:00',
                '2024-01-01T00:02:00'
            ])
        })

        g = CGFull()\
            .edges(edges_df)\
            .nodes(nodes_df)\
            .bind(source='source', destination='target', node='node')

        result = execute_call(
            g,
            'time_ring_layout',
            {
                'time_col': 'ts',
                'time_start': '2024-01-01T00:00:00',
                'time_end': '2024-01-01T00:02:00',
                'num_rings': 3
            },
            Engine.PANDAS
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)
        assert result._point_x == 'x'
        assert result._point_y == 'y'
    
    def test_execute_with_validation_error(self, sample_graph):
        """Test that validation errors are properly raised."""
        with pytest.raises(GFQLTypeError) as exc_info:
            execute_call(
                sample_graph,
                'hop',
                {'hops': 'invalid'},  # Should be int
                Engine.PANDAS
            )
        assert exc_info.value.code == ErrorCode.E201
    
    def test_execute_unknown_method(self, sample_graph):
        """Test execution of unknown method."""
        with pytest.raises(GFQLTypeError) as exc_info:
            execute_call(
                sample_graph,
                'unknown_method',
                {},
                Engine.PANDAS
            )
        assert exc_info.value.code == ErrorCode.E303


class TestCallInDAG:
    """Test ASTCall execution within DAGs."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
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
        """Test executing ASTCall within a DAG."""
        dag = ASTLet({
            'filtered': Chain([n({'type': 'user'})]),
            'with_degrees': ASTCall('get_degrees', {'col': 'degree'})
        })

        result = chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)

        # Should have degree column
        assert 'degree' in result._nodes.columns
        # When Chain filters, subsequent operations see filtered data
        assert len(result._nodes) == 3  # Only 'user' nodes
    
    def test_call_referencing_binding(self, sample_graph):
        """Test ASTCall that operates on whole graph (not in chain)."""
        from graphistry.compute.ast import ASTRef

        # Call operations work on the accumulated graph state in DAG
        dag = ASTLet({
            'users': Chain([n({'type': 'user'})]),
            'with_degrees': ASTCall('get_degrees', {'col': 'degree'})
        })

        result = chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)

        # DAG returns last binding result, which has filtered nodes from Chain
        assert len(result._nodes) == 3  # Only 'user' nodes
        assert 'degree' in result._nodes.columns
    
    def test_multiple_calls(self, sample_graph):
        """Test multiple call operations in sequence."""
        # First add degrees
        dag1 = ASTLet({
            'with_degrees': ASTCall('get_degrees', {'col': 'deg'})
        })
        result1 = chain_let_impl(sample_graph, dag1, EngineAbstract.PANDAS)
        assert 'deg' in result1._nodes.columns
        
        # Then filter - use the graph that has degrees
        dag2 = ASTLet({
            'filtered': ASTCall('filter_nodes_by_dict', {'filter_dict': {'deg': 2}})
        })
        result2 = chain_let_impl(result1, dag2, EngineAbstract.PANDAS)
        
        # Should have nodes with degree 2
        assert len(result2._nodes) > 0
        assert all(result2._nodes['deg'] == 2)
    
    @patch('graphistry.compute.gfql.call_executor.getattr')
    def test_call_execution_error(self, mock_getattr, sample_graph):
        """Test handling of execution errors in calls."""
        # Make the method raise an error
        mock_method = Mock(side_effect=RuntimeError("Method failed"))
        mock_getattr.return_value = mock_method
        
        dag = ASTLet({
            'failing': ASTCall('get_degrees', {})
        })
        
        with pytest.raises(RuntimeError) as exc_info:
            chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)
        assert "Failed to execute node 'failing'" in str(exc_info.value)


class TestGraphAlgorithmCalls:
    """Test calls to graph algorithm methods."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2],
            'target': [1, 2, 0, 3]
        })
        
        return CGFull()\
            .edges(edges_df)\
            .bind(source='source', destination='target')
    
    def test_compute_cugraph_params(self):
        """Test compute_cugraph parameter validation."""
        # Valid params
        params = validate_call_params('compute_cugraph', {
            'alg': 'pagerank',
            'out_col': 'pr_score',
            'directed': True
        })
        assert params['alg'] == 'pagerank'
        
        # Missing required alg
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('compute_cugraph', {
                'out_col': 'pr_score'
            })
        assert exc_info.value.code == ErrorCode.E105
    
    def test_compute_igraph_params(self):
        """Test compute_igraph parameter validation."""
        params = validate_call_params('compute_igraph', {
            'alg': 'community_louvain',
            'directed': False
        })
        assert params['alg'] == 'community_louvain'
    
    def test_layout_methods(self):
        """Test layout method parameter validation."""
        # layout_cugraph
        params = validate_call_params('layout_cugraph', {
            'layout': 'force_atlas2',
            'params': {'iterations': 100}
        })
        assert params['layout'] == 'force_atlas2'
        
        # layout_igraph
        params = validate_call_params('layout_igraph', {
            'layout': 'fruchterman_reingold',
            'directed': True
        })
        assert params['layout'] == 'fruchterman_reingold'
        
        # fa2_layout
        params = validate_call_params('fa2_layout', {
            'fa2_params': {'iterations': 500}
        })
        assert params['fa2_params']['iterations'] == 500
    
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

    def test_ring_continuous_layout_params(self):
        """Test ring_continuous_layout parameter validation."""
        params = validate_call_params('ring_continuous_layout', {
            'ring_col': 'score',
            'min_r': 200.0
        })
        assert params['min_r'] == 200.0

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('ring_continuous_layout', {
                'ring_col': ['not', 'valid']
            })
        assert exc_info.value.code == ErrorCode.E201

    def test_ring_categorical_layout_params(self):
        """Test ring_categorical_layout parameter validation."""
        params = validate_call_params('ring_categorical_layout', {
            'ring_col': 'segment',
            'order': ['a', 'b']
        })
        assert params['ring_col'] == 'segment'

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('ring_categorical_layout', {})
        assert exc_info.value.code == ErrorCode.E105

    def test_time_ring_layout_params(self):
        """Test time_ring_layout parameter validation."""
        params = validate_call_params('time_ring_layout', {
            'time_col': 'ts',
            'time_start': '2024-01-01T00:00:00'
        })
        assert params['time_col'] == 'ts'

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('time_ring_layout', {
                'time_col': 'ts',
                'time_unit': 'invalid'
            })
        assert exc_info.value.code == ErrorCode.E201
    
    def test_group_in_a_box_layout_params(self):
        """Test group_in_a_box_layout parameter validation."""
        # Valid params with all types
        params = validate_call_params('group_in_a_box_layout', {
            'partition_alg': 'louvain',
            'partition_params': {'resolution': 1.0},
            'layout_alg': 'force_atlas2',
            'layout_params': {'iterations': 100},
            'x': 0,
            'y': 0,
            'w': 1000,
            'h': 1000,
            'encode_colors': True,
            'colors': ['#ff0000', '#00ff00'],
            'partition_key': 'community',
            'engine': 'pandas'
        })
        assert params['partition_alg'] == 'louvain'
        assert params['engine'] == 'pandas'
        
        # Minimal params (all optional)
        params = validate_call_params('group_in_a_box_layout', {})
        assert params == {}
        
        # Test type validations
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('group_in_a_box_layout', {
                'x': 'not_a_number'  # Should be numeric
            })
        assert exc_info.value.code == ErrorCode.E201
        
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('group_in_a_box_layout', {
                'engine': 'invalid_engine'  # Should be in allowed list
            })
        assert exc_info.value.code == ErrorCode.E201


@pytest.fixture
def names_graph():
    """Graph with rich schema for testing matcher naming & mark()."""
    nodes_df = pd.DataFrame({
        'id': [0, 1, 2, 3],
        'region': ['NA', 'EU', 'NA', 'APAC'],
        'tier': ['gold', 'silver', 'silver', 'bronze']
    })
    edges_df = pd.DataFrame({
        'source': [0, 0, 1, 2],
        'target': [1, 2, 3, 3],
        'amount': [10, 7, 5, 3]
    })
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'source', 'target')
    return g


class TestNameConflicts:
    """TDD coverage for matcher name conflict policy."""

    def test_duplicate_node_names_default_any(self, names_graph):
        chain_ops = [
            n({'region': 'NA'}, name='dup'),
            e_forward(),
            n({'region': 'EU'}, name='dup'),
        ]

        result = names_graph.gfql(chain_ops)

        nodes = result._nodes.set_index(names_graph._node)
        assert 'dup' in nodes.columns
        assert nodes['dup'].dtype == bool
        assert set(nodes.index[nodes['dup']]) == {0, 1}

    def test_duplicate_edge_names_default_any(self, names_graph):
        chain_ops = [
            n({'region': 'NA'}),
            e_forward({'amount': {'$gte': 7}}, name='dup', to_fixed_point=True),
            n({'region': 'EU'}),
            e_forward({'amount': {'$lte': 5}}, name='dup', to_fixed_point=True),
            n({'region': 'APAC'}),
        ]

        result = names_graph.gfql(chain_ops)

        edges = result._edges[['source', 'target', 'dup']]
        assert edges['dup'].dtype == bool

    def test_name_conflicts_error_on_duplicates(self, names_graph):
        chain_ops = [
            n({'region': 'NA'}, name='dup'),
            e_forward(),
            n({'region': 'EU'}, name='dup'),
        ]

        with pytest.raises(GFQLSchemaError):
            names_graph.gfql(chain_ops, name_conflicts='error')

    def test_name_conflicts_existing_column_error(self, names_graph):
        graph_with_dup_col = names_graph.nodes(
            names_graph._nodes.assign(dup=False),
            names_graph._node
        )

        chain_ops = [
            n({'region': 'NA'}, name='dup'),
            e_forward(),
            n({'region': 'EU'}, name='dup'),
        ]

        with pytest.raises(GFQLSchemaError):
            graph_with_dup_col.gfql(chain_ops, name_conflicts='error')

    def test_name_conflicts_existing_column_any_overwrites(self, names_graph):
        graph_with_dup_col = names_graph.nodes(
            names_graph._nodes.assign(dup=False),
            names_graph._node
        )

        chain_ops = [
            n({'region': 'NA'}, name='dup'),
            e_forward(),
            n({'region': 'EU'}, name='dup'),
        ]

        result = graph_with_dup_col.gfql(chain_ops, name_conflicts='any')
        nodes = result._nodes.set_index(names_graph._node)
        assert nodes['dup'].dtype == bool
        assert set(nodes.index[nodes['dup']]) == {0, 1}

    def test_name_conflicts_node_and_edge_same_name(self, names_graph):
        chain_ops = [
            n({'region': 'NA'}, name='shared'),
            e_forward({'amount': {'$gte': 7}}, name='shared'),
            n({'region': 'EU'})
        ]

        result = names_graph.gfql(chain_ops)

        nodes = result._nodes.set_index(names_graph._node)
        edges = result._edges[['source', 'target', 'shared']]
        assert 'shared' in nodes.columns
        assert 'shared' in edges.columns
        assert nodes['shared'].dtype == bool
        assert edges['shared'].dtype == bool

    def test_name_conflicts_propagates_through_let(self, names_graph):
        chain_ops = Chain([
            n({'region': 'NA'}, name='dup'),
            e_forward(),
            n({'region': 'EU'}, name='dup'),
        ])

        dag = ASTLet({
            'base': chain_ops,
            'with_degrees': ASTCall('get_degrees', {'col': 'deg'})
        })

        result = names_graph.gfql(dag)
        assert 'dup' in result._nodes.columns

        with pytest.raises(GFQLSchemaError):
            names_graph.gfql(dag, name_conflicts='error')

    def test_name_conflicts_handles_astref_chains(self, names_graph):
        dag = ASTLet({
            'seed': Chain([n({'region': 'NA'})]),
            'expanded': ASTRef('seed', Chain([
                e_forward(name='dup'),
                n({'region': 'EU'}, name='dup')
            ]).chain)
        })

        result = names_graph.gfql(dag)
        assert 'dup' in result._nodes.columns



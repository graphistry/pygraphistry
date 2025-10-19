"""Tests for GFQL mark() operation."""

import pytest
import pandas as pd

from graphistry.tests.test_compute import CGFull
from graphistry.Engine import Engine, EngineAbstract
from graphistry.compute.ast import ASTCall, ASTLet, n, e_forward, ref, call
from graphistry.compute.chain import Chain
from graphistry.compute.chain_let import chain_let_impl
from graphistry.compute.gfql.call_safelist import validate_call_params
from graphistry.compute.gfql.call_executor import execute_call
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLSyntaxError, GFQLSchemaError


class TestMarkSafelist:
    """Test mark() in safelist validation."""

    def test_mark_in_safelist(self):
        """Test mark is registered in safelist."""
        params = validate_call_params('mark', {
            'gfql': [n()],
            'name': 'is_matched'
        })
        # Just check keys, not deep equality of AST objects
        assert 'gfql' in params
        assert params['name'] == 'is_matched'

    def test_mark_required_params(self):
        """Test mark requires gfql parameter (name is optional)."""
        # Missing gfql
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('mark', {'name': 'is_matched'})
        assert exc_info.value.code == ErrorCode.E105
        assert 'Missing required parameters' in str(exc_info.value)

        # Missing name is OK (will use default)
        params = validate_call_params('mark', {'gfql': [n()]})
        assert 'gfql' in params

    def test_mark_param_types(self):
        """Test mark parameter type validation."""
        # Valid list
        params = validate_call_params('mark', {
            'gfql': [n()],
            'name': 'is_matched'
        })
        assert params['name'] == 'is_matched'

        # Valid dict (JSON representation)
        params = validate_call_params('mark', {
            'gfql': {'type': 'Chain', 'chain': []},
            'name': 'is_matched'
        })
        assert 'gfql' in params

        # Invalid name type
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('mark', {
                'gfql': [n()],
                'name': 123  # Should be string
            })
        assert exc_info.value.code == ErrorCode.E201


class TestMarkBasic:
    """Test basic mark() functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user']
        })
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2],
            'target': [1, 2, 0, 3],
            'rel': ['friend', 'friend', 'manages', 'manages']
        })

        return CGFull()\
            .nodes(nodes_df)\
            .edges(edges_df)\
            .bind(node='node', source='source', destination='target')

    def test_mark_nodes_basic(self, sample_graph):
        """Test marking nodes with simple pattern."""
        result = sample_graph.mark(
            gfql=[n({'type': 'user'})],
            name='is_user'
        )

        # All nodes should be preserved
        assert len(result._nodes) == 4

        # Should have is_user column
        assert 'is_user' in result._nodes.columns

        # Should have True for user nodes, False for others
        user_marks = result._nodes.set_index('node')['is_user']
        assert user_marks[0] == True  # noqa: E712
        assert user_marks[1] == True  # noqa: E712
        assert user_marks[2] == False  # noqa: E712
        assert user_marks[3] == True  # noqa: E712

    def test_mark_edges_basic(self, sample_graph):
        """Test marking edges with simple pattern."""
        result = sample_graph.mark(
            gfql=[e_forward({'rel': 'friend'})],
            name='is_friend'
        )

        # All edges should be preserved
        assert len(result._edges) == 4

        # Should have is_friend column
        assert 'is_friend' in result._edges.columns

        # Should have True for friend edges, False for others
        friend_marks = result._edges['is_friend']
        assert friend_marks.iloc[0] == True  # noqa: E712
        assert friend_marks.iloc[1] == True  # noqa: E712
        assert friend_marks.iloc[2] == False  # noqa: E712
        assert friend_marks.iloc[3] == False  # noqa: E712

    def test_mark_all_match(self, sample_graph):
        """Test marking when all entities match."""
        result = sample_graph.mark(
            gfql=[n()],  # All nodes
            name='is_node'
        )

        # All nodes marked True
        assert all(result._nodes['is_node'])

    def test_mark_none_match(self, sample_graph):
        """Test marking when no entities match."""
        result = sample_graph.mark(
            gfql=[n({'type': 'nonexistent'})],
            name='is_nonexistent'
        )

        # All nodes marked False
        assert not any(result._nodes['is_nonexistent'])

    def test_mark_default_name_nodes(self, sample_graph):
        """Test marking nodes with default name."""
        result = sample_graph.mark(
            gfql=[n({'type': 'user'})]
            # No name parameter - should use default
        )

        # Should have default column name
        assert 'is_matched_node' in result._nodes.columns

        # Should have correct marks
        user_marks = result._nodes.set_index('node')['is_matched_node']
        assert user_marks[0] == True  # noqa: E712
        assert user_marks[1] == True  # noqa: E712
        assert user_marks[2] == False  # noqa: E712

    def test_mark_default_name_edges(self, sample_graph):
        """Test marking edges with default name."""
        result = sample_graph.mark(
            gfql=[e_forward({'rel': 'friend'})]
            # No name parameter - should use default
        )

        # Should have default column name
        assert 'is_matched_edge' in result._edges.columns

        # Should have correct marks
        friend_marks = result._edges['is_matched_edge']
        assert friend_marks.iloc[0] == True  # noqa: E712
        assert friend_marks.iloc[1] == True  # noqa: E712


class TestMarkValidation:
    """Test mark() parameter validation."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2],
            'type': ['user', 'user', 'admin']
        })
        edges_df = pd.DataFrame({
            'source': [0, 1],
            'target': [1, 2]
        })

        return CGFull()\
            .nodes(nodes_df)\
            .edges(edges_df)\
            .bind(node='node', source='source', destination='target')

    def test_mark_invalid_gfql_type(self, sample_graph):
        """Test mark with invalid gfql type."""
        with pytest.raises(GFQLTypeError) as exc_info:
            sample_graph.mark(
                gfql="not a list or chain",  # Invalid
                name='is_matched'
            )
        assert exc_info.value.code == ErrorCode.E201
        assert 'Chain, List[ASTObject], or JSON dict' in str(exc_info.value)

    def test_mark_empty_gfql(self, sample_graph):
        """Test mark with empty gfql."""
        with pytest.raises(GFQLSyntaxError) as exc_info:
            sample_graph.mark(
                gfql=[],  # Empty
                name='is_matched'
            )
        assert exc_info.value.code == ErrorCode.E105
        assert 'cannot be empty' in str(exc_info.value)

    def test_mark_gfql_not_ending_with_matcher(self, sample_graph):
        """Test mark with gfql not ending with node/edge matcher."""
        with pytest.raises(GFQLSyntaxError) as exc_info:
            sample_graph.mark(
                gfql=[n(), call('get_degrees')],  # Ends with call
                name='is_matched'
            )
        assert exc_info.value.code == ErrorCode.E104
        assert 'ending with node or edge matcher' in str(exc_info.value)

    def test_mark_invalid_name_type(self, sample_graph):
        """Test mark with invalid name type."""
        with pytest.raises(GFQLTypeError) as exc_info:
            sample_graph.mark(
                gfql=[n()],
                name=123  # Should be string
            )
        assert exc_info.value.code == ErrorCode.E201

    def test_mark_empty_name(self, sample_graph):
        """Test mark with empty name."""
        with pytest.raises(GFQLTypeError) as exc_info:
            sample_graph.mark(
                gfql=[n()],
                name=''  # Empty
            )
        assert exc_info.value.code == ErrorCode.E106
        assert 'cannot be empty' in str(exc_info.value)

    def test_mark_internal_column_name(self, sample_graph):
        """Test mark with internal column name."""
        with pytest.raises(GFQLSchemaError) as exc_info:
            sample_graph.mark(
                gfql=[n()],
                name='__gfql_temp__'  # Internal prefix
            )
        assert '__gfql_' in str(exc_info.value)
        assert 'reserved for internal use' in str(exc_info.value)

    def test_mark_column_name_collision(self, sample_graph):
        """Test mark with existing column name."""
        with pytest.raises(GFQLSchemaError) as exc_info:
            sample_graph.mark(
                gfql=[n()],
                name='type'  # Already exists
            )
        assert exc_info.value.code == ErrorCode.E301
        assert 'already exists' in str(exc_info.value)


class TestMarkAccumulation:
    """Test accumulating multiple marks."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user'],
            'region': ['US', 'EU', 'US', 'EU']
        })
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 3]
        })

        return CGFull()\
            .nodes(nodes_df)\
            .edges(edges_df)\
            .bind(node='node', source='source', destination='target')

    def test_multiple_marks_accumulate(self, sample_graph):
        """Test that multiple marks accumulate."""
        g2 = sample_graph.mark(
            gfql=[n({'type': 'user'})],
            name='is_user'
        )

        g3 = g2.mark(
            gfql=[n({'region': 'US'})],
            name='is_us'
        )

        # Both columns should exist
        assert 'is_user' in g3._nodes.columns
        assert 'is_us' in g3._nodes.columns

        # Check specific node values
        nodes = g3._nodes.set_index('node')
        assert nodes.loc[0, 'is_user'] == True  # noqa: E712
        assert nodes.loc[0, 'is_us'] == True  # noqa: E712
        assert nodes.loc[1, 'is_user'] == True  # noqa: E712
        assert nodes.loc[1, 'is_us'] == False  # noqa: E712
        assert nodes.loc[2, 'is_user'] == False  # noqa: E712
        assert nodes.loc[2, 'is_us'] == True  # noqa: E712

    def test_mark_then_filter(self, sample_graph):
        """Test using mark column for subsequent filtering."""
        g2 = sample_graph.mark(
            gfql=[n({'type': 'user'})],
            name='is_user'
        )

        # Filter using mark column
        g3 = g2.gfql([n({'is_user': True})])

        # Should only have marked nodes
        assert len(g3._nodes) == 3
        assert all(g3._nodes['is_user'])


class TestMarkInCall:
    """Test mark() as call() operation."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2],
            'type': ['user', 'admin', 'user']
        })
        edges_df = pd.DataFrame({
            'source': [0, 1],
            'target': [1, 2]
        })

        return CGFull()\
            .nodes(nodes_df)\
            .edges(edges_df)\
            .bind(node='node', source='source', destination='target')

    def test_call_mark_basic(self, sample_graph):
        """Test mark via call() operation."""
        result = execute_call(
            sample_graph,
            'mark',
            {
                'gfql': [n({'type': 'user'})],
                'name': 'is_user'
            },
            Engine.PANDAS
        )

        # Should have mark column
        assert 'is_user' in result._nodes.columns
        assert len(result._nodes) == 3  # All nodes preserved

        # Verify exact boolean values for specific nodes
        nodes = result._nodes.set_index('node')
        assert nodes.loc[0, 'is_user'] == True  # noqa: E712
        assert nodes.loc[1, 'is_user'] == False  # noqa: E712
        assert nodes.loc[2, 'is_user'] == True  # noqa: E712

        # Verify other columns preserved
        assert 'type' in result._nodes.columns

        # Verify edges unchanged
        assert len(result._edges) == len(sample_graph._edges)
        pd.testing.assert_frame_equal(result._edges, sample_graph._edges)

    def test_call_mark_with_json_gfql(self, sample_graph):
        """Test mark with JSON-serialized GFQL."""
        # Simulate remote execution with JSON
        gfql_json = Chain([n({'type': 'user'})]).to_json()

        result = execute_call(
            sample_graph,
            'mark',
            {
                'gfql': gfql_json,
                'name': 'is_user'
            },
            Engine.PANDAS
        )

        # Should work same as list form
        assert 'is_user' in result._nodes.columns
        user_count = result._nodes['is_user'].sum()
        assert user_count == 2

        # Verify exact values match list form
        list_result = execute_call(
            sample_graph,
            'mark',
            {
                'gfql': [n({'type': 'user'})],
                'name': 'is_user_list'
            },
            Engine.PANDAS
        )

        # Boolean columns should match
        nodes_json = result._nodes.set_index('node')['is_user']
        nodes_list = list_result._nodes.set_index('node')['is_user_list']
        pd.testing.assert_series_equal(nodes_json, nodes_list, check_names=False)


class TestMarkInLet:
    """Test mark() in let() DAG compositions."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user'],
            'vip': [True, False, True, False]
        })
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 3]
        })

        return CGFull()\
            .nodes(nodes_df)\
            .edges(edges_df)\
            .bind(node='node', source='source', destination='target')

    def test_mark_in_let_binding(self, sample_graph):
        """Test mark() within let() binding."""
        dag = ASTLet({
            'marked': ASTCall('mark', {
                'gfql': [n({'type': 'user'})],
                'name': 'is_user'
            })
        })

        result = chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)

        # Should have mark column
        assert 'is_user' in result._nodes.columns
        assert result._nodes['is_user'].sum() == 3

    def test_mark_accumulation_in_let(self, sample_graph):
        """Test multiple marks accumulating in let()."""
        dag = ASTLet({
            'marked_users': ASTCall('mark', {
                'gfql': [n({'type': 'user'})],
                'name': 'is_user'
            }),
            'marked_vips': ASTCall('mark', {
                'gfql': [n({'vip': True})],
                'name': 'is_vip'
            })
        })

        result = chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)

        # Both marks should exist (last binding wins in let)
        # DAG returns last binding result
        assert 'is_vip' in result._nodes.columns

    def test_mark_with_ref_in_let(self, sample_graph):
        """Test mark referencing previous binding."""
        from graphistry.compute.ast import ASTRef

        dag = ASTLet({
            'marked_users': ASTCall('mark', {
                'gfql': [n({'type': 'user'})],
                'name': 'is_user'
            }),
            'marked_vips': ref('marked_users', [
                ASTCall('mark', {
                    'gfql': [n({'vip': True})],
                    'name': 'is_vip'
                })
            ])
        })

        result = chain_let_impl(sample_graph, dag, EngineAbstract.PANDAS)

        # Second binding should have both marks
        assert 'is_user' in result._nodes.columns
        assert 'is_vip' in result._nodes.columns


class TestMarkChain:
    """Test mark() in chain contexts."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user']
        })
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 3]
        })

        return CGFull()\
            .nodes(nodes_df)\
            .edges(edges_df)\
            .bind(node='node', source='source', destination='target')

    def test_mark_with_chain(self, sample_graph):
        """Test mark with Chain object."""
        chain = Chain([n({'type': 'user'})])

        result = sample_graph.mark(
            gfql=chain,
            name='is_user'
        )

        assert 'is_user' in result._nodes.columns
        assert result._nodes['is_user'].sum() == 3


class TestMarkEdgeCases:
    """Test edge cases for mark()."""

    def test_mark_on_graph_with_no_nodes(self):
        """Test marking when graph has no nodes."""
        edges_df = pd.DataFrame({
            'source': [0, 1],
            'target': [1, 2]
        })
        g = CGFull().edges(edges_df).bind(source='source', destination='target')

        # Should materialize nodes first
        result = g.mark(
            gfql=[n()],
            name='is_node'
        )

        # Should have materialized nodes with mark
        assert result._nodes is not None
        assert 'is_node' in result._nodes.columns

    def test_mark_edges_on_graph_with_no_edges(self):
        """Test marking edges when graph has no edges."""
        nodes_df = pd.DataFrame({'node': [0, 1, 2]})
        g = CGFull().nodes(nodes_df).bind(node='node')

        # Should raise error
        with pytest.raises(ValueError) as exc_info:
            g.mark(
                gfql=[e_forward()],
                name='is_edge'
            )
        assert 'no edges' in str(exc_info.value).lower()

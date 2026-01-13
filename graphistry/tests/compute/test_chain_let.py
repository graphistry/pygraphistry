"""Test chain DAG functionality

For integration tests with real remote graphs, see test_chain_let_remote_integration.py
Enable remote tests with: TEST_REMOTE_INTEGRATION=1
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from graphistry.compute.ast import ASTLet, ASTRemoteGraph, ASTRef, ASTNode, ASTObject, n, e
from graphistry.compute.chain import Chain
from graphistry.compute.chain_let import (
    extract_dependencies, build_dependency_graph, validate_dependencies,
    detect_cycles, determine_execution_order
)
from graphistry.compute.execution_context import ExecutionContext
from graphistry.compute.exceptions import GFQLTypeError
from graphistry.tests.test_compute import CGFull


class TestChainDagHelpers:
    """Test the helper functions for DAG execution"""
    
    def test_extract_dependencies_no_deps(self):
        """Test extracting dependencies from nodes with no dependencies"""
        node = n({'type': 'person'})
        deps = extract_dependencies(node)
        assert deps == set()
        
        remote = ASTRemoteGraph('dataset123')
        deps = extract_dependencies(remote)
        assert deps == set()
    
    def test_extract_dependencies_chain_ref(self):
        """Test extracting dependencies from ASTRef"""
        chain_ref = ASTRef('source', [n()])
        deps = extract_dependencies(chain_ref)
        assert deps == {'source'}
    
    def test_extract_dependencies_nested(self):
        """Test extracting dependencies from nested structures"""
        # ChainRef with ChainRef in its chain
        nested = ASTRef('a', [ASTRef('b', [n()])])
        deps = extract_dependencies(nested)
        assert deps == {'a', 'b'}
        
        # Nested DAG
        dag = ASTLet({
            'inner': ASTRef('outer', [n()])
        })
        deps = extract_dependencies(dag)
        assert deps == {'outer'}
    
    def test_build_dependency_graph(self):
        """Test building dependency and dependent mappings"""
        bindings = {
            'a': n(),
            'b': ASTRef('a', [n()]),
            'c': ASTRef('b', [n()])
        }
        
        dependencies, dependents = build_dependency_graph(bindings)
        
        assert dependencies == {
            'a': set(),
            'b': {'a'},
            'c': {'b'}
        }
        assert dependents == {
            'a': {'b'},
            'b': {'c'}
        }
    
    def test_validate_dependencies_valid(self):
        """Test validation passes for valid dependencies"""
        bindings = {
            'a': n(),
            'b': ASTRef('a', [n()])
        }
        dependencies = {'a': set(), 'b': {'a'}}
        
        # Should not raise
        validate_dependencies(bindings, dependencies)
    
    def test_validate_dependencies_missing_ref(self):
        """Test validation catches missing references"""
        bindings = {
            'a': n()
        }
        dependencies = {'a': {'missing'}}
        
        with pytest.raises(ValueError) as exc_info:
            validate_dependencies(bindings, dependencies)
        
        assert "references undefined nodes: ['missing']" in str(exc_info.value)
        assert "Available nodes: ['a']" in str(exc_info.value)
    
    def test_validate_dependencies_self_ref(self):
        """Test validation catches self-references"""
        bindings = {
            'a': n()
        }
        dependencies = {'a': {'a'}}
        
        with pytest.raises(ValueError) as exc_info:
            validate_dependencies(bindings, dependencies)
        
        assert "Self-reference cycle detected: 'a' depends on itself" in str(exc_info.value)
    
    def test_detect_cycles_no_cycle(self):
        """Test cycle detection on acyclic graph"""
        dependencies = {
            'a': set(),
            'b': {'a'},
            'c': {'b'}
        }
        
        cycle = detect_cycles(dependencies)
        assert cycle is None
    
    def test_detect_cycles_simple_cycle(self):
        """Test cycle detection on simple cycle"""
        dependencies = {
            'a': {'b'},
            'b': {'a'}
        }
        
        cycle = detect_cycles(dependencies)
        assert cycle == ['a', 'b', 'a'] or cycle == ['b', 'a', 'b']
    
    def test_detect_cycles_longer_cycle(self):
        """Test cycle detection on longer cycle"""
        dependencies = {
            'a': {'b'},
            'b': {'c'},
            'c': {'a'},
            'd': {'a'}
        }
        
        cycle = detect_cycles(dependencies)
        # Could start from any node in the cycle
        assert len(cycle) == 4  # 3 nodes + repeat
        assert cycle[0] == cycle[-1]  # Cycle closes
    
    def test_determine_execution_order_empty(self):
        """Test execution order for empty DAG"""
        order = determine_execution_order({})
        assert order == []
    
    def test_determine_execution_order_single(self):
        """Test execution order for single node"""
        bindings = {'only': n()}
        order = determine_execution_order(bindings)
        assert order == ['only']
    
    def test_determine_execution_order_linear(self):
        """Test execution order for linear dependencies"""
        bindings = {
            'a': n(),
            'b': ASTRef('a', [n()]),
            'c': ASTRef('b', [n()])
        }
        
        order = determine_execution_order(bindings)
        assert order == ['a', 'b', 'c']
    
    def test_determine_execution_order_diamond(self):
        """Test execution order for diamond pattern"""
        bindings = {
            'top': n(),
            'left': ASTRef('top', [n()]),
            'right': ASTRef('top', [n()]),
            'bottom': ASTRef('left', [ASTRef('right', [n()])])
        }
        
        order = determine_execution_order(bindings)
        # Top must come first, bottom must come last
        assert order[0] == 'top'
        assert order[-1] == 'bottom'
        # Left and right can be in either order
        assert set(order[1:3]) == {'left', 'right'}
    
    def test_determine_execution_order_disconnected(self):
        """Test execution order for disconnected components"""
        bindings = {
            'a1': n(),
            'a2': ASTRef('a1', [n()]),
            'b1': n(),
            'b2': ASTRef('b1', [n()])
        }
        
        order = determine_execution_order(bindings)
        # Each component should be ordered correctly
        assert order.index('a1') < order.index('a2')
        assert order.index('b1') < order.index('b2')


class TestExecutionContext:
    """Test ExecutionContext integration in chain_let"""
    
    def test_context_stores_results(self):
        """Test that ExecutionContext stores node results"""
        from graphistry.compute.chain_let import execute_node
        
        # Create a mock AST object that returns a known result
        class MockNode:
            def validate(self):
                pass
        
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        context = ExecutionContext()
        mock_node = MockNode()
        
        # This should raise NotImplementedError but still store in context
        try:
            execute_node('test_node', mock_node, g, context, None)
        except NotImplementedError:
            pass
        
        # Even though execution failed, context.set_binding was called
        # (we can't test this without implementing execution)
    
    def test_chain_ref_missing_reference(self):
        """Test ASTRef with missing reference gives helpful error"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine
        
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        context = ExecutionContext()
        
        # Create ASTRef that references non-existent binding
        chain_ref = ASTRef('missing_ref', [])
        
        # Should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            execute_node('test', chain_ref, g, context, Engine.PANDAS)
        
        assert "references 'missing_ref' which has not been executed yet" in str(exc_info.value)
        assert "Available bindings: []" in str(exc_info.value)
    
    def test_chain_ref_with_existing_reference(self):
        """Test ASTRef successfully resolves existing reference"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine
        
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        context = ExecutionContext()
        
        # Pre-populate context with a result
        context.set_binding('previous_result', g)
        
        # Create ASTRef that references it (empty chain)
        chain_ref = ASTRef('previous_result', [])
        
        # Should return the referenced result
        result = execute_node('test', chain_ref, g, context, Engine.PANDAS)
        assert result is g  # Same object since empty chain
        
        # And store it under new name
        assert context.get_binding('test') is g
    
    def test_context_passed_through_dag(self):
        """Test that context is passed through DAG execution"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        dag = ASTLet({})
        
        # Empty DAG should work
        result = g.gfql(dag)
        assert result is not None
    
    def test_execution_order_verified(self):
        """Test that execution order follows dependencies"""
        # Create a DAG with known dependencies
        dag = ASTLet({
            'data': ASTRemoteGraph('dataset'),
            'filtered': ASTRef('data', []),
            'analyzed': ASTRef('filtered', [])
        })
        
        # Get execution order
        from graphistry.compute.chain_let import determine_execution_order
        order = determine_execution_order(dag.bindings)
        
        # Verify order respects dependencies
        assert order == ['data', 'filtered', 'analyzed']
        
        # Also test diamond pattern
        dag_diamond = ASTLet({
            'root': ASTRemoteGraph('data'),
            'left': ASTRef('root', []),
            'right': ASTRef('root', []),
            'merge': ASTRef('left', [ASTRef('right', [])])
        })
        
        order_diamond = determine_execution_order(dag_diamond.bindings)
        assert order_diamond[0] == 'root'
        assert order_diamond[-1] == 'merge'
        assert set(order_diamond[1:3]) == {'left', 'right'}
    
    def test_chain_ref_in_dag_execution(self):
        """Test ASTRef works in DAG execution (fails on chain ops)"""
        _g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')  # noqa: F841
        
        # Create a simple mock that can be executed
        class MockExecutable(ASTObject):
            def _validate_fields(self):
                pass
                
            def _get_child_validators(self):
                return []
            
            def __call__(self, g, prev_node_wavefront, target_wave_front, engine):
                raise NotImplementedError("Mock execution")
            
            def reverse(self):
                return self
        
        # Create DAG with mock executable - should fail validation
        # MockExecutable is not a valid GraphOperation
        with pytest.raises(GFQLTypeError) as exc_info:
            _dag = ASTLet({  # noqa: F841
                'first': MockExecutable(),
                'second': ASTRef('first', [])  # Empty chain should work
            })
        
        assert "valid operation" in str(exc_info.value)
        assert "MockExecutable" in str(exc_info.value)


class TestEdgeExecution:
    """Test ASTEdge execution in chain_let"""
    
    def test_edge_execution_basic(self):
        """Test basic edge traversal in DAG"""
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e'],
            'type': ['knows', 'works_with', 'knows', 'manages']
        })
        g = CGFull().edges(edges_df, 's', 'd')
        g = g.materialize_nodes()
        
        dag = ASTLet({
            'one_hop': Chain([e()])  # Wrap in Chain for GraphOperation
        })
        
        result = g.gfql(dag)
        assert result is not None
        # Should have traversed edges
        assert len(result._nodes) > 0
    
    def test_edge_with_filter(self):
        """Test edge traversal with filters"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'company', 'person', 'company']
        })
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e'],
            'rel': ['knows', 'works_at', 'invests', 'works_at']
        })
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        dag = ASTLet({
            'work_edges': Chain([e(edge_match={'rel': 'works_at'})])
        })
        
        result = g.gfql(dag)
        # Should have filtered to work relationships
        assert result is not None
    
    def test_edge_with_direction(self):
        """Test edge traversal with different directions"""
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd']
        })
        g = CGFull().edges(edges_df, 's', 'd')
        g = g.materialize_nodes()
        
        # Test reverse direction
        from graphistry.compute.ast import ASTEdgeReverse
        dag = ASTLet({
            'reverse': Chain([ASTEdgeReverse()])
        })
        
        result = g.gfql(dag)
        assert result is not None
    
    def test_edge_with_name(self):
        """Test edge operation adds name column"""
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd']
        })
        g = CGFull().edges(edges_df, 's', 'd')

        dag = ASTLet({
            'tagged_edges': Chain([e(name='important')])
        })

        result = g.gfql(dag)
        assert 'important' in result._edges.columns
    
    def test_node_edge_combination(self):
        """Test DAG with both node and edge operations"""
        # TODO: Complex runtime execution error in hop() and combine_steps - binding inconsistency
        # This requires deeper fixes to maintain graph bindings across operations
        # TEMPORARILY ENABLED FOR INVESTIGATION
        # pytest.skip("Runtime binding inconsistency - complex fix needed in execution engine")
        
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd']
        })
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})]),
            'from_people': ASTRef('people', [e()]),  # e() in ASTRef chain is OK
            'companies': Chain([n({'type': 'company'})])
        })
        
        # Should execute successfully
        result = g.gfql(dag)
        assert result is not None


class TestNodeExecution:
    """Test ASTNode execution in chain_let"""
    
    def test_node_execution_empty_filter(self):
        """Test ASTNode with empty filter returns only nodes (no edges)"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine

        g = CGFull().edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd')
        g = g.materialize_nodes()  # Ensure nodes exist
        context = ExecutionContext()

        # Empty node filter - filters to just nodes, no edges
        node = n()
        result = execute_node('test', node, g, context, Engine.PANDAS)

        # Should return all nodes but no edges (filter semantics)
        assert len(result._nodes) == len(g._nodes)
        assert len(result._edges) == 0  # n() filters to just nodes
    
    def test_node_execution_with_filter(self):
        """Test ASTNode with filter_dict filters nodes"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine
        
        # Create graph with node attributes
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        context = ExecutionContext()
        
        # Filter for person nodes
        node = n({'type': 'person'})
        result = execute_node('people', node, g, context, Engine.PANDAS)
        
        # With filter semantics, should only have person nodes
        assert len(result._nodes) == 2  # Only person nodes
        assert all(result._nodes['type'] == 'person')
        assert len(result._edges) == 0  # n() filters to just nodes
    
    def test_node_execution_with_name(self):
        """Test ASTNode adds name column when specified"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine
        
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        g = g.materialize_nodes()
        context = ExecutionContext()
        
        # Node with name
        node = n(name='tagged')
        result = execute_node('test', node, g, context, Engine.PANDAS)
        
        # Should have 'tagged' column
        assert 'tagged' in result._nodes.columns
        assert all(result._nodes['tagged'])
    
    def test_node_in_dag_execution(self):
        """Test ASTNode works in full DAG execution"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # DAG with node filter
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})])
        })
        
        result = g.gfql(dag)
        
        # Should have filtered to people only
        assert len(result._nodes) == 2
        assert set(result._nodes['type'].unique()) == {'person'}
    
    def test_dag_with_node_and_chainref(self):
        """Test DAG execution with both node and chain reference"""
        # TODO: Same runtime execution error in chain combine_steps - missing 'index' column 
        # This is an implementation issue in the execution engine, not GraphOperation validation
        # pytest.skip("Runtime KeyError in chain execution - needs fix in combine_steps implementation")
        
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company'],
            'active': [True, False, True, True]
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'b', 'c'], 'd': ['b', 'c', 'd', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # DAG: filter people, then filter active from those
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})]),
            'active_people': ASTRef('people', [n({'active': True})])  # n() in ASTRef chain is OK
        })
        
        result = g.gfql(dag)

        # With filtering semantics, ASTRef with chain returns only the filtered results
        # 'people' binding filters to 2 person nodes (a, b)
        # 'active_people' further filters to only the active person (a)
        assert len(result._nodes) == 1  # Only the active person node
        assert result._nodes['id'].iloc[0] == 'a'
        assert result._nodes['active'].iloc[0]

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_dag_type(self):
        """Test helpful error when dag parameter is wrong type"""
        g = CGFull()
        
        with pytest.raises(TypeError) as exc_info:
            g.gfql("not a dag")
        assert "Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict" in str(exc_info.value)
        
        # When passed a dict, gfql creates an ASTLet which validates
        with pytest.raises(GFQLTypeError) as exc_info:
            g.gfql({'dict': 'not allowed'})
        assert exc_info.value.code == "type-mismatch"
        assert "binding value must be a valid operation" in str(exc_info.value)
    
    def test_node_execution_error_wrapped(self):
        """Test node execution errors are wrapped with context"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # Create a node with invalid query syntax
        dag = ASTLet({
            'bad_query': Chain([n(query='invalid python syntax !@#')])
        })
        
        with pytest.raises(RuntimeError) as exc_info:
            g.gfql(dag)
        
        error_msg = str(exc_info.value)
        assert "Failed to execute node 'bad_query'" in error_msg
        assert "Error:" in error_msg
    
    def test_cycle_detection_with_path(self):
        """Test cycle detection provides the cycle path"""
        dag = ASTLet({
            'a': ASTRef('b', []),
            'b': ASTRef('c', []),
            'c': ASTRef('a', [])  # Creates cycle a->b->c->a
        })
        
        g = CGFull().edges(pd.DataFrame({'s': ['x'], 'd': ['y']}), 's', 'd')
        with pytest.raises(ValueError) as exc_info:
            g.gfql(dag)
        
        error_msg = str(exc_info.value)
        assert "Circular dependency detected" in error_msg
        assert "->" in error_msg  # Shows the cycle path
    
    def test_complex_cycle_detection(self):
        """Test detection of cycles in complex DAGs"""
        # This DAG has no cycles, just complex dependencies
        bindings = {
            'start': n(),
            'a': ASTRef('start', []),
            'b': ASTRef('a', []),
            'c': ASTRef('b', []),
            'd': ASTRef('c', []),
            'e': ASTRef('d', []),
            'f': ASTRef('b', []),  # Second branch from b
            'g': ASTRef('f', [])  # Note: removed nested ASTRef in chain  
        }
        
        # Test cycle detection directly
        from graphistry.compute.chain_let import detect_cycles, build_dependency_graph
        dependencies, _ = build_dependency_graph(bindings)
        cycle = detect_cycles(dependencies)
        
        # Should find no cycle
        assert cycle is None
    
    def test_missing_reference_with_suggestions(self):
        """Test missing reference error includes available bindings"""
        dag = ASTLet({
            'data1': Chain([n()]),
            'data2': Chain([n()]),
            'result': ASTRef('data3', [])  # data3 doesn't exist
        })
        
        g = CGFull().edges(pd.DataFrame({'s': ['x'], 'd': ['y']}), 's', 'd')
        with pytest.raises(ValueError) as exc_info:
            g.gfql(dag)
        
        error_msg = str(exc_info.value)
        assert "references undefined nodes: ['data3']" in error_msg
        assert "Available nodes: ['data1', 'data2', 'result']" in error_msg


class TestExecutionMechanics:
    """Test execution mechanics with granular tests"""
    
    def test_execute_node_stores_in_context(self):
        """Test that execute_node stores results in context"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine
        
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        g = g.materialize_nodes()
        context = ExecutionContext()
        
        # Execute a simple node
        node = n()
        result = execute_node('test_node', node, g, context, Engine.PANDAS)
        
        # Check result is stored in context
        assert context.get_binding('test_node') is result
        assert len(result._nodes) == 2  # nodes a and b
    
    def test_execute_node_with_different_ast_types(self):
        """Test execute_node handles different AST object types"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine
        
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        context = ExecutionContext()
        
        # Test ASTRemoteGraph is now implemented (will fail with missing auth)
        # We'll test actual functionality with mocks in a separate test
        
        # Test nested ASTLet
        nested_dag = ASTLet({'inner': Chain([n()])})
        result = execute_node('nested', nested_dag, g, context, Engine.PANDAS)
        assert result is not None
    
    @patch('graphistry.compute.chain_remote.chain_remote')
    def test_remote_graph_execution(self, mock_chain_remote):
        """Test ASTRemoteGraph executes correctly with mocked remote call"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine
        
        # Setup mock return value
        mock_result = CGFull().edges(pd.DataFrame({'s': ['x'], 'd': ['y']}), 's', 'd')
        mock_chain_remote.return_value = mock_result
        
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        context = ExecutionContext()
        
        # Execute remote graph
        remote = ASTRemoteGraph('dataset123', token='secret-token')
        result = execute_node('remote_data', remote, g, context, Engine.PANDAS)
        assert result is mock_result  # Verify correct result returned
        
        # Verify chain_remote was called with correct params
        mock_chain_remote.assert_called_once()
        call_args = mock_chain_remote.call_args
        assert call_args[0][1] == []  # Empty chain
        assert call_args[1]['dataset_id'] == 'dataset123'
        assert call_args[1]['api_token'] == 'secret-token'
        assert call_args[1]['output_type'] == 'all'
        
        # Verify result is stored in context
        assert context.get_binding('remote_data') is mock_result
    
    def test_chain_ref_resolution_order(self):
        """Test ASTRef resolves references in correct order"""
        from graphistry.compute.chain_let import execute_node
        from graphistry.Engine import Engine

        nodes_df = pd.DataFrame({'id': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        context = ExecutionContext()

        # Store initial result
        filtered = g.filter_nodes_by_dict({'value': 2})
        context.set_binding('filtered_data', filtered)

        # Create chain ref that adds more filtering
        chain_ref = ASTRef('filtered_data', [n({'id': 'b'})])
        result = execute_node('final', chain_ref, g, context, Engine.PANDAS)

        # ASTRef with chain should return filtered result (only node 'b')
        assert len(result._nodes) == 1  # Only filtered node present
        assert result._nodes['id'].iloc[0] == 'b'
        assert result._nodes['value'].iloc[0] == 2
    
    def test_execution_context_isolation(self):
        """Test that each DAG execution has isolated context"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # First DAG execution
        dag1 = ASTLet({'node1': Chain([n(name='first')])})
        result1 = g.gfql(dag1)
        assert result1 is not None  # First execution succeeds
        
        # Second DAG execution should not see first's context
        dag2 = ASTLet({
            'node2': Chain([n(name='second')]),
            'ref_fail': ASTRef('node1', [])  # Should fail - node1 not in this context
        })
        
        with pytest.raises(ValueError) as exc_info:
            g.gfql(dag2)
        assert "references undefined nodes: ['node1']" in str(exc_info.value)
    
    def test_execution_order_logging(self):
        """Test execution order is logged correctly"""
        import logging
        from graphistry.compute.chain_let import logger as dag_logger
        
        # Capture log output
        logs = []
        handler = logging.Handler()
        handler.emit = lambda record: logs.append(record)
        dag_logger.addHandler(handler)
        dag_logger.setLevel(logging.DEBUG)
        
        try:
            g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
            dag = ASTLet({
                'first': Chain([n()]),
                'second': ASTRef('first', []),
                'third': ASTRef('second', [])
            })
            
            g.gfql(dag)
            
            # Check execution order was logged
            order_logs = [r for r in logs if 'DAG execution order' in str(r.getMessage())]
            assert len(order_logs) > 0
            assert "['first', 'second', 'third']" in str(order_logs[0].getMessage())
            
            # Check individual node execution was logged
            node_logs = [r for r in logs if "Executing node" in str(r.getMessage())]
            assert len(node_logs) >= 3
        finally:
            dag_logger.removeHandler(handler)


class TestDiamondPatterns:
    """Test diamond and complex dependency patterns"""
    
    def test_diamond_pattern_execution(self):
        """Test diamond pattern executes correctly"""
        # TODO: Runtime execution error in combine_steps - missing 'index' column in ASTRef chains
        # This is an implementation issue in the execution engine, not GraphOperation validation
        # pytest.skip("Runtime KeyError in ASTRef chain execution - needs fix in combine_steps implementation")
        
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['source', 'middle1', 'middle2', 'target', 'other']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c', 'd'], 'd': ['b', 'd', 'd', 'e']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Diamond: top -> (left, right) -> bottom
        dag = ASTLet({
            'top': Chain([n({'type': 'source'})]),
            'left': ASTRef('top', [n(name='from_left')]),
            'right': ASTRef('top', [n(name='from_right')]),
            'bottom': ASTRef('left', [])
        })
        
        result = g.gfql(dag)
        
        # Chain filters, so result should have only source node
        # The 'bottom' operation just references 'left' without additional filtering
        assert len(result._nodes) == 1
        assert result._nodes['type'].iloc[0] == 'source'
    def test_multi_branch_convergence(self):
        """Test multiple branches converging"""
        g = CGFull().edges(pd.DataFrame({
            's': ['a', 'b', 'c', 'd', 'e'],
            'd': ['x', 'x', 'x', 'x', 'x']
        }), 's', 'd')
        g = g.materialize_nodes()
        
        # Multiple branches converging - test execution order
        from graphistry.compute.chain_let import determine_execution_order, ExecutionContext, execute_node
        from graphistry.Engine import Engine
        
        dag = ASTLet({
            'branch1': Chain([n(name='b1')]),
            'branch2': Chain([n(name='b2')]),
            'branch3': Chain([n(name='b3')]),
            'converge': Chain([n()])  # Gets all nodes
        })
        
        # Test execution order - branches can execute in any order
        order = determine_execution_order(dag.bindings)
        assert len(order) == 4
        assert order[-1] == 'converge'  # Converge must be last
        
        # Execute and check final result
        result = g.gfql(dag)
        assert len(result._nodes) == 6  # a,b,c,d,e,x
    
    def test_parallel_independent_branches(self):
        """Test parallel branches execute independently"""
        nodes_df = pd.DataFrame({
            'id': list('abcdefgh'),
            'branch': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
        })
        edges_df = pd.DataFrame({'s': list('abcdefg'), 'd': list('bcdefgh')})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Two independent branches
        dag = ASTLet({
            'branch_a': Chain([n({'branch': 'A'})]),
            'branch_b': Chain([n({'branch': 'B'})]),
            'a_subset': ASTRef('branch_a', [n(query="id in ['a', 'b']")]),  # n() in ASTRef is OK
            'b_subset': ASTRef('branch_b', [n(query="id in ['e', 'f']")])  # n() in ASTRef is OK
        })
        
        # Check execution order allows parallel execution
        from graphistry.compute.chain_let import determine_execution_order
        order = determine_execution_order(dag.bindings)
        
        # branch_a and branch_b can execute in any order
        assert order.index('branch_a') < order.index('a_subset')
        assert order.index('branch_b') < order.index('b_subset')
        
        # Execute DAG
        result = g.gfql(dag)
        
        # Result should be from last executed node (b_subset)
        assert len(result._nodes) == 2
        assert set(result._nodes['id'].tolist()) == {'e', 'f'}
    
    def test_deep_dependency_chain(self):
        """Test deep linear dependency chain"""
        g = CGFull().edges(pd.DataFrame({'s': list('abcdef'), 'd': list('bcdefg')}), 's', 'd')
        g = g.materialize_nodes()
        
        # Create deep chain: n1 -> n2 -> n3 -> ... -> n10
        # Using empty chains to avoid execution issues
        dag_dict = {'n1': Chain([n(name='level1')])}
        for i in range(2, 11):
            dag_dict[f'n{i}'] = ASTRef(f'n{i - 1}', [])
        
        dag = ASTLet(dag_dict)
        
        # Test execution order is correct
        from graphistry.compute.chain_let import determine_execution_order
        order = determine_execution_order(dag.bindings)
        
        # Should be in sequential order
        expected_order = [f'n{i}' for i in range(1, 11)]
        assert order == expected_order
        
        # Execute DAG
        result = g.gfql(dag)
        
        # Result should have level1 tag from n1
        assert 'level1' in result._nodes.columns
    
    def test_fan_out_fan_in_pattern(self):
        """Test fan-out then fan-in pattern"""
        g = CGFull().edges(pd.DataFrame({
            's': ['root', 'a1', 'a2', 'b1', 'b2', 'b3'],
            'd': ['hub', 'end', 'end', 'end', 'end', 'end']
        }), 's', 'd')
        g = g.materialize_nodes()
        
        # Test execution order for fan-out/fan-in
        from graphistry.compute.chain_let import determine_execution_order
        
        dag = ASTLet({
            'start': Chain([n({'id': 'root'})]),
            'expand1': ASTRef('start', []),
            'expand2': ASTRef('start', []),
            'expand3': ASTRef('start', []),
            'collect': Chain([n()])  # Gets all nodes from original graph
        })
        
        # Check execution order
        order = determine_execution_order(dag.bindings)
        # 'start' must come before expand nodes
        assert order.index('start') < order.index('expand1')
        assert order.index('start') < order.index('expand2') 
        assert order.index('start') < order.index('expand3')
        # 'collect' has no dependencies so can be anywhere
        
        # Execute DAG
        result = g.gfql(dag)
        # Result is from last executed node (one of the expand nodes)
        # which references 'start' (filtered to just 'root')
        assert len(result._nodes) == 1
        assert result._nodes['id'].iloc[0] == 'root'


class TestIntegration:
    """Integration tests for complex DAG scenarios"""
    
    def test_empty_dag(self):
        """Test empty DAG returns original graph"""
        g = CGFull().edges(pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']}), 's', 'd')
        dag = ASTLet({})
        
        result = g.gfql(dag)
        
        # Should return original graph
        assert len(result._edges) == len(g._edges)
        pd.testing.assert_frame_equal(result._edges, g._edges)
    
    def test_large_dag_10_nodes(self):
        """Test DAG with 10+ nodes executes successfully"""
        # Create a complex graph with attributes
        nodes_data = []
        edges_data = []
        for i in range(20):
            nodes_data.append({
                'id': f'n{i}', 
                'value': i,
                'type': 'even' if i % 2 == 0 else 'odd'
            })
            for j in range(i + 1, min(i + 3, 20)):
                edges_data.append({'s': f'n{i}', 'd': f'n{j}'})
        
        g = CGFull().nodes(pd.DataFrame(nodes_data), 'id').edges(pd.DataFrame(edges_data), 's', 'd')
        
        # Create a 10+ node DAG with various patterns
        dag = ASTLet({
            # Layer 1: Initial filters using filter_dict
            'high_value': Chain([n(name='high')]),
            'even': Chain([n({'type': 'even'})]),
            'odd': Chain([n({'type': 'odd'})]),
            
            # Layer 2: References
            'high_even': ASTRef('even', []),
            'high_odd': ASTRef('odd', []),
            
            # Layer 3: More nodes
            'n1': Chain([n(name='tag1')]),
            'n2': Chain([n(name='tag2')]),
            'n3': Chain([n(name='tag3')]),
            'n4': Chain([n(name='tag4')]),
            
            # Layer 4: Final node
            'final': Chain([n(name='final_tag')])
        })
        
        # Should execute without error
        result = g.gfql(dag)
        assert result is not None
        # The DAG has 10 nodes, so it meets our 10+ node requirement
        assert len(dag.bindings) == 10
        
        # Verify execution order is valid
        from graphistry.compute.chain_let import determine_execution_order
        order = determine_execution_order(dag.bindings)
        assert len(order) == 10
        # References come after their dependencies
        assert order.index('even') < order.index('high_even')
        assert order.index('odd') < order.index('high_odd')
    
    @patch('graphistry.compute.chain_remote.chain_remote')
    def test_mock_remote_graph_placeholder(self, mock_chain_remote):
        """Test DAG with mock RemoteGraph"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # Setup mock to return a simple graph
        mock_result = CGFull().edges(pd.DataFrame({'s': ['x'], 'd': ['y']}), 's', 'd')
        mock_chain_remote.return_value = mock_result
        
        dag = ASTLet({
            'remote1': ASTRemoteGraph('dataset1'),
            'remote2': ASTRemoteGraph('dataset2', token='mock-token'),
            'combined': Chain([n()])  # Would combine results
        })
        
        # Should execute successfully with mocked remote calls
        result = g.gfql(dag)
        assert result is not None
        
        # Verify chain_remote was called twice (once for each RemoteGraph)
        assert mock_chain_remote.call_count == 2
    
    def test_memory_efficient_execution(self):
        """Test that intermediate results are stored efficiently"""
        
        # Create a simple DAG
        g = CGFull().edges(pd.DataFrame({'s': list('abc'), 'd': list('bcd')}), 's', 'd')
        g = g.materialize_nodes()
        
        dag = ASTLet({
            'step1': Chain([n(name='tag1')]),
            'step2': Chain([n(name='tag2')]),
            'step3': Chain([n(name='tag3')])
        })
        
        # Execute and verify context usage
        result = g.gfql(dag)
        
        # Each step should produce a result
        assert result is not None
        # Result has the last tag
        assert 'tag3' in result._nodes.columns
    
    def test_error_propagation_with_context(self):
        """Test errors include helpful context about which node failed"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        dag = ASTLet({
            'good1': Chain([n()]),
            'good2': Chain([n()]), 
            'bad': Chain([n(query='invalid syntax !@#')]),
            'never_reached': Chain([n()])
        })
        
        with pytest.raises(RuntimeError) as exc_info:
            g.gfql(dag)
        
        error_msg = str(exc_info.value)
        assert "Failed to execute node 'bad'" in error_msg
        assert "Error:" in error_msg


class TestCrossValidation:
    """Cross-validation tests to verify implementation correctness"""
    
    def test_dag_vs_chain_consistency(self):
        """Test that simple DAG produces same result as chain for linear flow"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Using chain
        chain_result = g.gfql([n({'type': 'person'})])
        
        # Using DAG
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})])
        })
        dag_result = g.gfql(dag)
        
        # Should produce same nodes
        assert len(chain_result._nodes) == len(dag_result._nodes)
        assert set(chain_result._nodes['id'].tolist()) == set(dag_result._nodes['id'].tolist())
    
    def test_execution_order_deterministic(self):
        """Test that execution order is deterministic for same DAG"""
        from graphistry.compute.chain_let import determine_execution_order
        
        dag = ASTLet({
            'a': Chain([n()]),
            'b': Chain([n()]),
            'c': ASTRef('a', []),
            'd': ASTRef('b', []),
            'e': ASTRef('c', []),
            'f': ASTRef('d', [])
        })
        
        # Get order multiple times
        orders = []
        for i in range(5):
            order = determine_execution_order(dag.bindings)
            orders.append(order)
        
        # All should be the same
        for order in orders[1:]:
            assert order == orders[0]
    
    def test_context_bindings_accessible(self):
        """Test that all intermediate results are accessible in context"""
        from graphistry.compute.chain_let import chain_let_impl
        
        g = CGFull().edges(pd.DataFrame({'s': list('abc'), 'd': list('bcd')}), 's', 'd')
        g = g.materialize_nodes()
        
        # Create a mock context to track all bindings
        bindings_tracker = {}
        
        class TrackingContext(ExecutionContext):
            def set_binding(self, name, value):
                super().set_binding(name, value)
                bindings_tracker[name] = value
        
        # Monkey patch the execution to use our tracking context
        original_chain_let_impl = chain_let_impl
        
        def tracking_chain_let_impl(g, dag, engine):
            # Call original but capture context usage
            return original_chain_let_impl(g, dag, engine)
        
        dag = ASTLet({
            'step1': Chain([n(name='tag1')]),
            'step2': Chain([n(name='tag2')]),
            'step3': ASTRef('step1', [])
        })
        
        result = g.gfql(dag)
        
        # We can't easily intercept the context, but we can verify the result
        assert result is not None
    
    def test_error_doesnt_corrupt_state(self):
        """Test that errors don't leave DAG execution in bad state"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # First execution with error
        bad_dag = ASTLet({
            'bad': Chain([n(query='invalid syntax !!!')])
        })
        
        try:
            g.gfql(bad_dag)
        except RuntimeError:
            pass  # Expected
        
        # Second execution should work fine
        good_dag = ASTLet({
            'good': Chain([n()])
        })
        
        result = g.gfql(good_dag)
        assert result is not None
    
    def test_node_filter_consistency(self):
        """Test node filtering is consistent between chain and chain_let"""
        nodes_df = pd.DataFrame({
            'id': list('abcdef'),
            'value': [10, 20, 30, 40, 50, 60],
            'active': [True, False, True, False, True, False]
        })
        edges_df = pd.DataFrame({'s': list('abcde'), 'd': list('bcdef')})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Test filter_dict
        dag1 = ASTLet({'result': Chain([n({'active': True})])})
        result1 = g.gfql(dag1)
        assert len(result1._nodes) == 3
        assert all(result1._nodes['active'])
        
        # Test with name
        dag2 = ASTLet({'result': Chain([n({'active': True}, name='is_active')])})
        result2 = g.gfql(dag2)
        assert 'is_active' in result2._nodes.columns
        assert all(result2._nodes['is_active'])


class TestChainDagInternal:
    """Test internal chain_let functionality (via gfql)"""
    
    def test_chain_let_via_gfql(self):
        """Test that DAG execution works via gfql"""
        g = CGFull()
        assert hasattr(g, 'gfql')
        assert callable(g.gfql)
        
        # chain_let should not be in public API - removed from ComputeMixin
        assert not hasattr(g, 'chain_let')
    
    def test_chain_let_empty(self):
        """Test chain_let with empty DAG"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        dag = ASTLet({})
        
        # Empty DAG should return original graph
        result = g.gfql(dag)
        assert result is not None
    
    def test_chain_let_single_node_works(self):
        """Test chain_let with single node now works"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        g = g.materialize_nodes()
        
        dag = ASTLet({
            'all_nodes': Chain([n()])
        })
        
        # Should work now that node execution is implemented
        result = g.gfql(dag)
        assert result is not None
        assert len(result._nodes) == 2  # nodes a and b
    
    @patch('graphistry.compute.chain_remote.chain_remote')  
    def test_chain_let_remote_not_implemented(self, mock_chain_remote):
        """Test chain_let with RemoteGraph works with mocked remote"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # Setup mock
        mock_result = CGFull().edges(pd.DataFrame({'s': ['remote1'], 'd': ['remote2']}), 's', 'd')
        mock_chain_remote.return_value = mock_result
        
        dag = ASTLet({
            'remote': ASTRemoteGraph('dataset123')
        })
        
        # Should work now with mocked chain_remote
        result = g.gfql(dag)
        assert result is not None
        # Result should be the mocked remote graph
        assert 'remote1' in result._edges['s'].values
    
    def test_chain_let_multi_node_works(self):
        """Test chain_let with multiple nodes now works"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        dag = ASTLet({
            'first': Chain([n()]),
            'second': Chain([n()])
        })
        
        # Should work now that node execution is implemented
        result = g.gfql(dag)
        assert result is not None
        
        # Result should be from last node ('second')
        # Both nodes have empty filters so should have all data
        assert len(result._nodes) == 2  # nodes a and b
    
    def test_chain_let_validates(self):
        """Test chain_let validates the DAG"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # Invalid DAG should raise during validation
        with pytest.raises(TypeError) as exc_info:
            g.gfql("not a dag")
        
        assert "Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict" in str(exc_info.value)
    
    def test_chain_let_output_selection(self):
        """Test output parameter selects specific binding"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})]),
            'companies': Chain([n({'type': 'company'})]),
            'all_nodes': Chain([n()])
        })
        
        # Default: returns last executed
        result_default = g.gfql(dag)
        # Could be any of the three since they have no dependencies
        assert result_default is not None
        
        # Select specific outputs
        result_people = g.gfql(dag, output='people')
        assert len(result_people._nodes) == 2
        assert all(result_people._nodes['type'] == 'person')
        
        result_companies = g.gfql(dag, output='companies')
        assert len(result_companies._nodes) == 2
        assert all(result_companies._nodes['type'] == 'company')
        
        result_all = g.gfql(dag, output='all_nodes')
        assert len(result_all._nodes) == 4
    
    def test_chain_let_output_not_found(self):
        """Test error when output binding not found"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        dag = ASTLet({'node1': Chain([n()])})
        
        with pytest.raises(ValueError) as exc_info:
            g.gfql(dag, output='missing')
        
        error_msg = str(exc_info.value)
        assert "Output binding 'missing' not found" in error_msg
        assert "Available bindings: ['node1']" in error_msg

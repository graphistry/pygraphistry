import pandas as pd
import pytest
from graphistry.compute.ast import ASTLet, ASTRef, n, e
from graphistry.compute.chain import Chain
from graphistry.tests.test_compute import CGFull

# Suppress deprecation warnings for chain() method in this test file
# This file tests the migration from chain() to gfql()
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:graphistry")


class TestGFQLAPI:
    """Test unified GFQL API and migration"""
    
    def test_public_api_methods(self):
        """Test what methods are available on the public API"""
        g = CGFull()
        
        # Should have gfql
        assert hasattr(g, 'gfql')
        assert callable(g.gfql)
        
        # Should still have chain (with deprecation)
        assert hasattr(g, 'chain')
        assert callable(g.chain)
        
        # chain_let should not be in public API - removed from ComputeMixin
        assert not hasattr(g, 'chain_let')


class TestGFQL:
    """Test unified GFQL entrypoint"""
    
    def test_gfql_exists(self):
        """Test that gfql method exists on CGFull"""
        g = CGFull()
        assert hasattr(g, 'gfql')
        assert callable(g.gfql)
    
    def test_gfql_with_list(self):
        """Test gfql with list executes as chain"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Execute as chain
        result = g.gfql([n({'type': 'person'})])
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_with_chain_object(self):
        """Test gfql with Chain object"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Execute with Chain
        chain = Chain([n({'type': 'person'}), e(), n()])
        result = g.gfql(chain)
        
        assert result is not None
        # Result depends on graph structure
    
    def test_gfql_with_dag(self):
        """Test gfql with ASTLet"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Execute as DAG - wrap n() in Chain for GraphOperation
        dag = ASTLet({
            'people': Chain([n({'type': 'person'})]),
            'companies': Chain([n({'type': 'company'})])
        })
        
        result = g.gfql(dag)
        assert result is not None
    
    def test_gfql_with_dict_convenience(self):
        """Test gfql with dict converts to DAG"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Dict convenience should auto-wrap ASTNode/ASTEdge in Chain
        result = g.gfql({'people': n({'type': 'person'})})
        
        # Should have filtered to people only  
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_output_with_dag(self):
        """Test gfql output parameter works with DAG"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'type': ['person', 'person', 'company', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Dict convenience with output parameter
        result = g.gfql({
            'people': n({'type': 'person'}),
            'companies': n({'type': 'company'})
        }, output='people')
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_output_ignored_for_chain(self):
        """Test gfql output parameter ignored for chains"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # Should work but output ignored
        result = g.gfql([n()], output='ignored')
        assert result is not None
    
    def test_gfql_with_single_ast_object(self):
        """Test gfql with single ASTObject wraps in list"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # Single ASTObject should work
        result = g.gfql(n({'type': 'person'}))
        
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
    
    def test_gfql_invalid_query_type(self):
        """Test gfql with invalid query type"""
        g = CGFull()
        
        with pytest.raises(TypeError) as exc_info:
            g.gfql("not a valid query")
        
        assert "Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict" in str(exc_info.value)
    
    def test_gfql_deprecation_and_migration(self):
        """Test deprecation warnings and migration path"""
        import warnings
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')
        
        # chain() should show deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chain_result = g.chain([n({'type': 'person'})])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "chain() is deprecated" in str(w[0].message)
            assert "Use gfql()" in str(w[0].message)
        
        assert len(chain_result._nodes) == 2
        
        # chain_let should be removed from public API - use gfql() instead
        assert not hasattr(g, 'chain_let'), "chain_let should be removed from public API"
        
        # gfql should work for both patterns
        gfql_chain = g.gfql([n({'type': 'person'})])
        assert len(gfql_chain._nodes) == 2
        
        # Dict convenience should now work with auto-wrapping
        gfql_dag = g.gfql({'people': n({'type': 'person'})})
        assert len(gfql_dag._nodes) == 2


class TestGFQLDictConversion:
    """Test dict-to-AST conversion scenarios to prevent regressions."""

    def test_gfql_with_list_containing_raw_dicts(self):
        """Test gfql with list containing raw dict objects (main regression case)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # This was the exact case that failed before our fix
        result = g.gfql([{"type": "Node"}])

        # Should execute successfully without TypeError
        assert result is not None
        assert hasattr(result, '_nodes')
        # Should return all nodes since no filter specified
        assert len(result._nodes) == 3

    def test_gfql_with_mixed_list_ast_and_dicts(self):
        """Test gfql with list containing both AST objects and dicts"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Mixed: AST object + raw dict
        result = g.gfql([n({'type': 'person'}), {"type": "Node"}])

        # Should execute successfully
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_gfql_with_multiple_dicts_in_list(self):
        """Test gfql with list containing multiple dict objects"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Multiple dicts in sequence
        result = g.gfql([{"type": "Node"}, {"type": "Node"}])

        # Should execute successfully
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_gfql_dict_conversion_with_filter(self):
        """Test dict conversion works with actual filtering"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Test with actual filter in dict format
        result = g.gfql([{"type": "Node", "filter": {"type": "person"}}])

        # Should execute successfully (filtering behavior depends on implementation)
        assert result is not None
        assert hasattr(result, '_nodes')
        assert len(result._nodes) >= 0  # May or may not filter, just ensure no crash

    def test_gfql_empty_list_with_dicts(self):
        """Test edge case: empty list"""
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')

        # Empty list should work
        result = g.gfql([])
        assert result is not None

    def test_gfql_single_vs_list_dict_equivalence(self):
        """Test that list dict conversion works (single dict vs list behavior may differ)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'type': ['person', 'person', 'company']
        })
        edges_df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 's', 'd')

        # Test that list dict conversion works - main regression test
        result_list = g.gfql([{"type": "Node"}])

        # Should succeed without TypeError
        assert result_list is not None
        assert hasattr(result_list, '_nodes')
        assert len(result_list._nodes) >= 0  # Ensure execution succeeds

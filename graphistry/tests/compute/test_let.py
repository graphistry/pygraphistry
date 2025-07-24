"""Tests for Let bindings and related AST nodes validation"""
import pytest
from graphistry.compute.ast import ASTLet, ASTRemoteGraph, ASTChainRef, n, e
from graphistry.compute.execution_context import ExecutionContext


class TestLetValidation:
    """Test validation for Let bindings"""
    
    def test_let_valid(self):
        """Valid Let should pass validation"""
        dag = ASTLet({'a': n(), 'b': e()})
        dag.validate()  # Should not raise
    
    def test_let_invalid_key_type(self):
        """Let with non-string key should fail"""
        with pytest.raises(AssertionError, match="binding key must be string"):
            dag = ASTLet({123: n()})
            dag.validate()
    
    def test_let_invalid_value_type(self):
        """Let with non-ASTObject value should fail"""
        with pytest.raises(AssertionError, match="binding value must be ASTObject"):
            dag = ASTLet({'a': 'not an AST object'})
            dag.validate()
    
    def test_let_nested_validation(self):
        """Let should validate nested objects"""
        # This should work - nested validation of valid objects
        dag = ASTLet({
            'a': n({'type': 'person'}),
            'b': ASTRemoteGraph('dataset123')
        })
        dag.validate()


class TestRemoteGraphValidation:
    """Test validation for RemoteGraph"""
    
    def test_remoteGraph_valid(self):
        """Valid RemoteGraph should pass validation"""
        rg = ASTRemoteGraph('my-dataset')
        rg.validate()  # Should not raise
        
        rg_with_token = ASTRemoteGraph('my-dataset', token='secret')
        rg_with_token.validate()  # Should not raise
    
    def test_remoteGraph_invalid_dataset_type(self):
        """RemoteGraph with non-string dataset_id should fail"""
        with pytest.raises(AssertionError, match="dataset_id must be a string"):
            rg = ASTRemoteGraph(123)
            rg.validate()
    
    def test_remoteGraph_empty_dataset(self):
        """RemoteGraph with empty dataset_id should fail"""
        with pytest.raises(AssertionError, match="dataset_id cannot be empty"):
            rg = ASTRemoteGraph('')
            rg.validate()
    
    def test_remoteGraph_invalid_token_type(self):
        """RemoteGraph with non-string token should fail"""
        with pytest.raises(AssertionError, match="token must be string or None"):
            rg = ASTRemoteGraph('dataset', token=123)
            rg.validate()


class TestChainRefValidation:
    """Test validation for ChainRef"""
    
    def test_chainRef_valid(self):
        """Valid ChainRef should pass validation"""
        cr = ASTChainRef('myref', [n(), e()])
        cr.validate()  # Should not raise
        
        cr_empty = ASTChainRef('myref', [])
        cr_empty.validate()  # Empty chain is valid
    
    def test_chainRef_invalid_ref_type(self):
        """ChainRef with non-string ref should fail"""
        with pytest.raises(AssertionError, match="ref must be a string"):
            cr = ASTChainRef(123, [])
            cr.validate()
    
    def test_chainRef_empty_ref(self):
        """ChainRef with empty ref should fail"""
        with pytest.raises(AssertionError, match="ref cannot be empty"):
            cr = ASTChainRef('', [])
            cr.validate()
    
    def test_chainRef_invalid_chain_type(self):
        """ChainRef with non-list chain should fail"""
        with pytest.raises(AssertionError, match="chain must be a list"):
            cr = ASTChainRef('ref', 'not a list')
            cr.validate()
    
    def test_chainRef_invalid_chain_element(self):
        """ChainRef with non-ASTObject in chain should fail"""
        with pytest.raises(AssertionError, match="must be ASTObject"):
            cr = ASTChainRef('ref', [n(), 'not an AST object'])
            cr.validate()
    
    def test_chainRef_nested_validation(self):
        """ChainRef should validate nested operations"""
        cr = ASTChainRef('ref', [n({'type': 'person'}), e()])
        cr.validate()  # Should validate nested nodes


class TestExecutionContext:
    """Test ExecutionContext functionality"""
    
    def test_context_basic_operations(self):
        """Test basic get/set operations"""
        ctx = ExecutionContext()
        
        # Set and get
        ctx.set_binding('a', 'value_a')
        assert ctx.get_binding('a') == 'value_a'
        
        # Has binding
        assert ctx.has_binding('a') is True
        assert ctx.has_binding('b') is False
        
        # Multiple bindings
        ctx.set_binding('b', 123)
        assert ctx.get_binding('b') == 123
        assert len(ctx.get_all_bindings()) == 2
    
    def test_context_missing_binding(self):
        """Test error on missing binding"""
        ctx = ExecutionContext()
        with pytest.raises(KeyError, match="No binding found for 'missing'"):
            ctx.get_binding('missing')
    
    def test_context_invalid_name_type(self):
        """Test error on non-string binding names"""
        ctx = ExecutionContext()
        
        with pytest.raises(TypeError, match="Binding name must be string"):
            ctx.set_binding(123, 'value')
        
        with pytest.raises(TypeError, match="Binding name must be string"):
            ctx.get_binding(123)
        
        with pytest.raises(TypeError, match="Binding name must be string"):
            ctx.has_binding(123)
    
    def test_context_clear(self):
        """Test clearing all bindings"""
        ctx = ExecutionContext()
        ctx.set_binding('a', 1)
        ctx.set_binding('b', 2)
        assert len(ctx.get_all_bindings()) == 2
        
        ctx.clear()
        assert len(ctx.get_all_bindings()) == 0
        assert ctx.has_binding('a') is False
    
    def test_context_overwrite(self):
        """Test overwriting existing binding"""
        ctx = ExecutionContext()
        ctx.set_binding('a', 'first')
        assert ctx.get_binding('a') == 'first'
        
        ctx.set_binding('a', 'second')
        assert ctx.get_binding('a') == 'second'


class TestChainRefReverse:
    """Test reverse operation for ChainRef"""
    
    def test_chainRef_reverse(self):
        """Test ChainRef reverse reverses operations"""
        cr = ASTChainRef('data', [n(), e(), n()])
        reversed_cr = cr.reverse()
        
        assert isinstance(reversed_cr, ASTChainRef)
        assert reversed_cr.ref == 'data'
        assert len(reversed_cr.chain) == 3
        # Operations should be reversed
        # Original: n, e, n
        # Reversed: n, e.reverse(), n (each op is individually reversed)

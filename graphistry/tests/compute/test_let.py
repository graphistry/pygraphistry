"""Tests for Let bindings and related AST nodes validation"""
import pytest
from graphistry.compute.ast import ASTLet, ASTRemoteGraph, ASTRef, n, e
from graphistry.compute.chain import Chain
from graphistry.compute.execution_context import ExecutionContext
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError


class TestLetValidation:
    """Test validation for Let bindings"""
    
    def test_let_valid(self):
        """Valid Let should pass validation"""
        # Now requires GraphOperations - wrap n()/e() in Chain
        dag = ASTLet({
            'a': Chain([n()]),  # Chain produces a Plottable
            'b': Chain([e()])   # Chain produces a Plottable
        })
        dag.validate()  # Should not raise
    
    def test_let_invalid_key_type(self):
        """Let with non-string key should fail"""
        # Note: This validation happens at runtime in _validate_fields
        # Use valid GraphOperation but invalid key
        dag = ASTLet({123: Chain([n()])}, validate=False)  # type: ignore
        with pytest.raises(GFQLTypeError) as exc_info:
            dag.validate()
        assert exc_info.value.code == ErrorCode.E102
        assert "binding key must be string" in str(exc_info.value)
    
    def test_let_invalid_value_type(self):
        """Let with non-GraphOperation value should fail"""
        dag = ASTLet({'a': 'not an AST object'}, validate=False)  # type: ignore
        with pytest.raises(GFQLTypeError) as exc_info:
            dag.validate()
        assert exc_info.value.code == ErrorCode.E201
        assert "GraphOperation" in str(exc_info.value)
    
    def test_let_nested_validation(self):
        """Let should validate nested objects"""
        # This should work - nested validation of valid objects
        dag = ASTLet({
            'a': Chain([n({'type': 'person'})]),  # Wrap in Chain
            'b': ASTRemoteGraph('dataset123')     # Already valid GraphOperation
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
        rg = ASTRemoteGraph(123)  # type: ignore
        with pytest.raises(GFQLTypeError) as exc_info:
            rg.validate()
        assert exc_info.value.code == "type-mismatch"
        assert "dataset_id must be a string" in str(exc_info.value)
    
    def test_remoteGraph_empty_dataset(self):
        """RemoteGraph with empty dataset_id should fail"""
        rg = ASTRemoteGraph('')
        with pytest.raises(GFQLTypeError) as exc_info:
            rg.validate()
        assert exc_info.value.code == "empty-chain"
        assert "dataset_id cannot be empty" in str(exc_info.value)
    
    def test_remoteGraph_invalid_token_type(self):
        """RemoteGraph with non-string token should fail"""
        rg = ASTRemoteGraph('dataset', token=123)  # type: ignore
        with pytest.raises(GFQLTypeError) as exc_info:
            rg.validate()
        assert exc_info.value.code == "type-mismatch"
        assert "token must be string or None" in str(exc_info.value)


class TestChainRefValidation:
    """Test validation for ChainRef"""
    
    def test_chainRef_valid(self):
        """Valid ChainRef should pass validation"""
        cr = ASTRef('myref', [n(), e()])
        cr.validate()  # Should not raise
        
        cr_empty = ASTRef('myref', [])
        cr_empty.validate()  # Empty chain is valid
    
    def test_chainRef_invalid_ref_type(self):
        """ChainRef with non-string ref should fail"""
        cr = ASTRef(123, [])  # type: ignore
        with pytest.raises(GFQLTypeError) as exc_info:
            cr.validate()
        assert exc_info.value.code == "type-mismatch"
        assert "ref must be a string" in str(exc_info.value)
    
    def test_chainRef_empty_ref(self):
        """ChainRef with empty ref should fail"""
        cr = ASTRef('', [])
        with pytest.raises(GFQLTypeError) as exc_info:
            cr.validate()
        assert exc_info.value.code == "empty-chain"
        assert "ref cannot be empty" in str(exc_info.value)
    
    def test_chainRef_invalid_chain_type(self):
        """ChainRef with non-list chain should fail"""
        cr = ASTRef('ref', 'not a list')  # type: ignore
        with pytest.raises(GFQLTypeError) as exc_info:
            cr.validate()
        assert exc_info.value.code == "type-mismatch"
        assert "chain must be a list" in str(exc_info.value)
    
    def test_chainRef_invalid_chain_element(self):
        """ChainRef with non-ASTObject in chain should fail"""
        cr = ASTRef('ref', [n(), 'not an AST object'])  # type: ignore
        with pytest.raises(GFQLTypeError) as exc_info:
            cr.validate()
        assert exc_info.value.code == "type-mismatch"
        assert "must be ASTObject" in str(exc_info.value)
    
    def test_chainRef_nested_validation(self):
        """ChainRef should validate nested operations"""
        cr = ASTRef('ref', [n({'type': 'person'}), e()])
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
        cr = ASTRef('data', [n(), e(), n()])
        reversed_cr = cr.reverse()
        
        assert isinstance(reversed_cr, ASTRef)
        assert reversed_cr.ref == 'data'
        assert len(reversed_cr.chain) == 3
        # Operations should be reversed
        # Original: n, e, n
        # Reversed: n, e.reverse(), n (each op is individually reversed)

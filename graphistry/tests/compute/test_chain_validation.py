"""Tests for Chain validation with new error system."""

import pytest
from graphistry.compute.chain import Chain
from graphistry.compute.ast import n, e_forward, ASTNode, ASTEdge, ASTLet
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError, GFQLSyntaxError, GFQLTypeError


class TestChainValidation:
    """Test Chain validation with structured errors."""
    
    def test_valid_chain(self):
        """Valid chains pass validation."""
        # Empty operations list is now invalid, but single operation is valid
        chain = Chain([n()])
        chain.validate()  # Should not raise
        
        # Multiple operations
        chain = Chain([n(), e_forward(), n()])
        chain.validate()  # Should not raise
        
        # With collect_all
        errors = chain.validate(collect_all=True)
        assert errors == []

    def test_constructor_auto_validates(self):
        """Chain constructor should validate by default."""
        with pytest.raises(GFQLTypeError) as exc_info:
            Chain([n(), e_forward(hops=-1), n()])

        assert exc_info.value.code == ErrorCode.E103

    def test_constructor_validate_opt_out(self):
        """Allow skipping validation when explicitly requested."""
        chain = Chain([n(), e_forward(hops=-1), n()], validate=False)

        with pytest.raises(GFQLTypeError) as exc_info:
            chain.validate()

        assert exc_info.value.code == ErrorCode.E103

    def test_chain_not_list(self):
        """Chain must be a list."""
        chain = Chain.__new__(Chain)  # Skip __init__
        chain.chain = "not a list"
        
        with pytest.raises(GFQLTypeError) as exc_info:
            chain.validate()
        
        assert exc_info.value.code == ErrorCode.E101
        assert "must be a list" in str(exc_info.value)
        assert "str" in str(exc_info.value)
        assert "Wrap your operations" in str(exc_info.value)
    
    def test_empty_chain(self):
        """Empty chain is valid for backward compatibility."""
        chain = Chain([])
        chain.validate()  # Should not raise
        
        errors = chain.validate(collect_all=True)
        assert errors == []
    
    def test_invalid_operation_type(self):
        """Operations must be ASTObject instances."""
        chain = Chain.__new__(Chain)
        chain.chain = [n(), "not an operation", e_forward()]
        
        with pytest.raises(GFQLTypeError) as exc_info:
            chain.validate()
        
        assert exc_info.value.code == ErrorCode.E101
        assert "index 1" in str(exc_info.value)
        assert "not a valid GFQL operation" in str(exc_info.value)
        assert exc_info.value.context['operation_index'] == 1
        assert "Use n() for nodes" in str(exc_info.value)
    
    def test_chain_validates_children(self):
        """Chain validates child operations."""
        # Create an invalid node (we'll implement this validation next)
        # For now, test that child validation is called
        node = n()
        edge = e_forward()
        chain = Chain([node, edge])
        
        # Should validate children
        chain.validate()  # Currently passes since we haven't updated Node/Edge yet
    
    def test_chain_from_json_valid(self):
        """from_json works with valid data."""
        data = {
            'type': 'Chain',
            'chain': [
                {'type': 'Node', 'filter_dict': {}},
                {'type': 'Edge', 'direction': 'forward'},
                {'type': 'Node', 'filter_dict': {}}
            ]
        }
        
        chain = Chain.from_json(data)
        assert len(chain.chain) == 3
        assert isinstance(chain.chain[0], ASTNode)
        assert isinstance(chain.chain[1], ASTEdge)
    
    def test_chain_from_json_not_dict(self):
        """from_json requires dictionary."""
        with pytest.raises(GFQLSyntaxError) as exc_info:
            Chain.from_json("not a dict")
        
        assert exc_info.value.code == ErrorCode.E101
        assert "must be a dictionary" in str(exc_info.value)
    
    def test_chain_from_json_missing_chain_field(self):
        """from_json requires 'chain' field."""
        with pytest.raises(GFQLSyntaxError) as exc_info:
            Chain.from_json({'type': 'Chain'})
        
        assert exc_info.value.code == ErrorCode.E105
        assert "missing required 'chain' field" in str(exc_info.value)
    
    def test_chain_from_json_chain_not_list(self):
        """'chain' field must be a list."""
        with pytest.raises(GFQLSyntaxError) as exc_info:
            Chain.from_json({'type': 'Chain', 'chain': 'not a list'})
        
        assert exc_info.value.code == ErrorCode.E101
        assert "Chain field must be a list" in str(exc_info.value)
    
    def test_chain_from_json_skip_validation(self):
        """from_json can skip validation on operations."""
        # Operations with invalid field values
        data = {
            'type': 'Chain',
            'chain': [
                {'type': 'Edge', 'direction': 'forward', 'hops': -1},  # Invalid hops
            ]
        }
        
        # Should not raise with validate=False
        chain = Chain.from_json(data, validate=False)
        assert len(chain.chain) == 1
        assert chain.chain[0].hops == -1
        
        # But manual validation would fail
        with pytest.raises(GFQLTypeError) as exc_info:
            chain.validate()
        # Should fail on invalid hops
        assert exc_info.value.code == ErrorCode.E103
    
    def test_chain_from_json_validate_true_invalid_hops(self):
        """from_json validate=True should fail on invalid hops."""
        data = {
            'type': 'Chain',
            'chain': [
                {'type': 'Edge', 'direction': 'forward', 'hops': -1},
            ]
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            Chain.from_json(data, validate=True)

        assert exc_info.value.code == ErrorCode.E103

    def test_nested_chain_validation_via_let(self):
        """Let should validate nested chains by default."""
        bad_chain = Chain([n(), e_forward(hops=-1), n()], validate=False)
        with pytest.raises(GFQLTypeError) as exc_info:
            ASTLet({'bad': bad_chain})  # validate=True default

        assert exc_info.value.code == ErrorCode.E103

    def test_nested_chain_validation_deferred_then_explicit(self):
        """Nested deferred validation should fail when validate() is called."""
        bad_chain = Chain([n(), e_forward(hops=-1), n()], validate=False)
        let = ASTLet({'bad': bad_chain}, validate=False)

        with pytest.raises(GFQLTypeError) as exc_info:
            let.validate()

        assert exc_info.value.code == ErrorCode.E103

    def test_nested_chain_from_json_validate_true_invalid_hops(self):
        """Let.from_json validate=True should fail on nested invalid hops."""
        data = {
            'type': 'Let',
            'bindings': {
                'bad': {
                    'type': 'Chain',
                    'chain': [
                        {'type': 'Edge', 'direction': 'forward', 'hops': -1}
                    ]
                }
            }
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            ASTLet.from_json(data, validate=True)

        assert exc_info.value.code == ErrorCode.E103
    
    def test_chain_collect_all_errors(self):
        """Chain can collect multiple errors."""
        # Create chain with multiple invalid operations
        chain = Chain.__new__(Chain)
        chain.chain = ["invalid1", "invalid2", n(), "invalid3"]
        
        errors = chain.validate(collect_all=True)
        assert len(errors) >= 3  # At least 3 invalid operations
        
        # All should be type errors for invalid operations
        for i, error in enumerate(errors[:3]):
            assert error.code == ErrorCode.E101
            assert "not a valid GFQL operation" in error.message

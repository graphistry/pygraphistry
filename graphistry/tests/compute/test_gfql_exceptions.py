"""Tests for GFQL validation exceptions."""

import os
import warnings
import pandas as pd
import pytest
from graphistry.compute.exceptions import (
    ErrorCode, GFQLValidationError, GFQLSyntaxError,
    GFQLTypeError, GFQLSchemaError
)

_CUDF = pytest.mark.skipif(
    not (os.environ.get("TEST_CUDF") == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestErrorCode:
    """Test error code constants."""
    
    def test_error_codes_exist(self):
        """Error codes are defined."""
        assert ErrorCode.E101 == "invalid-chain-type"
        assert ErrorCode.E102 == "invalid-filter-key"
        assert ErrorCode.E201 == "type-mismatch"
        assert ErrorCode.E301 == "column-not-found"
    
    def test_error_code_ranges(self):
        """Error codes follow range convention."""
        # E1xx for syntax
        assert ErrorCode.E101.startswith("invalid")
        assert ErrorCode.E106 == "empty-chain"
        
        # E2xx for types
        assert ErrorCode.E201 == "type-mismatch"
        assert ErrorCode.E202 == "predicate-type-mismatch"
        
        # E3xx for schema
        assert ErrorCode.E301 == "column-not-found"
        assert ErrorCode.E302 == "incompatible-column-type"


class TestGFQLValidationError:
    """Test base validation error."""
    
    def test_basic_error_creation(self):
        """Can create basic error."""
        error = GFQLValidationError(ErrorCode.E101, "Test message")
        assert error.code == ErrorCode.E101
        assert error.message == "Test message"
        assert str(error).startswith("[invalid-chain-type]")
        assert "Test message" in str(error)
    
    def test_error_with_context(self):
        """Error includes context fields."""
        error = GFQLValidationError(
            ErrorCode.E102,
            "Invalid filter key",
            field="filter_dict.123",
            value=123,
            suggestion="Use string keys"
        )
        
        formatted = str(error)
        assert "[invalid-filter-key]" in formatted
        assert "field: filter_dict.123" in formatted
        assert "value: 123" in formatted
        assert "suggestion: Use string keys" in formatted
    
    def test_error_with_operation_index(self):
        """Error includes operation index."""
        error = GFQLValidationError(
            ErrorCode.E101,
            "Bad operation",
            operation_index=2
        )
        assert "at operation 2" in str(error)
    
    def test_error_truncates_long_values(self):
        """Long values are truncated in string representation."""
        long_value = "x" * 100
        error = GFQLValidationError(
            ErrorCode.E201,
            "Value too long",
            value=long_value
        )
        formatted = str(error)
        assert "..." in formatted
        assert len(formatted) < 200  # Reasonable length
    
    def test_error_to_dict(self):
        """Error converts to dictionary."""
        error = GFQLValidationError(
            ErrorCode.E102,
            "Test error",
            field="test_field",
            value="test_value",
            custom_field="custom"
        )
        
        d = error.to_dict()
        assert d['code'] == ErrorCode.E102
        assert d['message'] == "Test error"
        assert d['field'] == "test_field"
        assert d['value'] == "test_value"
        assert d['custom_field'] == "custom"
    
    def test_error_filters_none_context(self):
        """None values are filtered from context."""
        error = GFQLValidationError(
            ErrorCode.E101,
            "Test",
            field=None,
            value="something",
            suggestion=None
        )
        
        assert 'field' not in error.context
        assert 'suggestion' not in error.context
        assert error.context['value'] == "something"


class TestErrorSubclasses:
    """Test error subclasses."""
    
    def test_syntax_error(self):
        """GFQLSyntaxError works correctly."""
        error = GFQLSyntaxError(
            ErrorCode.E101,
            "Invalid syntax",
            field="chain"
        )
        assert isinstance(error, GFQLValidationError)
        assert error.code == ErrorCode.E101
        assert "Invalid syntax" in str(error)
    
    def test_type_error(self):
        """GFQLTypeError works correctly."""
        error = GFQLTypeError(
            ErrorCode.E201,
            "Type mismatch",
            field="hops",
            value="not_a_number"
        )
        assert isinstance(error, GFQLValidationError)
        assert error.code == ErrorCode.E201
        assert "Type mismatch" in str(error)
        assert "not_a_number" in str(error)
    
    def test_schema_error(self):
        """GFQLSchemaError works correctly."""
        error = GFQLSchemaError(
            ErrorCode.E301,
            "Column not found",
            field="user_id",
            suggestion="Available columns: id, name, email"
        )
        assert isinstance(error, GFQLValidationError)
        assert error.code == ErrorCode.E301
        assert "Column not found" in str(error)
        assert "Available columns" in str(error)
    
    def test_error_inheritance(self):
        """All errors inherit from base class."""
        errors = [
            GFQLSyntaxError(ErrorCode.E101, "test"),
            GFQLTypeError(ErrorCode.E201, "test"),
            GFQLSchemaError(ErrorCode.E301, "test")
        ]
        
        for error in errors:
            assert isinstance(error, GFQLValidationError)
            assert isinstance(error, Exception)
            assert hasattr(error, 'code')
            assert hasattr(error, 'to_dict')


class TestBoolLabelPredicate:
    """#876: filtering on bool label__ columns must not raise GFQLSchemaError."""

    def _make_graph(self):
        import graphistry
        nodes = pd.DataFrame({
            'id': [0, 1, 2, 3],
            'label__A': [True, True, False, False],
            'label__B': [False, False, True, True],
        })
        edges = pd.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 3]})
        return graphistry.nodes(nodes, 'id').edges(edges, 'src', 'dst')

    def test_bool_label_filter_does_not_raise(self):
        """Filtering n({'label__A': True}) must not raise GFQLSchemaError (#876)."""
        from graphistry.compute.ast import n, e_forward
        g = self._make_graph()
        # Must not raise "column is string but filter value is numeric"
        result = g.gfql([n({'label__A': True}), e_forward(), n({'label__B': True})])
        assert len(result._nodes) > 0

    def test_bool_label_filter_returns_correct_nodes(self):
        """Filtering on bool label__ columns returns only matching nodes."""
        from graphistry.compute.ast import n
        g = self._make_graph()
        result = g.gfql([n({'label__A': True})])
        assert set(result._nodes['id'].tolist()) == {0, 1}

    def test_bool_label_false_filter(self):
        """Filtering n({'label__A': False}) also works without error."""
        from graphistry.compute.ast import n
        g = self._make_graph()
        result = g.gfql([n({'label__A': False})])
        assert set(result._nodes['id'].tolist()) == {2, 3}


class TestChainFillnaNoFutureWarning:
    """#881: chain tag columns must not emit FutureWarning on fillna."""

    def _make_graph(self):
        import graphistry
        nodes = pd.DataFrame({'id': [0, 1, 2], 'val': [10, 20, 30]})
        edges = pd.DataFrame({'src': [0, 1], 'dst': [1, 2]})
        return graphistry.nodes(nodes, 'id').edges(edges, 'src', 'dst')

    def test_chain_hop_no_future_warning(self):
        """A basic gfql hop chain must not emit FutureWarning (#881)."""
        from graphistry.compute.ast import n, e_forward
        g = self._make_graph()
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            # Must not raise FutureWarning about fillna downcasting
            g.gfql([n(), e_forward(), n()])

    @_CUDF
    def test_chain_hop_no_future_warning_cudf(self):
        """cuDF: chain hop tag columns emit no FutureWarning on fillna (#881)."""
        import cudf
        import graphistry
        from graphistry.compute.ast import n, e_forward
        nodes = cudf.DataFrame({'id': [0, 1, 2], 'val': [10, 20, 30]})
        edges = cudf.DataFrame({'src': [0, 1], 'dst': [1, 2]})
        g = graphistry.nodes(nodes, 'id').edges(edges, 'src', 'dst')
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            g.gfql([n(), e_forward(), n()])


class TestBoolLabelPredicateCuDF:
    """#876 cuDF: filtering on bool label__ columns must not raise GFQLSchemaError."""

    def _make_graph(self):
        import cudf
        import graphistry
        nodes = cudf.DataFrame({
            'id': [0, 1, 2, 3],
            'label__A': [True, True, False, False],
            'label__B': [False, False, True, True],
        })
        edges = cudf.DataFrame({'src': [0, 1, 2], 'dst': [1, 2, 3]})
        return graphistry.nodes(nodes, 'id').edges(edges, 'src', 'dst')

    @_CUDF
    def test_bool_label_filter_does_not_raise_cudf(self):
        """cuDF: n({'label__A': True}) must not raise GFQLSchemaError (#876)."""
        from graphistry.compute.ast import n, e_forward
        g = self._make_graph()
        result = g.gfql([n({'label__A': True}), e_forward(), n({'label__B': True})])
        assert len(result._nodes) > 0

    @_CUDF
    def test_bool_label_filter_returns_correct_nodes_cudf(self):
        """cuDF: filtering on bool label__ columns returns only matching nodes."""
        from graphistry.compute.ast import n
        g = self._make_graph()
        result = g.gfql([n({'label__A': True})])
        assert set(result._nodes['id'].to_pandas().tolist()) == {0, 1}

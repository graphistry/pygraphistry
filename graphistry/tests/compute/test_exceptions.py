"""Unit tests for GFQL exceptions module."""

import pytest
from graphistry.compute.gfql.exceptions import (
    GFQLException, GFQLValidationError, GFQLSyntaxError,
    GFQLSchemaError, GFQLSemanticError, GFQLTypeError,
    GFQLColumnNotFoundError
)
from graphistry.tests.common import NoAuthTestCase


class TestGFQLException(NoAuthTestCase):
    """Test base GFQL exception."""
    
    def test_init_without_context(self):
        exc = GFQLException("Test error")
        self.assertEqual(str(exc), "Test error")
        self.assertEqual(exc.context, {})
    
    def test_init_with_context(self):
        context = {'field': 'test', 'value': 123}
        exc = GFQLException("Test error", context)
        self.assertEqual(exc.context, context)
    
    def test_str_with_context(self):
        exc = GFQLException("Test error", {'field': 'test', 'value': 123})
        exc_str = str(exc)
        self.assertIn("Test error", exc_str)
        self.assertIn("field=test", exc_str)
        self.assertIn("value=123", exc_str)


class TestGFQLValidationError(NoAuthTestCase):
    """Test validation error hierarchy."""
    
    def test_inheritance(self):
        exc = GFQLValidationError("Validation failed")
        self.assertIsInstance(exc, GFQLException)
        self.assertIsInstance(exc, ValueError)
    
    def test_subclasses(self):
        syntax_err = GFQLSyntaxError("Syntax error")
        schema_err = GFQLSchemaError("Schema error")
        semantic_err = GFQLSemanticError("Semantic error")
        
        self.assertIsInstance(syntax_err, GFQLValidationError)
        self.assertIsInstance(schema_err, GFQLValidationError)
        self.assertIsInstance(semantic_err, GFQLValidationError)


class TestGFQLTypeError(NoAuthTestCase):
    """Test type error exception."""
    
    def test_init(self):
        exc = GFQLTypeError(
            column='age',
            column_type='string',
            predicate='gt',
            expected_type='numeric'
        )
        
        self.assertIsInstance(exc, GFQLSchemaError)
        self.assertIn("'age'", str(exc))
        self.assertIn("'string'", str(exc))
        self.assertIn("'gt'", str(exc))
        self.assertIn("'numeric'", str(exc))
    
    def test_context(self):
        exc = GFQLTypeError('col', 'str', 'pred', 'num')
        self.assertEqual(exc.context['column'], 'col')
        self.assertEqual(exc.context['column_type'], 'str')
        self.assertEqual(exc.context['predicate'], 'pred')
        self.assertEqual(exc.context['expected_type'], 'num')


class TestGFQLColumnNotFoundError(NoAuthTestCase):
    """Test column not found exception."""
    
    def test_init(self):
        exc = GFQLColumnNotFoundError(
            column='missing_col',
            table='node',
            available_columns=['id', 'name', 'type']
        )
        
        self.assertIsInstance(exc, GFQLSchemaError)
        self.assertIn("'missing_col'", str(exc))
        self.assertIn("node", str(exc))
    
    def test_context(self):
        exc = GFQLColumnNotFoundError('col', 'table', ['a', 'b'])
        self.assertEqual(exc.context['column'], 'col')
        self.assertEqual(exc.context['table'], 'table')
        self.assertEqual(exc.context['available_columns'], ['a', 'b'])


class TestExceptionUsage(NoAuthTestCase):
    """Test exception usage patterns."""
    
    def test_catching_specific_errors(self):
        """Ensure exceptions can be caught at different levels."""
        
        def raise_syntax_error():
            raise GFQLSyntaxError("Invalid syntax")
        
        # Can catch as GFQLSyntaxError
        with self.assertRaises(GFQLSyntaxError):
            raise_syntax_error()
        
        # Can catch as GFQLValidationError
        with self.assertRaises(GFQLValidationError):
            raise_syntax_error()
        
        # Can catch as GFQLException
        with self.assertRaises(GFQLException):
            raise_syntax_error()
        
        # Can catch as ValueError (for backwards compatibility)
        with self.assertRaises(ValueError):
            raise_syntax_error()
    
    def test_exception_messages(self):
        """Test that exception messages are informative."""
        
        exc1 = GFQLTypeError('age', 'object', 'gt', 'int64')
        self.assertIn("Column 'age' has type 'object'", str(exc1))
        self.assertIn("predicate 'gt' expects 'int64'", str(exc1))
        
        exc2 = GFQLColumnNotFoundError('foo', 'edge', ['bar', 'baz'])
        self.assertIn("Column 'foo' not found", str(exc2))
        self.assertIn("edge data", str(exc2))


if __name__ == '__main__':
    pytest.main([__file__])

"""Unit tests for GFQL validation module."""

import pandas as pd
import pytest
from typing import List

from graphistry.compute.gfql import (
    validate_syntax, validate_schema, validate_query,
    extract_schema, extract_schema_from_dataframes,
    format_validation_errors, suggest_fixes,
    ValidationIssue, Schema
)
# Import internals directly from the module
from graphistry.compute.gfql.validate.validate import _format_error, _get_type_category
from graphistry.compute.ast import n, e_forward, e_reverse, e
from graphistry.compute.predicates.numeric import gt, lt, between
from graphistry.compute.predicates.str import contains, startswith
from graphistry.compute.chain import Chain
from graphistry.tests.common import NoAuthTestCase


class TestValidationIssue(NoAuthTestCase):
    """Test ValidationIssue class."""
    
    def test_init(self):
        issue = ValidationIssue('error', 'Test message')
        self.assertEqual(issue.level, 'error')
        self.assertEqual(issue.message, 'Test message')
        self.assertIsNone(issue.operation_index)
        self.assertIsNone(issue.field)
        self.assertIsNone(issue.suggestion)
        self.assertIsNone(issue.error_type)
    
    def test_init_with_all_fields(self):
        issue = ValidationIssue(
            'warning', 'Test warning',
            operation_index=2,
            field='test_field',
            suggestion='Fix it',
            error_type='TEST_ERROR'
        )
        self.assertEqual(issue.level, 'warning')
        self.assertEqual(issue.operation_index, 2)
        self.assertEqual(issue.field, 'test_field')
        self.assertEqual(issue.suggestion, 'Fix it')
        self.assertEqual(issue.error_type, 'TEST_ERROR')
    
    def test_repr(self):
        issue = ValidationIssue('error', 'Test message', suggestion='Fix it')
        repr_str = repr(issue)
        self.assertIn('ERROR', repr_str)
        self.assertIn('Test message', repr_str)
        self.assertIn('Fix it', repr_str)
    
    def test_to_dict(self):
        issue = ValidationIssue(
            'error', 'Test', 
            operation_index=1,
            field='field',
            suggestion='Suggestion',
            error_type='TYPE'
        )
        d = issue.to_dict()
        self.assertEqual(d['level'], 'error')
        self.assertEqual(d['message'], 'Test')
        self.assertEqual(d['operation_index'], 1)
        self.assertEqual(d['field'], 'field')
        self.assertEqual(d['suggestion'], 'Suggestion')
        self.assertEqual(d['error_type'], 'TYPE')


class TestSchema(NoAuthTestCase):
    """Test Schema class."""
    
    def test_init_empty(self):
        schema = Schema()
        self.assertEqual(schema.node_columns, {})
        self.assertEqual(schema.edge_columns, {})
    
    def test_init_with_data(self):
        node_cols = {'id': 'int64', 'name': 'object'}
        edge_cols = {'source': 'int64', 'target': 'int64'}
        schema = Schema(node_cols, edge_cols)
        self.assertEqual(schema.node_columns, node_cols)
        self.assertEqual(schema.edge_columns, edge_cols)
    
    def test_repr(self):
        schema = Schema({'id': 'int64', 'name': 'object'}, {'source': 'int64'})
        repr_str = repr(schema)
        self.assertIn('nodes', repr_str)
        self.assertIn('edges', repr_str)
        self.assertIn('id', repr_str)
        self.assertIn('name', repr_str)


class TestSyntaxValidation(NoAuthTestCase):
    """Test syntax validation functions."""
    
    def test_valid_chain(self):
        chain = [n({"type": "person"}), e_forward(), n()]
        issues = validate_syntax(chain)
        self.assertEqual(len(issues), 0)
    
    def test_invalid_chain_type(self):
        issues = validate_syntax("not a list")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].level, 'error')
        self.assertEqual(issues[0].error_type, 'INVALID_CHAIN_TYPE')
    
    def test_empty_chain(self):
        issues = validate_syntax([])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].level, 'error')
        self.assertEqual(issues[0].error_type, 'EMPTY_CHAIN')
    
    def test_invalid_operation(self):
        chain = [n(), "not an operation", e_forward()]
        issues = validate_syntax(chain)
        errors = [i for i in issues if i.level == 'error']
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].operation_index, 1)
        self.assertEqual(errors[0].error_type, 'INVALID_OPERATION')
    
    def test_invalid_filter_key(self):
        chain = [n({123: "value"})]  # Non-string key
        issues = validate_syntax(chain)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].error_type, 'INVALID_FILTER_KEY')
    
    def test_invalid_hops(self):
        chain = [n(), e_forward(hops=-1), n()]
        issues = validate_syntax(chain)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].error_type, 'INVALID_HOPS')
    
    def test_unbounded_hops_warning(self):
        chain = [n(), e_forward(to_fixed_point=True), n()]
        issues = validate_syntax(chain)
        warnings = [i for i in issues if i.level == 'warning']
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].error_type, 'UNBOUNDED_HOPS_WARNING')
    
    def test_orphaned_edge_warning(self):
        chain = [e_forward(), e_reverse()]
        issues = validate_syntax(chain)
        warnings = [i for i in issues if i.level == 'warning']
        self.assertEqual(len(warnings), 2)
        self.assertTrue(all(w.error_type == 'ORPHANED_EDGE' for w in warnings))
    
    def test_chain_object(self):
        chain_obj = Chain([n(), e_forward(), n()])
        issues = validate_syntax(chain_obj)
        self.assertEqual(len(issues), 0)


class TestSchemaValidation(NoAuthTestCase):
    """Test schema validation functions."""
    
    def setUp(self):
        super().setUp()
        self.schema = Schema(
            node_columns={'id': 'int64', 'name': 'object', 'age': 'int64'},
            edge_columns={'source': 'int64', 'target': 'int64', 'type': 'object', 'weight': 'float64'}
        )
    
    def test_valid_query(self):
        chain = [n({"name": "Alice"}), e_forward({"type": "knows"}), n()]
        issues = validate_schema(chain, self.schema)
        self.assertEqual(len(issues), 0)
    
    def test_column_not_found_node(self):
        chain = [n({"missing_col": "value"})]
        issues = validate_schema(chain, self.schema)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].error_type, 'COLUMN_NOT_FOUND')
        self.assertIn('node', issues[0].message)
    
    def test_column_not_found_edge(self):
        chain = [n(), e_forward({"missing_edge_col": "value"}), n()]
        issues = validate_schema(chain, self.schema)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].error_type, 'COLUMN_NOT_FOUND')
        self.assertIn('edge', issues[0].message)
    
    def test_type_mismatch_numeric(self):
        chain = [n({"age": contains("25")})]  # String predicate on numeric
        issues = validate_schema(chain, self.schema)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].error_type, 'TYPE_MISMATCH')
        self.assertIn('string', issues[0].message)
    
    def test_type_mismatch_string(self):
        chain = [n({"name": gt(10)})]  # Numeric predicate on string
        issues = validate_schema(chain, self.schema)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].error_type, 'TYPE_MISMATCH')
        self.assertIn('numeric', issues[0].message)
    
    def test_edge_node_filters(self):
        chain = [
            n(),
            e_forward(
                source_node_match={"missing": "value"},
                destination_node_match={"id": 1}
            ),
            n()
        ]
        issues = validate_schema(chain, self.schema)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].field, 'source_node_match.missing')


class TestValidateQuery(NoAuthTestCase):
    """Test combined validation function."""
    
    def test_without_data(self):
        chain = [n(), e_forward(hops=-1), n()]
        issues = validate_query(chain)
        # Should only do syntax validation
        self.assertTrue(any(i.error_type == 'INVALID_HOPS' for i in issues))
    
    def test_with_data(self):
        nodes_df = pd.DataFrame({'id': [1, 2, 3], 'type': ['A', 'B', 'C']})
        edges_df = pd.DataFrame({'source': [1, 2], 'target': [2, 3]})
        
        chain = [n({"missing": "value"})]
        issues = validate_query(chain, nodes_df, edges_df)
        # Should include schema validation
        self.assertTrue(any(i.error_type == 'COLUMN_NOT_FOUND' for i in issues))


class TestSchemaExtraction(NoAuthTestCase):
    """Test schema extraction functions."""
    
    def test_extract_from_dataframes(self):
        nodes_df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        edges_df = pd.DataFrame({'source': [1], 'target': [2], 'weight': [1.0]})
        
        schema = extract_schema_from_dataframes(nodes_df, edges_df)
        
        self.assertIn('id', schema.node_columns)
        self.assertIn('name', schema.node_columns)
        self.assertIn('source', schema.edge_columns)
        self.assertIn('weight', schema.edge_columns)
        self.assertEqual(schema.edge_columns['weight'], 'float64')
    
    def test_extract_none_dataframes(self):
        schema = extract_schema_from_dataframes(None, None)
        self.assertEqual(schema.node_columns, {})
        self.assertEqual(schema.edge_columns, {})
    
    def test_extract_partial_dataframes(self):
        nodes_df = pd.DataFrame({'id': [1, 2]})
        schema = extract_schema_from_dataframes(nodes_df, None)
        self.assertIn('id', schema.node_columns)
        self.assertEqual(schema.edge_columns, {})


class TestErrorFormatting(NoAuthTestCase):
    """Test error formatting functions."""
    
    def test_format_no_issues(self):
        result = format_validation_errors([])
        self.assertEqual(result, "No validation issues found.")
    
    def test_format_errors_and_warnings(self):
        issues = [
            ValidationIssue('error', 'Error 1', operation_index=0, suggestion='Fix 1'),
            ValidationIssue('warning', 'Warning 1', operation_index=1, suggestion='Fix 2')
        ]
        result = format_validation_errors(issues)
        self.assertIn('ERRORS (1)', result)
        self.assertIn('WARNINGS (1)', result)
        self.assertIn('Error 1', result)
        self.assertIn('Warning 1', result)
        self.assertIn('Fix 1', result)
    
    def test_suggest_fixes(self):
        issues = [
            ValidationIssue('error', 'Column not found', error_type='COLUMN_NOT_FOUND', field='col1'),
            ValidationIssue('error', 'Column not found', error_type='COLUMN_NOT_FOUND', field='col2'),
            ValidationIssue('error', 'Type mismatch', error_type='TYPE_MISMATCH')
        ]
        suggestions = suggest_fixes([], issues)
        self.assertTrue(any('Missing columns' in s for s in suggestions))
        self.assertTrue(any('Type mismatches' in s for s in suggestions))


class TestHelperFunctions(NoAuthTestCase):
    """Test internal helper functions."""
    
    def test_format_error(self):
        message, suggestion = _format_error('EMPTY_CHAIN')
        self.assertEqual(message, 'Chain is empty')
        self.assertIn('Add at least one operation', suggestion)
    
    def test_format_error_with_kwargs(self):
        message, suggestion = _format_error('COLUMN_NOT_FOUND', 
                                          column='test', 
                                          table='node',
                                          available='id, name')
        self.assertIn('test', message)
        self.assertIn('node', message)
        self.assertIn('id, name', suggestion)
    
    def test_get_type_category(self):
        self.assertEqual(_get_type_category('int64'), 'numeric')
        self.assertEqual(_get_type_category('float32'), 'numeric')
        self.assertEqual(_get_type_category('object'), 'string')
        self.assertEqual(_get_type_category('str'), 'string')
        self.assertEqual(_get_type_category('datetime64'), 'temporal')
        self.assertEqual(_get_type_category('bool'), 'boolean')
        self.assertEqual(_get_type_category('custom_type'), 'unknown')


if __name__ == '__main__':
    pytest.main([__file__])

"""Tests for improved hypergraph opts parameter validation."""

import pytest
from graphistry.compute.gfql.call_safelist import (
    validate_hypergraph_opts,
    validate_call_params
)
from graphistry.compute.exceptions import GFQLTypeError


class TestHypergraphOptsValidation:
    """Test the improved validation for hypergraph opts parameter."""

    def test_opts_empty_dict(self):
        """Test that empty dict is valid for opts."""
        assert validate_hypergraph_opts({}) is True

    def test_opts_with_string_params(self):
        """Test opts with valid string parameters."""
        opts = {
            'TITLE': 'myTitle',
            'DELIM': '|',
            'NODEID': 'node_id',
            'ATTRIBID': 'attr_id',
            'EVENTID': 'event_id',
            'EVENTTYPE': 'evt',
            'SOURCE': 'from',
            'DESTINATION': 'to',
            'CATEGORY': 'cat',
            'NODETYPE': 'ntype',
            'EDGETYPE': 'etype',
            'NULLVAL': 'NA'
        }
        assert validate_hypergraph_opts(opts) is True

    def test_opts_with_skip_list(self):
        """Test opts with SKIP parameter."""
        opts = {
            'SKIP': ['col1', 'col2', 'col3']
        }
        assert validate_hypergraph_opts(opts) is True

    def test_opts_with_categories(self):
        """Test opts with CATEGORIES mapping."""
        opts = {
            'CATEGORIES': {
                'n': ['aa', 'bb', 'cc'],
                'type1': ['x', 'y', 'z']
            }
        }
        assert validate_hypergraph_opts(opts) is True

    def test_opts_with_edges(self):
        """Test opts with EDGES mapping."""
        opts = {
            'EDGES': {
                'aa': ['cc', 'bb'],
                'cc': ['cc'],
                'user': ['product', 'session']
            }
        }
        assert validate_hypergraph_opts(opts) is True

    def test_opts_with_all_params(self):
        """Test opts with all valid parameters combined."""
        opts = {
            'TITLE': 'myTitle',
            'DELIM': '::',
            'NODEID': 'id',
            'SKIP': ['timestamp', 'metadata'],
            'CATEGORIES': {'n': ['type1', 'type2']},
            'EDGES': {'user': ['product'], 'product': ['user']}
        }
        assert validate_hypergraph_opts(opts) is True

    def test_opts_invalid_string_param_type(self):
        """Test opts with invalid type for string parameter."""
        opts = {
            'TITLE': 123  # Should be string
        }
        assert validate_hypergraph_opts(opts) is False

    def test_opts_invalid_skip_not_list(self):
        """Test opts with invalid SKIP (not a list)."""
        opts = {
            'SKIP': 'col1'  # Should be list
        }
        assert validate_hypergraph_opts(opts) is False

    def test_opts_invalid_skip_not_strings(self):
        """Test opts with invalid SKIP (list contains non-strings)."""
        opts = {
            'SKIP': ['col1', 123, 'col3']  # All items should be strings
        }
        assert validate_hypergraph_opts(opts) is False

    def test_opts_invalid_categories_not_dict(self):
        """Test opts with invalid CATEGORIES (not a dict)."""
        opts = {
            'CATEGORIES': ['type1', 'type2']  # Should be dict
        }
        assert validate_hypergraph_opts(opts) is False

    def test_opts_invalid_categories_value_not_list(self):
        """Test opts with invalid CATEGORIES value (not a list)."""
        opts = {
            'CATEGORIES': {
                'n': 'aa'  # Should be list of strings
            }
        }
        assert validate_hypergraph_opts(opts) is False

    def test_opts_invalid_edges_structure(self):
        """Test opts with invalid EDGES structure."""
        opts = {
            'EDGES': {
                'aa': 'cc'  # Should be list, not string
            }
        }
        assert validate_hypergraph_opts(opts) is False

    def test_opts_not_dict(self):
        """Test that non-dict opts is rejected."""
        assert validate_hypergraph_opts("not a dict") is False
        assert validate_hypergraph_opts(None) is False
        assert validate_hypergraph_opts([]) is False

    def test_opts_unknown_keys_allowed(self):
        """Test that unknown keys are allowed for forward compatibility."""
        opts = {
            'TITLE': 'test',
            'CUSTOM_PARAM': 'value',  # Unknown but should be allowed
            'ANOTHER': 123
        }
        assert validate_hypergraph_opts(opts) is True

    def test_opts_invalid_key_type(self):
        """Test opts with non-string keys."""
        opts = {
            123: 'value'  # Keys must be strings
        }
        assert validate_hypergraph_opts(opts) is False

    def test_validate_call_params_with_valid_opts(self):
        """Test full validation through validate_call_params."""
        params = {
            'entity_types': ['user', 'product'],
            'opts': {
                'CATEGORIES': {'n': ['type1', 'type2']},
                'EDGES': {'user': ['product']}
            },
            'drop_na': True,
            'direct': False
        }

        # Should not raise
        validated = validate_call_params('hypergraph', params)
        assert validated == params

    def test_validate_call_params_with_invalid_opts(self):
        """Test that invalid opts raises error through validate_call_params."""
        params = {
            'entity_types': ['user', 'product'],
            'opts': {
                'CATEGORIES': 'not_a_dict'  # Invalid structure
            }
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('hypergraph', params)

        assert "Invalid type for parameter 'opts'" in str(exc_info.value)

    def test_real_world_opts_example(self):
        """Test real-world opts examples from existing tests."""
        # From test_hypergraph.py
        opts1 = {"CATEGORIES": {"n": ["aa", "bb", "cc"]}}
        assert validate_hypergraph_opts(opts1) is True

        opts2 = {"EDGES": {"aa": ["cc"], "cc": ["cc"]}}
        assert validate_hypergraph_opts(opts2) is True

        opts3 = {"EDGES": {"aa": ["cc", "bb", "aa"], "cc": ["cc"]}}
        assert validate_hypergraph_opts(opts3) is True

        opts4 = {"EDGES": {"id": ["a1"], "a1": ["🙈"]}}  # Unicode should work
        assert validate_hypergraph_opts(opts4) is True

    def test_opts_complex_nested_structure(self):
        """Test opts with complex nested structure."""
        opts = {
            'TITLE': 'Complex Graph',
            'DELIM': '||',
            'SKIP': ['meta1', 'meta2', 'timestamp'],
            'CATEGORIES': {
                'department': ['sales', 'engineering', 'marketing'],
                'level': ['junior', 'senior', 'lead'],
                'region': ['NA', 'EU', 'APAC']
            },
            'EDGES': {
                'user': ['product', 'session', 'transaction'],
                'product': ['category', 'supplier'],
                'session': ['user', 'device']
            },
            'CUSTOM_FIELD': 'allowed'
        }
        assert validate_hypergraph_opts(opts) is True

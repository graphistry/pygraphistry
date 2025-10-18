"""Test static validation of GFQL call() operations across multiple methods.

This test ensures that call parameter validation happens BEFORE execution,
catching issues client-side before sending to server.

Tests cover diverse operation types:
- Graph transformations: hypergraph, umap
- Graph algorithms: compute_igraph, get_degrees
- Visual encodings: encode_point_color
- Graph traversals: hop
- Metadata: name, description
"""
import pandas as pd
import pytest
from graphistry.compute.ast import call, ASTCall
from graphistry.compute.gfql.call_safelist import validate_call_params
from graphistry.compute.exceptions import GFQLTypeError


class TestHypergraphValidation:
    """Test validation of hypergraph() call parameters."""

    def test_valid_hypergraph_call_basic(self):
        """Test that basic hypergraph call passes validation."""
        params = {
            'entity_types': ['col1', 'col2', 'col3'],
            'drop_na': True,
            'direct': True,
            'engine': 'pandas'
        }

        # Should not raise
        validated = validate_call_params('hypergraph', params)
        assert validated == params

    def test_invalid_hypergraph_engine_value(self):
        """Test that invalid engine value is rejected."""
        params = {
            'entity_types': ['col1', 'col2'],
            'engine': 'invalid_engine'  # Should be pandas, cudf, dask, or auto
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('hypergraph', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'engine' in exc_info.value.message

    def test_invalid_hypergraph_return_as(self):
        """Test that invalid return_as value is rejected."""
        params = {
            'entity_types': ['col1', 'col2'],
            'return_as': 'invalid_option'
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('hypergraph', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'return_as' in exc_info.value.message


class TestUmapValidation:
    """Test validation of umap() call parameters."""

    def test_valid_umap_call_basic(self):
        """Test that basic umap call passes validation."""
        params = {
            'X': ['col1', 'col2'],
            'n_neighbors': 15,
            'min_dist': 0.1,
            'engine': 'auto'
        }

        validated = validate_call_params('umap', params)
        assert validated == params

    def test_invalid_umap_kind(self):
        """Test that invalid kind value is rejected."""
        params = {
            'X': ['col1', 'col2'],
            'kind': 'invalid'  # Should be 'nodes' or 'edges'
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('umap', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'kind' in exc_info.value.message

    def test_invalid_umap_n_neighbors_type(self):
        """Test that invalid n_neighbors type is rejected."""
        params = {
            'X': ['col1', 'col2'],
            'n_neighbors': 'fifteen'  # Should be int
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('umap', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'n_neighbors' in exc_info.value.message

    def test_invalid_umap_inplace(self):
        """Test that inplace=True is rejected (only False allowed)."""
        params = {
            'X': ['col1', 'col2'],
            'inplace': True  # Only False is allowed
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('umap', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'inplace' in exc_info.value.message


class TestGetDegreesValidation:
    """Test validation of get_degrees() call parameters."""

    def test_valid_get_degrees_call(self):
        """Test that valid get_degrees call passes validation."""
        params = {
            'col': 'degree',
            'col_in': 'degree_in',
            'col_out': 'degree_out'
        }

        validated = validate_call_params('get_degrees', params)
        assert validated == params

    def test_invalid_get_degrees_col_type(self):
        """Test that invalid col type is rejected."""
        params = {
            'col': 123  # Should be string
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('get_degrees', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'col' in exc_info.value.message

    def test_get_degrees_unknown_param(self):
        """Test that unknown parameters are rejected."""
        params = {
            'col': 'degree',
            'unknown_param': 'value'
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('get_degrees', params)

        assert 'Unknown parameters' in exc_info.value.message


class TestHopValidation:
    """Test validation of hop() call parameters."""

    def test_valid_hop_call(self):
        """Test that valid hop call passes validation."""
        params = {
            'hops': 2,
            'direction': 'forward',
            'edge_match': {'type': 'transaction'}
        }

        validated = validate_call_params('hop', params)
        assert validated == params

    def test_invalid_hop_direction(self):
        """Test that invalid direction is rejected."""
        params = {
            'hops': 2,
            'direction': 'sideways'  # Should be forward/reverse/undirected
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('hop', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'direction' in exc_info.value.message

    def test_invalid_hop_hops_type(self):
        """Test that invalid hops type is rejected."""
        params = {
            'hops': 'two'  # Should be int
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('hop', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'hops' in exc_info.value.message


class TestEncodePointColorValidation:
    """Test validation of encode_point_color() call parameters."""

    def test_valid_encode_point_color_call(self):
        """Test that valid encode_point_color call passes validation."""
        params = {
            'column': 'type',
            'as_categorical': True,
            'palette': ['#FF0000', '#00FF00', '#0000FF']
        }

        validated = validate_call_params('encode_point_color', params)
        assert validated == params

    def test_encode_point_color_missing_required(self):
        """Test that missing required column parameter is rejected."""
        params = {
            'palette': ['#FF0000', '#00FF00']
            # Missing required 'column' parameter
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert 'Missing required parameters' in exc_info.value.message
        # Check that error mentions encode_point_color
        assert 'encode_point_color' in exc_info.value.message

    def test_invalid_encode_point_color_column_type(self):
        """Test that invalid column type is rejected."""
        params = {
            'column': 123  # Should be string
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'column' in exc_info.value.message


class TestComputeIgraphValidation:
    """Test validation of compute_igraph() call parameters."""

    def test_valid_compute_igraph_call(self):
        """Test that valid compute_igraph call passes validation."""
        params = {
            'alg': 'pagerank',
            'out_col': 'pagerank_score',
            'directed': True
        }

        validated = validate_call_params('compute_igraph', params)
        assert validated == params

    def test_compute_igraph_missing_required_alg(self):
        """Test that missing required alg parameter is rejected."""
        params = {
            'out_col': 'score'
            # Missing required 'alg' parameter
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('compute_igraph', params)

        assert 'Missing required parameters' in exc_info.value.message
        # Check that error mentions compute_igraph
        assert 'compute_igraph' in exc_info.value.message

    def test_invalid_compute_igraph_directed_type(self):
        """Test that invalid directed type is rejected."""
        params = {
            'alg': 'pagerank',
            'directed': 'yes'  # Should be bool
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('compute_igraph', params)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'directed' in exc_info.value.message


class TestMetadataValidation:
    """Test validation of metadata call parameters (name, description)."""

    def test_valid_name_call(self):
        """Test that valid name call passes validation."""
        params = {
            'name': 'My Graph Visualization'
        }

        validated = validate_call_params('name', params)
        assert validated == params

    def test_name_missing_required(self):
        """Test that missing required name parameter is rejected."""
        params = {}

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('name', params)

        assert 'Missing required parameters' in exc_info.value.message

    def test_valid_description_call(self):
        """Test that valid description call passes validation."""
        params = {
            'description': 'A graph showing user interactions'
        }

        validated = validate_call_params('description', params)
        assert validated == params


class TestUnknownFunctionValidation:
    """Test validation of unknown functions."""

    def test_unknown_function_rejected(self):
        """Test that unknown functions are rejected."""
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('dangerous_method', {})

        assert "not in the safelist" in exc_info.value.message
        assert 'dangerous_method' in exc_info.value.message

    def test_unknown_function_shows_available(self):
        """Test that error message contains function name."""
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('unknown', {})

        assert 'not in the safelist' in exc_info.value.message
        assert 'unknown' in exc_info.value.message


class TestASTCallDeserialization:
    """Test that ASTCall validates during from_json (remote execution path)."""

    def test_astcall_from_json_validates_params(self):
        """Test that ASTCall.from_json() validates parameters."""
        # Valid params should work
        call_json = {
            'type': 'Call',
            'function': 'get_degrees',
            'params': {
                'col': 'degree'
            }
        }
        call_obj = ASTCall.from_json(call_json, validate=True)
        assert call_obj.function == 'get_degrees'
        assert call_obj.params == {'col': 'degree'}

    def test_astcall_from_json_rejects_invalid_params(self):
        """Test that ASTCall.from_json() rejects invalid parameters."""
        call_json = {
            'type': 'Call',
            'function': 'hypergraph',
            'params': {
                'engine': 'invalid_engine'
            }
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            ASTCall.from_json(call_json, validate=True)

        assert 'Invalid type for parameter' in exc_info.value.message
        assert 'engine' in exc_info.value.message

    def test_astcall_from_json_rejects_unknown_params(self):
        """Test that ASTCall.from_json() rejects unknown parameters."""
        call_json = {
            'type': 'Call',
            'function': 'umap',
            'params': {
                'X': ['col1', 'col2'],
                'bad_param': 'value'
            }
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            ASTCall.from_json(call_json, validate=True)

        assert 'Unknown parameters' in exc_info.value.message


class TestCallObjectCreation:
    """Test that call() helper validates parameters."""

    def test_call_helper_validates_params(self):
        """Test that call() creates valid ASTCall that validates on construction."""
        # Valid call
        call_obj = call('get_degrees', {'col': 'degree'})
        assert call_obj.function == 'get_degrees'
        assert call_obj.params == {'col': 'degree'}

        # Invalid params should raise during validation
        call_obj = call('umap', {'engine': 'invalid'})
        with pytest.raises(GFQLTypeError):
            call_obj.validate()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])

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
from graphistry.compute.gfql.call.validation import validate_call_params
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
            'degree_in': 'degree_in',
            'degree_out': 'degree_out'
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
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert 'Missing required parameters' in exc_info.value.message
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

    def test_encode_point_color_rejects_non_string_palette_entry(self):
        """Test that palette entries are validated before execution."""
        params = {
            'column': 'type',
            'palette': ['#FF0000', 123]
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'palette' in 'encode_point_color': "
            "palette[1] must be a string"
        )

    def test_encode_point_color_rejects_non_list_palette(self):
        """Test that palette must use the direct encode list shape."""
        params = {
            'column': 'type',
            'palette': '#FF0000'
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'palette' in 'encode_point_color': "
            "palette must be a list of strings"
        )

    def test_encode_point_color_rejects_non_string_mapping_value(self):
        """Test that categorical mapping values are validated before execution."""
        params = {
            'column': 'type',
            'categorical_mapping': {'admin': 1}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'categorical_mapping' in 'encode_point_color': "
            "categorical_mapping['admin'] must be a string"
        )

    def test_encode_point_color_rejects_non_dict_mapping(self):
        """Test that categorical mappings must be dictionaries."""
        params = {
            'column': 'type',
            'categorical_mapping': [('admin', '#FF0000')]
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'categorical_mapping' in 'encode_point_color': "
            "categorical_mapping must be a dictionary"
        )

    def test_encode_point_color_rejects_non_string_mapping_key(self):
        """Test that categorical mapping keys are string categories."""
        params = {
            'column': 'type',
            'categorical_mapping': {1: '#FF0000'}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'categorical_mapping' in 'encode_point_color': "
            "categorical_mapping keys must be strings"
        )

    def test_encode_point_color_rejects_non_string_default_mapping(self):
        """Test that default color mappings are strings or None."""
        params = {
            'column': 'type',
            'default_mapping': 1
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_color', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'default_mapping' in 'encode_point_color': "
            "default_mapping must be a string or None"
        )


class TestEncodeEdgeIconValidation:
    """Test validation of encode_edge_icon() call parameters."""

    def test_valid_encode_edge_icon_call(self):
        """Test that valid encode_edge_icon call passes validation."""
        params = {
            'column': 'edge_kind',
            'categorical_mapping': {'email': 'envelope'},
            'default_mapping': 'question',
            'as_text': False
        }

        validated = validate_call_params('encode_edge_icon', params)
        assert validated == params

    def test_encode_edge_icon_missing_required(self):
        """Test that missing required column parameter is rejected."""
        params = {
            'categorical_mapping': {'email': 'envelope'}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_edge_icon', params)

        assert exc_info.value.message == "Missing required parameters for 'encode_edge_icon'"

    def test_encode_edge_icon_rejects_non_string_mapping_value(self):
        """Test that categorical mapping values are validated before execution."""
        params = {
            'column': 'edge_kind',
            'categorical_mapping': {'email': 1}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_edge_icon', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'categorical_mapping' in 'encode_edge_icon': "
            "categorical_mapping['email'] must be a string"
        )


class TestEncodeAxisValidation:
    """Test validation of encode_axis() call parameters."""

    def test_valid_encode_axis_call(self):
        """Test that valid encode_axis call passes validation."""
        params = {
            'rows': [
                {'r': 10, 'external': True, 'label': 'outer'},
                {'y': 2, 'internal': True}
            ]
        }

        validated = validate_call_params('encode_axis', params)
        assert validated == params

    def test_valid_encode_axis_radial_row(self):
        """Test documented radial axis row validation."""
        params = {
            'rows': [{'r': 10, 'external': True, 'label': 'outer'}]
        }

        validated = validate_call_params('encode_axis', params)
        assert validated == params

    def test_valid_encode_axis_linear_row(self):
        """Test documented linear axis row validation."""
        params = {
            'rows': [{'y': 40, 'width': 20, 'bounds': {'min': 40, 'max': 400}}]
        }

        validated = validate_call_params('encode_axis', params)
        assert validated == params

    def test_encode_axis_allows_extension_subtype_rows(self):
        """Test extension subtypes remain payload-compatible."""
        params = {
            'rows': [{'kind': 'polar-v2', 'radius': 10, 'label': 7}]
        }

        validated = validate_call_params('encode_axis', params)
        assert validated == params

    def test_encode_axis_allows_default_rows(self):
        """Test that encode_axis mirrors direct Plottable default rows behavior."""
        params = {}

        validated = validate_call_params('encode_axis', params)
        assert validated == params

    def test_encode_axis_rejects_non_dict_row(self):
        """Test that rows entries are validated before execution."""
        params = {
            'rows': [{'r': 10}, 'bad']
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_axis', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'rows' in 'encode_axis': rows[1] must be a dictionary"
        )

    def test_encode_axis_rejects_radial_wrong_key(self):
        """Test row-indexed diagnostics for documented radial row keys."""
        params = {
            'rows': [{'r': 10, 'radius': 10}]
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_axis', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'rows' in 'encode_axis': "
            "rows[0] radial axis row has unexpected key 'radius'; expected keys: "
            "axisKind, axis_subtype, bounds, external, internal, kind, label, r, space, width, x, y"
        )

    def test_encode_axis_rejects_missing_position_key(self):
        """Test row-indexed diagnostics for axis rows missing required position keys."""
        params = {
            'rows': [{'label': 'missing position'}]
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_axis', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'rows' in 'encode_axis': "
            "rows[0] axis row is missing required key one of r, x, y; expected keys: "
            "axisKind, axis_subtype, bounds, external, internal, kind, label, r, space, width, x, y"
        )

    def test_encode_axis_rejects_non_list_rows(self):
        """Test that rows use the direct encode_axis list shape."""
        params = {
            'rows': {'r': 10}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_axis', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'rows' in 'encode_axis': rows must be a list of dictionaries"
        )


class TestEncodePointSizeValidation:
    """Test validation of encode_point_size() call parameters."""

    def test_valid_encode_point_size_mapping_call(self):
        """Test that numeric size mappings and defaults pass validation."""
        params = {
            'column': 'kind',
            'categorical_mapping': {'admin': 10},
            'default_mapping': 1.5
        }

        validated = validate_call_params('encode_point_size', params)
        assert validated == params

    def test_encode_point_size_rejects_non_dict_mapping(self):
        """Test that numeric categorical mappings must be dictionaries."""
        params = {
            'column': 'kind',
            'categorical_mapping': [('admin', 10)]
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_size', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'categorical_mapping' in 'encode_point_size': "
            "categorical_mapping must be a dictionary"
        )

    def test_encode_point_size_rejects_non_string_mapping_key(self):
        """Test that numeric categorical mapping keys are string categories."""
        params = {
            'column': 'kind',
            'categorical_mapping': {1: 10}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_size', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'categorical_mapping' in 'encode_point_size': "
            "categorical_mapping keys must be strings"
        )

    def test_encode_point_size_rejects_boolean_mapping_value(self):
        """Test that bool is not accepted as a numeric size."""
        params = {
            'column': 'kind',
            'categorical_mapping': {'admin': True}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_size', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'categorical_mapping' in 'encode_point_size': "
            "categorical_mapping['admin'] must be a number"
        )

    def test_encode_point_size_rejects_boolean_default_mapping(self):
        """Test that bool is not accepted as a default numeric size."""
        params = {
            'column': 'kind',
            'default_mapping': True
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('encode_point_size', params)

        assert exc_info.value.message == (
            "Invalid value for parameter 'default_mapping' in 'encode_point_size': "
            "default_mapping must be a number"
        )


class TestEncodeParityValidation:
    """Test validation of encode parity call parameters."""

    @pytest.mark.parametrize("function,mapping_value,default_value", [
        ('encode_edge_size', 10, 1),
        ('encode_edge_weight', 0.8, 0.1),
        ('encode_point_opacity', 0.8, 0.1),
        ('encode_edge_opacity', 0.8, 0.1),
    ])
    def test_valid_numeric_encode_parity_call(self, function, mapping_value, default_value):
        params = {
            'column': 'kind',
            'categorical_mapping': {'admin': mapping_value},
            'default_mapping': default_value,
        }

        validated = validate_call_params(function, params)
        assert validated == params

    @pytest.mark.parametrize("function", [
        'encode_edge_size',
        'encode_edge_weight',
        'encode_point_opacity',
        'encode_edge_opacity',
    ])
    def test_numeric_encode_parity_rejects_non_numeric_mapping_value(self, function):
        params = {
            'column': 'kind',
            'categorical_mapping': {'admin': 'large'}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params(function, params)

        assert exc_info.value.message == (
            f"Invalid value for parameter 'categorical_mapping' in '{function}': "
            "categorical_mapping['admin'] must be a number"
        )

    @pytest.mark.parametrize("function,mapping_value,default_value", [
        ('encode_point_label', 'Admin', 'Other'),
        ('encode_edge_label', 'Admin', 'Other'),
        ('encode_point_title', 'Admin', 'Other'),
        ('encode_edge_title', 'Admin', 'Other'),
    ])
    def test_valid_text_encode_parity_call(self, function, mapping_value, default_value):
        params = {
            'column': 'kind',
            'categorical_mapping': {'admin': mapping_value},
            'default_mapping': default_value,
        }

        validated = validate_call_params(function, params)
        assert validated == params

    @pytest.mark.parametrize("function", [
        'encode_point_label',
        'encode_edge_label',
        'encode_point_title',
        'encode_edge_title',
    ])
    def test_text_encode_parity_rejects_non_string_mapping_value(self, function):
        params = {
            'column': 'kind',
            'categorical_mapping': {'admin': 1}
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params(function, params)

        assert exc_info.value.message == (
            f"Invalid value for parameter 'categorical_mapping' in '{function}': "
            "categorical_mapping['admin'] must be a string"
        )


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
        }

        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('compute_igraph', params)

        assert 'Missing required parameters' in exc_info.value.message
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
        call_obj = call('get_degrees', {'col': 'degree'})
        assert call_obj.function == 'get_degrees'
        assert call_obj.params == {'col': 'degree'}

        call_obj = call('umap', {'engine': 'invalid'})
        with pytest.raises(GFQLTypeError):
            call_obj.validate()


class TestLayoutCallValidation:
    """Validation for GFQL layout calls backed by Python layout methods."""

    def test_circle_layout_params_valid(self):
        params = validate_call_params('circle_layout', {
            'bounding_box': [0, 0, 100, 100],
            'partition_by': ['type', 'cluster'],
            'sort_by': 'degree',
            'ascending': [True, False],
            'na_position': 'first',
            'engine': 'pandas',
        })
        assert params['bounding_box'] == [0, 0, 100, 100]

    def test_circle_layout_rejects_non_json_dataframe_boundary(self):
        with pytest.raises(GFQLTypeError):
            validate_call_params('circle_layout', {
                'bounding_box': {'x': 0, 'y': 0, 'w': 100, 'h': 100},
            })

    def test_tree_layout_params_valid(self):
        params = validate_call_params('tree_layout', {
            'level_col': 'level',
            'level_sort_values_by': ['type', 'rank'],
            'level_sort_values_by_ascending': False,
            'width': 100,
            'height': 50,
            'rotate': 90,
            'allow_cycles': True,
            'root': 'a',
        })
        assert params['root'] == 'a'

    def test_mercator_layout_params_valid(self):
        params = validate_call_params('mercator_layout', {
            'scale_for_graphistry': False,
        })
        assert params['scale_for_graphistry'] is False

    def test_modularity_weighted_layout_params_valid(self):
        params = validate_call_params('modularity_weighted_layout', {
            'community_col': 'community',
            'community_alg': None,
            'community_params': None,
            'same_community_weight': 2.0,
            'cross_community_weight': 0.3,
            'edge_influence': 2.0,
            'engine': 'pandas',
        })
        assert params['community_col'] == 'community'


class TestRingAxisValidation:
    """Deep axis payload validation for ring layout calls."""

    def test_ring_categorical_axis_label_map_valid(self):
        params = validate_call_params('ring_categorical_layout', {
            'ring_col': 'segment',
            'axis': {'a': 'A', 'b': 'B'},
        })
        assert params['axis'] == {'a': 'A', 'b': 'B'}

    def test_ring_categorical_axis_rows_valid(self):
        params = validate_call_params('ring_categorical_layout', {
            'ring_col': 'segment',
            'axis': [{'r': 200, 'label': 'outer', 'external': True}],
        })
        assert isinstance(params['axis'], list)

    def test_ring_categorical_axis_rejects_invalid_rows(self):
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('ring_categorical_layout', {
                'ring_col': 'segment',
                'axis': [{'label': 'missing_pos'}],
            })
        assert exc_info.value.message == (
            "Invalid value for parameter 'axis' in 'ring_categorical_layout': "
            "axis[0] is missing one of required position keys: r, x, y; expected keys: "
            "bounds, external, internal, label, r, space, width, x, y"
        )

    def test_ring_continuous_axis_accepts_string_labels(self):
        params = validate_call_params('ring_continuous_layout', {
            'ring_col': 'score',
            'axis': ['low', 'mid', 'high'],
        })
        assert params['axis'] == ['low', 'mid', 'high']

    def test_ring_continuous_axis_accepts_numeric_map(self):
        params = validate_call_params('ring_continuous_layout', {
            'ring_col': 'score',
            'axis': {100.0: 'inner', 200.0: 'outer'},
        })
        assert isinstance(params['axis'], dict)

    def test_ring_continuous_axis_rejects_bad_bounds(self):
        with pytest.raises(GFQLTypeError) as exc_info:
            validate_call_params('ring_continuous_layout', {
                'ring_col': 'score',
                'axis': [{'y': 40, 'bounds': {'min': '40', 'max': 100}}],
            })
        assert exc_info.value.message == (
            "Invalid value for parameter 'axis' in 'ring_continuous_layout': "
            "axis[0].bounds['min'] must be a number"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

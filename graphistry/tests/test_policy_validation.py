"""Tests for policy modification schema validation."""

import pytest
from typing import Optional

from graphistry.compute.gfql.policy import (
    PolicyModification,
    validate_modification
)


class TestSchemaValidation:
    """Test modification schema validation."""

    def test_valid_engine_modification(self):
        """Test that valid engine values pass validation."""
        # Valid engines
        for engine in ['pandas', 'cudf', 'dask', 'dask_cudf', 'auto']:
            mod = {'engine': engine}
            result = validate_modification(mod, 'preload')
            assert result['engine'] == engine

    def test_invalid_engine_rejected(self):
        """Test that invalid engine values are rejected."""
        with pytest.raises(ValueError, match="Invalid engine"):
            validate_modification({'engine': 'quantum'}, 'preload')

        with pytest.raises(ValueError, match="Invalid engine"):
            validate_modification({'engine': 'invalid'}, 'call')

    def test_valid_params_modification(self):
        """Test that params as dict passes validation."""
        mod = {'params': {'n_components': 2, 'metric': 'euclidean'}}
        result = validate_modification(mod, 'call')
        assert result['params'] == {'n_components': 2, 'metric': 'euclidean'}

    def test_invalid_params_type_rejected(self):
        """Test that non-dict params are rejected."""
        with pytest.raises(ValueError, match="'params' must be a dict"):
            validate_modification({'params': 'invalid'}, 'call')

        with pytest.raises(ValueError, match="'params' must be a dict"):
            validate_modification({'params': ['list', 'not', 'dict']}, 'call')

    def test_query_modification_only_in_preload(self):
        """Test that query modification is only allowed in preload phase."""
        # Should work in preload
        mod = {'query': ['modified', 'query']}
        result = validate_modification(mod, 'preload')
        assert result['query'] == ['modified', 'query']

        # Should fail in postload
        with pytest.raises(ValueError, match="Query modifications only allowed in preload"):
            validate_modification({'query': ['modified']}, 'postload')

        # Should fail in call
        with pytest.raises(ValueError, match="Query modifications only allowed in preload"):
            validate_modification({'query': ['modified']}, 'call')

    def test_unknown_fields_rejected(self):
        """Test that unknown fields are rejected."""
        with pytest.raises(ValueError, match="Unknown modification fields"):
            validate_modification({'engine': 'pandas', 'turbo': True}, 'preload')

        with pytest.raises(ValueError, match="Unknown modification fields"):
            validate_modification({'timeout': 30}, 'call')

        with pytest.raises(ValueError, match="Unknown modification fields"):
            validate_modification({'unknown_field': 'value'}, 'postload')

    def test_empty_modification_valid(self):
        """Test that empty modification is valid."""
        result = validate_modification({}, 'preload')
        assert result == {}

        result = validate_modification(None, 'postload')
        assert result == {}

    def test_multiple_valid_modifications(self):
        """Test that multiple valid modifications work together."""
        # Preload can have all three
        mod = {
            'engine': 'cudf',
            'params': {'test': 1},
            'query': ['new', 'query']
        }
        result = validate_modification(mod, 'preload')
        assert result['engine'] == 'cudf'
        assert result['params'] == {'test': 1}
        assert result['query'] == ['new', 'query']

        # Call can have engine and params, not query
        mod = {
            'engine': 'pandas',
            'params': {'n_components': 3}
        }
        result = validate_modification(mod, 'call')
        assert result['engine'] == 'pandas'
        assert result['params'] == {'n_components': 3}

    def test_modification_not_dict_rejected(self):
        """Test that non-dict modifications are rejected."""
        with pytest.raises(ValueError, match="Modifications must be a dict"):
            validate_modification("not a dict", 'preload')

        with pytest.raises(ValueError, match="Modifications must be a dict"):
            validate_modification(['list', 'not', 'dict'], 'call')

        with pytest.raises(ValueError, match="Modifications must be a dict"):
            validate_modification(123, 'postload')

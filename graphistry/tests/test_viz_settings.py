from typing import get_args

import graphistry
import graphistry.viz_settings as viz_settings
from graphistry.models.surfaces.graphistry_frontend import (
    APPLY_ENCODINGS_REACT_KEY_SET,
    APPLY_ENCODINGS_REACT_KEYS,
    REACT_SETTING_NAME_SET,
    REACT_SETTING_NAMES,
    URL_PARAM_NAME_SET,
    URL_PARAM_NAMES,
)
from graphistry.validate import normalize_react_settings, normalize_url_params


def _flat_type_args(t):
    out = []
    for arg in get_args(t):
        nested = get_args(arg)
        if nested:
            out.extend(nested)
        else:
            out.append(arg)
    return tuple(out)


def test_public_viz_settings_module_is_exported_from_package():
    assert graphistry.viz_settings is viz_settings
    assert "URL_PARAM_NAMES" in viz_settings.__all__
    assert "ReactSettingName" in viz_settings.__all__


def test_public_key_constants_match_canonical_frontend_contracts():
    assert viz_settings.URL_PARAM_NAMES == URL_PARAM_NAMES
    assert viz_settings.URL_PARAM_NAME_SET == URL_PARAM_NAME_SET
    assert viz_settings.REACT_SETTING_NAMES == REACT_SETTING_NAMES
    assert viz_settings.REACT_SETTING_NAME_SET == REACT_SETTING_NAME_SET
    assert viz_settings.APPLY_ENCODINGS_REACT_KEYS == APPLY_ENCODINGS_REACT_KEYS
    assert viz_settings.APPLY_ENCODINGS_REACT_KEY_SET == APPLY_ENCODINGS_REACT_KEY_SET


def test_public_literal_key_types_match_runtime_key_constants():
    assert tuple(get_args(viz_settings.URLParamName)) == viz_settings.URL_PARAM_NAMES
    assert tuple(get_args(viz_settings.ReactSettingName)) == viz_settings.REACT_SETTING_NAMES
    assert set(_flat_type_args(viz_settings.VizSettingName)) == viz_settings.VIZ_SETTING_NAME_SET


def test_public_typed_dict_keys_match_runtime_key_constants():
    assert set(viz_settings.KnownURLParamsDict.__annotations__) == viz_settings.URL_PARAM_NAME_SET
    assert set(viz_settings.KnownReactSettingsDict.__annotations__) == viz_settings.REACT_SETTING_NAME_SET
    assert (
        set(viz_settings.ApplyEncodingsReactSettingsDict.__annotations__)
        == viz_settings.APPLY_ENCODINGS_REACT_KEY_SET
    )


def test_public_key_constants_are_validator_accepted():
    url_params = normalize_url_params(
        {name: None for name in viz_settings.URL_PARAM_NAMES},
        validate="strict",
    )
    react_settings = normalize_react_settings(
        {name: None for name in viz_settings.REACT_SETTING_NAMES},
        validate="strict",
    )
    assert set(url_params) == viz_settings.URL_PARAM_NAME_SET
    assert set(react_settings) == viz_settings.REACT_SETTING_NAME_SET

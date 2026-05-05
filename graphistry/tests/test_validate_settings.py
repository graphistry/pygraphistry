import pytest

import graphistry
from graphistry.io.contracts import (
    GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURE,
    GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURES_BY_VERSION,
    GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION,
    GRAPHISTRY_SERVER_DATASET_UPSTREAM_VERSIONS,
    graphistry_server_dataset_contract_version_info,
)
from graphistry.validate import (
    APPLY_ENCODINGS_REACT_KEY_SET,
    GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE,
    GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURES_BY_VERSION,
    GRAPHISTRY_FRONTEND_CONTRACT_VERSION,
    GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS,
    ApplyEncodingsReactSettingsDict,
    AXIS_BOUNDS_ALLOWED_KEYS,
    AXIS_ROW_ALLOWED_KEYS,
    AXIS_ROW_BOOL_KEYS,
    AXIS_ROW_NUMERIC_KEYS,
    AXIS_ROW_POSITION_KEYS,
    LINEAR_AXIS_URL_DEFAULTS,
    RADIAL_AXIS_URL_DEFAULTS,
    REACT_SETTING_NAME_SET,
    KnownReactSettingsDict,
    KnownURLParamsDict,
    URL_PARAM_NAME_SET,
    apply_axis_url_defaults,
    axis_url_defaults,
    is_axis_bounds_payload,
    is_axis_row_payload,
    is_axis_rows_payload,
    is_ring_categorical_axis_payload,
    is_ring_continuous_axis_payload,
    normalize_react_settings,
    normalize_url_params,
    graphistry_frontend_contract_version_info,
    apply_encodings_keys,
    react_setting_keys,
    url_param_keys,
)


def test_settings_key_sets_exported():
    assert "play" in URL_PARAM_NAME_SET
    assert "encodeAxis" in REACT_SETTING_NAME_SET
    assert "dissuadeHubs" in REACT_SETTING_NAME_SET
    assert "encodePointColor" in APPLY_ENCODINGS_REACT_KEY_SET


def test_known_typed_dict_keyspaces_align_with_exported_sets():
    assert set(KnownURLParamsDict.__annotations__.keys()) == URL_PARAM_NAME_SET
    assert set(KnownReactSettingsDict.__annotations__.keys()) == REACT_SETTING_NAME_SET
    assert set(ApplyEncodingsReactSettingsDict.__annotations__.keys()) == APPLY_ENCODINGS_REACT_KEY_SET


def test_introspection_key_accessors():
    assert set(url_param_keys()) == URL_PARAM_NAME_SET
    assert set(react_setting_keys()) == REACT_SETTING_NAME_SET
    assert set(apply_encodings_keys()) == APPLY_ENCODINGS_REACT_KEY_SET


def test_bundle_contract_version_exports():
    assert GRAPHISTRY_FRONTEND_CONTRACT_VERSION >= 1
    assert "graphistry_js_client_api_react" in GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS
    frontend_info = graphistry_frontend_contract_version_info()
    assert frontend_info["bundle"] == "graphistry_frontend"
    assert frontend_info["contract_version"] == GRAPHISTRY_FRONTEND_CONTRACT_VERSION
    assert frontend_info["contract_signature"] == GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE
    assert (
        GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURES_BY_VERSION[GRAPHISTRY_FRONTEND_CONTRACT_VERSION]
        == GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE
    )

    assert GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION >= 1
    assert "graphistry_server" in GRAPHISTRY_SERVER_DATASET_UPSTREAM_VERSIONS
    server_info = graphistry_server_dataset_contract_version_info()
    assert server_info["bundle"] == "graphistry_server_dataset"
    assert server_info["contract_version"] == GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION
    assert server_info["contract_signature"] == GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURE
    assert (
        GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURES_BY_VERSION[GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION]
        == GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURE
    )


def test_axis_row_allowed_keys_contract():
    assert set(AXIS_BOUNDS_ALLOWED_KEYS) == {"min", "max"}
    assert set(AXIS_ROW_ALLOWED_KEYS) == {
        "label", "r", "x", "y", "internal", "external", "space", "width", "bounds",
    }
    assert set(AXIS_ROW_POSITION_KEYS) == {"r", "x", "y"}
    assert set(AXIS_ROW_BOOL_KEYS) == {"internal", "external", "space"}
    assert set(AXIS_ROW_NUMERIC_KEYS) == {"r", "x", "y", "width"}


def test_axis_payload_validators_exported_contract():
    assert is_axis_bounds_payload({"min": 1, "max": 2})
    assert not is_axis_bounds_payload({"min": "1"})

    assert is_axis_row_payload({"r": 200, "label": "outer", "external": True})
    assert not is_axis_row_payload({"label": "missing_pos"})

    assert is_axis_rows_payload([{"y": 40, "label": "mid", "internal": True}])
    assert not is_axis_rows_payload([{"label": "missing_pos"}])

    assert is_ring_continuous_axis_payload({100.0: "inner", 200.0: "outer"})
    assert is_ring_continuous_axis_payload(["low", "mid", "high"])
    assert not is_ring_continuous_axis_payload({100.0: 1})

    assert is_ring_categorical_axis_payload({"a": "A", "b": "B"})
    assert not is_ring_categorical_axis_payload({"a": 1})


def test_normalize_url_params_strict_unknown_key_raises():
    with pytest.raises(ValueError):
        normalize_url_params({"badKey": True}, validate="strict")


def test_normalize_url_params_autofix_drops_unknown_key():
    out = normalize_url_params({"play": 0, "badKey": True}, validate="autofix", warn=False)
    assert out == {"play": 0}


def test_normalize_react_settings_strict_invalid_type_raises():
    with pytest.raises(ValueError):
        normalize_react_settings({"encodeAxis": {1: "bad"}}, validate="strict")


def test_normalize_react_settings_accepts_dissuade_hubs():
    out = normalize_react_settings({"dissuadeHubs": True}, validate="strict")
    assert out == {"dissuadeHubs": True}


def test_normalize_url_params_accepts_neighborhood_string_modes():
    out = normalize_url_params(
        {
            "neighborhoodHighlight": "incoming",
            "neighborhoodHighlightHops": 2,
        },
        validate="strict",
    )
    assert out == {
        "neighborhoodHighlight": "incoming",
        "neighborhoodHighlightHops": 2,
    }


def test_normalize_url_params_accepts_extended_logo_position_values():
    out = normalize_url_params({"logoPosition": "top-left"}, validate="strict")
    assert out == {"logoPosition": "top-left"}


def test_plotter_settings_strict_rejects_unknown_key():
    with pytest.raises(ValueError):
        graphistry.bind().settings(url_params={"badKey": True}, validate="strict")


def test_plotter_settings_autofix_drops_unknown_key():
    g2 = graphistry.bind().settings(url_params={"play": 0, "badKey": True}, validate="autofix", warn=False)
    assert "play" in g2._url_params
    assert "badKey" not in g2._url_params


def test_axis_url_defaults_contract():
    assert axis_url_defaults("radial") == RADIAL_AXIS_URL_DEFAULTS
    assert axis_url_defaults("linear") == LINEAR_AXIS_URL_DEFAULTS


def test_apply_axis_url_defaults_radial():
    complex_encodings = {
        "node_encodings": {
            "default": {
                "pointAxisEncoding": {
                    "rows": [{"r": 200, "label": "outer", "external": True}]
                }
            },
            "current": {},
        }
    }
    out = apply_axis_url_defaults({"play": 10}, complex_encodings)
    assert out is not None
    assert out["play"] == 10
    assert out["lockedR"] is True
    assert out["splashAfter"] is False


def test_apply_axis_url_defaults_linear():
    complex_encodings = {
        "node_encodings": {
            "default": {
                "pointAxisEncoding": {
                    "rows": [{"y": 40, "label": "level", "internal": True}]
                }
            },
            "current": {},
        }
    }
    out = apply_axis_url_defaults({}, complex_encodings)
    assert out is not None
    assert out["lockedX"] is True
    assert out["lockedY"] is True
    assert out["splashAfter"] is False

"""Version metadata for the graphistry_frontend contract bundle."""

import hashlib
import json
from typing import Dict, Optional
from typing_extensions import Final, Literal, TypedDict

from .axis import AXIS_BOUNDS_ALLOWED_KEYS, AXIS_ROW_ALLOWED_KEYS
from .react_settings import APPLY_ENCODINGS_REACT_KEYS, REACT_SETTING_NAMES
from .url_params import LINEAR_AXIS_URL_DEFAULTS, RADIAL_AXIS_URL_DEFAULTS, URL_PARAM_NAMES

FrontendBundleName = Literal["graphistry_frontend"]


class FrontendContractVersionInfo(TypedDict):
    bundle: FrontendBundleName
    contract_version: int
    contract_signature: str
    upstream_versions: Dict[str, Optional[str]]


# Increment only when exported frontend contract keyspace/shape changes.
GRAPHISTRY_FRONTEND_CONTRACT_VERSION: Final[int] = 1

# Optional pins to upstream runtime/library versions when known.
GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS: Final[Dict[str, Optional[str]]] = {
    "graphistry_js_client_api_react": None,
    "graphistry_js_client_api": None,
    "graphistry_server": None,
}


def _frontend_contract_signature_payload() -> Dict[str, object]:
    return {
        "url_param_names": list(URL_PARAM_NAMES),
        "react_setting_names": list(REACT_SETTING_NAMES),
        "apply_encodings_keys": list(APPLY_ENCODINGS_REACT_KEYS),
        "axis_bounds_allowed_keys": list(AXIS_BOUNDS_ALLOWED_KEYS),
        "axis_row_allowed_keys": list(AXIS_ROW_ALLOWED_KEYS),
        "radial_axis_url_defaults": dict(RADIAL_AXIS_URL_DEFAULTS),
        "linear_axis_url_defaults": dict(LINEAR_AXIS_URL_DEFAULTS),
    }


def _contract_signature(payload: Dict[str, object]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE: Final[str] = _contract_signature(
    _frontend_contract_signature_payload()
)
GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURES_BY_VERSION: Final[Dict[int, str]] = {
    1: "bf54644c941670c46000fd7590077e33bfbda881095e77ebcf2f11821045edbd",
}

if GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURES_BY_VERSION.get(GRAPHISTRY_FRONTEND_CONTRACT_VERSION) != GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE:
    raise ValueError(
        "GRAPHISTRY_FRONTEND_CONTRACT_VERSION/signature mismatch: "
        "bump version and register new signature"
    )


def graphistry_frontend_contract_version_info() -> FrontendContractVersionInfo:
    return {
        "bundle": "graphistry_frontend",
        "contract_version": GRAPHISTRY_FRONTEND_CONTRACT_VERSION,
        "contract_signature": GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE,
        "upstream_versions": dict(GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS),
    }

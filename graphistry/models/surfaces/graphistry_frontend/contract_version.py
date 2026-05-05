"""Version metadata for the graphistry_frontend contract bundle."""

from typing import Dict, Optional
from typing_extensions import Final, Literal, TypedDict

FrontendBundleName = Literal["graphistry_frontend"]


class FrontendContractVersionInfo(TypedDict):
    bundle: FrontendBundleName
    contract_version: int
    upstream_versions: Dict[str, Optional[str]]


# Increment only when exported frontend contract keyspace/shape changes.
GRAPHISTRY_FRONTEND_CONTRACT_VERSION: Final[int] = 1

# Optional pins to upstream runtime/library versions when known.
GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS: Final[Dict[str, Optional[str]]] = {
    "graphistry_js_client_api_react": None,
    "graphistry_js_client_api": None,
    "graphistry_server": None,
}


def graphistry_frontend_contract_version_info() -> FrontendContractVersionInfo:
    return {
        "bundle": "graphistry_frontend",
        "contract_version": GRAPHISTRY_FRONTEND_CONTRACT_VERSION,
        "upstream_versions": dict(GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS),
    }

"""Version metadata for Graphistry server dataset contract bundle."""

from typing import Dict, Optional
from typing_extensions import Final, Literal, TypedDict

GraphistryServerBundleName = Literal["graphistry_server_dataset"]


class GraphistryServerContractVersionInfo(TypedDict):
    bundle: GraphistryServerBundleName
    contract_version: int
    upstream_versions: Dict[str, Optional[str]]


# Increment only when exported graphistry_server dataset contract changes.
GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION: Final[int] = 1

# Optional pins to upstream server/runtime versions when known.
GRAPHISTRY_SERVER_DATASET_UPSTREAM_VERSIONS: Final[Dict[str, Optional[str]]] = {
    "graphistry_server": None,
}


def graphistry_server_dataset_contract_version_info() -> GraphistryServerContractVersionInfo:
    return {
        "bundle": "graphistry_server_dataset",
        "contract_version": GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION,
        "upstream_versions": dict(GRAPHISTRY_SERVER_DATASET_UPSTREAM_VERSIONS),
    }

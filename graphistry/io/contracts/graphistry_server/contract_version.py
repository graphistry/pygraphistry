"""Version metadata for Graphistry server dataset contract bundle."""

import hashlib
import json
from typing import Dict, Optional
from typing_extensions import Final, Literal, TypedDict

from .dataset import GRAPHISTRY_SERVER_BINDING_TO_PLOTTABLE_ENCODING_MAP

GraphistryServerBundleName = Literal["graphistry_server_dataset"]


class GraphistryServerContractVersionInfo(TypedDict):
    bundle: GraphistryServerBundleName
    contract_version: int
    contract_signature: str
    upstream_versions: Dict[str, Optional[str]]


# Increment only when exported graphistry_server dataset contract changes.
GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION: Final[int] = 1

# Optional pins to upstream server/runtime versions when known.
GRAPHISTRY_SERVER_DATASET_UPSTREAM_VERSIONS: Final[Dict[str, Optional[str]]] = {
    "graphistry_server": None,
}


def _server_dataset_contract_signature_payload() -> Dict[str, object]:
    return {
        "binding_to_plottable_encoding_map": dict(GRAPHISTRY_SERVER_BINDING_TO_PLOTTABLE_ENCODING_MAP),
    }


def _contract_signature(payload: Dict[str, object]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURE: Final[str] = _contract_signature(
    _server_dataset_contract_signature_payload()
)
# Signature map is keyed by contract version.
# Each value must be the output of:
#   _contract_signature(_server_dataset_contract_signature_payload())
# for the payload shape shipped with that version.
GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURES_BY_VERSION: Final[Dict[int, str]] = {
    1: "1096ca85da67020943be3d886250a4ec837773c0cc4314ffae7c9b9f7e273b9d",
}

if GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURES_BY_VERSION.get(
    GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION
) != GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURE:
    raise ValueError(
        "graphistry_server_dataset contract drift detected. "
        "Update graphistry/io/contracts/graphistry_server/contract_version.py: "
        "(1) bump GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION, "
        "(2) add the new signature to GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURES_BY_VERSION, "
        "(3) optionally set GRAPHISTRY_SERVER_DATASET_UPSTREAM_VERSIONS pins. "
        f"computed_signature={GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURE}, "
        "expected_signature="
        f"{GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURES_BY_VERSION.get(GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION)}."
    )


def graphistry_server_dataset_contract_version_info() -> GraphistryServerContractVersionInfo:
    return {
        "bundle": "graphistry_server_dataset",
        "contract_version": GRAPHISTRY_SERVER_DATASET_CONTRACT_VERSION,
        "contract_signature": GRAPHISTRY_SERVER_DATASET_CONTRACT_SIGNATURE,
        "upstream_versions": dict(GRAPHISTRY_SERVER_DATASET_UPSTREAM_VERSIONS),
    }

from typing import List, Any, Optional, TYPE_CHECKING
from typing_extensions import TypedDict, NotRequired
from dataclasses import dataclass

if TYPE_CHECKING:
    from azure.kusto.data import KustoClient
else:
    KustoClient = Any

class KustoConnectionError(Exception):
    pass


class KustoQueryResult:
    def __init__(self, data: List[List[Any]], column_names: List[str], column_types: List[str]):
        self.data = data
        self.column_names = column_names
        self.column_types = column_types


class KustoConfig(TypedDict):
    cluster: str
    database: str
    client_id: NotRequired[str]
    client_secret: NotRequired[str]
    tenant_id: NotRequired[str]


@dataclass
class KustoSession:
    client: Optional[KustoClient] = None
    database: Optional[str] = None
    config: Optional[KustoConfig] = None

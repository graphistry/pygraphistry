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


@dataclass
class KustoConfig:
    database: str
    cluster: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = None

    _client: Optional[KustoClient] = None

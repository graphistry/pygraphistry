from typing import Optional, List, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import timedelta

if TYPE_CHECKING:
    from azure.monitor.query import LogsQueryClient
    from azure.core.credentials import TokenCredential
else:
    LogsQueryClient = Any
    TokenCredential = Any


class SentinelConnectionError(Exception):
    """Raised when connection to Log Analytics workspace fails"""
    pass


class SentinelQueryError(Exception):
    """Raised when query execution fails"""
    pass


class SentinelQueryResult:
    """Container for a single query result table from Microsoft Sentinel"""

    def __init__(
        self,
        data: List[List[Any]],
        column_names: List[str],
        column_types: List[str],
        table_name: Optional[str] = None
    ):
        """
        Initialize a Sentinel query result.

        :param data: List of rows, where each row is a list of values
        :param column_names: List of column names
        :param column_types: List of column types (e.g., 'string', 'datetime', 'int')
        :param table_name: Optional name of the result table
        """
        self.data = data
        self.column_names = column_names
        self.column_types = column_types
        self.table_name = table_name


@dataclass
class SentinelConfig:
    """Configuration for Microsoft Sentinel Log Analytics connection"""

    workspace_id: str
    """The Log Analytics workspace ID (GUID format)"""

    tenant_id: Optional[str] = None
    """Azure AD tenant ID for authentication"""

    client_id: Optional[str] = None
    """Azure AD application (client) ID for service principal auth"""

    client_secret: Optional[str] = None
    """Azure AD application secret for service principal auth"""

    credential: Optional[TokenCredential] = None
    """Custom credential object for authentication"""

    default_timespan: timedelta = timedelta(hours=24)
    """Default time range for queries when not specified"""

    use_device_auth: bool = False
    """Use device code authentication flow"""

    _client: Optional[LogsQueryClient] = None
    """Cached client instance (internal use)"""

from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential
else:
    TokenCredential = Any


class SentinelGraphConnectionError(Exception):
    """Raised when connection to Sentinel Graph API fails"""
    pass


class SentinelGraphQueryError(Exception):
    """Raised when a Sentinel Graph query fails"""
    pass


@dataclass
class SentinelGraphConfig:
    """Configuration for Microsoft Sentinel Graph API connection"""
    graph_instance: str

    # Endpoint configuration
    api_endpoint: str = "api.securityplatform.microsoft.com"
    auth_scope: str = "73c2949e-da2d-457a-9607-fcc665198967/.default"

    # HTTP configuration
    timeout: int = 60
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    verify_ssl: bool = True

    # Authentication options
    credential: Optional[TokenCredential] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    use_device_auth: bool = False

    # Internal state (not user-configurable)
    _token: Optional[str] = field(default=None, repr=False)
    _token_expiry: Optional[float] = field(default=None, repr=False)

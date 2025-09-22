import time
import pandas as pd
from typing import Any, List, Optional, TYPE_CHECKING, Union, overload, Literal
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from azure.monitor.query import LogsQueryClient
    from azure.core.credentials import TokenCredential
else:
    LogsQueryClient = Any
    TokenCredential = Any

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from graphistry.plugins_types.sentinel_types import (
    SentinelConfig,
    SentinelConnectionError,
    SentinelQueryError,
    SentinelQueryResult
)

logger = setup_logger(__name__)


class SentinelMixin(Plottable):
    """
    Microsoft Sentinel Log Analytics integration for Graphistry.

    This mixin allows you to query Microsoft Sentinel (Azure Log Analytics)
    using KQL (Kusto Query Language) and visualize the results with Graphistry.
    """

    def configure_sentinel(
        self,
        workspace_id: str,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        credential: Optional["TokenCredential"] = None,
        default_timespan: Optional[timedelta] = None,
    ) -> Plottable:
        """Configure Microsoft Sentinel Log Analytics connection settings.

        Sets up the connection parameters for accessing a Log Analytics workspace.
        Authentication can be done via:
        - Custom credential object (highest priority)
        - Service principal (client_id, client_secret, tenant_id)
        - DefaultAzureCredential (includes Azure CLI, Managed Identity, etc.)

        :param workspace_id: Log Analytics workspace ID (GUID format)
        :type workspace_id: str
        :param tenant_id: Azure AD tenant ID for authentication
        :type tenant_id: Optional[str]
        :param client_id: Azure AD application (client) ID for service principal auth
        :type client_id: Optional[str]
        :param client_secret: Azure AD application secret for service principal auth
        :type client_secret: Optional[str]
        :param credential: Custom credential object for authentication
        :type credential: Optional[TokenCredential]
        :param default_timespan: Default time range for queries (defaults to 24 hours)
        :type default_timespan: Optional[timedelta]
        :returns: Self for method chaining
        :rtype: Plottable

        **Example: Azure CLI authentication (development)**
            ::

                import graphistry
                # First run: az login
                g = graphistry.configure_sentinel(
                    workspace_id="12345678-1234-1234-1234-123456789abc"
                )

        **Example: Service principal authentication (production)**
            ::

                import graphistry
                g = graphistry.configure_sentinel(
                    workspace_id="12345678-1234-1234-1234-123456789abc",
                    tenant_id="your-tenant-id",
                    client_id="your-client-id",
                    client_secret="your-client-secret"
                )

        **Example: Custom credential**
            ::

                from azure.identity import DeviceCodeCredential
                import graphistry

                credential = DeviceCodeCredential()
                g = graphistry.configure_sentinel(
                    workspace_id="12345678-1234-1234-1234-123456789abc",
                    credential=credential
                )
        """
        self.session.sentinel = SentinelConfig(
            workspace_id=workspace_id,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            credential=credential,
            default_timespan=default_timespan or timedelta(hours=24),
        )
        return self

    def sentinel_from_client(
        self,
        client: LogsQueryClient,
        workspace_id: str,
        default_timespan: Optional[timedelta] = None
    ) -> Plottable:
        """Configure Sentinel using an existing LogsQueryClient connection.

        Use this method when you already have a configured LogsQueryClient
        and want to reuse it with Graphistry.

        :param client: Pre-configured LogsQueryClient
        :type client: azure.monitor.query.LogsQueryClient
        :param workspace_id: Log Analytics workspace ID
        :type workspace_id: str
        :param default_timespan: Default time range for queries
        :type default_timespan: Optional[timedelta]
        :returns: Self for method chaining
        :rtype: Plottable

        **Example**
            ::

                from azure.monitor.query import LogsQueryClient
                from azure.identity import DefaultAzureCredential
                import graphistry

                # Create client
                credential = DefaultAzureCredential()
                logs_client = LogsQueryClient(credential)

                # Use with Graphistry
                g = graphistry.sentinel_from_client(
                    logs_client,
                    "12345678-1234-1234-1234-123456789abc"
                )
        """
        # Clean up existing client if different
        if self.session.sentinel is not None and client is not self.session.sentinel._client:
            self.sentinel_close()

        self.session.sentinel = SentinelConfig(
            workspace_id=workspace_id,
            default_timespan=default_timespan or timedelta(hours=24),
            _client=client,
        )
        return self

    @property
    def _sentinel_config(self) -> SentinelConfig:
        """Get the current Sentinel configuration."""
        if self.session.sentinel is None:
            raise ValueError("SentinelMixin is not configured")
        return self.session.sentinel

    @property
    def sentinel_client(self) -> LogsQueryClient:
        """Get or create the LogsQueryClient instance."""
        if self._sentinel_config._client is not None:
            return self._sentinel_config._client
        client = init_sentinel_client(self._sentinel_config)
        self._sentinel_config._client = client
        return client

    def sentinel_close(self) -> None:
        """Close the Sentinel client connection.

        Note: LogsQueryClient doesn't require explicit cleanup,
        but this method is provided for API consistency.

        **Example**
            ::

                import graphistry
                g = graphistry.configure_sentinel(...)
                # ... perform queries ...
                g.sentinel_close()
        """
        if self.session.sentinel is None:
            return
        # LogsQueryClient doesn't need explicit cleanup
        # Just clear the cached client reference
        self.session.sentinel._client = None

    def sentinel_health_check(self) -> None:
        """Perform a health check on the Sentinel connection.

        Executes a simple query (Heartbeat | take 1) to verify that the connection
        to the Log Analytics workspace is working properly.

        :raises SentinelConnectionError: If the connection test fails

        **Example**
            ::

                import graphistry
                g = graphistry.configure_sentinel(...)
                g.sentinel_health_check()  # Verify connection works
        """
        try:
            self._sentinel_query("Heartbeat | take 1", timespan=timedelta(hours=1))
            logger.info("Sentinel health check successful")
        except Exception as e:
            raise SentinelConnectionError(f"Health check failed: {e}") from e

    # Query methods will be added next...
    def _sentinel_query(
        self,
        query: str,
        timespan: Optional[Union[timedelta, tuple[datetime, datetime]]] = None
    ) -> List[SentinelQueryResult]:
        """Execute KQL query and return raw results.

        Internal method for executing KQL queries and returning raw Sentinel
        query results without DataFrame conversion.

        :param query: KQL query string to execute
        :type query: str
        :param timespan: Time range for the query
        :type timespan: Optional[Union[timedelta, tuple[datetime, datetime]]]
        :returns: List of raw query results
        :rtype: List[SentinelQueryResult]
        :raises SentinelQueryError: If the query execution fails
        """
        from azure.monitor.query import LogsQueryStatus
        from azure.core.exceptions import HttpResponseError

        logger.debug(f"SentinelMixin._sentinel_query(): {query}")

        # Use default timespan if not provided
        if timespan is None:
            timespan = self._sentinel_config.default_timespan

        try:
            start = time.time()
            response = self.sentinel_client.query_workspace(
                workspace_id=self._sentinel_config.workspace_id,
                query=query,
                timespan=timespan
            )

            # Check for partial failures
            if response.status == LogsQueryStatus.PARTIAL:
                logger.warning(f"Query returned partial results: {response.partial_error}")
            elif response.status == LogsQueryStatus.FAILURE:
                raise SentinelQueryError(f"Query failed: {response.partial_error}")

            results = []
            row_lengths = []

            # Process each table in the response
            for table in response.tables:
                rows = [list(row) for row in table.rows]
                col_names = [col.name for col in table.columns]
                col_types = [col.type for col in table.columns]

                results.append(SentinelQueryResult(
                    data=rows,
                    column_names=col_names,
                    column_types=col_types,
                    table_name=table.name
                ))
                row_lengths.append((len(rows), len(col_names)))

            logger.info(f"Query returned {len(results)} tables shapes: {row_lengths} in {time.time() - start:.3f} sec")
            return results

        except HttpResponseError as e:
            logger.error(f"Sentinel query failed: {e}")
            raise SentinelQueryError(f"Query failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}")
            raise SentinelQueryError(f"Unexpected error: {e}") from e


def init_sentinel_client(cfg: SentinelConfig) -> "LogsQueryClient":
    """Initialize Sentinel Log Analytics client with appropriate authentication.

    Authentication precedence:
    1. Custom credential object (if provided)
    2. Service Principal (if credentials provided)
    3. DefaultAzureCredential (tries multiple methods automatically)

    For Azure CLI auth: Run 'az login' before using this method.
    """
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.monitor.query import LogsQueryClient

    try:
        assert cfg.workspace_id is not None, "workspace_id is not set"

        if cfg.credential:
            credential = cfg.credential
            logger.info("Using custom credential object for Sentinel")
        elif cfg.client_id and cfg.client_secret and cfg.tenant_id:
            credential = ClientSecretCredential(
                tenant_id=cfg.tenant_id,
                client_id=cfg.client_id,
                client_secret=cfg.client_secret
            )
            logger.info(f"Using Service Principal authentication for workspace {cfg.workspace_id}")
        else:
            credential = DefaultAzureCredential()
            logger.info(f"Using DefaultAzureCredential (Azure CLI, Managed Identity, etc.) for workspace {cfg.workspace_id}")

        client = LogsQueryClient(credential)
        return client

    except Exception as exc:
        raise SentinelConnectionError(f"Failed to initialize Sentinel client: {exc}") from exc
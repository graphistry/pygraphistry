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
        use_device_auth: bool = False,
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
        :param use_device_auth: Use device code authentication (shows code and URL)
        :type use_device_auth: bool
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

        **Example: Device code authentication (interactive)**
            ::

                import graphistry
                g = graphistry.configure_sentinel(
                    workspace_id="12345678-1234-1234-1234-123456789abc",
                    use_device_auth=True
                )
                # This will show a code and URL for authentication

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
            use_device_auth=use_device_auth,
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

    @overload
    def kql(
        self,
        query: str,
        *,
        timespan: Optional[Union[timedelta, tuple[datetime, datetime]]] = None,
        unwrap_nested: Optional[bool] = None,
        single_table: Literal[True] = True,
        include_statistics: bool = False
    ) -> pd.DataFrame:
        ...

    @overload
    def kql(
        self,
        query: str,
        *,
        timespan: Optional[Union[timedelta, tuple[datetime, datetime]]] = None,
        unwrap_nested: Optional[bool] = None,
        single_table: Literal[False],
        include_statistics: bool = False
    ) -> List[pd.DataFrame]:
        ...

    @overload
    def kql(
        self,
        query: str,
        *,
        timespan: Optional[Union[timedelta, tuple[datetime, datetime]]] = None,
        unwrap_nested: Optional[bool] = None,
        single_table: bool = True,
        include_statistics: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        ...

    def kql(
        self,
        query: str,
        *,
        timespan: Optional[Union[timedelta, tuple[datetime, datetime]]] = None,
        unwrap_nested: Optional[bool] = None,
        single_table: bool = True,
        include_statistics: bool = False
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Execute KQL query and return result tables as DataFrames.

        Submits a Kusto Query Language (KQL) query to Microsoft Sentinel (Log Analytics)
        and returns the results. By default, expects a single table result and returns
        it as a DataFrame. If multiple tables are returned, only the first is returned
        with a warning. Set single_table=False to get all result tables.

        :param query: KQL query string to execute
        :type query: str
        :param timespan: Time range for the query (default: 24 hours)
        :type timespan: Optional[Union[timedelta, tuple[datetime, datetime]]]
        :param unwrap_nested: Strategy for handling nested/dynamic columns
        :type unwrap_nested: Optional[bool]
        :param single_table: If True, return single DataFrame; if False, return list
        :type single_table: bool
        :param include_statistics: Include query statistics in DataFrame attrs
        :type include_statistics: bool
        :returns: Single DataFrame if single_table=True, else list of DataFrames
        :rtype: Union[pd.DataFrame, List[pd.DataFrame]]

        **unwrap_nested semantics:**

        - **True**: Always attempt to unwrap nested columns; raise on failure
        - **None**: Use heuristic - unwrap if the result looks nested
        - **False**: Never attempt to unwrap nested columns

        **Example: Basic security query (single table mode)**
            ::

                import graphistry
                from datetime import timedelta
                g = graphistry.configure_sentinel(...)

                query = '''
                SecurityEvent
                | where TimeGenerated > ago(1d)
                | where EventID == 4625  // Failed logon
                | project TimeGenerated, Account, Computer, IpAddress
                | take 1000
                '''

                # Query last 7 days
                df = g.kql(query, timespan=timedelta(days=7))
                print(f"Found {len(df)} failed logon events")

        **Example: Get all tables as list**
            ::

                # Always get a list of all tables
                dfs = g.kql(query, single_table=False)
                df = dfs[0]

        **Example: Query with specific time range**
            ::

                from datetime import datetime, timedelta

                # Query specific time window
                start = datetime(2024, 1, 1)
                end = datetime(2024, 1, 7)
                df = g.kql(query, timespan=(start, end))

        **Example: Multi-table query**
            ::

                query = '''
                SecurityEvent | summarize Count=count() by EventID | top 5 by Count;
                SecurityAlert | take 10
                '''

                # With single_table=False, returns all tables
                frames = g.kql(query, single_table=False)
                events_df = frames[0]
                alerts_df = frames[1]
        """
        results = self._sentinel_query(query, timespan=timespan)

        if not results:
            if single_table:
                raise ValueError("Query returned no results")
            return []

        dfs: List[pd.DataFrame] = []

        for result in results:
            # Determine if we should unwrap nested data
            do_unwrap = (
                unwrap_nested is True or
                (unwrap_nested is None and _should_unwrap(result))
            )

            if do_unwrap:
                try:
                    df_unwrapped = _unwrap_nested(result)
                    dfs.append(df_unwrapped)
                    continue
                except Exception as exc:
                    if unwrap_nested is True:
                        raise RuntimeError(f"Failed to unwrap nested data: {exc}") from exc
                    # Heuristic miss - fall back to flat table
                    pass

            # Default: flat table
            if not result.column_names:
                # Safety fallback
                dfs.append(pd.DataFrame(result.data))
            else:
                dfs.append(pd.DataFrame(result.data, columns=result.column_names))

        # Auto-unbox single table result if requested
        if single_table:
            if len(dfs) > 1:
                logger.warning(f"Query returned {len(dfs)} tables, returning first table only")
            return dfs[0]

        return dfs

    def kql_last(
        self,
        query: str,
        *,
        hours: float = 1,
        **kwargs
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Execute KQL query for the last N hours.

        Convenience wrapper for kql() that automatically sets the timespan
        to the last N hours from now.

        :param query: KQL query string to execute
        :type query: str
        :param hours: Number of hours to look back (default: 1)
        :type hours: float
        :param kwargs: Additional arguments passed to kql()
        :returns: Query results as DataFrame(s)
        :rtype: Union[pd.DataFrame, List[pd.DataFrame]]

        **Example: Get security alerts from last 24 hours**
            ::

                import graphistry
                g = graphistry.configure_sentinel(...)

                alerts = g.kql_last('''
                    SecurityAlert
                    | project TimeGenerated, AlertName, Severity
                    | order by TimeGenerated desc
                ''', hours=24)

        **Example: Get recent failed logins (last hour)**
            ::

                # Default is 1 hour
                recent_failures = g.kql_last('''
                    SecurityEvent
                    | where EventID == 4625
                    | summarize FailCount=count() by Account
                ''')
        """
        return self.kql(query, timespan=timedelta(hours=hours), **kwargs)

    def sentinel_tables(self) -> pd.DataFrame:
        """List all available tables in the Log Analytics workspace.

        :returns: DataFrame with table names
        :rtype: pd.DataFrame

        **Example**
            ::

                import graphistry
                g = graphistry.configure_sentinel(...)

                # Get list of all tables
                tables = g.sentinel_tables()
                print(f"Found {len(tables)} tables")
                print(tables.head(10))
        """
        query = "union withsource=TableName * | distinct TableName | sort by TableName asc"
        return self.kql(query, timespan=timedelta(minutes=5))

    def sentinel_schema(self, table: str) -> pd.DataFrame:
        """Get schema information for a specific table.

        :param table: Name of the table to inspect
        :type table: str
        :returns: DataFrame with column names and types
        :rtype: pd.DataFrame

        **Example**
            ::

                import graphistry
                g = graphistry.configure_sentinel(...)

                # Get schema for SecurityEvent table
                schema = g.sentinel_schema("SecurityEvent")
                print(schema[['ColumnName', 'DataType']])
        """
        query = f"{table} | getschema"
        return self.kql(query, timespan=timedelta(minutes=5))

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
    3. Device code authentication (if use_device_auth=True)
    4. DefaultAzureCredential (tries multiple methods automatically)

    For Azure CLI auth: Run 'az login' before using this method.
    """
    from azure.identity import DefaultAzureCredential, ClientSecretCredential, DeviceCodeCredential
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
        elif cfg.use_device_auth:
            credential = DeviceCodeCredential(
                tenant_id=cfg.tenant_id  # Optional, uses common tenant if not provided
            )
            logger.info(f"Using Device Code authentication for workspace {cfg.workspace_id}")
            logger.info("You will be prompted to visit a URL and enter a code to authenticate")
        else:
            credential = DefaultAzureCredential()
            logger.info(f"Using DefaultAzureCredential (Azure CLI, Managed Identity, etc.) for workspace {cfg.workspace_id}")

        client = LogsQueryClient(credential)
        return client

    except Exception as exc:
        raise SentinelConnectionError(f"Failed to initialize Sentinel client: {exc}") from exc


# Sentinel Utils - adapted from Kusto plugin
def _is_dynamic(val: Any) -> bool:
    """Check if value is a nested/dynamic JSON type."""
    return isinstance(val, (dict, list))


def _unwrap_nested(result: SentinelQueryResult) -> pd.DataFrame:
    """
    Transform a Sentinel result whose columns contain nested/dynamic objects.

    - dict          -> dot-flattened
    - list[dict]    -> explode + flatten
    - list[scalar]  -> keep as-is
    """
    df = pd.DataFrame(result.data, columns=result.column_names)
    if not result.column_types:
        return df

    for col, col_type in zip(result.column_names, result.column_types):
        # Check for dynamic/object types (common in Sentinel)
        if col_type.lower() in ["dynamic", "object", "string"]:
            # Check if column contains JSON strings that need parsing
            if col_type.lower() == "string" and len(df) > 0:
                try:
                    # Try to parse first non-null value as JSON
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample and isinstance(sample, str) and (sample.startswith('{') or sample.startswith('[')):
                        import json
                        df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else x)
                except (json.JSONDecodeError, IndexError):
                    continue  # Not JSON, keep as string

            # Handle lists of dicts - need to explode
            list_of_dicts = df[col].apply(
                lambda v: isinstance(v, list) and (not v or all(isinstance(x, dict) for x in v))
            )
            if list_of_dicts.any():
                df[col] = df[col].where(
                    list_of_dicts,
                    df[col].apply(lambda x: [x] if pd.notna(x) else x)
                )
                df = df.explode(col, ignore_index=True)

            # Flatten dict columns
            dict_rows = df[col].apply(lambda v: isinstance(v, dict))
            if dict_rows.any():
                flat = pd.json_normalize(df.loc[dict_rows, col].tolist(), sep='.').add_prefix(f"{col}.")
                flat.index = df.loc[dict_rows].index
                df = df.join(flat, how='left')
                df[col] = df[col].mask(dict_rows, pd.NA)

        # Drop column if all values are NA after processing
        if df[col].isna().all():
            df = df.drop(columns=[col])

    # Clean up - replace pd.NA with None for consistency
    df = df.astype(object).where(pd.notna(df), None)
    return df.reset_index(drop=True)


def _should_unwrap(result: SentinelQueryResult, sample_rows: int = 5) -> bool:
    """
    Decide whether result looks like it contains nested/dynamic columns.

    Strategy:
      1. Check column types for 'dynamic' or 'object'
      2. Inspect sample rows for dict/list values
      3. Check for JSON strings
    """
    # Check column types
    if result.column_types:
        for col_type in result.column_types:
            if col_type.lower() in ["dynamic", "object"]:
                return True

    # Sample data for nested structures
    for col_idx in range(len(result.column_names)):
        sample = (row[col_idx] for row in result.data[:sample_rows] if row)
        for val in sample:
            if _is_dynamic(val):
                return True
            # Check for JSON strings
            if isinstance(val, str) and val and (val.startswith('{') or val.startswith('[')):
                try:
                    import json
                    json.loads(val)
                    return True
                except (json.JSONDecodeError, ValueError):
                    continue

    return False
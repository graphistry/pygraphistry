import json
import time
import requests
import pandas as pd
from typing import Optional, Union, TYPE_CHECKING
from functools import wraps

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential
else:
    TokenCredential = object

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from graphistry.plugins_types.sentinel_graph_types import (
    SentinelGraphConfig,
    SentinelGraphConnectionError,
    SentinelGraphQueryError
)

logger = setup_logger(__name__)


def retry_on_request_exception(func):
    """Decorator for HTTP retry with exponential backoff"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        cfg = self._sentinel_graph_config

        for attempt in range(cfg.max_retries):
            try:
                return func(self, *args, **kwargs)
            except requests.exceptions.RequestException as e:
                if attempt < cfg.max_retries - 1:
                    wait_time = cfg.retry_backoff_factor ** attempt
                    # Security: Log exception type but not details (might contain URLs with sensitive data)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{cfg.max_retries}): "
                        f"{type(e).__name__}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)

        # Security: Provide generic error message
        raise SentinelGraphConnectionError(
            f"Request failed after {cfg.max_retries} retries. "
            f"Check network connectivity and endpoint configuration."
        )

    return wrapper


class SentinelGraphMixin(Plottable):
    """
    Microsoft Sentinel Graph API integration for graph queries.

    This mixin allows you to query Microsoft Security Platform Graph API
    using GQL (Graph Query Language) and visualize the results with Graphistry.

    Security Notes:
        - Authentication tokens are cached in memory with repr=False to prevent accidental exposure
        - HTTPS is enforced for all API endpoints
        - SSL certificate verification is enabled by default
        - Credentials (client_secret, tokens) are never logged
        - Error messages are sanitized to prevent information disclosure
        - Query content is not logged to prevent exposure of sensitive data
    """

    def configure_sentinel_graph(
        self,
        graph_instance: str,
        credential: Optional["TokenCredential"] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        use_device_auth: bool = False,
        api_endpoint: str = "api.securityplatform.microsoft.com",
        auth_scope: str = "73c2949e-da2d-457a-9607-fcc665198967/.default",
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        verify_ssl: bool = True
    ) -> Plottable:
        """Configure Microsoft Sentinel Graph API connection.

        Sets up the connection parameters for accessing a Sentinel Graph instance.
        Authentication can be done via:
        - Custom credential object (highest priority)
        - Service principal (client_id + client_secret + tenant_id)
        - Device code auth (use_device_auth=True)
        - Interactive browser credential (fallback)

        :param graph_instance: Graph instance name (e.g., "YourGraphInstance")
        :type graph_instance: str
        :param credential: Custom credential object for authentication
        :type credential: Optional[TokenCredential]
        :param tenant_id: Azure AD tenant ID for service principal auth
        :type tenant_id: Optional[str]
        :param client_id: Azure AD application (client) ID
        :type client_id: Optional[str]
        :param client_secret: Azure AD application secret
        :type client_secret: Optional[str]
        :param use_device_auth: Use device code authentication flow
        :type use_device_auth: bool
        :param api_endpoint: API endpoint hostname
        :type api_endpoint: str
        :param auth_scope: OAuth scope for authentication
        :type auth_scope: str
        :param timeout: Request timeout in seconds
        :type timeout: int
        :param max_retries: Maximum number of retry attempts
        :type max_retries: int
        :param retry_backoff_factor: Exponential backoff factor for retries
        :type retry_backoff_factor: float
        :param verify_ssl: Verify SSL certificates (default: True, recommended for security)
        :type verify_ssl: bool
        :returns: Self for method chaining
        :rtype: Plottable

        **Example: Interactive browser authentication**
            ::

                import graphistry
                g = graphistry.configure_sentinel_graph(
                    graph_instance="YourGraphInstance"
                )

        **Example: Service principal authentication**
            ::

                import graphistry
                g = graphistry.configure_sentinel_graph(
                    graph_instance="YourGraphInstance",
                    tenant_id="your-tenant-id",
                    client_id="your-client-id",
                    client_secret="your-client-secret"
                )

        **Example: Custom scope for different environment**
            ::

                import graphistry
                g = graphistry.configure_sentinel_graph(
                    graph_instance="CustomGraphInstance",
                    auth_scope="custom-scope/.default",
                    api_endpoint="custom.endpoint.com"
                )
        """
        # Security: Validate endpoint doesn't use HTTP
        if api_endpoint.startswith('http://'):
            raise ValueError(
                "HTTP endpoints are not allowed for security reasons. "
                "Please use HTTPS or provide hostname only."
            )

        # Strip https:// prefix if provided (we'll add it in the request)
        api_endpoint_clean = api_endpoint.replace('https://', '')

        self.session.sentinel_graph = SentinelGraphConfig(
            graph_instance=graph_instance,
            api_endpoint=api_endpoint_clean,
            auth_scope=auth_scope,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            verify_ssl=verify_ssl,
            credential=credential,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            use_device_auth=use_device_auth
        )
        return self

    def sentinel_graph_from_credential(
        self,
        credential: "TokenCredential",
        graph_instance: str,
        **kwargs
    ) -> Plottable:
        """Configure Sentinel Graph using an existing credential.

        Use this method when you already have a configured credential
        and want to reuse it with Graphistry.

        :param credential: Pre-configured TokenCredential
        :type credential: TokenCredential
        :param graph_instance: Graph instance name
        :type graph_instance: str
        :param kwargs: Additional configuration options (see configure_sentinel_graph)
        :returns: Self for method chaining
        :rtype: Plottable

        **Example**
            ::

                from azure.identity import DefaultAzureCredential
                import graphistry

                credential = DefaultAzureCredential()
                g = graphistry.sentinel_graph_from_credential(
                    credential,
                    "YourGraphInstance"
                )
        """
        return self.configure_sentinel_graph(
            graph_instance=graph_instance,
            credential=credential,
            **kwargs
        )

    @property
    def _sentinel_graph_config(self) -> SentinelGraphConfig:
        """Get the current Sentinel Graph configuration."""
        if self.session.sentinel_graph is None:
            raise ValueError(
                "SentinelGraphMixin is not configured. Call configure_sentinel_graph() first."
            )
        return self.session.sentinel_graph

    def sentinel_graph_close(self) -> None:
        """Clear cached authentication token.

        **Example**
            ::

                import graphistry

                graphistry.configure_sentinel_graph(...)
                # ... perform queries ...
                graphistry.sentinel_graph_close()
        """
        if self.session.sentinel_graph is not None:
            self.session.sentinel_graph._token = None
            self.session.sentinel_graph._token_expiry = None

    def sentinel_graph(
        self,
        query: str,
        language: str = 'GQL'
    ) -> Plottable:
        """Execute graph query and return Plottable with nodes/edges bound.

        This is the main method - handles auth, query execution, and parsing automatically.

        :param query: GQL query string
        :type query: str
        :param language: Query language (default: 'GQL')
        :type language: str
        :returns: Plottable with nodes and edges bound
        :rtype: Plottable

        **Example: Query graph data**
            ::

                import graphistry

                graphistry.configure_sentinel_graph('YourGraphInstance')

                viz = graphistry.sentinel_graph('''
                    MATCH (n)-[e]->(m)
                    RETURN *
                    LIMIT 100
                ''')

                viz.plot()

        **Example: Multiple queries**
            ::

                import graphistry

                graphistry.configure_sentinel_graph('YourGraphInstance')

                # Query 1
                result1 = graphistry.sentinel_graph('MATCH (n) RETURN * LIMIT 10')

                # Query 2
                result2 = graphistry.sentinel_graph('MATCH (a)-[r]->(b) RETURN * LIMIT 20')
        """
        # Execute query
        response_bytes = self._sentinel_graph_query(query, language)

        # Parse and return Plottable
        return self._parse_graph_response(response_bytes)

    @retry_on_request_exception
    def _sentinel_graph_query(self, query: str, language: str) -> bytes:
        """Internal: Execute query and return raw response bytes"""
        cfg = self._sentinel_graph_config
        token = self._get_auth_token()

        url = f"https://{cfg.api_endpoint}/graphs/graph-instances/{cfg.graph_instance}/query"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "pygraphistry-sentinel-graph"
        }

        payload = {
            "query": query,
            "queryLanguage": language
        }

        # Security: Don't log query content (could contain sensitive data)
        logger.debug(f"Executing {language} query against graph instance: {cfg.graph_instance}")

        # Security: Explicit SSL verification
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=cfg.timeout,
            verify=cfg.verify_ssl
        )

        if response.status_code == 200:
            logger.info(f"Query successful: {len(response.content)} bytes returned")
            return response.content
        else:
            # Security: Don't expose raw API error messages which could contain sensitive info
            # Instead provide generic error with status code only
            raise SentinelGraphQueryError(
                f"Query failed with status {response.status_code}. "
                f"Check your query syntax and permissions."
            )

    def _get_auth_token(self) -> str:
        """Internal: Get or refresh authentication token with 5-minute expiry buffer."""
        cfg = self._sentinel_graph_config

        # Check cached token (5 min buffer)
        if cfg._token and cfg._token_expiry:
            time_remaining = cfg._token_expiry - time.time()
            if time_remaining > 300:  # 5 min buffer
                logger.debug(f"Using cached token (expires in {int(time_remaining)}s)")
                return cfg._token

        # Get new token
        from azure.identity import (
            ClientSecretCredential,
            DeviceCodeCredential,
            InteractiveBrowserCredential,
            DefaultAzureCredential
        )

        try:
            # Determine credential type
            if cfg.credential:
                logger.debug("Using provided credential")
                credential = cfg.credential
            elif cfg.client_id and cfg.client_secret and cfg.tenant_id:
                logger.debug("Using service principal authentication")
                credential = ClientSecretCredential(
                    tenant_id=cfg.tenant_id,
                    client_id=cfg.client_id,
                    client_secret=cfg.client_secret
                )
            elif cfg.use_device_auth:
                logger.info("Using device code authentication")
                credential = DeviceCodeCredential()
            else:
                logger.debug("Using interactive browser authentication")
                try:
                    credential = InteractiveBrowserCredential()
                except Exception:
                    # Security: Don't log exception details which might contain sensitive info
                    logger.warning(
                        "Interactive browser auth failed. "
                        "Falling back to DefaultAzureCredential"
                    )
                    credential = DefaultAzureCredential()

            # Get token
            token_obj = credential.get_token(cfg.auth_scope)
            cfg._token = token_obj.token
            cfg._token_expiry = token_obj.expires_on

            logger.info("Successfully obtained authentication token")
            return cfg._token

        except Exception:
            # Security: Don't expose credential details or exception messages
            raise SentinelGraphConnectionError(
                "Authentication failed. Please verify your credentials, tenant ID, "
                "and that you have the correct permissions for the auth scope."
            )

    def _parse_graph_response(self, response: Union[bytes, dict]) -> Plottable:
        """Internal: Parse response and return Plottable"""
        # Parse JSON
        if isinstance(response, bytes):
            try:
                parsed = json.loads(response.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise SentinelGraphQueryError(f"Failed to parse response as JSON: {e}")
        else:
            parsed = response

        # Extract nodes and edges
        nodes_df = self._extract_nodes(parsed)
        edges_df = self._extract_edges(parsed)

        logger.info(f"Extracted {len(nodes_df)} nodes and {len(edges_df)} edges")

        if nodes_df.empty and edges_df.empty:
            logger.warning("No graph data found in response")

        # Return bound Plottable
        return (
            self.nodes(nodes_df, node='id')
            .edges(edges_df, source='source', destination='target')
        )

    def _extract_nodes(self, data: dict) -> pd.DataFrame:
        """Internal: Extract and deduplicate nodes from response"""
        nodes_list = []

        # Extract from Graph.Nodes section
        try:
            graph_nodes = data.get('Graph', {}).get('Nodes', [])
            for node in graph_nodes:
                if isinstance(node, dict):
                    nodes_list.append({
                        'id': node.get('Id'),
                        'label': node.get('Label', []),
                        'properties': node.get('Properties', {})
                    })
        except Exception as e:
            logger.warning(f"Failed to extract from Graph.Nodes: {e}")

        # Extract from RawData.Rows
        try:
            raw_rows = data.get('RawData', {}).get('Rows', [])
            for row in raw_rows:
                for col in row.get('Cols', []):
                    try:
                        value_str = col.get('Value', '{}')
                        value = json.loads(value_str) if isinstance(value_str, str) else value_str

                        # Node detection: has label/sys_label but not source/target edge fields
                        # Support both _label (original) and label/sys_label (Sentinel Graph API)
                        has_label = isinstance(value, dict) and (
                            '_label' in value or 'label' in value or 'sys_label' in value
                        )
                        is_edge = (
                            '_sourceId' in value or 'sys_sourceId' in value or
                            '_targetId' in value or 'sys_targetId' in value
                        ) if isinstance(value, dict) else False

                        if has_label and not is_edge:
                            # Start with all properties from the value
                            node_data = {k: v for k, v in value.items() if v is not None}
                            # Normalize key fields
                            node_data['id'] = value.get('_id') or value.get('id') or value.get('sys_id')
                            node_data['label'] = value.get('_label') or value.get('label')
                            if node_data.get('id'):  # Must have ID
                                nodes_list.append(node_data)
                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.debug(f"Skipping unparseable column value: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Failed to extract from RawData.Rows: {e}")

        # Create DataFrame and deduplicate
        if not nodes_list:
            logger.debug("No nodes found in response")
            return pd.DataFrame(columns=['id', 'label'])

        nodes_df = pd.DataFrame(nodes_list)

        if 'id' in nodes_df.columns and not nodes_df['id'].isna().all():
            # Keep row with most information (most non-null values)
            nodes_df['_info_count'] = nodes_df.notna().sum(axis=1)
            nodes_df = nodes_df.sort_values('_info_count', ascending=False)
            nodes_df = nodes_df.drop_duplicates(subset='id', keep='first')
            nodes_df = nodes_df.drop('_info_count', axis=1)

        return nodes_df.reset_index(drop=True)

    def _extract_edges(self, data: dict) -> pd.DataFrame:
        """Internal: Extract edges from response"""
        edges_list = []

        # Extract from Graph.Edges section
        try:
            graph_edges = data.get('Graph', {}).get('Edges', [])
            for edge in graph_edges:
                if isinstance(edge, dict):
                    edges_list.append(edge)
        except Exception as e:
            logger.warning(f"Failed to extract from Graph.Edges: {e}")

        # Extract from RawData.Rows
        try:
            raw_rows = data.get('RawData', {}).get('Rows', [])
            for row in raw_rows:
                for col in row.get('Cols', []):
                    try:
                        value_str = col.get('Value', '{}')
                        value = json.loads(value_str) if isinstance(value_str, str) else value_str

                        # Edge detection: has source/target IDs
                        # Support both _sourceId/_targetId (original) and sys_sourceId/sys_targetId (Sentinel Graph API)
                        has_source = isinstance(value, dict) and (
                            '_sourceId' in value or 'sys_sourceId' in value
                        )
                        has_target = isinstance(value, dict) and (
                            '_targetId' in value or 'sys_targetId' in value
                        )

                        if has_source and has_target:
                            # Start with all properties from the value
                            edge_data = {k: v for k, v in value.items() if v is not None}
                            # Normalize key fields
                            edge_data['source'] = value.get('_sourceId') or value.get('sys_sourceId')
                            edge_data['target'] = value.get('_targetId') or value.get('sys_targetId')
                            edge_data['edge'] = value.get('_label') or value.get('type') or value.get('sys_label')
                            if edge_data.get('source') and edge_data.get('target'):  # Must have source/target
                                edges_list.append(edge_data)
                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.debug(f"Skipping unparseable column value: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Failed to extract from RawData.Rows: {e}")

        if not edges_list:
            logger.debug("No edges found in response")
            return pd.DataFrame(columns=['source', 'target'])

        return pd.DataFrame(edges_list).reset_index(drop=True)

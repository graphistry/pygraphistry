import time
import pandas as pd
from typing import Any, List, Optional, TYPE_CHECKING, Union, overload, Literal

if TYPE_CHECKING:
    from azure.kusto.data import KustoClient
else:
    KustoClient = Any

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from graphistry.plugins_types.kusto_types import KustoConfig, KustoConnectionError, KustoQueryResult

logger = setup_logger(__name__)


class KustoMixin(Plottable):
    """
    KustoMixin is a Graphistry Mixin that allows you to plot data from Kusto.
    """

    def configure_kusto(
        self,
        cluster: str,
        database: str = "NetDefaultDB",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Plottable:
        """Configure Azure Data Explorer (Kusto) connection settings.
        
        Sets up the connection parameters for accessing a Kusto cluster.
        Authentication can be done via service principal (client_id, client_secret, tenant_id)
        or managed identity (omit authentication parameters).
        
        :param cluster: Kusto cluster URL (e.g., 'https://mycluster.westus2.kusto.windows.net')
        :type cluster: str
        :param database: Database name (defaults to 'NetDefaultDB')
        :type database: str
        :param client_id: Azure AD application (client) ID for service principal auth
        :type client_id: Optional[str]
        :param client_secret: Azure AD application secret for service principal auth
        :type client_secret: Optional[str]
        :param tenant_id: Azure AD tenant ID for service principal auth
        :type tenant_id: Optional[str]
        :returns: Self for method chaining
        :rtype: Plottable
        
        **Example: Service principal authentication**
            ::
            
                import graphistry
                g = graphistry.configure_kusto(
                    cluster="https://mycluster.westus2.kusto.windows.net",
                    database="SecurityDatabase",
                    client_id="your-client-id",
                    client_secret="your-client-secret",
                    tenant_id="your-tenant-id"
                )
                
        **Example: Managed identity authentication**
            ::
            
                import graphistry
                g = graphistry.configure_kusto(
                    cluster="https://mycluster.westus2.kusto.windows.net",
                    database="SecurityDatabase"
                    # No auth params - uses managed identity
                )
        """
        self.session.kusto = KustoConfig(
            cluster=cluster,
            database=database,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
        )
        return self
    

    def kusto_from_client(self, client: KustoClient, database: str = "NetDefaultDB") -> Plottable:
        """Configure Kusto using an existing client connection.
        
        Use this method when you already have a configured Kusto client connection
        and want to reuse it with Graphistry.
        
        :param client: Pre-configured Kusto client
        :type client: azure.kusto.data.KustoClient
        :param database: Database name to query against
        :type database: str
        :returns: Self for method chaining
        :rtype: Plottable
        
        **Example**
            ::
            
                from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
                import graphistry
                
                # Create Kusto client
                kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(
                    "https://mycluster.kusto.windows.net"
                )
                kusto_client = KustoClient(kcsb)
                
                # Use with Graphistry
                g = graphistry.kusto_from_client(kusto_client, "MyDatabase")
        """
        # Don't close the client if it's the same one
        if self.session.kusto is not None and client is not self.session.kusto._client:
            self.kusto_close()
        self.session.kusto = KustoConfig(
            cluster="unkown cluster kusto_from_client",
            database=database,
            _client=client,
        )
        return self

    
    @property
    def _kusto_config(self) -> KustoConfig:
        if self.session.kusto is None:
            raise ValueError("KustoMixin is not configured")
        return self.session.kusto
    
    @property
    def kusto_client(self) -> KustoClient:
        if self._kusto_config._client is not None:
            return self._kusto_config._client
        client = init_kusto_client(self._kusto_config)
        self._kusto_config._client = client
        return client

    def kusto_close(self) -> None:
        """Close the active Kusto client connection.
        
        Properly closes the underlying Kusto client connection to free resources.
        This should be called when you're done using the Kusto connection.
        
        **Example**
            ::
            
                import graphistry
                g = graphistry.configure_kusto(...)
                # ... perform queries ...
                g.kusto_close()  # Clean up connection
        """
        if self.session.kusto is None:
            return
        if self.session.kusto._client is not None:
            self.session.kusto._client.close()
        self.session.kusto._client = None


    # ---- Query API ---------------------------------------------------- #

    def kusto_health_check(self) -> None:
        """Perform a health check on the Kusto connection.
        
        Executes a simple query (.show tables) to verify that the connection
        to the Kusto cluster is working properly.
        
        :raises RuntimeError: If the connection test fails
        
        **Example**
            ::
            
                import graphistry
                g = graphistry.configure_kusto(...)
                g.kusto_health_check()  # Verify connection works
        """
        self.kusto_client.execute(self._kusto_config.database, ".show tables")

    @overload
    def kql(
        self,
        query: str,
        *,
        unwrap_nested: Optional[bool] = None,
        single_table: Literal[True] = True
    ) -> List[pd.DataFrame]:
        ...
    
    @overload
    def kql(
        self,
        query: str,
        *,
        unwrap_nested: Optional[bool] = None,
        single_table: Literal[False]
    ) -> pd.DataFrame:
        ...
    
    @overload
    def kql(
        self,
        query: str,
        *,
        unwrap_nested: Optional[bool] = None,
        single_table: bool = True
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        ...
    
    def kql(
        self,
        query: str,
        *,
        unwrap_nested: Optional[bool] = None,
        single_table: bool = True
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Execute KQL query and return result tables as DataFrames.
        
        Submits a Kusto Query Language (KQL) query to Azure Data Explorer and returns
        the results. By default, expects a single table result and returns it as a DataFrame.
        If multiple tables are returned, only the first is returned with a warning.
        Set single_table=False to always get a list of all result tables.
        
        :param query: KQL query string to execute
        :type query: str
        :param unwrap_nested: Strategy for handling nested/dynamic columns
        :type unwrap_nested: Optional[bool]
        :param single_table: If True, return single DataFrame (first table if multiple); if False, return list
        :type single_table: bool
        :returns: Single DataFrame if single_table=True, else list of DataFrames
        :rtype: Union[pd.DataFrame, List[pd.DataFrame]]
        
        **unwrap_nested semantics:**
        
        - **True**: Always attempt to unwrap nested columns; raise on failure
        - **None**: Use heuristic - unwrap if the first result looks nested
        - **False**: Never attempt to unwrap nested columns
        
        **Example: Basic security query (single table mode)**
            ::
            
                import graphistry
                g = graphistry.configure_kusto(...)
                
                query = '''
                SecurityEvent
                | where TimeGenerated > ago(1d)
                | where EventID == 4624  // Successful logon
                | project TimeGenerated, Account, Computer, IpAddress
                | take 1000
                '''
                
                # Single table mode returns DataFrame directly (default)
                df = g.kql(query)
                print(f"Found {len(df)} logon events")
                
        **Example: Get all tables as list**
            ::
            
                # Always get a list of all tables
                dfs = g.kql(query, single_table=False)
                df = dfs[0]
                
        **Example: Multi-table query**
            ::
            
                query = '''
                SecurityEvent | take 10;
                Heartbeat | take 5
                '''
                
                # With single_table=True (default), returns first table with warning
                df = g.kql(query)  # Returns SecurityEvent data, warns about multiple tables
                
                # With single_table=False, returns all tables
                frames = g.kql(query, single_table=False)
                security_df = frames[0]
                heartbeat_df = frames[1]
        """
        results = self._kql(query)
        if not results:
            if single_table:
                raise ValueError("Query returned no results")
            return []

        dfs: List[pd.DataFrame] = []

        for result in results:
            do_unwrap = (unwrap_nested is True or (unwrap_nested is None and _should_unwrap(result)))

            if do_unwrap:
                try:
                    frames_od = _unwrap_nested(result)
                    dfs.append(frames_od)
                    continue
                except Exception as exc:
                    if unwrap_nested is True:
                        raise RuntimeError(f"_unwrap_nested failed: {exc}") from exc
                    # Heuristic miss – fall back silently to flat table
                    pass

            # Default: flat table
            if not result.column_names:
                # safety 
                dfs.append(pd.DataFrame(result.data))
            else:
                dfs.append(pd.DataFrame(result.data, columns=result.column_names))

        # Auto-unbox single table result if requested
        if single_table:
            if len(dfs) > 1:
                logger.warning("Query returned multiple tables, returning first table")
            return dfs[0]
            
        return dfs


    def kusto_graph(self, graph_name: str, snap_name: Optional[str] = None) -> Plottable:
        """Fetch a Kusto graph entity as a Graphistry visualization object.
        
        Retrieves a named graph entity (and optional snapshot) from Kusto using
        the graph() operator and graph-to-table transformation. The result is
        automatically bound as nodes and edges for visualization.
        
        :param graph_name: Name of the Kusto graph entity to fetch
        :type graph_name: str
        :param snap_name: Optional snapshot/version identifier
        :type snap_name: Optional[str]
        :returns: Plottable object ready for visualization or further transforms
        :rtype: Plottable
        
        **Example: Basic graph visualization**
            ::
            
                import graphistry
                g = graphistry.configure_kusto(...)
                
                # Fetch and visualize a named graph
                graph_viz = g.kusto_graph("NetworkTopology")
                graph_viz.plot()
                
        **Example: Specific snapshot**
            ::
            
                # Fetch a specific snapshot of the graph
                graph_viz = g.kusto_graph("NetworkTopology", "2023-12-01")
                graph_viz.plot()
        """
        if snap_name:
            graph_query = f'graph("{graph_name}", "{snap_name}") | graph-to-table nodes as N with_node_id=g_NodeId, edges as E with_source_id=g_src with_target_id=g_dst; N;E'
        else:
            graph_query = f'graph("{graph_name}") | graph-to-table nodes as N with_node_id=g_NodeId, edges as E with_source_id=g_src with_target_id=g_dst; N;E'
        results = self._kql(graph_query)
        if len(results) != 2:
            raise ValueError(f"Expected 2 results, got {len(results)}")
        nodes = pd.DataFrame(results[0].data, columns=results[0].column_names)
        edges = pd.DataFrame(results[1].data, columns=results[1].column_names)
        return self.nodes(nodes, node='g_NodeId').edges(edges, source='g_src', destination='g_dst')  # type: ignore


    def _kql(self, query: str) -> List[KustoQueryResult]:
        """Execute KQL query and return raw results.
        
        Internal method for executing KQL queries and returning raw Kusto
        query results without DataFrame conversion.
        
        :param query: KQL query string to execute
        :type query: str
        :returns: List of raw query results
        :rtype: List[KustoQueryResult]
        :raises RuntimeError: If the query execution fails
        """
        from azure.kusto.data.exceptions import KustoServiceError
        logger.debug(f"KustoMixin._query(): {query}")

        try:
            start = time.time()
            response = self.kusto_client.execute(self._kusto_config.database, query)

            results = []
            row_lengths = []
            for result in response.primary_results:
                rows = [list(r) for r in result.rows]
                col_names = [col.column_name for col in result.columns]
                col_types = [col.column_type for col in result.columns]
                results.append(KustoQueryResult(rows, col_names, col_types))
                row_lengths.append((len(rows), len(col_names)))

            logger.info(f"Query returned {len(results)} results shapes: {row_lengths} in {time.time() - start:.3f} sec")
            print(f"Query returned {len(results)} results shapes: {row_lengths} in {time.time() - start:.3f} sec")
            return results

        except KustoServiceError as e:
            logger.error(f"Kusto query failed: {e}")
            raise RuntimeError(f"Kusto query failed: {e}")


def init_kusto_client(cfg: KustoConfig) -> "KustoClient":
    from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
    try:
        assert cfg.cluster is not None, "config.cluster is not set"
        if cfg.client_id is not None and cfg.client_secret is not None and cfg.tenant_id is not None:
            kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
                cfg.cluster,
                cfg.client_id,
                cfg.client_secret,
                cfg.tenant_id,
            )
            logger.info("Connecting to Kusto cluster %s", cfg.cluster)
        else:
            kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cfg.cluster)
            logger.info("Interactive login to Kusto cluster %s", cfg.cluster)

        client = KustoClient(kcsb)
    except Exception as exc:
        raise KustoConnectionError(f"Failed to connect to Kusto cluster: {exc}") from exc

    return client



# Kusto Utils
def _is_dynamic(val: Any) -> bool:
    """ADT check for Kusto 'dynamic' JSON values."""
    return isinstance(val, (dict, list))


# Core transformer
def _unwrap_nested(result: "KustoQueryResult") -> pd.DataFrame:
    """
    Transform one Kusto result whose columns contain *dynamic* objects

    - dict          -> dot-flattened
    - list[dict]    -> explode + flatten 
    - list[scalar]  -> keep as-is
    """

    df = pd.DataFrame(result.data, columns=result.column_names)
    if not result.column_types:
        return df

    for col, col_type in zip(result.column_names, result.column_types):
        if col_type.lower() != "dynamic":
            continue

        list_of_dicts = df[col].apply(
            lambda v: isinstance(v, list) and (not v or all(isinstance(x, dict) for x in v))
        )
        if list_of_dicts.any():
            df[col] = df[col].where(list_of_dicts,
                                    df[col].apply(lambda x: [x]))
            df = df.explode(col, ignore_index=True)

        # flatten dict rows
        dict_rows: pd.Series = df[col].apply(lambda v: isinstance(v, dict))
        if dict_rows.any():
            flat = pd.json_normalize(df.loc[dict_rows, col].tolist(), sep='.').add_prefix(f"{col}.")
            flat.index = df.loc[dict_rows].index
            df = df.join(flat, how='left')
            df[col] = df[col].mask(dict_rows, pd.NA)

        if df[col].isna().all():
            df = df.drop(columns=[col])

    df = df.astype(object).where(pd.notna(df), None)
    return df.reset_index(drop=True)


# Heuristic for ``unwrap_nested is None``
def _should_unwrap(result: "KustoQueryResult", sample_rows: int = 5) -> bool:
    """
    Decide whether result *looks* like it contains nested/dynamic columns.
    Strategy:
      1. Prefer explicit type info (column_type == 'dynamic') if present.
      2. Otherwise inspect up to `sample_rows` rows for dict / list values.
    """
    try:
        if any(c.lower() == "dynamic" for c in result.column_types):
            return True
    except AttributeError:
        pass  # `.column_type` not available in older SDK versions.

    for col_idx in range(len(result.column_names)):
        sample = (row[col_idx] for row in result.data[:sample_rows])
        if any(_is_dynamic(v) for v in sample):
            return True
    return False

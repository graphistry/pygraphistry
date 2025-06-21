import time
import pandas as pd
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from azure.kusto.data import KustoClient
else:
    KustoClient = Any

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from graphistry.plugins_types.kusto_types import KustoConfig, KustoConnectionError, KustoQueryResult, KustoSession
from graphistry.client_session import ClientSession

logger = setup_logger(__name__)


class KustoGraph:
    """
    KustoGraph is a Graphistry plugin that allows you to plot data from Kusto.

    Usable stand-alone *or* as a cooperative mix-in on the Plottable.
    """

    _kusto_session: KustoSession

    def __init__(
        self,
        *args,
        **kwargs: Any,
    ) -> None:
        # NOTE: Cooperative Mixin initialization passes args and kwargs along
        super().__init__(*args, **kwargs)

        session = getattr(kwargs.get('pygraphistry_session', {}), '_session', None)
        self._kusto_session = session.kusto if isinstance(session, ClientSession) else KustoSession()

    def from_kusto_client(self, client: KustoClient, database: str) -> 'KustoGraph':
        self._kusto_session.client = client
        self._kusto_session.database = database
        return self

    def kusto_connect(self, config: Optional[KustoConfig] = None) -> 'KustoGraph':
        """
        Connect to a Kusto cluster.
        
        KustoConfig
            - "cluster": The Kusto cluster name.
            - "database": The Kusto database name.
          For AAD authentication:
            - "client_id": The Kusto client ID.
            - "client_secret": The Kusto client secret.
            - "tenant_id": The Kusto tenant ID.
          Otherwise: process will use web browser to authenticate.


        :param config: A dictionary containing the Kusto configuration.
        :type  config: KustoConfig
        :returns: KustoGraph with a Kusto connection
        :rtype:  KustoGraph
        """
        if config is not None:
            # Write back to the session, may be shared
            self._kusto_session.config = config
            self._kusto_session.client = None
            self._kusto_session.database = None

        _ = self.kusto_client  # trigger initialization
        return self

    @property
    def kusto_client(self) -> KustoClient:
        if self._kusto_session.client is not None:
            return self._kusto_session.client
        client = init_kusto_client(self._kusto_session)
        self._kusto_session.client = client
        return client

    def kusto_close(self) -> None:
        self.kusto_client.close()
        self._kusto_session.client = None
        self._kusto_session.database = None
    

    @property
    def kusto_database(self) -> str:
        if self._kusto_session.database is not None:
            return self._kusto_session.database
        if self._kusto_session.config is not None:
            return self._kusto_session.config['database']
        raise ValueError("KustoGraph requires a database to be set")
    
    @kusto_database.setter
    def kusto_database(self, database: str) -> None:
        self._kusto_session.database = database

    # ---- query api ---------------------------------------------------- #

    def kql(
        self,
        query: str,
        *,
        unwrap_nested: Optional[bool] = True
    ) -> List[pd.DataFrame]:
        """
        Submit a Kusto/Azure Data Explorer *query* and return result tables.
        Because a Kusto request may emit multiple tables, a **list of
        DataFrames** is always returned; most queries yield a single entry.

        unwrap_nested semantics
        -----------------------
        • True   - Always attempt to unwrap; raise on failure.
        • None   - Use heuristic: unwrap iff the first result *looks* nested.
        • False  - Never attempt to unwrap.

        :param query: Kusto query string
        :type  query: str
        :param unwrap_nested: flatten strategy above
        :type  unwrap_nested: bool | None
        :returns: list of Pandas DataFrames
        :rtype:  List[pd.DataFrame]

        **Example**
            ::
                frames = graphistry.kusto_query("StormEvents | take 100")
                df = frames[0]
        """
        results = self._kql(query)
        if not results:
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

        return dfs


    def kql_graph(self, graph_name: str, snap_name: Optional[str] = None) -> Plottable:
        """
        Fetch a Kusto *graph* (and optional *snapshot*) as a Graphistry object.
        Under the hood: `graph(..)` + `graph-to-table` to pull **nodes** and
        **edges**, then binds them to *self*.

        :param graph_name: name of Kusto graph entity
        :type  graph_name: str
        :param snap_name: optional snapshot/version
        :type  snap_name: str | None
        :returns: Plottable ready for `.plot()` or further transforms
        :rtype:  Plottable
        
        **Example**
            ::
                g = graphistry.kusto_query_graph("HoneypotNetwork").plot()
        """
        from ..plotter import Plotter
        g = self if isinstance(self, Plottable) else Plotter()  # type: ignore

        if snap_name:
            graph_query = f'graph("{graph_name}", "{snap_name}" | graph-to-table nodes as N with_node_id=NodeId, edges as E with_source_id=src with_target_id=dst; N;E'
        else:
            graph_query = f'graph("{graph_name}") | graph-to-table nodes as N with_node_id=NodeId, edges as E with_source_id=src with_target_id=dst; N;E'
        results = self._kql(graph_query)
        if len(results) != 2:
            raise ValueError(f"Expected 2 results, got {len(results)}")
        nodes = pd.DataFrame(results[0].data, columns=results[0].column_names)
        edges = pd.DataFrame(results[1].data, columns=results[1].column_names)
        return g.nodes(nodes, node='NodeId').edges(edges, source='src', destination='dst')  # type: ignore


    def _kql(self, query: str) -> List[KustoQueryResult]:
        from azure.kusto.data.exceptions import KustoServiceError
        logger.debug(f"KustoGraph._query(): {query}")

        try:
            start = time.time()
            response = self.kusto_client.execute(self.kusto_database, query)

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


def init_kusto_client(session: KustoSession) -> "KustoClient":
    if session.config is None:
        raise ValueError("Kusto initialization requires a kusto_config")
    cfg = session.config

    from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
    try:
        cluster = cfg["cluster"]
    except KeyError as e:
        raise ValueError(f"Missing required kusto_config key: '{e}'") from e

    try:
        client_id = cfg.get("client_id")
        client_secret = cfg.get("client_secret")
        tenant_id = cfg.get("tenant_id")

        if client_id is not None and client_secret is not None and tenant_id is not None:
            kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
                cluster,
                client_id,
                client_secret,
                tenant_id,
            )
            logger.info("Connecting to Kusto cluster %s", cluster)
        else:
            kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster)
            logger.info("Interactive login to Kusto cluster %s", cluster)

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
        dict_rows = df[col].apply(lambda v: isinstance(v, dict))
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

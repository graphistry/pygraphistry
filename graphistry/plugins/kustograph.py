import time
import pandas as pd
from typing import Any, List, Mapping, Dict, Iterable, Tuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from azure.kusto.data import KustoClient
else:
    KustoClient = Any

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from graphistry.pygraphistry import PyGraphistry
from graphistry.plugins_types.kusto_types import KustoConfig, KustoConnectionError, KustoQueryResult

logger = setup_logger(__name__)





class KustoGraph:
    def __init__(self, client: KustoClient, database: str):
        self.client = client
        self.database = database

    @classmethod
    def from_config(cls, cfg: KustoConfig) -> "KustoGraph":
        from azure.kusto.data import KustoClient, KustoConnectionStringBuilder

        try:
            cluster = cfg["cluster"]
            database = cfg["database"]
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
            else:
                kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster)

            logger.info("Connecting to Kusto cluster %s", cluster)
            client = KustoClient(kcsb)
        except Exception as exc:
            raise KustoConnectionError(f"Failed to connect to Kusto cluster: {exc}") from exc

        return cls(client, database=database)
    

    def close(self) -> None:
        self.client.close()


    def _query(self, query: str) -> List[KustoQueryResult]:
        from azure.kusto.data.exceptions import KustoServiceError
        logger.debug(f"KustoGraph._query(): {query}")

        try:
            start = time.time()
            response = self.client.execute(self.database, query)

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


    def query(
        self,
        query: str,
        *,
        unwrap_nested: Optional[bool] = True
    ) -> List[pd.DataFrame]:
        """
        Run *query* and return a list of DataFrames.

        unwrap_nested semantics
        -----------------------
        • True   - Always attempt to unwrap; raise on failure.  
        • None   - Use heuristic: unwrap iff the first result *looks* nested.
        • False  - Never attempt to unwrap.  
        """
        results = self._query(query)
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


    def query_graph(self, graph_name: str, snap_name: Optional[str] = None, g: Plottable = PyGraphistry.bind()) -> Plottable:
        if snap_name:
            graph_query = f'graph("{graph_name}", "{snap_name}" | graph-to-table nodes as N with_node_id=NodeId, edges as E with_source_id=src with_target_id=dst; N;E'
        else:
            graph_query = f'graph("{graph_name}") | graph-to-table nodes as N with_node_id=NodeId, edges as E with_source_id=src with_target_id=dst; N;E'
        results = self._query(graph_query)
        if len(results) != 2:
            raise ValueError(f"Expected 2 results, got {len(results)}")
        nodes = pd.DataFrame(results[0].data, columns=results[0].column_names)
        edges = pd.DataFrame(results[1].data, columns=results[1].column_names)
        return g.nodes(nodes, node='NodeId').edges(edges, source='src', destination='dst')


# Kusto Utils
def _is_dynamic(val: Any) -> bool:
    """ADT check for Kusto 'dynamic' JSON values."""
    return isinstance(val, (dict, list))

def _normalize(records: Iterable[Mapping], prefix: str = "") -> pd.DataFrame:
    """json_normalize + dedup + stable column ordering, with parent prefix."""
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize([dict(r) for r in records], sep='.')
    if prefix:
        df = df.add_prefix(f"{prefix}.")
    return df.loc[:, sorted(df.columns)].drop_duplicates().reset_index(drop=True)


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


class KustoGraphContext:
    def __init__(self, config: Union[KustoConfig, None] = None):
        config = config or PyGraphistry._config.get("kusto")
        if not config:
            raise ValueError("Missing kusto_config. Register globally with kusto_config or use with_kusto().")
        self.kusto_graph = KustoGraph.from_config(config)

    def __enter__(self):
        return self.kusto_graph

    def __exit__(self, exc_type, exc_value, traceback):
        self.kusto_graph.close()

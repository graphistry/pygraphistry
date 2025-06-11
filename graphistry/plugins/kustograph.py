import pandas as pd
import time
from typing import Any, List, TYPE_CHECKING
from collections import OrderedDict
from typing import Mapping, List, Dict, Any, Iterable, Tuple, Optional

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
            client_id, client_secret, tenant_id = cfg.get("client_id"), cfg.get("client_secret"), cfg.get("tenant_id")
            if all((client_id, client_secret, tenant_id)):
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
                rows = result.rows
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
        unwrap_nested: bool | None = True
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

        if unwrap_nested is False:
            return [pd.DataFrame(r.data, columns=r.column_names) for r in results]

        first = results[0]
        do_unwrap = (
            unwrap_nested is True or
            (unwrap_nested is None and _should_unwrap(first))
        )

        if not do_unwrap:
            return [pd.DataFrame(r.data, columns=r.column_names) for r in results]

        try:
            frames_od = _unwrap_nested(first)
            return list(frames_od.values())
        except Exception as exc:
            if unwrap_nested is True:
                raise RuntimeError(f"_unwrap_nested failed: {exc}") from exc
            # Heuristic miss – fall back silently
            return [pd.DataFrame(r.data, columns=r.column_names) for r in results]


    def query_graph(self, graph_name: str, snap_name: str | None = None, g: Plottable = PyGraphistry.bind()) -> Plottable:
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



# ================================================================
# kusto_graph.py  –  new imports
# ================================================================
from collections import OrderedDict
from typing import Any, Iterable, Mapping, List, Dict, OrderedDict as OD
import pandas as pd


# ================================================================
# Low‑level utilities
# ================================================================
def _is_dynamic(val: Any) -> bool:
    """ADT check for Kusto 'dynamic' JSON values."""
    return isinstance(val, (dict, list))

def _normalize(records: Iterable[Mapping]) -> pd.DataFrame:
    """json_normalize + dedup + stable column ordering."""
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(list(records))
    return df.loc[:, sorted(df.columns)].drop_duplicates().reset_index(drop=True)


# ================================================================
# Core transformer
# ================================================================
def _unwrap_nested(result: "KustoQueryResult") -> "OrderedDict[str, pd.DataFrame]":
    """
    Transform one Kusto result whose columns contain *dynamic* objects
    (typical from `graph-match … project a, edge, v`) into
    OrderedDict[col_name -> DataFrame].
    """
    frames: "OD[str, pd.DataFrame]" = OrderedDict()

    for col_idx, col_name in enumerate(result.column_names):
        col_vals = [row[col_idx] for row in result.data if row[col_idx] is not None]

        # Non‑dynamic column → trivial DF
        if not col_vals or not _is_dynamic(col_vals[0]):
            frames[col_name] = pd.DataFrame({col_name: col_vals})
            continue

        # Dynamic column → flatten each JSON object
        recs: List[Mapping] = []
        for cell in col_vals:
            if isinstance(cell, list):
                recs.extend(cell)     # explode lists
            else:
                recs.append(cell)
        frames[col_name] = _normalize(recs)

    return frames


# ================================================================
# Heuristic for ``unwrap_nested is None``
# ================================================================
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
    def __init__(self, config: KustoConfig | None = None):
        config = config or PyGraphistry._config.get("kusto")
        if not config:
            raise ValueError("Missing kusto_config. Register globally with kusto_config or use with_kusto().")
        self.kusto_graph = KustoGraph.from_config(config)

    def __enter__(self):
        return self.kusto_graph

    def __exit__(self, exc_type, exc_value, traceback):
        self.kusto_graph.close()


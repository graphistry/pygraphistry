import pandas as pd
import time
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from typing_extensions import TypedDict, NotRequired

if TYPE_CHECKING:
    from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
    from azure.kusto.data.exceptions import KustoServiceError
else:
    KustoClient = Any
    KustoConnectionStringBuilder = Any
    KustoServiceError = Any

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger

logger = setup_logger(__name__)


class KustoConnectionError(Exception):
    pass


class KustoQueryResult:
    def __init__(self, data: List[List[Any]], column_names: List[str]):
        self.data = data
        self.column_names = column_names


class KustoConfig(TypedDict):
    cluster: str
    database: NotRequired[str]
    client_id: NotRequired[str]
    client_secret: NotRequired[str]
    tenant_id: NotRequired[str]


class KustoGraph:
    def __init__(self, client: KustoClient, database: str | None = None):
        self.client = client
        self.database = database

    @classmethod
    def from_config(cls, cfg: KustoConfig) -> "KustoGraph":
        from azure.kusto.data import KustoClient, KustoConnectionStringBuilder

        try:
            cluster = cfg["cluster"]
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

        return cls(client, database=cfg.get("database"))


    def query(self, query: str, database: str | None = None) -> KustoQueryResult:
        from azure.kusto.data.exceptions import KustoServiceError

        logger.debug(f"KustoGraph execute_query(): {query}")
        database = database or self.database
        if not database:
            raise ValueError("Database not specified and not set in KustoGraph")

        try:
            start = time.time()
            response = self.client.execute(database, query)

            rows = list(response.primary_results[0])
            col_names = [col.column_name for col in response.primary_results[0].columns]

            logger.info(f"Query returned {len(rows)} rows in {time.time() - start:.3f} sec")
            return KustoQueryResult(rows, col_names)

        except KustoServiceError as e:
            logger.error(f"Kusto query failed: {e}")
            raise RuntimeError(f"Kusto query failed: {e}")

    def query_to_df(self, query: str) -> pd.DataFrame:
        result = self.query(query)
        return pd.DataFrame(result.data, columns=result.column_names)

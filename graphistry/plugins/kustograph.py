import pandas as pd
import time
import json
from typing import Any, Dict, List, Optional

from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError

from graphistry.Plottable import Plottable
from graphistry.util import setup_logger

logger = setup_logger(__name__)


class KustoConnectionError(Exception):
    pass


class KustoQueryResult:
    def __init__(self, data: List[List[Any]], column_names: List[str]):
        self.data = data
        self.column_names = column_names


class KustoGraph:

    def __init__(self, g: Plottable, kusto_config: Dict[str, str]):
        required_keys = ["cluster", "database"]
        for key in required_keys:
            if not kusto_config.get(key):
                raise ValueError(f"Missing required kusto_config key: '{key}'")

        self.cluster = kusto_config["cluster"]
        self.database = kusto_config["database"]

        if all(kusto_config.get(k) for k in ["client_id", "client_secret", "tenant_id"]):
            self.credential_mode = "aad_app"
            self.client_id = kusto_config["client_id"]
            self.client_secret = kusto_config["client_secret"]
            self.tenant_id = kusto_config["tenant_id"]
        else:
            self.credential_mode = "device_code"

        self.client = self._connect()

    def _connect(self) -> KustoClient:
        try:
            if self.credential_mode == "aad_app":
                kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
                    self.cluster,
                    self.client_id,
                    self.client_secret,
                    self.tenant_id,
                )
            else:
                kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(self.cluster)

            logger.info(f"Connecting to Kusto cluster: {self.cluster}")
            return KustoClient(kcsb)

        except Exception as e:
            raise KustoConnectionError(f"Failed to connect to Kusto cluster: {e}")

    def execute_query(self, query: str) -> KustoQueryResult:
        logger.debug(f"KustoGraph execute_query(): {query}")
        try:
            start = time.time()
            response = self.client.execute(self.database, query)

            rows = list(response.primary_results[0])
            col_names = [col.column_name for col in response.primary_results[0].columns]

            logger.info(f"Query returned {len(rows)} rows in {time.time() - start:.3f} sec")
            return KustoQueryResult(rows, col_names)

        except KustoServiceError as e:
            logger.error(f"Kusto query failed: {e}")
            raise RuntimeError(f"Kusto query failed: {e}")

    def query_to_df(self, query: str) -> pd.DataFrame:
        result = self.execute_query(query)
        return pd.DataFrame(result.data, columns=result.column_names)

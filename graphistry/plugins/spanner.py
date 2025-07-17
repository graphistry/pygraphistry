# graphistry/plugins/spannergraph.py
import json, time
from typing import Any, List, Dict, Optional, TYPE_CHECKING

import pandas as pd
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from graphistry.plugins_types.spanner_types import (
    SpannerConfig,
    SpannerConnectionError,
    SpannerQueryResult,
)

if TYPE_CHECKING:
    from google.cloud.spanner_dbapi.connection import Connection
else:
    Connection = Any

logger = setup_logger(__name__)

class SpannerMixin(Plottable):
    """
    SpannerMixin is a Graphistry Mixin that allows you to plot data from Spanner.
    """

    def configure_spanner(self, instance_id: str, database_id: str, project_id: Optional[str] = None, credentials_file: Optional[str] = None) -> Plottable:
        """Configure Google Cloud Spanner connection settings.
        
        Sets up the connection parameters for accessing a Spanner database instance.
        Either project_id or credentials_file must be provided for authentication.
        
        :param instance_id: The Spanner instance identifier
        :type instance_id: str
        :param database_id: The Spanner database identifier  
        :type database_id: str
        :param project_id: Google Cloud project ID (optional if using credentials_file)
        :type project_id: Optional[str]
        :param credentials_file: Path to service account credentials JSON file
        :type credentials_file: Optional[str]
        :returns: Self for method chaining
        :rtype: Plottable
        :raises ValueError: If neither credentials_file nor project_id is provided
        
        **Example: Using project ID**
            ::
            
                import graphistry
                g = graphistry.configure_spanner(
                    project_id="my-project",
                    instance_id="my-instance", 
                    database_id="my-database"
                )
                
        **Example: Using service account credentials**
            ::
            
                import graphistry
                g = graphistry.configure_spanner(
                    instance_id="my-instance",
                    database_id="my-database",
                    credentials_file="/path/to/credentials.json"
                )
        """
        if credentials_file is None and project_id is None:
            raise ValueError("Either credentials_file or project_id must be set")
        self.session.spanner = SpannerConfig(project_id, instance_id, database_id, credentials_file)
        self.session.spanner.validate()
        return self
    
    def spanner_from_client(self, client: Connection) -> Plottable:
        """Configure Spanner using an existing client connection.
        
        Use this method when you already have a configured Spanner client connection
        and want to reuse it with Graphistry.
        
        :param client: Pre-configured Spanner database connection
        :type client: google.cloud.spanner_dbapi.connection.Connection
        :returns: Self for method chaining
        :rtype: Plottable
        
        **Example**
            ::
            
                from google.cloud import spanner
                import graphistry
                
                # Create Spanner client
                spanner_client = spanner.Client(project="my-project")
                instance = spanner_client.instance("my-instance")
                database = instance.database("my-database")
                
                # Use with Graphistry
                g = graphistry.spanner_from_client(database)
        """
        # Don't close the client if it's the same one
        if self.session.spanner is not None and client is not self.session.spanner._client:
            self.spanner_close()
        self.session.spanner = SpannerConfig(_client=client)
        self.session.spanner.validate()
        return self
    
    @property
    def spanner_config(self) -> SpannerConfig:
        if self.session.spanner is None:
            raise ValueError("SpannerMixin is not configured")
        return self.session.spanner
    
    @property
    def spanner_client(self) -> Connection:
        if self.spanner_config._client is not None:
            return self.spanner_config._client
        client = init_spanner_client(self.spanner_config)
        self.spanner_config._client = client
        return client

    def spanner_close(self) -> None:
        """Close the active Spanner database connection.
        
        Properly closes the underlying Spanner client connection to free resources.
        This should be called when you're done using the Spanner connection.
        
        **Example**
            ::
            
                import graphistry
                g = graphistry.configure_spanner(...)
                # ... perform queries ...
                g.spanner_close()  # Clean up connection
        """
        if self.session.spanner is None:
            return
        if self.session.spanner._client is not None:
            self.session.spanner._client.close()
        self.session.spanner._client = None


    # ---- Query API ---------------------------------------------------- #

    def _spanner_gql(self, query: str) -> SpannerQueryResult:
        """Execute a GQL query on the Spanner database.
        
        Internal method for executing Graph Query Language (GQL) queries on
        the configured Spanner database. Returns raw query results.
        
        :param query: The GQL query to execute
        :type query: str
        :returns: The results of the query execution
        :rtype: SpannerQueryResult
        :raises RuntimeError: If the query execution fails
        """
        logger.debug(f' SpannerMixin execute_query() query:{query}\n')

        try:
            start_time = time.time()
            cursor = self.spanner_client.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description or []]  # extract column names
            logger.debug(f'column names returned from query: {column_names}')
            execution_time_s = time.time() - start_time
            logger.info(f"Query completed in {execution_time_s:.3f} seconds.")
            return SpannerQueryResult(results, column_names)
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}") from e


    # ---- Static Helpers ------------------------------------------------ #

    @staticmethod
    def convert_spanner_json(data: List[Any]) -> List[Dict[str, Any]]:
        """Convert Spanner JSON query results to structured graph data.
        
        Transforms raw Spanner JSON query results into a standardized format
        with separate nodes and edges arrays for graph processing.
        
        :param data: Raw JSON data from Spanner query results
        :type data: List[Any]
        :returns: Structured graph data with 'nodes' and 'edges' arrays
        :rtype: List[Dict[str, Any]]
        """
        from google.cloud.spanner_v1.data_types import JsonObject
        json_list = []
        for item in data:
            for elements in item:
                json_entry: Dict[str, List] = {"nodes": [], "edges": []}
                for element in elements:
                    element_dict_list = json.loads(element.serialize()) if isinstance(element, JsonObject) else element
                    for element_dict in element_dict_list:
                        if element_dict.get('kind') == 'node':
                            labels = element_dict.get('labels', [])
                            for label in labels:
                                node_data = {
                                    "label": label,
                                    "identifier": element_dict.get('identifier'),
                                    "properties": element_dict.get('properties', {})
                                }
                                json_entry["nodes"].append(node_data)
                        elif element_dict.get('kind') == 'edge':
                            labels = element_dict.get('labels', [])
                            for label in labels:
                                edge_data = {
                                    "label": label,
                                    "identifier": element_dict.get('identifier'),
                                    "source": element_dict.get('source_node_identifier'),
                                    "destination": element_dict.get('destination_node_identifier'),
                                    "properties": element_dict.get('properties')
                                }
                                json_entry["edges"].append(edge_data)
                if json_entry["nodes"] or json_entry["edges"]:  # only add non-empty entries
                    json_list.append(json_entry)
        return json_list

    @staticmethod
    def add_type_from_label_to_df(df: pd.DataFrame) -> pd.DataFrame:
        """Add 'type' column from 'label' for Graphistry type handling.
        
        Creates a 'type' column from the 'label' column for proper visualization
        in Graphistry. If a 'type' column already exists, it is renamed to 'type_'
        before creating the new 'type' column.
        
        :param df: DataFrame containing node or edge data with 'label' column
        :type df: pd.DataFrame
        :returns: Modified DataFrame with the updated 'type' column
        :rtype: pd.DataFrame
        """

        # rename 'type' to 'type_' if it exists
        if "type" in df.columns:
            df.rename(columns={"type": "type_"}, inplace=True)
            logger.info("'type' column renamed to 'type_'")

        # check if 'label' column exists before assigning it to 'type'
        if "label" in df.columns:
            df["type"] = df["label"]
        else:
            # assign None value if 'label' is missing
            df["type"] = None  
            logger.warn("'label' column missing, 'type' set to None")

        return df

    @staticmethod
    def get_nodes_df(json_data: list) -> pd.DataFrame:
        """Convert Spanner JSON nodes into a pandas DataFrame.
        
        Extracts node data from structured JSON results and creates a DataFrame
        with columns for label, identifier, and all node properties.
        
        :param json_data: Structured JSON data containing graph nodes
        :type json_data: list
        :returns: DataFrame containing node data with properties as columns
        :rtype: pd.DataFrame
        """
        nodes = [
            { 
                "label": node.get("label"), 
                "identifier": node["identifier"], 
                **node.get("properties", {})
            }
            for entry in json_data
            for node in entry.get("nodes", [])
        ]
        nodes_df = pd.DataFrame(nodes).drop_duplicates()

        return SpannerMixin.add_type_from_label_to_df(nodes_df)

    @staticmethod
    def get_edges_df(json_data: list) -> pd.DataFrame:
        """Convert Spanner JSON edges into a pandas DataFrame.
        
        Extracts edge data from structured JSON results and creates a DataFrame
        with columns for label, identifier, source, destination, and all edge properties.
        
        :param json_data: Structured JSON data containing graph edges
        :type json_data: list
        :returns: DataFrame containing edge data with properties as columns
        :rtype: pd.DataFrame
        """
        edges = [
            {
                "label": edge.get("label"),
                "identifier": edge["identifier"],
                "source": edge["source"],
                "destination": edge["destination"],
                **edge.get("properties", {})
            }
            for entry in json_data
            for edge in entry.get("edges", [])
        ]
        edges_df = pd.DataFrame(edges).drop_duplicates()

        return SpannerMixin.add_type_from_label_to_df(edges_df)


    # ---- Ergonomic API ------------------------------------------------ #

    def spanner_gql(self, query: str) -> Plottable:
        """Execute GQL path query and return graph visualization.
        
        Executes a Graph Query Language (GQL) path query on the configured Spanner
        database and returns a Plottable object ready for visualization. The query
        must return path data using SAFE_TO_JSON(p) format.
        
        :param query: GQL path query string with SAFE_TO_JSON(path) format
        :type query: str
        :returns: Plottable object with nodes and edges populated from query results
        :rtype: Plottable
        
        **Example: Basic path query**
            ::
            
                import graphistry
                graphistry.configure_spanner(
                    project_id="my-project",
                    instance_id="my-instance", 
                    database_id="my-database"
                )
                
                query = '''
                GRAPH FinGraph
                MATCH p = (a:Account)-[t:Transfers]->(b:Account)
                LIMIT 10000
                RETURN SAFE_TO_JSON(p) as path
                '''
                
                g = graphistry.spanner_gql(query)
                g.plot()
        """
        query_result = self._spanner_gql(query)

        # convert json result set to a list 
        query_result_list = [ query_result.data ]

        json_data = self.convert_spanner_json(query_result_list)

        nodes_df = self.get_nodes_df(json_data)
        edges_df = self.get_edges_df(json_data)

        # TODO(tcook): add more error handling here if nodes or edges are empty
        return self.nodes(nodes_df, 'identifier').edges(edges_df, 'source', 'destination')  # type: ignore

    def spanner_gql_to_df(self, query: str) -> pd.DataFrame:
        """Execute GQL/SQL query and return results as DataFrame.
        
        Executes a Graph Query Language (GQL) or SQL query on the configured Spanner
        database and returns the results as a pandas DataFrame. This method is suitable
        for tabular queries that don't require graph visualization.
        
        :param query: GQL or SQL query string
        :type query: str
        :returns: DataFrame containing query results with column names
        :rtype: pd.DataFrame
        
        **Example: Aggregation query**
            ::
            
                import graphistry
                graphistry.configure_spanner(
                    project_id="my-project",
                    instance_id="my-instance", 
                    database_id="my-database"
                )
                
                query = '''
                GRAPH FinGraph
                MATCH (p:Person)-[:Owns]-(:Account)->(l:Loan)
                RETURN p.id as PersonID, p.name AS Name, 
                       SUM(l.loan_amount) AS TotalBorrowed
                ORDER BY TotalBorrowed DESC
                LIMIT 10
                '''
                
                df = graphistry.spanner_gql_to_df(query)
                print(df.head())
                
        **Example: SQL query**
            ::
            
                query = "SELECT * FROM Account WHERE type = 'checking' LIMIT 1000"
                df = graphistry.spanner_gql_to_df(query)
        """
        query_result = self._spanner_gql(query)

        # create DataFrame from json results, adding column names
        return pd.DataFrame(query_result.data, columns=query_result.column_names)

def init_spanner_client(cfg: SpannerConfig) -> "Connection":
    """
    Lazily establish a DB-API connection using the parameters in `session.config`.
    """
    from google.cloud.spanner_dbapi.connection import connect
    try:
        if cfg.instance_id is None or cfg.database_id is None:
            raise ValueError("Missing required configuration: instance_id & database_id")
        if cfg.credentials_file:
            # creds controls project implicitly
            logger.info("Connecting to Spanner instance %s", cfg.instance_id)
            conn = connect(cfg.instance_id, cfg.database_id, credentials=cfg.credentials_file)
        else:
            if not cfg.project_id:
                raise ValueError("Need either `project_id` or `credentials_file` in SpannerConfig")
            logger.info("Interactive login to Spanner instance %s", cfg.instance_id)
            conn = connect(cfg.project_id, cfg.instance_id, cfg.database_id)

        conn.read_only = True
        return conn
    except Exception as exc:
        raise SpannerConnectionError(f"Failed to connect to Spanner: {exc}") from exc

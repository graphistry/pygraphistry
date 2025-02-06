import os
import pandas as pd
import json
import time
from typing import Any, List, Dict, Optional

from graphistry.Plottable import Plottable

from graphistry.util import setup_logger
logger = setup_logger(__name__)


class SpannerConnectionError(Exception):
    """Custom exception for errors related to Spanner connection."""
    pass

class SpannerQueryResult:
    """
    Encapsulates the results of a query, including metadata.

    :ivar list data: The raw query results.
    """

    def __init__(self, data: List[Any], column_names: List[str]):
        """
        Initializes a SpannerQueryResult instance.

        :param data: The raw query results.
        :type List[Any]
        :param column_names: a list of the column names from the cursor, defaults to None 
        :type: List[str]
        """
        self.data = data
        self.column_names = column_names


class SpannerGraph:
    """
    A comprehensive interface for interacting with Google Spanner Graph databases.

    :ivar str project_id: The Google Cloud project ID.
    :ivar str instance_id: The Spanner instance ID.
    :ivar str database_id: The Spanner database ID.
    :ivar Any connection: The active connection to the Spanner database.
    :ivar Any graphistry: The Graphistry parent object.
    """

    def __init__(self, g: Plottable, spanner_config: Dict[str, str]):
        """
        Initializes the SpannerGraph instance.

        :param graphistry: The Graphistry parent object.
        :param project_id: The Google Cloud project ID.
        :param instance_id: The Spanner instance ID.
        :param database_id: The Spanner database ID.
        """
        
        # check if valid 
        required_keys = ["project_id", "instance_id", "database_id"]
        for key in required_keys:
            value = spanner_config.get(key)
            if not value:  # check for None or empty values
                raise ValueError(f"Missing or invalid value for required Spanner configuration: '{key}'")

        self.project_id = spanner_config["project_id"]
        self.instance_id = spanner_config["instance_id"]
        self.database_id = spanner_config["database_id"]
    
        if spanner_config.get("credentials_file"):
            self.credentials_file = spanner_config["credentials_file"]
   
        self.connection = self.__connect()

    def __connect(self) -> Any:
        """
        Establishes a connection to the Spanner database.

        :return: A connection object to the Spanner database.
        :rtype: google.cloud.spanner_dbapi.connection
        :raises SpannerConnectionError: If the connection to Spanner fails.
        """
        from google.cloud.spanner_dbapi.connection import connect

        try:
            if hasattr(self, 'credentials_file') and self.credentials_file is not None:
                
                connection = connect(self.instance_id, self.database_id, credentials=self.credentials_file)
            else:
                connection = connect(self.instance_id, self.database_id)

            connection.autocommit = True
            logger.info("Connected to Spanner database.")
            return connection
        except Exception as e:
            raise SpannerConnectionError(f"Failed to connect to Spanner: {e}")

    def close_connection(self) -> None:
        """
        Closes the connection to the Spanner database.
        """
        if self.connection:
            self.connection.close()
            logger.info("Connection to Spanner database closed.")

    def execute_query(self, query: str) -> SpannerQueryResult:
        """
        Executes a GQL query on the Spanner database.

        :param query: The GQL query to execute
        :type str
        :return: The results of the query execution.
        :rtype: SpannerQueryResult
        :raises RuntimeError: If the query execution fails.
        """
        logger.debug(f' SpannerGraph execute_query() query:{query}\n')

        try:
            start_time = time.time()
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]  # extract column names
            logger.debug(f'column names returned from query: {column_names}')
            execution_time_s = time.time() - start_time
            logger.info(f"Query completed in {execution_time_s:.3f} seconds.")
            return SpannerQueryResult(results, column_names)
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}")

    @staticmethod
    def convert_spanner_json(data: List[Any]) -> List[Dict[str, Any]]:
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
        """
        Modify input DataFrame creating a 'type' column is created from 'label' for proper type handling in Graphistry
        If a 'type' column already exists, it is renamed to 'type_' before creating the new 'type' column.

        :param df: DataFrame containing node or edge data
        :type df: pd.DataFrame

        :return: Modified DataFrame with the updated 'type' column.
        :rtype: query: pd.DataFrame

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
        """
        Converts spanner json nodes into a pandas DataFrame.
    
        :param json_data: The structured JSON data containing graph nodes.
        :return: A DataFrame containing node data from Spanner, col names will match node properties.
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

        return SpannerGraph.add_type_from_label_to_df(nodes_df)


    
    @staticmethod
    def get_edges_df(json_data: list) -> pd.DataFrame:
        """
        Converts spanner json edges into a pandas DataFrame

        :param json_data: The structured JSON data containing graph edges.
        :type list 
        :return: A DataFrame containing edge data from Spanner, col names will match edge properties.
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

        return SpannerGraph.add_type_from_label_to_df(edges_df)


    def gql_to_graph(self, res: Plottable, query: str) -> Plottable:
        """
        Executes a query and constructs a Graphistry graph from the results.

        :param query: The GQL query to execute.
        :return: A Graphistry graph object constructed from the query results.
        :rtype: Plottable 
        """
        query_result = self.execute_query(query)

        # convert json result set to a list 
        query_result_list = [ query_result.data ]

        json_data = self.convert_spanner_json(query_result_list)

        nodes_df = self.get_nodes_df(json_data)
        edges_df = self.get_edges_df(json_data)

        # TODO(tcook): add more error handling here if nodes or edges are empty
        return res.nodes(nodes_df, 'identifier').edges(edges_df, 'source', 'destination')

    def query_to_df(self, query: str) -> pd.DataFrame:
        """
        Executes a query and returns a pandas dataframe of results

        :param query: The query to execute.
        :return: pandas dataframe of the query results 
        :rtype: pd.DataFrame
        """
        query_result = self.execute_query(query)

        # create DataFrame from json results, adding column names
        return pd.DataFrame(query_result.data, columns=query_result.column_names)

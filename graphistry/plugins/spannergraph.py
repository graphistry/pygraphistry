import os
import pandas as pd
import json
import time
from typing import Any, List, Dict

from graphistry.util import setup_logger
logger = setup_logger(__name__)

import logging
logging.basicConfig(level=logging.INFO)

from google.cloud.spanner_v1.data_types import JsonObject

class SpannerConnectionError(Exception):
    """Custom exception for errors related to Spanner connection."""
    pass

class SpannerQueryResult:
    """
    Encapsulates the results of a query, including metadata.

    :ivar list data: The raw query results.
    :ivar float execution_time: The time taken to execute the query.
    :ivar int record_count: The number of records returned.
    """

    def __init__(self, data: List[Any], execution_time: float):
        """
        Initializes a SpannerQueryResult instance.

        :param data: The raw query results.
        :param execution_time: The time taken to execute the query.
        """
        self.data = data
        self.execution_time = execution_time
        self.record_count = len(data)

    def summary(self) -> Dict[str, Any]:
        """
        Provides a summary of the query execution.

        :return: A summary of the query results.
        """
        return {
            "execution_time": self.execution_time,
            "record_count": self.record_count
        }


class SpannerGraph:
    """
    A comprehensive interface for interacting with Google Spanner Graph databases.

    :ivar str project_id: The Google Cloud project ID.
    :ivar str instance_id: The Spanner instance ID.
    :ivar str database_id: The Spanner database ID.
    :ivar Any connection: The active connection to the Spanner database.
    :ivar Any graphistry: The Graphistry parent object.
    """

    def __init__(self, graphistry: Any, project_id: str, instance_id: str, database_id: str):
        """
        Initializes the SpannerGraph instance.

        :param graphistry: The Graphistry parent object.
        :param project_id: The Google Cloud project ID.
        :param instance_id: The Spanner instance ID.
        :param database_id: The Spanner database ID.
        """
        self.graphistry = graphistry
        self.project_id = project_id
        self.instance_id = instance_id
        self.database_id = database_id
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
            connection = connect(self.instance_id, self.database_id)
            connection.autocommit = True
            logging.info("Connected to Spanner database.")
            return connection
        except Exception as e:
            raise SpannerConnectionError(f"Failed to connect to Spanner: {e}")

    def close_connection(self) -> None:
        """
        Closes the connection to the Spanner database.
        """
        if self.connection:
            self.connection.close()
            logging.info("Connection to Spanner database closed.")

    def execute_query(self, query: str) -> SpannerQueryResult:
        """
        Executes a GQL query on the Spanner database.

        :param query: The GQL query to execute.
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
            execution_time = time.time() - start_time
            logging.info(f"Query executed in {execution_time:.4f} seconds.")
            return SpannerQueryResult(results, execution_time)
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {e}")

    @staticmethod
    def convert_spanner_json(data):
        from google.cloud.spanner_v1.data_types import JsonObject
        json_list = []
        for item in data:
            for elements in item:
                json_entry = {"nodes": [], "edges": []}
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
    def get_nodes_df(json_data: list) -> pd.DataFrame:
        """
        Converts spanner json nodes into a pandas DataFrame.
    
        :param json_data: The structured JSON data containing graph nodes.
        :return: A DataFrame containing node information
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

        # if 'type' property exists, skip setting and warn
        if "type" not in nodes_df.columns:
            # check 'label' column exists before assigning it to 'type'
            if "label" in nodes_df.columns:
                nodes_df['type'] = nodes_df['label']
            else:
                nodes_df['type'] = None  # Assign a default value if 'label' is missing
        else: 
            logger.warn("unable to assign 'type' from label, column exists\n")
        
        return nodes_df

    @staticmethod
    def get_edges_df(json_data: list) -> pd.DataFrame:
        """
        Converts spanner json edges into a pandas DataFrame

        :param json_data: The structured JSON data containing graph edges.
        :return: A DataFrame containing edge information.
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

        # if 'type' property exists, skip setting and warn
        if "type" not in edges_df.columns:
            # check 'label' column exists before assigning it to 'type'
            if "label" in edges_df.columns:
                edges_df['type'] = edges_df['label']
            else:
                edges_df['type'] = None  # Assign a default value if 'label' is missing
        else: 
            logger.warn("unable to assign 'type' from label, column exists\n")

        return edges_df

    def gql_to_graph(self, query: str) -> Any:
        """
        Executes a query and constructs a Graphistry graph from the results.

        :param query: The GQL query to execute.
        :return: A Graphistry graph object constructed from the query results.
        """
        query_result = self.execute_query(query)
        # convert json result set to a list 
        query_result_list = [ query_result.data ]
        json_data = self.convert_spanner_json(query_result_list)
        nodes_df = self.get_nodes_df(json_data)
        edges_df = self.get_edges_df(json_data)
        # TODO(tcook): add more error handling here if nodes or edges are empty
        g = self.graphistry.nodes(nodes_df, 'identifier').edges(edges_df, 'source', 'destination')
        return g

    # TODO(tcook): add wrapper funcs in PlotterBase for these utility functions: 
    
    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Retrieves the schema of the Spanner database.

        :return: A dictionary containing table names and column details.
        """
        schema = {}
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT table_name, column_name, spanner_type FROM information_schema.columns")
            for row in cursor.fetchall():
                table_name, column_name, spanner_type = row
                if table_name not in schema:
                    schema[table_name] = []
                schema[table_name].append({"column_name": column_name, "type": spanner_type})
            logging.info("Database schema retrieved successfully.")
        except Exception as e:
            logging.error(f"Failed to retrieve schema: {e}")
        return schema

    def validate_data(self, data: Dict[str, List[Dict[str, Any]]], schema: Dict[str, List[Dict[str, str]]]) -> bool:
        """
        Validates input data against the database schema.

        :param data: The data to validate.
        :param schema: The schema of the database.
        :return: True if the data is valid, False otherwise.
        """
        for table, columns in data.items():
            if table not in schema:
                logging.error(f"Table {table} does not exist in schema.")
                return False
            for record in columns:
                for key in record.keys():
                    if key not in [col["column_name"] for col in schema[table]]:
                        logging.error(f"Column {key} is not valid for table {table}.")
                        return False
        logging.info("Data validation passed.")
        return True

    def dump_config(self) -> Dict[str, str]:
        """
        Returns the current configuration of the SpannerGraph instance.

        :return: A dictionary containing configuration details.
        """
        return {
            "project_id": self.project_id,
            "instance_id": self.instance_id,
            "database_id": self.database_id
        }

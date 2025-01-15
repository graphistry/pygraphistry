import os
import pandas as pd
import json
import time
import logging
from typing import Any, List, Dict

# logging.basicConfig(level=logging.INFO)

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
        :raises SpannerConnectionError: If the connection to Spanner fails.
        """
        try:
            from google.cloud.spanner_dbapi.connection import connect  # Lazy import
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
        :raises RuntimeError: If the query execution fails.
        """
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
    def parse_spanner_json(query_result: SpannerQueryResult) -> List[Dict[str, Any]]:
        """
        Converts Spanner JSON graph data into structured Python objects.

        :param query_result: The results of the executed query.
        :return: A list of dictionaries containing nodes and edges.
        """
        from google.cloud.spanner_v1.data_types import JsonObject  # Lazy import
        data = [query_result.data]
        json_list = []
        for record in data:
            for item in record:
                json_entry = {"nodes": [], "edges": []}
                elements = json.loads(item.serialize()) if isinstance(item, JsonObject) else item
                for element in elements:
                    if element.get('kind') == 'node':
                        for label in element.get('labels', []):
                            json_entry["nodes"].append({
                                "label": label,
                                "identifier": element.get('identifier'),
                                "properties": element.get('properties', {})
                            })
                    elif element.get('kind') == 'edge':
                        for label in element.get('labels', []):
                            json_entry["edges"].append({
                                "label": label,
                                "identifier": element.get('identifier'),
                                "source": element.get('source_node_identifier'),
                                "destination": element.get('destination_node_identifier'),
                                "properties": element.get('properties', {})
                            })
                if json_entry["nodes"] or json_entry["edges"]:
                    json_list.append(json_entry)
        return json_list

    @staticmethod
    def get_nodes_df(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts graph nodes into a pandas DataFrame.

        :param json_data: The structured JSON data containing graph nodes.
        :return: A DataFrame containing node information.
        """
        nodes = [
            {"label": node["label"], "identifier": node["identifier"], **node["properties"]}
            for entry in json_data
            for node in entry["nodes"]
        ]
        nodes_df = pd.DataFrame(nodes).drop_duplicates()
        nodes_df['type'] = nodes_df['label']
        return nodes_df

    @staticmethod
    def get_edges_df(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts graph edges into a pandas DataFrame.

        :param json_data: The structured JSON data containing graph edges.
        :return: A DataFrame containing edge information.
        """
        edges = [
            {
                "label": edge["label"],
                "identifier": edge["identifier"],
                "source": edge["source"],
                "destination": edge["destination"],
                **edge["properties"]
            }
            for entry in json_data
            for edge in entry["edges"]
        ]
        edges_df = pd.DataFrame(edges).drop_duplicates()
        edges_df['type'] = edges_df['label']
        return edges_df

    def gql_to_graph(self, query: str) -> Any:
        """
        Executes a query and constructs a Graphistry graph from the results.

        :param query: The GQL query to execute.
        :return: A Graphistry graph object constructed from the query results.
        """
        query_result = self.execute_query(query)
        json_data = self.parse_spanner_json(query_result)
        nodes_df = self.get_nodes_df(json_data)
        edges_df = self.get_edges_df(json_data)
        g = self.graphistry.nodes(nodes_df, 'identifier').edges(edges_df, 'source', 'destination')
        return g

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

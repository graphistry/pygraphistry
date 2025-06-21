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
    SpannerSession,
)

if TYPE_CHECKING:
    from google.cloud.spanner_dbapi.connection import Connection
else:
    Connection = Any

logger = setup_logger(__name__)

class SpannerGraph:
    """
    SpannerGraph is a Graphistry plugin that allows you to plot data from Spanner.

    Usable stand-alone *or* as a cooperative mix-in on the Plottable.
    """

    _spanner_session: SpannerSession

    def __init__(
        self,
        *args,
        spanner_session: Optional[SpannerSession] = None,
        **kwargs: Any,
    ) -> None:
        # NOTE: Cooperative Mixin initialization passes args and kwargs along
        kwargs["spanner_session"] = spanner_session
        super().__init__(*args, **kwargs)
        self._spanner_session = spanner_session or SpannerSession()

    def from_spanner_client(self, client: Connection, database: str) -> "SpannerGraph":
        self._spanner_session.client = client
        self._spanner_session.database = database
        return self

    def spanner_connect(self, config: Optional[SpannerConfig] = None) -> "SpannerGraph":
        if config is not None:
            # Write back to the session, may be shared
            self._spanner_session.config = config
            self._spanner_session.client = None
            self._spanner_session.database = None

        _ = self.spanner_client  # trigger initialization
        return self

    @property
    def spanner_client(self) -> Connection:
        if self._spanner_session.client:
            return self._spanner_session.client
        client = init_spanner_client(self._spanner_session)
        self._spanner_session.client = client
        return client

    def spanner_close(self) -> None:
        self.spanner_client.close()
        self._spanner_session.client = None
        self._spanner_session.database = None

    # ---- Query API ---------------------------------------------------- #

    def _gql(self, query: str) -> SpannerQueryResult:
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


    # ---- Ergonomic API ------------------------------------------------ #

    def gql_to_graph(self, query: str) -> Plottable:
        """
        Submit query to google spanner database and return a df of the results 
        
        query can be SQL or GQL as long as table of results are returned 
        query='SELECT * from Account limit 10000'
        :param query: query string 
        :type query: Str
        :returns: Pandas DataFrame with the results of query
        :rtype: pd.DataFrame

        **Example: calling spanner_gql_to_graph
                ::
                    import graphistry
                    # credentials_file is optional, all others are required
                    SPANNER_CONF = { "project_id":  PROJECT_ID,                 
                                     "instance_id": INSTANCE_ID, 
                                     "database_id": DATABASE_ID, 
                                     "credentials_file": CREDENTIALS_FILE }
                    graphistry.register(..., spanner_config=SPANNER_CONF)
                    query='SELECT * from Account limit 10000'
                    df = graphistry.spanner_gql_to_df(query)
                    g.plot()
     
        """
        from graphistry.pygraphistry import PyGraphistry
        g = self if isinstance(self, Plottable) else PyGraphistry.bind()
        query_result = self._gql(query)

        # convert json result set to a list 
        query_result_list = [ query_result.data ]

        json_data = self.convert_spanner_json(query_result_list)

        nodes_df = self.get_nodes_df(json_data)
        edges_df = self.get_edges_df(json_data)

        # TODO(tcook): add more error handling here if nodes or edges are empty
        return g.nodes(nodes_df, 'identifier').edges(edges_df, 'source', 'destination')

    def gql_to_df(self, query: str) -> pd.DataFrame:
        """
        Submit GQL query to google spanner graph database and return Plottable with nodes and edges populated  
        
        GQL must be a path query with a syntax similar to the following, it's recommended to return the path with
        SAFE_TO_JSON(p), TO_JSON() can also be used, but not recommend. LIMIT is optional, but for large graphs with millions
        of edges or more, it's best to filter either in the query or use LIMIT so as not to exhaust GPU memory.  
        query=f'''GRAPH my_graph
        MATCH p = (a)-[b]->(c) LIMIT 100000 return SAFE_TO_JSON(p) as path'''
        :param query: GQL query string 
        :type query: Str
        :returns: Plottable with the results of GQL query as a graph
        :rtype: Plottable

        **Example: calling spanner_gql
                ::
                    import graphistry
                    # credentials_file is optional, all others are required
                    SPANNER_CONF = { "project_id":  PROJECT_ID,                 
                                     "instance_id": INSTANCE_ID, 
                                     "database_id": DATABASE_ID, 
                                     "credentials_file": CREDENTIALS_FILE }
                    graphistry.register(..., spanner_config=SPANNER_CONF)
                    query=f'''GRAPH my_graph
                    MATCH p = (a)-[b]->(c) LIMIT 100000 return SAFE_TO_JSON(p) as path'''
                    g = graphistry.spanner_gql(query)
                    g.plot()
     
        """
        query_result = self._gql(query)

        # create DataFrame from json results, adding column names
        return pd.DataFrame(query_result.data, columns=query_result.column_names)

def init_spanner_client(session: SpannerSession) -> "Connection":
    """
    Lazily establish a DB-API connection using the parameters in `session.config`.
    """
    if not session.config:
        raise SpannerConnectionError("Missing Spanner config. Call `.spanner_connect()` first.")

    from google.cloud.spanner_dbapi.connection import connect

    cfg = session.config
    try:
        creds = cfg.get("credentials_file")
        if creds:
            # creds controls project implicitly
            logger.info("Connecting to Spanner instance %s", cfg["instance_id"])
            conn = connect(cfg["instance_id"], cfg["database_id"], credentials=creds)
        else:
            project = cfg.get("project_id")
            if not project:
                raise ValueError("Need either `project_id` or `credentials_file` in SpannerConfig")
            logger.info("Interactive login to Spanner instance %s", cfg["instance_id"])
            conn = connect(project, cfg["instance_id"], cfg["database_id"])

        conn.read_only = True
        return conn
    except Exception as exc:
        raise SpannerConnectionError(f"Failed to connect to Spanner: {exc}") from exc

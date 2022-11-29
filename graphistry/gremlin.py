from typing import Any, Callable, Iterable, List, Optional, Set, Union, TYPE_CHECKING
import os, pandas as pd
from .Plottable import Plottable
try:
    from gremlin_python.driver.client import Client
    from gremlin_python.driver.resultset import ResultSet
    from gremlin_python.driver.serializer import GraphSONSerializersV2d0
    from gremlin_python.structure.graph import Vertex, Edge, Path
except:
    Client = Any
    ResultSet = Any
    GraphSONSerializersV2d0 = Any
    Vertex = Any
    Edge = Any
    Path = Any
    1

from .util import setup_logger
logger = setup_logger(__name__)


if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object


def clean_str(v):
    if isinstance(v, str):
        return v.replace('"', r'\"')
    return str(v)


# ####

def ensure_imports():
    try:
        from gremlin_python.driver.client import Client
    except Exception as e:
        logger.warning('Could not import gremlin_python; try pip install --user graphistry[gremllin] or pip install --user gremlinpython')
        raise e

# ####

def node_to_query(
    d: dict, type_col: Optional[str] = None, untyped: bool = False, keys: List[str] = []
) -> str:
    """
    Convert node dictionary to gremlin node add string
    * Put remaining attributes as string-valued properties
    """

    if untyped:
        base = 'g.addV()'
    else:
        base = f'g.addV(\'{clean_str(d[type_col])}\')'

    for p in keys:
        if not pd.isna(d[p]):
            base += f'.property(\'{clean_str(p)}\', \'{clean_str(d[p])}\')'

    return base


def nodes_to_queries(g, type_col: Optional[str] = None, untyped: bool = False) -> Iterable[str]:
    """
    Convert graphistry object to stream of gremlin node add strings:
    * If type is None, try to default to fields  'category' or 'type'. Skip via untyped=True.
    * Put remaining attributes as string-valued properties
     """

    if g._nodes is None:
        raise ValueError('No nodes bound to graph yet, try calling g.nodes(df)')

    if (not untyped) and (type_col is None):
        if 'category' in g._nodes:
            type_col = 'category'
        elif 'type' in g._nodes:
            type_col = 'type'
        else:
            raise Exception('Must specify node type_col or provide node column category or type')

    skiplist = [type_col] if not untyped else []
    keys = [c for c in g._nodes if c not in skiplist]

    if type_col is not None:
        if type_col not in g._nodes:
            raise ValueError(f'type_col="{type_col}" specified yet not in data')

    return (node_to_query(row, type_col, untyped, keys) for index, row in g._nodes.iterrows())


# ####


def edge_to_query(d: dict, from_col: str, to_col: str, type_col: Optional[str] = None, untyped: bool = False) -> str:
    """
    Assumes properties from, to, type
    * If type is None, try to default to fields  'edgeType', 'category', or 'type'. Skip via untyped=True.
    """
    if (not untyped) and (type_col is None):
        if 'edgeType' in d:
            type_col = 'edgeType'
        elif 'category' in d:
            type_col = 'category'
        elif 'type' in d:
            type_col = 'type'
        else:
            raise Exception('Must specify edge type_col or provide edge column edgeType, category, or type')
    
    addE = f'addE(\'{clean_str(d[type_col])}\')' if not untyped else 'addE()'
    base = f'g.v(\'{clean_str(d[from_col])}\').{addE}.to(g.v(\'{clean_str(d[to_col])}\'))'
    if True:
        skiplist = [from_col, to_col, type_col]
        for p in d.keys():
            if not pd.isna(d[p]) and (p not in skiplist):
                base += f'.property(\'{clean_str(p)}\', \'{clean_str(d[p])}\')'
    return base


def edges_to_queries(g, type_col: Optional[str] = None, untyped: bool = False) -> Iterable[str]:
    """
    Return stream of edge add queries
    * If type is None, try to default to fields  'edgeType', 'category', or 'type'. Skip via untyped=True.
    """    

    if g._edges is None:
        raise ValueError('No edges bound to graph yet, try calling g.edges(df)')

    if type_col is not None:
        if type_col not in g._edges:
            raise ValueError(f'type_col="{type_col}" specified yet not in data')

    return (edge_to_query(row, g._source, g._destination, type_col, untyped) for index, row in g._edges.iterrows())


# ####


#https://github.com/graphistry/graph-app-kit/blob/master/src/python/neptune_helper/gremlin_helper.py
def flatten_vertex_dict(vertex: dict, id_col: str = 'id', label_col: str = 'label') -> dict:
    """
    Convert gremlin vertex (in dict form) to flat dict appropriate for pandas
    - Metadata names take priority over property names
    - Remap: T.id, T.label -> id, label (Neptune)
    - Drop field 'type'
    - id_col and label_col define output column name for metadata entries
    """

    d = {}
    props = {}
    for k in vertex.keys():

        if k == 'type':
            continue

        if k == 'id' or k == 'T.id':
            d[id_col] = vertex[k]
            continue

        if k == 'label' or k == 'T.label':
            d[label_col] = vertex[label_col]
            continue

        v = vertex[k]
        if isinstance(v, list):
            d[str(k)] = v[0]
            continue

        if k == 'properties' and isinstance(v, dict):
            for prop_k in v:
                if prop_k == id_col:
                    continue
                v2 = v[prop_k]
                # TODO: multi-prop to list?
                if isinstance(v2, list) and (len(v2) == 1) and isinstance(v2[0], dict) and 'id' in v2[0] and 'value' in v2[0]:
                    props[str(prop_k)] = v2[0]['value']
                    continue
                props[str(prop_k)] = str(v2)
            continue

        d[str(k)] = vertex[k]

    if len(props.keys()) > 0:
        d = {**props, **d}

    return d

def flatten_vertex_dict_adder(
    nodes: List, nodes_hits: Set[str],
    item: dict, id_col: str = 'id', label_col: str = 'label'
) -> Optional[dict]:
    """
    Return item when added as fresh
    """
    id = None
    if 'T.id' in item:
        id = item['T.id']
    elif 'id' in item: 
        id = item['id']
    if id is None or (id not in nodes_hits):
        d = flatten_vertex_dict(item, id_col, label_col)
        nodes.append(d)
        if id is not None:
            nodes_hits.add(id)
        return d
    return None


#https://github.com/apache/tinkerpop/blob/master/gremlin-python/src/main/python/gremlin_python/structure/graph.py
def flatten_edge_structure(
    edge: Edge,
    src_col: str = 'src', dst_col: str = 'dst',
    label_col: str = 'label', id_col: str = 'id'
) -> dict:
    return {
        id_col: edge.id,
        label_col: edge.label,
        src_col: edge.inV.id,
        dst_col: edge.outV.id
    }

def flatten_edge_structure_adder(
    edges: List, edges_hits: Set[str],
    edge: Edge,
    src_col: str = 'src', dst_col: str = 'dst',
    label_col: str = 'label', id_col: str = 'id'
) -> Optional[dict]:
    """
    Return item when added as fresh
    """
    if edge.id in edges_hits:
        return None
    d = flatten_edge_structure(edge, src_col, dst_col, label_col, id_col)
    edges.append(d)
    edges_hits.add(edge.id)
    return d


#https://github.com/apache/tinkerpop/blob/master/gremlin-python/src/main/python/gremlin_python/structure/graph.py
def flatten_vertex_structure(vertex: Vertex, id_col: str = 'id', label_col: str = 'label') -> dict:
    return {
        id_col: vertex.id,
        label_col: vertex.label
    }

def flatten_vertex_structure_adder(
    nodes: List,
    nodes_hits: Set[str],
    vertex: Vertex, id_col: str = 'id', label_col: str = 'label'
) -> Optional[dict]:
    """
    Return item when added as fresh
    """
    if vertex.id in nodes_hits:
        return None
    d = flatten_vertex_structure(vertex, id_col, label_col)
    nodes.append(d)
    nodes_hits.add(vertex.id)
    return d


#https://github.com/graphistry/graph-app-kit/blob/master/src/python/neptune_helper/gremlin_helper.py
def flatten_edge_dict(edge, src_col: str = 'src', dst_col: str = 'dst'):
    """
    Convert gremlin vertex (in dict form) to flat dict appropriate for pandas
    - Metadata names take priority over property names
    - Remap: T.inV, T.outV -> inV, outV (Neptune)
    - Drop field 'type'
    - src_col and dst_col define output column name for metadata entries
    """

    d = {}
    props = {}
    for k in edge.keys():

        if k == 'type':
            continue

        if k == 'inV':
            d[src_col] = edge[k]
            continue

        if k == 'outV':
            d[dst_col] = edge[k]
            continue

        if k == 'IN' and isinstance(edge[k], dict):
            d[src_col] = edge[k]['id']
            continue

        if k == 'OUT' and isinstance(edge[k], dict):
            d[dst_col] = edge[k]['id']
            continue

        v = edge[k]
        if isinstance(v, list):
            d[str(k)] = v[0]
            continue

        if k == 'properties' and isinstance(v, dict):
            for prop_k in v:
                if prop_k == src_col or prop_k == dst_col:
                    continue
                v2 = v[prop_k]
                if isinstance(v2, list) and (len(v2) == 1) and isinstance(v2[0], dict) and 'id' in v2[0] and 'value' in v2[0]:
                    props[str(prop_k)] = v2[0]['value']
                    continue
                props[str(prop_k)] = str(v2)
            continue

        d[str(k)] = edge[k]


    if len(props.keys()) > 0:
        d = {**props, **d}

    return d

def flatten_edge_dict_adder(
    edges: List, edges_hits: Set[str],
    item: dict, src_col: str = 'src', dst_col: str = 'dst'
):
    """
    Return item when added as fresh
    """

    #Neptune elementMap(): skip for now as gives id/label but not props
    #if ('IN' in item) and isinstance(item['IN'], dict):
    #    flatten_vertex_dict_adder(nodes, nodes_hits, item['IN'])
    #if ('OUT' in item) and isinstance(item['OUT'], dict):
    #    flatten_vertex_dict_adder(nodes, nodes_hits, item['OUT'])

    id = None
    if 'T.id' in item: 
        id = item['T.id']
    elif 'id' in item:
        id = item['id']
    if id is None or (id not in edges_hits):
        d = flatten_edge_dict(item, src_col, dst_col)
        edges.append(d)
        if id is not None:
            edges_hits.add(id)
        return d
    return None

def resultset_to_g_structured_item(
    edges: List, edges_hits: Set[str],
    nodes: List, nodes_hits: Set[str],
    item, ignore_errors
) -> bool:
    """
    Return true if matched
    """
    if isinstance(item, Edge):
        flatten_edge_structure_adder(edges, edges_hits, item)
        flatten_vertex_structure_adder(nodes, nodes_hits, item.inV)
        flatten_vertex_structure_adder(nodes, nodes_hits, item.outV)
        return True

    if isinstance(item, Vertex):
        flatten_vertex_structure_adder(nodes, nodes_hits, item)
        return True
    
    if isinstance(item, Path):
        for path_obj in item.objects:
            if isinstance(path_obj, Edge):
                flatten_edge_structure_adder(edges, edges_hits, path_obj)
                flatten_vertex_structure_adder(nodes, nodes_hits, path_obj.inV)
                flatten_vertex_structure_adder(nodes, nodes_hits, path_obj.outV)
            elif isinstance(path_obj, Vertex):
                flatten_vertex_structure_adder(nodes, nodes_hits, path_obj)
            else:
                if ignore_errors:
                    logger.info('Supressing path error for step :: %s', type(path_obj), exc_info=True)
                raise ValueError('unexpected Path step:', path_obj)
        return True
    
    return False


DROP_QUERY = 'g.V().drop()'

class GremlinMixin(MIXIN_BASE):
    """
    Universal Gremlin<>pandas/graphistry functionality across Gremlin connectors
    
    Currently serializes queries as strings instead of bytecode in order to support cosmosdb
    """

    _reconnect_gremlin : Optional[Callable[['GremlinMixin'], 'GremlinMixin']] = None
    _gremlin_client : Optional[Client]

    def __init__(self, *args, gremlin_client: Optional[Client] = None, **kwargs):
        if gremlin_client is not None:
            self._gremlin_client = gremlin_client

    def gremlin_client(
        self,
        gremlin_client: Client
    ):
        """Pass in a generic gremlin python client

            **Example: Login and plot **
            ::

                import graphistry
                from gremlin_python.driver.client import Client

                my_gremlin_client = Client(
                f'wss://MY_ACCOUNT.gremlin.cosmosdb.azure.com:443/',
                'g', 
                username=f"/dbs/MY_DB/colls/{self.COSMOS_CONTAINER}",
                password=self.COSMOS_PRIMARY_KEY,
                message_serializer=GraphSONSerializersV2d0())

                (graphistry
                    .gremlin_client(my_gremlin_client)
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        """    

        self._gremlin_client = gremlin_client
        return self

    def connect(self):
        """
        Use previously provided credentials to connect. Disconnect any preexisting clients.
        """

        if self._reconnect_gremlin is None:
            raise ValueError('No gremlin client; either pass one in or use a built-in like cosmos')

        return self._reconnect_gremlin(self)

    def drop_graph(self):
        """
            Remove all graph nodes and edges from the database
        """
        self.gremlin_run(DROP_QUERY)  # .iterate() ? follow by g.tx().commit() ? 
        return self


    # Tutorial: 
    # https://itnext.io/getting-started-with-graph-databases-azure-cosmosdb-with-gremlin-api-and-python-80e57cbd1c5e
    def gremlin_run(self, queries: Iterable[str], throw=False) -> ResultSet:
        for query in queries:
            logger.debug('query: %s', query)
            try:
                if self._gremlin_client is None:
                    raise ValueError('Must first set a gremlin client')
                callback = self._gremlin_client.submitAsync(query)  # type: ignore
                if callback.result() is not None:
                    results = callback.result()
                    logger.debug('results: %s', results)
                    if results is not None:
                        logger.debug('Query succeeded: %s', type(results))
                    yield results
                else:
                    logger.error('Erroroneous result on query: %s', query)
                    if throw:
                        raise Exception(f'Unexpected erroroneous result on query: {query}')
                    yield Exception(f'Unexpected erroroneous result on query: {query}')
            except Exception as e:
                logger.error('Exception on query: %s', query, exc_info=True)
                logger.info('Resuming after caught exception...')
                if throw:
                    raise e
                yield e

       
    def gremlin(self, queries: Union[str, Iterable[str]]) -> Plottable:
        """
            Run one or more gremlin queries and get back the result as a graph object
            To support cosmosdb, sends as strings

            **Example: Login and plot **
            ::

                import graphistry
                (graphistry
                    .gremlin_client(my_gremlin_client)
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        """
        ensure_imports()
        if isinstance(queries, str):
            queries = [ queries ]
        resultsets = self.gremlin_run(queries, throw=True)
        logger.debug('resultsets: %s', resultsets)
        g = self.resultset_to_g(resultsets)
        return g


    def resultset_to_g(
        self,
        resultsets: Union[ResultSet,Iterable[ResultSet]],
        mode: str = 'infer',
        verbose=False,
        ignore_errors=False
    ) -> Plottable:
        """
        Convert traversal results to graphistry object with ._nodes, ._edges
        If only received nodes or edges, populate that field
        For custom src/dst/node bindings, passing in a Graphistry instance with .bind(source=.., destination=..., node=...)
        Otherwise, will do src/dst/id
        For dict results (ex: valueMap/elementMap), specify mode='nodes' ('edges'), else will inspect field 'type'
        """
        

        if isinstance(resultsets, ResultSet):
            resultsets = [resultsets]
        
        nodes_hits: Set[str] = set()
        nodes: List[dict] = []
        edges_hits: Set[str] = set()
        edges: List[dict] = []
        for resultset in resultsets:
            if verbose:
                logger.debug('resultset: %s :: %s', resultset, type(resultset))
            
            try:
                for result in resultset:
                    if verbose:
                        logger.debug('result: %s :: %s', result, type(result))
                    if isinstance(result, dict):
                        result = [ result ]
                    if resultset_to_g_structured_item(edges, edges_hits, nodes, nodes_hits, result, ignore_errors):
                        continue
                    for item in result:
                        if verbose:
                            logger.debug('item: %s :: %s', item, type(item))

                        if resultset_to_g_structured_item(edges, edges_hits, nodes, nodes_hits, item, ignore_errors):
                            continue
                        elif isinstance(item, dict):
                            if (mode != 'infer') or ('type' in item):
                                item_kind = None
                                if mode == 'infer':
                                    item_kind = item['type']
                                elif mode == 'nodes':
                                    item_kind = 'vertex'
                                elif mode == 'edges':
                                    item_kind = 'edge'

                                if item_kind == 'vertex':
                                    flatten_vertex_dict_adder(nodes, nodes_hits, item)
                                elif item_kind == 'edge':
                                    flatten_edge_dict_adder(edges, edges_hits, item)
                                else:
                                    raise ValueError('unexpected item type', item['type'])
                            else:
                                for k in item.keys():
                                    item_k_val = item[k]
                                    if item_k_val['type'] == 'vertex':
                                        flatten_vertex_dict_adder(nodes, nodes_hits, item_k_val)
                                    elif item_k_val['type'] == 'edge':
                                        flatten_edge_dict_adder(edges, edges_hits, item_k_val)
                                    else:                                
                                        raise ValueError('unexpected item key val:', type(item[k]))
                        else:
                            raise ValueError('unexpected non-dict item type:', type(item))

            except Exception as e:
                if ignore_errors:
                    logger.info('Supressing error', exc_info=True)
                else:
                    raise e

        nodes_df = pd.DataFrame(nodes) if len(nodes) > 0 else None
        edges_df = pd.DataFrame(edges) if len(edges) > 0 else None
        

        g = self.nodes(nodes_df)

        if len(edges) > 0 and edges_df is not None:
            g = g.edges(edges_df)
        #elif len(nodes) > 0:
        #    v0 = nodes[0][g._node]
        #    g = g.edges(pd.DataFrame({
        #        g._source: pd.Series([v0], dtype=nodes_df[g._node].dtype),  # type: ignore
        #        g._destination: pd.Series([v0], dtype=nodes_df[g._node].dtype)  # type: ignore
        #    }))
        elif g._edges is None:
            g = g.edges(pd.DataFrame({
                g._source: pd.Series([], dtype='object'),
                g._destination: pd.Series([], dtype='object')
            }))

        bindings = {}
        if g._source is None:
            bindings['source'] = 'src'
        if g._destination is None:
            bindings['destination'] = 'dst'
        if g._node is None:
            bindings['node'] = 'id'
        if g._edge_title is None and g._edge_label is None and g._edges is not None:
            if 'label' in g._edges:
                bindings['edge_title'] = 'label'
        g = g.bind(**bindings)

        return g


    def fetch_nodes(self, batch_size = 1000, dry_run=False, verbose=False, ignore_errors=False) -> Union[Plottable, List[str]]:
        """
        Enrich nodes by matching g._node to gremlin nodes
        If no g._nodes table available, first synthesize g._nodes from g._edges
        """
        g = self
        nodes_df = g._nodes
        node_id = g._node
        if node_id is None:
            node_id = 'id'
            g = g.bind(node=node_id)
        if nodes_df is None:
            edges_df = g._edges
            if g._edges is None:
                raise Exception('Node enrichment requires either g._nodes or g._edges to be available')
            
            if g._source is None or g._destination is None:
                raise Exception('Cannot infer nodes table without having set g._source and g._destination bindings')

            nodes_df = pd.concat([
                edges_df[[g._source]].rename(columns={g._source: node_id}).drop_duplicates(),
                edges_df[[g._destination]].rename(columns={g._destination: node_id}).drop_duplicates()
            ], ignore_index=True, sort=False)
        
        if node_id not in nodes_df:
            raise Exception('Node id node in nodes table, excepted column', node_id)

        # Work in batches of 1000
        enriched_nodes_dfs = []
        dry_runs = []
        for start in range(0, len(nodes_df), batch_size):
            nodes_batch_df = nodes_df[start:(start + batch_size)]
            node_ids = ', '.join([f'"{x}"' for x in nodes_batch_df[node_id].to_list() ])
            query = f'g.V({node_ids}).elementMap()'  # TODO: alt for cosmos?
            if dry_run:
                dry_runs.append(query)
                continue
            resultset = self.gremlin_run([query], throw=True)
            g2 = self.resultset_to_g(resultset, 'nodes', verbose, ignore_errors)
            assert g2._nodes is not None
            enriched_nodes_dfs.append(g2._nodes)
        if dry_run:
            return dry_runs
        nodes2_df = pd.concat(enriched_nodes_dfs, sort=False, ignore_index=True)
        g2 = g.nodes(nodes2_df, node_id)
        return g2

    def fetch_edges(self, batch_size = 1000, dry_run=False, verbose=False, ignore_errors=False) -> Union[Plottable, List[str]]:
        """
        Enrich edges by matching g._edges to gremlin edges
        """
        g = self
        edges_df = g._edges
        edge_id = 'id'
        if edges_df is None:
            raise Exception('Edge enrichment requires g._edges to be available')
        
        if edge_id not in edges_df:
            raise Exception('Edge id not in edges table', edge_id)

        # Work in batches of 1000
        enriched_edges_dfs = []
        dry_runs = []
        for start in range(0, len(edges_df), batch_size):
            edges_batch_df = edges_df[start:(start + batch_size)]
            edge_ids = ', '.join([f'"{x}"' for x in edges_batch_df[edge_id].to_list() ])
            query = f'g.E({edge_ids}).elementMap()'  # TODO: alt for cosmos?
            if dry_run:
                dry_runs.append(query)
                continue
            resultset = self.gremlin_run([query], throw=True)
            g2 = self.resultset_to_g(resultset, 'edges', verbose, ignore_errors)
            assert g2._edges is not None
            enriched_edges_dfs.append(g2._edges)
        if dry_run:
            return dry_runs
        edges2_df = pd.concat(enriched_edges_dfs, sort=False, ignore_index=True)
        g2 = g.edges(edges2_df)
        return g2


if TYPE_CHECKING:
    COSMOS_BASE = GremlinMixin
    NEPTUNE_BASE = GremlinMixin
else:
    COSMOS_BASE = object
    NEPTUNE_BASE = object


class NeptuneMixin(NEPTUNE_BASE):

    def __init__(self, *args, **kwargs):
        pass

    def neptune(
        self,
        NEPTUNE_READER_HOST : Optional[str] = None,
        NEPTUNE_READER_PORT : Optional[str] = None,
        NEPTUNE_READER_PROTOCOL : Optional[str] = None,
        endpoint : Optional[str] = None,
        gremlin_client: Optional[Any] = None
    ):
        """
           Provide credentials as arguments, as environment variables, or by providing a gremlinpython client
           Environment variable names are the same as the constructor argument names
           If endpoint provided, do not need host/port/protocol
           If no client provided, create (connect)

        **Example: Login and plot via parrams**

            ::

                import graphistry
                (graphistry
                    .neptune(
                        NEPTUNE_READER_PROTOCOL='wss'
                        NEPTUNE_READER_HOST='neptunedbcluster-xyz.cluster-ro-abc.us-east-1.neptune.amazonaws.com'
                        NEPTUNE_READER_PORT='8182'
                    )
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        **Example: Login and plot via env vars**

            ::

                import graphistry
                (graphistry
                    .neptune()
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        **Example: Login and plot via endpoint**

            ::

                import graphistry
                (graphistry
                    .neptune(endpoint='wss://neptunedbcluster-xyz.cluster-ro-abc.us-east-1.neptune.amazonaws.com:8182/gremlin')
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())

        **Example: Login and plot via client**

            ::

                import graphistry
                (graphistry
                    .neptune(gremlin_client=client)
                    .gremlin('g.E().sample(10)')
                    .fetch_nodes()  # Fetch properties for nodes
                    .plot())
        """

        #alt:
        #from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
        #from gremlin_python.structure.graph import Graph
        #remoteConn = DriverRemoteConnection(endpoint,'g')
        #graph = Graph()
        #g = graph.traversal().withRemote(remoteConn)

        #versions:
        # Downgrade gremlin python to match neptune and tornado to match old gremlin
        #! pip install --user gremlinpython==3.4.10
        #! pip install --user tornado==4.5.3
        #
        #! pip list | grep gremlinpython
        #! pip list | grep tornado
        #
        ##import nest_asyncio
        ##nest_asyncio.apply()
        #
        ##import asyncio
        ##loop = asyncio.new_event_loop()
        ##asyncio.set_event_loop(loop)

        ensure_imports()

        if gremlin_client is None:
            if endpoint is None:
                self.NEPTUNE_READER_HOST = NEPTUNE_READER_HOST if NEPTUNE_READER_HOST is not None else os.environ['NEPTUNE_READER_HOST']
                self.NEPTUNE_READER_PORT = NEPTUNE_READER_PORT if NEPTUNE_READER_PORT is not None else os.environ['NEPTUNE_READER_PORT']
                self.NEPTUNE_READER_PROTOCOL = NEPTUNE_READER_PROTOCOL if NEPTUNE_READER_PROTOCOL is not None else os.environ['NEPTUNE_READER_PROTOCOL']
                self.endpoint = f'{self.NEPTUNE_READER_PROTOCOL}://{self.NEPTUNE_READER_HOST}:{self.NEPTUNE_READER_PORT}/gremlin'
            else:
                self.endpoint = endpoint
            gremlin_client = Client(self.endpoint, 'g',
                message_serializer=GraphSONSerializersV2d0(),
                pool_size=1)
        self._gremlin_client = gremlin_client

        def connect(self: NeptuneMixin) -> NeptuneMixin:
            """Reconnect. Requires initialization was via credentials."""

            if self._gremlin_client is not None:
                self._gremlin_client.close()

            self._gremlin_client = Client(self.endpoint, 'g',
                message_serializer=GraphSONSerializersV2d0(),
                pool_size=1)

            return self

        self._reconnect_gremlin : Optional[Callable[[NeptuneMixin], NeptuneMixin]] = connect  # type: ignore

        return self



class CosmosMixin(COSMOS_BASE):

    def __init__(self, *args, **kwargs):
        pass

    def cosmos(
        self,
        COSMOS_ACCOUNT: Optional[str] = None,
        COSMOS_DB: Optional[str] = None,
        COSMOS_CONTAINER: Optional[str] = None,
        COSMOS_PRIMARY_KEY: Optional[str] = None,
        gremlin_client: Optional[Client] = None
    ):
        """
           Provide credentials as arguments, as environment variables, or by providing a gremlinpython client
           Environment variable names are the same as the constructor argument names
           If no client provided, create (connect)

        **Example: Login and plot **
                ::

                    import graphistry
                    (graphistry
                        .cosmos(
                            COSMOS_ACCOUNT='a',
                            COSMOS_DB='b',
                            COSMOS_CONTAINER='c',
                            COSMOS_PRIMARY_KEY='d')
                        .gremlin('g.E().sample(10)')
                        .fetch_nodes()  # Fetch properties for nodes
                        .plot())

        """

        ensure_imports()

        self.COSMOS_ACCOUNT = COSMOS_ACCOUNT if COSMOS_ACCOUNT is not None else os.environ['COSMOS_ACCOUNT']
        self.COSMOS_DB = COSMOS_DB if COSMOS_DB is not None else os.environ['COSMOS_DB']
        self.COSMOS_CONTAINER = COSMOS_CONTAINER if COSMOS_CONTAINER is not None else os.environ['COSMOS_CONTAINER']
        self.COSMOS_PRIMARY_KEY = COSMOS_PRIMARY_KEY if COSMOS_PRIMARY_KEY is not None else os.environ['COSMOS_PRIMARY_KEY']
        self._gremlin_client = gremlin_client

        def connect(self: CosmosMixin) -> CosmosMixin:

            if self._gremlin_client is not None:
                self._gremlin_client.close()

            self._gremlin_client = Client(
                f'wss://{self.COSMOS_ACCOUNT}.gremlin.cosmosdb.azure.com:443/',
                'g', 
                username=f"/dbs/{self.COSMOS_DB}/colls/{self.COSMOS_CONTAINER}",
                password=self.COSMOS_PRIMARY_KEY,
                message_serializer=GraphSONSerializersV2d0())
            return self

        self._reconnect_gremlin : Optional[Callable[[CosmosMixin], CosmosMixin]] = connect  # type: ignore

        if gremlin_client is None:
            if self._reconnect_gremlin is None:
                raise ValueError('Missing _reconnect_gremlin')
            else:
                self._reconnect_gremlin(self)

        return self

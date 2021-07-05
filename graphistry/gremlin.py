from typing import Callable, Iterable, List, Optional, Union, TYPE_CHECKING
import logging, os, pandas as pd
from .Plottable import Plottable
try:
    from gremlin_python.driver.client import Client
    from gremlin_python.driver.resultset import ResultSet
    from gremlin_python.driver.serializer import GraphSONSerializersV2d0
except:
    1
logger = logging.getLogger('gremlin')


if TYPE_CHECKING:
    MIXIN_BASE = Plottable
else:
    MIXIN_BASE = object


def clean_str(v):
    if isinstance(v, str):
        return v.replace('"', r'\"')
    return str(v)


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
        if isinstance(queries, str):
            queries = [ queries ]
        resultsets = self.gremlin_run(queries, throw=True)
        logger.debug('resultsets: %s', resultsets)
        g = self.resultset_to_g(resultsets)
        return g


    def resultset_to_g(self, resultsets: Union[ResultSet, Iterable[ResultSet]], verbose=False, ignore_errors=False) -> Plottable:
        """
        Convert traversal results to graphistry object with ._nodes, ._edges
        If only received nodes or edges, populate that field
        For custom src/dst/node bindings, passing in a Graphistry instance with .bind(source=.., destination=..., node=...)
        Otherwise, will do src/dst/id
        """
        

        if isinstance(resultsets, ResultSet):
            resultsets = [resultsets]
            
        nodes = []
        edges = []
        for resultset in resultsets:
            if verbose:
                logger.debug('resultset: %s :: %s', resultset, type(resultset))
            
            try:
                for result in resultset:
                    if verbose:
                        logger.debug('result: %s :: %s', result, type(result))
                    if isinstance(result, dict):
                        result = [ result ]
                    for item in result:
                        if verbose:
                            logger.debug('item: %s :: %s', item, type(item))
                        if isinstance(item, dict):
                            if 'type' in item:
                                if item['type'] == 'vertex':
                                    nodes.append(flatten_vertex_dict(item))
                                elif item['type'] == 'edge':
                                    edges.append(flatten_edge_dict(item))
                                else:
                                    raise ValueError('unexpected item type', item['item'])
                            else:
                                for k in item.keys():
                                    item_k_val = item[k]
                                    if item_k_val['type'] == 'vertex':
                                        nodes.append(flatten_vertex_dict(item_k_val))
                                    elif item_k_val['type'] == 'edge':
                                        edges.append(flatten_edge_dict(item_k_val))
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
        
        bindings = {}
        if self._source is None:
            bindings['source'] = 'src'
        if self._destination is None:
            bindings['destination'] = 'dst'
        if self._node is None:
            bindings['node'] = 'id'
        g = self.bind(**bindings)

        g = g.nodes(nodes_df)

        if len(edges) > 0 and edges_df is not None:
            g = g.edges(edges_df)
        #elif len(nodes) > 0:
        #    v0 = nodes[0][g._node]
        #    g = g.edges(pd.DataFrame({
        #        g._source: pd.Series([v0], dtype=nodes_df[g._node].dtype),  # type: ignore
        #        g._destination: pd.Series([v0], dtype=nodes_df[g._node].dtype)  # type: ignore
        #    }))
        else:
            g = g.edges(pd.DataFrame({
                g._source: pd.Series([], dtype='object'),
                g._destination: pd.Series([], dtype='object')
            }))
        
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
            query = f'g.V({node_ids})'
            if dry_run:
                dry_runs.append(query)
                continue
            resultset = self.gremlin_run([query], throw=True)
            g2 = self.resultset_to_g(resultset, verbose, ignore_errors)
            assert g2._nodes is not None
            enriched_nodes_dfs.append(g2._nodes)
        if dry_run:
            return dry_runs
        nodes2_df = pd.concat(enriched_nodes_dfs, sort=False, ignore_index=True)
        g2 = g.nodes(nodes2_df, node_id)
        return g2


if TYPE_CHECKING:
    COSMOS_BASE = GremlinMixin
else:
    COSMOS_BASE = object

class CosmosMixin(COSMOS_BASE):

    def __init__(self, *args, **kwargs):
        pass

    def cosmos(
        self,
        COSMOS_ACCOUNT: str = None,
        COSMOS_DB: str = None,
        COSMOS_CONTAINER: str = None,
        COSMOS_PRIMARY_KEY: str = None,
        gremlin_client: Client = None
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

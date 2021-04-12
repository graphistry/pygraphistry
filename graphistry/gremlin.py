from typing import Iterable, List, Optional, Union
import logging, os, pandas as pd
from .Plottable import Plottable
try:
    from gremlin_python.driver.client import Client
    from gremlin_python.driver.resultset import ResultSet
    from gremlin_python.driver.serializer import GraphSONSerializersV2d0
except:
    1
logger = logging.getLogger('gremlin')



def clean_str(v):
    if isinstance(v, str):
        return v.replace('"', r'\"')
    return str(v)


# ####

def node_to_query(d: dict, type_col: Optional[str] = None) -> str:
    """
    Assumes properties type, id
    """
    if type_col is None:
        if 'category' in d:
            type_col = 'category'
        elif 'type' in d:
            type_col = 'type'
        else:
            raise Exception('Must specify node type_col or provide node column category or type')
    base = f'g.addV(\'{clean_str(d[type_col])}\')'
    skiplist = ['type']
    for p in d.keys():
        if not pd.isna(d[p]) and (p not in skiplist):
            base += f'.property(\'{clean_str(p)}\', \'{clean_str(d[p])}\')'
    return base


def nodes_to_queries(g, partition_key_name: str, partition_key_value: str = '1', target_id_col: str = 'id', type_col: Optional[str] = None) -> Iterable[str]:
    """
    Return stream of node add queries:
      * Sets partition_key if not available
      * Automatically renames g._node to column target_id_col (default: 'id')
    """
    
    nodes_df = (
        g._nodes
            .assign(**({partition_key_name: '1'} if partition_key_name not in g._nodes else {}))
            .rename(columns={g._node: target_id_col})
    )
    
    return (node_to_query(row) for index, row in nodes_df.iterrows())


# ####


def edge_to_query(d: dict, from_col: str, to_col: str, type_col: Optional[str] = None) -> str:
    """
    Assumes properties from, to, type
    """
    if type_col is None:
        if 'edgeType' in d:
            type_col = 'edgeType'
        elif 'category' in d:
            type_col = 'category'
        elif 'type' in d:
            type_col = 'type'
        else:
            raise Exception('Must specify edge type_col or provide edge column edgeType, category, or type')
    base = f'g.v(\'{clean_str(d[from_col])}\').addE(\'{clean_str(d[type_col])}\').to(g.v(\'{clean_str(d[to_col])}\'))'
    if True:
        skiplist = [from_col, to_col, type_col]
        for p in d.keys():
            if not pd.isna(d[p]) and (p not in skiplist):
                base += f'.property(\'{clean_str(p)}\', \'{clean_str(d[p])}\')'
    return base


def edges_to_queries(g, type_col: Optional[str] = None) -> Iterable[str]:
    """
    Return stream of edge add queries
    """    
    edges_df = g._edges
    return (edge_to_query(row, g._source, g._destination) for index, row in edges_df.iterrows())


# ####


#https://github.com/graphistry/graph-app-kit/blob/master/src/python/neptune_helper/gremlin_helper.py
def vertex_to_dict(vertex, id_col: str = 'id', label_col: str = 'label'):
    d = {}
    for k in vertex.keys():
        v = vertex[k]
        if isinstance(v, list):
            d[str(k)] = v[0]
            continue
        if k == 'properties' and isinstance(v, dict):
            for prop_k in v:
                v2 = v[prop_k]
                if isinstance(v2, list) and (len(v2) == 1) and isinstance(v2[0], dict) and 'id' in v2[0] and 'value' in v2[0]:
                    d[str(prop_k)] = v2[0]['value']
                    continue
                d[str(prop_k)] = str(v2)
            continue
        d[str(k)] = vertex[k]
    if 'T.id' in d:  # fixme: from neptune?
        d[id_col] = d.pop('T.id')
    if 'T.label' in d:
        d[label_col] = d.pop('T.label')  # fixme: from neptune?
    return d

#https://github.com/graphistry/graph-app-kit/blob/master/src/python/neptune_helper/gremlin_helper.py
def edge_to_dict(edge, src_col: str = 'src', dst_col: str = 'dst'):
    d = {}
    for k in edge.keys():
        v = edge[k]
        if isinstance(v, list):
            d[str(k)] = v[0]
            continue
        if k == 'properties' and isinstance(v, dict):
            for prop_k in v:
                v2 = v[prop_k]
                if isinstance(v2, list) and (len(v2) == 1) and isinstance(v2[0], dict) and 'id' in v2[0] and 'value' in v2[0]:
                    d[str(prop_k)] = v2[0]['value']
                    continue
                d[str(prop_k)] = str(v2)
            continue
        d[str(k)] = edge[k]
    if 'inV' in d:
        d[src_col] = d.pop('inV')
    if 'outV' in d:
        d[dst_col] = d.pop('outV')
    return d



class GremlinMixin(Plottable):
    """
    Currently serializes queries as strings instead of bytecode in order to support cosmosdb
    """
    def __init__(
        self: Plottable,
        COSMOS_ACCOUNT: str = None,
        COSMOS_DB: str = None,
        COSMOS_CONTAINER: str = None,
        COSMOS_PRIMARY_KEY: str = None,
        COSMOS_PARTITION_KEY: str = None,
        client = None
    ):
        """
           Provide credentials as arguments, as environment variables, or by providing a gremlinpython client
           Environment variable names are the same as the constructor argument names
        """
        self.COSMOS_ACCOUNT = COSMOS_ACCOUNT if COSMOS_ACCOUNT is not None else os.environ['COSMOS_ACCOUNT']
        self.COSMOS_DB = COSMOS_DB if COSMOS_DB is not None else os.environ['COSMOS_DB']
        self.COSMOS_CONTAINER = COSMOS_CONTAINER if COSMOS_CONTAINER is not None else os.environ['COSMOS_CONTAINER']
        self.COSMOS_PRIMARY_KEY = COSMOS_PRIMARY_KEY if COSMOS_PRIMARY_KEY is not None else os.environ['COSMOS_PRIMARY_KEY']
        self.COSMOS_PARTITION_KEY = COSMOS_PARTITION_KEY if COSMOS_PARTITION_KEY is not None else os.environ['COSMOS_PARTITION_KEY']

        self.client = client


    def drop_graph(self):
        """
            Remove all graph nodes and edges from the database
        """
        self.run('g.V().drop()')  # .iterate() ? follow by g.tx().commit() ? 
        return self

    def connect(self):
        """
            Use 
        """
        self.client = Client(
            f'wss://{self.COSMOS_ACCOUNT}.gremlin.cosmosdb.azure.com:443/',
            'g', 
            username=f"/dbs/{self.COSMOS_DB}/colls/{self.COSMOS_CONTAINER}",
            password=self.COSMOS_PRIMARY_KEY,
            message_serializer=GraphSONSerializersV2d0())
        return self


    # Tutorial: 
    # https://itnext.io/getting-started-with-graph-databases-azure-cosmosdb-with-gremlin-api-and-python-80e57cbd1c5e
    def run(self, queries: Iterable[str], throw=False) -> ResultSet:
        for query in queries:
            logger.debug('query: %s', query)
            try:
                callback = self.client.submitAsync(query)
                if callback.result() is not None:
                    results = callback.result()
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

       
    def gremlin(self, queries: Iterable[str], g = None):
        """
            Run one or more gremlin queries and get back the result as a graph object
            To support cosmosdb, sends as strings
        """
        if isinstance(queries, str):
            queries = [ queries ]
        resultsets = self.run(queries, throw=True)
        g = self.resultset_to_g(resultsets, g)
        return g


    def resultset_to_g(self, resultsets: Union[ResultSet, Iterable[ResultSet]], g = None):
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
            for result in resultset:
                for item in result:
                    if isinstance(item, dict):
                        if 'type' in item:
                            if item['type'] == 'vertex':
                                nodes.append(vertex_to_dict(item))
                            elif item['type'] == 'edge':
                                edges.append(edge_to_dict(item))
                        else:
                            for k in item.keys():
                                item_k_val = item[k]
                                if item_k_val['type'] == 'vertex':
                                    nodes.append(vertex_to_dict(item_k_val))
                                elif item_k_val['type'] == 'edge':
                                    edges.append(edge_to_dict(item_k_val))
                                else:                                
                                    raise Exception('unexpected item key val:', type(item[k]))

                    else:
                        raise Exception('unexpected non-dict item type:', type(item))
        
        
        nodes_df = pd.DataFrame(nodes) if len(nodes) > 0 else None
        edges_df = pd.DataFrame(edges) if len(edges) > 0 else None

        if g is None:
            g = self.bind(source='src', destination='dst')  # defer node binding
            if nodes_df is not None:
                g = g.bind(node='id')

        if nodes_df is not None:
            g = g.nodes(nodes_df)

        if len(edges) > 0:
            g = g.edges(edges_df)
        elif len(nodes) > 0:
            v0 = nodes[0][g._node]
            g = g.edges(pd.DataFrame({
                g._source: pd.Series([v0], dtype=nodes_df[g._node].dtype),  # type: ignore
                g._destination: pd.Series([v0], dtype=nodes_df[g._node].dtype)  # type: ignore
            }))
        else:
            g = g.edges(pd.DataFrame({
                g._source: pd.Series([], dtype='object'),
                g._destination: pd.Series([], dtype='object')
            }))
        
        return g


    def enrich_nodes(self, g, batch_size = 1000):
        """
        Enrich nodes by matching g._node to gremlin nodes
        If no g._nodes table available, synthesize from g._edges
        """
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
        enrichd_nodes_dfs = []
        for start in range(0, len(nodes_df), batch_size):
            nodes_batch_df = nodes_df[start:(start + batch_size)]
            node_ids = ', '.join([f'"{x}"' for x in nodes_batch_df[node_id].to_list() ])
            query = f'g.V({node_ids})'
            resultset = self.run(query, throw=True)
            g2 = self.resultset_to_g(resultset)
            enrichd_nodes_dfs.append(g2._nodes)
        nodes2_df = pd.concat(enrichd_nodes_dfs, sort=False, ignore_index=True)
        g2 = g.nodes(nodes2_df, node_id)
        return g2

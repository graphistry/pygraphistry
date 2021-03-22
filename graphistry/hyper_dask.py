#
# Like hypergraph(); adds engine = 'pandas' | 'cudf' | 'dask' | 'dask-cudf'
#

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from .Engine import Engine, DataframeLike, DataframeLocalLike
import logging, pandas as pd, pyarrow as pa, sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: When Python 3.8+, switch to TypedDict
class HyperBindings():
    def __init__(
        self,
        TITLE: str = 'nodeTitle',
        DELIM: str = '::',
        NODEID: str = 'nodeID',
        ATTRIBID: str = 'attribID',
        EVENTID: str = 'EventID',
        SOURCE: str = 'src',
        DESTINATION: str = 'dst',
        CATEGORY: str = 'category',
        NODETYPE: str = 'type',
        EDGETYPE: str = 'edgeType',
        NULLVAL: str = 'null',
        SKIP: List[str] = [],
        CATEGORIES: Dict[str, List[str]] = {},
        EDGES: Optional[Dict[str, List[str]]] = None
    ):
        self.title = TITLE
        self.delim = DELIM
        self.node_id = NODEID
        self.attrib_id = ATTRIBID
        self.event_id = EVENTID
        self.source = SOURCE
        self.destination = DESTINATION
        self.category = CATEGORY
        self.node_type = NODETYPE
        self.edge_type = EDGETYPE
        self.categories = CATEGORIES
        self.edges = EDGES
        self.null_val = NULLVAL

        self.skip = SKIP.copy()
        # Prevent metadata fields from turning into nodes 
        # bindings = vars(self)
        # for key in bindings.keys():
        #     if (key != 'SKIP') and (bindings[key] not in self.skip):
        #         self.skip.append(bindings[key])

def screen_entities(events: DataframeLike, entity_types: Optional[List[str]], defs: HyperBindings) -> List[str]:
    """
    List entity columns: Unskipped user-specified entities when provided, else unskipped cols
    """
    base = entity_types if entity_types is not None else [x for x in events.columns]
    return [x for x in base if x not in defs.skip]

def col2cat(cat_lookup: Dict[str, str], col: str):
    return cat_lookup[col] if col in cat_lookup else col

def make_reverse_lookup(categories):
    lookup = {}
    for category in categories:
        for col in categories[category]:
            lookup[col] = str(category)
    return lookup



def format_entities_from_col(
    defs: HyperBindings,
    cat_lookup: Dict[str, str],
    drop_na: bool,
    engine: Engine,
    df_hd: DataframeLocalLike,
    col_name: str,
    df_with_col: DataframeLike
) -> DataframeLocalLike:
    """
    For unique v in column col, create [{col: str(v), title: str(v), nodetype: col, nodeid: `<cat><delim><v>`}]
        - respect drop_na
        - respect colname overrides
        - receive+return pd.DataFrame / cudf.DataFrame depending on engine
    """
    logger.debug('@format_entities: [drop: %s], %s / %s', drop_na, col_name, [c for c in df_with_col])

    try:
        df_with_col_pre = df_with_col[col_name].dropna() if drop_na else df_with_col[col_name]
        try:
            unique_vals = df_with_col_pre.drop_duplicates()
        except:
            unique_vals = df_with_col_pre.astype(str).drop_duplicates()
            logger.warning('Coerced col %s to string type for entity names', col_name)
        unique_safe_val_strs = unique_vals.astype(str).fillna(defs.null_val)
    except NotImplementedError:
        logger.warning('Dropped col %s from entity list due to errors')
        unique_vals = df_with_col[[]].reset_index()['index'][:0]
        unique_safe_val_strs = unique_vals.astype(str).fillna(defs.null_val)

    if engine in [Engine.PANDAS, Engine.DASK]:
        cons = pd.DataFrame
    else:
        import cudf
        cons = cudf.DataFrame
    
    return cons({
        col_name: unique_vals,
        defs.title: unique_safe_val_strs,
        defs.node_type: col_name,
        defs.category: col2cat(cat_lookup, col_name),
        defs.node_id: (col2cat(cat_lookup, col_name) + defs.delim) + unique_safe_val_strs
    })

def concat(dfs: List[DataframeLike], engine: Engine):

    if engine == Engine.PANDAS:
        return pd.concat(dfs, ignore_index=True, sort=False)

    if engine == Engine.DASK:
        import dask.dataframe
        return dask.dataframe.concat(dfs).reset_index(drop=True)
    
    if engine == Engine.CUDF:
        import cudf
        try:
            return cudf.concat(dfs)
        except TypeError as e:
            logger.warning('Failed to concat, likely due to column type issue, try converting to a string; columns')
            for df in dfs:
                logger.warning('df types :: %s', df.dtypes)
            raise e

    if engine == Engine.DASK_CUDF:
        import dask_cudf
        return dask_cudf.concat(dfs)

def mt_df(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.DataFrame()

    if engine == Engine.DASK:
        import dask.dataframe
        return dask.dataframe.DataFrame()

    if engine == Engine.CUDF:
        import cudf
        return cudf.DataFrame()

    if engine == Engine.DASK_CUDF:
        import dask_cudf
        return dask_cudf.DataFrame()

#ex output: DataFrameLike([{'val::state': 'CA', 'nodeType': 'state', 'nodeID': 'state::CA'}])
def format_entities(
    events: DataframeLike,
    entity_types: List[str],
    defs: HyperBindings,
    drop_na: bool,
    engine: Engine,
    npartitions: Optional[int],
    chunksize: Optional[int]) -> DataframeLike:

    logger.debug('@format_entities :: %s', entity_types)


    if engine in [Engine.PANDAS, Engine.DASK]:
        df_cons = pd.DataFrame
        series_cons = pd.Series
    else:
        import cudf
        df_cons = cudf.DataFrame
        series_cons = cudf.Series

    # handle mt + dask meta
    df_hd = df_cons({
        defs.title: series_cons([], dtype='object'),
        defs.node_type: series_cons([], dtype='object'),
        defs.category: series_cons([], dtype='object'),
        defs.node_id: series_cons([], dtype='object'),
        **{
            col_name: series_cons([], dtype=events[col_name].dtype)
            for col_name in entity_types
        }
    })

    cat_lookup = make_reverse_lookup(defs.categories)
    logger.debug('@format_entities cat_lookup [ %s ] => [ %s ]', defs.categories, cat_lookup)

    entity_dfs = [
        format_entities_from_col(
            defs, cat_lookup, drop_na, engine, df_hd,
            col_name, events[[col_name]])
        for col_name in entity_types
    ]
    for df in entity_dfs:
        logger.debug('sub df: %s', df)

    df = concat(entity_dfs, engine).drop_duplicates([defs.node_id])

    return df


#ex output: pd.DataFrame([{'edgeType': 'state', 'attribID': 'state::CA', 'eventID': 'eventID::0'}])
def format_hyperedges(engine, events, entity_types, defs, drop_na, drop_edge_attrs):
    is_using_categories = len(defs.categories.keys()) > 0
    cat_lookup = make_reverse_lookup(defs.categories)

    subframes = []
    for col in sorted(entity_types):
        fields = list(set([defs.event_id] + ([x for x in events.columns] if not drop_edge_attrs else [col])))
        raw = events[ fields ]
        if drop_na:
            raw = raw.dropna(axis=0, subset=[col])            
        raw = raw.copy()
        if len(raw):            
            if is_using_categories:
                raw[defs.edge_type] = col2cat(cat_lookup, col)
                raw[defs.category] = col
            else:
                raw[defs.edge_type] = col
            try:
                raw[defs.attrib_id] = (col2cat(cat_lookup, col) + defs.delim) + raw[col].astype(str).fillna(defs.null_val)
            except NotImplementedError:
                logger.warning('Did not create hyperedges for column %s as does not support astype(str)', col)
                continue
            subframes.append(raw)

    if len(subframes):
        result_cols = list(set(
            ([x for x in events.columns.tolist() if not x == defs.node_type] 
                if not drop_edge_attrs 
                else [])
            + [defs.edge_type, defs.attrib_id, defs.event_id]  # noqa: W503
            + ([defs.category] if is_using_categories else []) ))  # noqa: W503
        out = concat(subframes, engine).reset_index(drop=True)[ result_cols ]
        return out
    else:
        return pd.DataFrame([])


def direct_edgelist_shape(entity_types: List[str], defs: HyperBindings) -> Dict[str, List[str]]:
    """
        Edges take format {src_col: [dest_col1, dest_col2], ....}
        If None, create connect all to all, leaving up to algorithm in which direction
    """
    if defs.edges is not None:
        return defs.edges
    else:
        out = {}
        for entity_i in range(len(entity_types)):
            out[ entity_types[entity_i] ] = entity_types[(entity_i + 1):]
        return out
  
      
#ex output: DataFrameLike([{'edgeType': 'state', 'attribID': 'state::CA', 'eventID': 'eventID::0'}])
def format_direct_edges(
    engine: Engine, events: DataframeLike, entity_types, defs: HyperBindings, edge_shape, drop_na: bool, drop_edge_attrs: bool
) -> DataframeLike:
    is_using_categories = len(defs.categories.keys()) > 0
    cat_lookup = make_reverse_lookup(defs.categories)

    subframes = []
    for col1 in sorted(edge_shape.keys()):
        for col2 in sorted(edge_shape[col1]):
            fields = list(set([defs.event_id] + ([x for x in events.columns] if not drop_edge_attrs else [col1, col2])))
            raw = events[ fields ]
            if drop_na:
                raw = raw.dropna(axis=0, subset=[col1, col2])
            raw = raw.copy()
            if len(raw):
                if is_using_categories:
                    raw[defs.edge_type] = col2cat(cat_lookup, col1) + defs.delim + col2cat(cat_lookup, col2)
                    raw[defs.category] = col1 + defs.delim + col2
                else:
                    raw[defs.edge_type] = col1 + defs.delim + col2
                raw[defs.source] = (col2cat(cat_lookup, col1) + defs.delim) + raw[col1].astype(str).fillna(defs.null_val)
                raw[defs.destination] = (col2cat(cat_lookup, col2) + defs.delim) + raw[col2].astype(str).fillna(defs.null_val)
                subframes.append(raw)

    if len(subframes):
        result_cols = list(set(
            ([x for x in events.columns.tolist() if not x == defs.node_type] 
                if not drop_edge_attrs 
                else [])
            + [defs.edge_type, defs.source, defs.destination, defs.event_id]  # noqa: W503
            + ([defs.category] if is_using_categories else []) ))  # noqa: W503
        out = concat(subframes, engine=engine)[ result_cols ]
        return out
    else:
        return events[:0][[]]


def format_hypernodes(events, defs, drop_na):
    event_nodes = events.copy()
    event_nodes[defs.node_type] = defs.event_id
    event_nodes[defs.category] = 'event'
    event_nodes[defs.node_id] = event_nodes[defs.event_id]
    event_nodes[defs.title] = event_nodes[defs.event_id]
    return event_nodes

def hyperbinding(g, defs, entities, event_entities, edges, source, destination):
    nodes = pd.concat([entities, event_entities], ignore_index=True, sort=False).reset_index(drop=True)
    return {
        'entities': entities,
        'events': event_entities,
        'edges': edges,
        'nodes': nodes,
        'graph':
            g
            .bind(source=source, destination=destination).edges(edges)
            .bind(node=defs.node_id, point_title=defs.title).nodes(nodes)
    }     

#turn lists etc to strs, and preserve nulls
def flatten_objs_inplace(df, cols):
    for c in cols:
        name = df[c].dtype.name
        if name == 'category':
            #Avoid warning
            df[c] = df[c].astype(str).where(~df[c].isnull(), df[c])
        elif name == 'object':
            df[c] = df[c].where(df[c].isnull(), df[c].astype(str))
 
#

def shallow_copy(df: DataframeLike, engine: Engine) -> DataframeLike:
    if engine in [Engine.DASK, Engine.DASK_CUDF]:
        df2 = df.copy()
    else:
        df2 = df.copy(deep=False)
    return df2


def df_coercion(  # noqa: C901
    df: DataframeLike,
    engine: Engine,
    npartitions: Optional[int] = None,
    chunksize: Optional[int] = None
) -> DataframeLike:
    """
    Supported coercions:
        pd <- pd
        cudf <- pd, cudf
        ddf <- pd, ddf
        dgdf <- pd, cudf, dgdf
    """
    if engine == Engine.PANDAS:
        if isinstance(df, pd.DataFrame):
            return df
        raise ValueError('pandas engine mode requires pd.DataFrame input, received: %s' % str(type(df)))

    if engine == Engine.CUDF:
        import cudf
        if isinstance(df, pd.DataFrame):
            return cudf.from_pandas(df)
        if isinstance(df, cudf.DataFrame):
            return df
        raise ValueError('cudf engine mode requires pd.DataFrame/cudf.DataFrame input, received: %s' % str(type(df)))

    if engine == Engine.DASK:
        import dask
        if isinstance(df, pd.DataFrame):
            return dask.dataframe.from_pandas(df, npartitions=npartitions, chunksize=chunksize)
        if isinstance(df, dask.dataframe.DataFrame):
            return df
        raise ValueError('dask engine mode requires pd.DataFrame/dask.dataframe.DataFrame input, received: %s' % str(type(df)))

    if engine == Engine.DASK_CUDF:
        import cudf, dask_cudf, dask.dataframe
        if isinstance(df, pd.DataFrame):
            ddf = df_coercion(df, Engine.DASK, npartitions, chunksize)
            return df_coercion(ddf, Engine.DASK_CUDF, npartitions, chunksize)
        if isinstance(df, cudf.DataFrame):
            return dask_cudf.from_cudf(df, npartitions=npartitions, chunksize=chunksize)
        if isinstance(df, dask_cudf.DataFrame):
            return df
        if isinstance(df, dask.dataframe.DataFrame):
            return df.map_partitions(cudf.from_pandas)  # FIXME How does this get the right type?
        raise ValueError('dask_cudf engine mode requires pd.DataFrame/cudf.DataFrame/dask.dataframe.DataFrame/dask_cudf.DataFrame input, received: %s' % str(type(df)))

    return ValueError('Did not get a value Engine type: %s' % engine)


def clean_events(
    events: DataframeLike,
    engine: Engine,
    npartitions: Optional[int] = None,
    chunksize: Optional[int] = None
) -> DataframeLike:
    """
    Copy with reset index and in the target engine format
    """
    logger.debug('@clean: %s', [c for c in events.columns])
    out_events = df_coercion(events, engine, npartitions, chunksize)
    out_events = shallow_copy(out_events, engine)
    out_events = out_events.reset_index(drop=True)
    return out_events


class Hypergraph():
    def __init__(
        self, g,
        defs, entities: DataframeLike, event_entities: DataframeLike, edges: DataframeLike,
        source: str, destination: str,
        engine: Engine = Engine.PANDAS
    ):
        self.engine = engine
        self.entities = entities
        self.events = event_entities
        self.edges = edges
        self.nodes = concat([entities, event_entities], engine=engine)
        self.graph = (g
            .edges(edges, source, destination)
            .nodes(self.nodes, defs.node_id)
            .bind(point_title=defs.title))


def hypergraph(
    g,
    raw_events: DataframeLike, 
    entity_types: Optional[List[str]] = None,
    opts: dict = {},
    drop_na: bool = True,
    drop_edge_attrs: bool = False,
    verbose: bool = True,
    direct: bool = False,
    engine: str = 'pandas',  # see Engine for valid values
    npartitions: Optional[int] = None,
    chunksize: Optional[int] = None
):
    """
    Internal details:
        - IDs currently strings: `${namespace(col)}${delim}${str(val)}`
    """
    # TODO: String -> categorical
    # TODO: col_name column can be prohibitively wide & sparse: drop / warning?

    engine_resolved : Engine
    if not isinstance(engine, Engine):
        engine_resolved = getattr(Engine, str(engine).upper())  # type: ignore
    else:
        engine_resolved = engine
    defs = HyperBindings(**opts)
    entity_types = screen_entities(raw_events, entity_types, defs)
    events = clean_events(raw_events, engine_resolved, npartitions, chunksize)  # type: ignore

    if defs.event_id in events.columns:
        events[defs.event_id] = (defs.event_id + defs.delim) + events[defs.event_id].astype(str).fillna(defs.null_val) 
    else:
        events[defs.event_id] = (defs.event_id + defs.delim) + events[[]].reset_index()['index'].astype(str)
    events[defs.node_type] = 'event'
    
    entities = format_entities(events, entity_types, defs, drop_na, engine_resolved, npartitions, chunksize)  # type: ignore

    event_entities = None
    edges = None
    if direct:
        edge_shape = direct_edgelist_shape(entity_types, defs)
        event_entities = mt_df(engine_resolved)
        edges = format_direct_edges(engine_resolved, events, entity_types, defs, edge_shape, drop_na, drop_edge_attrs)
    else:        
        event_entities = format_hypernodes(events, defs, drop_na)
        edges = format_hyperedges(engine_resolved, events, entity_types, defs, drop_na, drop_edge_attrs)
    if verbose:
        print('# links', len(edges))
        print('# events', len(events))
        print('# attrib entities', len(entities))
    return Hypergraph(
        g,
        defs, entities, event_entities, edges,
        defs.source if direct else defs.attrib_id,
        defs.destination if direct else defs.event_id,
        engine_resolved)

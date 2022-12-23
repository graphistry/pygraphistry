#
# Like hypergraph(); adds engine = 'pandas' | 'cudf' | 'dask' | 'dask-cudf'
#

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from .Engine import Engine, DataframeLike, DataframeLocalLike
import numpy as np, pandas as pd, pyarrow as pa, sys
from .util import setup_logger
logger = setup_logger(__name__)

# TODO: When Python 3.8+, switch to TypedDict
class HyperBindings():
    def __init__(
        self,
        TITLE: str = 'nodeTitle',
        DELIM: str = '::',
        NODEID: str = 'nodeID',
        ATTRIBID: str = 'attribID',
        EVENTID: str = 'EventID',
        EVENTTYPE: str = 'event',
        SOURCE: str = 'src',
        DESTINATION: str = 'dst',
        CATEGORY: str = 'category',
        NODETYPE: str = 'type',
        EDGETYPE: str = 'edgeType',
        NULLVAL: str = 'null',
        SKIP: Optional[List[str]] = None,
        CATEGORIES: Dict[str, List[str]] = {},
        EDGES: Optional[Dict[str, List[str]]] = None
    ):
        self.title = TITLE
        self.delim = DELIM
        self.node_id = NODEID
        self.attrib_id = ATTRIBID
        self.event_id = EVENTID
        self.event_type = EVENTTYPE
        self.source = SOURCE
        self.destination = DESTINATION
        self.category = CATEGORY
        self.node_type = NODETYPE
        self.edge_type = EDGETYPE
        self.categories = CATEGORIES
        self.edges = EDGES
        self.null_val = NULLVAL

        self.skip = (SKIP or []).copy()
        # Prevent metadata fields from turning into nodes
        bindings = vars(self)
        if SKIP is None:
            key : str
            for key in [
                    #'title', 'node_id', 'attrib_id', 'event_id', 'node_type', 'edge_type'
                ]:
                if (key in bindings) and (bindings[key] not in self.skip):
                    self.skip.append(bindings[key])
            self.skip.sort()

def screen_entities(events: DataframeLike, entity_types: Optional[List[str]], defs: HyperBindings) -> List[str]:
    """
    List entity columns: Unskipped user-specified entities when provided, else unskipped cols
    """
    logger.debug('@screen_entities: skip [ %s ]', defs.skip)
    base = entity_types if entity_types is not None else [x for x in events.columns]
    out = [x for x in base if x not in defs.skip]
    logger.debug('////screen_entities: %s', out)
    return out

def col2cat(cat_lookup: Dict[str, str], col: str):
    return cat_lookup[col] if col in cat_lookup else col

def make_reverse_lookup(categories):
    lookup = {}
    for category in categories:
        for col in categories[category]:
            lookup[col] = str(category)
    return lookup

def coerce_col_safe(s, to_dtype):
    if s.dtype.name == to_dtype.name:
        return s
    if to_dtype.name == 'int64':
        return s.fillna(0).astype('int64')
    if to_dtype.name == 'timedelta64[ns]':
        return s.fillna(np.datetime64('NAT')).astype(str)
    logger.debug('CEORCING %s :: %s -> %s', s.name, s.dtype, to_dtype)
    return s.astype(to_dtype)

def format_entities_from_col(
    defs: HyperBindings,
    cat_lookup: Dict[str, str],
    drop_na: bool,
    engine: Engine,
    col_name: str,
    df_with_col: DataframeLike,
    meta: pd.DataFrame,
    debug: bool
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
        if debug and engine in [ Engine.DASK, Engine.DASK_CUDF ]:
            df_with_col_pre = df_with_col_pre.persist()
            logger.debug('col [ %s ] entities with dropna [ %s ]: %s', col_name, drop_na, df_with_col_pre.compute())
        try:
            unique_vals = df_with_col_pre.drop_duplicates()
        except:
            unique_vals = df_with_col_pre.astype(str).drop_duplicates()
            logger.warning('Coerced col %s to string type for entity names', col_name)
        unique_safe_val_strs = unique_vals.astype(str).fillna(defs.null_val).astype(str)
    except NotImplementedError:
        logger.warning('Dropped col %s from entity list due to errors')
        unique_vals = mt_series(engine)
        unique_safe_val_strs = unique_vals.astype(str).fillna(defs.null_val).astype(str)

    if debug and engine in [ Engine.DASK, Engine.DASK_CUDF ]:
        unique_vals = unique_vals.persist()
        logger.debug('unique_vals: %s', unique_vals.compute())

    base_df = unique_vals.rename(col_name).to_frame()
    base_df = base_df.assign(**{
        defs.title: unique_safe_val_strs,
        defs.node_type: col_name,
        defs.category: col2cat(cat_lookup, col_name),
        defs.node_id: (col2cat(cat_lookup, col_name) + defs.delim) + unique_safe_val_strs
    })

    if debug and engine in [ Engine.DASK, Engine.DASK_CUDF ]:
        base_df = base_df.persist()
        logger.debug('base_df1: %s', base_df.compute())

    missing_cols : List = [ c for c in meta.columns if c not in base_df ]
    base_df = base_df.assign(**{
        c: np.nan
        for c in missing_cols
    })
    logger.debug('==== BASE 2 ====')
    if debug and engine in [ Engine.DASK, Engine.DASK_CUDF ]:
        base_df = base_df.persist()
        logger.debug('base_df2: %s', base_df.compute())
        logger.debug('needs conversions: %s',
            [(
                c,
                base_df[c].dtype.name,
                meta[c].dtype.name  # type: ignore
            ) for c in missing_cols])
        for c in base_df:
            logger.debug('test base_df2 col [ %s ]: %s', c, base_df[c].dtype)
            logger.debug('base_df2[ %s ]: %s', c, base_df[c].compute())
            logger.debug('convert [ %s ] %s -> %s', c, base_df[c].dtype.name, meta[c].dtype.name)
            logger.debug('orig: %s', base_df[c].compute())
            logger.debug('was a missing col needing coercion: %s', c in missing_cols)
            if c in missing_cols:
                logger.debug('coerced 1: %s', coerce_col_safe(base_df[c], meta[c].dtype).compute())
                logger.debug('coerced 2: %s', base_df.assign(**{c: coerce_col_safe(base_df[c], meta[c].dtype)}).compute())
    base_as_meta_df = base_df.assign(**{
        c: coerce_col_safe(
            base_df[c],
            meta[c].dtype)
            if base_df[c].dtype.name != meta[c].dtype.name  # type: ignore
            else base_df[c]
        for c in missing_cols
    })
    logger.debug('==== BASE 3 ====')
    if debug and engine in [ Engine.DASK, Engine.DASK_CUDF ]:
        base_as_meta_df = base_as_meta_df.persist()
        for c in base_df:
            logger.debug('test base_df3 col [ %s ]: %s -> %s', c, base_df[c].dtype, base_as_meta_df[c].dtype)
            logger.debug('base_df3[ %s ]: %s', c, base_as_meta_df[c].compute())

    return base_as_meta_df

def concat(dfs: List[DataframeLike], engine: Engine, debug=False):

    if debug and len(dfs) > 1:
        df0 = dfs[0]
        for c in df0:
            logger.debug('checking df0: %s :: %s', c, df0[c].dtype)
            for df_i in dfs[1:]:
                if c not in df_i:
                    logger.warning('missing df0[%s]::%s in df_i', c, df0[c].dtype)
                if df0[c].dtype != df_i[c].dtype:
                    logger.warning('mismatching df0[c]::%s vs df_i[c]::%s for %s', df0[c].dtype, df_i[c].dtype, c)
        for df_i in dfs[1:]:
            for c in df_i:
                logger.debug('checking df_i: %s', c)
                if c not in df0:
                    logger.warning('missing df_i[%s]::%s in df0', c, df_i[c].dtype)
        logger.debug('all checked!')

    if engine == Engine.PANDAS:
        return pd.concat(dfs, ignore_index=True, sort=False)

    if engine == Engine.DASK:
        import dask.dataframe
        return dask.dataframe.concat(dfs).reset_index(drop=True)
    
    if engine == Engine.CUDF:
        import cudf
        try:
            return cudf.concat(dfs, ignore_index=True)
        except TypeError as e:
            logger.warning('Failed to concat, likely due to column type issue, try converting to a string; columns')
            for df in dfs:
                logger.warning('df types :: %s', df.dtypes)
            raise e

    if engine == Engine.DASK:
        import dask.dataframe as dd
        return dd.concat(dfs)

    if engine == Engine.DASK_CUDF:
        import dask_cudf
        return dask_cudf.concat(dfs)

    raise NotImplementedError('Unknown engine')

def get_df_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.DataFrame

    if engine == Engine.DASK:
        import dask.dataframe
        return dask.dataframe.DataFrame

    if engine == Engine.CUDF:
        import cudf
        return cudf.DataFrame

    if engine == Engine.DASK_CUDF:
        import dask_cudf
        return dask_cudf.DataFrame

    raise NotImplementedError('Unknown engine')

def mt_df(engine: Engine):
    if engine == Engine.DASK:
        import dask.dataframe
        return dask.dataframe.from_pandas(pd.DataFrame(), npartitions=1)
    
    if engine == Engine.DASK_CUDF:
        import cudf, dask_cudf
        return dask_cudf.from_cudf(cudf.from_pandas(pd.DataFrame()), npartitions=1)

    cons = get_df_cons(engine)
    return cons()

def get_series_cons(engine: Engine, dtype='int32'):
    if engine == Engine.PANDAS:
        return pd.Series

    if engine == Engine.DASK:
        import dask.dataframe
        return dask.dataframe.Series

    if engine == Engine.CUDF:
        import cudf
        return cudf.Series

    if engine == Engine.DASK_CUDF:
        import dask_cudf
        return dask_cudf.Series

    raise NotImplementedError('Unknown engine')

def series_cons(engine: Engine, arr: List, dtype='int32', npartitions=None, chunksize=None):
    if engine == Engine.PANDAS:
        return pd.Series(arr, dtype=dtype)

    if engine == Engine.DASK:
        import dask.dataframe
        return dask.dataframe.from_pandas(pd.Series(arr, dtype=dtype), npartitions=npartitions, chunksize=chunksize).astype(dtype)

    if engine == Engine.CUDF:
        import cudf
        return cudf.Series(arr, dtype=dtype)

    if engine == Engine.DASK_CUDF:
        import cudf, dask_cudf
        gs = cudf.Series(arr, dtype=dtype)
        out = dask_cudf.from_cudf(gs, npartitions=npartitions, chunksize=chunksize)
        out2 = out.astype(dtype)
        logger.debug('series_cons :: %s => %s => %s', gs.dtype, out.dtype, out2.dtype)
        return out2
    raise NotImplementedError('Unknown engine')


def mt_series(engine: Engine, dtype='int32'):
    cons = get_series_cons(engine)
    return cons([], dtype=dtype)

# This will be slightly wrong: pandas will turn datetime64['ms'] into datetime64['ns']
def mt_nodes(defs: HyperBindings, events: DataframeLike, entity_types: List[str], direct: bool, engine: Engine) -> pd.DataFrame:

    single_engine = engine
    if engine == Engine.DASK_CUDF:
        single_engine = Engine.CUDF
    if engine == Engine.DASK:
        single_engine = Engine.PANDAS

    mt_obj_s = series_cons(single_engine, [], dtype='object', npartitions=1)

    out = ((events[ entity_types ] if direct else events)
        .head(0)
        .assign(
            **{
                defs.title: mt_obj_s,
                defs.event_id: mt_obj_s,
                defs.node_type: mt_obj_s,
                defs.category: mt_obj_s,
                defs.node_id: mt_obj_s,
            }
        ))

    logger.debug('mt_nodes init :: %s', out.dtypes)

    return out


#ex output: DataFrameLike([{'val::state': 'CA', 'nodeType': 'state', 'nodeID': 'state::CA'}])
def format_entities(
    events: DataframeLike,
    entity_types: List[str],
    defs: HyperBindings,
    direct: bool,
    drop_na: bool,
    engine: Engine,
    npartitions: Optional[int],
    chunksize: Optional[int],
    debug: bool = False) -> DataframeLike:

    logger.debug('@format_entities :: %s', entity_types)
    logger.debug('dtypes: %s', events.dtypes)

    cat_lookup = make_reverse_lookup(defs.categories)
    logger.debug('@format_entities cat_lookup [ %s ] => [ %s ]', defs.categories, cat_lookup)

    mt_df = mt_nodes(defs, events, entity_types, direct, engine)
    logger.debug('mt_df :: %s', mt_df.dtypes)

    entity_dfs = [
        format_entities_from_col(
            defs, cat_lookup, drop_na, engine,
            col_name, events[[col_name]], mt_df,
            debug)
        for col_name in entity_types
    ]
    if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
        entity_dfs = [ df.persist() for df in entity_dfs ]
        for df in entity_dfs:
            logger.debug('format_entities sub df dtypes: %s', df.dtypes)
            if df.dtypes.to_dict() != entity_dfs[0].dtypes.to_dict():
                logger.error('MISMATCHES')
                d1 = df.dtypes.to_dict()
                d2 = entity_dfs[0].dtypes.to_dict()
                for k, v in d1.items():
                    if k not in d2:
                        logger.error('key %s (::%s) missing in df_0', k, v)
                    elif d2[k] != v:
                        logger.error('%s:%s <> %s:%s', k, v, k, d2[k])
                for k, v in d2.items():
                    if k not in d1:
                        logger.error('key %s (::%s) missing in df_i', k, v)
            logger.debug('entity_df: %s', df.compute())

    df = concat(entity_dfs, engine, debug).drop_duplicates([defs.node_id])
    if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
        df = df.persist()
        df.compute()
        logger.debug('////format_entities')

    return df


#ex output: DataFrame([{'edgeType': 'state', 'attribID': 'state::CA', 'eventID': 'eventID::0'}])
def format_hyperedges(
    engine: Engine, events: DataframeLike, entity_types: List[str], defs: HyperBindings,
    drop_na: bool, drop_edge_attrs: bool, debug: bool = False
) -> DataframeLike:
    is_using_categories = len(defs.categories.keys()) > 0
    cat_lookup = make_reverse_lookup(defs.categories)

    # mt_pdf = pd.DataFrame({
    #     **{
    #         **({defs.category: pd.Series([], dtype='object')} if is_using_categories else {}),
    #         defs.edge_type: pd.Series([], dtype='object'),
    #         defs.attrib_id: pd.Series([], dtype='object'),
    #         defs.event_id: pd.Series([], dtype='object'),
    #         defs.category: pd.Series([], dtype='object'),
    #         defs.node_id: pd.Series([], dtype='object'),
    #     },
    #     **({
    #         x: pd.Series([], dtype=events[x].dtype)
    #         for x in entity_types
    #     } if drop_edge_attrs else {
    #         x: pd.Series([], dtype=events[x].dtype)
    #         for x in events.columns
    #     })
    # })

    subframes = []
    for col in sorted(entity_types):
        fields = list(set([defs.event_id] + ([x for x in events.columns] if not drop_edge_attrs else [ col ])))
        raw = events[ fields ]
        if drop_na:
            logger.debug('dropping na [ %s ] from available [ %s]  (fields: [ %s ])', col, raw.columns, fields)
            raw = raw.dropna(subset=[col])
        raw = raw.copy()
        if is_using_categories:
            raw[defs.edge_type] = col2cat(cat_lookup, col)
            raw[defs.category] = col
        else:
            raw[defs.edge_type] = col
        try:
            raw[defs.attrib_id] = (col2cat(cat_lookup, col) + defs.delim) + raw[col].astype(str).fillna(defs.null_val).astype(str)
        except NotImplementedError:
            logger.warning('Did not create hyperedges for column %s as does not support astype(str)', col)
            continue
        if drop_edge_attrs:
            logger.debug('dropping val col [ %s ] from [ %s ]', col, raw.columns)
            raw = raw.drop(columns=[col])
            logger.debug('dropped => [ %s ]', raw.columns)
        if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
            raw = raw.persist()
            raw.compute()
        subframes.append(raw)

    if len(subframes):
        result_cols = list(set(
            ([x for x in events.columns.tolist() if not x == defs.node_type] 
                if not drop_edge_attrs 
                else [])
            + [defs.edge_type, defs.attrib_id, defs.event_id]  # noqa: W503
            + ([defs.category] if is_using_categories else []) ))  # noqa: W503
        if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
            #subframes = [df.persist() for df in subframes]
            for df in subframes:
                logger.debug('edge sub: %s', df.dtypes)
        out = concat(subframes, engine, debug).reset_index(drop=True)[ result_cols ]
        if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
            out = out.persist()
            out.compute()
            logger.debug('////format_hyperedges')
        return out
    else:
        return mt_series(engine)


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
    engine: Engine, events: DataframeLike, entity_types, defs: HyperBindings, edge_shape, drop_na: bool, drop_edge_attrs: bool,
    debug: bool = False
) -> DataframeLike:
    is_using_categories = len(defs.categories.keys()) > 0
    cat_lookup = make_reverse_lookup(defs.categories)

    subframes = []
    for col1 in sorted(edge_shape.keys()):
        for col2 in sorted(edge_shape[col1]):
            fields = list(set([defs.event_id] + ([x for x in events.columns] if not drop_edge_attrs else [col1, col2])))
            raw = events[ fields ]
            if drop_na:
                raw = raw.dropna(subset=[col1, col2])
            raw = raw.copy()
            if is_using_categories:
                raw[defs.edge_type] = col2cat(cat_lookup, col1) + defs.delim + col2cat(cat_lookup, col2)
                raw[defs.category] = col1 + defs.delim + col2
            else:
                raw[defs.edge_type] = col1 + defs.delim + col2
            raw[defs.source] = (col2cat(cat_lookup, col1) + defs.delim) + raw[col1].astype(str).fillna(defs.null_val)
            raw[defs.destination] = (col2cat(cat_lookup, col2) + defs.delim) + raw[col2].astype(str).fillna(defs.null_val)
            if drop_edge_attrs:
                raw = raw.drop(columns=[col1, col2])
            if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
                raw = raw.persist()
                raw.compute()
            subframes.append(raw)

    if len(subframes):
        result_cols = list(set(
            ([x for x in events.columns.tolist() if not x == defs.node_type] 
                if not drop_edge_attrs 
                else [])
            + [defs.edge_type, defs.source, defs.destination, defs.event_id]  # noqa: W503
            + ([defs.category] if is_using_categories else []) ))  # noqa: W503
        if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
            # subframes = [ df.persist() for df in subframes ]
            for df in subframes:
                logger.debug('format_direct_edges subdf: %s', df.dtypes)
        out = concat(subframes, engine=engine, debug=debug)[ result_cols ]
        if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
            out = out.persist()
            out.compute()
            logger.debug('////format_direct_edges')
        return out
    else:
        return events[:0][[]]


def format_hypernodes(events, defs, drop_na):
    event_nodes = events.copy()
    event_nodes[defs.node_type] = defs.event_id
    event_nodes[defs.category] = defs.event_type
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

 
#

def shallow_copy(df: DataframeLike, engine: Engine, debug: bool = False) -> DataframeLike:
    if engine in [Engine.DASK, Engine.DASK_CUDF]:
        df2 = df.copy()
    else:
        df2 = df.copy(deep=False)

    if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
        df2 = df2.persist()
        df2.compute()

    return df2


def df_coercion(  # noqa: C901
    df: DataframeLike,
    engine: Engine,
    npartitions: Optional[int] = None,
    chunksize: Optional[int] = None,
    debug: bool = False
) -> DataframeLike:
    """
    Go from df to engine of choice

    Supported coercions:
        pd <- pd
        cudf <- pd, cudf
        ddf <- pd, ddf
        dgdf <- pd, cudf, dgdf
    """
    logger.debug('@df_coercion %s -> %s', type(df), engine)

    if engine == Engine.PANDAS:
        if isinstance(df, pd.DataFrame):
            return df
        raise ValueError('pandas engine mode requires pd.DataFrame input, received: %s' % str(type(df)))

    if engine == Engine.CUDF:
        import cudf
        if isinstance(df, pd.DataFrame):
            try:
                return cudf.from_pandas(df)
            except:
                failed_cols = [ ]
                out_gdf = cudf.from_pandas(df[[]])
                for c in df.columns:
                    try:
                        out_gdf[c] = cudf.from_pandas(df[c])
                    except:
                        failed_cols.append(c)
                        out_gdf[c] = cudf.from_pandas(df[c].astype(str))
                logger.warning('CPU->GPU coercion failures on columns, converted their individual values to strings: %s',
                    ', '.join([f'[{c} :: {df[c].dtype.name}]' for c in failed_cols]))
                return out_gdf

        if isinstance(df, cudf.DataFrame):
            return df
        raise ValueError('cudf engine mode requires pd.DataFrame/cudf.DataFrame input, received: %s' % str(type(df)))

    if engine == Engine.DASK:
        import dask.dataframe
        if isinstance(df, pd.DataFrame):
            out = dask.dataframe.from_pandas(df, **{
                **({'npartitions': npartitions} if npartitions is not None else {}) , 
                **({'chunksize': chunksize} if chunksize is not None else {})
            })
            if debug:
                out = out.persist()
                logger.debug('pdf -> ddf: %s', out.compute())
            return out
        if isinstance(df, dask.dataframe.DataFrame):
            return df
        raise ValueError('dask engine mode requires pd.DataFrame/dask.dataframe.DataFrame input, received: %s' % str(type(df)))

    if engine == Engine.DASK_CUDF:
        import cudf, dask_cudf, dask.dataframe
        if isinstance(df, pd.DataFrame):
            gdf = df_coercion(df, Engine.CUDF)
            return df_coercion(gdf, Engine.DASK_CUDF, npartitions=npartitions, chunksize=chunksize)
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
    defs: HyperBindings,
    engine: Engine,
    npartitions: Optional[int] = None,
    chunksize: Optional[int] = None,
    dropna: bool = False,  # FIXME https://github.com/rapidsai/cudf/issues/7735
    debug: bool = False
) -> DataframeLike:
    """
    Copy with reset index and in the target engine format
    """
    logger.debug('@clean: %s', events.dtypes)

    # FIXME https://github.com/rapidsai/cudf/issues/7735
    # None -> np.nan
    if dropna and (engine == Engine.DASK_CUDF):
        import cudf, numpy as np
        if isinstance(events, pd.DataFrame):  # or isinstance(events, cudf.DataFrame):
            events = events.copy()
            for c in events.columns:
                if events[c].dtype.name == 'object' and events[c].isna().any():
                    logger.debug('None -> nan workaround for col [ %s ] => %s', c, events[c])
                    events[c] = events[c].fillna(np.nan)
                    logger.debug('... with drop: %s', events[c].dropna())

    out_events = df_coercion(events, engine, npartitions, chunksize, debug)
    if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
        out_events = out_events.persist()
        logger.debug('coerced events: %s', out_events.compute())

    if engine in [Engine.CUDF, Engine.DASK_CUDF]:
        for c in out_events:
            if out_events[c].dtype.name == 'timedelta64[ns]':
                logger.debug('timedelta concats may conflict when nans; coerce col [ %s ] => str', c)
                out_events[c] = out_events[c].astype(str)
    
    out_events = shallow_copy(out_events, engine)

    out_events = out_events.reset_index(drop=True)
    if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
        out_events = out_events.persist()
        logger.debug('copied events: %s', out_events.compute())

    if defs.event_id in events.columns:
        out_events[defs.event_id] = (defs.event_id + defs.delim) + out_events[defs.event_id].astype(str).fillna(defs.null_val) 
    else:
        out_events[defs.event_id] = (defs.event_id + defs.delim) + out_events[[]].reset_index()['index'].astype(str)
    if debug and (engine in [Engine.DASK, Engine.DASK_CUDF]):
        out_events = out_events.persist()
        logger.debug('tagged events: %s', out_events.compute())
        logger.debug('////clean_events')

    logger.debug('////clean: %s', out_events.dtypes)
    return out_events


class Hypergraph():
    def __init__(
        self, g,
        defs, entities: DataframeLike, event_entities: DataframeLike, edges: DataframeLike,
        source: str, destination: str,
        engine: Engine = Engine.PANDAS, debug: bool = False
    ):
        self.engine = engine
        self.entities = entities
        self.events = event_entities
        self.edges = edges
        logger.debug('final nodes dtypes - entities: %s', entities.dtypes)
        logger.debug('final nodes dtypes - event_entities: %s', event_entities.dtypes)
        self.nodes = concat([entities, event_entities], engine=engine, debug=debug)
        if debug and engine in [Engine.DASK, Engine.DASK_CUDF]:
            self.nodes = self.nodes.persist()
            self.nodes.compute()
            logger.debug('////Hypergraph nodes')
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
    chunksize: Optional[int] = None,
    debug: bool = False
):
    """
    Internal details:
        - IDs currently strings: `${namespace(col)}${delim}${str(val)}`
        - debug: sprinkle persist() to catch bugs earlier
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
    events = clean_events(raw_events, defs, dropna=drop_na, engine=engine_resolved, npartitions=npartitions, chunksize=chunksize, debug=debug)  # type: ignore
    if debug and (engine in [ Engine.DASK, Engine.DASK_CUDF ]):
        logger.debug('==== events: %s', events.compute())
    
    entities = format_entities(events, entity_types, defs, direct, drop_na, engine_resolved, npartitions, chunksize, debug)  # type: ignore

    event_entities = None
    edges = None
    if direct:
        edge_shape = direct_edgelist_shape(entity_types, defs)
        event_entities = df_coercion(mt_nodes(defs, events, entity_types, direct, engine_resolved), engine_resolved, npartitions=1)
        if debug:
            logger.debug('mt event_entities: %s', event_entities.dtypes)
        edges = format_direct_edges(engine_resolved, events, entity_types, defs, edge_shape, drop_na, drop_edge_attrs, debug)
    else:        
        event_entities = format_hypernodes(events, defs, drop_na)
        edges = format_hyperedges(engine_resolved, events, entity_types, defs, drop_na, drop_edge_attrs, debug)

    if debug:
        logger.debug('==== edges: %s', edges.compute() if engine_resolved in [Engine.DASK, Engine.DASK_CUDF] else edges)

    if verbose:
        print('# links', len(edges))
        print('# events', len(events))
        print('# attrib entities', len(entities))
    return Hypergraph(
        g,
        defs, entities, event_entities, edges,
        defs.source if direct else defs.attrib_id,
        defs.destination if direct else defs.event_id,
        engine_resolved,
        debug)

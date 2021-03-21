import logging, pandas as pd
from datetime import datetime
from .pygraphistry import util

logger = logging.getLogger(__name__)

node_id_key = u'_bolt_node_id_key'
node_type_key = u'type'
node_label_prefix_key = u'_lbl_'
start_node_id_key = u'_bolt_start_node_id_key'
end_node_id_key = u'_bolt_end_node_id_key'
relationship_id_key = u'_bolt_relationship_id'
relationship_type_key = u'type'

t0 = datetime.min.time()

try:
    import neo4j
except:
    pass


def to_bolt_driver(driver=None):
    if driver is None:
        return None
    try:
        from neo4j import GraphDatabase, Driver
        if isinstance(driver, Driver):
            return driver
        return GraphDatabase.driver(**driver)
    except ImportError:
        raise BoltSupportModuleNotFound()

#TODO catch additional encodings
def bolt_graph_to_edges_dataframe(graph):
    df = pd.DataFrame([
        util.merge_two_dicts(
            { key: value for (key, value) in relationship.items() },
            {
                relationship_id_key:   relationship.id,  # noqa: E241
                relationship_type_key: relationship.type,  # noqa: E241
                start_node_id_key:     relationship.start_node.id,  # noqa: E241
                end_node_id_key:       relationship.end_node.id  # noqa: E241
            }
        )
        for relationship in graph.relationships
    ])
    if len(df) == 0:
        util.warn('Query returned no edges; may have surprising visual results or need to add missing columns for encodings')
        return pd.DataFrame({
            relationship_id_key: pd.Series([], dtype='int32'),
            relationship_type_key: pd.Series([], dtype='int32'),
            start_node_id_key: pd.Series([], dtype='int32'),
            end_node_id_key: pd.Series([], dtype='int32')
        })
    return neo_df_to_pd_df(df)


def bolt_graph_to_nodes_dataframe(graph) -> pd.DataFrame:
    df = pd.DataFrame([
        util.merge_two_dicts(
            { key: value for (key, value) in node.items() },
            util.merge_two_dicts(
                { 
                    node_id_key: node.id, 
                    node_type_key: ",".join(sorted([str(label) for label in node.labels])) 
                },
                { node_label_prefix_key + str(label): True for label in node.labels }))
        for node in graph.nodes
    ])
    if len(df) == 0:
        util.warn('Query returned no nodes')
        return pd.DataFrame({
            node_id_key: pd.Series([], dtype='int32'),
            node_type_key: pd.Series([], dtype='object')
        })
    return neo_df_to_pd_df(df)


# Knowing a col is all-spatial, flatten into primitive cols
def flatten_spatial_col(df : pd.DataFrame, col : str) -> pd.DataFrame:  # noqa: C901

    out_df : pd.DataFrame = df.copy(deep=False)  # type: ignore

    ####

    #TODO: Can we do better than duck typing the spatial fields?
    try:
        out_df[f'{col}_x'] = df[col].apply(lambda v: None if v is None else v.x)
    except:
        pass

    try:
        out_df[f'{col}_y'] = df[col].apply(lambda v: None if v is None else v.y)
    except:
        pass

    try:
        out_df[f'{col}_z'] = df[col].apply(lambda v: None if v is None else v.z)
    except:
        pass

    try:
        out_df[f'{col}_srid'] = df[col].apply(lambda v: None if v is None else v.srid)
    except:
        pass

    try:
        out_df[f'{col}_longitude'] = df[col].apply(lambda v: None if v is None else v.longitude)
    except:
        pass

    try:
        out_df[f'{col}_latitude'] = df[col].apply(lambda v: None if v is None else v.latitude)
    except:
        pass

    ###

    out_df[col] = df[col].apply(str)

    return out_df





#dtype='obj' -> 'a
def neo_val_to_pd_val(v):

    if v is None:
        return v

    try:
        v_mod = v.__module__
    except:
        return v

    #neo4j 3
    if v_mod == 'neotime':
        return str(v)

    #neo4j 4
    if v_mod == 'neo4j.time':
        if v.__class__ == neo4j.time.DateTime:
            return v.to_native()  # datetime.datetime
        elif v.__class__ == neo4j.time.Date:
            return datetime.combine(v.to_native(), t0)  # datatime.datatime
        elif v.__class__ == neo4j.time.Time:
            return pd.to_timedelta(v.iso_format())  # timedelta
        elif v.__class__ == neo4j.time.Duration:
            #TODO expand out?
            return v.iso_format()  # str
        else:
            return str(v)

    #handle neo4j.spatial.* later

    return v


def stringify_spatial(v):
    if v is None:
        return None
    if isinstance(v, neo4j.spatial.Point):
        ##TODO rep as JSON / dict?
        return str(v)
    return v


def get_mod(v):
    try:
        return v.__module__
    except:
        return None


# if a col has spatials:
#   - all: flatten into new primitive cols
#   - some: stringify
def flatten_spatial(df : pd.DataFrame, col : str) -> pd.DataFrame:

    any_spatial = (df[col].apply(get_mod) == 'neo4j.spatial').any()  # type: ignore
    if not any_spatial:
        return df

    with_vals : pd.Series = df[col].dropna()  # type: ignore
    if len(with_vals) == 0:  # type: ignore
        return df

    out_df : pd.DataFrame = df.copy(deep=False)  # type: ignore

    t0 = with_vals[0]  # type: ignore
    try:
        all_t0 = (with_vals.apply(lambda s: s.__class__) == t0.__class__).all()  # type: ignore
    except:
        all_t0 = False
    
    if all_t0:
        out_df = flatten_spatial_col(df, col)
    else:
        out_df[col] = df[col].apply(stringify_spatial)
  
    return out_df


def neo_df_to_pd_df(df: pd.DataFrame) -> pd.DataFrame:
    out_df : pd.DataFrame = df.copy(deep=False)  # type: ignore
    for col in df:
        if df[col].dtype.name == 'object':
            out_df[col] = df[col].apply(neo_val_to_pd_val)
            out_df = flatten_spatial(out_df, col)
    return out_df


class BoltSupportModuleNotFound(Exception):
    def __init__(self):
        super(BoltSupportModuleNotFound, self).__init__(
            "The neo4j module was not found but is required for pygraphistry bolt support. Try running `!pip install --user graphistry[bolt]`."
        )

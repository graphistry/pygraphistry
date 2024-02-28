from typing import Union, Optional, List
import copy, logging, pandas as pd, numpy as np

from graphistry.PlotterBase import Plottable

logger = logging.getLogger("collapse")
#logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# best for development or debugging
consoleHandler = logging.StreamHandler()
#consoleHandler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(': %(message)s')

# add formatter to ch
consoleHandler.setFormatter(formatter)

# add ch to logger
logger.addHandler(consoleHandler)

COLLAPSE_NODE = "node_collapse"
COLLAPSE_SRC = "src_collapse"
COLLAPSE_DST = "dst_collapse"
FINAL_NODE = 'node_final'
FINAL_SRC = 'src_final'
FINAL_DST = 'dst_final'
WRAP = "~"
DEFAULT_VAL = "None"
VERBOSE = False


def unpack(g: Plottable):
    """Helper method that unpacks graphistry instance

    ex:

        ndf, edf, src, dst, node = unpack(g)

    :param g: graphistry instance

    :returns: node DataFrame, edge DataFrame, source column, destination column, node column
    """
    ndf = g._nodes
    edf = g._edges
    src = g._source
    dst = g._destination
    node = g._node
    return ndf, edf, src, dst, node


def get_children(g: Plottable, node_id: Union[str, int], hops: int = 1):
    """Helper that gets children at k-hops from node `node_id`

    :returns graphistry instance of hops
    """
    g2 = g.hop(pd.DataFrame({g._node: [node_id]}), hops=hops)
    return g2


def has_edge(
    g: Plottable, n1: Union[str, int], n2: Union[str, int], directed: bool = True
) -> bool:
    """Checks if `n1` and `n2` share an (directed or not) edge

    :param g: graphistry instance
    :param n1: `node` to check if has edge to `n2`
    :param n2: `node` to check if has edge to `n1`
    :param directed: bool, if True, checks only outgoing edges from `n1`->`n2`, else finds undirected edges

    :returns: bool, if edge exists between `n1` and `n2`
    """
    ndf, edf, src, dst, node = unpack(g)
    if directed:
        if n2 in edf[edf[src] == n1][dst].values:
            return True
    else:
        if (n2 in edf[edf[src] == n1][dst].values) or (
            n1 in edf[edf[src] == n2][dst].values
        ):
            return True
    return False


def get_edges_of_node(
    g: Plottable, node_id: Union[str, int], outgoing_edges: bool = True, hops: int = 1
):
    """Gets edges of node at k-hops from node

    :param g: graphistry instance
    :param node_id: `node` to find edges from
    :param outgoing_edges: bool, if true, finds all outgoing edges of `node`, default True
    :param hops: the number of hops from `node` to take, default = 1

    :returns: DataFrame of edges
    """
    _, _, src, dst, _ = unpack(g)
    g2 = get_children(g, node_id, hops=hops)
    if outgoing_edges:
        edges = g2._edges[dst].drop_duplicates()
    else:
        edges = pd.concat([g2._edges[src], g2._edges[dst]]).drop_duplicates()
    return edges


def get_edges_in_out_cluster(
    g: Plottable,
    node_id: Union[str, int],
    attribute: Union[str, int],
    column: Union[str, int],
    directed: bool = True,
):
    """Traverses children of `node_id` and separates them into incluster and outcluster sets depending if they have `attribute` in node DataFrame `column`

    :param g: graphistry instance
    :param node_id: `node` with `attribute` in `column`
    :param attribute: `attribute` to collapse in `column` over
    :param column: `column` to collapse over
    :param directed:
    """
    g2 = get_children(g, node_id, hops=1)
    e = get_edges_of_node(
        g, node_id, outgoing_edges=directed
    )  # False just includes the src node
    ndf, edf, src, dst, node = unpack(g2)
    tdf = ndf[ndf[column] == attribute]
    if not tdf.empty:
        # Get edges that are not in attribute (outside edges)
        outcluster = set(e.values).difference(set(tdf.node.values))
        # get edges that are internal to attribute, we will use these later to collapse those edges to supernode
        incluster = set(tdf.node.values).intersection(set(e.values))
        if VERBOSE:
            if len(outcluster):
                logger.info(
                    f"{outcluster} are edges *not* in [[ {column}:{attribute} ]] for node {node_id}"
                )
                # get directionality
            if len(incluster):
                logger.info(
                    f"{incluster} are edges in [[ {column}:{attribute} ]] for node {node_id}"
                )
        return outcluster, incluster, tdf
    return None, None, None


def get_cluster_store_keys(ndf: pd.DataFrame, node: Union[str, int]):
    """Main innovation in finding and adding to super node. Checks if node is a segment in any collapse_node in COLLAPSE column of nodes DataFrame

    :param ndf: node DataFrame
    :param node: node to find

    :returns: DataFrame of bools of where `wrap_key(node)` exists in COLLAPSE column
    """
    node = wrap_key(node)
    return ndf[COLLAPSE_NODE].astype(str).str.contains(node, na=False)


def in_cluster_store_keys(ndf: pd.DataFrame, node: Union[str, int]) -> bool:
    """checks if node is in collapse_node in COLLAPSE column of nodes DataFrame

    :param ndf: nodes DataFrame
    :param node: node to find

    :returns: bool
    """
    return any(get_cluster_store_keys(ndf, node))


def reduce_key(key: Union[str, int]) -> str:
    """Takes "1 1 2 1 2 3" -> "1 2 3

    :param key: node name

    :returns: new node name with duplicates removed
    """
    uniques = " ".join(np.unique(str(key).split()))
    return uniques


def unwrap_key(name: Union[str, int]) -> str:
    """Unwraps node name: ~name~ -> name

    :param name: node to unwrap

    :returns: unwrapped node name
    """
    return str(name).replace(WRAP, "")


def wrap_key(name: Union[str, int]) -> str:
    """Wraps node name -> ~name~

    :param name: node name

    :returns: wrapped node name
    """

    name = str(name)
    if WRAP in name:  # idempotency
        return name
    return f"{WRAP}{name}{WRAP}"


def melt(ndf: pd.DataFrame, node: Union[str, int]) -> str:
    """Reduces node if in cluster store, otherwise passes it through.
    ex:

        node = "4" will take any sequence from get_cluster_store_keys, "1 2 3", "4 3 6" and returns "1 2 3 4 6"
        when they have a common entry (3).

    :param ndf, node DataFrame
    :param node: node to melt
    :returns new_parent_name of super node

    """
    rdf = ndf[get_cluster_store_keys(ndf, node)]
    topkey = wrap_key(node)
    if not rdf.empty:
        for key in rdf[COLLAPSE_NODE].values:  # all these are already wrapped
            # add the keys to cluster
            topkey += f" {key}"  # keep whitespace
        topkey = reduce_key(topkey)
    return topkey

def check_has_set(ndf, parent, child):

    ckey = in_cluster_store_keys(ndf, child)
    pkey = in_cluster_store_keys(ndf, parent)

    if ckey and pkey:
        return True
    return False


def get_new_node_name(
    ndf: pd.DataFrame, parent: Union[str, int], child: Union[str, int]
) -> str:
    """If child in cluster group, melts name, else makes new parent_name from parent, child

    :param ndf: node DataFrame
    :param parent: `node` with `attribute` in `column`
    :param child: `node` with `attribute` in `column`

    :returns new_parent_name
    """
    # THIS IS IMPORTANT FUNCTION -- it is where we wrap the parent/child in WRAP
    # if child in cluster group, we melt it
    ckey = in_cluster_store_keys(ndf, child)
    pkey = in_cluster_store_keys(ndf, parent)
    # new_parent_name = wrap_key(parent)
    if ckey and pkey:
        new_parent_name = melt(ndf, child)
        new_parent_name = f"{new_parent_name} {wrap_key(parent)}"

    else:  # if not, then append child to parent as the start of a new cluster group
        new_parent_name = melt(ndf, parent)
        new_parent_name = f"{new_parent_name} {wrap_key(child)}"
    if VERBOSE:
        logger.info(
            f"Renaming (parent:{parent}:{pkey}, child:{child}:{ckey})  ->  {new_parent_name}"
        )
    return reduce_key(new_parent_name)



def collapse_nodes_and_edges(
    g: Plottable, parent: Union[str, int], child: Union[str, int]
):
    """
        Asserts that parent and child node in ndf should be collapsed into super node.
        Sets new ndf with COLLAPSE nodes in graphistry instance g

        # this asserts that we SHOULD merge parent and child as super node
        # outside logic controls when that is the case
        # for example, it assumes parent is already in cluster keys of COLLAPSE node

    :param g: graphistry instance
    :param parent: `node` with `attribute` in `column`
    :param child: `node` with `attribute` in `column`
    :returns: graphistry instance
    """

    ndf, edf, src, dst, node = unpack(g)
    
    new_parent_name = get_new_node_name(ndf, parent, child)

    ndf.loc[ndf[node] == parent, COLLAPSE_NODE] = new_parent_name
    ndf.loc[ndf[node] == child, COLLAPSE_NODE] = new_parent_name

    edf.loc[edf[src] == parent, COLLAPSE_SRC] = new_parent_name
    edf.loc[edf[dst] == parent, COLLAPSE_DST] = new_parent_name

    edf.loc[edf[src] == child, COLLAPSE_SRC] = new_parent_name
    edf.loc[edf[dst] == child, COLLAPSE_DST] = new_parent_name

    g._edges = edf
    g._nodes = ndf


def has_property(
    g: Plottable, ref_node: Union[str, int], attribute: Union[str, int], column: Union[str, int]
) -> bool:
    """Checks if ref_node is in node dataframe in column with attribute
    :param attribute:
    :param column:
    :param g: graphistry instance
    :param ref_node: `node` to check if it as `attribute` in `column`

    :returns: bool
    """
    ndf, edf, src, dst, node = unpack(g)
    ref_node = unwrap_key(ref_node)
    return ref_node in ndf[ndf[column] == attribute][node].values


def check_default_columns_present_and_coerce_to_string(g: Plottable):
    """Helper to set COLLAPSE columns to nodes and edges dataframe, while converting src, dst, node to dtype(str)
    :param g: graphistry instance

    :returns: graphistry instance
    """
    ndf, edf, src, dst, node = unpack(g)
    if COLLAPSE_NODE not in ndf.columns:
        ndf[COLLAPSE_NODE] = DEFAULT_VAL
        ndf[node] = ndf[node].astype(str)
        logger.info(f"Converted ndf to type({type(ndf[node].values[0])})")
        g._nodes = ndf
    if COLLAPSE_SRC not in edf.columns:
        edf[COLLAPSE_SRC] = DEFAULT_VAL
        edf[COLLAPSE_DST] = DEFAULT_VAL
        edf[src] = edf[src].astype(str)
        edf[dst] = edf[dst].astype(str)
        logger.info(f"Converted edf to type({type(edf[src].values[0])})")
        g._edges = edf
    return g


def collapse_algo(
    g: Plottable,
    child: Union[str, int],
    parent: Union[str, int],
    attribute: Union[str, int],
    column: Union[str, int],
    seen: dict,
):
    """Basically candy crush over graph properties in a topology aware manner

        Checks to see if child node has desired property from parent, we will need to check if (start_node=parent: has_attribute , children nodes: has_attribute) by case (T, T), (F, T), (T, F) and (F, F),we start recursive collapse (or not) on the children, reassigning nodes and edges.

        if (T, T), append children nodes to start_node, re-assign the name of the node, and update the edge table with new name,

        if (F, T) start k-(potentially new) super nodes, with k the number of children of start_node. Start node keeps k outgoing edges.

        if (T, F) it is the end of the cluster, and we keep new node as is; keep going

        if (F, F); keep going
  
    :param seen:
    :param g: graphistry instance
    :param child: child node to start traversal, for first traversal, set child=parent or vice versa.
    :param parent: parent node to start traversal, in main call, this is set to child.
    :param attribute: attribute to collapse by
    :param column: column in nodes dataframe to collapse over.

    :returns: graphistry instance with collapsed nodes.
    """

    compute_key = f"{parent} {child}"
    
    if compute_key in seen:  # it has already traversed this path, skip
        return g
    else:
        if has_property(g, parent, attribute, column):  # if (T, *)
            # add start node to super node index
            tkey = f"{parent} {parent}"  # it will reduce this to `parent` but can add to `seen`
            if tkey not in compute_key:  # its love!
                seen[tkey] = 1
                collapse_nodes_and_edges(g, parent, parent)
            if has_property(g, child, attribute, column):  # if (T, T)
                if VERBOSE:
                    logger.info("-" * 80)
                    logger.info(
                        f" ** [ parent: {parent}, child: {child} ] both have property"
                    )
                collapse_nodes_and_edges(
                    g, parent, child
                )  # will make a new parent off of parent, child names
                # add to seen
                seen[compute_key] = 1
                for e in get_edges_of_node(
                    g, parent, outgoing_edges=True, hops=1
                ).values:  # False just includes the child node and goes into infinite loop when parent = child
                    collapse_algo(
                        g, e, child, attribute, column, seen
                    )  # now child is the parent, and the edges are the start node
        # else do nothing collapse-y to parent, move on to child
        else:  # if (F, *)
            #  do nothing to child, parent is child, and child is edge and recurse
            for e in get_edges_of_node(g, child, outgoing_edges=True, hops=1).values:
                if VERBOSE:
                    logger.info(
                        f" -- Parent {parent} does not have property, looking at node <[ {e} from {child} ]>"
                    )

                if (e == child) and (parent == child):
                    # get it unstuck
                    return g
                
                collapse_algo(
                    g, e, child, attribute, column, seen
                )  # now child is the parent, and the edges are the start node
    return g


def normalize_graph(
    g: Plottable,
    self_edges: bool = False,
    unwrap: bool = False
) -> Plottable:
    """Final step after collapse traversals are done, removes duplicates and moves COLLAPSE columns into respective(node, src, dst) columns of node, edges dataframe from Graphistry instance g.

    :param g: graphistry instance
    :param self_edges: bool, whether to keep duplicates from ndf, edf, default False
    :param unwrap: bool, whether to unwrap node text with `~`, default True

    :returns: final graphistry instance
    """

    ndf, edf, src, dst, node = unpack(g)

    # we set src and COLLAPSE to FINAL so that we can run algo idempotently and in chain without having to know new node ids
    # move the new node names from COLLAPSE COL to the FINAL COLS
    ndf.loc[ndf[COLLAPSE_NODE] != DEFAULT_VAL, FINAL_NODE] = ndf.loc[
        ndf[COLLAPSE_NODE] != DEFAULT_VAL, COLLAPSE_NODE
    ]
    # set the untouched nodes to FINAL
    ndf.loc[ndf[COLLAPSE_NODE] == DEFAULT_VAL, FINAL_NODE] = ndf.loc[
        ndf[COLLAPSE_NODE] == DEFAULT_VAL, node
    ]
    ndf = ndf.drop_duplicates()

    # set the new data in FINAL edges
    edf.loc[edf[COLLAPSE_SRC] != DEFAULT_VAL, FINAL_SRC] = edf.loc[
        edf[COLLAPSE_SRC] != DEFAULT_VAL, COLLAPSE_SRC
    ]
    edf.loc[edf[COLLAPSE_DST] != DEFAULT_VAL, FINAL_DST] = edf.loc[
        edf[COLLAPSE_DST] != DEFAULT_VAL, COLLAPSE_DST
    ]
    # set the old data in FINAL edges
    edf.loc[edf[COLLAPSE_SRC] == DEFAULT_VAL, FINAL_SRC] = edf.loc[
        edf[COLLAPSE_SRC] == DEFAULT_VAL, src
    ]
    edf.loc[edf[COLLAPSE_DST] == DEFAULT_VAL, FINAL_DST] = edf.loc[
        edf[COLLAPSE_DST] == DEFAULT_VAL, dst
    ]
    
    if not self_edges:
        edf = edf.drop_duplicates()

    if unwrap:  # this is only to make things more readable.
        ndf[FINAL_NODE] = ndf[FINAL_NODE].astype(str).apply(lambda x: unwrap_key(x))
        edf[FINAL_SRC] = edf[FINAL_SRC].astype(str).apply(lambda x: unwrap_key(x))
        edf[FINAL_DST] = edf[FINAL_DST].astype(str).apply(lambda x: unwrap_key(x))

    # set the dataframes according to FINAL nodes
    g._nodes = ndf
    g._edges = edf
    g._node = FINAL_NODE
    g._source = FINAL_SRC
    g._destination = FINAL_DST
    return g


def collapse_by(
    self: Plottable,
    parent: Union[str, int],
    start_node: Union[str, int],
    attribute: Union[str, int],
    column: Union[str, int],
    seen: dict,
    self_edges: bool = False,
    unwrap: bool = False,
    verbose: bool = True
) -> Plottable:
    """
        Main call in collapse.py, collapses nodes and edges by attribute, and returns normalized graphistry object.

    :param self: graphistry instance
    :param parent: parent node to start traversal, in main call, this is set to child.
    :param start_node:
    :param attribute: attribute to collapse by
    :param column: column in nodes dataframe to collapse over.
    :param seen: dict of previously collapsed pairs -- {n1, n2) is seen as different from (n2, n1)
    :param verbose: bool, default True

    :returns graphistry instance with collapsed and normalized nodes.
    """
    from time import time
    
    
    g = copy.deepcopy(self.bind())
    g = check_default_columns_present_and_coerce_to_string(g)

    n_edges = len(g._edges)
    complexity_min = int(n_edges * np.log(n_edges))
    complexity_max = int(n_edges ** (3 / 2))
    if (VERBOSE or verbose) and n_edges > 5000:
        logger.info("-" * 108)
        logger.info(
            "This Algorithm runs approximately between n_edges*log(n_edges) and n_edges**(3/2) in un-normalized units"
        )
        logger.info(
            f"Hence, in this case, between O({complexity_min/n_edges:.2f} - {complexity_max/n_edges:.2f}) for "
            f"this graph normalized by {n_edges} edges"
        )
        logger.info(
            "It is not recommended for large graphs -- one can expect a modern laptop CPU to scan 1-6k edges per minute"
        )
        logger.info(f"Here we expect collapse to run in under {n_edges/1000:.3f} minutes")
        logger.info("*" * 100)
    t = time()
    
    collapse_algo(g, parent, start_node, attribute, column, seen)
    
    t2 = time()
    delta_mins = (t2 - t) / 60
    if VERBOSE or verbose:
        logger.info("-" * 80)
        logger.info(
            f"Total Collapse took {delta_mins:.2f} minutes or {n_edges/delta_mins:.2f} edges per minute"
        )
    return normalize_graph(
        g, self_edges=self_edges, unwrap=unwrap
    )

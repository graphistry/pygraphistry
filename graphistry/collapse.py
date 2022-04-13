# query WIP
# want to be able to do better graph collapse and query
import pandas as pd
import numpy as np


"""input: g, root

queue = [ root ]
g2 = g

while parent = queue.pop():
  for pbg in parent.children.pbg.unque():
     pbg_children = parent.subgraph({'pbg': pbg}).remove(parent)
     pbg_children_node = as_node(pbg_children)
     g2 = g2.collapse(pbg_children, pbg_children_node)
     queue.append(pbg_children_node)

return g2      """


COLLAPSE_NODE = "collapse_node"
COLLAPSE_SRC = "collapse_src"
COLLAPSE_DST = "collapse_dst"
WRAP = '~'

VERBOSE = True


def unpack(g):
    #  ndf, edf, src, dst, node = unpack(g)
    ndf = g._nodes
    edf = g._edges
    src = g._source
    dst = g._destination
    node = g._node
    return ndf, edf, src, dst, node


def get_children(g, node_id, hops=1):
    g2 = g.hop(pd.DataFrame({g._node: [node_id]}), hops=hops)
    return g2


def has_edge(g, n1, n2, directed=True):
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


def get_edges_of_node(g, node_id, directed=True, hops=1):
    _, _, src, dst, _ = unpack(g)
    g2 = get_children(g, node_id, hops=hops)
    if directed:
        edges = g2._edges[dst]
    else:
        edges = pd.concat([g2._edges[src], g2._edges[dst]]).drop_duplicates()
    return edges


def is_outgoing(e, edf, src, dst):
    # edf is understood to be the local edf, not global
    # returns edge direction outgoing, or not
    if e in edf[src].values:
        return False
    if e in edf[dst].values:
        return True


def remove_edges(g, edgelist):
    # src_edges, dst_edges = np.array(edgelist).T
    ndf, edf, src, dst, node = unpack(g)
    tedf = edf[~edf[src].isin(edgelist) & ~edf[dst].isin(edgelist)]
    return tedf


def get_edges_in_out_cluster(
    g, node_id, attribute=1, attribute_col="level", directed=False
):
    g2 = get_children(g, node_id, hops=1)
    e = get_edges_of_node(
        g, node_id, directed=directed
    )  # False just includes the src node
    ndf, edf, src, dst, node = unpack(g2)
    tdf = ndf[ndf[attribute_col] == attribute]
    if not tdf.empty:
        # Get edges that are not in attribute (outside edges)
        outcluster = set(e.values).difference(set(tdf.node.values))
        # get edges that are internal to attribute, we will use these later to collapse those edges to supernode
        incluster = set(tdf.node.values).intersection(set(e.values))
        if VERBOSE:
            if len(outcluster):
                print(
                    f"{outcluster} are edges *not* in {attribute_col}:{attribute} for node {node_id}"
                )
                # get directionality
            if len(incluster):
                print(
                    f"{incluster} are edges in {attribute_col}:{attribute} for node {node_id}"
                )
        return outcluster, incluster, tdf

    # else: # if tdf is empty, we are at a parent node without the collapsable property
    # print(f'{node_id} did not have any attributes')
    return None, None, None


def in_cluster_store(df, node):
    return node in df[COLLAPSE_NODE].values


def get_cluster_store_keys(df, node):
    # checks if node is a segment in any collapse_node
    # this is where you wrap the keys
    node = wrap_key(node)
    return df[COLLAPSE_NODE].astype(str).str.contains(node, na=False)


def in_cluster_store_keys(df, node):
    return any(get_cluster_store_keys(df, node))


def reduce_key(key):
    # takes "1 1 2 1 2 3" -> "1 2 3"
    uniques = " ".join(np.unique(key.split()))
    return uniques


def unwrap_key(name):
    return name.replace(WRAP, '')

def wrap_key(name):
    name = str(name)
    if WRAP in name:
        return name
    return f'{WRAP}{name}{WRAP}'

def melt(df, node):
    """MAIN AWESOME"""
    # suppose node = "4" will take any sequence from get_cluster_store_keys, "1 2 3", "4 3" and returns "1 2 3 4"
    rdf = df[get_cluster_store_keys(df, node)]
    topkey = wrap_key(node)
    if not rdf.empty:
        for key in rdf[COLLAPSE_NODE].values: # all these are already wrapped
            # print(f'collapse key iter: {key}')
            # add the keys to cluster
            topkey += f" {key}" # keep whitespace
    topkey = reduce_key(topkey)
    if VERBOSE:
        print(f"start of collapse: {node}")
        print(f"Collapse Final Key: {topkey}")
    return topkey


def get_new_node_name(ndf, parent, child) -> str:
    # THIS IS IMPORTANT FUNCTION -- it is where we wrap the parent child in WRAP
    # if child in cluster group, we melt it
    ckey = in_cluster_store_keys(ndf, child)

    if ckey:  # and pkey: # pkey should be redundant if recursion has done its job
        new_parent_name = melt(ndf, child)
    else:  # if not, then append child to parent as the start of a new cluster group
        # might have to escape parent and child if node names are dumb eg, 'this value key'
        new_parent_name = f"{wrap_key(parent)} {wrap_key(child)}"
    if VERBOSE:
        print(f"Renaming parent {parent} with child {child} as {new_parent_name}")
    return reduce_key(new_parent_name)


def collapse_nodes(g, parent, child):
    # this asserts that we SHOULD merge parent and child as super node
    # outside logic controls when that is the case
 
    ndf, edf, src, dst, node = unpack(g)

    new_parent_name = get_new_node_name(ndf, parent, child)
    
    ndf.loc[ndf[node] == parent, COLLAPSE_NODE] = new_parent_name
    ndf.loc[ndf[node] == child, COLLAPSE_NODE] = new_parent_name

    g._nodes = ndf
    return g, new_parent_name


def collapse_edges_and_nodes(g, parent, attribute, column, new_parent_name):
    # get out of cluster and in cluster nodes from parent node
    # then feed incluster nodes to new_parent_name
    outcluster, incluster, tdf = get_edges_in_out_cluster(g, parent, attribute, column)
    # keep incluster nodes and assign them to new parent
    ndf, edf, src, dst, node = unpack(g)

    for node in incluster:
        # so we don't corrupt the OG src dst table
        edf.loc[edf[src] == parent, COLLAPSE_SRC] = new_parent_name
        edf.loc[edf[dst] == parent, COLLAPSE_DST] = new_parent_name

        edf.loc[edf[dst] == node, COLLAPSE_DST] = new_parent_name
        edf.loc[edf[src] == node, COLLAPSE_SRC] = new_parent_name

    g._edges = edf
    return g


def has_property(g, ref_node, attribute, column):
    ndf, edf, src, dst, node = unpack(g)
    ref_node = unwrap_key(ref_node)
    return ref_node in ndf[ndf[column] == attribute][node].values


def check_default_columns_present(g):
    ndf, edf, src, dst, node = unpack(g)
    if COLLAPSE_NODE not in ndf.columns:
        ndf[COLLAPSE_NODE] = "None"
        g._nodes = ndf
    if COLLAPSE_SRC not in edf.columns:
        edf[COLLAPSE_SRC] = "None"
        edf[COLLAPSE_DST] = "None"
        g._edges = edf
    return g


def collapse(g, child, parent, attribute, column, seen):
    """
    Basically candy crush over graph properties in a topology aware situation
    
    basic ingress, use
    
    >>> seen={}
    >>> collapse(g, parent, parent, attribute, column, seen=seen)
    
    where parent and child are same node.
    
    see if child node has desired property (can be a new node without attribute, a node with attribute, and a new collapsed node with attribute)
    we will need to check if (start_node: has_attribute , children nodes: has_attribute) by case (T, T) and (F, T), (T, F) and (F, F) and split them,
     so we start recursive the collapse (or not) on the children, reassigning nodes and edges.
    if (T, T), append children nodes to start_node, re-assign the name of the node, and update the edge table with new name,
    if (F, T) start k-new super nodes, with k the number of children of start_node. Start node keeps k outgoing edges.
    if (T, F) is the end of the cluster, and should keep new node as is
    if (F, F) keep going
    """
    g = check_default_columns_present(g)
    
    # parent = wrap_key(parent)
    # child = wrap_key(child)
    
    compute_key = f"{parent} {child}"

    if compute_key in seen:  # it has already traversed this path, skip
        return g
    else:
        if has_property(g, parent, attribute, column):  # if (T, *)
            # add start node to super node index
            # g, rename = collapse_nodes(g, parent, start_node)
            # g = collapse_edges_and_nodes(g, parent, attribute, column)
            tkey = f"{parent} {parent}"
            if tkey not in compute_key:  # its love!
                seen[tkey] = 1
                g, new_parent_name = collapse_nodes(g, parent, parent) # will make a new parent key'd off of parent name
            if has_property(g, child, attribute, column):# and has_edge(g, parent, child):  # if (T, T)
                # now add to seen
                seen[compute_key] = 1
                if VERBOSE:
                    print("-" * 80)
                    print(
                        f" ** [ parent: {parent}, child: {child} ] both have property"
                    )
                g, new_parent_name = collapse_nodes(g, parent, child) # will make a new parent off of parent, child names
                # like simplicial collapse
                g = collapse_edges_and_nodes(
                    g, parent, attribute, column, new_parent_name
                )
                for e in get_edges_of_node(
                    g, parent, directed=True
                ).values:  # False just includes the src node
                    collapse(
                        g, e, child, attribute, column, seen
                    )  # now start_node is the parent, and the edges are the start node
        # else do nothing collapse-y to parent, move on
        else:  # if (F, *)
            #  do nothing to start_node, parent is start_node, and start_node is edge and recurse
            for e in get_edges_of_node(g, child, directed=True).values:
                if VERBOSE:
                    print(
                        f" -- Parent {parent} does not have property, looking at node <[ {e} from {child} ]>"
                    )
                collapse(
                    g, e, child, attribute, column, seen
                )  # now start_node is the parent, and the edges are the start node
    return g


def normalize_graph(g, self_edges=False):
    # at the end of collapse, move anything untouched to new graph
    ndf, edf, src, dst, node = unpack(g)

    # move the new node names fromo COLLAPSE COL to the node column
    ndf.loc[ndf[COLLAPSE_NODE] != "None", node] = ndf.loc[
        ndf[COLLAPSE_NODE] != "None", COLLAPSE_NODE
    ]
    ndf = ndf.drop_duplicates()

    edf.loc[edf[COLLAPSE_SRC] != "None", src] = edf.loc[
        edf[COLLAPSE_SRC] != "None", COLLAPSE_SRC
    ]
    edf.loc[edf[COLLAPSE_DST] != "None", dst] = edf.loc[
        edf[COLLAPSE_DST] != "None", COLLAPSE_DST
    ]
    if not self_edges:
        edf = edf.drop_duplicates()

    ## convert to str
    ndf[node] = ndf[node].astype(str)
    edf[src] = edf[src].astype(str)
    edf[dst] = edf[dst].astype(str)

    g._nodes = ndf
    g._edges = edf
    return g


def collapse_by(g, parent, start_node, attribute, column, seen):
    collapse(g, parent, start_node, attribute, column, seen)
    return normalize_graph(g)


def brute_collapse(g, start_node, attribute, column, hops=3):
    # redundant?
    seen = {}
    collapse(g, start_node, start_node, attribute, column, seen)
    for n in get_edges_of_node(g, start_node, hops=hops):
        if f"{n} {n}" not in seen:
            collapse(g, n, attribute, column, start_node, seen)
    return normalize_graph(g)


# def filterby(df, attribute, column, negative=False):
#     if negative:
#         tdf = df[df[column] != attribute]
#     else:
#         tdf = df[df[column] == attribute]
#     return tdf
#
#
# def splitby(df, attribute, column):
#     pos, neg = filterby(df, attribute, column, negative=False), filterby(
#         df, attribute, column, negative=True
#     )
#     return pos, neg
#

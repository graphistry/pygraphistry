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


COLLAPSE_NODE = 'collapse_nodes'
COLLAPSE_SRC_EDGE = 'collapse_src'
COLLAPSE_DST_EDGE = 'collapse_dst'

def _unpack(g):
    #  ndf, edf, src, dst, node = _unpack(g)
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
    ndf, edf, src, dst, node = _unpack(g)
    if directed:
        if n2 in edf[edf[src] == n1][dst].values:
            return True
    else:
        if (n2 in edf[edf[src] == n1][dst].values) or (n1 in edf[edf[src] == n2][dst].values):
            return True
    return False

def get_edges_of_node(g, node_id, directed=True):
    _, _, src, dst, _ = _unpack(g)
    g2 = get_children(g, node_id, hops=1)
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
    ndf, edf, src, dst, node = _unpack(g)
    tedf = edf[~edf[src].isin(edgelist) & ~edf[dst].isin(edgelist)]
    return tedf

def get_edges_in_out_cluster(g, node_id, attribute_col='level', attribute=1, directed=False):
    g2 = get_children(g, node_id, hops=1)
    e = get_edges_of_node(g, node_id, directed=directed)  # False just includes the src node
    ndf, edf, src, dst, node = _unpack(g2)
    tdf = ndf[ndf[attribute_col] == attribute]
    if not tdf.empty:
        # Get edges that are not in attribute (outside edges)
        outcluster = set(e.values).difference(set(tdf.node.values))
        # get edges that are internal to attribute, we will use these later to collapse those edges to supernode
        incluster = set(tdf.node.values).intersection(set(e.values))
        if len(outcluster):
            print(f'{outcluster} are edges *not* in {attribute_col}:{attribute} for node {node_id}')
            # get directionality
        if len(incluster):
            print(f'{incluster} are edges in {attribute_col}:{attribute} for node {node_id}')
        return outcluster, incluster, tdf
    
    # else: # if tdf is empty, we are at a parent node without the collapsable property
    # print(f'{node_id} did not have any attributes')
    return None, None, None

def in_cluster_store(df, node):
    return node in df[COLLAPSE_NODE].values

def get_cluster_store_keys(df, node):
    # checks if node is a segment in any collapse_node
    return df[COLLAPSE_NODE].str.contains(node)

def in_cluster_store_keys(df, node):
    return any(get_cluster_store_keys(df, node))

def reduce_key(key):
    # takes "1 1 2 1 2 3" -> "1 2 3"
    uniques = ' '.join(np.unique(key.split()))
    return uniques

def melt(df, node):
    """MAIN AWESOME"""
    # suppose node = "4" will take any sequence from get_cluster_store_keys, "1 2 3", "4 3" and returns "1 2 3 4"
    rdf = df[get_cluster_store_keys(df, node)]
    topkey = node
    if not rdf.empty:
        for key in rdf[COLLAPSE_NODE].values:
            # add the keys to cluster
            topkey += f' {key}' # keep whitespace
    return reduce_key(topkey)
    
def get_new_node_name(ndf, parent, child):
    # if child in cluster group, we melt it
    ckey = in_cluster_store_keys(ndf, child)
    
    if ckey:
        new_parent_name = melt(ndf, child)
    else: # if not, then append child to parent as the start of a new cluster group
        # might have to escape parent and child if node names are dumb eg, 'this value key'
        new_parent_name = f'{parent} {child}'
    return new_parent_name


def collapse_nodes(g, parent, child):
    # this asserts that we SHOULD merge parent and child as super node
    # outside logic controls when that is the case
    # we only call this when we
    # the parent is guaranteed to be there
    #pkey = in_cluster_store_keys(ndf, parent)
    ndf, edf, src, dst, node = _unpack(g)

    new_parent_name = get_new_node_name(ndf, parent, child)
    
    rename = {parent: new_parent_name}
    ndf[COLLAPSE_NODE] = ndf[COLLAPSE_NODE].apply(lambda x: rename[x] if x in rename else x)
    g._nodes = ndf
    return g, rename
    

def collapse_edges_and_nodes(g, parent, attribute, column):
    #if has_edge(g, parent, child): # this should be redundant, but good check
    # get out of cluster and in cluster nodes from parent node
    outcluster, incluster, tdf = get_edges_in_out_cluster(g, parent, attribute, column)
    # keep out cluster nodes and assign them to
    # this takes care of outgoing edges
    ndf, edf, src, dst, node = _unpack(g)

    #new_edf2 = remove_edges(g, incluster)
    for node in incluster:
        g, rename = collapse_nodes(g, parent, node)
        # so we don't corrupt the OG src dst table
        edf[COLLAPSE_SRC_EDGE] = edf[COLLAPSE_SRC_EDGE].apply(lambda x: rename[x] if x in rename else x)
        edf[COLLAPSE_DST_EDGE] = edf[COLLAPSE_DST_EDGE].apply(lambda x: rename[x] if x in rename else x)
    g._edges = edf
    return g


def has_property(g, ref_node, attribute, column):
    ndf, edf, src, dst, node = _unpack(g)
    return ref_node in ndf[ndf[column] == attribute][node].values

def collapse(g, start_node, attribute, column, parent): #Basically candy crush over graph properties
    # at ingress use collapse(g, node, attribute, column, node)
    # see if start_node has desired property (start node can be a new node without attribute, a node with attribute, and a new collapsed node with attribute)
    # we will need to check if (start_node: has_attribute , children nodes: has_attribute) by case (T, T) and (F, T), (T, F) and (F, F) and split them,
    #  so we start recursive the collapse (or not) on the children, reassigning nodes and edges.
    # if (T, T), append children nodes to start_node, re-assign the name of the node, and update the edge table with new name,
    # if (F, T) start k-new super nodes, with k the number of children of start_node. Start node keeps k outgoing edges.
    # if (T, F) is the end of the cluster, and should keep new node as is
    # if (F, F) keep going
    if has_property(g, parent, attribute, column): # if (T, *)
        # add start node to super node index
        #g, rename = collapse_nodes(g, parent, start_node)
        if has_property(g, start_node, attribute, column): # if (T, T)
            g = collapse_edges_and_nodes(g, parent, attribute, column)
            #collapse_edges(g, child, start_node, attribute, column)
        # else: # if (T, F)
        #     # recurse over children
        #     for e in get_edges_of_node(g, start_node, directed=True): # False just includes the src node
        #         collapse(g, e, attribute, column, start_node) # now start_node is the parent, and the edges are the start node
    # else do nothing collapsy
    else: # if (F, *)
    #     # do nothing to start_node, and recurse
    # #
    #     g2 = get_children(g, start_node, hops=1)
        for e in get_edges_of_node(g, start_node, directed=True):
            collapse(g, e, attribute, column, start_node)
    return g

    
""":param
We are in the uncanny value of globalism -- war the best means of exploitation. Exploitation serves markets we all touch.
Cheaper labor, minerals, and energy mean gentler hills for procurement and steeper profits.
where some cost function
"""


def filterby(df, attribute, column, negative=False):
    if negative:
        tdf = df[df[column] != attribute]
    else:
        tdf = df[df[column] == attribute]
    return tdf

def splitby(df, attribute, column):
    pos, neg = filterby(df, attribute, column, negative=False), filterby(df, attribute, column, negative=True)
    return pos, neg

    


# try again
# get one hop children starting at node_id
#
def get_edges_in_out_cluster(g, node_id, attribute_col='level', attribute=1, directed=False):
    g2 = get_children(g, node_id, hops=1)
    e = get_edges_of_node(g, node_id, directed=directed)  # False just includes the src node
    ndf, edf, src, dst, node = _unpack(g2)
    tdf = ndf[ndf[attribute_col] == attribute]
    nodes = tdf[node].values
    

    
    notdf = ndf[ndf[attribute_col] != attribute] # nodes not attribute
    
    if not tdf.empty and not e.empty:




def collapse(g, attribute, attribute_col, new_node_name=None, start_node=None, directed=False):
    new_edgelist = []
    ndf, edf, src, dst, node = _unpack(g)
    
    if start_node is None:
        nodes = ndf[node]
    else:
        nodes = [start_node]
    
    if new_node_name is None:
        if node_type(nodes) is str:
            new_node_name = 'collapsed'
        else:
            new_node_name = np.max(nodes) + 1
    
    incluster = []
    for n in nodes:  # iterate over nodes
        # we want to find the edges that are/not connected to attribute 'cluster'
        outclust, inclust, _, etdf = get_edges_in_out_cluster(gg, n, attribute=attribute, attribute_col=attribute_col,
                                                              directed=directed)
        if outclust is not None and len(outclust):
            # outcluster[attribute].update(d)
            for out_edge in outclust:
                print(f'edge: {out_edge}')
                outgoing = is_outgoing(out_edge, etdf, src, dst)
                if outgoing:
                    new_edgelist.append([new_node_name, out_edge])
                else:
                    new_edgelist.append([out_edge, new_node_name])
        if inclust is not None and len(inclust):
            for in_edge in inclust:
                print(f'incluster node: {in_edge}')
                incluster.append(in_edge)
    
    print(incluster)
    incluster = np.unique(incluster)
    old_edges = remove_edges(g, incluster)
    good_nodes = remove_nodes_and_add_supernodes(g, incluster, [new_node_name], attribute, attribute_col)
    
    if len(new_edgelist):
        src_edges, dst_edges = np.array(new_edgelist).T
        
        edf = pd.DataFrame({src: src_edges, dst: dst_edges}).drop_duplicates()
        edf = pd.concat([old_edges, edf])
        print(etdf)
    
    g2 = graphistry.edges(edf, src, dst).nodes(good_nodes, node)
    # g2 = g2.materialize_nodes() #relabels nodes... dumb
    return g2


def get_parent(g, node_id):
    ndf, edf, src, dst, node = _unpack(g)


def get_children_by_attribute(g, node_id, attribute, attribute_col, accumulation_graph):
    g2 = get_children(g, node_id, hops=1)
    ndf, edf, src, dst, node = _unpack(g2)
    tdf = ndf[ndf[attribute_col] == attribute]
    if tdf.empty():  # grab edges and repeat
        edges = g2._edges[dst].unique()
        if len(edges):
            for e in edges:
                accumulation_graph = get_children_by_attribute(g, e, attribute, attribute_col, accumulation_graph)
        else:
            return accumulation_graph
    else:
        g3 = g.nodes(tdf, node)
        super_g = as_node(g3)
    
    return accumulation_graph


###############################################
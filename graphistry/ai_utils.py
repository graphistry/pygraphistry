import pandas as pd
import numpy as np

import graphistry

from .constants import DISTANCE, WEIGHT, BATCH
from logging import getLogger

try:
    import faiss  # type ignore
except:
    faiss = None

logger = getLogger(__name__)


# #################################################################################################
#
#   Finding subgraphs within a large graph.
#
# #################################################################################################


def search_to_df(word, col, df, as_string=False):
    """
    A simple way to retrieve entries from a given (edge) col in a dataframe
    :eg
        search_to_df('BlackRock', 'to_node', edf)

    :param word: str search term
    :param col: given column of dataframe
    :param df: pandas dataframe
    :param as_string: if True, will coerce the column `col` to string, default False.
    :returns
        DataFrame of results, or empty DataFrame if None are found
    """
    try:
        if as_string:
            res = df[df[col].astype(str).str.contains(word, case=False)]
        else:
            res = df[df[col].str.contains(word, case=False)]
    except TypeError as e:
        logger.error(e)
        return pd.DataFrame([], columns=df.columns)
    return res


def get_nearest(search_term, src, dst, edf):
    """
        finds `search_term` in src or dst nodes.
    :param search_term: str search term
    :param src: source node
    :param dst: destination node
    :param edf: edges dataframe
    :return: pandas.DataFrame
    """
    logger.info(f"Finding {search_term} in both {src} and {dst} columns")
    tdf = pd.concat(
        [search_to_df(search_term, src, edf), search_to_df(search_term, dst, edf)],
        axis=0,
    )
    return tdf


def get_graphistry_from_search(search_term, src, dst, node_col, edf, ndf):
    """
        Helper function to get subgraph from a search term (a node) located in edges dataframe
    :param search_term: Note this retrieves all nodes that have `search_term` in it -- ie, not strict matches
    :param src: source node
    :param dst: destination node
    :param node_col: node column
    :param edf: edges dataframe
    :param ndf: nodes dataframe
    :return: graphistry instance
    """
    tdf = get_nearest(search_term, src, dst, edf)
    gcols = pd.concat(
        [tdf[dst], tdf[src]], axis=0
    )  # get all node entries that show up in edge graph
    ntdf = ndf[ndf[node_col].isin(gcols)]
    g = graphistry.edges(tdf, src, dst).nodes(ntdf, node_col)
    return g


# lets make simple functions to find subgraphs by search_term = a given node
def get_milieu_graph_from_search(search_term, src, dst, edf, both=False):
    """
    Can think of this as finding all 2-hop connections from a given `search_term`. It will find all direct edges to
    `search_term` as well as the edges of all those entities. It shows the `milieu graph` of the `search_term`
    :param search_term:
    :param src:
    :param dst:
    :param edf:
    :param both: to retrieve edges from both src and dst columns of dataframe -- if true, returns a larger edgeDataFrame
    :return:
    """
    # get all the entities in either column
    # tdf = pd.concat([search_to_df(search_term, src, edf), search_to_df(search_term, dst, edf)], axis=0)
    tdf = get_nearest(search_term, src, dst, edf)
    # now find all their nearest connections.
    if both:
        gcols = pd.concat([tdf[dst], tdf[src]], axis=0)
        logger.info(
            f"Then finding all edges from {search_term} in {src} and {dst} columns of full edgeDataFrame"
        )
        df = edf[(edf[src].isin(gcols) | edf[dst].isin(gcols))]
    else:
        # logger.info(f'Finding {search_term} in {src} columns')
        # tdf = search_to_df(search_term, src, edf)
        logger.info(
            f"Then finding {src} columns with edges from {search_term} in {dst} column of full edgeDataFrame"
        )
        df = edf[edf[src].isin(list(tdf[dst]) + [search_term])]
    return df


def get_graphistry_from_milieu_search(
    search_term, src, dst, node_col, edf, ndf, both=False
):
    """
        Produces large graphs of neighbors from a given search term
    :param search_term:
    :param src:
    :param dst:
    :param node_col:
    :param edf:
    :param ndf:
    :param both:
    :return:
    """
    tdf = get_milieu_graph_from_search(search_term, src, dst, edf, both=both)
    gcols = pd.concat(
        [tdf[dst], tdf[src]], axis=0
    )  # get all node entries that show up in edge graph
    ntdf = ndf[ndf[node_col].isin(gcols)]
    g = graphistry.edges(tdf, src, dst).nodes(ntdf, node_col)
    return g




# #########################################################################################################################
#
#  Graphistry Vector Search Index
#
##########################################################################################################################


class FaissVectorSearch:
    def __init__(self, M):
        import faiss
        self.index = faiss.IndexFlatL2(M.shape[1])
        self.index.add(M.astype('float32'))

    def search(self, q, k=5):
        """
        Search for the k nearest neighbors of a query vector q.

        Parameters:
        - q: the query vector to search for
        - k: the number of nearest neighbors to return (default: 5)

        Returns:
        - Index: a numpy array of size (k,) containing the indices of the k nearest neighbors
        - Distances: a numpy array of size (k,) containing the distances to the k nearest neighbors
        """
        q = np.asarray(q, dtype=np.float32)
        Distances, Index = self.index.search(q.reshape(1, -1), k)
        return Index[0], Distances[0]
    
    def search_df(self, q, df, k):
        """ Query by vector using index and append distance to results
    
        it is assumed len(vect) == len(df) == len(search_index)
        args:
            vect: query vector
            df: dataframe to query
            search_index: annoy index
            top_n: number of results to return
        returns:
            sorted dataframe with top_n results and distance
        """

        indices, distances = self.search(q.values[0], k=k)

        results = df.iloc[indices]
        results.loc[:, DISTANCE] = distances
        results = results.sort_values(by=[DISTANCE])

        return results


# #########################################################################################################################
#
#  Graphistry Graph Inference
#
##########################################################################################################################

def edgelist_to_weighted_adjacency(g, weights=None):
    """ Convert edgelist to weighted adjacency matrix in sparse coo_matrix"""
    import scipy.sparse as ss
    import numpy as np
    res = g._edges[[g._source, g._destination]].values.astype(np.int64)
    rows, cols = res.T[0], res.T[1]
    if weights is None:
        weights = np.ones(len(rows))
    M = ss.coo_matrix((weights, (rows, cols)))
    return M.tocsr()

def hydrate_graph(res, new_nodes, new_edges, node, src, dst, new_emb, new_features, new_targets):
    # #########################################################
    g = res.nodes(new_nodes, node).edges(new_edges, src, dst)

    # TODO this needs more work since edgelist_to_weighted_adjacency produces non square matrices (since infer_graph will add new nodes)
    #g._weighted_adjacency = edgelist_to_weighted_adjacency(g)
    g._node_embedding = new_emb
    g._node_features = new_features
    g._node_targets = new_targets
    g = g.settings(url_params={'play': 0})
    return g
    

def infer_graph(
    res, emb, X, y, df, infer_on_umap_embedding=False, eps="auto", sample=None, n_neighbors=7, verbose=False, 
):
    """
    Infer a graph from a graphistry object

    args:
        res: graphistry object
        df: outside minibatch dataframe to add to existing graph
        X: minibatch transformed dataframe
        emb: minibatch UMAP embedding distance threshold for a minibatch point to cluster to existing graph
        eps: if 'auto' will find a good epsilon from the data; distance threshold for a minibatch point to cluster to existing graph
        sample: number of nearest neighbors to add from existing graphs edges, if None, ignores existing edges.
            This sets the global stickiness of the graph, and is a good way to control the number of edges incuded from the old graph.
        n_neighbors, int: number of nearest neighbors to include per batch point within epsilon.
            This sets the local stickiness of the graph, and is a good way to control the number of edges between 
            an added point and the existing graph.
    returns:
        graphistry Plottable object
    """
    #enhanced = is_notebook()
    
    print("-" * 50) if verbose else None
    
    if infer_on_umap_embedding and emb is not None:
        X_previously_fit = res._node_embedding
        X_new = emb
        print("Infering edges over UMAP embedding") if verbose else None
    else:  # can still be umap, but want to do the inference on the higher dimensional features
        X_previously_fit = res._node_features
        X_new = X
        print("Infering edges over features embedding") if verbose else None

    print("-" * 45) if verbose else None

    FEATS = res._node_features
    if FEATS is None:
        raise ValueError("Must have node features to infer edges")
    EMB = res._node_embedding if res._node_embedding is not None else FEATS.index
    Y = res._node_target if res._node_target is not None else FEATS.index

    assert (
        df.shape[0] == X.shape[0]
    ), "minibatches df and X must have same number of rows since f(df) = X"
    if emb is not None:
        assert (
            emb.shape[0] == df.shape[0]
        ), "minibatches emb and X must have same number of rows since h(df) = emb"
        df = df.assign(x=emb.x, y=emb.y)  # add x and y to df for graphistry instance

    # if umap, need to add '_n' as node id to df, adding new indices to existing graph
    numeric_indices = range(
        X_previously_fit.shape[0], X_previously_fit.shape[0] + X_new.shape[0]
    )
    df["_n"] = numeric_indices
    df[BATCH] = 1  # 1 for minibatch, 0 for existing graph
    node = res._node

    if node not in df.columns:
        df[node] = numeric_indices

    NDF = res._nodes
    NDF[BATCH] = 0
    EDF = res._edges
    EDF[BATCH] = 0
    src = res._source
    dst = res._destination

    #new_nodes = []
    new_edges = []
    old_edges = []
    old_nodes = []
    mdists = []

    # check if pandas or cudf
    if 'cudf.core.dataframe' in str(type(X_previously_fit)):
        #  move it out of memory...
        X_previously_fit = X_previously_fit.to_pandas()

    for i in range(X_new.shape[0]):
        diff = X_previously_fit - X_new.iloc[i, :]
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        mdists.append(dist)

    m, std = np.mean(mdists), np.std(mdists)
    logger.info(f"--Mean distance to existing nodes  {m:.2f} +/- {std:.2f}")
    print(f' Mean distance to existing nodes {m:.2f} +/- {std:.2f}') if verbose else None
    if eps == "auto":
        eps = np.min([np.abs(m - std), m])
    logger.info(
        f"-epsilon = {eps:.2f} max distance threshold to be considered a neighbor"
    )
    print(f' Max distance threshold; epsilon = {eps:.2f}') if verbose else None
    
    print(f' Finding {n_neighbors} nearest neighbors') if verbose else None
    nn = []
    for i, dist in enumerate(mdists):
        record_df = df.iloc[i, :]
        nearest = np.where(dist < eps)[0]
        nn.append(len(nearest))
        for j in nearest[:n_neighbors]:  # add n_neighbors nearest neighbors, if any, super speedup hack
            this_ndf = NDF.iloc[j, :]
            if sample:
                local_edges = EDF[
                    (EDF[src] == this_ndf[node]) | (EDF[dst] == this_ndf[node])
                ]
                if not local_edges.empty:
                    old_edges.append(local_edges.sample(sample, replace=True))
                    
            weight = min(1 / (dist[j] + 1e-3), 1)
            new_edges.append([this_ndf[node], record_df[node], weight, 1])
            old_nodes.append(this_ndf)
            #new_nodes.extend([record_df, this_ndf])
            
    print(f' {np.mean(nn):.2f} neighbors per node within epsilon {eps:.2f}') if verbose else None
    
    new_edges = pd.DataFrame(new_edges, columns=[src, dst, WEIGHT, BATCH])

    all_nodes = []
    if len(old_edges):
        old_edges = pd.concat(old_edges, axis=0).assign(_batch=0)
        all_nodes = pd.concat([old_edges[src], old_edges[dst], new_edges[src], new_edges[dst]]).drop_duplicates()
        print('', len(all_nodes), "nodes in new graph") if verbose else None

    if sample:
        new_edges = pd.concat([new_edges, old_edges], axis=0).drop_duplicates()
        print(' Sampled', len(old_edges.drop_duplicates()), 'previous old edges') if verbose else None
    new_edges = new_edges.drop_duplicates()
    print('', len(new_edges), 'total edges after dropping duplicates') if verbose else None

    if len(old_nodes):
        old_nodes = pd.DataFrame(old_nodes)
        old_nodes = pd.concat(
            [old_nodes, NDF[NDF[node].isin(all_nodes)]], axis=0
        ).drop_duplicates(subset=[node])
    else:
        old_nodes = NDF[NDF[node].isin(all_nodes)]

    old_emb = None
    if EMB is not None:
        old_emb = EMB.loc[old_nodes.index]

    new_emb = None
    if emb is not None:
        if 'cudf.core.dataframe.DataFrame' in str(type(old_emb)):  # convert to pd
            old_emb = old_emb.to_pandas()
        new_emb = pd.concat([emb, old_emb], axis=0)

    new_features = pd.concat([X, FEATS.loc[old_nodes.index]], axis=0)

    new_nodes = pd.concat([df, old_nodes], axis=0)  # append minibatch at top
    print(" ** Final graph has", len(new_nodes), "nodes") if verbose else None
    print(" - Batch has", len(df), "nodes") if verbose else None
    print(" - Brought in", len(old_nodes), "nodes") if verbose else None

    new_targets = pd.concat([y, Y.loc[old_nodes.index]]) if y is not None else Y

    print("-" * 50) if verbose else None
    return hydrate_graph(res, new_nodes, new_edges, node, src, dst, new_emb, new_features, new_targets)


def infer_self_graph(res, 
    emb, X, y, df, infer_on_umap_embedding=False, eps="auto", n_neighbors=7, verbose=False, 
):
    """
    Infer a graph from a graphistry object

    args:
        df: outside minibatch dataframe to add to existing graph
        X: minibatch transformed dataframe
        emb: minibatch UMAP embedding distance threshold for a minibatch point to cluster to existing graph
        eps: if 'auto' will find a good epsilon from the data; distance threshold for a minibatch point to cluster to existing graph
        sample: number of nearest neighbors to add from existing graphs edges, if None, ignores existing edges.
            This sets the global stickiness of the graph, and is a good way to control the number of edges incuded from the old graph.
        n_neighbors, int: number of nearest neighbors to include per batch point within epsilon.
            This sets the local stickiness of the graph, and is a good way to control the number of edges between 
            an added point and the existing graph.
    returns:
        graphistry Plottable object
    """
    #enhanced = is_notebook()
    
    print("-" * 50) if verbose else None
    
    if infer_on_umap_embedding and emb is not None:
        X_previously_fit = emb
        X_new = emb
        print("Infering edges over UMAP embedding") if verbose else None
    else:  # can still be umap, but want to do the inference on the higher dimensional features
        X_previously_fit = X
        X_new = X
        print("Infering edges over features embedding") if verbose else None

    print("-" * 45) if verbose else None

    assert (
        df.shape[0] == X.shape[0]
    ), "minibatches df and X must have same number of rows since f(df) = X"
    if emb is not None:
        assert (
            emb.shape[0] == df.shape[0]
        ), "minibatches emb and X must have same number of rows since h(df) = emb"
        df = df.assign(x=emb.x, y=emb.y)  # add x and y to df for graphistry instance
    else:  # if umap has been fit, but only transforming over features, need to add x and y or breaks plot binds of res
        df['x'] = np.random.random(df.shape[0])
        df['y'] = np.random.random(df.shape[0])

    #  if umap, need to add '_n' as node id to df, adding new indices to existing graph
    numeric_indices = np.arange(
        X_previously_fit.shape[0],
        dtype=np.float64  # this seems off but works
        )
    df["_n"] = numeric_indices
    df[BATCH] = 1  # 1 for minibatch, 0 for existing graph, here should all be `1` 
    node = res._node
    if node not in df.columns:
        df[node] = numeric_indices

    src = res._source
    dst = res._destination
    
    old_nodes = []
    new_edges = []
    mdists = []

    for i in range(X_new.shape[0]):
        diff = X_previously_fit - X_new.iloc[i, :]
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        mdists.append(dist)

    m, std = np.mean(mdists), np.std(mdists)
    logger.info(f"--Mean distance to existing nodes  {m:.2f} +/- {std:.2f}")
    print(f' Mean distance to existing nodes {m:.2f} +/- {std:.2f}') if verbose else None
    if eps == "auto":
        eps = np.min([np.abs(m - std), m])
    logger.info(
        f" epsilon = {eps:.2f} max distance threshold to be considered a neighbor"
    )
    print(f' Max distance threshold; epsilon = {eps:.2f}') if verbose else None
    
    print(f' Finding {n_neighbors} nearest neighbors') if verbose else None
    nn = []
    for i, dist in enumerate(mdists):
        record_df = df.iloc[i, :]
        nearest = np.where(dist < eps)[0]
        nn.append(len(nearest))
        for j in nearest[:n_neighbors]:  # add n_neighbors nearest neighbors, if any, super speedup hack
            if i != j:
                this_ndf = df.iloc[j, :]
                weight = min(1 / (dist[j] + 1e-3), 1)
                new_edges.append([this_ndf[node], record_df[node], weight, 1])
                old_nodes.append(this_ndf)
            
    print(f' {np.mean(nn):.2f} neighbors per node within epsilon {eps:.2f}') if verbose else None
    
    new_edges = pd.DataFrame(new_edges, columns=[src, dst, WEIGHT, BATCH])
    new_edges = new_edges.drop_duplicates()
    print('', len(new_edges), 'total edges after dropping duplicates') if verbose else None
    print(" ** Final graph has", len(df), "nodes") if verbose else None
    # #########################################################
    print("-" * 50) if verbose else None
    return hydrate_graph(res, df, new_edges, node, src, dst, emb, X, y)

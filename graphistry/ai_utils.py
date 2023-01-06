import pandas as pd
import numpy as np

import graphistry

from .constants import N_TREES, DISTANCE
from logging import getLogger

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


def build_annoy_index(X, angular, n_trees=None):
    """Builds an Annoy Index for fast vector search

    Args:
        X (_type_): _description_
        angular (_type_): _description_
        n_trees (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    from annoy import AnnoyIndex  # type: ignore

    logger.info(f"Building Index of size {X.shape}")

    if angular:
        logger.info("-using angular metric")
        metric = "angular"
    else:
        logger.info("-using euclidean metric")
        metric = "euclidean"

    search_index = AnnoyIndex(X.shape[1], metric)
    # Add all the feature vectors to the search index
    for i in range(len(X)):
        search_index.add_item(i, X.values[i])
    if n_trees is None:
        n_trees = N_TREES

    logger.info(f"-building index with {n_trees} trees")
    search_index.build(n_trees)
    return search_index


def query_by_vector(vect, df, search_index, top_n):
    indices, distances = search_index.get_nns_by_vector(
        vect.values[0], top_n, include_distances=True
    )

    results = df.iloc[indices]
    results[DISTANCE] = distances
    results = results.sort_values(by=[DISTANCE])

    return results


# #########################################################################################################################
#
#  Graphistry Graph Inference
#
##########################################################################################################################


def infer_graph(
    res, emb, X, y, df, infer_on_umap_embedding=False, eps="auto", sample=None
):
    """
    Infer a graph from a graphistry object

    args:
        res: graphistry object
        df: outside minibatch dataframe to add to existing graph
        X: minibatch transformed dataframe
        emb: minibatch UMAP embedding
        kind: 'nodes' or 'edges'
        eps: if 'auto' will find a good epsilon from the data; distance threshold for a minibatchh point to cluster to existing graph
        n_nearest: number of nearest neighbors to add from existing graphs edges, if None, ignores existing edges.
    """

    # new_index = df.index
    if infer_on_umap_embedding and emb is not None:
        X_previously_fit = res._node_embedding
        X_new = emb
    else:  # can still be umap, but want to do the inference on the higher dimensional features
        X_previously_fit = res._node_features
        X_new = X

    FEATS = res._node_features
    EMB = res._node_embedding
    Y = res._node_target

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
        X_previously_fit.shape[0], X_previously_fit.shape[0] + df.shape[0]
    )
    df["_n"] = numeric_indices
    df["_batch"] = 1  # 1 for minibatch, 0 for existing graph
    node = res._node
    NDF = res._nodes
    NDF["_batch"] = 0
    EDF = res._edges
    EDF["_batch"] = 0
    src = res._source
    dst = res._destination

    new_edges = []
    old_edges = []
    old_nodes = []
    mdists = []

    # vsearch = build_search_index(X_previously_fit, angular=False)

    for i in range(X_new.shape[0]):
        # record_df = df.iloc[i, :]
        diff = X_previously_fit - X_new.iloc[i, :]
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        mdists.append(dist)

    m, std = np.mean(mdists), np.std(mdists)
    logger.info(f"--Mean distance to existing nodes  {m:.2f} +/- {std:.2f}")
    # print(f'--Mean distance to existing nodes {m:.2f} +/- {std:.2f}')
    if eps == "auto":
        eps = np.min([np.abs(m - 2 * std), m])
    logger.info(
        f"{eps:.2f} epsilon for max distance threshold to be considered a neighbor"
    )

    for i, dist in enumerate(mdists):
        record_df = df.iloc[i, :]
        for j in np.where(dist < eps)[0]:
            this_ndf = NDF.iloc[j, :]
            if sample:
                local_edges = EDF[
                    (EDF[src] == this_ndf[node]) | (EDF[dst] == this_ndf[node])
                ]
                if not local_edges.empty:
                    old_edges.append(local_edges.sample(sample, replace=True))
            new_edges.append([this_ndf[node], record_df[node], 1, 1])
            old_nodes.append(this_ndf)

    new_edges = pd.DataFrame(new_edges, columns=[src, dst, "_weight", "_batch"])

    all_nodes = []
    if len(old_edges):
        old_edges = pd.concat(old_edges, axis=0).assign(_batch=0)
        all_nodes = (
            old_edges[src]
            .append(old_edges[dst])
            .append(new_edges[src])
            .append(new_edges[dst])
        )

    if sample:
        new_edges = pd.concat([new_edges, old_edges], axis=0)
        # print('sampled', len(new_edges), 'new edges')
    new_edges = new_edges.drop_duplicates()
    # print(len(new_edges), 'new edges after dropping duplicates')

    if len(old_nodes):
        old_nodes = pd.DataFrame(old_nodes)
        old_nodes = pd.concat(
            [old_nodes, NDF[NDF[node].isin(all_nodes)]], axis=0
        ).drop_duplicates(subset=[node])

    old_emb = None
    if EMB is not None:
        old_emb = EMB.loc[old_nodes.index]

    new_emb = None
    if emb is not None:
        new_emb = pd.concat([emb, old_emb], axis=0)

    new_features = pd.concat([X, FEATS.loc[old_nodes.index]], axis=0)

    new_nodes = pd.concat([df, old_nodes], axis=0)  # append minibatch at top

    new_targets = pd.concat([y, Y.loc[old_nodes.index]]) if y is not None else Y

    # #########################################################
    g = res.nodes(new_nodes, node).edges(new_edges, src, dst)

    g._node_embedding = new_emb
    g._node_features = new_features
    g._node_targets = new_targets

    return g

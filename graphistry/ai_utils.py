# import cugraph
import logging
from collections import Counter

import numpy as np
import pandas as pd
from dirty_cat import SimilarityEncoder
from sklearn.inspection import permutation_importance
from sklearn.manifold import MDS
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

import graphistry
from . import constants as config


def setup_logger(name, verbose=True):
    if verbose:
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ]\n   %(message)s\n"
    else:
        FORMAT = "   %(message)s\n"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


# need outside config setting this
verbose = True
logger = setup_logger(__name__, verbose)


def tqdm_progress_bar(total, *args, **kwargs):
    from tqdm import tqdm
    global pbar
    pbar = tqdm(total=total, *args, **kwargs)

    def decorator(func):
        def wrapper(*args, **kwargs):
            pbar.update()
            return func(*args, **kwargs)

        return wrapper

    return decorator


# @tqdm_progress_bar(len(df))
def estimate_encoding_time(df, y):
    # Todo
    pass


# #################################################################################################
#
#   Finding subgraphs within a large graph.
#
# #################################################################################################


def search_to_df(word, col, df):
    """
    A simple way to retrieve entries from a given col in a dataframe
    :eg
        search_to_df('BlackRock', 'to_node', edf)

    :param word: str search term
    :param col: given column of dataframe
    :param df: pandas dataframe
    :returns
        DataFrame of results
    """
    try:
        res = df[df[col].str.contains(word, case=False)]
    except TypeError as e:
        logger.error(e)
        return df
    return res


def get_nearest(search_term, src, dst, edf):
    """
    :param search_term:
    :param src:
    :param dst:
    :param edf:
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
        Useful helper function to get subgraph from a search term
    :param search_term: Note this retrieves all nodes that have `search_term` in it -- ie, not strict matches
    :param src:
    :param dst:
    :param node_col:
    :param edf:
    :param ndf:
    :return:
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


# #################################################################################################
#
#   DGL helpers
#
# #################################################################################################


def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    import torch
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device, gpu_ids


def reindex_edgelist(df, src, dst):
    """Since DGL needs integer contiguous node labels, this relabels as preprocessing step
    :eg
        df, ordered_nodes_dict = reindex_edgelist(df, 'to_node', 'from_node')
        creates new columns given by config.SRC and config.DST
    :param df: edge dataFrame
    :param src: source column of dataframe
    :param dst: destination column of dataframe

    :returns
        df, pandas DataFrame with new edges.
        ordered_nodes_dict, dict ordered from most common src and dst nodes.
    """
    srclist = df[src]
    dstlist = df[dst]
    cnt = Counter(
        pd.concat([srclist, dstlist], axis=0)
    )  # can also use pd.Factorize but doesn't order by count, which is satisfying
    ordered_nodes_dict = {k: i for i, (k, c) in enumerate(cnt.most_common())}
    df[config.SRC] = df[src].apply(lambda x: ordered_nodes_dict[x])
    df[config.DST] = df[dst].apply(lambda x: ordered_nodes_dict[x])
    return df, ordered_nodes_dict


def pandas_to_sparse_adjacency(df, src, dst, weight_col):
    """
        Takes a Pandas Dataframe and named src and dst columns into a sparse adjacency matrix
    :param df:
    :param src:
    :param dst:
    :param weight_col:
    :return:
    """
    # use scipy sparse to encode matrix
    from scipy.sparse import coo_matrix

    df, ordered_nodes_dict = reindex_edgelist(df, src, dst)

    eweight = np.array([1] * len(df))
    if weight_col is not None:
        eweight = df[weight_col].values

    shape = len(ordered_nodes_dict)
    sp_mat = coo_matrix(
        (eweight, (df[config.SRC], df[config.DST])), shape=(shape, shape)
    )
    return sp_mat, ordered_nodes_dict


# def pandas_to_cugraph(df, src, dst, weight_col=None):
#     """ CuGraph uses graph algorithms stored in GPU DataFrames, NetworkX Graphs, or even CuPy or SciPy sparse Matrices
#
#     :param df: A DataFrame that contains edge information
#     :param src: source column name or array of column names
#     :param dst: destination column name or array of column names
#     :param weight_col: the weights column name. Default is None
#     :return: cugraph.Graph object
#     """
#     G = cugraph.Graph()
#     G.from_pandas_edgelist(df, source=src, destination =dst,  edge_attrs = weight_col)
#     logger.info(f"Graph Type: {type(G)}")
#     return G


# #################################################################################################
#
#   Fitting simple model pipelines
#
# #################################################################################################


def fit_pipeline(pipeline, X, y, scoring="r2"):
    """
        Standard Sklearn pipeline fitting method using cross validation
    :param pipeline:
    :param X:
    :param y:
    :return:
    """
    scores = cross_val_score(pipeline, X, y, scoring=scoring)

    logger.info(f"scores={scores}")
    logger.info(f"mean={np.mean(scores)}")
    logger.info(f"std={np.std(scores)}")

    logger.info("Calculating Permutation Feature Importance")
    result = permutation_importance(pipeline, X, y, n_repeats=10, random_state=0)
    # imean = result.importances_mean
    # istd = result.importances_std

    return scores, result


def _plot_feature_importances(importances, feature_names, n=20):
    import matplotlib.pyplot as plt
    indices = np.argsort(importances)
    # Sort from least to most
    indices = list(reversed(indices))
    plt.figure(figsize=(12, 9))
    plt.title("Feature importances")
    n_indices = indices[:n]
    labels = np.array(feature_names)[n_indices]
    plt.barh(range(n), importances[n_indices], color="b")
    plt.yticks(range(n), labels, size=15)
    plt.tight_layout(pad=1)
    plt.show()


# #################################################################################
#
#      ML Helpers and Plots
#
# ###############################################################################


def _calculate_column_similarity(y: pd.Series, n_points: int = 10):
    # y is a pandas series of labels for a given dataset
    sorted_values = y.sort_values().unique()
    similarity_encoder = SimilarityEncoder(similarity="ngram")
    transformed_values = similarity_encoder.fit_transform(sorted_values.reshape(-1, 1))

    _plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=n_points)
    return transformed_values, similarity_encoder, sorted_values


def _plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=15):
    # TODO try this with UMAP
    import matplotlib.pyplot as plt

    mds = MDS(dissimilarity="precomputed", n_init=10, random_state=42)
    two_dim_data = mds.fit_transform(1 - transformed_values)  # transformed values lie

    random_points = np.random.choice(
        len(similarity_encoder.categories_[0]), n_points, replace=False
    )

    nn = NearestNeighbors(n_neighbors=2).fit(transformed_values)
    _, indices_ = nn.kneighbors(transformed_values[random_points])
    indices = np.unique(indices_.squeeze())

    f, ax = plt.subplots()
    ax.scatter(x=two_dim_data[indices, 0], y=two_dim_data[indices, 1])
    # adding the legend
    for x in indices:
        ax.text(
            x=two_dim_data[x, 0], y=two_dim_data[x, 1], s=sorted_values[x], fontsize=8
        )
    ax.set_title(
        "multi-dimensional-scaling representation using a 3gram similarity matrix"
    )

    f2, ax2 = plt.subplots(figsize=(7, 7))
    cax2 = ax2.matshow(transformed_values[indices, :][:, indices])
    ax2.set_yticks(np.arange(len(indices)))
    ax2.set_xticks(np.arange(len(indices)))
    ax2.set_yticklabels(sorted_values[indices], rotation="30")
    ax2.set_xticklabels(sorted_values[indices], rotation="60", ha="right")
    ax2.xaxis.tick_bottom()
    ax2.set_title("Similarities across categories")
    f2.colorbar(cax2)
    f2.tight_layout()

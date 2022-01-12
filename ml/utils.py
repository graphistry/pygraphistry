import numpy as np
import scipy
import pandas as pd
import dgl
import torch
import graphistry
from dirty_cat import SimilarityEncoder, TargetEncoder, MinHashEncoder, GapEncoder
from dirty_cat import SuperVectorizer

from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from scipy.sparse import coo_matrix

import seaborn as sb
import matplotlib.pyplot as plt
from time import time
from collections import Counter

from ml import constants as config

import logging
logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


encoders_dirty = {
    'similarity': SimilarityEncoder(similarity='ngram'),
    'target': TargetEncoder(handle_unknown='ignore'),
    'minhash': MinHashEncoder(n_components=100),
    'gap': GapEncoder(n_components=100),
    'super': SuperVectorizer(auto_cast=True),
}

def estimate_encoding_time(df, y):
    # TODO
    pass

def process_dirty_dataframes(Xdf, y):
    t = time()
    sup_vec = SuperVectorizer(auto_cast=True)
    sup_label = None
    y_enc = None
    logger.info('Sit tight, this might take a few minutes --')
    X_enc = sup_vec.fit_transform(Xdf)
    logger.info(f'Fitting SuperVectorizer on DATA took {(time()-t)/60:.2f} minutes')
    all_transformers = sup_vec.transformers
    features_transformed = sup_vec.get_feature_names()
    logger.info(f'Shape of data {X_enc.shape}')
    logger.info(f'Transformers: {all_transformers}\n')
    logger.info(f'Transformed Columns: {features_transformed[:20]}...\n')
    X_enc = pd.DataFrame(X_enc, columns=features_transformed)
    X_enc = X_enc.fillna(0)
    if y is not None:
        logger.info(f'Fitting Targets --')
        sup_label = SuperVectorizer(auto_cast=True)
        y_enc = sup_label.fit_transform(y)
        if type(y_enc) == scipy.sparse.csr.csr_matrix:
            y_enc = y_enc.toarray()
        labels_transformed = sup_label.get_feature_names()
        y_enc = pd.DataFrame(np.array(y_enc), columns=labels_transformed)
        y_enc = y_enc.fillna(0)
        logger.info(f'Shape of target {y_enc.shape}')
        logger.info(f'Target Transformers used: {sup_label.transformers}')
    return X_enc, y_enc, sup_vec, sup_label


def calculate_column_similarity(y, n_points=10):
    # y is a pandas series of labels for a given dataset
    sorted_values = y.sort_values().unique()
    similarity_encoder = SimilarityEncoder(similarity='ngram')
    transformed_values = similarity_encoder.fit_transform(
        sorted_values.reshape(-1, 1))
    
    plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=n_points)
    return transformed_values, similarity_encoder, sorted_values


def plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=15):
    # TODO try this with UMAP
    mds = MDS(dissimilarity='precomputed', n_init=10, random_state=42)
    two_dim_data = mds.fit_transform(
        1 - transformed_values)  # transformed values lie
    
    random_points = np.random.choice(len(similarity_encoder.categories_[0]),
                                     n_points, replace=False)
    
    nn = NearestNeighbors(n_neighbors=2).fit(transformed_values)
    _, indices_ = nn.kneighbors(transformed_values[random_points])
    indices = np.unique(indices_.squeeze())
    
    f, ax = plt.subplots()
    ax.scatter(x=two_dim_data[indices, 0], y=two_dim_data[indices, 1])
    # adding the legend
    for x in indices:
        ax.text(x=two_dim_data[x, 0], y=two_dim_data[x, 1], s=sorted_values[x],
                fontsize=8)
    ax.set_title(
        'multi-dimensional-scaling representation using a 3gram similarity matrix')

    f2, ax2 = plt.subplots(figsize=(7, 7))
    cax2 = ax2.matshow(transformed_values[indices, :][:, indices])
    ax2.set_yticks(np.arange(len(indices)))
    ax2.set_xticks(np.arange(len(indices)))
    ax2.set_yticklabels(sorted_values[indices], rotation='30')
    ax2.set_xticklabels(sorted_values[indices], rotation='60', ha='right')
    ax2.xaxis.tick_bottom()
    ax2.set_title('Similarities across categories')
    f2.colorbar(cax2)
    f2.tight_layout()
    
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
    word = word.lower()
    try:
        res = df[df[col].apply(lambda x: word in str(x).lower())]
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
    logger.info(f'Finding {search_term} in both {src} and {dst} columns')
    tdf = pd.concat([search_to_df(search_term, src, edf), search_to_df(search_term, dst, edf)], axis=0)
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
    gcols = pd.concat([tdf[dst], tdf[src]], axis=0) # get all node entries that show up in edge graph
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
    #tdf = pd.concat([search_to_df(search_term, src, edf), search_to_df(search_term, dst, edf)], axis=0)
    tdf = get_nearest(search_term, src, dst, edf)
    # now find all their nearest connections.
    if both:
        gcols = pd.concat([tdf[dst], tdf[src]], axis=0)
        logger.info(f'Then finding all edges from {search_term} in {src} and {dst} columns of full edgeDataFrame')
        df = edf[(edf[src].isin(gcols) | edf[dst].isin(gcols))]
    else:
        # logger.info(f'Finding {search_term} in {src} columns')
        # tdf = search_to_df(search_term, src, edf)
        logger.info(f'Then finding {src} columns with edges from {search_term} in {dst} column of full edgeDataFrame')
        df = edf[edf[src].isin(list(tdf[dst])+[search_term])]
    return df

def get_graphistry_from_milieu_search(search_term, src, dst, node_col, edf, ndf, both=False):
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
    gcols = pd.concat([tdf[dst], tdf[src]], axis=0) # get all node entries that show up in edge graph
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
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

def reindex_edgelist(df, src, dst):
    """Since DGL needs integer contiguous node labels, this relabels as preprocessing step
        :eg
            df, ordered_nodes_dict = reindex_edgelist(df, 'to_node', 'from_node')
        :param df: edge dataFrame
        :param src: source column of dataframe
        :param dst: destination column of dataframe
        
        :returns
            df, pandas DataFrame with new edges.
            ordered_nodes_dict, dict ordered from most common src and dst nodes.
    """
    srclist = df[src]
    dstlist = df[dst]
    cnt = Counter(pd.concat([srclist, dstlist], axis=0))
    ordered_nodes_dict = {k: i for i, (k, c) in enumerate(cnt.most_common())}
    df[config.SRC] = df[src].apply(lambda x: ordered_nodes_dict[x])
    df[config.DST] = df[dst].apply(lambda x: ordered_nodes_dict[x])
    return df, ordered_nodes_dict


def pandas2dgl(df, src, dst, weight_col=None, device='cpu'):
    """ Turns an edge DataFrame with named src and dst nodes, to DGL graph
        :eg
            g, sp_mat, ordered_nodes_dict = pandas2dgl(df, 'to_node', 'from_node')
        :returns
            g: dgl graph
            sp_mat: sparse scipy matrix
            ordered_nodes_dict: dict ordered from most common src and dst nodes
    """
    # use scipy sparse to encode matrix
    from scipy.sparse import coo_matrix
    
    df, ordered_nodes_dict = reindex_edgelist(df, src, dst)
    
    eweight = np.array([1]*len(df))
    if weight_col is not None:
        eweight = df[weight_col].values
    
    shape = len(ordered_nodes_dict)
    sp_mat = coo_matrix((eweight, (df[config.SRC], df[config.DST])), shape=(shape, shape))
    
    g = dgl.from_scipy(sp_mat, device=device)  # there are other ways too, like
    logger.info(f'Graph Type: {type(g)}')  # why is this making a heterograph???
    return g, sp_mat, ordered_nodes_dict



# #################################################################################################
#
#   Fitting simple model pipelines
#
# #################################################################################################
def fit_pipeline(pipeline, X, y):
    scores = cross_val_score(pipeline, X, y, scoring='r2')
    
    print(f'scores={scores}')
    print(f'mean={np.mean(scores)}')
    print(f'std={np.std(scores)}')
    
    print('Calculating Permutation Feature Importance')
    result = permutation_importance(pipeline, X, y, n_repeats=10, random_state=0)
    imean = result.importances_mean
    istd = result.importances_std
    
    return scores, result


def plot_confidence_scores(all_scores):
    # all_scores is a list of sklearn cross validation scores
    plt.figure(figsize=(4, 3))
    ax = sb.boxplot(data=pd.DataFrame(all_scores), orient='h')
    plt.ylabel('Encoding', size=20)
    plt.xlabel('Prediction accuracy     ', size=20)
    plt.yticks(size=20)
    plt.tight_layout()


def plot_feature_importances(importances, feature_names, n=20):
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

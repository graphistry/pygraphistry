from time import time
from typing import Any, Callable, Iterable, List, Optional, Set, Union, TYPE_CHECKING, Dict
from graphistry.plotter import PlotterBase
from ml.umap_utils import baseUmap
from sklearn.utils import check_X_y


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

from dirty_cat import (
    SuperVectorizer,
    SimilarityEncoder,
    TargetEncoder,
    MinHashEncoder,
    GapEncoder,
)
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

import ml.constants as config
from ml.utils import setup_logger

logger = setup_logger(__name__)

encoders_dirty: Dict = {
    "similarity": SimilarityEncoder(similarity="ngram"),
    "target": TargetEncoder(handle_unknown="ignore"),
    "minhash": MinHashEncoder(n_components=config.N_HASHERS_DEFAULT),
    "gap": GapEncoder(n_components=config.N_TOPICS_DEFAULT),
    "super": SuperVectorizer(auto_cast=True),
}



# #################################################################################
#
#      Pandas Helpers
#
# ###############################################################################

def check_target_not_in_features(df: pd.DataFrame,
                                 y: Union[pd.DataFrame, pd.Series, np.ndarray, List],
                                 remove: bool = True) -> Union[pd.DataFrame, pd.Series, np.ndarray, List]:
    """

    :param df: model dataframe
    :param y: target dataframe
    :param remove: whether to remove columns from df, default True
    :return: dataframes of model and target
    """
    if y is None:
        return df, y
    remove_cols = []
    if hasattr(y, "columns") and hasattr(df, "columns"):
        yc = y.columns
        xc = df.columns
        for c in yc:
            if c in xc:
                remove_cols.append(c)
    else:
        logger.warning(f"Target is not of type(DataFrame) and has no columns")
    if remove:
        logger.info(f"Removing {remove_cols} columns from DataFrame")
        tf = df.drop(columns=remove_cols)
        return tf, y
    return df, y  # will just pass through data

def remove_internal_namespace_if_present(df):
    # here we drop all _namespace like _x, _y, etc, so that featurization doesn't include them idempotently
    reserved_namespace = ['_x', '_y', config.SRC, config.DST, config.WEIGHT]
    df = df.drop(columns=reserved_namespace, errors='ignore')
    return df
    

def featurize_to_torch(df: pd.DataFrame, y: pd.DataFrame, vectorizer, remove=True):
    """
        Vectorize pandas DataFrames of model and target features into Torch compatible tensors
    :param df: DataFrame of model features
    :param y: DataFrame of target features
    :param vectorizer: must impliment (X, y) encoding and output (X_enc, y_enc, sup_vec, sup_label)
    :param remove: whether to remove target from df
    :return:
        data: dict of {feature, target} encodings
        encoders: dict of {feature, target} encoding objects (which store vocabularity and column information)
    """
    df, y = check_target_not_in_features(df, y, remove=remove)
    X_enc, y_enc, sup_vec, sup_label = vectorizer(df, y)
    if y_enc is not None:
        data = {
            config.FEATURE: torch.tensor(X_enc.values),
            config.TARGET: torch.tensor(y_enc.values),
        }
    else:
        data = {config.FEATURE: torch.tensor(X_enc.values)}
    encoders = {config.FEATURE: sup_vec, config.TARGET: sup_label}
    return data, encoders


# #################################################################################
#
#      Featurization Functions
#
# ###############################################################################


def process_dirty_dataframes(Xdf, y):
    t = time()
    sup_vec = SuperVectorizer(auto_cast=True, high_card_cat_transformer=GapEncoder(42))
    sup_label = None
    y_enc = None
    logger.info("Encoding might take a few minutes --")
    X_enc = sup_vec.fit_transform(Xdf, y)
    logger.info(f"Fitting SuperVectorizer on DATA took {(time()-t)/60:.2f} minutes")
    all_transformers = sup_vec.transformers
    features_transformed = sup_vec.get_feature_names()
    logger.info(f"Shape of data {X_enc.shape}")
    logger.info(f"Transformers: {all_transformers}\n")
    logger.info(f"Transformed Columns: {features_transformed[:20]}...\n")
    X_enc = pd.DataFrame(X_enc, columns=features_transformed)
    X_enc = X_enc.fillna(0)
    if y is not None:
        assert len(Xdf) == len(y), f'Targets must be same size as data'
        logger.info(f"Fitting Targets --")
        sup_label = SuperVectorizer(auto_cast=True)
        y_enc = sup_label.fit_transform(y)
        if type(y_enc) == scipy.sparse.csr.csr_matrix:
            y_enc = y_enc.toarray()
        labels_transformed = sup_label.get_feature_names()
        y_enc = pd.DataFrame(np.array(y_enc), columns=labels_transformed)
        y_enc = y_enc.fillna(0)
        logger.info(f"Shape of target {y_enc.shape}")
        logger.info(f"Target Transformers used: {sup_label.transformers}")
    return X_enc, y_enc, sup_vec, sup_label

def process_edges_dataframe(edf, y, src, dst):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    source = edf[src]
    destination = edf[dst]
    T = mlb.fit_transform(zip(source, destination))
    
    other_df = edf.drop(columns=[src, dst])
    sup_vec = SuperVectorizer(auto_cast=True)
    T2 = sup_vec.fit_transform(other_df, y)
    
    columns = list(mlb.classes_) + sup_vec.get_feature_names_out()
    T = np.c_[T, T2]
    X_enc = pd.DataFrame(T, columns=columns)
    logger.info(f'Created an edge feature matrix of size {T.shape}')
    
    y_enc, sup_label = None, None
    if y is not None:
        sup_label = SuperVectorizer(auto_cast=True)
        y_enc = sup_label.fit_transform(y)
        y_enc = pd.DataFrame(y_enc, columns=sup_label.get_feature_names_out())
        logger.info(f'Created an edge target of size {y_enc.shape}')

    # get's us close to `process_dirty_dataframe
    # TODO how can I meld mlb and sup_vec???
    return X_enc, y_enc, [mlb, sup_vec], sup_label
    
    

# #################################################################################
#
#      Vectorizer Class
#
# ###############################################################################



class FeatureMixin(PlotterBase, baseUmap):
    """
        Notes:
            ~1) Given nothing but a graphistry Plottable `g`, we may minimally generate the (N, N)
            adjacency matrix as a node level feature set, ironically as an edge level feature set over N unique nodes.
            This is the structure/topology of the graph itself, gotten from encoding `g._edges` as an adjacency matrix
            
            ~2) with `node_df = g._nodes` one has row level data over many columns, we may featurize it appropriately,
            generating another node level set. The advantage here is that we are not constrained as we would be in
            a node level adjacency matrix, given M records or rows from `node_df`, with M >= N
            
            ~3) with `edge_df = g._edges` one may also generate a row level encoding, but here we face immediate problems.
                A given edge list is minimally of the form `(src, relationship, dst)`, and so we may form many different
                graphs graded by the cardinality in the `relationships`. Or we may form one single one.
                There is also no notion of how to associate the features produced, unless we use the LineGraph of `g`
                
        Encoding Strategies:
            1) compute the (N, N) adjacency matrix and associate with implicit node level features
            2) feature encode `node_df` as explicit node level features, with M >= N
            3) feature encode `edge_df` as explicit edge level features and associate it with the LineGraph of `g`
        Next:
            A) use UMAP or Louvian, Spectral, etc Embedding to encode 1-3 above, and reduce feature vectors to
            lower dimensional embedding
                a) UMAP projects vectors of length `n` down to, say, 2-dimensions but also generates a
                weighted adjacency matrix under projection, giving another node level feature set (though not distinct,
                or with other
        
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        PlotterBase.__init__(self, *args, **kwargs)
        baseUmap.__init__(self, *args, **kwargs)
        self._node_featurizer = process_dirty_dataframes
        self._edge_featurizer = process_edges_dataframe

        
    def _featurize_nodes(self, y):
        ndf = self._nodes
        ndf, y = check_target_not_in_features(ndf, y, remove=True)  # removes column in ndf if present
        ndf = remove_internal_namespace_if_present(ndf)
        X_enc, y_enc, data_vec, label_vec = self._node_featurizer(ndf, y)
        self.node_features = X_enc
        self.node_target = y_enc
        self.node_encoder = data_vec
        self.node_target_encoder = label_vec
    
    def _featurize_edges(self, y):
        edf = self._edges
        edf, y = check_target_not_in_features(edf, y, remove=True)  # removes column in edf if present
        edf = remove_internal_namespace_if_present(edf)
        X_enc, y_enc, [mlb, data_vec], label_vec = self._edge_featurizer(edf, y, self._source, self._destination)
        self.edge_features = X_enc
        self.edge_target = y_enc
        self.edge_encoders = [mlb, data_vec]
        self.edge_target_encoder = label_vec


    def featurize(self, kind: str = 'nodes', y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None):
        if kind == 'nodes':
            self._featurize_nodes(y)
        elif kind == 'edges':
            self._featurize_edges(y)
        else:
            logger.warning('One may only featurize nodes or edges')
        return self

    def _umap_nodes(self, X, y):
        if X is None:
            if hasattr(self, 'node_features'):
                X = self.node_features
            else:
                logger.warning('Must call `featurize` or supply a feature matrix X')
        if y is None:
            if hasattr(self, 'node_target') and self.node_target is not None:
                y = self.node_target
        return X, y

    def _umap_edges(self, X, y):
        if X is None:
            if hasattr(self, 'edge_features'):
                X = self.edge_features
            else:
                logger.warning('Must call `featurize` or supply a feature matrix X')
        if y is None:
            if hasattr(self, 'edge_target') and self.edge_target is not None:
                y = self.edge_target
        return X, y

    def _bind_xy_from_umap(self, xy):
        if len(xy) != len(self._nodes) or len(xy) != len(self._edges):
            return
        if len(xy) == len(self._nodes):
            df = self._nodes
        elif len(xy) == len(self._edges):
            df = self._edges
        df['_x'] = xy.T[0]
        df['_y'] = xy.T[1]
        

    def umap(self, kind: str = 'nodes', X: np.ndarray = None, y: np.ndarray = None, engine: str = "umap_learn"):
        # UMAP
        if kind == 'nodes':
            X, y = self._umap_nodes(X, y)
            xy = super().fit_transform(X, y)
            self.weighted_adjacency_nodes = self._weighted_adjacency
            self.xy_nodes = xy
        elif kind == 'edges':
            X, y = self._umap_edges(X, y)
            xy = super().fit_transform(X, y)
            self.weighted_adjacency_edges = self._weighted_adjacency
            self.xy_edges = xy
        else:
            logger.warning(f'kind should be one of `nodes` or `edges` unless you are passing explicit matrices')
            try:
                X, y = check_X_y(X, y)
            except ValueError as e:
                logger.error(e)
                return self
            xy = super().fit_transform(X, y)
            self._xy = xy
            logger.info(f'Reduced Coordinates are stored in `_xy`')
            
        self._bind_xy_from_umap(xy)
            
        return self
    




# #################################################################################
#
#      ML Helpers and Plots
#
# ###############################################################################


def calculate_column_similarity(y: pd.Series, n_points: int = 10):
    # y is a pandas series of labels for a given dataset
    sorted_values = y.sort_values().unique()
    similarity_encoder = SimilarityEncoder(similarity="ngram")
    transformed_values = similarity_encoder.fit_transform(sorted_values.reshape(-1, 1))

    plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=n_points)
    return transformed_values, similarity_encoder, sorted_values


def plot_MDS(transformed_values, similarity_encoder, sorted_values, n_points=15):
    # TODO try this with UMAP
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

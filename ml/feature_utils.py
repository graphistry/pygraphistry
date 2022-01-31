from time import time
from typing import List, Union, Dict, Callable

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
from sklearn.preprocessing import MultiLabelBinarizer

from sentence_transformers import SentenceTransformer

import ml.constants as config
from graphistry.plotter import PlotterBase
from ml.umap_utils import BaseUMAPMixin
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


def check_target_not_in_features(
    df: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series, np.ndarray, List],
    remove: bool = True,
) -> Union[pd.DataFrame, pd.Series, np.ndarray, List]:
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
    """
        Some tranformations below add columns to the dataframe, this method removes them before featurization
    :param df: dataframe
    :return: dataframe with dropped columns in reserved namespace
    """
    # here we drop all _namespace like _x, _y, etc, so that featurization doesn't include them idempotently
    reserved_namespace = [config.X, config.Y, config.SRC, config.DST, config.WEIGHT]
    df = df.drop(columns=reserved_namespace, errors="ignore")
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


def check_if_textual_column(
    df: pd.DataFrame, col: str, confidence: float = 0.35, min_words: float = 3.5
) -> bool:
    isstring = df[col].apply(lambda x: isinstance(x, str))
    abundance = sum(isstring) / len(df)
    assert (
        min_words > 1
    ), f"probably best to have at least a word if you want to consider this a textual column?"
    if abundance >= confidence:
        # now check how many words
        n_words = df[col].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        mean_n_words = n_words.mean()
        if mean_n_words >= min_words:
            logger.info(
                f"\n\tColumn `{col}` looks textual with mean number of words {mean_n_words:.2f}"
            )
            return True
        else:
            return False
    else:
        return False


def get_textual_columns(df: pd.DataFrame) -> List:
    text_cols = []
    for col in df.columns:
        if check_if_textual_column(df, col):
            text_cols.append(col)
    if len(text_cols) == 0:
        logger.info(f'No Textual Columns were found')
    return text_cols


def process_textual_or_other_dataframes(
    df: pd.DataFrame, y: pd.DataFrame, z_scale: bool = True, model_name: str = "paraphrase-MiniLM-L6-v2"
) -> Union[pd.DataFrame, Callable]:
    """
        Automatic Deep Learning Embedding of Textual Features, with the rest taken care of by dirty_cat
    :param df: pandas dataframe of data
    :param y: pandas dataframe of targets
    :param model_name: SentenceTransformer model name. See available list at
            https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    :return:
    """
    model = SentenceTransformer(model_name)

    text_cols = get_textual_columns(df)
    embeddings = np.zeros((len(df), 1))  # just a placeholder so we can use np.c_
    if text_cols:
        for col in text_cols:
            logger.info(f"Calculating Embeddings for column `{col}`")
            emb = model.encode(df[col].values)
            embeddings = np.c_[embeddings, emb]

    other_df = df.drop(columns=text_cols, errors="ignore")
    X_enc, y_enc, data_encoder, label_encoder = process_dirty_dataframes(
        other_df, y, z_scale=False
    )

    faux_columns = list(
        range(embeddings.shape[1] - 1)
    )  # minus 1 since the first column is just placeholder

    embeddings = np.c_[embeddings, X_enc.values]
    # now remove the zeros
    embeddings = embeddings[:, 1:]
    if z_scale:  # sort of, but don't remove mean
        embeddings /= embeddings.std(0) + 1

    X_enc = pd.DataFrame(
        embeddings, columns=faux_columns + data_encoder.get_feature_names_out()
    )

    return X_enc, y_enc, data_encoder, label_encoder


def process_dirty_dataframes(
    ndf: pd.DataFrame,
    y: pd.DataFrame,
    cardinality_threshold: int = 40,
    n_topics: int = config.N_TOPICS_DEFAULT,
    z_scale: bool = True,
) -> Union[pd.DataFrame, Callable]:
    """
        Dirty_Cat encoder for node-record level data. Will automatically turn
        inhomogeneous dataframe into matrix using smart conversion tricks.
    :param ndf: node dataframe
    :param y: target dataframe or series
    :param n_topics: number of topics for GapEncoder, default 42
    :param z_scale: bool, default True.
    :return: Encoded data matrix and target (if not None), the data encoder, and the label encoder.
    """
    t = time()
    data_encoder = SuperVectorizer(
        auto_cast=True,
        cardinality_threshold=cardinality_threshold,
        high_card_cat_transformer=GapEncoder(n_topics),
    )
    label_encoder = None
    y_enc = None
    logger.info("Encoding might take a few minutes --")
    X_enc = data_encoder.fit_transform(ndf, y)
    if z_scale:
        # X_enc -= X_enc.mean(0)
        X_enc /= X_enc.std(0) + 1
        logger.info(f"Z-Scaling the data")

    logger.info(f"Fitting SuperVectorizer on DATA took {(time()-t)/60:.2f} minutes\n")

    all_transformers = data_encoder.transformers
    features_transformed = data_encoder.get_feature_names()

    logger.info(f"Shape of data {X_enc.shape}\n")
    logger.info(f"Transformers: {all_transformers}\n")
    logger.info(f"Transformed Columns: {features_transformed[:20]}...\n")

    X_enc = pd.DataFrame(X_enc, columns=features_transformed)
    X_enc = X_enc.fillna(0)
    if y is not None:
        assert len(ndf) == len(y), f"Targets must be same size as data"
        logger.info(f"Fitting Targets --\n")
        label_encoder = SuperVectorizer(auto_cast=True)
        y_enc = label_encoder.fit_transform(y)
        if type(y_enc) == scipy.sparse.csr.csr_matrix:
            y_enc = y_enc.toarray()
        labels_transformed = label_encoder.get_feature_names()
        y_enc = pd.DataFrame(np.array(y_enc), columns=labels_transformed)
        y_enc = y_enc.fillna(0)

        logger.info(f"Shape of target {y_enc.shape}")
        logger.info(f"Target Transformers used: {label_encoder.transformers}\n")
    return X_enc, y_enc, data_encoder, label_encoder


def process_edge_dataframes(
    edf: pd.DataFrame, y: pd.DataFrame, src: str, dst: str, z_scale: bool = True
) -> Union[pd.DataFrame, Callable]:
    """
        Custom Edge-record encoder. Uses a MultiLabelBinarizer to generate a src/dst vector
        and then a Dirty_Cat SuperVectorizer that encodes any other data present in edf

    :param edf: pandas DataFrame of features
    :param y: pandas DataFrame of labels
    :param src: source column to select in edf
    :param dst: destination column to select in edf
    :return: Encoded data matrix and target (if not None), the data encoders, and the label encoder.
    """
    mlb_pairwise_edge_encoder = MultiLabelBinarizer()
    source = edf[src]
    destination = edf[dst]
    T = mlb_pairwise_edge_encoder.fit_transform(zip(source, destination))

    other_df = edf.drop(columns=[src, dst])
    X_enc, y_enc, data_encoder, label_encoder = process_textual_or_other_dataframes(
        other_df, y, z_scale=False
    )

    columns = (
        list(mlb_pairwise_edge_encoder.classes_) + data_encoder.get_feature_names_out()
    )

    T = np.c_[T, X_enc.values]
    if z_scale:
        T /= T.std(0) + 1

    X_enc = pd.DataFrame(T, columns=columns)
    logger.info(f"Created an edge feature matrix of size {T.shape}")

    # get's us close to `process_nodes_dataframe
    # TODO how can I meld mlb and sup_vec??? Difficult as it is not a per column transformer...
    return X_enc, y_enc, [mlb_pairwise_edge_encoder, data_encoder], label_encoder


# #################################################################################
#
#      Vectorizer Class
#
# ###############################################################################


def prune_weighted_edges_df(wdf: pd.DataFrame, scale: float = 2.0) -> pd.DataFrame:
    """
        Prune the weighted edge dataframe so to return high fidelity similarity scores.
    :param wdf: weighted edge dataframe gotten via UMAP
    :param scale: multiplicative scale for pruning weighted edge dataframe gotten from UMAP > (mean + scale * std)
    :return: pd.DataFrame
    """
    # we want to prune edges, so we calculate some statistics
    desc = wdf.describe()  # TODO, perhaps add Box-Cox transform, etc?
    mean = desc[config.WEIGHT]["mean"]
    std = desc[config.WEIGHT]["std"]
    wdf2 = wdf[wdf[config.WEIGHT] >= mean + scale * std]
    logger.info(f"Pruning weighted edge dataframe from {len(wdf)} to {len(wdf2)} edges")
    return wdf2


class FeatureMixin(PlotterBase, BaseUMAPMixin):
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
        BaseUMAPMixin.__init__(self, *args, **kwargs)
        self._node_featurizer = process_textual_or_other_dataframes
        self._edge_featurizer = process_edge_dataframes

    def _featurize_nodes(self, y, use_columns=None):
        ndf = self._nodes
        if use_columns is not None:
            ndf = ndf[use_columns]
        ndf, y = check_target_not_in_features(ndf, y, remove=True)
        ndf = remove_internal_namespace_if_present(ndf)
        X_enc, y_enc, data_vec, label_vec = self._node_featurizer(ndf, y)
        self.node_features = X_enc
        self.node_target = y_enc
        self.node_encoder = data_vec
        self.node_target_encoder = label_vec

    def _featurize_edges(self, y, use_columns=None):
        edf = self._edges
        if use_columns is not None:
            edf = edf[use_columns]
        edf, y = check_target_not_in_features(edf, y, remove=True)
        edf = remove_internal_namespace_if_present(edf)
        X_enc, y_enc, [mlb, data_vec], label_vec = self._edge_featurizer(
            edf, y, self._source, self._destination
        )
        self.edge_features = X_enc
        self.edge_target = y_enc
        self.edge_encoders = [mlb, data_vec]
        self.edge_target_encoder = label_vec

    def featurize(
        self,
        kind: str = "nodes",
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None,
        use_columns: Union[None, List] = None,
    ):
        """
            Featurize Nodes or Edges of the Graph.

        :param kind: specify whether to featurize `nodes` or `edges`
        :param y: Optional Target, default None. If .featurize came with a target, it will use that target.
        :param use_columns: Specify which dataframe columns to use for featurization, if any.
        :return: self, with new attributes set by the featurization process

        """
        if kind == "nodes":
            self._featurize_nodes(y, use_columns)
        elif kind == "edges":
            self._featurize_edges(y, use_columns)
        else:
            logger.warning("One may only featurize nodes or edges")
        return self

    def _umap_nodes(self, X, y, use_columns):
        # helper method gets node feature and target matrix if X, y are not specified
        if X is None:
            if hasattr(self, "node_features"):
                X = self.node_features
            else:
                logger.warning(
                    "Calling `featurize` to create data matrix X over nodes dataframe"
                )
                self._featurize_nodes(y, use_columns)
                return self._umap_nodes(X, y, use_columns)
        if y is None:
            if hasattr(self, "node_target") and self.node_target is not None:
                y = self.node_target
        return X, y

    def _umap_edges(self, X, y, use_columns):
        # helper method gets edge feature and target matrix if X, y are not specified
        if X is None:
            if hasattr(self, "edge_features"):
                X = self.edge_features
            else:
                logger.warning(
                    "Call `featurize` to create data matrix X over edges dataframe"
                )
                self._featurize_edges(y, use_columns)
                return self._umap_edges(X, y, use_columns)
        if y is None:
            if hasattr(self, "edge_target") and self.edge_target is not None:
                y = self.edge_target
        return X, y

    def _bind_xy_from_umap(self, xy):
        # binds reduced coordinates
        if xy is None:
            return
        if len(xy) != len(self._nodes) or len(xy) != len(self._edges):
            return

        if len(xy) == len(self._nodes):
            df = self._nodes
        elif len(xy) == len(self._edges):
            df = self._edges
        else:
            return

        df[config.X] = xy.T[0]
        df[config.Y] = xy.T[1]
        self.bind(point_x=config.X, point_y=config.Y)

    def umap(
        self,
        kind: str = "nodes",
        use_columns: Union[List, None] = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        scale: float = 2,
        n_components: int = 2,
        metric: str = "euclidean",
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        engine: str = "umap_learn",
    ):
        """
            UMAP the featurized node or edges data, or pass in your own X, y (optional).

        :param kind: `nodes` or `edges` or None. If None, expects explicit X, y (optional) matrices, and will Not
                associate them to nodes or edges. If X, y (optional) is given, with kind = [nodes, edges],
                it will associate new matrices to nodes or edges attributes.
        :param use_columns: List of columns to use for featurization if featurization hasn't been applied.
        :param X: ndarray of features
        :param y: ndarray of targets
        :param scale: multiplicative scale for pruning weighted edge dataframe gotten from UMAP (mean + scale *std)
        :param n_components: number of components in the UMAP projection, default 2
        :param metric: UMAP metric, default 'euclidean'. Other useful ones are 'hellinger', '..'
                see (UMAP-LEARN)[https://umap-learn.readthedocs.io/en/latest/parameters.html] documentation for more.
        :param n_neighbors: number of nearest neighbors to include for UMAP connectivity, lower makes more compact layouts. Minimum 2.
        :param min_dist: float between 0 and 1, lower makes more compact layouts.
        :param engine: selects which engine to use to calculate UMAP: NotImplemented yet, default UMAP-LEARN
        :return: self, with attributes set with new data
        """
        xy = None
        umap_kwargs = dict(
            n_components=n_components,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        super()._set_new_kwargs(**umap_kwargs)
        if kind == "nodes":
            X, y = self._umap_nodes(X, y, use_columns)
            xy = super().fit_transform(X, y)
            self.weighted_adjacency_nodes = self._weighted_adjacency
            self.xy_nodes = xy
            self.weighted_edges_df_from_nodes = prune_weighted_edges_df(
                self._weighted_edges_df, scale=scale
            )
        elif kind == "edges":
            X, y = self._umap_edges(X, y, use_columns)
            xy = super().fit_transform(X, y)
            self.weighted_adjacency_edges = self._weighted_adjacency
            self.xy_edges = xy
            self.weighted_edges_df_from_edges = prune_weighted_edges_df(
                self._weighted_edges_df, scale=scale
            )
        else:
            logger.warning(
                f"kind should be one of `nodes` or `edges` unless you are passing explicit matrices"
            )
            if X is not None:
                xy = super().fit_transform(X, y)
                self._xy = xy
                self._weighted_edges_df = prune_weighted_edges_df(
                    self._weighted_edges_df, scale=self.scale
                )
                logger.info(
                    f"Reduced Coordinates are stored in `._xy` attribute and "
                    f"pruned weighted edge df in `._weighted_edges_df` attribute"
                )

        # FIXME
        # self._bind_xy_from_umap(xy) # doesn't work ...

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

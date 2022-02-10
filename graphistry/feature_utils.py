from time import time
from typing import List, Union, Dict, Callable, Any, Tuple, Optional

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
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from graphistry.plotter import PlotterBase

from . import constants as config
from .umap_utils import BaseUMAPMixin
from .ai_utils import  setup_logger

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


def safe_divide(a, b):
    # so we can do X /= X.std(0) safely , etc
    a = np.array(a)
    b = np.array(b)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0.0, casting="unsafe")


def check_target_not_in_features(
    df: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series, np.ndarray, List],
    remove: bool = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series, np.ndarray, List]]:
    """
        Checks if y DataFrame column name is in df, and removes it from df if so

    :param df: model DataFrame
    :param y: target DataFrame
    :param remove: whether to remove columns from df, default True
    :return: DataFrames of model and target
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
        tf = df.drop(columns=remove_cols, errors="ignore")
        return tf, y
    return df, y  # will just pass through data


def remove_node_column_from_ndf_and_return_ndf_from_res(res, remove_node_column: bool):
    """
        Helper method to make sure that node name is not featurized

    :param res:
    :param remove_node_column: bool, whether to remove node column or not
    :return: node DataFrame with or without node column
    """
    if remove_node_column:
        if hasattr(res, "_node"):
            node_label = res._node
            if node_label is not None:
                logger.info(
                    f"removing node column `{node_label}` so we do not featurize it"
                )
                return res._nodes.drop(columns=[node_label])
    return res._nodes  # just pass through if false


def remove_internal_namespace_if_present(df: pd.DataFrame):
    """
        Some tranformations below add columns to the DataFrame, this method removes them before featurization

    :param df: DataFrame
    :return: DataFrame with dropped columns in reserved namespace
    """
    if df is None:
        return None
    # here we drop all _namespace like _x, _y, etc, so that featurization doesn't include them idempotently
    reserved_namespace = [config.X, config.Y, config.SRC, config.DST, config.WEIGHT]
    df = df.drop(columns=reserved_namespace, errors="ignore")
    # logger.info(df.head(3))
    return df


def convert_to_torch(X_enc: pd.DataFrame, y_enc: Union[pd.DataFrame, None]):
    """
        Converts X, y to torch tensors compatible with ndata/edata of DGL graph

    :param X_enc:
    :param y_enc:
    :return: dictionary of torch encoded arrays
    """
    if y_enc is not None:
        data = {
            config.FEATURE: torch.tensor(X_enc.values),
            config.TARGET: torch.tensor(y_enc.values),
        }
    else:
        data = {config.FEATURE: torch.tensor(X_enc.values)}
    return data


# #################################################################################
#
#      Featurization Functions
#
# ###############################################################################


def check_if_textual_column(
    df: pd.DataFrame, col: str, confidence: float = 0.35, min_words: float = 3.5
) -> bool:
    """
        Checks if `col` column of df is textual or not using basic heuristics

    :param df: DataFrame
    :param col: column name
    :param confidence: threshold float value between 0 and 1. If column `col` has `confidence` more elements as `str`
            it will pass it onto next stage of evaluation. Default 0.35
    :param min_words: mean minimum words threshold. If mean words across `col` is greater than this, it is deemed textual.
            Default 3.5
    :return: bool, whether column is textual or not
    """
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
    """
        Collects columns from df that it deems are textual.

    :param df: DataFrame
    :return: list of columns names
    """
    text_cols = []
    for col in df.columns:
        if check_if_textual_column(df, col):
            text_cols.append(col)
    if len(text_cols) == 0:
        logger.info(f"No Textual Columns were found")
    return text_cols


def process_textual_or_other_dataframes(
    df: pd.DataFrame,
    y: pd.DataFrame,
    z_scale: bool = True,
    model_name: str = "paraphrase-MiniLM-L6-v2",
) -> Union[pd.DataFrame, Callable, Any]:
    """
        Automatic Deep Learning Embedding of Textual Features,
        with the rest of the columns taken care of by dirty_cat

    :param df: pandas DataFrame of data
    :param y: pandas DataFrame of targets
    :param model_name: SentenceTransformer model name. See available list at
            https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    :return: X_enc, y_enc, data_encoder, label_encoder
    """
    t = time()
    model = SentenceTransformer(model_name)

    if len(df) == 0 or df.empty:
        logger.info(f"DataFrame seems to be Empty")

    text_cols = get_textual_columns(df)
    embeddings = np.zeros((len(df), 1))  # just a placeholder so we can use np.c_
    if text_cols:
        for col in text_cols:
            logger.info(f"-Calculating Embeddings for column `{col}`")
            emb = model.encode(df[col].values)
            embeddings = np.c_[embeddings, emb]

    other_df = df.drop(columns=text_cols)
    X_enc, y_enc, data_encoder, label_encoder = process_dirty_dataframes(
        other_df, y, z_scale=False
    )

    faux_columns = list(
        range(embeddings.shape[1] - 1)
    )  # minus 1 since the first column is just placeholder

    if data_encoder is not None:
        embeddings = np.c_[embeddings, X_enc.values]
        # now remove the zeros
        columns = faux_columns + data_encoder.get_feature_names_out()
    else:
        columns = faux_columns

    embeddings = embeddings[:, 1:]
    if z_scale:  # sort of, but don't remove mean
        embeddings = safe_divide(embeddings, embeddings.std(0))

    X_enc = pd.DataFrame(embeddings, columns=columns)
    logger.info(
        f"--The entire Textual or Other encoding process took {(time()-t)/60:.2f} minutes"
    )
    return X_enc, y_enc, data_encoder, label_encoder


def process_dirty_dataframes(
    ndf: pd.DataFrame,
    y: pd.DataFrame,
    cardinality_threshold: int = 40,
    n_topics: int = config.N_TOPICS_DEFAULT,
    z_scale: bool = True,
) -> Union[pd.DataFrame, Callable, Any]:
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
    if not ndf.empty:
        logger.info("Encoding might take a few minutes --------")
        X_enc = data_encoder.fit_transform(ndf, y)
        X_enc = X_enc.astype(float)  # otherwise the safe divide is borqued
        if z_scale:
            X_enc = safe_divide(X_enc, np.std(X_enc, axis=0))
            logger.info(f"Z-Scaling the data")

        logger.info(
            f"-Fitting SuperVectorizer on DATA took {(time()-t)/60:.2f} minutes\n"
        )

        all_transformers = data_encoder.transformers
        features_transformed = data_encoder.get_feature_names_out()

        logger.info(f"-Shape of data {X_enc.shape}\n")
        logger.info(f"-Transformers: {all_transformers}\n")
        logger.info(f"-Transformed Columns: {features_transformed[:20]}...\n")

        X_enc = pd.DataFrame(X_enc, columns=features_transformed)
        X_enc = X_enc.fillna(0)
    else:
        X_enc = None
        data_encoder = None
        logger.info(f"*Given DataFrame seems to be empty")

    if y is not None:
        logger.info(f"-Fitting Targets --\n")
        label_encoder = SuperVectorizer(auto_cast=True)
        y_enc = label_encoder.fit_transform(y)
        if type(y_enc) == scipy.sparse.csr.csr_matrix:
            y_enc = y_enc.toarray()
        labels_transformed = label_encoder.get_feature_names_out()
        y_enc = pd.DataFrame(np.array(y_enc), columns=labels_transformed)
        y_enc = y_enc.fillna(0)

        logger.info(f"-Shape of target {y_enc.shape}")
        logger.info(f"-Target Transformers used: {label_encoder.transformers}\n")

    return X_enc, y_enc, data_encoder, label_encoder


def process_edge_dataframes(
    edf: pd.DataFrame, y: pd.DataFrame, src: str, dst: str, z_scale: bool = True
) -> Union[pd.DataFrame, Callable, Any]:
    """
        Custom Edge-record encoder. Uses a MultiLabelBinarizer to generate a src/dst vector
        and then process_textual_or_other_dataframes that encodes any other data present in edf,
        textual or not.

    :param edf: pandas DataFrame of features
    :param y: pandas DataFrame of labels
    :param src: source column to select in edf
    :param dst: destination column to select in edf
    :return: Encoded data matrix and target (if not None), the data encoders, and the label encoder.
    """
    t = time()
    mlb_pairwise_edge_encoder = MultiLabelBinarizer()
    source = edf[src]
    destination = edf[dst]
    logger.info(f"Encoding Edges using MultiLabelBinarizer")
    T = mlb_pairwise_edge_encoder.fit_transform(zip(source, destination))
    T = 1.0 * T  # coerce to float, or divide= will throw error under z_scale below
    logger.info(f"-Shape of Edge encoder {T.shape}")

    other_df = edf.drop(columns=[src, dst])
    logger.info(
        f"-Rest of DataFrame has columns: {other_df.columns} and is empty: {other_df.empty}"
    )

    X_enc, y_enc, data_encoder, label_encoder = process_textual_or_other_dataframes(
        other_df, y, z_scale=False
    )
    if data_encoder is not None:
        columns = (
            list(mlb_pairwise_edge_encoder.classes_)
            + data_encoder.get_feature_names_out()
        )
        T = np.c_[T, X_enc.values]
    else:  # if this other_df is empty
        logger.info("-Other_df is empty")
        columns = list(mlb_pairwise_edge_encoder.classes_)

    if z_scale:
        T = safe_divide(T, T.std(0))

    X_enc = pd.DataFrame(T, columns=columns)
    logger.info(f"--Created an Edge feature matrix of size {T.shape}")
    logger.info(f"**The entire Edge encoding process took {(time()-t)/60:.2f} minutes")
    # get's us close to `process_nodes_dataframe
    # TODO how can I meld mlb and sup_vec??? Difficult as it is not a per column transformer...
    return X_enc, y_enc, [mlb_pairwise_edge_encoder, data_encoder], label_encoder


# #################################################################################
#
#      Vectorizer Class
#
# ###############################################################################


def prune_weighted_edges_df(
    wdf: pd.DataFrame, scale: float = 1.0, index_to_nodes_dict: Dict = None
) -> pd.DataFrame:
    """
        Prune the weighted edge DataFrame so to return high fidelity similarity scores.

    :param wdf: weighted edge DataFrame gotten via UMAP
    :param scale: multiplicative scale for pruning weighted edge DataFrame gotten from UMAP > (mean + scale * std)
    :return: pd.DataFrame
    """
    # we want to prune edges, so we calculate some statistics
    desc = wdf.describe()  # TODO, perhaps add Box-Cox transform, etc?
    mean = desc[config.WEIGHT]["mean"]
    std = desc[config.WEIGHT]["std"]
    wdf2 = wdf[wdf[config.WEIGHT] >= mean + scale * std]
    # TODO remove either src -> dst and dst -> src so that we don't have double edges
    logger.info(f"Pruning weighted edge DataFrame from {len(wdf)} to {len(wdf2)} edges")
    if index_to_nodes_dict is not None:
        wdf2[config.SRC] = wdf2[config.SRC].apply(lambda x: index_to_nodes_dict[x])
        wdf2[config.DST] = wdf2[config.DST].apply(lambda x: index_to_nodes_dict[x])
    return wdf2


def get_dataframe_columns(df: pd.DataFrame, columns: Union[List, None] = None):  # TODO
    """
        helper method to get columns from DataFrame -- does not check if column is in DataFrame, that is up to user

    :param df: DataFrame
    :param columns: columns you want to include in analysis/featurization
    :return: DataFrame with columns
    """
    if columns is None:
        # just pass through df
        return df
    if columns == []:  # hmmm do i want this behavior? #FIXME??
        logger.warning(
            f"Passing an empty column list [] returns None rather than original DataFrame"
        )
        return None
    if len(columns):
        logger.info(f"returning DataFrame with columns `{columns}`")
        return df[columns]


class FeatureMixin(PlotterBase, BaseUMAPMixin):
    """
    FeatureMixin for automatic featurization of nodes and edges DataFrames.
    Subclasses BaseUMAPMixin for umap-ing of automatic features.

    TODO: add example usage doc
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        PlotterBase.__init__(self, *args, **kwargs)
        BaseUMAPMixin.__init__(self, *args, **kwargs)
        self._node_featurizer = process_textual_or_other_dataframes
        self._edge_featurizer = process_edge_dataframes

    def _featurize_nodes(
        self,
        res: Any,
        y: Union[pd.DataFrame, np.ndarray],
        use_columns: Union[List, None] = None,
        remove_node_column: bool = True,  # since node label might be index or unique names
    ):

        ndf = remove_node_column_from_ndf_and_return_ndf_from_res(
            res, remove_node_column
        )
        # TODO move the columns select after the featurizer
        ndf = get_dataframe_columns(ndf, use_columns)
        ndf, y = check_target_not_in_features(ndf, y, remove=True)
        ndf = remove_internal_namespace_if_present(ndf)
        # now vectorize it all
        X_enc, y_enc, data_vec, label_vec = self._node_featurizer(ndf, y)
        # set the new variables
        res.node_features = X_enc
        res.node_target = y_enc
        res.node_encoder = data_vec
        res.node_target_encoder = label_vec
        return res

    def _featurize_edges(
        self,
        res: Any,
        y: Union[pd.DataFrame, np.ndarray],
        use_columns: Union[List, None] = None,
    ):
        # TODO move the columns select after the featurizer
        edf = get_dataframe_columns(res._edges, use_columns)
        edf, y = check_target_not_in_features(edf, y, remove=True)
        edf = remove_internal_namespace_if_present(edf)
        X_enc, y_enc, [mlb, data_vec], label_vec = self._edge_featurizer(
            edf, y, res._source, res._destination
        )
        res.edge_features = X_enc
        res.edge_target = y_enc
        res.edge_encoders = [mlb, data_vec]
        res.edge_target_encoder = label_vec
        return res

    def featurize(
        self,
        kind: str = "nodes",
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None,
        use_columns: Union[None, List] = None,
        inplace: bool = False,
        remove_node_column: bool = True,
    ):
        """
            Featurize Nodes or Edges of the Graph.

        :param kind: specify whether to featurize `nodes` or `edges`
        :param y: Optional Target, default None. If .featurize came with a target, it will use that target.
        :param use_columns: Specify which DataFrame columns to use for featurization, if any.
        :return: self, with new attributes set by the featurization process

        """
        if inplace:
            res = self
        else:
            res = self.bind()
        if kind == "nodes":
            res = self._featurize_nodes(res, y, use_columns, remove_node_column)
        elif kind == "edges":
            res = self._featurize_edges(res, y, use_columns)
        else:
            logger.warning("One may only featurize `nodes` or `edges`")
            return self
        if inplace:
            return None
        return res

    def _featurize_or_get_nodes_data_if_X_is_None(
        self,
        res: Any,
        X: Union[np.ndarray, None],
        y: Union[np.ndarray, List, None],
        use_columns: Union[List, None],
    ):
        """
            helper method gets node feature and target matrix if X, y are not specified.
            if X, y are specified will set them as `node_target` and `node_target` attributes
        :param X: ndArray Data Matrix, default None
        :param y: target, default None
        :param use_columns: which columns to featurize if X is None
        :return: data `X` and `y`
        """
        if X is None:
            if hasattr(res, "node_features"):
                X = res.node_features
                logger.info(f"Found Node features in `res`")
            else:
                logger.warning(
                    "Calling `featurize` to create data matrix `X` over nodes DataFrame"
                )
                res = self._featurize_nodes(res, y, use_columns)
                return self._featurize_or_get_nodes_data_if_X_is_None(
                    res, res.node_features, res.node_target, use_columns
                )  # now we are guaranteed to have node feature and target matrices.
        if y is None:
            if hasattr(res, "node_target"):
                y = res.node_target
                logger.info(
                    f"Fetching `node_target` in `res`. Target is type {type(y)}"
                )

        # now on the return the X, y will
        return X, y

    def _featurize_or_get_edges_dataframe_if_X_is_None(
        self,
        res: Any,
        X: Union[np.ndarray, None],
        y: Union[np.ndarray, List, None],
        use_columns: Union[List, None],
    ):
        """
            helper method gets edge feature and target matrix if X, y are not specified

        :param X: ndArray Data Matrix
        :param y: target, default None
        :param use_columns: which columns to featurize if X is None
        :return: data `X` and `y`
        """
        if X is None:
            if hasattr(res, "edge_features"):
                X = res.edge_features
            else:
                logger.warning(
                    "Calling `featurize` to create data matrix `X` over edges DataFrame"
                )
                res = self._featurize_edges(res, y, use_columns)
                return self._featurize_or_get_edges_dataframe_if_X_is_None(
                    res, res.edge_features, res.edge_target, use_columns
                )
        return X, y

    def umap(
        self,
        kind: str = "nodes",
        use_columns: Union[List, None] = None,
        featurize: bool = True,  # TODO ask Leo what this was for again?
        encode_position: bool = True,
        encode_weight: bool = True,
        inplace: bool = False,
        X: np.ndarray = None,
        y: Union[np.ndarray, List] = None,
        scale: float = 2,
        n_components: int = 2,
        metric: str = "euclidean",
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        suffix: str = "",
        play: Optional[int] = 0,
        engine: str = "umap_learn",
    ):
        """
            UMAP the featurized node or edges data, or pass in your own X, y (optional).

        :param kind: `nodes` or `edges` or None. If None, expects explicit X, y (optional) matrices, and will Not
                associate them to nodes or edges. If X, y (optional) is given, with kind = [nodes, edges],
                it will associate new matrices to nodes or edges attributes.
        :param use_columns: List of columns to use for featurization if featurization hasn't been applied.
        :param featurize: Whether to re-featurize, or use previous features, and just slice into appropriate columns
        :param encode_position: whether to set default plotting bindings -- positions x,y from umap for .plot()
        :param X: ndarray of features
        :param y: ndarray of targets
        :param scale: multiplicative scale for pruning weighted edge DataFrame gotten from UMAP (mean + scale *std)
        :param n_components: number of components in the UMAP projection, default 2
        :param metric: UMAP metric, default 'euclidean'. Other useful ones are 'hellinger', '..'
                see (UMAP-LEARN)[https://umap-learn.readthedocs.io/en/latest/parameters.html] documentation for more.
        :param n_neighbors: number of nearest neighbors to include for UMAP connectivity, lower makes more compact layouts. Minimum 2.
        :param min_dist: float between 0 and 1, lower makes more compact layouts.
        :param suffix: optional suffix to add to x, y attributes of umap.
        :param engine: selects which engine to use to calculate UMAP: NotImplemented yet, default UMAP-LEARN
        :return: self, with attributes set with new data
        """
        self.suffix = suffix
        xy = None
        umap_kwargs = dict(
            n_components=n_components,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        if inplace:
            res = self
        else:
            res = self.bind()

        res._set_new_kwargs(**umap_kwargs)

        if kind == "nodes":
            # make a node_entity to index dict
            # nodes = res.materialize_nodes()
            if hasattr(res, '_node') and res._node is not None:
                nodes = res._nodes[res._node].values
                assert len(nodes) == len(
                    np.unique(nodes)
                ), "There are repeat entities in node table"
                index_to_nodes_dict = dict(zip(range(len(nodes)), nodes))
            else:
                index_to_nodes_dict = None

            X, y = self._featurize_or_get_nodes_data_if_X_is_None(
                res, X, y, use_columns
            )
            xy = res.fit_transform(X, y)
            res.weighted_adjacency_nodes = res._weighted_adjacency
            res.node_embedding = xy
            # TODO add edge filter so graph doesn't have double edges
            res.weighted_edges_df_from_nodes = prune_weighted_edges_df(
                res._weighted_edges_df,
                scale=scale,
                index_to_nodes_dict=index_to_nodes_dict,
            )
        elif kind == "edges":
            X, y = self._featurize_or_get_edges_dataframe_if_X_is_None(
                res, X, y, use_columns
            )
            xy = res.fit_transform(X, y)
            res.weighted_adjacency_edges = res._weighted_adjacency
            res.edge_embedding = xy
            res.weighted_edges_df_from_edges = prune_weighted_edges_df(
                res._weighted_edges_df, scale=scale, index_to_nodes_dict=None
            )
        elif kind is None:
            logger.warning(
                f"kind should be one of `nodes` or `edges` unless you are passing explicit matrices"
            )
            if X is not None:
                logger.info(f"New Matrix `X` passed in for UMAP-ing")
                xy = res.fit_transform(X, y)
                res._xy = xy
                res._weighted_edges_df = prune_weighted_edges_df(
                    res._weighted_edges_df, scale=scale
                )
                logger.info(
                    f"Reduced Coordinates are stored in `._xy` attribute and "
                    f"pruned weighted_edge_df in `._weighted_edges_df` attribute"
                )
            else:
                logger.error(
                    f"If `kind` is `None`, `X` and optionally `y` must be given"
                )
        else:
            raise ValueError(
                f"`kind` needs to be one of `nodes`, `edges`, `None`, got {kind}"
            )
        res = self._bind_xy_from_umap(res, kind, encode_position, encode_weight, play)
        if inplace:
            return None
        return res

    def _bind_xy_from_umap(
        self,
        res: Any,
        kind: str,
        encode_position: bool,
        encode_weight: bool,
        play: Optional[int],
    ):
        # todo make sure xy is two dim, might be 3 or more....
        df = res._nodes if kind == "nodes" else res._edges

        df = df.copy(deep=False)
        x_name = config.X + self.suffix
        y_name = config.Y + self.suffix
        df[x_name] = res.embedding_.T[0]
        df[y_name] = res.embedding_.T[1]

        res = res.nodes(df) if kind == "nodes" else res.edges(df)

        if encode_weight and kind == "nodes":
            w_name = config.WEIGHT + self.suffix
            umap_df = res.weighted_edges_df_from_nodes.copy(deep=False)
            umap_df = umap_df.rename({config.WEIGHT: w_name})
            res = res.edges(umap_df, config.SRC, config.DST)
            res = res.bind(edge_weight=w_name)

        if encode_position and kind == "nodes":
            if play is not None:
                return res.bind(point_x=x_name, point_y=y_name).layout_settings(
                    play=play
                )
            else:
                return res.bind(point_x=x_name, point_y=y_name)

        if encode_weight and kind == "edges":
            w_name = config.WEIGHT + self.suffix
            umap_df = res.weighted_edges_df_from_edges.copy(deep=False)
            umap_df = umap_df.rename({config.WEIGHT: w_name})
            res = res.edges(umap_df, config.SRC, config.DST)
            res = res.bind(edge_weight=w_name)

        if encode_position and kind == "edges":
            if play is not None:
                return res.bind(point_x=x_name, point_y=y_name).layout_settings(
                    play=play
                )
            else:
                return res.bind(point_x=x_name, point_y=y_name)

        return res


# def get_columns(use_columns, X):
#     cols = X.columns
#     good_cols = []
#     for c in cols:
#         pass
#
#
# def safe_suffix(suffix: str):
#     if suffix.startswith("_"):
#         return suffix
#     else:
#         return "_" + suffix


__notes__ = """
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

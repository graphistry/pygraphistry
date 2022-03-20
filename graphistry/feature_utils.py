import copy, numpy as np, pandas as pd
from functools import partial
from time import time
from typing import List, Union, Dict, Callable, Any, Tuple, Optional

from .ai_utils import setup_logger
from .compute import ComputeMixin
from . import constants as config
from .umap_utils import UMAPMixin

logger = setup_logger(name=__name__, verbose=False)


import_exn = None
try:
    import scipy, scipy.sparse, torch
    from dirty_cat import (
        SuperVectorizer,
        #SimilarityEncoder,
        #TargetEncoder,
        #MinHashEncoder,
        GapEncoder,
    )
    from sentence_transformers import SentenceTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        MinMaxScaler,
        QuantileTransformer,
        StandardScaler,
        RobustScaler,
        MultiLabelBinarizer,
        KBinsDiscretizer,
    )
    has_dependancy = True

except ModuleNotFoundError as e:
    logger.debug(
        f"AI Packages not found, trying running `pip install graphistry[ai]`",
        exc_info=True,
    )
    import_exn = e
    has_dependancy = False
    scipy = (Any,)
    torch = Any
    SuperVectorizer = (Any,)
    #SimilarityEncoder = (Any,)
    #TargetEncoder = (Any,)
    #MinHashEncoder = (Any,)
    GapEncoder = (Any,)
    SentenceTransformer = Any
    SimpleImputer = (Any,)
    MultiLabelBinarizer = Any
    MinMaxScaler = Any
    QuantileTransformer = Any
    StandardScaler = Any
    RobustScaler = Any
    KBinsDiscretizer = Any
    scipy = Any
    torch = Any

def assert_imported():
    if not has_dependancy:
        raise import_exn


def get_train_test_sets(X, y, test_size):
    if test_size is None:
        return X, None, y, None

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


# #################################################################################
#
#      Pandas Helpers
#
# ###############################################################################


def safe_divide(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0.0, casting="unsafe")


def features_without_target(
    df: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series, np.ndarray, List]
) -> pd.DataFrame:
    """
        Checks if y DataFrame column name is in df, and removes it from df if so
    _________________________________________________________________________

    :param df: model DataFrame
    :param y: target DataFrame
    :return: DataFrames of model and target
    """
    if y is None:
        return df
    remove_cols = []
    if hasattr(y, "columns") and hasattr(df, "columns"):
        yc = y.columns
        xc = df.columns
        for c in yc:
            if c in xc:
                remove_cols.append(c)
    else:
        logger.warning(f"Target is not of type(DataFrame) and has no columns")
    if len(remove_cols):
        logger.info(f"Removing {remove_cols} columns from DataFrame")
        tf = df.drop(columns=remove_cols, errors="ignore")
        return tf
    return df

def remove_node_column_from_ndf_and_return_ndf(g):
    """
        Helper method to make sure that node name is not featurized
    _________________________________________________________________________

    :param res:
    :return: node DataFrame with or without node column
    """
    if g._node is not None:
        node_label = g._node
        if node_label is not None and node_label in g.columns:
            logger.info(
                f"removing node column `{node_label}` so we do not featurize it"
            )
            return g._nodes.drop(columns=[node_label], errors="ignore")
    return g._nodes


def remove_internal_namespace_if_present(df: pd.DataFrame):
    """
        Some tranformations below add columns to the DataFrame, this method removes them before featurization
        Will not drop if suffix is added during UMAP-ing
    _________________________________________________________________________

    :param df: DataFrame
    :return: DataFrame with dropped columns in reserved namespace
    """
    if df is None:
        return None
    # here we drop all _namespace like _x, _y, etc, so that featurization doesn't include them idempotent-ly
    reserved_namespace = [
        config.X,
        config.Y,
        config.SRC,
        config.DST,
        config.WEIGHT,
        config.IMPLICIT_NODE_ID,
        'index' # in umap, we add
    ]
    df = df.drop(columns=reserved_namespace, errors="ignore")
    return df


# #########################################################################################
#
#  Torch helpers
#
# #########################################################################################


def convert_to_torch(X_enc: pd.DataFrame, y_enc: Union[pd.DataFrame, None]):
    """
        Converts X, y to torch tensors compatible with ndata/edata of DGL graph
    _________________________________________________________________________
    :param X_enc: DataFrame Matrix of Values for Model Matrix
    :param y_enc: DataFrame Matrix of Values for Target
    :return: Dictionary of torch encoded arrays
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
#      Featurization Functions and Utils
#
# ###############################################################################

def get_dataframe_by_column_dtype(df, include=None, exclude=None):
    # verbose function that might be overkill.
    if exclude is not None:
        df = df.select_dtypes(exclude=exclude)
    if include is not None:
        df = df.select_dtypes(include=include)
    return df


def group_columns_by_dtypes(df: pd.DataFrame, verbose: bool = True) -> Dict:
    # very useful on large DataFrames, super useful if we use a feature_column type transformer too
    gtypes = df.columns.to_series().groupby(df.dtypes).groups
    gtypes = {k.name: list(v) for k, v in gtypes.items()}
    if verbose:
        for k, v in gtypes.items():
            logger.info(f"{k} has {len(v)} members")
    return gtypes


def set_to_numeric(df: pd.DataFrame, cols: List, fill_value: float = 0.0):
    df[cols] = pd.to_numeric(df[cols], errors="coerce").fillna(fill_value)


def set_to_datetime(df: pd.DataFrame, cols: List, new_col: str):
    # eg df["Start_Date"] = pd.to_datetime(df[['Month', 'Day', 'Year']])
    df[new_col] = pd.to_datetime(df[cols], errors="coerce").fillna(0)


def set_to_bool(df: pd.DataFrame, col: str, value: Any):
    df[col] = np.where(df[col] == value, True, False)


def where_is_currency_column(df: pd.DataFrame, col: str):
    # simple heuristics:
    def check_if_currency(x: str):
        if "$" in x:  ## hmmm need to add for ALL currencies...
            return True
        if "," in x:  # and ints next to it
            return True
        try:
            x = float(x)
            return True
        except:
            return False

    mask = df[col].apply(lambda x: check_if_currency)
    return mask


def set_currency_to_float(df: pd.DataFrame, col: str, return_float: bool = True):
    from re import sub
    from decimal import Decimal

    def convert_money_string_to_float(money: str):
        value = Decimal(sub(r"[^\d\-.]", "", money))  # preserves minus signs
        if return_float:
            return float(value)
        return value

    mask = where_is_currency_column(df, col)
    df[col, mask] = df[col, mask].apply(convert_money_string_to_float)


def is_dataframe_all_numeric(df: pd.DataFrame) -> bool:
    is_all_numeric = True
    for k in df.dtypes.unique():
        if k in ["float64", "int64", "float32", "int32"]:
            continue
        else:
            is_all_numeric = False
    return is_all_numeric


def find_bad_set_columns(df: pd.DataFrame, bad_set: List = ["[]"]):
    """
        Finds columns that if not coerced to strings, will break processors.
    ----------------------------------------------------------------------------
    :param df: DataFrame
    :param bad_set: List of strings to look for.
    :return: list
    """
    gtypes = group_columns_by_dtypes(df, verbose=True)
    bad_cols = []
    for k in gtypes.keys():
        for col in gtypes[k]:
            if col in df.columns:
                mask = df.astype(str)[col].isin(bad_set)
                if any(mask):
                    print(k, col)
                    bad_cols.append(col)
    return bad_cols


# #########################################################################################
#
#      Text Utils
#
# #########################################################################################


def check_if_textual_column(
    df: pd.DataFrame, col: str, confidence: float = 0.35, min_words: float = 2.5
) -> bool:
    """
        Checks if `col` column of df is textual or not using basic heuristics
    ___________________________________________________________________________

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


def get_textual_columns(
    df: pd.DataFrame, confidence: float = 0.35, min_words: float = 2.5
) -> List:
    """
        Collects columns from df that it deems are textual.
    _________________________________________________________________________

    :param df: DataFrame
    :return: list of columns names
    """
    text_cols = []
    for col in df.columns:
        if check_if_textual_column(df, col, confidence=confidence, min_words=min_words):
            text_cols.append(col)
    if len(text_cols) == 0:
        logger.info(f"No Textual Columns were found")
    return text_cols


# #########################################################################################
#
#      Featurization Utils
#
# #########################################################################################

def impute_and_scale_matrix(
    X: np.ndarray,
    use_scaler: str = "minmax",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 5,
    encode: str = "ordinal",
    strategy: str = "uniform",
    keep_n_decimals: int = 5,
):
    """
        Helper function for imputing and scaling np.ndarray data using different scaling transformers.
    :param X: np.ndarray
    :param impute: whether to run imputing or not
    :param use_scaler: string in ["minmax", "quantile", "zscale", "robust"], selects scaling transformer
    :param n_quantiles: if use_scaler = 'quantile', sets the quantile bin size.
    :param output_distribution: if use_scaler = 'quantile', can return distribution as ["normal", "uniform"]
    :param quantile_range: if use_scaler = 'robust', sets the quantile range.
    :params TODO add kbins desc
    :return: scaled array, imputer instances or None, scaler instance or None
    """
    available_preprocessors = ["minmax", "quantile", "zscale", "robust", "kbins"]
    available_quantile_distributions = ["normal", "uniform"]

    imputer = None
    res = X
    if impute:
        logger.info(f"Imputing Values using mean strategy")
        # impute values
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer = imputer.fit(X)
        res = imputer.transform(X)

    scaler = None
    if use_scaler == "minmax":
        # scale the resulting values column-wise between min and max column values and sets them between 0 and 1
        scaler = MinMaxScaler()
    elif use_scaler == "quantile":
        assert output_distribution in available_quantile_distributions, logger.error(
            f"output_distribution must be in {available_quantile_distributions}, got {output_distribution}"
        )
        scaler = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution=output_distribution
        )
    elif use_scaler == "zscale":
        scaler = StandardScaler()
    elif use_scaler == "robust":
        scaler = RobustScaler(quantile_range=quantile_range)
    elif use_scaler == "kbins":
        scaler = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    elif use_scaler is None:
        return res, imputer, scaler
    else:
        logger.error(
            f"`scaling` must be on of {available_preprocessors} or {None}, got {scaler}.\nData is not scaled"
        )
        return res, imputer, scaler

    logger.info(f"Applying {use_scaler}-Scaling")
    res = scaler.fit_transform(res)
    res = np.round(
        res, decimals=keep_n_decimals
    )  # since zscale with have small negative residuals (-1e-17) and that kills Hellinger in umap..
    return res, imputer, scaler


def impute_and_scale_df(
    df,
    use_scaler: str = "minmax",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 5,
    encode: str = "ordinal",
    strategy: str = "uniform",
    keep_n_decimals: int = 5,

):
    columns = df.columns
    index = df.index

    if not is_dataframe_all_numeric(df):
        logger.warn(
            f"Impute and Scaling can only happen on a Numeric DataFrame.\n -- Try featurizing the DataFrame first using graphistry.featurize(..)"
        )
        return df

    X = df.values
    res, imputer, scaler = impute_and_scale_matrix(
        X,
        impute=impute,
        use_scaler=use_scaler,
        n_quantiles=n_quantiles,
        quantile_range=quantile_range,
        output_distribution=output_distribution,
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        keep_n_decimals=keep_n_decimals
    )

    return pd.DataFrame(res, columns=columns, index=index), imputer, scaler



def encode_textual(
    df: pd.DataFrame,
    confidence: float = 0.35,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
):
    t = time()
    model = SentenceTransformer(model_name)

    text_cols = get_textual_columns(df, confidence=confidence, min_words=min_words)
    embeddings = np.zeros((len(df), 1))  # just a placeholder so we can use np.c_

    if text_cols:
        for col in text_cols:
            logger.info(f"-Calculating Embeddings for column `{col}`")
            emb = model.encode(df[col].values)
            embeddings = np.c_[embeddings, emb]
        logger.info(
            f"Encoded Textual data at {len(df)/(len(text_cols)*(time()-t)/60):.2f} rows per column minute"
        )

    return embeddings, text_cols


def process_textual_or_other_dataframes(
    df: pd.DataFrame,
    y: pd.DataFrame,
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    use_scaler: Union[str, None] = "robust",
    confidence: float = 0.35,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    # test_size: Union[bool, None] = None,
) -> Union[pd.DataFrame, Callable, Any]:
    """
        Automatic Deep Learning Embedding of Textual Features,
        with the rest of the columns taken care of by dirty_cat
    _________________________________________________________________________

    :param df: pandas DataFrame of data
    :param y: pandas DataFrame of targets
    :param use_scaler: None or string in ['minmax', 'zscale', 'robust', 'quantile']
    :param n_topics: number of topics in Gap Encoder
    :param use_scaler:
    :param confidence: Number between 0 and 1, will pass column for textual processing if total entries are string
            like in a column and above this relative threshold.
    :param min_words: Number greater than 1 that sets the threshold for average number of words to include column for
            textual sentence encoding. Lower values means that columns will be labeled textual and sent to sentence-encoder
    :param model_name: SentenceTransformer model name. See available list at
            https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
    :return: X_enc, y_enc, data_encoder, label_encoder
    """
    t = time()
    if len(df) == 0 or df.empty:
        logger.warning(f"DataFrame seems to be Empty")

    embeddings, text_cols = encode_textual(
        df, confidence=confidence, min_words=min_words, model_name=model_name
    )

    other_df = df.drop(columns=text_cols, errors="ignore")

    X_enc, y_enc, data_encoder, label_encoder = process_dirty_dataframes(
        other_df,
        y,
        cardinality_threshold=cardinality_threshold,
        cardinality_threshold_target=cardinality_threshold_target,
        n_topics=n_topics,
        use_scaler=None,  # set to None so that it happens later
    )

    faux_columns = list(
        range(embeddings.shape[1] - 1)
    )  # minus 1 since the first column is just placeholder

    if data_encoder is not None:
        embeddings = np.c_[embeddings, X_enc.values]
        columns = faux_columns + list(X_enc.columns.values)
    else:
        logger.warning(f'! Data Encoder is {data_encoder}')
        columns = faux_columns  # just sentence-transformers

    # now remove the leading zeros
    embeddings = embeddings[:, 1:]
    if use_scaler:
        embeddings, _, _ = impute_and_scale_matrix(embeddings, use_scaler=use_scaler)

    X_enc = pd.DataFrame(embeddings, columns=columns)
    logger.info(
        f"--The entire Textual and/or other encoding process took {(time()-t)/60:.2f} minutes"
    )
    return X_enc, y_enc, data_encoder, label_encoder


def get_cardinality_ratio(df: pd.DataFrame):
    """Calculates ratio of unique values to total number of rows of DataFrame
    --------------------------------------------------------------------------
    :param df: DataFrame
    """
    ratios = {}
    for col in df.columns:
        ratio = df[col].nunique() / len(df)
        ratios[col] = ratio
    return ratios


def make_dense(X):
    if scipy.sparse.issparse(X):
        return X.todense()
    return X


def process_dirty_dataframes(
    ndf: pd.DataFrame,
    y: pd.DataFrame,
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    use_scaler: Union[str, None] = None,
) -> Union[pd.DataFrame, Callable, Any]:
    """
        Dirty_Cat encoder for record level data. Will automatically turn
        inhomogeneous dataframe into matrix using smart conversion tricks.
    _________________________________________________________________________

    :param ndf: node DataFrame
    :param y: target DataFrame or series
    :param cardinality_threshold: For ndf columns, below this threshold, encoder is OneHot, above, it is GapEncoder
    :param cardinality_threshold_target: For target columns, below this threshold, encoder is OneHot, above, it is GapEncoder
    :param n_topics: number of topics for GapEncoder, default 42
    :param use_scaler: None or string in ['minmax', 'zscale', 'robust', 'quantile']
    :return: Encoded data matrix and target (if not None), the data encoder, and the label encoder.
    """
    t = time()
    data_encoder = SuperVectorizer(
        auto_cast=True,
        cardinality_threshold=cardinality_threshold,
        high_card_cat_transformer=GapEncoder(n_topics),
        # numerical_transformer=StandardScaler(), This breaks since -- AttributeError: Transformer numeric (type StandardScaler)
        #  does not provide get_feature_names.
        datetime_transformer=None,  # TODO add a smart datetime -> histogram transformer
    )
    label_encoder = None
    y_enc = None
    if not ndf.empty:
        if not is_dataframe_all_numeric(ndf):
            logger.info("Encoding DataFrame might take a few minutes --------")
            X_enc = data_encoder.fit_transform(ndf, y)
            X_enc = make_dense(X_enc)
            # X_enc = X_enc.astype(float)
            all_transformers = data_encoder.transformers
            features_transformed = data_encoder.get_feature_names_out()
            logger.info(f"-Shape of data {X_enc.shape}\n")
            logger.info(f"-Transformers: \n{all_transformers}\n")
            logger.info(f"-Transformed Columns: \n{features_transformed[:20]}...\n")
            logger.info(f"--Fitting on Data took {(time() - t) / 60:.2f} minutes\n")
            X_enc = pd.DataFrame(X_enc, columns=features_transformed)
            X_enc = X_enc.fillna(0)
        else:
            # if we pass only a numeric DF, data_encoder throws
            # RuntimeError: No transformers could be generated !
            logger.info(f"-*-*-DataFrame is already completely numeric")
            X_enc = ndf.astype(float)
            data_encoder = False  ## DO NOT SET THIS TO NONE
            features_transformed = ndf.columns
            logger.info(f"-Shape of data {X_enc.shape}\n")
            logger.info(f"-Columns: {features_transformed[:20]}...\n")

        if use_scaler is not None:
            X_enc, _, _ = impute_and_scale_df(X_enc, use_scaler=use_scaler)
    else:
        X_enc = None
        data_encoder = None
        logger.info(f"*Given DataFrame seems to be empty")

    if y is not None:
        if not is_dataframe_all_numeric(y):
            t2 = time()
            logger.info(f"-Fitting Targets --\n")
            label_encoder = SuperVectorizer(
                auto_cast=True,
                cardinality_threshold=cardinality_threshold_target,
                datetime_transformer=None,  # TODO add a smart datetime -> histogram transformer
            )
            y_enc = label_encoder.fit_transform(y)
            y_enc = make_dense(y_enc)
            labels_transformed = label_encoder.get_feature_names_out()
            y_enc = pd.DataFrame(np.array(y_enc), columns=labels_transformed)
            y_enc = y_enc.fillna(0)

            logger.info(f"-Shape of target {y_enc.shape}")
            logger.info(f"-Target Transformers used: {label_encoder.transformers}\n")
            logger.info(
                f"--Fitting SuperVectorizer on TARGET took {(time()-t2)/60:.2f} minutes\n"
            )
        else:
            logger.info(f"-*-*-Target DataFrame is already completely numeric")
            y_enc = y

    return X_enc, y_enc, data_encoder, label_encoder


def process_edge_dataframes(
    edf: pd.DataFrame,
    y: pd.DataFrame,
    src: str,
    dst: str,
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    use_scaler: Union[str, None] = None,
    confidence: float = 0.35,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    featurize=True
) -> Union[pd.DataFrame, Callable, Any]:
    """
        Custom Edge-record encoder. Uses a MultiLabelBinarizer to generate a src/dst vector
        and then process_textual_or_other_dataframes that encodes any other data present in edf,
        textual or not.

    :param edf: pandas DataFrame of features
    :param y: pandas DataFrame of labels
    :param src: source column to select in edf
    :param dst: destination column to select in edf
    :param use_scaler: None or string in ['minmax', 'zscale', 'robust', 'quantile']
    :return: Encoded data matrix and target (if not None), the data encoders, and the label encoder.
    """

    #FIXME what to return?
    if not featurize:
        return edf, y, None, None

    t = time()
    mlb_pairwise_edge_encoder = MultiLabelBinarizer()
    source = edf[src]
    destination = edf[dst]
    logger.info(f"Encoding Edges using MultiLabelBinarizer")
    T = mlb_pairwise_edge_encoder.fit_transform(zip(source, destination))
    T = 1.0 * T  # coerce to float, or divide= will throw error under z_scale below
    logger.info(f"-Shape of Edge-2-Edge encoder {T.shape}")

    other_df = edf.drop(columns=[src, dst])
    logger.info(
        f"-Rest of DataFrame has columns: {other_df.columns} and is not empty"
        if not other_df.empty
        else f"-Rest of DataFrame has columns: {other_df.columns} and is empty"
    )
    X_enc, y_enc, data_encoder, label_encoder = process_textual_or_other_dataframes(
        other_df,
        y,
        cardinality_threshold=cardinality_threshold,
        cardinality_threshold_target=cardinality_threshold_target,
        n_topics=n_topics,
        use_scaler=None,
        confidence=confidence,
        min_words=min_words,
        model_name=model_name,
    )

    if data_encoder is not None:
        columns = list(mlb_pairwise_edge_encoder.classes_) + list(X_enc.columns)
        T = np.c_[T, X_enc.values]
    elif (
        data_encoder is False and not X_enc.empty
    ):  # means other_df was all numeric, data_encoder is False, and we can't get feature names
        T = np.c_[T, X_enc.values]
        columns = list(mlb_pairwise_edge_encoder.classes_) + list(other_df.columns)
    else:  # if other_df is empty
        logger.info("-other_df is empty")
        columns = list(mlb_pairwise_edge_encoder.classes_)

    if use_scaler:
        T, _, _ = impute_and_scale_matrix(
            T,
            use_scaler=use_scaler,
            impute=True,
            n_quantiles=100,
            quantile_range=(25, 75),
            output_distribution="normal",
        )

    X_enc = pd.DataFrame(T, columns=columns)
    logger.info(f"--Created an Edge feature matrix of size {T.shape}")
    logger.info(f"**The entire Edge encoding process took {(time()-t)/60:.2f} minutes")
    # get's us close to `process_nodes_dataframe
    # TODO how can I meld mlb and sup_vec??? Difficult as it is not a per column transformer...
    return X_enc, y_enc, [mlb_pairwise_edge_encoder, data_encoder], label_encoder


# #################################################################################
#
#      Assemble Processors into useful options
#
# ###############################################################################

processors_node = {
    "highCardTarget": partial(
        process_textual_or_other_dataframes,
    )
}


# #################################################################################
#
#      Vectorizer Class + Helpers
#
# ###############################################################################


def prune_weighted_edges_df_and_relabel_nodes(
    wdf: pd.DataFrame, scale: float = 0.1, index_to_nodes_dict: Dict = None
) -> pd.DataFrame:
    """
        Prune the weighted edge DataFrame so to return high fidelity similarity scores.

    :param wdf: weighted edge DataFrame gotten via UMAP
    :param scale: lower values means less edges > (max - scale * std)
    :param index_to_nodes_dict: dict of index to node name; remap src/dst values if provided 
    :return: pd.DataFrame
    """
    # we want to prune edges, so we calculate some statistics
    desc = wdf.describe()
    eps = 1e-3

    mean = desc[config.WEIGHT]["mean"]
    std = desc[config.WEIGHT]["std"]
    max_val = desc[config.WEIGHT]["max"]+eps
    min_val = desc[config.WEIGHT]["min"]-eps
    thresh = np.max([max_val - scale, min_val]) # if std =0 we add eps so we still have scale in the equation
    
    logger.info(
        f"edge weights: mean({mean:.2f}), std({std:.2f}), max({max_val}), min({min_val:.2f}), thresh({thresh:.2f})"
    )
    wdf2 = wdf[
        wdf[config.WEIGHT] >= thresh
    ]  # adds eps so if scale = 0, we have small window/wiggle room
    logger.info(
        f"Pruning weighted edge DataFrame from {len(wdf):,} to {len(wdf2):,} edges."
    )
    if index_to_nodes_dict is not None:
        wdf2 = wdf2.replace({
            config.SRC: index_to_nodes_dict,
            config.DST: index_to_nodes_dict,
        })
    return wdf2


class FeatureMixin(ComputeMixin, UMAPMixin):
    """
        FeatureMixin for automatic featurization of nodes and edges DataFrames.
        Subclasses UMAPMixin for umap-ing of automatic features.
    
        TODO: add example usage doc
    """

    def __init__(self, *args, **kwargs):
        pass
        #super().__init__()
        #ComputeMixin.__init__(self, *args, **kwargs)
        # FeatureMixin.__init__(self, *args, **kwargs)
        #UMAPMixin.__init__(self, *args, **kwargs)
        
    def _node_featurizer(self, *args, **kwargs):
        return process_textual_or_other_dataframes(*args, **kwargs)

    def _featurize_nodes(
        self,
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None,
        use_columns: Union[List, None] = None,
        use_scaler: Union[str, None] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 120,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        remove_node_column: bool = True,
        featurize: bool = True
    ):
        ndf = remove_node_column_from_ndf_and_return_ndf(self, remove_node_column)
        # TODO move the columns select after the featurizer?
        ndf = ndf[use_columns] if use_columns is not None else ndf
        ndf = features_without_target(ndf, y)
        ndf = remove_internal_namespace_if_present(ndf)


        res = self.nodes(
            self._nodes.assign(**{
                #FIXME unnecessary?
                config.IMPLICIT_NODE_ID: range(len(ndf))
            })
        )

        if featurize:

            assert_imported()

            # now vectorize it all
            X_enc, y_enc, data_vec, label_vec = self._node_featurizer(
                ndf,
                y=y,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
            )
            res.node_features = X_enc
            res.node_target = y_enc
            res.node_encoder = data_vec
            res.node_target_encoder = label_vec

        else:
            res.node_features = ndf

        return res

    def _featurize_edges(
        self,
        y: Union[pd.DataFrame, np.ndarray],
        use_columns: Union[List, None] = None,
        use_scaler: Union[str, None] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 20,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        featurize: bool = True
    ):
        # TODO move the columns select after the featurizer
        if use_columns is not None:
            use_columns = list(set(use_columns + list([self._source, self._destination])))
            
        edf = self._edges[ use_columns ] if use_columns is not None else self._edges
        edf = features_without_target(edf, y)

        res = self.bind()

        if featurize:

            assert_imported()

            X_enc, y_enc, [mlb, data_vec], label_vec = process_edge_dataframes(
                edf=edf,
                y=y,
                src=self._source,
                dst=self._destination,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
                featurize=featurize
            )
            res.edge_features = X_enc
            res.edge_target = y_enc
            res.edge_encoders = [mlb, data_vec]
            res.edge_target_encoder = label_vec
        else:
            res.edge_features = edf

        return res

    def featurize(
        self,
        kind: str = "nodes",
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None,
        use_columns: Union[List, None] = None,
        use_scaler: Union[str, None] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        remove_node_column: bool = True,
        inplace: bool = False,
    ):
        """
            Featurize Nodes or Edges of the Graph.

        :param kind: specify whether to featurize `nodes` or `edges`
        :param y: Optional Target, default None. If .featurize came with a target, it will use that target.
        :param use_columns: Specify which DataFrame columns to use for featurization, if any.
        :param remove_node_column:
        :param use_scaler:
        :param inplace: whether to not return new graphistry instance or not, default False
        :return: self, with new attributes set by the featurization process

        """
        assert_imported()
        if inplace:
            res = self
        else:
            res = self.bind()

        if kind == "nodes":
            res = res._featurize_nodes(
                y=y,
                use_columns=use_columns,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
                remove_node_column=remove_node_column,
                featurize=True
            )
        elif kind == "edges":
            res = res._featurize_edges(
                y=y,
                use_columns=use_columns,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
                featurize=True
            )
        else:
            logger.warning(f"One may only featurize `nodes` or `edges`, got {kind}")
            return self
        if not inplace:
            return res

    def _featurize_or_get_nodes_dataframe_if_X_is_None(
        self,
        X: Union[np.ndarray, None],
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None,
        use_columns: Union[List, None] = None,
        use_scaler: Union[str, None] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        remove_node_column: bool = True,
        featurize: bool = True
    ):
        """
            helper method gets node feature and target matrix if X, y are not specified.
            if X, y are specified will set them as `node_target` and `node_target` attributes
        ---------------------------------------------------------------------------------------
        :param X: ndArray Data Matrix, default None
        :param y: target, default None
        :param use_columns: which columns to featurize if X is None
        :param use_scaler:
        :param featurize: if True, will force re-featurization -- useful in DGL_utils if we want to run other scalers
        :return: data `X` and `y`
        """
        if featurize:
            # remove node_features
            if hasattr(self, "node_features"):
                delattr(self, "node_features")

        if X is None:
            if hasattr(self, "node_features"):
                X = self.node_features
                logger.info(f"Found Node features in `res`")
            else:
                logger.warning(
                    "Calling `featurize` to create data matrix `X` over nodes DataFrame"
                )
                res = res._featurize_nodes(
                    y=y,
                    use_columns=use_columns,
                    use_scaler=use_scaler,
                    cardinality_threshold=cardinality_threshold,
                    cardinality_threshold_target=cardinality_threshold_target,
                    n_topics=n_topics,
                    confidence=confidence,
                    min_words=min_words,
                    model_name=model_name,
                    remove_node_column=remove_node_column,
                    featurize=featurize
                )
                return res._featurize_or_get_nodes_dataframe_if_X_is_None(
                    res.node_features, res.node_target, use_columns, use_scaler, featurize=featurize
                )  # now we are guaranteed to have node feature and target matrices.
        if y is None:
            if hasattr(self, "node_target"):
                y = self.node_target
                logger.info(
                    f"Fetching `node_target` in `res`. Target is type {type(y)}"
                )
        # now on the return the X, y will
        return X, y

    def _featurize_or_get_edges_dataframe_if_X_is_None(
        self: Any,
        X: Union[np.ndarray, None],
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None,
        use_columns: Union[List, None] = None,
        use_scaler: Union[str, None] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 20,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        featurize: bool = True
    ):
        """
            helper method gets edge feature and target matrix if X, y are not specified
        --------------------------------------------------------------------------------
        :param X: ndArray Data Matrix
        :param y: target, default None
        :param use_columns: which columns to featurize if X is None
        :return: data `X` and `y`
        """
        if X is None:
            if hasattr(self, "edge_features"):
                X = self.edge_features
            else:
                logger.info(
                    "Calling `featurize` to create data matrix `X` over edges DataFrame"
                )
                res = self._featurize_edges(
                    y=y,
                    use_columns=use_columns,
                    use_scaler=use_scaler,
                    cardinality_threshold=cardinality_threshold,
                    cardinality_threshold_target=cardinality_threshold_target,
                    n_topics=n_topics,
                    confidence=confidence,
                    min_words=min_words,
                    model_name=model_name,
                    featurize=featurize
                )
                return res._featurize_or_get_edges_dataframe_if_X_is_None(
                    res.edge_features, res.edge_target, use_columns, use_scaler
                )
        return X, y


    def filter_edges(
        self,
        scale: float = 0.1,
        index_to_nodes_dict: Optional[Dict] = None,
        inplace: bool = False,
    ):
        if inplace:
            res = self
        else:
            res = self.bind()

        if hasattr(res, "_weighted_edges_df"):
            res.weighted_edges_df_from_nodes = (
                prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df,
                    scale=scale,
                    index_to_nodes_dict=index_to_nodes_dict,
                )
            )
        else:
            logger.error(f"UMAP has not been run, run g.featurize(...).umap(...) first")

        # write new res._edges df
        res = self._bind_xy_from_umap(
            res, "nodes", encode_position=True, encode_weight=True, play=0
        )

        if not inplace:
            return res


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

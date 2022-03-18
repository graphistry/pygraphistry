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


def check_target_not_in_features(
    df: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series, np.ndarray, List],
    remove: bool = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series, np.ndarray, List]]:
    """
        Checks if y DataFrame column name is in df, and removes it from df if so
    _________________________________________________________________________

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
    if remove and len(remove_cols):
        logger.info(f"Removing {remove_cols} columns from DataFrame")
        tf = df.drop(columns=remove_cols, errors="ignore")
        return tf, y
    return df, y  # will just pass through data


def remove_node_column_from_ndf_and_return_ndf_from_res(res, remove_node_column: bool):
    """
        Helper method to make sure that node name is not featurized
    _________________________________________________________________________

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
                return res._nodes.drop(columns=[node_label], errors="ignore")
    return res._nodes  # just pass through dataframe if false


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
    df = df.copy(deep=False).drop(columns=reserved_namespace, errors="ignore")
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
        wdf2[config.SRC] = wdf2[config.SRC].apply(lambda x: index_to_nodes_dict[x])
        wdf2[config.DST] = wdf2[config.DST].apply(lambda x: index_to_nodes_dict[x])
    return wdf2


def get_dataframe_columns(df: pd.DataFrame, columns: Union[List, None] = None):
    """
        helper method to get columns from DataFrame -- does not check if column is in DataFrame, that is up to user

    :param df: DataFrame
    :param columns: columns you want to include in analysis/featurization
    :return: DataFrame with columns
    """
    if columns is None:
        # just pass through df
        return df
    if columns == []:
        # just pass through df
        return df
    if len(columns):
        logger.info(f"returning DataFrame with columns `{columns}`")
        return df[columns]


def add_implicit_node_identifier(res):
    ndf = res._nodes
    ndf[config.IMPLICIT_NODE_ID] = range(len(ndf))


class FeatureMixin(ComputeMixin, UMAPMixin):
    """
        FeatureMixin for automatic featurization of nodes and edges DataFrames.
        Subclasses UMAPMixin for umap-ing of automatic features.
    
        TODO: add example usage doc
    """

    def __init__(self, *args, **kwargs):
        from functools import partial

        super().__init__()
        ComputeMixin.__init__(self, *args, **kwargs)
        # FeatureMixin.__init__(self, *args, **kwargs)
        UMAPMixin.__init__(self, *args, **kwargs)
        self._node_featurizer = (
            process_textual_or_other_dataframes  # partial(.., *args, **kwargs)
        )
        self._edge_featurizer = process_edge_dataframes

    def _featurize_nodes(
        self,
        res: Any,
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
    ):
        ndf = remove_node_column_from_ndf_and_return_ndf_from_res(
            res, remove_node_column
        )
        # TODO move the columns select after the featurizer?
        ndf = get_dataframe_columns(ndf, use_columns)
        ndf, y = check_target_not_in_features(ndf, y, remove=True)
        ndf = remove_internal_namespace_if_present(ndf)
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
        # set the new variables
        add_implicit_node_identifier(res)
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
        use_scaler: Union[str, None] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 20,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
    ):
        # TODO move the columns select after the featurizer
        if use_columns is not None:
            # have to add src, dst or get errors
            use_columns = copy.copy(use_columns)
            use_columns += list([res._source, res._destination])
            use_columns = list(set(use_columns))
            
        edf = get_dataframe_columns(res._edges, use_columns)
        edf, y = check_target_not_in_features(edf, y, remove=True)

        X_enc, y_enc, [mlb, data_vec], label_vec = self._edge_featurizer(
            edf=edf,
            y=y,
            src=res._source,
            dst=res._destination,
            use_scaler=use_scaler,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            confidence=confidence,
            min_words=min_words,
            model_name=model_name,
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
            res = self._featurize_nodes(
                res,
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
            )
        elif kind == "edges":
            res = self._featurize_edges(
                res,
                y=y,
                use_columns=use_columns,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
            )
        else:
            logger.warning(f"One may only featurize `nodes` or `edges`, got {kind}")
            return self
        if not inplace:
            return res

    def _featurize_or_get_nodes_dataframe_if_X_is_None(
        self,
        res: Any,
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
        featurize: bool = False,
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
            if hasattr(res, "node_features"):
                delattr(res, "node_features")

        if X is None:
            if hasattr(res, "node_features"):
                X = res.node_features
                logger.info(f"Found Node features in `res`")
            else:
                logger.warning(
                    "Calling `featurize` to create data matrix `X` over nodes DataFrame"
                )
                res = self._featurize_nodes(
                    res,
                    y=y,
                    use_columns=use_columns,
                    use_scaler=use_scaler,
                    cardinality_threshold=cardinality_threshold,
                    cardinality_threshold_target=cardinality_threshold_target,
                    n_topics=n_topics,
                    confidence=confidence,
                    min_words=min_words,
                    model_name=model_name,
                    remove_node_column=remove_node_column
                )
                return self._featurize_or_get_nodes_dataframe_if_X_is_None(
                    res, res.node_features, res.node_target, use_columns, use_scaler
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
        y: Union[pd.DataFrame, pd.Series, np.ndarray, List] = None,
        use_columns: Union[List, None] = None,
        use_scaler: Union[str, None] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 20,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
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
            if hasattr(res, "edge_features"):
                X = res.edge_features
            else:
                logger.warning(
                    "Calling `featurize` to create data matrix `X` over edges DataFrame"
                )
                res = self._featurize_edges(
                    res,
                    y=y,
                    use_columns=use_columns,
                    use_scaler=use_scaler,
                    cardinality_threshold=cardinality_threshold,
                    cardinality_threshold_target=cardinality_threshold_target,
                    n_topics=n_topics,
                    confidence=confidence,
                    min_words=min_words,
                    model_name=model_name,
                )
                return self._featurize_or_get_edges_dataframe_if_X_is_None(
                    res, res.edge_features, res.edge_target, use_columns, use_scaler
                )
        return X, y

    def umap(
        self,
        kind: str = "nodes",
        use_columns: Union[List, None] = None,
        featurize: bool = False,  # TODO ask Leo what this was for again?
        encode_position: bool = True,
        encode_weight: bool = True,
        inplace: bool = False,
        X: np.ndarray = None,
        y: Union[np.ndarray, List] = None,
        scale: float = 0.1,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        scale_xy: float = 10,
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
        :param encode_weight: if True, will set new edges_df from implicit UMAP, default True.
        :param encode_position: whether to set default plotting bindings -- positions x,y from umap for .plot()
        :param X: ndarray of features
        :param y: ndarray of targets
        :param scale: multiplicative scale for pruning weighted edge DataFrame gotten from UMAP (mean + scale *std)
        :param n_neighbors: UMAP number of nearest neighbors to include for UMAP connectivity, lower makes more compact layouts. Minimum 2.
        :param min_dist: UMAP float between 0 and 1, lower makes more compact layouts.
        :param spread: UMAP spread of values for relaxation
        :param local_connectivity: UMAP connectivity parameter
        :param repulsion_strength: UMAP repulsion strength
        :param negative_sample_rate: UMAP negative sampling rate
        :param n_components: number of components in the UMAP projection, default 2
        :param metric: UMAP metric, default 'euclidean'. Other useful ones are 'hellinger', '..'
                see (UMAP-LEARN)[https://umap-learn.readthedocs.io/en/latest/parameters.html] documentation for more.
        :param suffix: optional suffix to add to x, y attributes of umap.
        :param play: Graphistry play parameter, default 0, how much to evolve the network during clustering
        :param engine: selects which engine to use to calculate UMAP: NotImplemented yet, default UMAP-LEARN
        :return: self, with attributes set with new data
        """
        assert_imported()

        self.suffix = suffix
        xy = None
        umap_kwargs = dict(
            n_components=n_components,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=spread,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
        )

        if inplace:
            res = self
        else:
            res = self.bind()

        res._set_new_kwargs(**umap_kwargs)

        if kind == "nodes":
            # make a node_entity to index dict to match with the implicit edges gotten from UMAPing
            index_to_nodes_dict = None
            if hasattr(res, "_node") and res._node is None:  # thanks Leo
                res = res.nodes(res._nodes.reset_index(), config.IMPLICIT_NODE_ID)

            if (
                hasattr(res, "_nodes")
                and hasattr(res, "_node")
                and res._node is not None
                and hasattr(res._nodes, config.IMPLICIT_NODE_ID)
            ):
                implicit_nodes = res._nodes[
                    config.IMPLICIT_NODE_ID
                ].values  # these are the integer node ids that line up with UMAP's calculation

                nodes = res._nodes[
                    res._node
                ].values  # the named node in g.nodes(ndf, 'node_name')
                if len(np.unique(nodes)) == len(np.unique(implicit_nodes)):
                    logger.info(f"Relabeling nodes")
                    # we use this to relabel from integer values to 'node_name' given in g.nodes(ndf, 'node_name')
                    index_to_nodes_dict = dict(zip(range(len(nodes)), nodes))

            X, y = self._featurize_or_get_nodes_dataframe_if_X_is_None(
                res, X, y, use_columns
            )
            xy = scale_xy * res.fit_transform(X, y)
            res.weighted_adjacency_nodes = res._weighted_adjacency
            res.node_embedding = xy
            # TODO add edge filter so graph doesn't have double edges
            res.weighted_edges_df_from_nodes = (
                prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df,
                    scale=scale,
                    index_to_nodes_dict=index_to_nodes_dict,
                )
            )
        elif kind == "edges":
            X, y = self._featurize_or_get_edges_dataframe_if_X_is_None(
                res, X, y, use_columns
            )
            xy = scale_xy * res.fit_transform(X, y)
            res.weighted_adjacency_edges = res._weighted_adjacency
            res.edge_embedding = xy
            res.weighted_edges_df_from_edges = (
                prune_weighted_edges_df_and_relabel_nodes(
                    res._weighted_edges_df, scale=scale, index_to_nodes_dict=None
                )
            )
        elif kind is None:
            logger.warning(
                f"kind should be one of `nodes` or `edges` unless you are passing explicit matrices"
            )
            if X is not None:
                logger.info(f"New Matrix `X` passed in for UMAP-ing")
                xy = res.fit_transform(X, y)
                res._xy = xy
                res._weighted_edges_df = prune_weighted_edges_df_and_relabel_nodes(
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
        if not inplace:
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
        if kind == "nodes":
            emb = res.node_embedding
        else:
            emb = res.edge_embedding
        df[x_name] = emb.T[0]
        df[y_name] = emb.T[1]

        res = res.nodes(df) if kind == "nodes" else res.edges(df)

        if encode_weight and kind == "nodes":
            w_name = config.WEIGHT + self.suffix
            umap_df = res.weighted_edges_df_from_nodes.copy(deep=False)
            umap_df = umap_df.rename({config.WEIGHT: w_name})
            res = res.edges(umap_df, config.SRC, config.DST)
            logger.info(
                f"Wrote new edges_dataframe from UMAP embedding of shape {res._edges.shape}"
            )
            res = res.bind(edge_weight=w_name)

        if encode_position and kind == "nodes":
            if play is not None:
                return res.bind(point_x=x_name, point_y=y_name).layout_settings(
                    play=play
                )
            else:
                return res.bind(point_x=x_name, point_y=y_name)

        return res

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

import numpy as np, pandas as pd
from time import time
from typing import List, Union, Dict, Any, Optional, Tuple, TYPE_CHECKING
from typing_extensions import Literal  # Literal native to py3.8+

from graphistry.compute.ComputeMixin import ComputeMixin
from . import constants as config
from .PlotterBase import WeakValueDictionary, Plottable
from .util import setup_logger, check_set_memoize

logger = setup_logger(name=__name__, verbose=config.VERBOSE)

if TYPE_CHECKING:
    MIXIN_BASE = ComputeMixin
else:
    MIXIN_BASE = object

import_min_exn = None
import_text_exn = None

try:
    from sentence_transformers import SentenceTransformer

    has_dependancy_text = True

except ModuleNotFoundError as e:
    import_text_exn = e
    has_dependancy_text = False

try:
    import scipy, scipy.sparse
    from dirty_cat import (
        SuperVectorizer,
        GapEncoder,
    )

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        MinMaxScaler,
        QuantileTransformer,
        StandardScaler,
        RobustScaler,
        MultiLabelBinarizer,
        KBinsDiscretizer,
    )

    has_min_dependancy = True

except ModuleNotFoundError as e:
    import_min_exn = e
    has_min_dependancy = False
    SuperVectorizer = Any
    Pipeline = Any


def assert_imported_text():
    if not has_dependancy_text:
        logger.error(
            "AI Package sentence_transformers not found, trying running `pip install graphistry[ai]`"
        )
        raise import_text_exn


def assert_imported():
    if not has_min_dependancy:
        logger.error(
            "AI Packages not found, trying running `pip install graphistry[ai]`"
        )
        raise import_min_exn


# #################################################################################
#
#     Rough calltree
#
# #################################################################################

# umap
#     _featurize_or_get_nodes_dataframe_if_X_is_None
#         _featurize_nodes
#             _node_featurizer
#                 process_textual_or_other_dataframes
#                     encode_textual
#                     process_dirty_dataframes
#                     impute_and_scale_matrix
#
#    _featurize_or_get_edges_dataframe_if_X_is_None
#      _featurize_edges
#             _edge_featurizer
#                 featurize_edges:
#                 rest of df goes to equivalent of _node_featurizer
#
#      _featurize_or_get_edges_dataframe_if_X_is_None
FeatureEngineConcrete = Literal["none", "pandas", "dirty_cat", "torch"]
FeatureEngine = Literal[FeatureEngineConcrete, "auto"]


def resolve_feature_engine(feature_engine: FeatureEngine) -> FeatureEngineConcrete:

    if feature_engine in ["none", "pandas", "dirty_cat", "torch"]:
        return feature_engine  # type: ignore

    if feature_engine == "auto":
        if has_dependancy_text:
            return "torch"
        if has_min_dependancy:
            return "dirty_cat"
        return "pandas"

    raise ValueError(
        f'feature_engine expected to be "none", "pandas", "dirty_cat", "torch", or "auto" but received: {feature_engine} :: {type(feature_engine)}'
    )


YSymbolic = Optional[Union[List[str], str, pd.DataFrame]]


def resolve_y(df: Optional[pd.DataFrame], y: YSymbolic) -> pd.DataFrame:

    if isinstance(y, pd.DataFrame):
        return y

    if df is None:
        raise ValueError("Missing data for featurization")

    if y is None:
        return df[[]]  # oh brills, basically index
    elif isinstance(y, str):
        return df[[y]]
    elif isinstance(y, list):
        return df[y]
    else:
        raise ValueError(f"Unexpected type for y: {type(y)}")


XSymbolic = Optional[Union[List[str], str, pd.DataFrame]]


def resolve_X(df: Optional[pd.DataFrame], X: XSymbolic) -> pd.DataFrame:

    if isinstance(X, pd.DataFrame):
        return X

    if df is None:
        raise ValueError("Missing data for featurization")

    if X is None:
        return df
    elif isinstance(X, str):
        return df[[X]]
    elif isinstance(X, list):
        return df[X]
    else:
        raise ValueError(f"Unexpected type for X: {type(X)}")


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
    df: pd.DataFrame, y: Optional[Union[List[str], str, pd.DataFrame]] = None
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
    if y is None:
        pass
    elif isinstance(y, pd.DataFrame):
        yc = y.columns
        xc = df.columns
        for c in yc:
            if c in xc:
                remove_cols.append(c)
    elif isinstance(y, pd.Series):
        if y.name and (y.name in df.columns):
            remove_cols = [y.name]
    elif isinstance(y, List):
        remove_cols = y
    elif isinstance(y, str):
        remove_cols = [y]
    else:
        logger.warning("Target is not of type(DataFrame) and has no columns")
    if len(remove_cols):
        logger.debug(f"Removing {remove_cols} columns from DataFrame")
        tf = df.drop(columns=remove_cols, errors="ignore")
        return tf
    return df


def remove_node_column_from_ndf_and_return_ndf(g):
    """
        Helper method to make sure that node name is not featurized
    _________________________________________________________________________

    :param g: graphistry instance
    :return: node DataFrame with or without node column
    """
    if g._node is not None:
        node_label = g._node
        if node_label is not None and node_label in g._nodes.columns:
            logger.debug(
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
        "index",  # in umap, we add as reindex
    ]
    df = df.drop(columns=reserved_namespace, errors="ignore")
    return df


# #################################################################################
#
#      Featurization Functions and Utils
#
# ###############################################################################

# can also just do np.number to get all of these, thanks @LEO!
numeric_dtypes = [
    "float64",
    "float32",
    "float",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
]


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
            logger.debug(f"{k} has {len(v)} members")
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
        if isinstance(x, str):
            if "$" in x:  # hmmm need to add for ALL currencies...
                return True
            if "," in x:  # and ints next to it
                return True
        try:
            float(x)
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
        if k in numeric_dtypes:
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
    ), "probably best to have at least a word if you want to consider this a textual column?"
    if abundance >= confidence:
        # now check how many words
        n_words = df[col].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        mean_n_words = n_words.mean()
        if mean_n_words >= min_words:
            logger.debug(
                f"\n\tColumn `{col}` looks textual with mean number of words {mean_n_words:.2f}"
            )
            return True
        else:
            return False
    else:
        return False


def get_textual_columns(
    df: pd.DataFrame, confidence: float = 0.35, min_words: float = 2.5
) -> List[str]:
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
        logger.debug("No Textual Columns were found")
    return text_cols


# #########################################################################################
#
#      Featurization Utils
#
# #########################################################################################


def identity(x):
    return x


def get_ordinal_preprocessing_pipeline(
    use_scaler: str = "robust",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 5,
    encode: str = "ordinal",
    strategy: str = "uniform",
) -> Pipeline:
    """
        Helper function for imputing and scaling np.ndarray data using different scaling transformers.
    :param X: np.ndarray
    :param impute: whether to run imputing or not
    :param use_scaler: string in ["minmax", "quantile", "zscale", "robust", "kbins"], selects scaling transformer,
            default `robust`
    :param n_quantiles: if use_scaler = 'quantile', sets the quantile bin size.
    :param output_distribution: if use_scaler = 'quantile', can return distribution as ["normal", "uniform"]
    :param quantile_range: if use_scaler = 'robust', sets the quantile range.
    :param n_bins: number of bins to use in kbins discretizer
    :return: scaled array, imputer instances or None, scaler instance or None
    """
    available_preprocessors = ["minmax", "quantile", "zscale", "robust", "kbins"]
    available_quantile_distributions = ["normal", "uniform"]

    imputer = identity
    if impute:
        logger.debug("Imputing Values using mean strategy")
        # impute values
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

    scaler = identity
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
    else:
        logger.error(
            f"`scaling` must be on of {available_preprocessors} or {None}, got {scaler}.\nData is not scaled"
        )
    logger.info(f"Using {use_scaler} scaling")
    ordinal_transformer = Pipeline(steps=[("imputer", imputer), ("scaler", scaler)])

    return ordinal_transformer


def fit_pipeline(X: pd.DataFrame, transformer, keep_n_decimals: int = 5):
    """
     Helper to fit DataFrame over transformer pipeline.
     Rounds resulting matrix X by keep_n_digits if not 0,
     which helps for when transformer pipeline is scaling or imputer which sometime introduce small negative numbers,
     and umap metrics like Hellinger need to be positive
    :param X, DataFrame to transform.
    :param transformer: Pipeline object to fit and transform
    :param keep_n_decimals: Int of how many decimal places to keep in rounded transformed data
    """
    X = transformer.fit_transform(X)
    if keep_n_decimals:
        X = np.round(
            X, decimals=keep_n_decimals
        )  # since zscale with have small negative residuals (-1e-17) and that kills Hellinger in umap..
    return X


def impute_and_scale_df(
    df: pd.DataFrame,
    use_scaler: str = "robust",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 5,
    encode: str = "ordinal",
    strategy: str = "uniform",
    keep_n_decimals: int = 5,
) -> Tuple[pd.DataFrame, Optional[Pipeline]]:

    columns = df.columns
    index = df.index

    if not is_dataframe_all_numeric(df):
        logger.warning(
            "Impute and Scaling can only happen on a Numeric DataFrame.\n -- Try featurizing the DataFrame first using graphistry.featurize(..)"
        )
        return df, None

    transformer = get_ordinal_preprocessing_pipeline(
        impute=impute,
        use_scaler=use_scaler,
        n_quantiles=n_quantiles,
        quantile_range=quantile_range,
        output_distribution=output_distribution,
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
    )
    res = fit_pipeline(df, transformer, keep_n_decimals=keep_n_decimals)

    return pd.DataFrame(res, columns=columns, index=index), transformer


def encode_textual(
    df: pd.DataFrame,
    confidence: float = 0.35,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
) -> Tuple[np.ndarray, List, List]:
    import os

    t = time()
    model_name = os.path.split(model_name)[-1]
    model = SentenceTransformer(f"{model_name}")

    text_cols = get_textual_columns(df, confidence=confidence, min_words=min_words)
    embeddings = np.zeros((len(df), 1))  # just a placeholder so we can use np.c_
    columns = []
    if text_cols:
        for col in text_cols:
            logger.debug(f"-Calculating Embeddings for column `{col}`")
            # coerce to string in case there are ints, floats, nans, etc mixed into column
            emb = model.encode(df[col].astype(str).values)
            columns.extend(
                [f"{col}_{k}" for k in range(emb.shape[1])]
            )  # so we can slice by original column name
            # assuming they are unique across cols
            embeddings = np.c_[embeddings, emb]
        logger.debug(
            f"Encoded Textual data at {len(df)/(len(text_cols)*(time()-t)/60):.2f} rows per column minute"
        )

    return embeddings, text_cols, columns


def process_textual_or_other_dataframes(
    df: pd.DataFrame,
    y: pd.DataFrame,
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    use_scaler: Optional[str] = "robust",
    confidence: float = 0.35,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    feature_engine: FeatureEngineConcrete = "pandas"
    # test_size: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Any, SuperVectorizer, SuperVectorizer, Optional[Pipeline]]:
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

    logger.info("process_textual_or_other_dataframes[%s]", feature_engine)

    t = time()
    if len(df) == 0 or df.empty:
        logger.warning("DataFrame seems to be Empty")

    embeddings = np.zeros((len(df), 1))  # just a placeholder so we can use np.c_
    text_cols: List[str] = []
    columns_text: List[str] = []
    if has_dependancy_text and (feature_engine in ["torch", "auto"]):
        embeddings, text_cols, columns_text = encode_textual(
            df, confidence=confidence, min_words=min_words, model_name=model_name
        )
    else:
        logger.debug(
            f"! Skipping encoding any textual features since dependency {import_text_exn} is not met"
        )

    other_df = df.drop(columns=text_cols, errors="ignore")

    X_enc, y_enc, data_encoder, label_encoder, _ = process_dirty_dataframes(
        other_df,
        y,
        cardinality_threshold=cardinality_threshold,
        cardinality_threshold_target=cardinality_threshold_target,
        n_topics=n_topics,
        use_scaler=None,  # set to None so that it happens later
        feature_engine=feature_engine,
    )

    if data_encoder is not None:  # can be False!
        embeddings = np.c_[embeddings, X_enc.values]
        columns = columns_text + list(X_enc.columns.values)
    elif len(columns_text):
        columns = columns_text  # just sentence-transformers
    else:
        logger.warning(
            f" WARNING: Data Encoder is {data_encoder} and textual_columns are {columns_text}"
        )
        columns = list(X_enc.columns.values)  # try with this if nothing else

    # now remove the leading zeros
    embeddings = embeddings[:, 1:]
    X_enc = pd.DataFrame(embeddings, columns=columns)
    ordinal_pipeline = None
    if use_scaler and not X_enc.empty:
        X_enc, ordinal_pipeline = impute_and_scale_df(X_enc, use_scaler=use_scaler)

    logger.debug(
        f"--The entire Textual and/or other encoding process took {(time()-t)/60:.2f} minutes"
    )
    return X_enc, y_enc, data_encoder, label_encoder, ordinal_pipeline


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
    y: Optional[pd.DataFrame],
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    use_scaler: Optional[str] = None,
    feature_engine: FeatureEngineConcrete = "pandas",
) -> Tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    SuperVectorizer,
    SuperVectorizer,
    Optional[Pipeline]
]:
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

    if feature_engine == "none" or feature_engine == "pandas":
        logger.warning(
            "Featurizer returning only numeric entries in DataFrame, if any exist. No real featurizations has taken place."
        )
        return (
            ndf.select_dtypes(include=[np.number]),
            y.select_dtypes(include=[np.number]) if y is not None else None,
            False,
            False,
            False
        )

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
    ordinal_pipeline = None
    if not ndf.empty:
        if not is_dataframe_all_numeric(ndf):
            logger.debug("Encoding DataFrame might take a few minutes --------")
            X_enc = data_encoder.fit_transform(ndf, y)
            X_enc = make_dense(X_enc)
            all_transformers = data_encoder.transformers

            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                features_transformed = data_encoder.get_feature_names_out()

            logger.debug(f"-Shape of data {X_enc.shape}\n")
            logger.debug(f"-Transformers: \n{all_transformers}\n")
            logger.debug(f"-Transformed Columns: \n{features_transformed[:20]}...\n")
            logger.debug(f"--Fitting on Data took {(time() - t) / 60:.2f} minutes\n")
            X_enc = pd.DataFrame(X_enc, columns=features_transformed)
            X_enc = X_enc.fillna(0)
        else:
            # if we pass only a numeric DF, data_encoder throws
            # RuntimeError: No transformers could be generated !
            logger.debug("-*-*-DataFrame is already completely numeric")
            X_enc = ndf.astype(float)
            data_encoder = False  # DO NOT SET THIS TO NONE
            features_transformed = ndf.columns
            logger.debug(f"-Shape of data {X_enc.shape}\n")
            logger.debug(f"-Columns: {features_transformed[:20]}...\n")

        if use_scaler is not None:
            X_enc, ordinal_pipeline = impute_and_scale_df(X_enc, use_scaler=use_scaler)
    else:
        X_enc = ndf
        data_encoder = None
        logger.debug("**Given DataFrame seems to be empty")

    if y is not None and len(y.columns) > 0 and not is_dataframe_all_numeric(y):
        t2 = time()
        logger.debug("-Fitting Targets --\n%s", y.columns)
        label_encoder = SuperVectorizer(
            auto_cast=True,
            cardinality_threshold=cardinality_threshold_target,
            datetime_transformer=None,  # TODO add a smart datetime -> histogram transformer
        )
        y_enc = label_encoder.fit_transform(y)
        y_enc = make_dense(y_enc)

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            labels_transformed = label_encoder.get_feature_names_out()

        y_enc = pd.DataFrame(np.array(y_enc), columns=labels_transformed)
        y_enc = y_enc.fillna(0)

        logger.debug(f"-Shape of target {y_enc.shape}")
        logger.debug(f"-Target Transformers used: {label_encoder.transformers}\n")
        logger.debug(
            f"--Fitting SuperVectorizer on TARGET took {(time()-t2)/60:.2f} minutes\n"
        )
    else:
        y_enc = y

    return X_enc, y_enc, data_encoder, label_encoder, ordinal_pipeline


def process_edge_dataframes(
    edf: pd.DataFrame,
    y: pd.DataFrame,
    src: str,
    dst: str,
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    use_scaler: Optional[str] = None,
    confidence: float = 0.35,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    feature_engine: FeatureEngineConcrete = "pandas"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Any], Any, Optional[Pipeline]]:
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

    if feature_engine in ["none", "pandas"]:
        edf2 = edf.select_dtypes(include=[np.number])
        return edf2, y, [False, False], False, False

    t = time()
    mlb_pairwise_edge_encoder = MultiLabelBinarizer()
    source = edf[src]
    destination = edf[dst]
    logger.debug("Encoding Edges using MultiLabelBinarizer")
    T = mlb_pairwise_edge_encoder.fit_transform(zip(source, destination))
    T = 1.0 * T  # coerce to float
    logger.debug(f"-Shape of Edge-2-Edge encoder {T.shape}")

    other_df = edf.drop(columns=[src, dst])
    logger.debug(
        f"-Rest of DataFrame has columns: {other_df.columns} and is not empty"
        if not other_df.empty
        else f"-Rest of DataFrame has columns: {other_df.columns} and is empty"
    )
    (
        X_enc,
        y_enc,
        data_encoder,
        label_encoder,
        _,
    ) = process_textual_or_other_dataframes(
        other_df,
        y,
        cardinality_threshold=cardinality_threshold,
        cardinality_threshold_target=cardinality_threshold_target,
        n_topics=n_topics,
        use_scaler=None,
        confidence=confidence,
        min_words=min_words,
        model_name=model_name,
        feature_engine=feature_engine
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
        logger.debug("-other_df is empty")
        columns = list(mlb_pairwise_edge_encoder.classes_)

    X_enc = pd.DataFrame(T, columns=columns)
    ordinal_pipeline = None
    if use_scaler:
        X_enc, ordinal_pipeline = impute_and_scale_df(
            X_enc,
            use_scaler=use_scaler,
            impute=True,
            n_quantiles=100,
            quantile_range=(25, 75),
            output_distribution="normal"
        )

    logger.debug(f"--Created an Edge feature matrix of size {T.shape}")
    logger.debug(f"**The entire Edge encoding process took {(time()-t)/60:.2f} minutes")
    # get's us close to `process_nodes_dataframe
    # TODO how can I meld mlb and sup_vec??? Difficult as it is not a per column transformer...
    return (
        X_enc,
        y_enc,
        [mlb_pairwise_edge_encoder, data_encoder],
        label_encoder,
        ordinal_pipeline
    )


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
    max_val = desc[config.WEIGHT]["max"] + eps
    min_val = desc[config.WEIGHT]["min"] - eps
    thresh = np.max(
        [max_val - scale, min_val]
    )  # if std =0 we add eps so we still have scale in the equation

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
        wdf2 = wdf2.replace(
            {
                config.SRC: index_to_nodes_dict,
                config.DST: index_to_nodes_dict,
            }
        )
    return wdf2


# #################################################################################
#
#      Fast Memoize
#
# ###############################################################################


def reuse_featurization(g: Plottable, metadata: Any):  # noqa: C901
    return check_set_memoize(
        g, metadata, attribute="_feat_param_to_g", name="featurize", memoize=True
    )


class FeatureMixin(MIXIN_BASE):
    """
    FeatureMixin for automatic featurization of nodes and edges DataFrames.
    Subclasses UMAPMixin for umap-ing of automatic features.

    TODO: add example usage doc
    """

    _feature_memoize: WeakValueDictionary = WeakValueDictionary()
    _feature_params: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def _node_featurizer(self, *args, **kwargs):
        return process_textual_or_other_dataframes(*args, **kwargs)

    def _featurize_nodes(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[str] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 120,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        remove_node_column: bool = True,
        feature_engine: FeatureEngineConcrete = "pandas",
    ):
        res = self.copy()
        ndf = res._nodes
        if remove_node_column:
            ndf = remove_node_column_from_ndf_and_return_ndf(res)

        # resolve everything before setting dict so that `X = ndf[cols]` and `X = cols` resolve to same thing
        X_resolved = resolve_X(ndf, X)
        y_resolved = resolve_y(ndf, y)

        feature_engine = resolve_feature_engine(feature_engine)

        fkwargs = dict(
            X=X_resolved,
            y=y_resolved,
            use_scaler=use_scaler,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            confidence=confidence,
            min_words=min_words,
            model_name=model_name,
            remove_node_column=remove_node_column,
            feature_engine=feature_engine,
        )

        res._feature_params["nodes"] = fkwargs

        old_res = reuse_featurization(res, fkwargs)
        if old_res:
            logger.info(" --- RE-USING NODE FEATURIZATION")
            return old_res

        if self._nodes is None:
            raise ValueError(
                "Expected nodes; try running nodes.materialize_nodes() first if you only have edges"
            )

        X_resolved = features_without_target(X_resolved, y_resolved)
        X_resolved = remove_internal_namespace_if_present(X_resolved)

        if feature_engine == "none":
            X_enc = X_resolved.select_dtypes(include=np.number)
            y_enc = y_resolved
            data_vec = False
            label_vec = False
            ordinal_pipeline = False
        else:
            assert_imported()
            # now vectorize it all
            X_enc, y_enc, data_vec, label_vec, ordinal_pipeline = self._node_featurizer(
                X_resolved,
                y=y_resolved,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
                feature_engine=feature_engine,
            )

        res._node_features = X_enc
        res._node_target = y_enc
        res._node_encoder = data_vec
        res._node_target_encoder = label_vec
        res._node_ordinal_pipeline = ordinal_pipeline

        return res

    def _featurize_edges(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[str] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 20,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        feature_engine: FeatureEngineConcrete = "pandas",
    ):

        res = self.copy()
        edf = res._edges
        X_resolved = resolve_X(edf, X)
        y_resolved = resolve_y(edf, y)

        if res._source not in X_resolved:
            logger.debug("adding g._source to edge features")
            X_resolved = X_resolved.assign(**{res._source: res._edges[res._source]})
        if res._destination not in X_resolved:
            logger.debug("adding g._destination to edge features")
            X_resolved = X_resolved.assign(**{res._destination: res._edges[res._destination]})

        # now that everything is set
        fkwargs = dict(
            X=X_resolved,
            y=y_resolved,
            use_scaler=use_scaler,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            confidence=confidence,
            min_words=min_words,
            model_name=model_name,
            feature_engine=feature_engine,
        )

        res._feature_params["edges"] = fkwargs

        old_res = reuse_featurization(res, fkwargs)
        if old_res:
            logger.info(" --- RE-USING EDGE FEATURIZATION")
            return old_res

        X_resolved = features_without_target(X_resolved, y_resolved)

        if feature_engine == "none":
            X_enc = X_resolved.select_dtypes(include=np.number)
            y_enc = y_resolved
            data_vec = False
            label_vec = False
            ordinal_pipeline = None
            mlb = False

        else:
            assert_imported()

            if res._source is None:
                raise ValueError(
                    'Must have a source column to featurize edges, try g.bind(source="my_col") or g.edges(df, source="my_col")'
                )

            if res._destination is None:
                raise ValueError(
                    'Must have a destination column to featurize edges, try g.bind(destination="my_col") or g.edges(df, destination="my_col")'
                )

            (
                X_enc,
                y_enc,
                [mlb, data_vec],
                label_vec,
                ordinal_pipeline
            ) = process_edge_dataframes(
                X_resolved,
                y=y_resolved,
                src=res._source,
                dst=res._destination,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
                feature_engine=feature_engine
            )

        res._edge_features = X_enc
        res._edge_target = y_enc
        res._edge_encoders = [mlb, data_vec]
        res._edge_target_encoder = label_vec
        res._edge_ordinal_pipeline = ordinal_pipeline

        return res

    def featurize(
        self,
        kind: str = "nodes",
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[str] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        remove_node_column: bool = True,
        inplace: bool = False,
        feature_engine: FeatureEngine = "auto",
    ):
        """
            Featurize Nodes or Edges of the Graph.

        :param kind: specify whether to featurize `nodes` or `edges`
        :param X: Optional input, default None. If symbolic, evaluated against self data based on kind.
        :param y: Optional Target, default None. If .featurize came with a target, it will use that target.
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

        feature_engine = resolve_feature_engine(feature_engine)

        if kind == "nodes":
            res = res._featurize_nodes(
                X=X,
                y=y,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
                remove_node_column=remove_node_column,
                feature_engine=feature_engine,
            )
        elif kind == "edges":
            res = res._featurize_edges(
                X=X,
                y=y,
                use_scaler=use_scaler,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                confidence=confidence,
                min_words=min_words,
                model_name=model_name,
                feature_engine=feature_engine,
            )
        else:
            logger.warning(f"One may only featurize `nodes` or `edges`, got {kind}")
            return self
        if not inplace:
            return res

    def _featurize_or_get_nodes_dataframe_if_X_is_None(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[str] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        remove_node_column: bool = True,
        feature_engine: FeatureEngineConcrete = "pandas",
        reuse_if_existing=False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], MIXIN_BASE]:
        """
        helper method gets node feature and target matrix if X, y are not specified.
        if X, y are specified will set them as `_node_target` and `_node_target` attributes
        ---------------------------------------------------------------------------------------
        """
        
        res = self.bind()

        if not reuse_if_existing:  # will cause re-featurization
            res._node_features = None
            res._node_target = None

        if reuse_if_existing and res._node_features is not None:
            return res._node_features, res._node_target, res

        res = res._featurize_nodes(
            X=X,
            y=y,
            use_scaler=use_scaler,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            confidence=confidence,
            min_words=min_words,
            model_name=model_name,
            remove_node_column=remove_node_column,
            feature_engine=feature_engine,
        )

        assert res._node_features is not None  # ensure no infinite loop

        return res._featurize_or_get_nodes_dataframe_if_X_is_None(
            res._node_features,
            res._node_target,
            use_scaler,
            feature_engine=feature_engine,
            reuse_if_existing=True,
        )  # now we are guaranteed to have node feature and target matrices.

    def _featurize_or_get_edges_dataframe_if_X_is_None(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[str] = "robust",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 20,
        n_topics: int = config.N_TOPICS_DEFAULT,
        confidence: float = 0.35,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        feature_engine: FeatureEngineConcrete = "pandas",
        reuse_if_existing=False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], MIXIN_BASE]:
        """
        helper method gets edge feature and target matrix if X, y are not specified
        --------------------------------------------------------------------------------
        :param X: ndArray Data Matrix
        :param y: target, default None
        :return: data `X` and `y`
        """

        res = self.bind()

        if not reuse_if_existing:
            res._edge_features = None
            res._edge_target = None

        if reuse_if_existing and res._edge_features is not None:
            return res._edge_features, res._edge_target, res

        res = res._featurize_edges(
            X=X,
            y=y,
            use_scaler=use_scaler,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            confidence=confidence,
            min_words=min_words,
            model_name=model_name,
            feature_engine=feature_engine,
        )

        assert res._edge_features is not None  # ensure no infinite loop

        return res._featurize_or_get_edges_dataframe_if_X_is_None(
            res._edge_features, res._edge_target, use_scaler, reuse_if_existing=True
        )


__notes__ = """
    Notes:
        ~1) Given nothing but a graphistry Plottable `g`, we may minimally generate the (N, N)
        adjacency matrix as a node l(conformance test from year n-evel feature set, ironically as an edge level feature set over N unique nodes.
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

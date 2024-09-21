import copy
import numpy as np
import os
import pandas as pd
from time import time
from inspect import getmodule
import warnings
from functools import partial

from typing import (
    List,
    Union,
    Dict,
    Any,
    Optional,
    Tuple,
    TYPE_CHECKING, 
)  # noqa
from typing_extensions import Literal  # Literal native to py3.8+

from graphistry.compute.ComputeMixin import ComputeMixin
from graphistry.config import config as graphistry_config
from graphistry.features import ScalerType
from graphistry.utils.lazy_import import (
    lazy_sentence_transformers_import,
    lazy_import_has_min_dependancy,
    lazy_dirty_cat_import,
    assert_imported_text,
    assert_imported
)
from . import constants as config
from .PlotterBase import WeakValueDictionary, Plottable
from .util import setup_logger, check_set_memoize
from .ai_utils import infer_graph, infer_self_graph

# add this inside classes and have a method that can set log level
logger = setup_logger(__name__)

if TYPE_CHECKING:
    MIXIN_BASE = ComputeMixin
    try:
        from sklearn.pipeline import Pipeline
    except ImportError:
        Pipeline = Any
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = Any  # type:ignore
    try:
        from dirty_cat import (
            SuperVectorizer,
            GapEncoder,
            SimilarityEncoder,
        )
    except:
        SuperVectorizer = Any
        GapEncoder = Any
        SimilarityEncoder = Any
    try:
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.base import BaseEstimator, TransformerMixin
    except ImportError:
        FunctionTransformer = Any
        BaseEstimator = object
        TransformerMixin = object
else:
    MIXIN_BASE = object
    Pipeline = Any
    SentenceTransformer = Any
    SuperVectorizer = Any
    GapEncoder = Any
    SimilarityEncoder = Any
    FunctionTransformer = Any
    BaseEstimator = Any
    TransformerMixin = Any


def is_cudf_df(df: Any) -> bool:
    mod_str = str(getmodule(df))
    return 'cudf' in mod_str and 'dataframe' in mod_str

def is_cudf_s(s: Any) -> bool:
    mod_str = str(getmodule(s))
    return 'cudf' in mod_str and 'series' in mod_str


# ############################################################################
#
#     Rough calltree
#
# ############################################################################

# umap
#     featurize_or_get_nodes_dataframe_if_X_is_None
#         _featurize_nodes
#             _node_featurizer
#                 process_textual_or_other_dataframes
#                     encode_textual
#                     process_dirty_dataframes
#                     impute_and_scale_matrix
#
#    featurize_or_get_edges_dataframe_if_X_is_None
#      _featurize_edges
#             _edge_featurizer
#                 featurize_edges:
#                 rest of df goes to equivalent of _node_featurizer
#
#      featurize_or_get_edges_dataframe_if_X_is_None

FeatureEngineConcrete = Literal["none", "pandas", "dirty_cat", "torch"]
FeatureEngine = Literal[FeatureEngineConcrete, "auto"]


def resolve_feature_engine(
    feature_engine: FeatureEngine,
) -> FeatureEngineConcrete:  # noqa

    if feature_engine in ["none", "pandas", "dirty_cat", "torch"]:
        return feature_engine  # type: ignore

    if feature_engine == "auto":
        has_dependancy_text_, _, _ = lazy_sentence_transformers_import()
        if has_dependancy_text_:
            return "torch"
        has_min_dependancy_, _ = lazy_import_has_min_dependancy()
        if has_min_dependancy_:
            return "dirty_cat"
        return "pandas"

    raise ValueError(  # noqa
        f'feature_engine expected to be "none", '
        '"pandas", "dirty_cat", "torch", or "auto"'
        f'but received: {feature_engine} :: {type(feature_engine)}'
    )


YSymbolic = Optional[Union[List[str], str, pd.DataFrame]]


def resolve_y(df: Optional[pd.DataFrame], y: YSymbolic) -> pd.DataFrame:

    if isinstance(y, pd.DataFrame) or is_cudf_df(y):
        return y  # type: ignore

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

    if isinstance(X, pd.DataFrame) or is_cudf_df(X):
        return X  # type: ignore

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



def resolve_scaler(use_scaler: Optional[ScalerType], feature_engine: FeatureEngineConcrete) -> ScalerType:

    if use_scaler is None:
        return "none" if feature_engine == "none" else "robust"

    if feature_engine == "none" and (use_scaler is not None and use_scaler != "none"):
        raise ValueError(f"Scaling is not supported with feature_engine='none', received {use_scaler} and downgrading to 'none'")
    
    return use_scaler


def resolve_scaler_target(use_scaler_target: Optional[ScalerType], feature_engine: FeatureEngineConcrete, multilabel: bool) -> ScalerType:

    if use_scaler_target is None:

        if multilabel:
            return "none"

        return "none" if feature_engine == "none" else "robust"

    if feature_engine == "none" and (use_scaler_target is not None and use_scaler_target != "none"):
        raise ValueError(f"Scaling is not supported with feature_engine='none', received {use_scaler_target} and downgrading to 'none'")
    
    return use_scaler_target


# #########################################################################
#
#      Pandas Helpers
#
# #########################################################################


def safe_divide(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.divide(
        a, b, out=np.zeros_like(a), where=b != 0.0, casting="unsafe"
    )  # noqa


def features_without_target(
    df: pd.DataFrame, y: Optional[Union[List, str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """Checks if y DataFrame column name is in df, and removes it from df if so

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
    elif is_cudf_df(y):
        import cudf
        assert isinstance(y, cudf.DataFrame)
        yc = y.columns
        xc = df.columns
        for c in yc:
            if c in xc:
                remove_cols.append(c)
    elif isinstance(y, pd.Series):
        if y.name and (y.name in df.columns):
            remove_cols = [y.name]
    elif is_cudf_s(y):
        import cudf
        assert isinstance(y, cudf.Series)
        if y.name and (y.name in df.columns):
            remove_cols = [y.name]
    elif isinstance(y, List):
        remove_cols = y  # noqa
    elif isinstance(y, str):
        remove_cols = [y]
    else:
        logger.warning(
            "Target is not of type(DataFrame) and has no columns"
        )  # noqa
    if len(remove_cols):
        logger.debug(f"Removing {remove_cols} columns from DataFrame")
        tf = df.drop(columns=remove_cols, errors="ignore") # noqa
        return tf
    return df


def remove_node_column_from_symbolic(X_symbolic, node):
    if isinstance(X_symbolic, list):
        if node in X_symbolic:
            logger.info(f"Removing `{node}` from input X_symbolic list")
            X_symbolic.remove(node)
        return X_symbolic
    if isinstance(X_symbolic, pd.DataFrame) or is_cudf_df(X_symbolic):
        logger.info(f"Removing `{node}` from input X_symbolic DataFrame")
        return X_symbolic.drop(columns=[node], errors="ignore")


def remove_internal_namespace_if_present(df: pd.DataFrame):
    """
        Some tranformations below add columns to the DataFrame,
        this method removes them before featurization
        Will not drop if suffix is added during UMAP-ing

    :param df: DataFrame
    :return: DataFrame with dropped columns in reserved namespace
    """
    if df is None:
        return None
    # here we drop all _namespace like _x, _y, etc, so that
    # featurization doesn't include them idempotent-ly
    reserved_namespace : List[str] = [
        config.X,
        config.Y,
        config.SRC,
        config.DST,
        config.WEIGHT,
        config.IMPLICIT_NODE_ID,
        "index",  # in umap, we add as reindex
    ]

    if (len(df.columns) <= 2):
        df = df.rename(columns={c: c + '_1' for c in df.columns if c in reserved_namespace})
        # if (isinstance(df.columns.to_list()[0],int)):
        #     int_namespace = pd.to_numeric(df.columns, errors = 'ignore').dropna().to_list()  # type: ignore
        #     df = df.rename(columns={c: str(c) + '_1' for c in df.columns if c in int_namespace})
    else:
        df = df.drop(columns=reserved_namespace, errors="ignore")  # type: ignore
    return df


# ###########################################################################
#
#      Featurization Functions and Utils
#
# ###########################################################################

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
    # very useful on large DataFrames, super useful
    # if we use a feature_column type transformer too
    gtypes = df.columns.to_series().groupby(df.dtypes).groups
    gtypes = {k.name: list(v) for k, v in gtypes.items()}  # type: ignore
    if verbose:
        for k, v in gtypes.items():
            logger.debug(f"{k} has {len(v)} members")
    return gtypes


def set_to_numeric(df: pd.DataFrame, cols: List, fill_value: float = 0.0):
    df[cols] = pd.to_numeric(df[cols], errors="coerce").fillna(fill_value)  # type: ignore


def set_to_datetime(df: pd.DataFrame, cols: List, new_col: str):
    # eg df["Start_Date"] = pd.to_datetime(df[['Month', 'Day', 'Year']])
    df[new_col] = pd.to_datetime(df[cols], errors="coerce").fillna(0)


def set_to_bool(df: pd.DataFrame, col: str, value: Any):
    df[col] = np.where(df[col] == value, True, False)


def where_is_currency_column(df: pd.DataFrame, col: str):
    #  simple heuristics:
    def check_if_currency(x: str):
        if isinstance(x, str):
            if "$" in x:  # hmmm need to add for ALL currencies...
                return True
            if "," in x:  # and ints next to it
                return True
        try:
            float(x)
            return True
        except Exception as e:
            logger.warning(e)
            return False

    mask = df[col].apply(check_if_currency)
    return mask


def set_currency_to_float(
    df: pd.DataFrame, col: str, return_float: bool = True
):
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
    """Finds columns that if not coerced to strings, will break processors.
    
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


# ############################################################################
#
#      Text Utils
#
# ############################################################################


def check_if_textual_column(
    df: pd.DataFrame,
    col: str,
    confidence: float = 0.35,
    min_words: float = 2.5,
) -> bool:
    """Checks if `col` column of df is textual or not using basic heuristics

    :param df: DataFrame
    :param col: column name
    :param confidence: threshold float value between 0 and 1.
            If column `col` has `confidence` more elements as type `str`
            it will pass it onto next stage of evaluation. Default 0.35
    :param min_words: mean minimum words threshold.
            If mean words across `col` is greater than this,
            it is deemed textual.
            Default 2.5
    :return: bool, whether column is textual or not
    """

    if df[col].dtype != "object" and df[col].dtype != "string":
        return False
    
    if is_cudf_df(df):
        import cudf
        assert isinstance(df, cudf.DataFrame)
        df2_small = df[[col]].head(100).to_pandas()
        return check_if_textual_column(df2_small, col, confidence, min_words)

    isstring = df[col].apply(lambda x: isinstance(x, str))
    abundance = sum(isstring) / len(df)
    if min_words == 0:  # force textual encoding of named columns
        return True
    
    if abundance >= confidence:
        # now check how many words
        n_words = df[col].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        mean_n_words = n_words.mean()
        if mean_n_words >= min_words:
            logger.info(
                f"\n\tColumn `{col}` looks textual with mean number "
                f"of words {mean_n_words:.2f}"
            )
            return True
        else:
            return False
    else:
        return False


def get_textual_columns(
    df: pd.DataFrame, min_words: float = 2.5
) -> List:
    """Collects columns from df that it deems are textual.

    :param df: DataFrame
    :return: list of columns names
    """
    text_cols: List = []
    for col in df.columns:
        if check_if_textual_column(
            df, col, confidence=0.35, min_words=min_words
        ):
            text_cols.append(col)
    if len(text_cols) == 0:
        logger.debug("No Textual Columns were found")
    return text_cols


# ######################################################################
#
#      Featurization Utils
#
# ######################################################################


class Embedding:
    """
    Generates random embeddings of a given dimension 
    that aligns with the index of the dataframe
    
    """
    def __init__(self, df: pd.DataFrame):
        self.index = df.index

    def fit(self, n_dim: int):
        logger.info(f"-Creating Random Embedding of dimension {n_dim}")
        self.vectors = np.random.randn(len(self.index), n_dim)
        self.columns = [f"emb_{k}" for k in range(n_dim)]
        self.get_feature_names_out = callThrough(self.columns)

    def transform(self, ids) -> pd.DataFrame:
        mask = self.index.isin(ids)
        index = self.index[mask]  # type: ignore
        res = self.vectors[mask]
        res = pd.DataFrame(res, index=index, columns=self.columns)  # type: ignore
        return res  # type: ignore

    def fit_transform(self, n_dim: int):
        self.fit(n_dim)
        return self.transform(self.index)


def identity(x):
    return x


def get_preprocessing_pipeline(
    use_scaler: ScalerType = "robust",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 10,
    encode: str = "ordinal",
    strategy: str = "quantile",
) -> Pipeline:  # noqa
    """Helper function for imputing and scaling np.ndarray data using different scaling transformers.

    :param X: np.ndarray
    :param impute: whether to run imputing or not
    :param use_scaler: Selects scaling transformer
    :param n_quantiles: if use_scaler = 'quantile',
            sets the quantile bin size.
    :param output_distribution: if use_scaler = 'quantile',
            can return distribution as ["normal", "uniform"]
    :param quantile_range: if use_scaler = 'robust'/'quantile', 
            sets the quantile range.
    :param n_bins: number of bins to use in kbins discretizer
    :param encode: encoding for KBinsDiscretizer, can be one of
            `onehot`, `onehot-dense`, `ordinal`, default 'ordinal'
    :param strategy: strategy for KBinsDiscretizer, can be one of
            `uniform`, `quantile`, `kmeans`, default 'quantile'
    :return: scaled array, imputer instances or None, scaler instance or None
    """
    from sklearn.preprocessing import (
        FunctionTransformer,
        KBinsDiscretizer,
        MinMaxScaler,
        MultiLabelBinarizer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    available_preprocessors = [
        "minmax",
        "quantile",
        "standard",
        "robust",
        "kbins",
    ]
    available_quantile_distributions = ["normal", "uniform"]

    imputer = identity
    if impute:
        logger.debug("Imputing Values using “median” strategy")
        # impute values
        imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    scaler = identity

    if use_scaler == "minmax":
        # scale the resulting values column-wise between min and max
        # column values and sets them between 0 and 1
        scaler = MinMaxScaler()
    elif use_scaler == "quantile":
        assert (
            output_distribution in available_quantile_distributions
        ), logger.error(
            f"output_distribution must be in "
            f"{available_quantile_distributions}, got {output_distribution}"
        )
        scaler = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution=output_distribution
        )
    elif use_scaler == "standard":
        scaler = StandardScaler()
    elif use_scaler == "robust":
        scaler = RobustScaler(quantile_range=quantile_range)
    elif use_scaler == "kbins":
        scaler = KBinsDiscretizer(
            n_bins=n_bins, encode=encode, strategy=strategy
        )
    elif use_scaler == "none":
        pass
    else:
        raise ValueError(
            f"Invalid scaler type. Received {use_scaler}. Available types are {available_preprocessors + ['none']}"
        )
    logger.debug(f"Using {use_scaler} scaling")
    transformer = Pipeline(steps=[("imputer", imputer), ("scaler", scaler)])

    return transformer


def fit_pipeline(
    X: pd.DataFrame, transformer, keep_n_decimals: int = 5
) -> pd.DataFrame:
    """Helper to fit DataFrame over transformer pipeline. Rounds resulting matrix X by keep_n_digits if not 0, which helps for when transformer pipeline is scaling or imputer which sometime introduce small negative numbers, and umap metrics like Hellinger need to be positive
    :param X: DataFrame to transform.
    :param transformer: Pipeline object to fit and transform
    :param keep_n_decimals: Int of how many decimal places to keep in rounded transformed data
    """
    columns = X.columns
    index = X.index

    X = transformer.fit_transform(X)
    if keep_n_decimals:
        X = np.round(X, decimals=keep_n_decimals)  #  type: ignore  # noqa

    return pd.DataFrame(X, columns=columns, index=index)


def impute_and_scale_df(
    df: pd.DataFrame,
    use_scaler: ScalerType = "robust",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 10,
    encode: str = "ordinal",
    strategy: str = "uniform",
    keep_n_decimals: int = 5,
) -> Tuple[pd.DataFrame, Pipeline]:

    transformer = get_preprocessing_pipeline(
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

    return res, transformer


def get_text_preprocessor(ngram_range=(1, 3), max_df=0.2, min_df=3):
    from sklearn.feature_extraction.text import (
        CountVectorizer,
        TfidfTransformer,
    )
    from sklearn.pipeline import Pipeline
    cvect = CountVectorizer(
        ngram_range=ngram_range, max_df=max_df, min_df=min_df
    )
    return Pipeline(
        [
            ("vect", cvect),
            ("tfidf", TfidfTransformer()),
        ]
    )


def concat_text(df, text_cols):
    res = df[text_cols[0]].astype(str)
    if len(text_cols) > 1:
        logger.debug(
            f"Concactinating columns {text_cols} "
            "into one text-column for encoding"
        )
        for col in text_cols[1:]:
            res += " " + df[col].astype(str)
    return res


def _get_sentence_transformer_headers(emb, text_cols):
    return [f"{'_'.join(text_cols)}_{k}" for k in range(emb.shape[1])]


def encode_textual(
    df: pd.DataFrame,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    use_ngrams: bool = False,
    ngram_range: tuple = (1, 3),
    max_df: float = 0.2,
    min_df: int = 3,
) -> Tuple[pd.DataFrame, List, Any]:
    _, _, SentenceTransformer = lazy_sentence_transformers_import()

    t = time()
    text_cols = get_textual_columns(
        df, min_words=min_words
    )
    embeddings = (
        []
    )  # np.zeros((len(df), 1)) just a placeholder so we can use np.c_
    transformed_columns = []
    model = None
    if text_cols:
        res = concat_text(df, text_cols)
        if use_ngrams:
            model = get_text_preprocessor(ngram_range, max_df, min_df)
            logger.debug(
                f"-Calculating Tfidf Vectorizer with"
                f" {ngram_range}-ngrams for column(s) `{text_cols}`"
            )
            embeddings = make_array(model.fit_transform(res))
            transformed_columns = list(model[0].vocabulary_.keys())
        else:
            model_name = os.path.split(model_name)[-1]
            model = SentenceTransformer(f"{model_name}")
            batch_size = graphistry_config.get('encode_textual.batch_size')
            embeddings = model.encode(res.values, **({'batch_size': batch_size} if batch_size is not None else {}))
            transformed_columns = _get_sentence_transformer_headers(
                embeddings, text_cols
            )
        logger.info(
            f"Encoded Textual Data using {model} at "
            f"{len(df) / ((time() - t) / 60):.2f} rows per minute"
        )
    res = pd.DataFrame(embeddings,
                       columns=transformed_columns,
                       index=df.index)

    return res, text_cols, model


def smart_scaler(
    X_enc,
    y_enc,
    use_scaler: ScalerType,
    use_scaler_target: ScalerType,
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 10,
    encode: str = "ordinal",
    strategy: str = "uniform",
    keep_n_decimals: int = 5,
):
    pipeline = None
    pipeline_target = None

    def encoder(X, use_scaler: ScalerType):
        return impute_and_scale_df(
            X,
            use_scaler=use_scaler,
            impute=impute,
            n_quantiles=n_quantiles,
            quantile_range=quantile_range,
            output_distribution=output_distribution,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
        )

    if use_scaler != "none" and not X_enc.empty:
        logger.info(f"-Feature scaling using {use_scaler}")
        X_enc, pipeline = encoder(X_enc, use_scaler)

    if use_scaler_target != "none" and not y_enc.empty:
        logger.info(f"-Target scaling using {use_scaler_target}")
        y_enc, pipeline_target = encoder(y_enc, use_scaler_target)

    return X_enc, y_enc, pipeline, pipeline_target


def get_cardinality_ratio(df: pd.DataFrame):
    """Calculates the ratio of unique values to total number of rows of DataFrame
   
    :param df: DataFrame
    """
    ratios = {}
    for col in df.columns:
        ratio = df[col].nunique() / len(df)
        ratios[col] = ratio
    return ratios


def make_array(X):
    import scipy, scipy.sparse
    if scipy.sparse.issparse(X):
        logger.debug("Turning sparse array into dense")
        return X.toarray()
    return X


def passthrough_df_cols(
    df, columns
):  # if lambdas, won't pickle in FunctionTransformer
    return df[columns]


class callThrough:
    def __init__(self, x):
        self.x = x

    def __call__(self, *args, **kwargs):
        return self.x


def get_numeric_transformers(ndf, y=None):
    # numeric selector needs to embody memorization of columns
    # for later .transform consistency.
    from sklearn.preprocessing import FunctionTransformer
    label_encoder = False
    data_encoder = False
    y_ = y
    if y is not None:
        y_ = y.select_dtypes(include=[np.number])
        label_encoder = FunctionTransformer(
            partial(passthrough_df_cols, columns=y_.columns)
        )  # takes dataframe and memorizes which cols to use.
        label_encoder.get_feature_names_out = callThrough(y_.columns)
        label_encoder.columns_ = y_.columns

    if ndf is not None:
        ndf_ = ndf.select_dtypes(include=[np.number])
        data_encoder = FunctionTransformer(
            partial(passthrough_df_cols, columns=ndf_.columns)
        )
        data_encoder.get_feature_names_out = callThrough(ndf_.columns)
        #data_encoder.columns_ = ndf_.columns
        data_encoder.get_feature_names_in = callThrough(ndf_.columns)
        
    return ndf_, y_, data_encoder, label_encoder


def process_dirty_dataframes(
    ndf: pd.DataFrame,
    y: Optional[pd.DataFrame],
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    n_topics_target: int = config.N_TOPICS_TARGET_DEFAULT,
    similarity: Optional[str] = None,  # "ngram",
    categories: Optional[str] = "auto",
    multilabel: bool = False,
    feature_engine: FeatureEngineConcrete = "pandas",
) -> Tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    Union[SuperVectorizer, FunctionTransformer],
    Union[SuperVectorizer, FunctionTransformer],
]:
    """
        Dirty_Cat encoder for record level data. Will automatically turn
        inhomogeneous dataframe into matrix using smart conversion tricks.
   
    :param ndf: node DataFrame
    :param y: target DataFrame or series
    :param cardinality_threshold: For ndf columns, below this threshold,
            encoder is OneHot, above, it is GapEncoder
    :param cardinality_threshold_target: For target columns, below this
            threshold, encoder is OneHot, above, it is GapEncoder
    :param n_topics: number of topics for GapEncoder, default 42
    :param similarity: one of 'ngram', 'levenshtein-ratio', 'jaro',
            or'jaro-winkler'}) – The type of pairwise string similarity
            to use. If None or False, uses a SuperVectorizer
    :return: Encoded data matrix and target (if not None),
            the data encoder, and the label encoder.
    """
    has_dirty_cat, _, dirty_cat = lazy_dirty_cat_import()
    if has_dirty_cat:
        from dirty_cat import SuperVectorizer, GapEncoder, SimilarityEncoder
    from sklearn.preprocessing import FunctionTransformer
    t = time()

    all_numeric = is_dataframe_all_numeric(ndf)
    if not all_numeric and has_dirty_cat and (feature_engine in ["dirty_cat", "torch"]):
        data_encoder = SuperVectorizer(
            auto_cast=True,
            cardinality_threshold=cardinality_threshold,
            high_card_cat_transformer=GapEncoder(n_topics),
            #  numerical_transformer=StandardScaler(), This breaks
            #  since -- AttributeError: Transformer numeric
            #  (type StandardScaler)
            #  does not provide get_feature_names.
        )

        logger.info(":: Encoding DataFrame might take a few minutes ------")
        
        try:
            X_enc = data_encoder.fit_transform(ndf, y)
        except TypeError:
            nndf = ndf.copy()
            object_columns = nndf.select_dtypes(include=['object']).columns
            nndf[object_columns] = nndf[object_columns].astype(str)
            X_enc = data_encoder.fit_transform(nndf, y)
            logger.info("obj columns: %s are being converted to str", object_columns)
        X_enc = make_array(X_enc)

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            features_transformed = data_encoder.get_feature_names_out()

        all_transformers = data_encoder.transformers
        logger.debug(f"-Shape of [[dirty_cat fit]] data {X_enc.shape}")
        logger.debug(f"-Transformers: \n{all_transformers}\n")
        logger.debug(
            f"-Transformed Columns: \n{features_transformed[:20]}...\n"
        )
        logger.debug(
            f"--Fitting on Data took {(time() - t) / 60:.2f} minutes\n"
        )
        #  now just set the feature names, since dirty cat changes them in
        #  a weird way...
        data_encoder.get_feature_names_out = callThrough(features_transformed)
        
        X_enc = pd.DataFrame(
            X_enc, columns=features_transformed, index=ndf.index
        )
        X_enc = X_enc.fillna(0.0)
    elif not all_numeric and (not has_dirty_cat or feature_engine in ["pandas", "none"]):
        numeric_ndf = ndf.select_dtypes(include=[np.number])  # type: ignore
        logger.warning("-*-*- DataFrame is not numeric and no dirty_cat, dropping non-numeric")
        X_enc, _, data_encoder, _ = get_numeric_transformers(numeric_ndf, None)
    else:
        logger.debug("-*-*- DataFrame is completely numeric")
        X_enc, _, data_encoder, _ = get_numeric_transformers(ndf, None)


    if multilabel and y is not None:
        y_enc, label_encoder = encode_multi_target(y, mlb=None)
    elif (
        y is not None
        and len(y.columns) > 0  # noqa: E126,W503
        and not is_dataframe_all_numeric(y)  # noqa: E126,W503
        and has_dirty_cat  # noqa: E126,W503
    ):
        t2 = time()
        logger.debug("-Fitting Targets --\n%s", y.columns)

        label_encoder = SuperVectorizer(
            auto_cast=True,
            cardinality_threshold=cardinality_threshold_target,
            high_card_cat_transformer=GapEncoder(n_topics_target)
            if not similarity
            else SimilarityEncoder(
                similarity=similarity, categories=categories, n_prototypes=2
            ),  # Similarity
        )

        y_enc = label_encoder.fit_transform(y)
        y_enc = make_array(y_enc)

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            if isinstance(label_encoder, SuperVectorizer) or isinstance(
                label_encoder, FunctionTransformer
            ):
                labels_transformed = label_encoder.get_feature_names_out()
            else:  # Similarity Encoding uses categories_
                labels_transformed = label_encoder.categories_

        y_enc = pd.DataFrame(y_enc,
                             columns=labels_transformed,
                             index=y.index)
        # y_enc = y_enc.fillna(0)
        # add for later
        label_encoder.get_feature_names_out = callThrough(labels_transformed)

        logger.debug(f"-Shape of target {y_enc.shape}")
        # logger.debug(f"-Target Transformers used:
        # {label_encoder.transformers}\n")
        logger.debug(
            "--Fitting SuperVectorizer on TARGET took"
            f" {(time() - t2) / 60:.2f} minutes\n"
        )
    elif (
        y is not None
        and len(y.columns) > 0  # noqa: E126,W503
        and not is_dataframe_all_numeric(y)  # noqa: E126,W503
        and not has_dirty_cat  # noqa: E126,W503
    ):
        logger.warning("-*-*- y is not numeric and no dirty_cat, dropping non-numeric")
        y2 = y.select_dtypes(include=[np.number])  # type: ignore
        y_enc, _, _, label_encoder = get_numeric_transformers(y2, None)
    else:
        y_enc, _, label_encoder, _ = get_numeric_transformers(y, None)

    return (X_enc, y_enc, data_encoder, label_encoder)


def process_nodes_dataframes(
    df: pd.DataFrame,
    y: pd.DataFrame,
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    n_topics_target: int = config.N_TOPICS_TARGET_DEFAULT,
    use_scaler: ScalerType = "robust",
    use_scaler_target: ScalerType = "kbins",
    multilabel: bool = False,
    embedding: bool = False,  # whether to produce random embeddings
    use_ngrams: bool = False,
    ngram_range: tuple = (1, 3),
    max_df: float = 0.2,
    min_df: int = 3,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    similarity: Optional[str] = None,
    categories: Optional[str] = "auto",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 10,
    encode: str = "ordinal",
    strategy: str = "uniform",
    keep_n_decimals: int = 5,
    feature_engine: FeatureEngineConcrete = "pandas"
    # test_size: Optional[bool] = None,
) -> Tuple[
    pd.DataFrame,
    Any,
    pd.DataFrame,
    Any,
    SuperVectorizer,
    SuperVectorizer,
    Optional[Pipeline],
    Optional[Pipeline],
    Any,
    List[str],
]:
    """Automatic Deep Learning Embedding/ngrams of Textual Features, with the rest of the columns taken care of by dirty_cat

    :param df: pandas DataFrame of data
    :param y: pandas DataFrame of targets
    :param n_topics: number of topics in Gap Encoder
    :param n_topics_target: number of topics in Gap Encoder for target
    :param use_scaler: Scaling transformer
    :param use_scaler_target: Scaling transformer for target
    :param confidence: Number between 0 and 1, will pass
            column for textual processing if total entries are string
            like in a column and above this relative threshold.
    :param min_words: Sets the threshold
            for average number of words to include column for
            textual sentence encoding. Lower values means that
            columns will be labeled textual and sent to sentence-encoder. 
            Set to 0 to force named columns as textual.
    :param model_name: SentenceTransformer model name. See available list at
            https://www.sbert.net/docs/pretrained_models.
            html#sentence-embedding-models

    :return: X_enc, y_enc, data_encoder, label_encoder,
        scaling_pipeline,
        scaling_pipeline_target,
        text_model,
        text_cols,
    """
    logger.info("process_nodes_dataframes[%s]", feature_engine)

    if feature_engine in ["none", "pandas"]:
        X_enc, y_enc, data_encoder, label_encoder = get_numeric_transformers(
            df, y
        )
        X_encs, y_encs, scaling_pipeline, scaling_pipeline_target = smart_scaler(
            X_enc,
            y_enc,
            use_scaler,
            use_scaler_target,
            impute=impute,
            n_quantiles=n_quantiles,
            quantile_range=quantile_range,
            output_distribution=output_distribution,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
        )

        logger.debug(
            f"Feature Engine {feature_engine},"
            "returning only Numeric Data, if any"
        )
        return (
            X_enc,
            y_enc,
            X_encs,
            y_encs,
            data_encoder,
            label_encoder,
            scaling_pipeline,
            scaling_pipeline_target,
            False,
            [],
        )

    t = time()
    if len(df) == 0 or df.empty:
        logger.warning("DataFrame **seems** to be Empty")

    text_cols: List[str] = []
    text_model: Any = None
    text_enc = pd.DataFrame([])
    has_deps_text, import_text_exn, _ = lazy_sentence_transformers_import()
    if has_deps_text and (feature_engine in ["torch", "auto"]):
        text_enc, text_cols, text_model = encode_textual(
            df,
            min_words=min_words,
            model_name=model_name,
            use_ngrams=use_ngrams,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
        )
    else:
        logger.debug(
            "! Skipping encoding any textual features"
            f"since dependency {import_text_exn} is not met"
        )

    other_df = df.drop(columns=text_cols, errors="ignore")  # type: ignore

    X_enc, y_enc, data_encoder, label_encoder = process_dirty_dataframes(
        other_df,
        y,
        cardinality_threshold=cardinality_threshold,
        cardinality_threshold_target=cardinality_threshold_target,
        n_topics=n_topics,
        n_topics_target=n_topics_target,
        similarity=similarity,
        categories=categories,
        multilabel=multilabel,
        feature_engine=feature_engine
    )

    if embedding:
        data_encoder = Embedding(df)
        X_enc = data_encoder.fit_transform(n_dim=n_topics)

    if not text_enc.empty and not X_enc.empty:
        logger.info("-" * 60)
        logger.info("<= Found both a textual embedding + dirty_cat =>")
        X_enc = pd.concat(
            [text_enc, X_enc], axis=1
        )  # np.c_[embeddings, X_enc.values]
    elif not text_enc.empty and X_enc.empty:
        logger.info("-" * 60)
        logger.info("<= Found only textual embedding =>")
        X_enc = text_enc

    logger.debug(
        f"--The entire Encoding process took {(time()-t)/60:.2f} minutes"
    )

    X_encs, y_encs, scaling_pipeline, scaling_pipeline_target = smart_scaler(
        X_enc,
        y_enc,
        use_scaler,
        use_scaler_target,
        impute=impute,
        n_quantiles=n_quantiles,
        quantile_range=quantile_range,
        output_distribution=output_distribution,
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        keep_n_decimals=keep_n_decimals,
    )

    return (
        X_enc,
        y_enc,
        X_encs,
        y_encs,
        data_encoder,
        label_encoder,
        scaling_pipeline,
        scaling_pipeline_target,
        text_model,
        text_cols  # type: ignore
    )
class FastMLB:
    def __init__(self, mlb, in_column, out_columns):
        if isinstance(in_column, str):
            in_column = [in_column]
        self.columns = in_column  # should be singe entry list ['cats']
        self.mlb = mlb
        self.out_columns = out_columns
        self.feature_names_in_ = in_column
    
    def __call__(self, df):
        ydf = df[self.columns]
        return self.mlb.transform(ydf.squeeze())
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, df):
        return self(df)
    
    def get_feature_names_out(self):
        return self.out_columns
    
    def get_feature_names_in(self):
        return self.feature_names_in_
    
    def __repr__(self):
        doc = f'FastMultiLabelBinarizer(In: {self.columns},  Out: {self.out_columns})'
        return doc 


def encode_multi_target(ydf, mlb = None):
    from sklearn.preprocessing import (
        MultiLabelBinarizer,
    )
    ydf = ydf.squeeze()  # since its a dataframe, we want series
    assert isinstance(ydf, pd.Series), 'Target needs to be a single column of (list of lists)'
    column_name = ydf.name
    
    if mlb is None:
        mlb = MultiLabelBinarizer()
        T = mlb.fit_transform(ydf) 
    else:
        T = mlb.transform(ydf)

    T = 1.0 * T
    columns = [
        str(k) for k in mlb.classes_
    ]
    T = pd.DataFrame(T, columns=columns, index=ydf.index)
    logger.info(f"Shape of Target Encoding: {T.shape}")
        
    label_encoder = FastMLB(mlb=mlb, in_column=[column_name], out_columns=columns)  # memorizes which cols to use.
 
    return T, label_encoder

def encode_edges(edf, src, dst, mlb, fit=False):
    """edge encoder -- creates multilabelBinarizer on edge pairs.

    Args:
        edf (pd.DataFrame): edge dataframe
        src (string): source column
        dst (string): destination column
        mlb (sklearn): multilabelBinarizer
        fit (bool, optional): If true, fits multilabelBinarizer. Defaults to False.

    :Returns: tuple: pd.DataFrame, multilabelBinarizer
    """
    # uses mlb with fit=T/F so we can use it in transform mode
    # to recreate edge feature concat definition
    source = edf[src]
    destination = edf[dst]
    logger.debug("Encoding Edges using MultiLabelBinarizer")
    if fit:
        T = mlb.fit_transform(zip(source, destination))
    else:
        T = mlb.transform(zip(source, destination))
    T = 1.0 * T  # coerce to float
    columns = [
        str(k) for k in mlb.classes_
    ]  # stringify the column names or scikits.base throws error
    mlb.get_feature_names_out = callThrough(columns)
    mlb.columns_ = [src, dst]
    T = pd.DataFrame(T, columns=columns, index=edf.index)
    logger.info(f"Shape of Edge Encoding: {T.shape}")
    return T, mlb


def process_edge_dataframes(
    edf: pd.DataFrame,
    y: pd.DataFrame,
    src: str,
    dst: str,
    cardinality_threshold: int = 40,
    cardinality_threshold_target: int = 400,
    n_topics: int = config.N_TOPICS_DEFAULT,
    n_topics_target: int = config.N_TOPICS_TARGET_DEFAULT,
    use_scaler: Optional[ScalerType] = None,
    use_scaler_target: Optional[ScalerType] = None,
    multilabel: bool = False,
    use_ngrams: bool = False,
    ngram_range: tuple = (1, 3),
    max_df: float = 0.2,
    min_df: int = 3,
    min_words: float = 2.5,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    similarity: Optional[str] = None,
    categories: Optional[str] = "auto",
    impute: bool = True,
    n_quantiles: int = 10,
    output_distribution: str = "normal",
    quantile_range=(25, 75),
    n_bins: int = 10,
    encode: str = "ordinal",
    strategy: str = "uniform",
    keep_n_decimals: int = 5,
    feature_engine: FeatureEngineConcrete = "pandas",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[Any],
    Any,
    Optional[Pipeline],
    Optional[Pipeline],
    Any,
    List[str],
]:
    """
        Custom Edge-record encoder. Uses a MultiLabelBinarizer
        to generate a src/dst vector
        and then process_textual_or_other_dataframes that encodes any
        other data present in edf,
        textual or not.

    :param edf: pandas DataFrame of edge features
    :param y: pandas DataFrame of edge labels
    :param src: source column to select in edf
    :param dst: destination column to select in edf
    :param use_scaler: Scaling transformer
    :param use_scaler_target': Scaling transformer for target

    :return: Encoded data matrix and target (if not None), the data encoders, and the label encoder.
    """
    lazy_import_has_min_dependancy()
    from sklearn.preprocessing import (
        MultiLabelBinarizer,
    )
    logger.info("process_edges_dataframes[%s]", feature_engine)

    t = time()
    mlb_pairwise_edge_encoder = (
        MultiLabelBinarizer()
    )  # create new one so we can use encode_edges later in
    # transform with fit=False
    T, mlb_pairwise_edge_encoder = encode_edges(
        edf, src, dst, mlb_pairwise_edge_encoder, fit=True
    )
    other_df = edf.drop(columns=[src, dst])
    logger.debug(
        f"-Rest of DataFrame has columns: {other_df.columns}"
        " and is not empty"
        if not other_df.empty
        else f"-Rest of DataFrame has columns: {other_df.columns}"
             " and is empty"
    )

    use_scaler_resolved = resolve_scaler(use_scaler, feature_engine)
    use_scaler_target_resolved = resolve_scaler_target(use_scaler_target, feature_engine, multilabel)

    if feature_engine in ["none", "pandas"]:

        X_enc, y_enc, data_encoder, label_encoder = get_numeric_transformers(
            other_df, y
        )
        # add the two datasets together
        X_enc = pd.concat([T, X_enc], axis=1)
        # then scale them
        X_encs, y_encs, scaling_pipeline, scaling_pipeline_target = smart_scaler(
            X_enc,
            y_enc,
            use_scaler_resolved,
            use_scaler_target_resolved,
            impute=impute,
            n_quantiles=n_quantiles,
            quantile_range=quantile_range,
            output_distribution=output_distribution,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
        )

        logger.info("Returning only Edge MLB and/or numeric features")

        return (
            X_enc,
            y_enc,
            X_encs,
            y_encs,
            [mlb_pairwise_edge_encoder, data_encoder],
            label_encoder,
            scaling_pipeline,
            scaling_pipeline_target,
            False,
            [],
        )

    (
        X_enc,
        y_enc,
        _,
        _,
        data_encoder,
        label_encoder,
        _,
        _,
        text_model,
        text_cols,
    ) = process_nodes_dataframes(
        other_df,
        y,
        cardinality_threshold=cardinality_threshold,
        cardinality_threshold_target=cardinality_threshold_target,
        n_topics=n_topics,
        n_topics_target=n_topics_target,
        use_scaler="none",
        use_scaler_target="none",
        multilabel=multilabel,
        use_ngrams=use_ngrams,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        min_words=min_words,
        model_name=model_name,
        similarity=similarity,
        categories=categories,
        feature_engine=feature_engine,
    )

    if not X_enc.empty and not T.empty:
        logger.debug("-" * 60)
        logger.debug("<= Found Edges and Dirty_cat encoding =>")
        X_enc = pd.concat([T, X_enc], axis=1)
    elif not T.empty and X_enc.empty:
        logger.debug("-" * 60)
        logger.debug("<= Found only Edges =>")
        X_enc = T

    logger.info(
        "**The entire Edge encoding process took"
        f" {(time()-t)/60:.2f} minutes"
    )

    X_encs, y_encs, scaling_pipeline, scaling_pipeline_target = smart_scaler(
        X_enc,
        y_enc,
        use_scaler_resolved,
        use_scaler_target_resolved,
        impute=impute,
        n_quantiles=n_quantiles,
        quantile_range=quantile_range,
        output_distribution=output_distribution,
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        keep_n_decimals=keep_n_decimals,
    )

    res = (
        X_enc,
        y_enc,
        X_encs,
        y_encs,
        [mlb_pairwise_edge_encoder, data_encoder],
        label_encoder,
        scaling_pipeline,
        scaling_pipeline_target,
        text_model,
        text_cols,
    )
    return res


# ############################################################################
#
#      Vectorizer Class + Helpers
#
# ############################################################################


def transform_text(
    df: pd.DataFrame,
    text_model: Union[SentenceTransformer, Pipeline],  # type: ignore
    text_cols: Union[List, str],
) -> pd.DataFrame:
    from sklearn.pipeline import Pipeline
    _, _, SentenceTransformer = lazy_sentence_transformers_import()

    logger.debug("Transforming text using:")
    if isinstance(text_model, Pipeline):
        logger.debug(f"--Ngram tfidf {text_model}")
        tX = text_model.transform(df)
        tX = make_array(tX)
        tX = pd.DataFrame(
            tX,
            columns=list(text_model[0].vocabulary_.keys()),
            index=df.index
            )
    elif isinstance(text_model, SentenceTransformer):
        logger.debug(f"--HuggingFace Transformer {text_model}")
        tX = text_model.encode(df.values)
        tX = pd.DataFrame(
            tX,
            columns=_get_sentence_transformer_headers(tX, text_cols),
            index=df.index,
        )
    else:
        raise ValueError(
            "`text_model` should be instance of"
            "sklearn.pipeline.Pipeline or SentenceTransformer,"
            f"got {text_model}"
        )
    return tX


def transform_dirty(
    df: pd.DataFrame,
    data_encoder: Union[SuperVectorizer, FunctionTransformer],  # type: ignore
    name: str = "",
) -> pd.DataFrame:
    # from sklearn.preprocessing import MultiLabelBinarizer
    logger.debug(f"-{name} Encoder:")
    logger.debug(f"\t{data_encoder}\n")
    # print(f"-{name} Encoder:")
    # print(f"\t{data_encoder}\n")
    # try:
    #     logger.debug(f"{data_encoder.get_feature_names_in}")
    # except Exception as e:
    #     logger.warning(e)
    #     pass
    logger.debug(f"TRANSFORM pre as df -- \t{df.shape}")

    # #####################################  for dirty_cat 0.3.0
    use_columns = getattr(data_encoder, 'columns_', [])
    if len(use_columns):
        #print(f"Using columns: {use_columns}")
        X = data_encoder.transform(df[df.columns.intersection(use_columns)])
    # #####################################  with dirty_cat 0.2.0
    else:
        X = data_encoder.transform(df)
    # ###################################
    # X = data_encoder.transform(df)

    logger.debug(f"TRANSFORM DIRTY as Matrix -- \t{X.shape}")
    X = make_array(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        X = pd.DataFrame(
            X, columns=data_encoder.get_feature_names_out(), index=df.index
        )
        logger.debug(f"TRANSFORM DIRTY dataframe -- \t{X.shape}")

    return X


def transform(
    df: pd.DataFrame, ydf: pd.DataFrame, res: List, kind: str, src, dst
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # here res is the featurization result,
    # this function aligns with what is computed during
    # processing nodes or edges.
    (
        _,
        _,
        _,
        _,
        data_encoder,
        label_encoder,
        _,
        _,
        text_model,
        text_cols,
    ) = res

    logger.info("-" * 90)
    
    # index = df.index
    y = pd.DataFrame([])
    T = pd.DataFrame([])
    # encode nodes
    if kind == "nodes":
        logger.info("-Transforming Nodes--")
        X = transform_dirty(
            df, data_encoder, name="Numeric or Dirty Node-Features"
        )
    # encode edges
    elif kind == "edges":
        logger.info("-Transforming Edges--")
        mlb, data_encoder = data_encoder
        T, mlb = encode_edges(df, src, dst, mlb, fit=False)
        X = transform_dirty(df,
                            data_encoder,
                            "Numeric or Dirty Edge-Features")

    if ydf is not None:
        logger.info("-Transforming Target--")
        y = transform_dirty(ydf, label_encoder,
                            name=f"{kind.title()}-Label")

    # Concat in the Textual features, if any
    tX = pd.DataFrame([])
    if text_cols:
        logger.info(f"-textual columns found: {text_cols}")
        res_df = concat_text(df, text_cols)
        if text_model:
            tX = transform_text(res_df, text_model, text_cols)
        logger.info("** text features are empty") if tX.empty else None

    # concat text to dirty_cat, with text in front.
    if not tX.empty and not X.empty:
        X = pd.concat([tX, X], axis=1)
        logger.info("--Combining both Textual and Numeric/Dirty_Cat")
    elif not tX.empty and X.empty:
        X = tX  # textual
        logger.info("--Just textual")
    elif not X.empty:
        logger.info("--Just Numeric/Dirty_Cat transformer")
        X = X  # dirty/Numeric
    else:
        logger.info("-" * 60)
        logger.info(f"** IT'S ALL EMPTY NOTHINGNESS [[ {X} ]]")
        logger.info("-" * 60)

    # now if edges, add T at front
    if kind == "edges":
        X = pd.concat([T, X], axis=1)  # edges, text, dirty_cat
        logger.info("-Combining MultiLabelBinarizer with previous features")

    logger.info("-" * 40)
    logger.info(f"--Features matrix shape: {X.shape}")
    logger.info(f"--Target matrix shape: {y.shape}")

    # if scaling_pipeline and not X.empty:
    #     logger.info("--Scaling Features")
    #     X = pd.DataFrame(scaling_pipeline.transform(X), columns=X.columns, index=index)
    # if scaling_pipeline_target and not y.empty:
    #     logger.info(f"--Scaling Target {scaling_pipeline_target}")
    #     y = pd.DataFrame(
    #         scaling_pipeline_target.transform(y), columns=y.columns, index=index
    #     )

    return X, y


class FastEncoder:
    def __init__(self, df, y=None, kind="nodes"):
        self._df = df
        self.feature_names_in = df.columns  
        self._y = pd.DataFrame([], index=df.index) if y is None else y
        self.target_names_in = self._y.columns
        self.kind = kind
        self._assertions()
        # these are the parts we can use to reconstruct transform.
        self.res_names = ("X_enc y_enc data_encoder label_encoder "
                          "scaling_pipeline scaling_pipeline_target "
                          "text_model text_cols".split(" "))

    def _assertions(self):
        # add smart checks here
        if not self._y.empty:
            assert (
                self._df.shape[0] == self._y.shape[0]
            ), "Data and Targets must have same number of rows,"
            f"found {self._df.shape[0], self._y.shape[0]}, resp"

    def _encode(self, df, y, kind, src, dst, *args, **kwargs):
        if kind == "nodes":
            res = process_nodes_dataframes(df, y, *args, **kwargs)
        elif kind == "edges":
            res = process_edge_dataframes(
                df, y, src=src, dst=dst, *args, **kwargs
            )
        else:
            raise ValueError(
                f'`kind` should be one of "nodes" or "edges", found {kind}'
            )

        return res

    def _hecho(self, res):
        logger.info("\n-- Setting Encoder Parts from Fit ::")
        logger.info(f'Feature Columns In: {self.feature_names_in}')
        logger.info(f'Target Columns In: {self.target_names_in}')

        for name, value in zip(self.res_names, res):
            if name not in ["X_enc", "y_enc"]:
                logger.info(f"[[ {name} ]]:  {value}\n")

    def _set_result(self, res):
        self.res = list(res)
        [
            X_enc,
            y_enc,
            X_encs,
            y_encs,
            data_encoder,
            label_encoder,
            scaling_pipeline,
            scaling_pipeline_target,
            text_model,
            text_cols,
        ] = self.res

        self._hecho(res)
        # data_encoder.feature_names_in = self.feature_names_in
        # label_encoder.target_names_in = self.target_names_in
        self.feature_columns = X_enc.columns
        self.feature_columns_target = y_enc.columns
        self.X = X_encs
        self.y = y_encs
        self.X_orignal = X_enc
        self.y_orignal = y_enc
        self.data_encoder = data_encoder  # is list for edges
        self.label_encoder = label_encoder
        self.scaling_pipeline = scaling_pipeline
        self.scaling_pipeline_target = scaling_pipeline_target
        self.text_model = text_model
        self.text_cols = text_cols

    def fit(self, src=None, dst=None, *args, **kwargs):
        self.src = src
        self.dst = dst
        res = self._encode(
            self._df, self._y, self.kind, src, dst, *args, **kwargs
        )
        self._set_result(res)

    def transform(self, df, ydf=None):
        "Raw transform, no scaling."
        X, y = transform(df, ydf, self.res, self.kind, self.src, self.dst)
        return X, y
    
    def _transform_scaled(self, df, ydf, scaling_pipeline, scaling_pipeline_target):
        """Transform with scaling fit durning fit."""
        X, y = transform(df, ydf, self.res, self.kind, self.src, self.dst)
        if scaling_pipeline is not None and not X.empty:
            X = pd.DataFrame(scaling_pipeline.transform(X), columns=X.columns, index=X.index)
        if scaling_pipeline_target is not None and y is not None and not y.empty:
            y = pd.DataFrame(scaling_pipeline_target.transform(y), columns=y.columns, index=y.index)
        return X, y
    
    def transform_scaled(self, df, ydf=None, scaling_pipeline=None, scaling_pipeline_target=None):
        if scaling_pipeline is None:
            scaling_pipeline = self.scaling_pipeline
        if scaling_pipeline_target is None:
            scaling_pipeline_target = self.scaling_pipeline_target
        return self._transform_scaled(df, ydf, scaling_pipeline, scaling_pipeline_target)

    def fit_transform(self, src=None, dst=None, *args, **kwargs):
        self.fit(src=src, dst=dst, *args, **kwargs)
        return self.X, self.y

    def scale(self, X=None, y=None, return_pipeline=False, *args, **kwargs):
        """Fits new scaling functions on df, y via args-kwargs
        
        **Example:**
            ::

                from graphisty.features import SCALERS, SCALER_OPTIONS
                print(SCALERS)
                g = graphistry.nodes(df)
                # set a scaling strategy for features and targets -- umap uses those and produces different results depending.
                g2 = g.umap(use_scaler='standard', use_scaler_target=None)
                
                # later if you want to scale new data, you can do so
                X, y = g2.transform(df, df, scaled=False)  # unscaled transformer output
                # now scale with new settings
                X_scaled, y_scaled = g2.scale(X, y, use_scaler='minmax', use_scaler_target='kbins', n_bins=5)
                # fit some other pipeline
                clf.fit(X_scaled, y_scaled)
                
            args:
            ::

                ;X: pd.DataFrame of features
                :y: pd.DataFrame of target features
                :kind: str, one of 'nodes' or 'edges'
                *args, **kwargs: passed to smart_scaler pipeline

            returns:
                scaled X, y
        """
        logger.info("-Fitting new scaler on raw features")
        X, y, scaling_pipeline, scaling_pipeline_target = smart_scaler(
            X_enc=X, y_enc=y, *args, **kwargs
        )
        if return_pipeline:
            return X, y, scaling_pipeline, scaling_pipeline_target
        return X, y


# ###########################################################################
#
#      Fast Memoize
#
# ###########################################################################


def reuse_featurization(
    g: Plottable, memoize: bool, metadata: Any
):  # noqa: C901
    return check_set_memoize(
        g,
        metadata,
        attribute="_feat_param_to_g",
        name="featurize",
        memoize=memoize,
    )

def get_matrix_by_column_part(X: pd.DataFrame, column_part: str) -> pd.DataFrame:
    """Get the feature matrix by column part existing in column names."""
    transformed_columns = X.columns[X.columns.map(lambda x: True if column_part in x else False)]  # type: ignore
    return X[transformed_columns] 

def get_matrix_by_column_parts(X: pd.DataFrame, column_parts: Optional[Union[list, str]]) -> pd.DataFrame:
    """Get the feature matrix by column parts list existing in column names."""
    if column_parts is None:
        return X
    if isinstance(column_parts, str):
        column_parts = [column_parts]
    res = pd.concat([get_matrix_by_column_part(X, column_part) for column_part in column_parts], axis=1)  # type: ignore
    res = res.loc[:, ~res.columns.duplicated()]  # type: ignore
    return res


class FeatureMixin(MIXIN_BASE):
    """FeatureMixin for automatic featurization of nodes and edges DataFrames. Subclasses UMAPMixin for umap-ing of automatic features.

    Usage:
    ::

        g = graphistry.nodes(df, 'node_column')
        g2 = g.featurize()

    or for edges,
    ::

        g = graphistry.edges(df, 'src', 'dst')
        g2 = g.featurize(kind='edges')

    or chain them for both nodes and edges,
    ::

        g = graphistry.edges(edf, 'src', 'dst').nodes(ndf, 'node_column')
        g2 = g.featurize().featurize(kind='edges')

    """

    _feature_memoize: WeakValueDictionary = WeakValueDictionary()
    _feature_params: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def _get_feature(self, kind):
        kind2 = kind.replace('s', '')
        assert kind2 in ['node', 'edge'], f'kind needs to be in `nodes` or `edges`, found {kind}'
        x = getattr(self, f'_{kind2}_features')
        return x
    
    def _get_target(self, kind):
        kind2 = kind.replace('s', '')
        assert kind2 in ['node', 'edge'], f'kind needs to be in `nodes` or `edges`, found {kind}'
        x = getattr(self, f'_{kind2}_target')
        return x
    
    def _featurize_nodes(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: ScalerType = "none",
        use_scaler_target: ScalerType = "none",
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        n_topics_target: int = config.N_TOPICS_TARGET_DEFAULT,
        multilabel: bool = False,
        embedding: bool = False,
        use_ngrams: bool = False,
        ngram_range: tuple = (1, 3),
        max_df: float = 0.2,
        min_df: int = 3,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        similarity: Optional[str] = None,
        categories: Optional[str] = "auto",
        impute: bool = True,
        n_quantiles: int = 10,
        output_distribution: str = "normal",
        quantile_range=(25, 75),
        n_bins: int = 10,
        encode: str = "ordinal",
        strategy: str = "uniform",
        keep_n_decimals: int = 5,
        remove_node_column: bool = True,
        feature_engine: FeatureEngineConcrete = "pandas",
        memoize: bool = True,
        verbose: bool = False,
    ):
        res = self.copy() 
        ndf = res._nodes
        node = res._node

        if remove_node_column:
            ndf = remove_node_column_from_symbolic(ndf, node)
            X = remove_node_column_from_symbolic(X, node)

        if ndf is None:
            logger.info(
                "! Materializing Nodes and setting `embedding=True`"
                f"with latent dimension = n_topics: {n_topics}"
            )
            embedding = True
            res = res.materialize_nodes()
            ndf = res._nodes
            col = list(ndf.columns)[0]
            ndf = ndf.set_index(col)
            # in this case, X is not None, and is a DataFrame
            col = list(X.columns)[0]  # type: ignore
            X = X.set_index(col)  # type: ignore

        # resolve everything before setting dict so that
        # `X = ndf[cols]` and `X = cols` resolve to same thing
        X_resolved = resolve_X(ndf, X)
        y_resolved = resolve_y(ndf, y)
        
        from .features import ModelDict

        fkwargs = ModelDict("Featurize Params",
            X=X_resolved,
            y=y_resolved,
            use_scaler=use_scaler,
            use_scaler_target=use_scaler_target,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            n_topics_target=n_topics_target,
            multilabel=multilabel,
            embedding=embedding,
            use_ngrams=use_ngrams,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            min_words=min_words,
            model_name=model_name,
            similarity=similarity,
            categories=categories,
            impute=impute,
            n_quantiles=n_quantiles,
            quantile_range=quantile_range,
            output_distribution=output_distribution,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
            remove_node_column=remove_node_column,
            feature_engine=feature_engine,
        )

        res._feature_params = {
            **getattr(res, "_feature_params", {}),
            "nodes": fkwargs,
        }

        old_res = reuse_featurization(res, memoize, fkwargs)
        if old_res:
            logger.info("--- [[ RE-USING NODE FEATURIZATION ]]")
            fresh_res = copy.copy(res)
            for attr in ["_node_features", "_node_target", "_node_encoder"]:
                if hasattr(old_res, attr):
                    setattr(fresh_res, attr, getattr(old_res, attr))

            return fresh_res

        X_resolved = features_without_target(X_resolved, y_resolved)
        X_resolved = remove_internal_namespace_if_present(X_resolved)

        keys_to_remove = ["X", "y", "remove_node_column"]
        nfkwargs = dict()
        for key, value in fkwargs.items():
            if key not in keys_to_remove:
                nfkwargs[key] = value

        logger.debug("** Featurizing nodes")
        # ############################################################
        encoder = FastEncoder(X_resolved, y_resolved, kind="nodes")
        encoder.fit(**nfkwargs)
        # ###########################################################

        # if changing, also update fresh_res
        res._node_features = encoder.X
        res._node_features_raw = encoder.X_orignal  # .copy()
        res._node_target = encoder.y
        res._node_target_raw = encoder.y_orignal  # .copy()
        res._node_encoder = encoder  # now this does
        # all the work `._node_encoder.transform(df, y)` etc

        return res

    def _featurize_edges(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[ScalerType] = None,
        use_scaler_target: Optional[ScalerType] = None,
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        n_topics_target: int = config.N_TOPICS_TARGET_DEFAULT,
        use_ngrams: bool = False,
        ngram_range: tuple = (1, 3),
        max_df: float = 0.2,
        min_df: int = 3,
        min_words: float = 2.5,
        multilabel: bool = False,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        similarity: Optional[str] = "ngram",
        categories: Optional[str] = "auto",
        impute: bool = True,
        n_quantiles: int = 10,
        output_distribution: str = "normal",
        quantile_range=(25, 75),
        n_bins: int = 10,
        encode: str = "ordinal",
        strategy: str = "uniform",
        keep_n_decimals: int = 5,
        feature_engine: FeatureEngineConcrete = "pandas",
        memoize: bool = True,
        verbose: bool = False,
    ):

        res = self.copy()
        edf = res._edges
        X_resolved = resolve_X(edf, X)
        y_resolved = resolve_y(edf, y)

        if res._source not in X_resolved:
            logger.debug("adding g._source to edge features")
            X_resolved = X_resolved.assign(
                **{res._source: res._edges[res._source]}
            )
        if res._destination not in X_resolved:
            logger.debug("adding g._destination to edge features")
            X_resolved = X_resolved.assign(
                **{res._destination: res._edges[res._destination]}
            )

        # now that everything is set
        fkwargs = dict(
            X=X_resolved,
            y=y_resolved,
            use_scaler=use_scaler,
            use_scaler_target=use_scaler_target,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            n_topics_target=n_topics_target,
            use_ngrams=use_ngrams,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            min_words=min_words,
            model_name=model_name,
            similarity=similarity,
            categories=categories,
            multilabel=multilabel,
            impute=impute,
            n_quantiles=n_quantiles,
            quantile_range=quantile_range,
            output_distribution=output_distribution,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
            feature_engine=feature_engine,
        )

        res._feature_params = {
            **getattr(res, "_feature_params", {}),
            "edges": fkwargs,
        }

        old_res = reuse_featurization(res, memoize, fkwargs)
        if old_res:
            logger.info("--- [[ RE-USING EDGE FEATURIZATION ]]")
            fresh_res = copy.copy(res)
            for attr in ["_edge_encoder", "_edge_features", "_edge_target"]:
                setattr(fresh_res, attr, getattr(old_res, attr))

            return fresh_res

        X_resolved = features_without_target(X_resolved, y_resolved)

        keys_to_remove = [
            "X",
            "y",
        ]
        nfkwargs = {}
        for key, value in fkwargs.items():
            if key not in keys_to_remove:
                nfkwargs[key] = value

        print("** Featuring edges") if verbose else None
        ###############################################################
        encoder = FastEncoder(X_resolved, y_resolved, kind="edges")
        encoder.fit(src=res._source, dst=res._destination, **nfkwargs)
        ##############################################################

        # if editing, should also update fresh_res
        res._edge_features = encoder.X
        res._edge_features_raw = encoder.X_orignal  # .copy()
        res._edge_target = encoder.y
        res._edge_target_raw = encoder.y_orignal  # .copy()
        res._edge_encoder = encoder

        return res
    
    def _infer_edges(self, emb, X, y, df, eps='auto', n_neighbors=4, sample=None, infer_on_umap_embedding=False, 
                     verbose=False, merge_policy=False, **kwargs):
        res = self.bind()
        if merge_policy:
            # useful to cluster onto existing graph
            g = infer_graph(res, emb, X, y, df, infer_on_umap_embedding=infer_on_umap_embedding, 
                            n_neighbors=n_neighbors, eps=eps, sample=sample, verbose=verbose, **kwargs) 
        else:
            # useful to cluster onto self
            g = infer_self_graph(res, emb, X, y, df, infer_on_umap_embedding=infer_on_umap_embedding, 
                                 n_neighbors=n_neighbors, eps=eps, verbose=verbose, **kwargs)
        return g

    def _transform(self, encoder: str, df: pd.DataFrame, ydf: Optional[pd.DataFrame], scaled):
        if getattr(self, encoder) is not None:
            if scaled:
                return getattr(self, encoder).transform_scaled(df, ydf)
            return getattr(self, encoder).transform(df, ydf)
        else:
            logger.debug(
                "Fit on data (g.featurize(kind='..', ...))"
                "before being able to transform data"
            )

    def transform(self, df: pd.DataFrame, 
                  y: Optional[pd.DataFrame] = None, 
                  kind: str = 'nodes', 
                  min_dist: Union[str, float, int] = 'auto', 
                  n_neighbors: int = 7,
                  merge_policy: bool = False,
                  sample: Optional[int] = None, 
                  return_graph: bool = True,
                  scaled: bool = True,
                  verbose: bool = False):
        """Transform new data and append to existing graph, or return dataframes
        
            **args:**

                :df: pd.DataFrame, raw data to transform
                :ydf: pd.DataFrame, optional
                :kind: str  # one of `nodes`, `edges`
                :return_graph: bool, if True, will return a graph with inferred edges.
                :merge_policy: bool, if True, adds batch to existing graph nodes via nearest neighbors. If False, will infer edges only between nodes in the batch, default False
                :min_dist: float, if return_graph is True, will use this value in NN search, or 'auto' to infer a good value. min_dist represents the maximum distance between two samples for one to be considered as in the neighborhood of the other.
                :sample: int, if return_graph is True, will use sample edges of existing graph to fill out the new graph
                :n_neighbors: int, if return_graph is True, will use this value for n_neighbors in Nearest Neighbors search
                :scaled: bool, if True, will use scaled transformation of data set during featurization, default True
                :verbose: bool, if True, will print metadata about the graph construction, default False

            **Returns:**

                X, y: pd.DataFrame, transformed data if return_graph is False
                or a graphistry Plottable with inferred edges if return_graph is True
        """

        # This is temporary until cucat release 
        if 'cudf' in str(getmodule(df)):
            df = df.to_pandas()  # type: ignore
        if (y is not None) and ('cudf' in str(getmodule(y))):
            y = y.to_pandas()  # type: ignore

        if kind == "nodes":
            X, y_ = self._transform("_node_encoder", df, y, scaled=scaled)
        elif kind == "edges":
            X, y_ = self._transform("_edge_encoder", df, y, scaled=scaled)
        else:
            logger.debug("kind must be one of `nodes`,"
                         f"`edges`, found {kind}")
            
        if return_graph and kind not in ["edges"]:
            emb = None  # will not be able to infer graph from umap coordinates, 
            # but will be able to infer graph from features of existing edges
            g = self._infer_edges(emb, X, y_, df, eps=min_dist, sample=sample, n_neighbors=n_neighbors,
                                  infer_on_umap_embedding=False, merge_policy=merge_policy,
                                  verbose=verbose)
            return g
        return X, y_

    def scale(
        self,
        df: Optional[pd.DataFrame] = None,
        y: Optional[pd.DataFrame] = None,
        kind: str = "nodes",
        use_scaler: Optional[ScalerType] = None,
        use_scaler_target: Optional[ScalerType] = None,
        impute: bool = True,
        n_quantiles: int = 10,
        output_distribution: str = "normal",
        quantile_range=(25, 75),
        n_bins: int = 10,
        encode: str = "ordinal",
        strategy: str = "uniform",
        keep_n_decimals: int = 5,
        return_scalers: bool = False,
    ):
        """Scale data using the same scalers as used in the featurization step.
        
            **Example**
            ::

                g = graphistry.nodes(df)
                X, y = g.featurize().scale(kind='nodes', use_scaler='robust', use_scaler_target='kbins', n_bins=3)
                
                # or 
                g = graphistry.nodes(df)
                # set a scaling strategy for features and targets -- umap uses those and produces different results depending.
                g2 = g.umap(use_scaler='standard', use_scaler_target=None)
                
                # later if you want to scale new data, you can do so
                X, y = g2.transform(df, df, scale=False)
                X_scaled, y_scaled = g2.scale(X, y, use_scaler='minmax', use_scaler_target='kbins', n_bins=5)
                # fit some other pipeline
                clf.fit(X_scaled, y_scaled)

            **Args:**

                :df: pd.DataFrame, raw data to transform, if None, will use data from featurization fit
                :y: pd.DataFrame, optional target data
                :kind: str, one of `nodes`, `edges`
                :use_scaler: Scaling transformer
                :use_scaler_target: Scaling transformer on target
                :impute: bool, if True, will impute missing values
                :n_quantiles: int, number of quantiles to use for quantile scaler
                :output_distribution: str, one of `normal`, `uniform`, `lognormal` 
                :quantile_range: tuple, range of quantiles to use for quantile scaler
                :n_bins: int, number of bins to use for KBinsDiscretizer
                :encode: str, one of `ordinal`, `onehot`, `onehot-dense`, `binary`    
                :strategy: str, one of `uniform`, `quantile`, `kmeans`
                :keep_n_decimals: int, number of decimals to keep after scaling
                :return_scalers: bool, if True, will return the scalers used to scale the data

            **Returns:**

                (X, y) transformed data if return_graph is False or a graph with inferred edges if return_graph is True, or (X, y, scaler, scaler_target) if return_scalers is True
        """
                
        if df is None:  # use the original data
            X, y = (self._node_features_raw, self._node_target_raw) if kind == "nodes" else (self._edge_features_raw, self._edge_target_raw)  # type: ignore
        else:
            X, y = self.transform(df, y, kind=kind, return_graph=False, scaled=False)

        if kind == "nodes" and hasattr(self, "_node_encoder"):  # type: ignore
            if self._node_encoder is not None:  # type: ignore
                (
                    X,
                    y,
                    scaler,
                    scaler_target
                ) = self._node_encoder.scale(
                    X,
                    y,
                    use_scaler=use_scaler,
                    use_scaler_target=use_scaler_target,
                    impute=impute,
                    n_quantiles=n_quantiles,
                    quantile_range=quantile_range,
                    output_distribution=output_distribution,
                    n_bins=n_bins,
                    encode=encode,
                    strategy=strategy,
                    keep_n_decimals=keep_n_decimals,
                    return_pipeline=True
                )  # type: ignore
            else:
                raise AttributeError(
                    'Please run g.featurize(kind="nodes", *args, **kwargs) '
                    'first before scaling matrices and targets is possible.'
                )

        elif kind == "edges" and hasattr(self, "_edge_encoder"):
            # type: ignore
            if self._edge_encoder is not None:  # type: ignore
                (
                    X,
                    y,
                    scaler,
                    scaler_target
                ) = self._edge_encoder.scale(
                    X,
                    y,
                    use_scaler=use_scaler,
                    use_scaler_target=use_scaler_target,
                    impute=impute,
                    n_quantiles=n_quantiles,
                    quantile_range=quantile_range,
                    output_distribution=output_distribution,
                    n_bins=n_bins,
                    encode=encode,
                    strategy=strategy,
                    keep_n_decimals=keep_n_decimals,
                    return_pipeline=True
                )  # type: ignore
            else:
                raise AttributeError(
                    'Please run g.featurize(kind="edges", *args, **kwargs) '
                    'first before scaling matrices and targets is possible.'
                )
        if return_scalers:
            return X, y, scaler, scaler_target
        return X, y


    def featurize(
        self,
        kind: str = "nodes",
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[ScalerType] = None,
        use_scaler_target: Optional[ScalerType] = None,
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = 42,
        n_topics_target: int = 12,
        multilabel: bool = False,
        embedding: bool = False,
        use_ngrams: bool = False,
        ngram_range: tuple = (1, 3),
        max_df: float = 0.2,
        min_df: int = 3,
        min_words: float = 4.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        impute: bool = True,
        n_quantiles: int = 100,
        output_distribution: str = "normal",
        quantile_range = (25, 75),
        n_bins: int = 10,
        encode: str = "ordinal",
        strategy: str = "uniform",
        similarity: Optional[str] = None,  # turn this off in favor of Gap Encoder
        categories: Optional[str] = "auto",
        keep_n_decimals: int = 5,
        remove_node_column: bool = True,
        inplace: bool = False,
        feature_engine: FeatureEngine = "auto",
        dbscan: bool = False,
        min_dist: float = 0.5,  # DBSCAN eps
        min_samples: int = 1,  # DBSCAN min_samples
        memoize: bool = True,
        verbose: bool = False,
    ):
        r"""Featurize Nodes or Edges of the underlying nodes/edges DataFrames.
        
        :param kind: specify whether to featurize `nodes` or `edges`.
                Edge featurization includes a pairwise
                src-to-dst feature block using a MultiLabelBinarizer,
                with any other columns being treated the
                same way as with `nodes` featurization.
        :param X: Optional input, default None. If symbolic, evaluated
                against self data based on kind.
                If None, will featurize all columns of DataFrame
        :param y: Optional Target(s) columns or explicit DataFrame, default None
        :param use_scaler: selects which scaler (and automatically imputes
                missing values using mean strategy)
                to scale the data.
                Please see scikits-learn documentation
                https://scikit-learn.org/stable/modules/preprocessing.html
                Here 'standard' corresponds to 'StandardScaler' in scikits.
        :param use_scaler_target: selects which scaler to scale the target
        :param cardinality_threshold: dirty_cat threshold on cardinality of
                categorical labels across columns.
                If value is greater than threshold, will run GapEncoder
                (a topic model) on column.
                If below, will one-hot_encode. Default 40.
        :param cardinality_threshold_target: similar to cardinality_threshold,
                but for target features. Default is set high (400), as targets
                generally want to be one-hot encoded, but sometimes it can be
                useful to use
                GapEncoder (ie, set threshold lower) to create regressive
                targets, especially when those targets are
                textual/softly categorical and have semantic meaning across
                different labels.
                Eg, suppose a column has fields like
                ['Application Fraud', 'Other Statuses', 'Lost-Target scaling
                using/Stolen Fraud', 'Investigation Fraud', ...]
                the GapEncoder will concentrate the 'Fraud' labels together.
        :param n_topics: the number of topics to use in the GapEncoder if
                cardinality_thresholds is saturated.
                Default is 42, but good rule of thumb is to consult the
                Johnson-Lindenstrauss Lemma
                https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
                or use the simplified `random walk` estimate =>
                n_topics_lower_bound ~ (\pi/2) * (N-documents)**(1/4)
        :param n_topics_target: the number of topics to use in the GapEncoder
                if cardinality_thresholds_target is
                saturated for the target(s). Default 12.
        :param min_words: sets threshold on how many words to consider in a
                textual column if it is to be considered in
                the text processing pipeline. Set this very high if you want
                any textual columns to bypass the
                transformer, in favor of GapEncoder (topic modeling). Set to 
                0 to force all named columns to be encoded as textual (embedding)
        :param model_name: Sentence Transformer model to use. Default
                Paraphrase model makes useful vectors,
                but at cost of encoding time. If faster encoding is needed,
                `average_word_embeddings_komninos` is useful
                and produces less semantically relevant vectors.
                Please see sentence_transformer
                (https://www.sbert.net/) library for all available models.
        :param multilabel: if True, will encode a *single* target column composed of
                lists of lists as multilabel outputs. 
                This only works with y=['a_single_col'], default False
        :param embedding: If True, produces a random node embedding of size `n_topics`
                default, False. If no node features are provided, will produce random embeddings 
                (for GNN models, for example)
        :param use_ngrams: If True, will encode textual columns as TfIdf Vectors,
                default, False.
        :param ngram_range: if use_ngrams=True, can set ngram_range, eg: tuple = (1, 3)
        :param max_df:  if use_ngrams=True, set max word frequency to consider in vocabulary
                eg: max_df = 0.2,
        :param min_df:  if use_ngrams=True, set min word count to consider in vocabulary
                eg: min_df = 3 or 0.00001
        :param categories: Optional[str] in ["auto", "k-means", "most_frequent"], decides which 
                category to select in Similarity Encoding, default 'auto'
        :param impute: Whether to impute missing values, default True
        :param n_quantiles: if use_scaler = 'quantile',
                sets the quantile bin size.
        :param output_distribution: if use_scaler = 'quantile',
                can return distribution as ["normal", "uniform"]
        :param quantile_range: if use_scaler = 'robust'|'quantile', 
                sets the quantile range.
        :param n_bins: number of bins to use in kbins discretizer, default 10
        :param encode: encoding for KBinsDiscretizer, can be one of
                `onehot`, `onehot-dense`, `ordinal`, default 'ordinal'
        :param strategy: strategy for KBinsDiscretizer, can be one of
                `uniform`, `quantile`, `kmeans`, default 'quantile'
        :param n_quantiles: if use_scaler = "quantile", sets the number of quantiles, default=100
        :param output_distribution: if use_scaler="quantile"|"robust", 
                choose from ["normal", "uniform"]
        :param dbscan: whether to run DBSCAN, default False.
        :param min_dist: DBSCAN eps parameter, default 0.5.
        :param min_samples: DBSCAN min_samples parameter, default 5.
        :param keep_n_decimals: number of decimals to keep                
        :param remove_node_column: whether to remove node column so it is
                not featurized, default True.
        :param inplace: whether to not return new graphistry instance or
                not, default False.
        :param memoize: whether to store and reuse results across runs,
                default True.
        :return: graphistry instance with new attributes set by the featurization process.
        """
        assert_imported()
        if inplace:
            res = self
        else:
            res = self.bind()

        raw_feature_engine = feature_engine
        feature_engine = resolve_feature_engine(feature_engine)
        logger.debug("Resolved Feature Engine: %s => %s", raw_feature_engine, feature_engine)

        use_scaler_resolved = resolve_scaler(use_scaler, feature_engine)
        use_scaler_target_resolved = resolve_scaler_target(use_scaler_target, feature_engine, multilabel)
        
        if kind == "nodes":
            res = res._featurize_nodes(
                X=X,
                y=y,
                use_scaler=use_scaler_resolved,
                use_scaler_target=use_scaler_target_resolved,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                n_topics_target=n_topics_target,
                multilabel=multilabel,
                embedding=embedding,
                use_ngrams=use_ngrams,
                ngram_range=ngram_range,
                max_df=max_df,
                min_df=min_df,
                min_words=min_words,
                model_name=model_name,
                similarity=similarity,  
                categories=categories,
                impute=impute,
                n_quantiles=n_quantiles,
                quantile_range=quantile_range,
                output_distribution=output_distribution,
                n_bins=n_bins,
                encode=encode,
                strategy=strategy,
                keep_n_decimals=keep_n_decimals,
                remove_node_column=remove_node_column,
                feature_engine=feature_engine,
                memoize=memoize,
                verbose=verbose
            )
        elif kind == "edges":
            res = res._featurize_edges(
                X=X,
                y=y,
                use_scaler=use_scaler_resolved,
                use_scaler_target=use_scaler_target_resolved,
                cardinality_threshold=cardinality_threshold,
                cardinality_threshold_target=cardinality_threshold_target,
                n_topics=n_topics,
                n_topics_target=n_topics_target,
                multilabel=multilabel,
                use_ngrams=use_ngrams,
                ngram_range=ngram_range,
                max_df=max_df,
                min_df=min_df,
                min_words=min_words,
                model_name=model_name,
                similarity=similarity, 
                categories=categories,
                impute=impute,
                n_quantiles=n_quantiles,
                quantile_range=quantile_range,
                output_distribution=output_distribution,
                n_bins=n_bins,
                encode=encode,
                strategy=strategy,
                keep_n_decimals=keep_n_decimals,
                feature_engine=feature_engine,
                memoize=memoize,
                verbose=verbose
            )
        else:
            logger.warning(
                f"One may only featurize `nodes` or `edges`, got {kind}"
            )
            return self
        
        if dbscan:  # this adds columns to the dataframe, will break tests of pure featurization & umap, so set to False in those
            res = res.dbscan(min_dist=min_dist, min_samples=min_samples, kind=kind, fit_umap_embedding=False, verbose=verbose)  # type: ignore

        if not inplace:
            return res

    def featurize_or_get_nodes_dataframe_if_X_is_None(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[ScalerType] = None,
        use_scaler_target: Optional[ScalerType] = None,
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        n_topics_target: int = config.N_TOPICS_TARGET_DEFAULT,
        multilabel: bool = False,
        embedding: bool = False,
        use_ngrams: bool = False,
        ngram_range: tuple = (1, 3),
        max_df: float = 0.2,
        min_df: int = 3,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        similarity: Optional[
            str
        ] = None,  # turn this on to 'ngram' in favor of Similarity Encoder
        categories: Optional[str] = "auto",
        impute: bool = True,
        n_quantiles: int = 10,
        output_distribution: str = "normal",
        quantile_range=(25, 75),
        n_bins: int = 10,
        encode: str = "ordinal",
        strategy: str = "uniform",
        keep_n_decimals: int = 5,
        remove_node_column: bool = True,
        feature_engine: FeatureEngineConcrete = "pandas",
        reuse_if_existing=False,
        memoize: bool = True,
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, MIXIN_BASE]:
        """helper method gets node feature and target matrix if X, y are not specified. if X, y are specified will set them as `_node_target` and `_node_target` attributes
        """

        res = self.bind()

        if not reuse_if_existing:  # will cause re-featurization
            res._node_features = None
            res._node_target = None

        if reuse_if_existing and res._node_features is not None:
            logger.info('-Reusing Existing Node Featurization')
            return res._node_features, res._node_target, res

        use_scaler_resolved = resolve_scaler(use_scaler, feature_engine)
        use_scaler_target_resolved = resolve_scaler_target(use_scaler_target, feature_engine, multilabel)

        res = res._featurize_nodes(
            X=X,
            y=y,
            use_scaler=use_scaler_resolved,
            use_scaler_target=use_scaler_target_resolved,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            n_topics_target=n_topics_target,
            multilabel=multilabel,
            embedding=embedding,
            use_ngrams=use_ngrams,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            min_words=min_words,
            model_name=model_name,
            similarity=similarity,
            categories=categories,
            impute=impute,
            n_quantiles=n_quantiles,
            quantile_range=quantile_range,
            output_distribution=output_distribution,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
            remove_node_column=remove_node_column,
            feature_engine=feature_engine,
            memoize=memoize,
            verbose=verbose,
        )

        assert res._node_features is not None  # ensure no infinite loop

        return res.featurize_or_get_nodes_dataframe_if_X_is_None(
            res._node_features,
            res._node_target,
            use_scaler=use_scaler_resolved,
            use_scaler_target=use_scaler_target_resolved,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            n_topics_target=n_topics_target,
            multilabel=multilabel,
            embedding=embedding,
            use_ngrams=use_ngrams,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            min_words=min_words,
            model_name=model_name,
            similarity=similarity,
            categories=categories,
            impute=impute,
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            quantile_range=quantile_range,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
            remove_node_column=remove_node_column,
            feature_engine=feature_engine,
            reuse_if_existing=True,
            memoize=memoize,
        )  # now we are guaranteed to have node feature and target matrices.

    def featurize_or_get_edges_dataframe_if_X_is_None(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        use_scaler: Optional[ScalerType] = None,
        use_scaler_target: Optional[ScalerType] = None,
        cardinality_threshold: int = 40,
        cardinality_threshold_target: int = 400,
        n_topics: int = config.N_TOPICS_DEFAULT,
        n_topics_target: int = config.N_TOPICS_TARGET_DEFAULT,
        multilabel: bool = False,
        use_ngrams: bool = False,
        ngram_range: tuple = (1, 3),
        max_df: float = 0.2,
        min_df: int = 3,
        min_words: float = 2.5,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        similarity: Optional[
            str
        ] = None,  # turn this off in favor of Gap Encoder
        categories: Optional[str] = "auto",
        impute: bool = True,
        n_quantiles: int = 10,
        output_distribution: str = "normal",
        quantile_range=(25, 75),
        n_bins: int = 10,
        encode: str = "ordinal",
        strategy: str = "uniform",
        keep_n_decimals: int = 5,
        feature_engine: FeatureEngineConcrete = "pandas",
        reuse_if_existing=False,
        memoize: bool = True,
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], MIXIN_BASE]:
        """ helper method gets edge feature and target matrix if X, y are not specified
      
        :param X: Data Matrix
        :param y: target, default None
        :return: data `X` and `y`
        """

        res = self.bind()

        if not reuse_if_existing:
            res._edge_features = None
            res._edge_target = None

        if reuse_if_existing and res._edge_features is not None:
            logger.info('-Reusing Existing Edge Featurization')
            return res._edge_features, res._edge_target, res

        use_scaler_resolved = resolve_scaler(use_scaler, feature_engine)
        use_scaler_target_resolved = resolve_scaler_target(use_scaler_target, feature_engine, multilabel)

        res = res._featurize_edges(
            X=X,
            y=y,
            use_scaler=use_scaler_resolved,
            use_scaler_target=use_scaler_target_resolved,
            cardinality_threshold=cardinality_threshold,
            cardinality_threshold_target=cardinality_threshold_target,
            n_topics=n_topics,
            n_topics_target=n_topics_target,
            multilabel=multilabel,
            use_ngrams=use_ngrams,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            min_words=min_words,
            model_name=model_name,
            similarity=similarity,
            categories=categories,
            impute=impute,
            n_quantiles=n_quantiles,
            quantile_range=quantile_range,
            output_distribution=output_distribution,
            n_bins=n_bins,
            encode=encode,
            strategy=strategy,
            keep_n_decimals=keep_n_decimals,
            feature_engine=feature_engine,
            memoize=memoize,
            verbose=verbose,
        )

        assert res._edge_features is not None  # ensure no infinite loop

        return res.featurize_or_get_edges_dataframe_if_X_is_None(
            res._edge_features,
            res._edge_target,
            use_scaler,
            use_scaler_target,
            cardinality_threshold,
            cardinality_threshold_target,
            n_topics,
            n_topics_target,
            multilabel,
            use_ngrams,
            ngram_range,
            max_df,
            min_df,
            min_words,
            model_name,
            similarity,
            categories,
            impute,
            n_quantiles,
            output_distribution,
            quantile_range,
            n_bins,
            encode,
            strategy,
            keep_n_decimals,
            feature_engine,
            reuse_if_existing=True,
            memoize=memoize,
        )

    
    def get_matrix(self, columns: Optional[Union[List, str]] = None, kind: str = 'nodes', target: bool = False) -> pd.DataFrame:
        """Returns feature matrix, and if columns are specified, returns matrix with only the columns that contain the string `column_part` in their name.`X = g.get_matrix(['feature1', 'feature2'])` will retrieve a feature matrix with only the columns that contain the string `feature1` or `feature2` in their name. Most useful for topic modeling, where the column names are of the form `topic_0: descriptor`, `topic_1: descriptor`, etc. Can retrieve unique columns in original dataframe, or actual topic features like [ip_part, shoes, preference_x, etc]. Powerful way to retrieve features from a featurized graph by column or (top) features of interest.
            
            **Example:**
            ::
            
                # get the full feature matrices
                X = g.get_matrix()
                y = g.get_matrix(target=True)
            
                # get subset of features, or topics, given topic model encoding
                X = g2.get_matrix(['172', 'percent'])
                X.columns
                    => ['ip_172.56.104.67', 'ip_172.58.129.252', 'item_percent']
                # or in targets
                y = g2.get_matrix(['total', 'percent'], target=True)
                y.columns
                    => ['basket_price_total', 'conversion_percent', 'CTR_percent', 'CVR_percent']

                # not as useful for sbert features. 

            Caveats:
                - if you have a column name that is a substring of another column name, you may get unexpected results.

        Args:
            :columns (Union[List, str]): list of column names or a single column name that may exist in columns of the feature matrix. If None, returns original feature matrix
            :kind (str, optional): Node or Edge features. Defaults to 'nodes'.
            :target (bool, optional): If True, returns the target matrix. Defaults to False.

        Returns:
            pd.DataFrame: feature matrix with only the columns that contain the string `column_part` in their name.
        """

        if target:
            X = self._get_target(kind)
        else:
            X = self._get_feature(kind)

        return get_matrix_by_column_parts(X, columns)

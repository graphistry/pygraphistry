from typing_extensions import Literal
from .util import setup_logger
from .util import ModelDict

logger = setup_logger(__name__)

# ###############################################################
UNK = "UNK"
LENGTH_PRINT = 80
# ################ Encoded Global Models #################
EMBEDDING_MODEL_PATH = "embedding.model"
TOPIC_MODEL_PATH = "topic.model"
NGRAMS_MODEL_PATH = "ngrams.model"
SEARCH_MODEL_PATH = "search.model"

# ################ Actual Models #################
#  add specific instances of models here


# ###############################################################################
# ################# graphistry featurization config constants #################
N_TOPICS = 42
N_TOPICS_TARGET = 10
HIGH_CARD = 1e9  # forces one hot encoding
MID_CARD = 1e3  # todo: force hashing
LOW_CARD = 2

CARD_THRESH = 40
CARD_THRESH_TARGET = 400

FORCE_EMBEDDING_ALL_COLUMNS = 0  # min_words
HIGH_WORD_COUNT = 1024
MID_WORD_COUNT = 128
LOW_WORD_COUNT = 3

NGRAMS_RANGE = (1, 3)
MAX_DF = 0.2
MIN_DF = 3

KBINS_SCALER = "kbins"
STANDARD = 'standard'
ROBUST = 'robust'
MINMAX = 'minmax'
QUANTILE = 'quantile'
# for Optuna
ERROR = "error"

SCALERS = [STANDARD, ROBUST, MINMAX, KBINS_SCALER, QUANTILE]
ScalerType = Literal["none", "kbins", "standard", "robust", "minmax", "quantile"]
NO_SCALER = None
# Scaler options
N_BINS = 10
IMPUTE = "median"  # set to
N_QUANTILES = 100
OUTPUT_DISTRIBUTION = "normal"
QUANTILES_RANGE = (5, 95)
ENCODE = "ordinal"  # kbins, onehot, ordinal, label
STRATEGY = "uniform"  # uniform, quantile, kmeans
SIMILARITY = None  # 'ngram' , default None uses Gap
CATEGORIES = "auto"
SCALER_OPTIONS = {'impute': ['median', None], 'n_quantiles': [10,100], 'output_distribution': ['normal', 'uniform'],
                  'quantile_range': QUANTILES_RANGE, 
                  'encode': ['kbins', 'onehot', 'ordinal', 'label'], 
                  'strategy': ['uniform', 'quantile', 'kmeans'],
                  'similarity':[None, 'ngram'], 'categories': CATEGORIES, 'n_bins': [2, 100], 
                  'use_scaler': SCALERS, 'use_scaler_target': SCALERS
}

# precision in decimal places
KEEP_N_DECIMALS = 5  # TODO: check to see if this takes a lot of time
BATCH_SIZE_SMALL = 32
BATCH_SIZE = 1000
EXTRA_COLS_NEEDED = ["x", "y", "_n"]
# ###############################################################
# ################# graphistry umap config constants #################
N_COMPONENTS = 2
N_NEIGHBORS = 20
MIN_DIST = 0.1
SPREAD = 1
LOCAL_CONNECTIVITY = 1
REPULSION_STRENGTH = 2
NEGATIVE_SAMPLING_RATE = 5
METRIC = "euclidean"

UMAP_OPTIONS = {'n_components': [2, 10], 'n_neighbors': [2, 30], 'min_dist': [0.01, 0.99], 'spread': [0.5, 5], 'local_connectivity': [1, 30],
                'repulsion_strength': [1, 10], 'negative_sampling_rate': [5, 20], 
                'metric': ['euclidean', 'cosine', 'manhattan', 'l1', 'l2', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
}

# ###############################################################
# ################# enrichments
NMF_PATH = "nmf"
TIME_TOPIC = "time_topic"
TRANSLATED = "translated"
TRANSLATIONS = "translations"
SENTIMENT = "sentiment"

# ###############################################################
# ############ The Search key
SEARCH = "search"
# ############ Embeddings keys
TOPIC = "topic"  # topic model embeddings
EMBEDDING = "embedding"  # multilingual embeddings
QA = "qa"
NGRAMS = "ngrams"
# ############ Embedding Models
PARAPHRASE_SMALL_MODEL = "sentence-transformers/paraphrase-albert-small-v2"
PARAPHRASE_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MSMARCO2 = "sentence-transformers/msmarco-distilbert-base-v2"  # 768
MSMARCO3 = "sentence-transformers/msmarco-distilbert-base-v3"  # 512
QA_SMALL_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
LLM_SMALL = "sentence-transformers/llm-en-dim128"
LLM_LARGE = "sentence-transformers/llm-en-dim512"

EMBEDDING_MODELS = [PARAPHRASE_SMALL_MODEL, PARAPHRASE_MULTILINGUAL_MODEL, MSMARCO2, MSMARCO3, QA_SMALL_MODEL, LLM_SMALL, LLM_LARGE]
# #############################################################################
# Model Training Constants
# Used for seeding random state
RANDOM_STATE = 42
SPLIT_LOW = 0.1
SPLIT_MEDIUM = 0.2
SPLIT_HIGH = 0.5

# #############################################################################
# model training options

FEATURE_OPTIONS = {
    'kind': ['nodes', 'edges'],
    'cardinality_threshold': [1, HIGH_CARD],
    'cardinality_threshold_target': [1, HIGH_CARD],
    'n_topics': [4, 100],
    'n_topics_target': [4, 100],
    'multilabel': [True, False],
    'embedding': [True, False],
    'use_ngrams': [True, False],
    'ngram_range': (1, 5),
    'max_df': [0.1, 0.9],
    'min_df': [1, 10],
    'min_words': [0, 100],
    'model_name': [MSMARCO2, MSMARCO3, PARAPHRASE_SMALL_MODEL, PARAPHRASE_MULTILINGUAL_MODEL, QA_SMALL_MODEL],
}


# #############################################################################
# Model Training {params}

default_featurize_parameters = ModelDict(
    "Featurize Parameters",
    kind="nodes",
    use_scaler=NO_SCALER,
    use_scaler_target=NO_SCALER,
    cardinality_threshold=CARD_THRESH,
    cardinality_threshold_target=CARD_THRESH_TARGET,
    n_topics=N_TOPICS,
    n_topics_target=N_TOPICS_TARGET,
    multilabel=False,
    embedding=False,
    use_ngrams=False,
    ngram_range=NGRAMS_RANGE,
    max_df=MAX_DF,
    min_df=MIN_DF,
    min_words=LOW_WORD_COUNT,
    model_name=MSMARCO2,
    impute=IMPUTE,
    n_quantiles=N_QUANTILES,
    output_distribution=OUTPUT_DISTRIBUTION,
    quantile_range=QUANTILES_RANGE,
    n_bins=N_BINS,
    encode=ENCODE,  # kbins, onehot, ordinal, label
    strategy=STRATEGY,  # uniform, quantile, kmeans
    similarity=SIMILARITY,  # 'ngram'
    categories=CATEGORIES,
    keep_n_decimals=KEEP_N_DECIMALS,
    remove_node_column=True,
    inplace=False,
    feature_engine="auto",
    memoize=True,
)


default_umap_parameters = ModelDict("Umap Parameters",
        {"n_components": N_COMPONENTS,
        **({"metric": METRIC} if True else {}),
        "n_neighbors": N_NEIGHBORS,
        "min_dist": MIN_DIST,
        "spread": SPREAD,
        "local_connectivity": LOCAL_CONNECTIVITY,
        "repulsion_strength": REPULSION_STRENGTH,
        "negative_sample_rate": NEGATIVE_SAMPLING_RATE,
    }
)


umap_hellinger = ModelDict("Umap Parameters Hellinger",    
        {"n_components": N_COMPONENTS,
        "metric": "hellinger",  # info metric, can't use on
        # textual encodings since they contain negative values...
        "n_neighbors": 15,
        "min_dist": 0.3,
        "spread": 0.5,
        "local_connectivity": 1,
        "repulsion_strength": 1,
        "negative_sample_rate": 5
        }
)

umap_euclidean = ModelDict("Umap Parameters Euclidean",
        {"n_components": N_COMPONENTS,
        "metric": "euclidean",
        "n_neighbors": 12,
        "min_dist": 0.1,
        "spread": 0.5,
        "local_connectivity": 1,
        "repulsion_strength": 1,
        "negative_sample_rate": 5
        }
)

# #############################################################################
# Create useful presets for the user
# makes naming and encoding models consistently and testing different models against eachother easy
# customize the default parameters for each model you want to test

# Ngrams Model over features
ngrams_model = ModelDict(
    "Ngrams Model", use_ngrams=True, min_words=HIGH_CARD, verbose=True
)

# Topic Model over features
topic_model = ModelDict(
    "Reliable Topic Models on Features and Target",
    cardinality_threshold=LOW_CARD,  # force topic model
    cardinality_threshold_target=LOW_CARD,  # force topic model
    n_topics=N_TOPICS,
    n_topics_target=N_TOPICS_TARGET,
    min_words=HIGH_CARD,  # make sure it doesn't turn into sentence model, but rather topic models
    verbose=True,
)

# useful for text data that you want to paraphrase
embedding_model = ModelDict(
    f"{PARAPHRASE_SMALL_MODEL} sbert Embedding Model",
    min_words=FORCE_EMBEDDING_ALL_COLUMNS,
    model_name=PARAPHRASE_SMALL_MODEL,  # if we need multilingual support, use PARAPHRASE_MULTILINGUAL_MODEL
    verbose=True,
)

# useful for when search input is much smaller than the encoded documents
search_model = ModelDict(
    f"{MSMARCO2} Search Model",
    verbose=True,
    min_words=FORCE_EMBEDDING_ALL_COLUMNS,
    model_name=MSMARCO2,
)

# Question Answering encodings for search
qa_model = ModelDict(
    f"{QA_SMALL_MODEL} QA Model",
    min_words=FORCE_EMBEDDING_ALL_COLUMNS,
    model_name=QA_SMALL_MODEL,
    verbose=True,
)


BASE_MODELS = {
    EMBEDDING: embedding_model,
    SEARCH: search_model,
    QA: qa_model,
    TOPIC: topic_model,
    NGRAMS: ngrams_model,
}


if __name__ == "__main__":
    """python3 -m graphistry.features -m 'my awesome edge encoded model' -p '{"kind":"edges"}'"""
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default=SEARCH, help="description of your model"
    )
    parser.add_argument("-v", "--verbose", type=bool, default=True)
    parser.add_argument("-p", "--model_params", type=str)
    args = parser.parse_args()

    params = json.loads(args.model_params)
    print("----------- params -----------")
    print(params)
    model = ModelDict(args.model, verbose=args.verbose, **default_featurize_parameters)
    model.update(params)
    print(model)

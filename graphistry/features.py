from collections import UserDict
from .util import setup_logger
from .constants import VERBOSE, TRACE

logger = setup_logger("graphistry.features", verbose=VERBOSE, fullpath=TRACE)

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
HIGH_CARD = 4e7  # forces one hot encoding
MID_CARD = 2e3  # todo: forces hashing
LOW_CARD = 2

CARD_THRESH = 40
CARD_THRESH_TARGET = 400

FORCE_EMBEDDING_ALL_COLUMNS = 0  # min_words
HIGH_WORD_COUNT = 1024
LOW_WORD_COUNT = 3

NGRAMS_RANGE = (1, 3)
MAX_DF = 0.2
MIN_DF = 3

N_BINS = 10
KBINS_SCALER = "kbins"
IMPUTE = "median"  # set to
N_QUANTILES = 100
OUTPUT_DISTRIBUTION = "normal"
QUANTILES_RANGE = (25, 75)
N_BINS = 10
ENCODE = "ordinal"  # kbins, onehot, ordinal, label
STRATEGY = "uniform"  # uniform, quantile, kmeans
SIMILARITY = None  # 'ngram' , default None uses Gap
CATEGORIES = "auto"
KEEP_N_DECIMALS = 5

BATCH_SIZE = 1000
NO_SCALER = None
EXTRA_COLS_NEEDED = ["x", "y", "_n"]
# ###############################################################

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
PARAPHRASE_MULTILINGUAL_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
MSMARCO2 = "sentence-transformers/msmarco-distilbert-base-v2"  # 768
MSMARCO3 = "sentence-transformers/msmarco-distilbert-base-v3"  # 512
QA_SMALL_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
# #############################################################################
# Model Training Constants
# Used for seeding random state
RANDOM_STATE = 42
SPLIT_LOW = 0.1
SPLIT_MEDIUM = 0.2
SPLIT_HIGH = 0.5

# #############################################################################
class ModelDict(UserDict):
    """Helper class to print out model names

    Args:
        message: description of model
        verbose: print out model names, logging happens regardless
    """

    def __init__(self, message, verbose=True, *args, **kwargs):
        self._message = message
        self._verbose = verbose
        self._print_length = min(LENGTH_PRINT, len(message))
        self._updates = []
        super().__init__(*args, **kwargs)

    def __repr__(self):
        logger.info(self._message)
        if self._verbose:
            print("_" * self._print_length)
            print()
            print(self._message)
            print("_" * self._print_length)
            print()
        return super().__repr__()

    def update(self, *args, **kwargs):
        self._updates.append(args[0])
        if len(self._updates) > 1:  # don't take first update since its the init/default
            self._message += (
                "\n" + "_" * self._print_length + f"\n\nUpdated: {self._updates[-1]}"
            )
        return super().update(*args, **kwargs)


default_featurize_parameters = dict(
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


# #############################################################################
# Create useful presets for the user
# makes naming and encoding models consistently and testing different models against eachother easy
# customize the default parameters for each model you want to test

# Ngrams Model over features
ngrams_model = ModelDict("Ngrams Model", verbose=True, **default_featurize_parameters)
ngrams_model.update(dict(use_ngrams=True, min_words=HIGH_CARD))

# Topic Model over features
topic_model = ModelDict("Topic Model", verbose=True, **default_featurize_parameters)
topic_model.update(
    dict(
        cardinality_threshold=LOW_CARD,  # force topic model
        cardinality_threshold_target=LOW_CARD,  # force topic model
        n_topics=N_TOPICS,
        n_topics_target=N_TOPICS_TARGET,
        min_words=HIGH_CARD,  # make sure it doesn't turn into sentence model, but rather topic models
    )
)

# useful for text data that you want to paraphrase
embedding_model = ModelDict(
    f"{PARAPHRASE_SMALL_MODEL} Embedding Model",
    verbose=True,
    **default_featurize_parameters,
)
embedding_model.update(
    dict(
        min_words=FORCE_EMBEDDING_ALL_COLUMNS,
        model_name=PARAPHRASE_SMALL_MODEL,  # if we need multilingual support, use PARAPHRASE_MULTILINGUAL_MODEL
    )
)

# useful for when search input is much smaller than the encoded documents
search_model = ModelDict(
    f"{MSMARCO2} Search Model", verbose=True, **default_featurize_parameters
)
search_model.update(
    dict(
        min_words=FORCE_EMBEDDING_ALL_COLUMNS,
        model_name=MSMARCO2,
    )
)

# Question Answering encodings for search
qa_model = ModelDict(
    f"{QA_SMALL_MODEL} QA Model", verbose=True, **default_featurize_parameters
)
qa_model.update(
    dict(
        min_words=FORCE_EMBEDDING_ALL_COLUMNS,
        model_name=QA_SMALL_MODEL,
    )
)


BASE_MODELS = {
    EMBEDDING: embedding_model,
    SEARCH: search_model,
    QA: qa_model,
    TOPIC: topic_model,
    NGRAMS: ngrams_model,
}


if __name__ == "__main__":
    # python3 -m graphistry.features -m 'my awesome edge encoded model' -p '{"kind":"edges"}'
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

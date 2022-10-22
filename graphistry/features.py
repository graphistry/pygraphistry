from collections import UserDict
from .util import setup_logger
from .constants import VERBOSE, TRACE

logger = setup_logger('graphistry.features', verbose=VERBOSE, fullpath=TRACE)


UNK = 'UNK'
LENGTH_PRINT = 80
################# Encoded Global Models #################
EMBEDDING_MODEL_PATH = 'embedding.model'
TOPIC_MODEL_PATH = 'topic.model'
NGRAMS_MODEL_PATH = 'ngrams.model'

################# Actual Models #################


################################################################################
################## graphistry featurization config constants #################
N_TOPICS = 42
N_TOPICS_TARGET = 10
HIGH_CARD = 4e6  # forces one hot encoding
LOW_CARD = 2

FORCE_EMBEDDING_ALL_COLUMNS = 0 # min_words
HIGH_WORD_COUNT = 1024
LOW_WORD_COUNT = 2

N_BINS = 10 
KBINS_SCALER = "kbins"

BATCH_SIZE = 1000
NO_SCALER = None
EXTRA_COLS_NEEDED = ['x', 'y', '_n']
################################################################

################################################################
################## enrichments
NMF_PATH = "nmf"
TIME_TOPIC = "time_topic"
TRANSLATED = "translated"
TRANSLATIONS = "translations"
SENTIMENT = "sentiment"
############# The Search Instance /key
SEARCH = "search"
############# Embeddings
TOPIC = "topic"  # topic model embeddings
SEMANTIC = "semantic"  # multilingual embeddings
QA = "qa"
NGRAMS = "ngrams"
############# Embedding Models:
PARAPHRASE_SMALL_MODEL = 'sentence-transformers/paraphrase-albert-small-v2'
PARAPHRASE_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MSMARCO = "sentence-transformers/msmarco-distilbert-base-v3" # 512
QA_SMALL_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
# #############################################################################
# Model Training Constants
# Used for seeding random state
RANDOM_STATE = 42
SPLIT = 0.1


class ModelDict(UserDict):
    def __init__(self, message, verbose=True, *args, **kwargs):
        self._message = message
        self._verbose = verbose
        self._print_length = min(LENGTH_PRINT, len(message))
        super().__init__(*args, **kwargs)
        
    def __repr__(self):
        logger.info(self._message)
        if self._verbose:
            print('_'*self._print_length)
            print()
            print(self._message)
            print('_'*self._print_length)
            print()
            #print('-'*50)
        return super().__repr__()         


default_parameters = dict(kind='nodes', 
                            use_scaler=None,
                            use_scaler_target=None,
                            cardinality_threshold=40,
                            cardinality_threshold_target=400,
                            n_topics=42,
                            n_topics_target=10,
                            multilabel=False,
                            embedding=False,
                            use_ngrams=False,
                            ngram_range=(1, 3),
                            max_df=0.2,
                            min_df=3,
                            min_words=3,
                            model_name='paraphrase-MiniLM-L6-v2',
                            impute='median',
                            n_quantiles=100,
                            output_distribution='normal',
                            quantile_range=(25, 75),
                            n_bins=10,
                            encode='ordinal',
                            strategy='uniform',
                            similarity=None,
                            categories='auto',
                            keep_n_decimals=5,
                            remove_node_column=True,
                            inplace=False,
                            feature_engine='auto',
                            memoize=True
    )


ngrams_model = ModelDict('Ngrams Model', verbose=True)
ngrams_model.update(default_parameters)
ngrams_model.update(dict(
    use_ngrams=True,
    min_words=LOW_WORD_COUNT, 
                )
            )


topic_model = ModelDict('Topic Model', verbose=True)
topic_model.update(default_parameters)
topic_model.update(dict(
    cardinality_threshold=LOW_CARD,  # force topic model
    cardinality_threshold_target=LOW_CARD,  # force topic model
    n_topics=N_TOPICS,
    n_topics_target=N_TOPICS_TARGET,
    min_words=HIGH_CARD,  # make sure it doesn't turn into sentence model, but rather topic models
))


embedding_model = ModelDict(f'{PARAPHRASE_SMALL_MODEL} Embedding Model', verbose=True)
embedding_model.update(default_parameters)
# text -- paraphrase BERT style model
embedding_model.update(dict(
    min_words=FORCE_EMBEDDING_ALL_COLUMNS,  # make sentence transformer using paraphrase model #default in graphistry
    model_name=PARAPHRASE_SMALL_MODEL,  # if we need multilingual support, use PARAPHRASE_MULTILINGUAL_MODEL
))

search_model = ModelDict(f'{MSMARCO} Search Model', verbose=True)
search_model.update(default_parameters)
search_model.update(dict(
    min_words=FORCE_EMBEDDING_ALL_COLUMNS,  # make sentence transformer using paraphrase model #default in graphistry
    model_name=MSMARCO,  # if we need multilingual support, use PARAPHRASE_MULTILINGUAL_MODEL
    
))


qa_model = ModelDict(f'{QA_SMALL_MODEL} QA Model', verbose=True)
qa_model.update(default_parameters)
qa_model.update(dict(
    min_words=FORCE_EMBEDDING_ALL_COLUMNS,  # make sentence transformer
    model_name=QA_SMALL_MODEL,
))


BASE_MODELS = {
    SEMANTIC: embedding_model,
    SEARCH: search_model,
    TOPIC: topic_model,
    QA: qa_model,
    NGRAMS: ngrams_model,
}
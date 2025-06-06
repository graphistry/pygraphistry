# ###############################################################
VERBOSE = None  # set to true for info, false for debug, None for none
TRACE = False  # set to true for full trace of functions
# ###############################################################
# source and destination labels for consistent pipeline-ing across files
SRC = "_src_implicit"
DST = "_dst_implicit"
NODE = '_n_implicit'  # Is this being use anymore??
WEIGHT = "_weight"
BATCH = "_batch"
# for UMAP reserved namespace
X = "x"
Y = "y"
IMPLICIT_NODE_ID = (
    "_n"  # for g.featurize(..).umap(..) -> g.weighted_edges_from_nodes_df
)
# for text search db column
DISTANCE = '_distance'
# Scalers
SCALERS = ['quantile', 'standard', 'kbins', 'robust', 'minmax']

# dbscan reserved namespace
DBSCAN = '_dbscan'
DBSCAN_PARAMS = '_dbscan_params'

# ###############################################################
# consistent clf pipelining and constructor methods across files
DGL_GRAPH = "DGL_graph"  # TODO: change to _dgl_graph ? 
KG_GRAPH = '_kg_graph'
FEATURE = "feature"
TARGET = "target"
LABEL = "label"
LABEL_NODES = "node_label"
LABEL_EDGES = "edge_label"

# ENGINES
CUML = 'cuml'
UMAP_LEARN = 'umap_learn'

TRAIN_MASK = "train_mask"
TEST_MASK = "test_mask"


# ##############################################################
# for preprocessors namespace
SKRUB = 'skrub'
N_TOPICS_DEFAULT = 42
N_TOPICS_TARGET_DEFAULT = 7
N_HASHERS_DEFAULT = 100

# scikit-learn params
SKLEARN = "sklearn"

# #############################################################
# Caching and other internals
CACHE_COERCION_SIZE = 100

# #############################################################
# Annoy defaults
N_TREES = 10

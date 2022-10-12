# ###############################################################
VERBOSE = None  # set to true for info, false for debug, None for none
TRACE = False  # set to true for full trace of functions
# ###############################################################
# source and destination labels for consistent pipeline-ing across files
SRC = "_src_implicit"
DST = "_dst_implicit"
NODE = '_n_implicit'  # Is this being use anymore??
WEIGHT = "_weight"
# for UMAP reserved namespace
X = "x"
Y = "y"
IMPLICIT_NODE_ID = (
    "_n"  # for g.featurize(..).umap(..) -> g.weighted_edges_from_nodes_df
)
DISTANCE = '_distance'  # for text search db column
# ###############################################################
# consistent clf pipelining and constructor methods across files
DGL_GRAPH = "DGL_graph"
FEATURE = "feature"
TARGET = "target"
LABEL = "label"
LABEL_NODES = "node_label"
LABEL_EDGES = "edge_label"

TRAIN_MASK = "train_mask"
TEST_MASK = "test_mask"


# ##############################################################
# for preprocessors namespace
#   for dirty_cat params
DIRTY_CAT = "dirty_cat"
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

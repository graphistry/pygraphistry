# ###############################################################
# source and destination labels for consistent pipeline-ing across files
SRC = "_src"
DST = "_dst"
WEIGHT = "weight"
# for UMAP reserved namespace
X = "x"
Y = "y"

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
N_HASHERS_DEFAULT = 100

# scikit-learn params
SKLEARN = "sklearn"

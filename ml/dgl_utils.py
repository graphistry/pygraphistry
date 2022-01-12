# classes for converting a dataframe or Graphistry Plottable into a DGL
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import numpy as np

from ml.utils import pandas2dgl
from ml.utils import process_dirty_dataframes
from ml import constants as config

import logging

logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


# ndata_fields = ['feature', 'label', 'train_mask', 'test_mask', 'validation_mask']
# edata_fields = ['feature', 'relationship', 'train_mask', 'test_mask', 'validation_mask']


def get_vectorizer(name):
    if type(name) != str:
        logger.info(f"Returning {type(name)} vectorizer")
        return name
    if name == config.DIRTY_CAT:
        return process_dirty_dataframes
    if name == config.SKLEARN:
        # TODO
        return
    logging.warning(
        f"Vectorizer name must be one of [{config.DIRTY_CAT}, {config.SKLEARN}, ..]"
    )
    return


def get_dataframe_for_target(target, df, name):
    if type(target) == str:
        logger.info(f"Returning {name}-target {target} as DataFrame")
        return pd.DataFrame({target: df[target].values}, index=df.index)
    logger.info(f"Returning {name}-target as itself")
    return target


def check_target_not_in_features(df, y, remove=True):
    if y is None:
        return df, y
    remove_cols = []
    if hasattr(y, "columns"):
        yc = y.columns
        xc = df.columns
        for c in yc:
            if c in xc:
                remove_cols.append(c)
    else:
        logger.warning(f"Target is not of type(DataFrame) and has no columns")
    if remove:
        logger.info(f"Removing {remove_cols} columns from DataFrame")
        tf = df.drop(columns=remove_cols)
        return tf, y
    return df, y


def featurization(df, y, vectorizer, remove=True):
    df, y = check_target_not_in_features(df, y, remove=remove)
    X_enc, y_enc, sup_vec, sup_label = vectorizer(df, y)
    if y_enc is not None:
        y_enc = torch.tensor(y_enc.values)
        data = {config.FEATURE: torch.tensor(X_enc.values), config.TARGET: y_enc}
    else:
        data = {config.FEATURE: torch.tensor(X_enc.values)}
    encoders = {config.FEATURE: sup_vec, config.TARGET: sup_label}
    return data, encoders


def scatterplot(ux, color_labels=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.scatter(ux.T[0], ux.T[1], c=color_labels, s=100, alpha=0.4)


class BaseDGLGraphFromPandas:
    def __init__(
        self,
        ndf,
        edf,
        src,
        dst,
        node_column,
        node_target=None,
        edge_target=None,
        weight_col=None,
        use_node_label_as_feature=False,
        vectorizer=process_dirty_dataframes,
        device="cpu",
    ):
        """
        :param ndf: node DataFrame
        :param edf: edge DataFrame
        :param src: source column of edge DataFrame
        :param dst: destination column of edge DataFrame
        :param node_target: target node DataFrame (price, cluster, etc) or column name, of same length as nodes DataFrame
                If node_target is a string, it will find the target column in the node DataFrame ndf
                the target will be automatically removed before featurization
        :param edge_target: target node DataFrame (price, cluster, etc) or column name, of same length as nodes DataFrame
                If edge_target is a string, it will find the target column in the edge DataFrame ndf
                the target will be automatically removed before featurization
        :param node_column: node column of node DataFrame
        :param weight: weight column of edge DataFrame, default None
        :param use_node_label_as_feature:
        :param vectorizer: node level vectorizer, requires (df, target_df or None), eg: lambda x, y: f(x, y)
        :param device: Whether to put on cpu or gpu. Can always envoke .to(gpu) on returned DGL graph later, Default 'cpu'
        """
        self.edf = edf
        self.ndf = ndf
        self.src = src
        self.dst = dst
        self.node_target = get_dataframe_for_target(node_target, self.ndf, "node")
        self.edge_target = get_dataframe_for_target(edge_target, self.edf, "edge")
        self.node = node_column
        self.weight_col = weight_col
        self.use_node_label_as_feature = use_node_label_as_feature
        self.vectorizer = get_vectorizer(vectorizer)
        self._umapper = umap.UMAP()
        self.device = device

    def _remove_edges_not_in_nodes(self):
        # need to do this so we get the correct ndata size ...
        nodes = self.ndf[self.node]
        edf = self.edf
        logger.info(f"Length of edge DataFrame {len(edf)}")
        mask = edf[self.src].isin(nodes) & edf[self.dst].isin(nodes)
        self.edf = edf[mask]
        if self.edge_target is not None:
            edge_target = self.edge_target
            self.edge_target = edge_target[mask]
        logger.info(f"Length of edge DataFrame {len(self.edf)} after pruning")

    def _convert_edgeDF_to_DGL(self):
        logger.info("converting edge DataFrame to DGL")
        # recall that the adjacency is the graph itself, and really shouldn't be a node style feature (redundant),
        # but certainly can be
        self._remove_edges_not_in_nodes()
        self.graph, self.adjacency, self.entity_to_index = pandas2dgl(
            self.edf, self.src, self.dst, weight_col=self.weight_col, device=self.device
        )
        self.index_to_entity = {k: v for v, k in self.entity_to_index.items()}
        # this is a sanity check after _remove_edges_not_in_nodes
        self._check_nodes_lineup_with_edges()

    def _check_nodes_lineup_with_edges(self):
        nodes = self.ndf[self.node]
        unique_nodes = nodes.unique()
        logger.info(
            f"{len(nodes)} entities from column {self.node}\n with {len(unique_nodes)} unique entities"
        )
        if len(nodes) != len(unique_nodes):  # why would this be so?
            logger.warning(
                f"Nodes DataFrame has duplicate entries for column {self.node}"
            )
            # logger.warning(f'** Dropping duplicates in {self.node} and reassigning to self.ndf')
            # self.ndf = self.ndf.drop_duplicates(subset=[self.node])  # this might do damage to the representation
        # now check that self.entity_to_index is in 1-1 to with self.ndf[self.node]
        nodes = self.ndf[self.node]
        res = nodes.isin(self.entity_to_index)
        if res.sum() != len(nodes):
            logger.warning(
                f"Some Edges connect to Nodes not explicitly mentioned in nodes DataFrame (ndf)"
            )
        if len(self.entity_to_index) > len(nodes):
            logger.warning(
                f"There are more entities in edges DataFrame (edf) than in nodes DataFrame (ndf)"
            )

    def _node_featurization(self):
        logger.info("Running Node Featurization")
        ndf = self.ndf
        if not self.use_node_label_as_feature:
            logger.info(
                f"Dropping {self.node} from DataFrame so not to include it as an explicit feature, "
                f"which would just add np.eye({len(ndf), len(ndf)}) to the feature matrix under sklearn,"
                f"which might be useful for factorization machines"
                f"or under dirty_cat would mix rows with similar names, which seems odd"
            )
            ndf = ndf.drop(columns=[self.node])
        ndata, node_encoders = featurization(ndf, self.node_target, self.vectorizer)
        self.ndata = ndata
        self.node_encoders = node_encoders

    def _edge_featurization(self):
        logger.info("Running Edge Featurization")
        edata, edge_encoders = featurization(
            self.edf, self.edge_target, self.vectorizer
        )
        self.edata = edata
        self.edge_encoders = edge_encoders

    def build_simple_graph(self):
        self._convert_edgeDF_to_DGL()

    def embeddings(self):
        # here we make node and edge features and add them to the DGL graph instance self.graph
        if not hasattr(self, "graph"):
            self._convert_edgeDF_to_DGL()
        if not hasattr(self, "ndata"):
            self._node_featurization()
            # first add the ndata
            self.graph.ndata.update(self.ndata)
        if not hasattr(self, "edata"):
            self._edge_featurization()
            # then add any edata
            self.graph.edata.update(self.edata)

    def umap(self, scale=False, show=False, color_labels=None):
        if hasattr(self, "graph"):
            # y_node = None
            # if config.TARGET in self.graph.ndata:
            #     y_node = np.array(self.graph.ndata[config.TARGET])
            if config.FEATURE in self.graph.ndata:
                # take it out of a torch Tensor by wrapping it as ndarray
                X_node = np.array(self.graph.ndata[config.FEATURE])
                if scale:
                    logger.info(f"Zscaling matrix")
                    X_node = (X_node - X_node.mean(0)) / X_node.std(0)
                logger.info(f"Fitting Umap over matrix of size {X_node.shape}")
                # res = self._umapper.fit_transform(X_node, y_node)
                res = self._umapper.fit_transform(X_node)
                self._embedding = res
                if show:
                    scatterplot(res, color_labels=color_labels)
        else:
            logger.info(f"There are no Node Level embeddings to fit Umap over")

        # TODO add Edge Embedding


# ToDo
# class DGLGraphFromGraphistry(BaseDGLGraphFromPandas):
#
#     def __init__(self, g):
#         self.g = g
#         self._convert_graphistry_to_df()
#
#     def _convert_graphistry_to_df(self):
#         g = self.g
#         ndf = g._nodes
#         edf = g._edges
#         src = g._src

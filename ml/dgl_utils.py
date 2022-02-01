# classes for converting a dataframe or Graphistry Plottable into a DGL
from typing import List, Dict, Callable, Union, Any


import dgl
import torch
import numpy as np
import pandas as pd
from dgl.data import DGLDataset

from graphistry import Plottable

from ml import constants as config
from ml.feature_utils import convert_to_torch, process_textual_or_other_dataframes, process_edge_dataframes
#from ml.umap_utils import BaseUMAPMixin, umap_kwargs_euclidean
from ml.feature_utils import FeatureMixin
from ml.utils import pandas_to_sparse_adjacency, setup_logger

logger = setup_logger(__name__)


dgl_kwargs = dict(
    name="Graphistry", reverse=True, raw_dir=None, force_reload=False, verbose=True
)


# def get_vectorizer(name: Union[str, Any]):
#     if type(name) != str:
#         logger.info(f"Returning {type(name)} vectorizer")
#         return name
#     if name == config.DIRTY_CAT:
#         return process_dirty_dataframes
#     if name == config.SKLEARN:
#         # TODO
#         return
#     logger.warning(
#         f"Vectorizer name must be one of [{config.DIRTY_CAT}, {config.SKLEARN}, ..]"
#     )
#     return


# # FIXME this is not correct or robust enough
# def get_dataframe_for_target(target, df, name):
#     """
#         Might be doing too much -- general idea is to be able to pass in a target as column and select it from df, while returning a new dataframe with target as column
#         Likewise, if target is already a series, then return a dataframe with column 'name'
#         if target is already a dataframe, send it through without modification
#
#         usage examples:
#             -- target_dataframe = get_dataframe_for_target('some_col', df, '..')
#             -- target_dataframe = get_dataframe_for_target(pd.target_df, df, '..')
#             -- target_dataframe = get_dataframe_for_target(pd.Series, df, column_name)
#
#     :param target: str, series, or dataframe
#     :param df: aux dataframe to use if target is a string
#     :param name: useful for when target is a series, it will become the column name of the returned dataframe
#     :return: dataframe
#     """
#     if type(target) == str:
#         # get the target from the dataframe
#         logger.info(f"Returning {name}-target {target} as DataFrame")
#         return pd.DataFrame({target: df[target].values}, index=df.index)
#     if type(target) == pd.core.series.Series:
#         # use `name` as column header
#         logger.info(f"Returning target as DataFrame with column {name}")
#         return pd.DataFrame({name: target}, index=target.index)
#     if type(target) == pd.core.frame.DataFrame:
#         logger.info(f"Returning {name}-target as itself")
#         return target
#     logger.warning(f"Returning `None` for target")
#     return


def pandas_to_dgl_graph(df, src, dst, weight_col=None, device="cpu"):
    """Turns an edge DataFrame with named src and dst nodes, to DGL graph
    :eg
        g, sp_mat, ordered_nodes_dict = pandas_to_sparse_adjacency(df, 'to_node', 'from_node')
    :returns
        g: dgl graph
        sp_mat: sparse scipy matrix
        ordered_nodes_dict: dict ordered from most common src and dst nodes
    """
    sp_mat, ordered_nodes_dict = pandas_to_sparse_adjacency(df, src, dst, weight_col)

    g = dgl.from_scipy(sp_mat, device=device)  # there are other ways too
    logger.info(f"Graph Type: {type(g)}")  # why is this making a heterograph?

    return g, sp_mat, ordered_nodes_dict


def get_torch_train_test_mask(n: int, ratio: float = 0.8):
    """
        Generates random torch tensor mask
    :param n:
    :param ratio:
    :return:
    """
    train_mask = torch.zeros(n, dtype=torch.bool).bernoulli(ratio)
    test_mask = ~train_mask
    return train_mask, test_mask


class BaseDGLGraphMixin(FeatureMixin):
    def __init__(
        self,
        train_split: float = 0.8,
        device: str = "cpu",
        *args, **kwargs,
    ):
        """
        :param train_split: split percent between train and test, set in mask on dgl edata/ndata
        :param device: Whether to put on cpu or gpu. Can always envoke .to(gpu) on returned DGL graph later, Default 'cpu'
        """
        
        self.train_split = train_split
        self.device = device
        
        FeatureMixin.__init__(self, *args, **kwargs)

    def _prune_edge_target(self):
        if hasattr(self, 'edge_target'):
            if self.edge_target is not None:
                self.edge_target = self.edge_target[self._MASK]

    def _remove_edges_not_in_nodes(self, node_column):
        # need to do this so we get the correct ndata size ... and thus we call _prune_edge_target later too
        nodes = self._nodes[node_column]
        edf = self._edges
        n_initial = len(edf)
        logger.info(f"Length of edge DataFrame {n_initial}")
        mask = edf[self._source].isin(nodes) & edf[self._destination].isin(nodes)
        self._MASK = mask
        self._edges = edf[mask]
        self._prune_edge_target()
        n_final = len(self._edges)
        logger.info(f"Length of edge DataFrame {n_final} after pruning")
        n_final = len(self._edges)
        if n_final != n_initial:
            logger.warn('** Original Edge DataFrame has been changed, some elements have been dropped **')
        

    def _check_nodes_lineup_with_edges(self, node_column):
        nodes = self._nodes[node_column]
        unique_nodes = nodes.unique()
        logger.info(
            f"{len(nodes)} entities from column {node_column}\n with {len(unique_nodes)} unique entities"
        )
        if len(nodes) != len(
            unique_nodes
        ):  # why would this be so? Oh might be so for logs data...oof
            logger.warning(
                f"Nodes DataFrame has duplicate entries for column {node_column}"
            )
        # now check that self.entity_to_index is in 1-1 to with self.ndf[self.node]
        nodes = self._nodes[node_column]
        res = nodes.isin(self.entity_to_index)
        if res.sum() != len(nodes):
            logger.warning(
                f"Some Edges connect to Nodes not explicitly mentioned in nodes DataFrame (ndf)"
            )
        if len(self.entity_to_index) > len(nodes):
            logger.warning(
                f"There are more entities in edges DataFrame (edf) than in nodes DataFrame (ndf)"
            )

    def _convert_edgeDF_to_DGL(self, node_column, weight_column):
        logger.info("converting edge DataFrame to DGL")
        # recall that the adjacency is the graph itself, and really shouldn't be a node style feature (redundant),
        # but certainly can be.
        self._remove_edges_not_in_nodes(node_column)
        self.DGL_graph, self._adjacency, self.entity_to_index = pandas_to_dgl_graph(
            self._edges, self._source, self._destination, weight_col=weight_column, device=self.device
        )
        self.index_to_entity = {k: v for v, k in self.entity_to_index.items()}
        # this is a sanity check after _remove_edges_not_in_nodes
        self._check_nodes_lineup_with_edges(node_column)

    def _featurize_nodes_to_dgl(self, y, use_columns):
        logger.info("Running Node Featurization")
        self._featurize_nodes(y, use_columns)
        X_enc = self.node_features
        y_enc = self.node_target
        ndata = convert_to_torch(X_enc, y_enc)
        # add ndata to the graph
        self.DGL_graph.ndata.update(ndata)
        self._mask_nodes()

    def _featurize_edges_to_dgl(self, y, use_columns):
        logger.info("Running Edge Featurization")
        if hasattr(self, '_MASK'):
            y = y[self._MASK] # automatically prune target using mask
            # note, edf, ndf, should both have unique indices
        self._featurize_edges(y, use_columns)
        X_enc = self.edge_features
        y_enc = self.edge_target
        edata = convert_to_torch(X_enc, y_enc)
        # add edata to the graph
        self.DGL_graph.edata.update(edata)
        self._mask_edges()


    def build_dgl_graph(self, node_column, weight_column=None, y_nodes=None, y_edges=None, use_node_columns=None, use_edge_columns=None):
        # here we make node and edge features and add them to the DGL graph instance
        self._convert_edgeDF_to_DGL(node_column, weight_column)
        self._featurize_nodes_to_dgl(y_nodes, use_node_columns)
        self._featurize_edges_to_dgl(y_edges, use_edge_columns)
        

    def _mask_nodes(self):
        if config.FEATURE in self.DGL_graph.ndata:
            n = self.DGL_graph.ndata[config.FEATURE].shape[0]
            (
                self.DGL_graph.ndata[config.TRAIN_MASK],
                self.DGL_graph.ndata[config.TEST_MASK],
            ) = get_torch_train_test_mask(n, self.train_split)

    def _mask_edges(self):
        if config.FEATURE in self.DGL_graph.edata:
            n = self.DGL_graph.edata[config.FEATURE].shape[0]
            (
                self.DGL_graph.edata[config.TRAIN_MASK],
                self.DGL_graph.edata[config.TEST_MASK],
            ) = get_torch_train_test_mask(n, self.train_split)


    def __getitem__(self, idx):
        # get one example by index
        idx = 1  # only one graph here
        return self.DGL_graph

    def __len__(self):
        # number of data examples
        return 1
    
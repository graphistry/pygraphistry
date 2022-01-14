# classes for converting a dataframe or Graphistry Plottable into a DGL
import numpy as np
import pandas as pd
import torch

from ml import constants as config
from ml.utils import pandas_to_dgl_graph, process_dirty_dataframes, setup_logger
from ml.umap_utils import baseUmap, umap_kwargs_euclidean


logger = setup_logger(__name__)

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
    """
        Might be doing too much -- general idea is to be able to pass in a target as column and select it from df, while returning a new dataframe with target as column
        Likewise, if target is already a series, then return a dataframe with column 'name'
        if target is already a dataframe, send it through without modification
        
        usage examples:
            -- target_dataframe = get_dataframe_for_target('some_col', df, '..')
            -- target_dataframe = get_dataframe_for_target(pd.target_df, df, '..')
            -- target_dataframe = get_dataframe_for_target(pd.Series, df, column_name)
        
    :param target: str, series, or dataframe
    :param df: aux dataframe to use if target is a string
    :param name: useful for when target is a series, it will become the column name of the returned dataframe
    :return: dataframe
    """
    if type(target) == str:
        # get the target from the dataframe
        logger.info(f"Returning {name}-target {target} as DataFrame")
        return pd.DataFrame({target: df[target].values}, index=df.index)
    if type(target) == pd.core.series.Series:
        # use `name` as column header
        logger.info(f"Returning target as DataFrame with column {name}")
        return pd.DataFrame({name: target}, index=target.index)
    if type(target) == pd.core.frame.DataFrame:
        logger.info(f"Returning {name}-target as itself")
        return target
    logger.warning(f"Returning `None` for target")
    return


def check_target_not_in_features(df, y, remove=True):
    """
    
    :param df: model dataframe
    :param y: target dataframe
    :param remove: whether to remove columns from df, default True
    :return: dataframes of model and target
    """
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


def featurize_to_torch(df, y, vectorizer, remove=True):
    """
        Vectorize pandas DataFrames of model and target features into Torch compatible tensors
    :param df: DataFrame of model features
    :param y: DataFrame of target features
    :param vectorizer: must impliment (X, y) encoding and output (X_enc, y_enc, sup_vec, sup_label)
    :param remove: whether to remove target from df
    :return:
        data: dict of {feature, target} encodings
        encoders: dict of {feature, target} encoding objects (which store vocabularity and column information)
    """
    df, y = check_target_not_in_features(df, y, remove=remove)
    X_enc, y_enc, sup_vec, sup_label = vectorizer(df, y)
    if y_enc is not None:
        data = {config.FEATURE: torch.tensor(X_enc.values),
                config.TARGET: torch.tensor(y_enc.values)
                }
    else:
        data = {config.FEATURE: torch.tensor(X_enc.values)}
    encoders = {config.FEATURE: sup_vec, config.TARGET: sup_label}
    return data, encoders



class BaseDGLGraphFromPandas(baseUmap):
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
        umap_kwargs = umap_kwargs_euclidean,
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
        self.device = device
        self._has_node_features = False
        self._has_edge_features = False
        super().__init__(**umap_kwargs)


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
        self.graph, self.adjacency, self.entity_to_index = pandas_to_dgl_graph(
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
        if len(nodes) != len(unique_nodes):  # why would this be so? Oh might be so for logs data...oof
            logger.warning(
                f"Nodes DataFrame has duplicate entries for column {self.node}"
            )
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
        ndata, node_encoders = featurize_to_torch(ndf, self.node_target, self.vectorizer)
        # add ndata to the graph
        self.graph.ndata.update(ndata)
        self.node_encoders = node_encoders
        self._has_node_features = True

    def _edge_featurization(self):
        logger.info("Running Edge Featurization")
        edata, edge_encoders = featurize_to_torch(
            self.edf, self.edge_target, self.vectorizer
        )
        # add edata to the graph
        self.graph.edata.update(edata)
        self.edge_encoders = edge_encoders
        self._has_edge_features = True


    def build_simple_graph(self):
        self._convert_edgeDF_to_DGL()

    def embeddings(self):
        # here we make node and edge features and add them to the DGL graph instance self.graph
        if not hasattr(self, "graph"):
            self._convert_edgeDF_to_DGL()
        if not self._has_node_features:
            self._node_featurization()
        if not self._has_edge_features:
            self._edge_featurization()
        self.umap()
        
        
    def umap(self, scale=False):
        if hasattr(self, "graph"):
            y_node = None
            if config.TARGET in self.graph.ndata:
                y_node = np.array(self.graph.ndata[config.TARGET])  # umap only works on (N, 1) targets....too bad
            if config.FEATURE in self.graph.ndata:
                # take it out of a torch Tensor by wrapping it as ndarray
                X_node = np.array(self.graph.ndata[config.FEATURE])
                if scale:
                    logger.info(f"Z-scaling matrix")
                    X_node = (X_node - X_node.mean(0)) / X_node.std(0)
                logger.info(f"Fitting Umap over matrix of size {X_node.shape}")
                res = self.fit_transform(X_node, y_node)
                self.embedding_ = res
        else:
            logger.info(f"There are no node Level embeddings to fit Umap over")

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

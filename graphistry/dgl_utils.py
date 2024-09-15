# classes for converting a dataframe or Graphistry Plottable into a DGL
from collections import Counter
from typing import Dict, Optional, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

from graphistry.utils.lazy_import import (
    lazy_dgl_import,
    lazy_torch_import_has_dependency
)
from . import constants as config
from .feature_utils import (
    FeatureEngine,
    FeatureMixin,
    resolve_feature_engine,
    XSymbolic,
    YSymbolic,
    resolve_X,
    resolve_y,
)

from .util import setup_logger


if TYPE_CHECKING:
    import scipy
    MIXIN_BASE = FeatureMixin
    try:
        import torch
    except:
        pass
    try:
        import dgl
    except:
        pass
else:
    MIXIN_BASE = object


logger = setup_logger(name=__name__)



# #########################################################################################
#
#  Torch helpers
#
# #########################################################################################


def convert_to_torch(X_enc: pd.DataFrame, y_enc: Optional[pd.DataFrame]):  # type: ignore
    """
        Converts X, y to torch tensors compatible with ndata/edata of DGL graph
    _________________________________________________________________________
    :param X_enc: DataFrame Matrix of Values for Model Matrix
    :param y_enc: DataFrame Matrix of Values for Target
    :return: Dictionary of torch encoded arrays
    """
    _, _, torch = lazy_torch_import_has_dependency()  # noqa: F811

    if not y_enc.empty:  # type: ignore
        data = {
            config.FEATURE: torch.tensor(X_enc.values),
            config.TARGET: torch.tensor(y_enc.values),  # type: ignore
        }
    else:
        data = {config.FEATURE: torch.tensor(X_enc.values)}
    return data

# #################################################################################################
#
#   DGL helpers
#
# #################################################################################################


def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    _, _, torch = lazy_torch_import_has_dependency()  # noqa: F811

    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return device, gpu_ids


def reindex_edgelist(df, src, dst):
    """Since DGL needs integer contiguous node labels, this relabels as pre-processing step

    :eg
        df, ordered_nodes_dict = reindex_edgelist(df, 'to_node', 'from_node')
        creates new columns given by config.SRC and config.DST
    :param df: edge dataFrame
    :param src: source column of dataframe
    :param dst: destination column of dataframe

    :returns
        df, pandas DataFrame with new edges.
        ordered_nodes_dict, dict ordered from most common src and dst nodes.
    """
    srclist = df[src]
    dstlist = df[dst]
    cnt = Counter(
        pd.concat([srclist, dstlist], axis=0)
    )  # can also use pd.Factorize but doesn't order by count, which is satisfying
    ordered_nodes_dict = {k: i for i, (k, c) in enumerate(cnt.most_common())}
    df[config.SRC] = df[src].apply(lambda x: ordered_nodes_dict[x])
    df[config.DST] = df[dst].apply(lambda x: ordered_nodes_dict[x])
    return df, ordered_nodes_dict


def pandas_to_sparse_adjacency(df, src, dst, weight_col):
    """
        Takes a Pandas Dataframe and named src and dst columns into a sparse adjacency matrix in COO format
        Needed for DGL utils
    :param df: edges dataframe
    :param src: source column
    :param dst: destination column
    :param weight_col: optional weight column
    :return: COO sparse matrix, dictionary of src, dst nodes to index
    """
    # use scipy sparse to encode matrix
    from scipy.sparse import coo_matrix
    
    # have to reindex to align edge list with range(n_nodes) with new SRC and DST columns
    df, ordered_nodes_dict = reindex_edgelist(df, src, dst)
    
    eweight = np.array([1] * len(df))
    if weight_col is not None:
        eweight = df[weight_col].values
    
    shape = len(ordered_nodes_dict)
    sp_mat = coo_matrix(
        (eweight, (df[config.SRC], df[config.DST])), shape=(shape, shape)
    )
    return sp_mat, ordered_nodes_dict


# ##############################################################################

def pandas_to_dgl_graph(
    df: pd.DataFrame, src: str, dst: str, weight_col: Optional[str] = None, device: str = "cpu"
) -> Tuple["dgl.DGLGraph", "scipy.sparse.coo_matrix", Dict]:
    """Turns an edge DataFrame with named src and dst nodes, to DGL graph
    :eg
        g, sp_mat, ordered_nodes_dict = pandas_to_sparse_adjacency(df, 'to_node', 'from_node')
    :param df: DataFrame with source and destination and optionally weight column
    :param src: source column of DataFrame for coo matrix
    :param dst: destination column of DataFrame for coo matrix
    :param weight_col: optional weight column when constructing coo matrix
    :param device: whether to put dgl graph on cpu or gpu
    :return
        g: dgl graph
        sp_mat: sparse scipy matrix
        ordered_nodes_dict: dict ordered from most common src and dst nodes
    """
    _, _, dgl = lazy_dgl_import()  # noqa: F811
    sp_mat, ordered_nodes_dict = pandas_to_sparse_adjacency(df, src, dst, weight_col)
    g = dgl.from_scipy(sp_mat, device=device)  # there are other ways too
    logger.info(f"Graph Type: {type(g)}") 

    return g, sp_mat, ordered_nodes_dict


def get_torch_train_test_mask(n: int, ratio: float = 0.8):
    """
        Generates random torch tensor mask
    :param n: size of mask
    :param ratio: mimics train/test split. `ratio` sets number of True vs False mask entries.
    :return: train and test torch tensor masks
    """
    _, _, torch = lazy_torch_import_has_dependency()  # noqa: F811

    train_mask = torch.zeros(n, dtype=torch.bool).bernoulli(ratio)
    test_mask = ~train_mask
    return train_mask, test_mask


########################################################################################################################
#
#   DGL MIXIN
#
#######################################################################################################################


class DGLGraphMixin(MIXIN_BASE):
    """
        Automagic DGL models from Graphistry Instances.
        
    """
    def __init__(self):
        self.dgl_initialized = False

    def dgl_lazy_init(self, train_split: float = 0.8, device: str = "cpu"):
        """
        Initialize DGL graph lazily
        :return:  
        """

        if not self.dgl_initialized:
            lazy_dgl_import()
            lazy_torch_import_has_dependency()
            self.train_split = train_split
            self.device = device
            self._removed_edges_previously = False
            self._dgl_graph = None
            self.dgl_initialized = True

    def _prune_edge_target(self):
        if self._edge_target is not None and hasattr(self, "_MASK"):
            self._edge_target = self._edge_target[self._MASK]

    def _remove_edges_not_in_nodes(self, node_column: str):
        # need to do this so we get the correct ndata size ...
        if self._nodes is None:
            res = self.materialize_nodes()
            nodes = res._nodes[res._node]
        else:
            nodes = self._nodes[node_column]

        if self._source is None or self._destination is None:
            raise ValueError("Need to have source and destination columns bound, call bind() or edges()")
        
        if not isinstance(self._edges, pd.DataFrame):  # type: ignore
            raise ValueError("self._edges for DGLGraphMix must be pd.DataFrame, recieved: %s", type(self._edges))  # type: ignore
        edf: pd.DataFrame = self._edges  # type: ignore
        n_initial = len(edf)
        logger.info(f"Length of edge DataFrame {n_initial}")

        mask : pd.Series[bool] = edf[self._source].isin(nodes) & edf[self._destination].isin(nodes)  # type: ignore
        # print(f'MASK: length: {len(mask)}')
        # print(f'OG: length: {len(edf)}')

        assert (
            sum(mask) > 2
        ), f"mask slice is (practically) empty, will lead to bad graph, found {sum(mask)}"
        self._MASK = mask   # type: ignore
        self._edges = edf[mask]   # type: ignore

        logger.debug(f'new EDGES: length: {len(self._edges)}')

        self._prune_edge_target()
        n_final = len(self._edges)
        logger.info(f"-Length of edge DataFrame {n_final} after pruning")
        n_final = len(self._edges)
        if n_final != n_initial:
            logger.warning(
                "** Original Edge DataFrame has been changed, some elements have been dropped **"
            )
        self._removed_edges_previously = True

    def _check_nodes_lineup_with_edges(self):
        if self._nodes is None:
            res = self.materialize_nodes()
            node_column = res._node
            nodes = res._nodes[node_column]
        else:
            node_column = self._node
            nodes = self._nodes[node_column]
        # nodes = self._nodes[node_column]
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
        # now check that self._entity_to_index is in 1-1 to with self.ndf[node_column]
        # nodes = self._nodes[node_column]
        res = nodes.isin(self._entity_to_index)
        if res.sum() != len(nodes):
            logger.warning(
                "Some Edges connect to Nodes not explicitly mentioned in nodes DataFrame (ndf)"
            )
        if len(self._entity_to_index) > len(nodes):
            logger.warning(
                "There are more entities in edges DataFrame (edf) than in nodes DataFrame (ndf)"
            )

    def _convert_edge_dataframe_to_DGL(
        self, weight_column: Optional[str] = None, inplace: bool = False
    ):
        logger.info("-Converting edge DataFrame to DGL graph")

        if inplace:
            res = self
        else:
            res = self.bind()

        if res._node is None:
            res._node = config.IMPLICIT_NODE_ID

        if not res._removed_edges_previously:
            logger.info(
                f"--Node in convert dataframe to dgl: {res._node}"
            )
            res._remove_edges_not_in_nodes(res._node)

        if res._source is None:
            raise ValueError(
                'source column not set, try running g.bind(source="my_col") or g.edges(df, source="my_col")'
            )

        if res._destination is None:
            raise ValueError(
                'destination column not set, try running g.bind(destination="my_col") or g.edges(df, destination="my_col")'
            )

        res._dgl_graph, res._adjacency, res._entity_to_index = pandas_to_dgl_graph(
            res._edges,
            res._source,
            res._destination,
            weight_col=weight_column,
            device=res.device,
        )
        if res._entity_to_index is None:
            raise ValueError("entity_to_index is None, something went wrong")
        res._index_to_entity = {k: v for v, k in res._entity_to_index.items()}
        # this is a sanity check after _remove_edges_not_in_nodes
        res._check_nodes_lineup_with_edges()
        return res

    def _featurize_nodes_to_dgl(
        self,
        res,
        feature_engine: FeatureEngine = "auto",
        *args,
        **kwargs,
    ):
        logger.info("Running Node Featurization for DGL Graph")
        # logger.debug(f"=*=*=Input shapes are data: {X.shape}, target: {y.shape}")

        X_enc, y_enc, res = res.featurize_or_get_nodes_dataframe_if_X_is_None(
            feature_engine=resolve_feature_engine(feature_engine),
            *args,
            **kwargs
        )

        logger.debug(
            f"=*=*=Encoded Node shapes are data: {X_enc.shape}, target: {y_enc.shape}"

        )

        ndata = convert_to_torch(X_enc, y_enc)
        # add ndata to the graph
        res._dgl_graph.ndata.update(ndata)
        res._mask_nodes()
        return res


    def _featurize_edges_to_dgl(
        self,
        res,
        feature_engine: FeatureEngine = "auto",
        *args,
        **kwargs
    ):
        logger.info("Running Edge Featurization for DGL Graph")

        X_enc, y_enc, res = res.featurize_or_get_edges_dataframe_if_X_is_None(
            feature_engine=resolve_feature_engine(feature_engine),
            *args,
            **kwargs
        )
        
        logger.debug(
            f"=*=*=Encoded Edge shapes are data: {X_enc.shape}, target: {y_enc.shape}"
        )

        edata = convert_to_torch(X_enc, y_enc)
        # add edata to the graph
        res._dgl_graph.edata.update(edata)
        res._mask_edges()
        return res

    def convert_kwargs(self, *args, **kwargs):
        return dict(*args, **kwargs)
        
    def build_gnn(
        self,
        X_nodes: XSymbolic = None,
        X_edges: XSymbolic = None,
        y_nodes: YSymbolic = None,
        y_edges: YSymbolic = None,
        weight_column: Optional[str] = None,
        reuse_if_existing: bool = True,
        featurize_edges: bool = True,
        use_node_scaler: Optional[str] = None,
        use_node_scaler_target: Optional[str] = None,
        use_edge_scaler: Optional[str] = None,
        use_edge_scaler_target: Optional[str] = None,
        train_split: float = 0.8,
        device: str = "cpu",
        inplace: bool = False,
        *args,
        **kwargs
    ):
        """
        Builds GNN model using (DGL)[https://www.dgl.ai/]

        Will auto-featurize, and if no explicit edges are found, automatically UMAP to produce implicit edges.
        ________________________________________________________________________________________________________________

        :param X_nodes: Which node dataframe columns to featurize. If None, will use all columns.
                If passing in explicit dataframe, will set them as attributes.
        :param X_edges: Which edge dataframe columns to featurize. If None, will use all columns.
                If passing in explicit dataframe, will set them as attributes.
        :param y_nodes: Optional target column from nodes dataframe.
        :param y_edges: Optional target column from edges dataframe
        :param weight_column: Optional Weight column if explicit edges table exists with said weights.
                Otherwise, weight_column is inhereted by UMAP.
        :param train_split: Randomly assigns a train and test mask according to the split value, default 80%.
        :param use_node_scaler: selects which scaling to use on featurized nodes dataframe. Default None
        :param use_edge_scaler: selects which scaling to use on featurized edges dataframe. Default None
        :param device: device to run model, default `cpu`, with `gpu` the other choice. Can be handled in outer scope.
        :param inplace: default, False, whether to return Graphistry instance in place or not.

        """
        if inplace:
            res = self
        else:
            res = self.bind()

        res.dgl_lazy_init(train_split=train_split, device=device)

        try:
            m = res.materialize_nodes()
        except Exception as e:
            logger.debug(e)
            logger.info('No edges found, please call g.umap(...) to generate implicit edges')
            raise
        
        X_nodes_resolved = resolve_X(m._nodes, X_nodes)
        y_nodes_resolved = resolve_y(m._nodes, y_nodes)

        # here we check if edges are from UMAP, at which point X_edges should be none:
        if list(res._edges.columns) == ["_src_implicit", "_dst_implicit", "_weight"]:
            logger.debug(
                ">>> EDGES ARE FROM UMAP, discarding explicit mention of X_edges"
            )
            X_edges = None

        X_edges_resolved = resolve_X(res._edges, X_edges)
        y_edges_resolved = resolve_y(res._edges, y_edges)

        if hasattr(res, "_MASK"):
            if y_edges_resolved is not None:
                y_edges_resolved = y_edges_resolved[
                    res._MASK  # type: ignore
                ]  # automatically prune target using mask
                # note, edf, ndf, should both have unique indices

        # here we make node and edge features and add them to the DGL graph instance
        res = res._convert_edge_dataframe_to_DGL(weight_column, inplace)
        
        kwargs_nodes = self.convert_kwargs(X=X_nodes_resolved, y=y_nodes_resolved,
                                           use_scaler=use_node_scaler, use_scaler_target=use_node_scaler_target,
                                           reuse_if_existing=reuse_if_existing,
                                           *args, **kwargs)
        
        res = res._featurize_nodes_to_dgl(
            res,
            **kwargs_nodes
        )
        
        kwargs_edges = self.convert_kwargs(X=X_edges_resolved, y=y_edges_resolved,
                                           use_scaler=use_edge_scaler, use_scaler_target=use_edge_scaler_target,
                                           reuse_if_existing=reuse_if_existing,
                                           *args, **kwargs)
        if featurize_edges:
            res = res._featurize_edges_to_dgl(
                res,
                **kwargs_edges
            )
        if not inplace:
            return res

    def _mask_nodes(self):
        if config.FEATURE in self._dgl_graph.ndata:
            n = self._dgl_graph.ndata[config.FEATURE].shape[0]
            (
                self._dgl_graph.ndata[config.TRAIN_MASK],
                self._dgl_graph.ndata[config.TEST_MASK],
            ) = get_torch_train_test_mask(n, self.train_split)

    def _mask_edges(self):
        if config.FEATURE in self._dgl_graph.edata:
            n = self._dgl_graph.edata[config.FEATURE].shape[0]
            (
                self._dgl_graph.edata[config.TRAIN_MASK],
                self._dgl_graph.edata[config.TEST_MASK],
            ) = get_torch_train_test_mask(n, self.train_split)



# if __name__ == "__main__":
#     import graphistry
#     from graphistry.networks import LinkPredModelMultiOutput
#     #from graphistry.util import setup_logger
    
#     import torch
#     import torch.nn.functional as F
    
#     #logger = setup_logger("Main in DGL_utils", verbose=True)
    
#     edf = edf = pd.read_csv('https://gist.githubusercontent.com/silkspace/33bde3e69ae24fee1298a66d1e00b467/raw/dc66bd6f1687270be7098f94b3929d6a055b4438/malware_bots.csv', index_col=0)
#     edf = edf.drop_duplicates()
#     edf = edf.sample(100000).reset_index(drop=True)
#     src, dst = "to_node", "from_node"
#     edf["to_node"] = edf.SrcAddr
#     edf["from_node"] = edf.DstAddr
    
#     good_cols_without_label = [
#         "Dur",
#         "Proto",
#         "Sport",
#         "Dport",
#         "State",
#         "TotPkts",
#         "TotBytes",
#         "SrcBytes",
#         "to_node",
#         "from_node",
#     ]
    
#     good_cols_without_label_or_edges = [
#         "Dur",
#         "Proto",
#         "Sport",
#         "Dport",
#         "State",
#         "TotPkts",
#         "TotBytes",
#         "SrcBytes",
#     ]
    
#     node_cols = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "ip"]
    
#     use_cols = ["Dur", "TotPkts", "TotBytes", "SrcBytes"]
    
#     T = edf.Label.apply(
#         lambda x: 1 if "Botnet" in x else 0
#     )  # simple indicator, useful for slicing later df.loc[T==1]
    
#     y_edges = pd.DataFrame(
#         {"Label": edf.Label.values}, index=edf.index
#     )  # must include index or g._MASK will through error
    
#     # we can make an effective node_df using edf
#     tdf = edf.groupby(["to_node"], as_index=False).mean().assign(ip=lambda x: x.to_node)
#     fdf = (
#         edf.groupby(["from_node"], as_index=False)
#         .mean()
#         .assign(ip=lambda x: x.from_node)
#     )
#     ndf = pd.concat([tdf, fdf], axis=0)
#     ndf = ndf.fillna(0)
    
#     ndf = ndf[node_cols]
#     ndf = ndf.drop_duplicates(subset=["ip"])
    
#     src, dst = "from_node", "to_node"
#     g = graphistry.edges(edf, src, dst).nodes(ndf, "ip")
    
#     g2 = g.build_gnn(
#         y_edges=y_edges,
#         X_edges=good_cols_without_label_or_edges,
#         X_nodes=use_cols,
#         use_node_scaler="zscale",
#         use_edge_scaler="zscale",
#     )
#     # the DGL graph
#     G = g2._dgl_graph
#     print('G', G)
#     # to get a sense of the different parts in training loop above
#     # labels = torch.tensor(T.values, dtype=torch.float)
#     train_mask = G.edata["train_mask"]
#     test_mask = G.edata["test_mask"]
    
#     # define the model
#     n_feat = G.ndata["feature"].shape[1]
#     latent_dim = 32
#     n_output_feats = (
#         16  # this is equal to the latent dim output of the SAGE net, not n_targets
#     )
    
#     node_features = G.ndata["feature"].float()
#     edge_label = G.edata["target"]
#     n_targets = edge_label.shape[1]
#     labels = edge_label.argmax(1)
#     train_mask = G.edata["train_mask"]
    
#     # instantiate model
#     model = LinkPredModelMultiOutput(
#         n_feat, latent_dim, n_output_feats, n_targets
#     )  # 1) #LinkPredModel(n_feat, latent_dim, n_output_feats)
    
#     pred = model(G, node_features)  # the untrained graph
    
#     print(
#         f"output of model should have same length as the number of edges: {pred.shape[0]}"
#     )
#     print(f"number of edges: {G.num_edges()}")
#     assert G.num_edges() == pred.shape[0], "something went wrong"
    
#     # the optimizer does all the backprop
#     opt = torch.optim.Adam(model.parameters())
    
#     def evaluate(model, graph, features, labels, mask):
#         model.eval()
#         with torch.no_grad():
#             logits = model(graph, features)
#             logits = logits[mask]
#             labels = labels[mask]
#             _, indices = torch.max(logits, dim=1)
#             correct = torch.sum(indices == labels.argmax(1))
#             return correct.item() * 1.0 / len(labels)
    
#     use_cross_entropy_loss = False
#     # train the model
#     for epoch in range(2000):
#         logits = model(G, node_features)
    
#         if use_cross_entropy_loss:
#             loss = F.cross_entropy(logits[train_mask], edge_label[train_mask])
#         else:
#             loss = ((logits[train_mask] - edge_label[train_mask]) ** 2).mean()
    
#         pred = logits.argmax(1)
#         acc = sum(pred[test_mask] == labels[test_mask]) / len(pred[test_mask])
    
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         if epoch % 100 == 0:
#             print(
#                 f"epoch: {epoch} --------\nloss: {loss.item():.4f}\n\taccuracy: {acc:.4f}"
#             )
    
#     # trained comparison
#     logits = model(G, node_features)
#     pred = logits.argmax(1)
    
#     accuracy = sum(pred[test_mask] == labels[test_mask]) / len(
#         pred[test_mask]
#     )  # does pretty well, more data gets it up to 90+%, like RFRegressor!
#     print("-" * 60)
#     print(f"Final Accuracy: {100 * accuracy:.2f}%")

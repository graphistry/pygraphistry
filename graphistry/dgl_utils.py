# classes for converting a dataframe or Graphistry Plottable into a DGL
from typing import List, Any, Optional

import dgl
import pandas as pd
import torch

from . import constants as config
from .feature_utils import FeatureMixin, convert_to_torch
from .ai_utils import pandas_to_sparse_adjacency, setup_logger

logger = setup_logger(__name__, verbose=True)


def pandas_to_dgl_graph(
    df: pd.DataFrame, src: str, dst: str, weight_col: str = None, device: str = "cpu"
):
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
    sp_mat, ordered_nodes_dict = pandas_to_sparse_adjacency(df, src, dst, weight_col)

    g = dgl.from_scipy(sp_mat, device=device)  # there are other ways too
    logger.info(f"Graph Type: {type(g)}")  # why is this making a heterograph?

    return g, sp_mat, ordered_nodes_dict


def get_torch_train_test_mask(n: int, ratio: float = 0.8):
    """
        Generates random torch tensor mask
    :param n: size of mask
    :param ratio: mimics train/test split. `ratio` sets number of True vs False mask entries.
    :return: train and test torch tensor masks
    """
    train_mask = torch.zeros(n, dtype=torch.bool).bernoulli(ratio)
    test_mask = ~train_mask
    return train_mask, test_mask


class DGLGraphMixin(FeatureMixin):
    def __init__(
        self,
        train_split: float = 0.8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        :param train_split: split percent between train and test, set in mask on dgl edata/ndata
        :param device: Whether to put on cpu or gpu. Can always envoke .to(gpu) on returned DGL graph later, Default 'cpu'
        """

        self.train_split = train_split
        self.device = device
        self._removed_edges_previously = False
        FeatureMixin.__init__(self, *args, **kwargs)
        self.DGL_graph = None

    def _prune_edge_target(self):
        if hasattr(self, "edge_target") and hasattr(self, "_MASK"):
            if self.edge_target is not None:
                self.edge_target = self.edge_target[self._MASK]

    def _remove_edges_not_in_nodes(self, node_column: str):
        # need to do this so we get the correct ndata size ...
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
            logger.warn(
                "** Original Edge DataFrame has been changed, some elements have been dropped **"
            )
        self._removed_edges_previously = True

    def _check_nodes_lineup_with_edges(self, node_column: str):
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
        # now check that self.entity_to_index is in 1-1 to with self.ndf[node_column]
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

    def _convert_edgeDF_to_DGL(self, res: Any, node_column: str, weight_column: str):
        logger.info("converting edge DataFrame to DGL graph")

        if not res._removed_edges_previously:
            res._remove_edges_not_in_nodes(node_column)

        res.DGL_graph, res._adjacency, res.entity_to_index = pandas_to_dgl_graph(
            res._edges,
            res._source,
            res._destination,
            weight_col=weight_column,
            device=res.device,
        )
        res.index_to_entity = {k: v for v, k in res.entity_to_index.items()}
        # this is a sanity check after _remove_edges_not_in_nodes
        res._check_nodes_lineup_with_edges(node_column)
        return res

    def _featurize_nodes_to_dgl(
        self,
        res: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        use_columns: List,
        use_scaler: str = None,
    ):
        logger.info("Running Node Featurization for DGL Graph")

        # res = self.featurize(kind="nodes", y=y, use_columns=use_columns, use_scaler=use_scaler, inplace=False)
        X_enc, y_enc = self._featurize_or_get_nodes_dataframe_if_X_is_None(
            res=res, X=X, y=y, use_columns=use_columns, use_scaler=use_scaler
        )

        ndata = convert_to_torch(X_enc, y_enc)
        # add ndata to the graph
        res.DGL_graph.ndata.update(ndata)
        res._mask_nodes()
        return res  # have to return despite inplace flag

    def _featurize_edges_to_dgl(
        self,
        res: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        use_columns: List,
        use_scaler: str = None,
    ):
        logger.info("Running Edge Featurization for DGL Graph")

        # res = self.featurize(kind="edges", y=y, use_columns=use_columns, use_scaler=use_scaler, inplace=False)
        X_enc, y_enc = self._featurize_or_get_edges_dataframe_if_X_is_None(
            res=res, X=X, y=y, use_columns=use_columns, use_scaler=use_scaler
        )

        edata = convert_to_torch(X_enc, y_enc)
        # add edata to the graph
        res.DGL_graph.edata.update(edata)
        res._mask_edges()
        return res  # have to return despite inplace flag

    def build_dgl_graph(
        self,
        node_column: str,
        weight_column: str = None,
        X_nodes: pd.DataFrame = None,
        X_edges: pd.DataFrame = None,
        y_nodes: pd.DataFrame = None,
        y_edges: pd.DataFrame = None,
        use_node_columns: List = None,
        use_edge_columns: List = None,
        use_node_scaler: str = None,
        use_edge_scaler: str = None,
        inplace: bool = False,
    ):
        if inplace:
            res = self
        else:
            res = self.bind()

        if hasattr(res, "_MASK"):
            if y_edges is not None:
                y_edges = y_edges[res._MASK]  # automatically prune target using mask
                # note, edf, ndf, should both have unique indices

        # here we make node and edge features and add them to the DGL graph instance
        res = self._convert_edgeDF_to_DGL(res, node_column, weight_column)
        res = res._featurize_nodes_to_dgl(
            res, X_nodes, y_nodes, use_node_columns, use_node_scaler
        )
        res = res._featurize_edges_to_dgl(
            res, X_edges, y_edges, use_edge_columns, use_edge_scaler
        )
        if not inplace:
            return res

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
        if self.DGL_graph is None:
            logger.warn(f"DGL graph is not built, run `g.build_dgl_graph(..)` first")
        return self.DGL_graph

    def __len__(self):
        # number of data examples
        return 1


if __name__ == "__main__":
    import graphistry
    from graphistry.networks import LinkPredModelMultiOutput
    from graphistry.ai_utils import setup_logger
    from data import get_botnet_dataframe

    import torch
    import torch.nn.functional as F

    logger = setup_logger('Main in DGL_utils', verbose=False)

    edf = get_botnet_dataframe(15000)
    edf = edf.drop_duplicates()
    src, dst = "to_node", "from_node"
    edf["to_node"] = edf.SrcAddr
    edf["from_node"] = edf.DstAddr

    good_cols_without_label = [
        "Dur",
        "Proto",
        "Sport",
        "Dport",
        "State",
        "TotPkts",
        "TotBytes",
        "SrcBytes",
        "to_node",
        "from_node",
    ]

    good_cols_without_label_or_edges = [
        "Dur",
        "Proto",
        "Sport",
        "Dport",
        "State",
        "TotPkts",
        "TotBytes",
        "SrcBytes",
    ]

    node_cols = ["Dur", "TotPkts", "TotBytes", "SrcBytes", "ip"]

    use_cols = ["Dur", "TotPkts", "TotBytes", "SrcBytes"]

    T = edf.Label.apply(
        lambda x: 1 if "Botnet" in x else 0
    )  # simple indicator, useful for slicing later df.loc[T==1]

    y_edges = pd.DataFrame(
        {"Label": edf.Label.values}, index=edf.index
    )  # must include index or g._MASK will through error

    # we can make an effective node_df using edf
    tdf = edf.groupby(["to_node"], as_index=False).mean().assign(ip=lambda x: x.to_node)
    fdf = (
        edf.groupby(["from_node"], as_index=False)
        .mean()
        .assign(ip=lambda x: x.from_node)
    )
    ndf = pd.concat([tdf, fdf], axis=0)
    ndf = ndf.fillna(0)

    ndf = ndf[node_cols]
    ndf = ndf.drop_duplicates(subset=["ip"])

    src, dst = "from_node", "to_node"
    g = graphistry.edges(edf, src, dst).nodes(ndf, "ip")

    g2 = g.build_dgl_graph(
        "ip",
        y_edges=y_edges,
        use_edge_columns=good_cols_without_label,
        use_node_columns=use_cols,
        use_node_scaler="robust",
        use_edge_scaler="robust",
    )
    # the DGL graph
    G = g2.DGL_graph

    # to get a sense of the different parts in training loop above
    # labels = torch.tensor(T.values, dtype=torch.float)
    train_mask = G.edata["train_mask"]
    test_mask = G.edata["test_mask"]

    # define the model
    n_feat = G.ndata["feature"].shape[1]
    latent_dim = 32
    n_output_feats = (
        16  # this is equal to the latent dim output of the SAGE net, not n_targets
    )

    node_features = G.ndata["feature"].float()
    edge_label = G.edata["target"]
    n_targets = edge_label.shape[1]
    labels = edge_label.argmax(1)
    train_mask = G.edata["train_mask"]

    # instantiate model
    model = LinkPredModelMultiOutput(
        n_feat, latent_dim, n_output_feats, n_targets
    )  # 1) #LinkPredModel(n_feat, latent_dim, n_output_feats)

    pred = model(G, node_features)  # the untrained graph

    print(
        f"output of model should have same length as the number of edges: {pred.shape[0]}"
    )
    print(f"number of edges: {G.num_edges()}")
    assert G.num_edges() == pred.shape[0], "something went wrong"

    # the optimizer does all the backprop
    opt = torch.optim.Adam(model.parameters())

    def evaluate(model, graph, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels.argmax(1))
            return correct.item() * 1.0 / len(labels)

    use_cross_entropy_loss = True
    # train the model
    for epoch in range(2000):
        logits = model(G, node_features)

        if use_cross_entropy_loss:
            loss = F.cross_entropy(logits[train_mask], edge_label[train_mask])
        else:
            loss = ((logits[train_mask] - edge_label[train_mask]) ** 2).mean()

        pred = logits.argmax(1)
        acc = sum(pred[test_mask] == labels[test_mask]) / len(pred[test_mask])
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print(f'epoch: {epoch} --------\nloss: {loss.item():.4f}\n\taccuracy: {acc:.4f}')

    # trained comparison
    logits = model(G, node_features)
    pred = logits.argmax(1)

    accuracy = sum(pred[test_mask] == labels[test_mask]) / len(pred[test_mask])  # does pretty well!
    print('-'*60)
    print(f'Final Accuracy: {100 * accuracy:.2f}%')

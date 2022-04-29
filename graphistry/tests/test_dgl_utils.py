
import unittest
import graphistry
import pandas as pd
from graphistry.util import setup_logger
import torch
import torch.nn.functional as F

logger = setup_logger("DGL_utils", verbose=True)

edf = pd.read_csv('graphistry/tests/data/malware_capture_bot.csv', index_col=0, nrows=50)
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
ndf = ndf.drop_duplicates(subset=["ip"]).reset_index(drop=True)

## a target
T = edf.Label.apply(
    lambda x: 1 if "Botnet" in x else 0
)  # simple indicator, useful for slicing later df.loc[T==1]

# explicit dataframe
y_edges = pd.DataFrame(
    {"Label": edf.Label.values}, index=edf.index
)
y_nodes = pd.DataFrame(
    {"Dur": ndf.Dur.values}, index=ndf.index
)
X_nodes = pd.DataFrame(
    {col: ndf[col] for col in node_cols}, index=ndf.index
)
X_edges = pd.DataFrame(
    {col: edf[col] for col in good_cols_without_label}, index=edf.index
)


class TestDGL(unittest.TestCase):
    
    def _test_cases_dgl(self, g):
        G = g.DGL_graph
        self.assertEqual(G.num_edges(), 50)
        self.assertEqual(G.num_nodes(), 60)
        keys = ['feature', 'target', 'train_mask', 'test_mask']
        keys_without_target = ['feature', 'train_mask', 'test_mask']
        if len(G.ndata.keys())==3:
            self.assertSequenceEqual(list(G.ndata.keys()),  keys_without_target)
        else:
            self.assertSequenceEqual(list(G.ndata.keys()),  keys)
        if len(G.edata.keys())==3:
            self.assertSequenceEqual(list(G.edata.keys()),  keys_without_target)
        else:
            self.assertSequenceEqual(list(G.edata.keys()),  keys)
        for k in keys:
            assert isinstance(G.ndata[k].sum(), torch.Tensor), f'Node {G.ndata[k]} for {k} is not a Tensor'
            assert isinstance(G.edata[k].sum(), torch.Tensor), f'Edge {G.edata[k]} for {k} is not a Tensor'

        
    def test_build_dgl_graph_from_column_names(self):
        g = graphistry.edges(edf, src, dst).nodes(ndf, "ip")

        g2 = g.build_dgl_graph(
            node_column="ip",
            y_edges='Label',
            y_nodes='Dur',
            X_edges=good_cols_without_label,
            X_nodes=use_cols,
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)
        
    def test_build_dgl_graph_from_dataframes(self):
        g = graphistry.edges(edf, src, dst).nodes(ndf, "ip")

        g2 = g.build_dgl_graph(
            node_column="ip",
            y_edges=y_edges,
            y_nodes=y_nodes,
            X_edges=X_edges,
            X_nodes=X_nodes,
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)
        
    def test_build_dgl_graph_from_umap(self):
        g = graphistry.nodes(ndf, 'ip')
        g._feature_params = {}  # so that we redo calcs
        g = g.umap()
        #g = g.umap(X=X_nodes, y=y_edges)

        g2 = g.build_dgl_graph(
            node_column='ip',
            y_edges=y_edges,
            y_nodes=y_nodes,
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)

    def test_build_dgl_graph_from_umap2(self):
        g = graphistry.nodes(ndf, 'ip')
        g._feature_params = {}  # so that we redo calcs
        g = g.umap()
        #g = g.umap(X=X_nodes, y=y_edges)

        g2 = g.build_dgl_graph(
            y_edges=y_edges,
            y_nodes=y_nodes,
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)
        
    def test_build_dgl_graph_from_umap_no_node_column(self):
        g = graphistry.nodes(ndf)
        g._feature_params = {}  # so that we redo calcs
        #g = g.umap(y=y_edges, X=X_nodes)
        g = g.umap()

        g2 = g.build_dgl_graph(
            X_edges=X_edges,
            X_nodes=X_nodes,
            y_edges=y_edges,
            y_nodes=y_nodes,
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)
        




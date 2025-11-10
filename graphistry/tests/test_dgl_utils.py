import unittest
import pytest
import graphistry
import pandas as pd
from graphistry.util import setup_logger
from graphistry.utils.lazy_import import lazy_dgl_import


has_dgl, _, dgl = lazy_dgl_import()

if has_dgl:
    import torch

logger = setup_logger(__name__)

edf = pd.read_csv(
    "graphistry/tests/data/malware_capture_bot.csv", index_col=0, nrows=50
)
edf = edf.drop_duplicates()
src, dst = "to_node", "from_node"
edf["to_node"] = edf.SrcAddr.astype(str)
edf["from_node"] = edf.DstAddr.astype(str)

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
tdf = edf.groupby(["to_node"], as_index=False).mean(numeric_only=True).assign(ip=lambda x: x.to_node)
fdf = edf.groupby(["from_node"], as_index=False).mean(numeric_only=True).assign(ip=lambda x: x.from_node)
ndf = pd.concat([tdf, fdf], axis=0)
ndf = ndf.fillna(0)

ndf = ndf[node_cols]
ndf = ndf.drop_duplicates(subset=["ip"]).reset_index(drop=True)

# a target
T = edf.Label.apply(
    lambda x: 1 if "Botnet" in x else 0
)  # simple indicator, useful for slicing later df.loc[T==1]

# explicit dataframe
y_edges = pd.DataFrame({"Label": edf.Label.values}, index=edf.index)
y_nodes = pd.DataFrame({"Dur": ndf.Dur.values}, index=ndf.index)
X_nodes = pd.DataFrame({col: ndf[col] for col in node_cols}, index=ndf.index)
X_edges = pd.DataFrame(
    {col: edf[col] for col in good_cols_without_label}, index=edf.index
)


class TestDGL(unittest.TestCase):
    def _test_cases_dgl(self, g):
        # simple test to see if DGL graph was set during different featurization + umap strategies
        G = g._dgl_graph
        keys = ["feature", "target", "train_mask", "test_mask"]
        keys_without_target = ["feature", "train_mask", "test_mask"]

        use_node_target = True
        use_edge_target = True

        if len(G.ndata.keys()) == 3:
            use_node_target = False
            self.assertSequenceEqual(list(G.ndata.keys()), keys_without_target)
        else:
            self.assertSequenceEqual(list(G.ndata.keys()), keys)
        if len(G.edata.keys()) == 3:
            use_edge_target = False
            self.assertSequenceEqual(list(G.edata.keys()), keys_without_target)
        else:
            self.assertSequenceEqual(list(G.edata.keys()), keys)
        if use_node_target:
            for k in keys:
                assert isinstance(
                    G.ndata[k].sum(), torch.Tensor
                ), f"Node {G.ndata[k]} for {k} is not a Tensor"
        else:
            for k in keys_without_target:
                assert isinstance(
                    G.ndata[k].sum(), torch.Tensor
                ), f"Node {G.ndata[k]} for {k} is not a Tensor"
        if use_edge_target:
            for k in keys:
                assert isinstance(
                    G.edata[k].sum(), torch.Tensor
                ), f"Edge {G.edata[k]} for {k} is not a Tensor"
        else:
            for k in keys_without_target:
                assert isinstance(
                    G.ndata[k].sum(), torch.Tensor
                ), f"Node {G.ndata[k]} for {k} is not a Tensor"

    @pytest.mark.skipif(not has_dgl, reason="requires DGL dependencies")
    def test_build_dgl_graph_from_column_names(self):
        g = graphistry.edges(edf, src, dst).nodes(ndf, "ip")

        g2 = g.build_gnn(
            y_edges="Label",
            y_nodes="Dur",
            X_edges=good_cols_without_label,
            X_nodes=use_cols,
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)

    @pytest.mark.skipif(not has_dgl, reason="requires DGL dependencies")
    def test_build_dgl_graph_from_dataframes(self):
        g = graphistry.edges(edf, src, dst).nodes(ndf, "ip")

        g2 = g.build_gnn(
            y_edges=y_edges,
            y_nodes=y_nodes,
            X_edges=X_edges,
            X_nodes=X_nodes,
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)

    @pytest.mark.skipif(not has_dgl, reason="requires DGL dependencies")
    def test_build_dgl_graph_from_umap(self):
        # explicitly set node in .nodes() and not in .build_gnn()
        g = graphistry.nodes(ndf, "ip")
        g.reset_caches()  # so that we redo calcs
        g = g.umap(scale=1)  # keep all edges with scale = 1

        g2 = g.build_gnn(
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)

    @pytest.mark.skipif(not has_dgl, reason="requires DGL dependencies")
    def test_build_dgl_graph_from_umap_no_node_column(self):
        g = graphistry.nodes(ndf)
        g.reset_caches()  # so that we redo calcs
        g = g.umap(scale=1)  # keep all edges with scale = 100

        g2 = g.build_gnn(
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)

    @pytest.mark.skipif(not has_dgl, reason="requires DGL dependencies")
    @pytest.mark.xfail(reason="Mishandling datetimes: https://github.com/graphistry/pygraphistry/issues/381")
    def test_build_dgl_with_no_node_features(self):
        g = graphistry.edges(edf, src, dst)
        g.reset_caches()  # so that we redo calcs
        #g = g.umap(scale=1) #keep all edges with scale = 100
        # should produce random features for nodes
        g2 = g.build_gnn(
            use_node_scaler="robust",
            use_edge_scaler="robust",
        )
        self._test_cases_dgl(g2)

    def test_entity_to_index_none_dereference(self):
        """
        Regression test for None dereference bug in dgl_utils.py:314 (Issue #801)

        When _entity_to_index is None, calling len() or isin() on it raises TypeError.
        This test ensures the fix prevents the crash by checking for None first.
        """
        df = pd.DataFrame({
            'src': ['A', 'B', 'C'],
            'dst': ['B', 'C', 'A']
        })

        g = graphistry.edges(df, 'src', 'dst')

        # Manually set _entity_to_index to None to trigger the bug scenario
        g._entity_to_index = None

        # After fix: This should NOT crash (None check prevents the bug)
        g._check_nodes_lineup_with_edges()

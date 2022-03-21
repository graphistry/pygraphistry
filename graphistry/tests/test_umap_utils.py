from typing import Any
import copy, datetime as dt, graphistry, numpy as np, os, pandas as pd
import pytest, unittest

from graphistry.umap_utils import (
    has_dependancy
)

from graphistry.tests.test_feature_utils import (

    ndf_reddit,
    text_cols_reddit,
    meta_cols_reddit,
    good_cols_reddit,
    single_target_reddit,
    double_target_reddit,
    edge_df,
    single_target_edge,
    double_target_edge,
    good_edge_cols,

    remove_internal_namespace_if_present
)


class TestUMAPMethods(unittest.TestCase):
    
    def _check_attributes(self, g, attributes):
        msg = "Graphistry instance after umap should have `{}` as attribute"
        for attribute in attributes:
            self.assertTrue(hasattr(g, attribute), msg.format(attribute))

    def cases_check_node_attributes(self, g):
        attributes = [
            "weighted_edges_df_from_nodes",
            "node_embedding",
            "weighted_adjacency_nodes",
            "_weighted_edges_df",
            "_weighted_adjacency",
            "_umap",
        ]
        self._check_attributes(g, attributes)

    def cases_check_edge_attributes(self, g):
        attributes = [
            "weighted_edges_df_from_edges",
            "edge_embedding",
            "weighted_adjacency_edges",
            "_weighted_edges_df",
            "_weighted_adjacency",
            "_umap",
        ]
        self._check_attributes(g, attributes)

    def cases_test_graph(self, g, kind="nodes", df=ndf_reddit):
        if kind == "nodes":
            ndf = g._nodes
            self.cases_check_node_attributes(g)
        else:
            ndf = g._edges
            self.cases_check_edge_attributes(g)

        ndf = remove_internal_namespace_if_present(ndf)
        cols = ndf.columns
        self.assertTrue(
            np.all(ndf.reset_index(drop=True) == df[cols].reset_index(drop=True)),
            f"Graphistry {kind}-dataframe does not match outside dataframe it was fed",
        )

    @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    def _test_umap(self, g, use_cols, targets, name, kind, df):
        for use_col in use_cols:
            for target in targets:
                print("*" * 90)
                value = [target, use_col]
                print(f"{kind} -- {name}")
                print(f"{value}")
                print("-" * 80)
                g2 = g.umap(kind=kind, y=target, use_columns=use_col)

                self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    def test_node_umap(self):
        g = graphistry.nodes(ndf_reddit)
        use_cols = [None, text_cols_reddit, good_cols_reddit, meta_cols_reddit]
        targets = [None, single_target_reddit, double_target_reddit]
        self._test_umap(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Node UMAP with `(target, use_col)=`",
            kind="nodes",
            df=ndf_reddit,
        )

    @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    def test_edge_umap(self):
        g = graphistry.edges(edge_df, "src", "dst")
        targets = [None, single_target_edge, double_target_edge]
        use_cols = [None, good_edge_cols]
        self._test_umap(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Edge UMAP with `(target, use_col)=`",
            kind="edges",
            df=edge_df,
        )

    @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    def test_filter_edges(self):
        for kind, g in [('nodes', graphistry.nodes(ndf_reddit))]:
            g2 = g.umap(kind=kind)
            last_shape = 0
            for scale in np.linspace(0, 6, 8):  # six sigma in 8 steps
                g3 = g2.filter_edges(scale=scale)
                shape = g3._edges.shape
                print('*' * 90)
                print(f'{kind} -- scale: {scale}: resulting edges dataframe shape: {shape}')
                print('-' * 80)
                self.assertGreaterEqual(shape[0], last_shape)
                last_shape = shape[0]

                
if __name__ == "__main__":
    unittest.main()

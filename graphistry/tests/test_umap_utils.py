from typing import Any
import copy, datetime as dt, graphistry, logging, numpy as np, os, pandas as pd
import pytest, unittest, warnings

from graphistry.umap_utils import has_dependancy
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
    model_avg_name,
    remove_internal_namespace_if_present,
    has_min_dependancy as has_featurize,
)

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


triangleEdges = pd.DataFrame(
    {
        "src": ["a", "b", "c", "d"] * 3,
        "dst": ["b", "c", "a", "a"] * 3,
        "int": [0, 1, 2, 3] * 3,
        "flt": [0.0, 1.0, 2.0, 3.0] * 3,
        "y": [0.0, 1.0, 2.0, 3.0] * 3,
    }
)
edge_ints = ["int"]
edge_floats = ["flt"]
edge_numeric = edge_ints + edge_floats
edge_target = triangleEdges[["y"]]

triangleNodes = pd.DataFrame(
    {
        "id": ["a", "b", "c"] * 10,
        "int": [1, 2, 3] * 10,
        "flt": [0.0, 1.0, 2.0] * 10,
        "y": [0.0, 1.0, 2.0] * 10,
    }
)
node_ints = ["int"]
node_floats = ["flt"]
node_numeric = node_ints + node_floats
node_target = triangleNodes[["y"]]


class TestUMAPMethods(unittest.TestCase):
    def _check_attributes(self, g, attributes):
        msg = "Graphistry instance after umap should have `{}` as attribute"
        for attribute in attributes:
            self.assertTrue(hasattr(g, attribute), msg.format(attribute))

    def cases_check_node_attributes(self, g):
        attributes = [
            "_weighted_edges_df_from_nodes",
            "_node_embedding",
            "_weighted_adjacency_nodes",
            "_weighted_edges_df",
            "_weighted_adjacency",
            "_umap",
        ]
        self._check_attributes(g, attributes)

    def cases_check_edge_attributes(self, g):
        attributes = [
            "_weighted_edges_df_from_edges",
            "_edge_embedding",
            "_weighted_adjacency_edges",
            "_weighted_edges_df",
            "_weighted_adjacency",
            "_umap",
        ]
        self._check_attributes(g, attributes)

    def cases_test_graph(self, g, kind="nodes", df=ndf_reddit, verbose=False):
        if kind == "nodes":
            ndf = g._nodes
            self.cases_check_node_attributes(g)
        else:
            ndf = g._edges
            self.cases_check_edge_attributes(g)

        ndf = remove_internal_namespace_if_present(ndf)
        cols = ndf.columns
        logger.debug("g_nodes", g._nodes)
        logger.debug("df", df)
        self.assertTrue(
            np.array_equal(ndf.reset_index(drop=True), df[cols].reset_index(drop=True)),
            f"Graphistry {kind}-dataframe does not match outside dataframe it was fed",
        )

    @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    def _test_umap(self, g, use_cols, targets, name, kind, df):
        for use_col in use_cols:
            for target in targets:
                for feature_engine in ["none", "auto", "pandas"]:
                    logger.debug("*" * 90)
                    value = [target, use_col]
                    logger.debug(f"{kind} -- {name}")
                    logger.debug(f"{value}: featurize umap {feature_engine}")
                    logger.debug("-" * 80)
                    g2 = g.umap(
                        kind=kind,
                        y=target,
                        X=use_col,
                        model_name=model_avg_name,
                        feature_engine=feature_engine,
                        n_neighbors=2,
                    )

                    self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    def test_node_umap(self):
        g = graphistry.nodes(triangleNodes)
        use_cols = [node_ints, node_floats, node_numeric]
        targets = [node_target]
        self._test_umap(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Node UMAP with `(target, use_col)=`",
            kind="nodes",
            df=triangleNodes,
        )

    @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    def test_edge_umap(self):
        g = graphistry.edges(triangleEdges, "src", "dst")
        use_cols = [edge_ints, edge_floats, edge_numeric]
        targets = [edge_target]
        self._test_umap(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Edge UMAP with `(target, use_col)=`",
            kind="edges",
            df=triangleEdges,
        )

    # @pytest.mark.skipif(not has_dependancy or not has_featurize, reason="requires umap feature dependencies")
    # def test_filter_edges(self):
    #     for kind, g in [("nodes", graphistry.nodes(triangleNodes))]:
    #         g2 = g.umap(kind=kind, feature_engine="none")
    #         last_shape = 0
    #         for scale in np.linspace(0, 3, 8):  # six sigma in 8 steps
    #             g3 = g2.filter_weighted_edges(scale=scale)
    #             shape = g3._edges.shape
    #             logger.debug("*" * 90)
    #             logger.debug(
    #                 f"{kind} -- scale: {scale}: resulting edges dataframe shape: {shape}"
    #             )
    #             logger.debug("-" * 80)
    #             self.assertGreaterEqual(shape[0], last_shape)  # should return more and more edges
    #             last_shape = shape[0]


class TestUMAPAIMethods(TestUMAPMethods):
    @pytest.mark.skipif(
        not has_dependancy or not has_featurize,
        reason="requires ai+umap feature dependencies",
    )
    def _test_umap(self, g, use_cols, targets, name, kind, df):
        for use_col in use_cols:
            for target in targets:
                logger.debug("*" * 90)
                value = [target, use_col]
                logger.debug(f"{kind} -- {name}")
                logger.debug(f"{value}")
                logger.debug("-" * 80)
                g2 = g.umap(kind=kind, y=target, X=use_col, model_name=model_avg_name)

                self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(
        not has_dependancy or not has_featurize,
        reason="requires ai+umap feature dependencies",
    )
    def test_node_umap(self):
        g = graphistry.nodes(ndf_reddit)
        use_cols = [None, text_cols_reddit, good_cols_reddit, meta_cols_reddit]
        targets = [None, single_target_reddit, double_target_reddit]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self._test_umap(
                g,
                use_cols=use_cols,
                targets=targets,
                name="Node UMAP with `(target, use_col)=`",
                kind="nodes",
                df=ndf_reddit,
            )

    @pytest.mark.skipif(
        not has_dependancy or not has_featurize,
        reason="requires ai+umap feature dependencies",
    )
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

    @pytest.mark.skipif(
        not has_dependancy or not has_featurize,
        reason="requires ai+umap feature dependencies",
    )
    def test_chaining_nodes(self):
        g = graphistry.nodes(ndf_reddit)
        g2 = g.umap()
        logger.debug('======= g.umap() done ======')
        g3a = g2.featurize()
        logger.debug('======= g3a.featurize() done ======')
        g3 = g3a.umap()
        logger.debug('======= g3.umap() done ======')
        assert g2._node_features.shape == g3._node_features.shape
        # since g3 has feature params with x and y. 
        g3._feature_params['nodes']['X'].pop('x')
        g3._feature_params['nodes']['X'].pop('y')
        assert all(g2._feature_params['nodes']['X'] == g3._feature_params['nodes']['X'])
        assert g2._feature_params['nodes']['y'].shape == g3._feature_params['nodes']['y'].shape  # None
        assert g2._node_embedding.shape == g3._node_embedding.shape
        
    @pytest.mark.skipif(
        not has_dependancy or not has_featurize,
        reason="requires ai+umap feature dependencies",
    )
    def test_chaining_edges(self):
        g = graphistry.edges(edge_df, "src", "dst")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            g2 = g.umap(kind='edges')
            g3 = g.featurize(kind='edges').umap(kind='edges')
        assert all(g2._edge_features == g3._edge_features)
        assert all(g2._feature_params['edges']['X'] == g3._feature_params['edges']['X'])
        assert all(g2._feature_params['edges']['y'] == g3._feature_params['edges']['y'])  # None
        assert g2._edge_embedding.sum() == g3._edge_embedding.sum()

    
    @pytest.mark.skipif(
        not has_dependancy or not has_featurize,
        reason="requires ai+umap feature dependencies",
    )
    def test_filter_edges(self):
        for kind, g in [("nodes", graphistry.nodes(ndf_reddit))]:
            g2 = g.umap(kind=kind, model_name=model_avg_name)
            last_shape = 0
            for scale in np.linspace(0, 1, 8):  # six sigma in 8 steps
                g3 = g2.filter_weighted_edges(scale=scale)
                shape = g3._edges.shape
                logger.debug("*" * 90)
                logger.debug(
                    f"{kind} -- scale: {scale}: resulting edges dataframe shape: {shape}"
                )
                logger.debug("-" * 80)
                self.assertGreaterEqual(shape[0], last_shape)
                last_shape = shape[0]


if __name__ == "__main__":
    unittest.main()

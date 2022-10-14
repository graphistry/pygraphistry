from xml.sax.handler import feature_external_ges
import pytest
import unittest
import warnings

import graphistry
import logging
import numpy as np
import pandas as pd
from graphistry.feature_utils import remove_internal_namespace_if_present
from graphistry.tests.test_feature_utils import (
    ndf_reddit,
    text_cols_reddit,
    meta_cols_reddit,
    good_cols_reddit,
    single_target_reddit,
    double_target_reddit,
    edge_df,
    edge_df2,
    edge2_target_df,
    model_avg_name,
    lazy_import_has_min_dependancy,
    check_allclose_fit_transform_on_same_data
)
from graphistry.umap_utils import lazy_umap_import_has_dependancy, lazy_cuml_import_has_dependancy

logger = logging.getLogger(__name__)

has_dependancy, _ = lazy_import_has_min_dependancy()
has_cuml, _, _ = lazy_cuml_import_has_dependancy()
has_umap, _, _ = lazy_umap_import_has_dependancy()

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


class TestUMAPFitTransform(unittest.TestCase):
    # check to see that .fit and transform gives similar embeddings on same data
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def setUp(self):

        g = graphistry.nodes(ndf_reddit)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            g2 = g.umap(y=double_target_reddit, 
                        use_ngrams=True, 
                        ngram_range=(1, 2), 
                        use_scaler='robust', 
                        cardinality_threshold=2)
            
        fenc = g2._node_encoder
        self.X, self.Y = fenc.X, fenc.y
        self.EMB = g2._node_embedding
        self.emb, self.x, self.y = g2.transform_umap(ndf_reddit, ydf=double_target_reddit, kind='nodes')

        edge_df22 = edge_df2.copy()
        edge_df22['rando'] = np.random.rand(edge_df2.shape[0])
        g = graphistry.edges(edge_df22, 'src', 'dst')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            g2 = g.umap(y=edge2_target_df, kind='edges',
                 use_ngrams=True, 
                 ngram_range=(1, 2),
                 use_scaler=None,
                 use_scaler_target=None,
                 cardinality_threshold=2, n_topics=4)
        
        fenc = g2._edge_encoder
        self.Xe, self.Ye = fenc.X, fenc.y
        self.EMBe = g2._edge_embedding
        self.embe, self.xe, self.ye = g2.transform_umap(edge_df22, ydf=edge2_target_df, kind='edges')

    # @pytest.mark.skipif(not has_dependancy, reason="requires umap feature dependencies")
    # def test_allclose_fit_transform_on_same_data(self):
    #     check_allclose_fit_transform_on_same_data(self.X, self.x, self.Y, self.y)
    #     check_allclose_fit_transform_on_same_data(self.Xe, self.xe, self.Ye, self.ye)

    #     check_allclose_fit_transform_on_same_data(self.EMB, self.emb, None, None)
    #     check_allclose_fit_transform_on_same_data(self.EMBe, self.embe, None, None)

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_columns_match(self):
        assert all(self.X.columns == self.x.columns), 'Node Feature Columns do not match'
        assert all(self.Y.columns == self.y.columns), 'Node Target Columns do not match'
        assert all(self.Xe.columns == self.xe.columns), 'Edge Feature Columns do not match'
        assert all(self.Ye.columns == self.ye.columns), 'Edge Target Columns do not match'


class TestUMAPMethods(unittest.TestCase):
    def _check_attributes(self, g, attributes):
        msg = "Graphistry instance after umap should have `{}` as attribute"
        msg2 = "Graphistry instance after umap should not have None values for `{}`"

        for attribute in attributes:
            self.assertTrue(hasattr(g, attribute), msg.format(attribute))
            self.assertTrue(getattr(g, attribute) is not None, msg2.format(attribute))
            if 'df' in attribute:
                self.assertIsInstance(getattr(g, attribute), pd.DataFrame, msg.format(attribute))
            if 'node_' in attribute:
                self.assertIsInstance(getattr(g, attribute), pd.DataFrame, msg.format(attribute))
            if 'edge_' in attribute:
                self.assertIsInstance(getattr(g, attribute), pd.DataFrame, msg.format(attribute))


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
        logger.debug("g_nodes: %s", g._nodes)
        logger.debug("df: %s", df)
        self.assertTrue(
            np.array_equal(ndf.reset_index(drop=True), df[cols].reset_index(drop=True)),
            f"Graphistry {kind}-dataframe does not match outside dataframe it was fed",
        )

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def _test_umap(self, g, use_cols, targets, name, kind, df):
        for use_col in use_cols:
            for target in targets:
                for feature_engine in ["none", "auto", "pandas"]:
                    logger.debug("*" * 90)
                    print("*" * 90)
                    value = [target, use_col]
                    print(f"{kind} -- {name}")
                    print(f"{value}: featurize umap {feature_engine}")
                    print("-" * 80)

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

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
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

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
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

    @pytest.mark.skipif(not has_dependancy or not has_umap, reason="requires umap feature dependencies")
    def test_filter_edges(self):
        for kind, g in [("nodes", graphistry.nodes(triangleNodes))]:
            g2 = g.umap(kind=kind, feature_engine="none")
            last_shape = 0
            for scale in np.linspace(0, 1, 8):
                g3 = g2.filter_weighted_edges(scale=scale)
                shape = g3._edges.shape
                logger.debug("*" * 90)
                logger.debug(
                    f"{kind} -- scale: {scale}: resulting edges dataframe shape: {shape}"
                )
                logger.debug("-" * 80)
                self.assertGreaterEqual(shape[0], last_shape)  # should return more and more edges
                last_shape = shape[0]


class TestUMAPAIMethods(TestUMAPMethods):
    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def _test_umap(self, g, use_cols, targets, name, kind, df):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for scaler in ['kbins', 'robust']:
                for cardinality in [2, 200]:
                    for use_ngram in [True, False]:
                        for use_col in use_cols:
                            for target in targets:
                                logger.debug("*" * 90)
                                value = [scaler, cardinality, use_ngram, target, use_col]
                                logger.debug(f"{value}")
                                logger.debug("-" * 80)
                                g2 = g.umap(kind=kind,
                                    X=use_col,
                                    y=target,
                                    model_name=model_avg_name,
                                    use_scaler=scaler,
                                    use_scaler_target=scaler,
                                    use_ngrams=use_ngram,
                                    engine='umap_learn',
                                    cardinality_threshold=cardinality,
                                    cardinality_threshold_target=cardinality,
                                    n_neighbors=3)

                                self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def test_node_umap(self):
        g = graphistry.nodes(ndf_reddit)
        use_cols = [None, text_cols_reddit, good_cols_reddit, meta_cols_reddit]
        targets = [None, single_target_reddit, double_target_reddit]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            self._test_umap(
                g,
                use_cols=use_cols,
                targets=targets,
                name="Node UMAP with `(target, use_col)=`",
                kind="nodes",
                df=ndf_reddit,
            )

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def test_edge_umap(self):
        g = graphistry.edges(edge_df2, "src", "dst")
        targets = [None, 'label']
        use_cols = [None, 'title']
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            self._test_umap(
                g,
                use_cols=use_cols,
                targets=targets,
                name="Edge UMAP with `(target, use_col)=`",
                kind="edges",
                df=edge_df2,
            )

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
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
        assert g2._node_embedding.shape == g3._node_embedding.shape  # kinda weak sauce
        
    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def test_chaining_edges(self):
        g = graphistry.edges(edge_df, "src", "dst")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            g2 = g.umap(kind='edges')
            g3 = g.featurize(kind='edges').umap(kind='edges')
            
        assert all(g2._feature_params['edges']['X'] == g3._feature_params['edges']['X'])
        assert all(g2._feature_params['edges']['y'] == g3._feature_params['edges']['y'])  # None
        assert all(g2._edge_features == g3._edge_features)

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def test_feature_kwargs_yield_different_values_using_umap_api(self):
        g = graphistry.nodes(ndf_reddit)
        n_topics_target = 6
    
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            g2 = g.umap(X="type", y="label", cardinality_threshold_target=3, n_topics_target=n_topics_target)  # makes a GapEncoded Target
            g3 = g.umap(X="type", y="label", cardinality_threshold_target=30000)  # makes a one-hot-encoded target
            
        assert all(g2._feature_params['nodes']['X'] == g3._feature_params['nodes']['X']), "features should be the same"
        assert all(g2._feature_params['nodes']['y'] != g3._feature_params['nodes']['y']), "targets in memoize should be different"  # None
        assert g2._node_target.shape[1] != g3._node_target.shape[1], 'Targets should be different'
        assert g2._node_target.shape[1] == n_topics_target, 'Targets '

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
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

@pytest.mark.skipif(
    not has_dependancy or not has_cuml,
    reason="requires cuml feature dependencies",
)
class TestCUMLMethods(TestUMAPMethods):

    def _test_umap(self, g, use_cols, targets, name, kind, df):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for scaler in ['kbins', 'robust']:
                for cardinality in [2, 200]:
                    for use_ngram in [True, False]:
                        for use_col in use_cols:
                            for target in targets:
                                logger.debug("*" * 90)
                                value = [scaler, cardinality, use_ngram, target, use_col]
                                logger.debug(f"{value}")
                                logger.debug("-" * 80)
                                g2 = g.umap(kind=kind,
                                    X=use_col,
                                    y=target,
                                    model_name=model_avg_name,
                                    use_scaler=scaler,
                                    use_scaler_target=scaler,
                                    use_ngrams=use_ngram,
                                    engine='cuml',
                                    cardinality_threshold=cardinality,
                                    cardinality_threshold_target=cardinality,
                                    n_neighbors=3)

                                self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_node_umap(self):
        g = graphistry.nodes(ndf_reddit)
        use_cols = [None, text_cols_reddit, good_cols_reddit, meta_cols_reddit]
        targets = [None, single_target_reddit, double_target_reddit]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            self._test_umap(
                g,
                use_cols=use_cols,
                targets=targets,
                name="Node UMAP with `(target, use_col)=`",
                kind="nodes",
                df=ndf_reddit,
            )

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_edge_umap(self):
        g = graphistry.edges(edge_df2, "src", "dst")
        targets = [None, 'label']
        use_cols = [None, 'title']
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            self._test_umap(
                g,
                use_cols=use_cols,
                targets=targets,
                name="Edge UMAP with `(target, use_col)=`",
                kind="edges",
                df=edge_df2,
            )

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
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
        assert g2._node_embedding.shape == g3._node_embedding.shape  # kinda weak sauce
        
    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_chaining_edges(self):
        g = graphistry.edges(edge_df, "src", "dst")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            g2 = g.umap(kind='edges')
            g3 = g.featurize(kind='edges').umap(kind='edges')
            
        assert all(g2._feature_params['edges']['X'] == g3._feature_params['edges']['X'])
        assert all(g2._feature_params['edges']['y'] == g3._feature_params['edges']['y'])  # None
        assert all(g2._edge_features == g3._edge_features)

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_feature_kwargs_yield_different_values_using_umap_api(self):
        g = graphistry.nodes(ndf_reddit)
        n_topics_target = 6
    
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            g2 = g.umap(X="type", y="label", cardinality_threshold_target=3, n_topics_target=n_topics_target)  # makes a GapEncoded Target
            g3 = g.umap(X="type", y="label", cardinality_threshold_target=30000)  # makes a one-hot-encoded target
            
        assert all(g2._feature_params['nodes']['X'] == g3._feature_params['nodes']['X']), "features should be the same"
        assert all(g2._feature_params['nodes']['y'] != g3._feature_params['nodes']['y']), "targets in memoize should be different"  # None
        assert g2._node_target.shape[1] != g3._node_target.shape[1], 'Targets should be different'
        assert g2._node_target.shape[1] == n_topics_target, 'Targets '

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
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

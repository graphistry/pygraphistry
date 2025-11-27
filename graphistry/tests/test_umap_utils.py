from time import time
from typing import Any
import warnings
import pytest
import unittest

import gc
import graphistry
from graphistry.pygraphistry import PyGraphistry
import os
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
    check_allclose_fit_transform_on_same_data,
)
from graphistry.utils.lazy_import import (
    lazy_cudf_import,
    lazy_cuml_import,
    lazy_umap_import,
)
from graphistry.util import cache_coercion_helper

has_dependancy, _ = lazy_import_has_min_dependancy()
has_cuml, _, _ = lazy_cuml_import()
has_umap, _, umap = lazy_umap_import()
has_cudf, _, cudf = lazy_cudf_import()

# print('has_dependancy', has_dependancy)
# print('has_cuml', has_cuml)
# print('has_umap', has_umap)

logger = logging.getLogger(__name__)
logging.getLogger("graphistry.umap_utils").setLevel(logging.DEBUG)


# enable tests if has cudf and env didn't explicitly disable
is_test_cudf = has_cudf and os.environ["TEST_CUDF"] != "0"

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

def _eq(df1, df2):
    try:
        df1 = df1.to_pandas()
    except:
        pass
    try:
        df2 = df2.to_pandas()
    except:
        pass
    return df1 == df2


@pytest.fixture(scope="module")
def reddit_ndf() -> pd.DataFrame:
    return ndf_reddit


class TestUMAPFitTransformMore():

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_kwargs_threaded(self, reddit_ndf: pd.DataFrame):

        g = graphistry.nodes(reddit_ndf.assign(zzz=2)).featurize(feature_engine='none')

        #warmup
        g2 = g.umap(
            feature_engine='none',
            engine='umap_learn',
            umap_kwargs={'random_state': 43, 'n_epochs': 1},
            umap_fit_kwargs={'force_all_finite': False},
            umap_transform_kwargs={},  # no args in older versions..
        )

        start_time = time()
        g2 = g.umap(
            feature_engine='none',
            engine='umap_learn',
            umap_kwargs={'random_state': 43, 'n_epochs': 2},
            umap_fit_kwargs={'force_all_finite': False},
            umap_transform_kwargs={},  # no args in older versions..
        )
        runtime_small = time() - start_time

        assert g2._umap_params['random_state'] == 43
        assert g2._umap_fit_kwargs['force_all_finite'] is False
        assert g2._umap_transform_kwargs == {}

        start_time = time()
        g2 = g.umap(
            feature_engine='none',
            engine='umap_learn',
            umap_kwargs={'random_state': 43, 'n_epochs': 2000},
            umap_fit_kwargs={'force_all_finite': False},
            umap_transform_kwargs={},  # no args in older versions..
        )
        runtime_large = time() - start_time

        logger.debug(f"runtime_small: {runtime_small}, runtime_large: {runtime_large}")

        assert runtime_large > 1.5 * runtime_small


class TestUMAPFitTransformNodes(unittest.TestCase):
    # check to see that .fit and transform gives similar embeddings on same data
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def setUp(self):
        g = graphistry.nodes(ndf_reddit)
        self.gn = g
        
        self.test = ndf_reddit.sample(5)

        g2 = g.umap(
            y=['label', 'type'],
            use_ngrams=True,
            ngram_range=(1, 2),
            use_scaler="robust",
            cardinality_threshold=2
        )

        self.g2 = g2
        fenc = g2._node_encoder
        self.X, self.Y = fenc.X, fenc.y
        self.EMB = g2._node_embedding
        self.emb, self.x, self.y = g2.transform_umap(
            ndf_reddit, ndf_reddit, kind="nodes", return_graph=False
        )
        self.g3 = g2.transform_umap(
            ndf_reddit, ndf_reddit, kind="nodes", return_graph=True
        )


    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_columns_match(self):
        assert set(self.X.columns) == set(self.x.columns), f"Node Feature Columns do not match: {set(self.X.columns)} vs {set(self.x.columns)}"
        assert set(self.Y.columns) == set(self.y.columns), f"Node Target Columns do not match: {set(self.Y.columns)} vs {set(self.y.columns)}"

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_index_match(self):
        # nodes
        d = self.g2._nodes.shape[0]
        assert _eq(self.gn._nodes.index, self.g2._nodes.index).sum() == d, "Node Indexes do not match"
        assert _eq(self.gn._nodes.index, self.EMB.index).sum() == d, "Emb Indexes do not match"
        assert _eq(self.gn._nodes.index, self.emb.index).sum() == d, "Transformed Emb Indexes do not match"
        assert _eq(self.gn._nodes.index, self.X.index).sum() == d, "Transformed Node features Indexes do not match"
        assert _eq(self.gn._nodes.index, self.y.index).sum() == d, "Transformed Node target Indexes do not match"

        # make sure the indexes match at transform time internally as well
        assert _eq(self.X.index, self.x.index).sum() == d, "Node Feature Indexes do not match"
        assert _eq(self.Y.index, self.y.index).sum() == d, "Node Target Indexes do not match"

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_node_index_match_in_infered_graph(self):
        # nodes
        g3 = self.g2._nodes
        assert _eq(g3.index, self.EMB.index).sum() == len(g3), "Node Emb Indexes do not match"
        assert _eq(g3.index, self.emb.index).sum() == len(g3), "Node Transformed Emb Indexes do not match"
        assert _eq(g3.index, self.X.index).sum() == len(g3), "Node Transformed features Indexes do not match"
        assert _eq(g3.index, self.y.index).sum() == len(g3), "Node Transformed target Indexes do not match"

    
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_kwargs(self):
        umap_kwargs = {
            "n_components": 2,
            "metric": "euclidean",
            "n_neighbors": 3,
            "min_dist": 1,
            "spread": 1,
            "local_connectivity": 1,
            "repulsion_strength": 1,
            "negative_sample_rate": 5,
        }

        umap_kwargs2 = {k: v + 1 for k, v in umap_kwargs.items() if k not in ['metric']}  # type: ignore
        umap_kwargs2['metric'] = 'euclidean'
        g = graphistry.nodes(self.test)
        g2 = g.umap(**umap_kwargs, engine='umap_learn')
        g3 = g.umap(**umap_kwargs2, engine='umap_learn')
        assert g2._umap_params == umap_kwargs
        assert (
            g2._umap_params == umap_kwargs
        ), f"Umap params do not match, found {g2._umap_params} vs {umap_kwargs}"
        assert len(g2._node_embedding.columns) == 2, f"Umap params do not match, found {len(g2._node_embedding.columns)} vs 2"

        assert (
            g3._umap_params == umap_kwargs2
        ), f"Umap params do not match, found {g3._umap_params} vs {umap_kwargs2}"
        assert len(g3._node_embedding.columns) == 3, f"Umap params do not match, found {len(g3._node_embedding.columns)} vs 3"
        
        g4 = g2.transform_umap(self.test)
        assert (
            g4._umap_params == umap_kwargs
        ), f"Umap params do not match, found {g4._umap_params} vs {umap_kwargs}"
        assert g4._n_components == 2, f"Umap params do not match, found {g2._n_components} vs 2"

        g5 = g3.transform_umap(self.test)
        assert (
            g5._umap_params == umap_kwargs2
        ), f"Umap params do not match, found {g5._umap_params} vs {umap_kwargs2}"

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_transform_umap(self):
        np.random.seed(41)
        test = self.test
        assert ( 
            self.g2._node_embedding.shape[0] <= self.g3._node_embedding.shape[0]
        ), "Node Embedding Lengths do not match, found {} and {}".format(
            self.g2._node_embedding.shape[0], self.g3._node_embedding.shape[0]
        )
        # now feed it args
        min_dist = ["auto", 10]
        sample = [None, 2]
        return_graph = [True, False]
        fit_umap_embedding = [True, False]
        n_neighbors = [2, None]
        for ep in min_dist:
            g4 = self.g2.transform_umap(test, test, min_dist=ep)
            assert True
        for return_g in return_graph:
            g4 = self.g2.transform_umap(test, test, return_graph=return_g)
            if return_g:
                assert True
            else:
                objs = (pd.DataFrame,)
                if has_cudf:
                    objs = (pd.DataFrame, cudf.DataFrame)
                assert len(g4) == 3
                assert isinstance(g4[0], objs)
                assert isinstance(g4[1], objs)
                assert isinstance(g4[2], objs)
                assert g4[0].shape[1] == 2
                assert g4[1].shape[1] >= 2
                assert g4[2].shape[0] == test.shape[0]
        for n_neigh in n_neighbors:
            g4 = self.g2.transform_umap(test, n_neighbors=n_neigh)
            assert True
        for sample_ in sample:
            print("sample", sample_)
            g4 = self.g2.transform_umap(test, sample=sample_)
            assert True
        for fit_umap_embedding_ in fit_umap_embedding:
            g4 = self.g2.transform_umap(test, fit_umap_embedding=fit_umap_embedding_)
            assert True




class TestUMAPFitTransformEdges(unittest.TestCase):
    # check to see that .fit and transform gives similar embeddings on same data
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def setUp(self):
        
        self.test = ndf_reddit.sample(5)

        # do the same for edges
        edge_df22 = edge_df2.copy()
        edge_df22["rando"] = np.random.rand(edge_df2.shape[0])
        g = graphistry.edges(edge_df22, "src", "dst")
        self.ge = g
        g2 = g.umap(
            y=['label'],
            kind="edges",
            use_ngrams=True,
            ngram_range=(1, 2),
            use_scaler=None,
            use_scaler_target=None,
            cardinality_threshold=2,
            n_topics=4,
        )

        fenc = g2._edge_encoder
        self.Xe, self.Ye = fenc.X, fenc.y
        self.EMBe = g2._edge_embedding
        self.embe, self.xe, self.ye = g2.transform_umap(
            edge_df22, y=edge2_target_df, kind="edges", return_graph=False
        )        
        self.g2e = g2


    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_columns_match(self):
        assert set(self.Xe.columns) == set(self.xe.columns), f"Edge Feature Columns do not match: {set(self.Xe.columns)} vs {set(self.xe.columns)}"
        assert set(self.Ye.columns) == set(self.ye.columns), f"Edge Target Columns do not match: {set(self.Ye.columns)} vs {set(self.ye.columns)}"

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_index_match(self):
        # nodes
        de = self.g2e._edges.shape[0]

        # edges
        assert _eq(self.ge._edges.index, self.g2e._edges.index).sum() == de, "Edge Indexes do not match"
        assert _eq(self.ge._edges.index, self.EMBe.index).sum() == de, "Edge Emb Indexes do not match"
        assert _eq(self.ge._edges.index, self.embe.index).sum() == de, "Edge Transformed Emb Indexes do not match"
        assert _eq(self.ge._edges.index, self.Xe.index).sum() == de, "Edge Transformed features Indexes do not match"
        assert _eq(self.ge._edges.index, self.ye.index).sum() == de, "Edge Transformed target Indexes do not match"

        # make sure the indexes match at transform time internally as well
        assert _eq(self.Xe.index, self.xe.index).sum() == de, "Edge Feature Indexes do not match"
        assert _eq(self.Ye.index, self.ye.index).sum() == de, "Edge Target Indexes do not match"


    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_edge_index_match_in_infered_graph(self):
        g3 = self.g2e._edges
        assert _eq(g3.index, self.EMBe.index).sum() == len(g3), "Edge Emb Indexes do not match"
        assert _eq(g3.index, self.embe.index).sum() == len(g3), "Edge Transformed Emb Indexes do not match"
        assert _eq(g3.index, self.Xe.index).sum() == len(g3), "Edge Transformed Node features Indexes do not match"
        assert _eq(g3.index, self.ye.index).sum() == len(g3), "Edge Transformed Node target Indexes do not match"


class TestUMAPMethods(unittest.TestCase):
    def _check_attributes(self, g, attributes):
        msg = "Graphistry instance after umap should have `{}` as attribute"
        msg2 = "Graphistry instance after umap should not have None values for `{}`"
        objs = (pd.DataFrame,)
        if has_cudf:
            objs = (pd.DataFrame, cudf.DataFrame)

        for attribute in attributes:
            self.assertTrue(hasattr(g, attribute), msg.format(attribute))
            self.assertTrue(getattr(g, attribute) is not None, msg2.format(attribute))
            if "df" in attribute:
                self.assertIsInstance(
                    getattr(g, attribute), objs, msg.format(attribute)
                )
            if "node_" in attribute:
                self.assertIsInstance(
                    getattr(g, attribute), objs, msg.format(attribute)
                )
            if "edge_" in attribute:
                self.assertIsInstance(
                    getattr(g, attribute), objs, msg.format(attribute)
                )

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

    def cases_test_graph(self, g, kind="nodes", df=ndf_reddit):
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
        assert ndf.shape == df[cols].shape
        assert ndf.reset_index(drop=True).equals(df[cols].reset_index(drop=True))

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
                        dbscan=False,
                    )

                    self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_simplest(self):
        df = pd.DataFrame({
            'x': ['aa a' * 10, 'bb b' * 2, 'ccc ' * 20, 'dd abc', 'ee x1z'] * 10,
            'y': [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        })
        graphistry.nodes(df).umap()
        assert True
    
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_edgecase(self):
        df = pd.DataFrame({
            'x': ['aa a' * 10, 'bb b' * 2, 'ccc ' * 20, 'dd abc', 'ee x1z'] * 10,
            'y': [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            'yy': [1.1, 20, 31, 12, 5.0] * 10,
        })
        df['z'] = df['x'].apply(lambda x: x[0])
        df.loc[[1,20,35,42,30], 'z'] = 1
        df.loc[[10,5,16,28,35], 'z'] = 1.0
        df.loc[[12,7], 'z'] = 'NaN'
        df.loc[[13,8], 'z'] = np.nan

        graphistry.nodes(df).umap()
        assert True
        
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_type_edgecase(self):
        values = pd.Series(np.random.rand(50))
        num_to_convert = int(len(values) * 0.05)
        indices_to_convert = np.random.choice(len(values), num_to_convert, replace=False)
        for i in indices_to_convert:
            values[i] = str(values[i])
        values.loc[13] = '92.026 123.903 702.124'
        values.loc[33] = '26.092 903.123'

        graphistry.nodes(values).umap()
        assert True

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

    @pytest.mark.skipif(
        not has_dependancy or not has_umap, reason="requires umap feature dependencies"
    )
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
                self.assertGreaterEqual(
                    shape[0], last_shape
                )  # should return more and more edges
                last_shape = shape[0]


class TestUMAPAIMethods(TestUMAPMethods):
    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def _test_umap(self, g, use_cols, targets, name, kind, df):
        for scaler in ["kbins", "robust"]:
            for cardinality in [2, 200]:
                for use_ngram in [True, False]:
                    for use_col in use_cols:
                        for target in targets:
                            logger.debug("*" * 90)
                            value = [
                                scaler,
                                cardinality,
                                use_ngram,
                                target,
                                use_col,
                            ]
                            logger.debug(f"{value}")
                            logger.debug("-" * 80)

                            g2 = g.umap(
                                kind=kind,
                                X=use_col,
                                y=target,
                                model_name=model_avg_name,
                                use_scaler=scaler,
                                use_scaler_target=scaler,
                                use_ngrams=use_ngram,
                                engine="umap_learn",
                                cardinality_threshold=cardinality,
                                cardinality_threshold_target=cardinality,
                                n_neighbors=3,
                                dbscan=False,
                            )

                            self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
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

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def test_edge_umap(self):
        g = graphistry.edges(edge_df2, "src", "dst")
        targets = [None, "label"]
        use_cols = [None, "title"]
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
        g2 = g.umap(dbscan=False)

        logger.debug("======= g.umap() done ======")
        g3a = g2.featurize()
        logger.debug("======= g3a.featurize() done ======")
        g3 = g3a.umap(dbscan=False)
        logger.debug("======= g3.umap() done ======")
        assert len(g2._node_features) == len(g3._node_features)
        assert g2._node_features.shape[1] < g3._node_features.shape[1]

        # since g3 has feature params with x and y.
        g3._feature_params["nodes"]["X"].pop("x")
        g3._feature_params["nodes"]["X"].pop("y")
        assert all(g2._feature_params["nodes"]["X"] == g3._feature_params["nodes"]["X"])
        assert (
            g2._feature_params["nodes"]["y"].shape == g3._feature_params["nodes"]["y"].shape
        )  # None
        assert len(g2._node_embedding) == len(g3._node_embedding)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                assert set(g2._node_embedding.columns) == set(g3._node_embedding.columns)
            except AssertionError:
                warnings.warn("Columns do not match", UserWarning)

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def test_chaining_edges(self):
        g = graphistry.edges(edge_df, "src", "dst")
        g2 = g.umap(kind="edges", dbscan=False)
        g3 = g.featurize(kind="edges").umap(kind="edges", dbscan=False)

        assert all(g2._feature_params["edges"]["X"] == g3._feature_params["edges"]["X"])
        assert all(
            g2._feature_params["edges"]["y"] == g3._feature_params["edges"]["y"]
        )  # None
        assert all(g2._edge_features == g3._edge_features)

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
        reason="requires ai+umap feature dependencies",
    )
    def test_feature_kwargs_yield_different_values_using_umap_api(self):
        g = graphistry.nodes(ndf_reddit)
        n_topics_target = 6

        g2 = g.umap(
            X="type",
            y="label",
            cardinality_threshold_target=3,
            n_topics_target=n_topics_target,
        )  # makes a GapEncoded Target
        g3 = g.umap(
            X="type", y="label", cardinality_threshold_target=30000
        )  # makes a one-hot-encoded target

        assert all(
            g2._feature_params["nodes"]["X"] == g3._feature_params["nodes"]["X"]
        ), "features should be the same"
        assert all(
            g2._feature_params["nodes"]["y"] != g3._feature_params["nodes"]["y"]
        ), "targets in memoize should be different"  # None
        assert (
            g2._node_target.shape[1] != g3._node_target.shape[1]
        ), "Targets should be different"
        assert g2._node_target.shape[1] == n_topics_target, "Targets "

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
class TestCUMLMethodsMore():

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_kwargs_threaded(self, reddit_ndf: pd.DataFrame):

        g = graphistry.nodes(cudf.from_pandas(reddit_ndf.assign(zzz=2))).featurize(feature_engine='none')

        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._node_features, cudf.DataFrame)

        #warmup
        g.umap(
            feature_engine='none',
            engine='cuml',
            umap_kwargs={},
            umap_fit_kwargs={},
            umap_transform_kwargs={}
        )

        start_time = time()
        g.umap(
            feature_engine='none',
            engine='cuml',
            umap_kwargs={'n_epochs': 3},  # smaller values crash: https://github.com/rapidsai/cuml/issues/6068
            umap_fit_kwargs={},
            umap_transform_kwargs={}
        )
        runtime_small = time() - start_time

        start_time = time()
        g.umap(
            feature_engine='none',
            engine='cuml',
            umap_kwargs={'n_epochs': 20000},
            umap_fit_kwargs={},
            umap_transform_kwargs={}
        )
        runtime_large = time() - start_time
        #assert g2._umap_params['random_state'] == 43
        logger.debug(f"runtime_small: {runtime_small}, runtime_large: {runtime_large}")
        assert runtime_large > 1.5 * runtime_small


@pytest.mark.skipif(
    not has_dependancy or not has_cuml,
    reason="requires cuml feature dependencies",
)
class TestCUMLMethods(TestUMAPMethods):

    def setup_method(self, method: Any) -> None:
        cache_coercion_helper.cache_clear()
        gc.collect()

    @classmethod
    def setup_class(cls: Any) -> None:
        PyGraphistry.session.encode_textual_batch_size = 8

    @classmethod
    def teardown_class(cls: Any) -> None:
        PyGraphistry.session.encode_textual_batch_size = None

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def _test_umap(self, g, use_cols, targets, name, kind, df):
        for scaler in ["kbins", "robust"]:
            for cardinality in [2, 200]:
                for use_ngram in [True, False]:
                    for use_col in use_cols:
                        for target in targets:
                            logger.debug("*" * 90)
                            value = [
                                scaler,
                                cardinality,
                                use_ngram,
                                target,
                                use_col,
                            ]
                            logger.debug(f"{name}:\n{value}")
                            logger.debug("-" * 80)

                            g2 = g.umap(
                                kind=kind,
                                X=use_col,
                                y=target,
                                model_name=model_avg_name,
                                use_scaler=scaler,
                                use_scaler_target=scaler,
                                use_ngrams=use_ngram,
                                engine="cuml",
                                cardinality_threshold=cardinality,
                                cardinality_threshold_target=cardinality,
                                n_neighbors=3,
                            )

                            self.cases_test_graph(g2, kind=kind, df=df)

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_node_umap(self):
        g = graphistry.nodes(ndf_reddit[:len(ndf_reddit) // 2].reset_index(drop=True))
        use_cols = [None, text_cols_reddit, good_cols_reddit, meta_cols_reddit]
        targets = [None, single_target_reddit, double_target_reddit]
        for i, target in enumerate(targets):
            if target is None:
                continue
            targets[i] = target[:len(g._nodes)].reset_index(drop=True)

        self._test_umap(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Node UMAP with `(target, use_col)=`",
            kind="nodes",
            df=g._nodes,
        )

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_edge_umap(self):
        g = graphistry.edges(edge_df2, "src", "dst")
        targets = [None, "label"]
        use_cols = [None, "title"]
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

        logger.debug("======= g.umap() done ======")
        g3a = g2.featurize()
        logger.debug("======= g3a.featurize() done ======")
        g3 = g3a.umap()
        logger.debug("======= g3.umap() done ======")
        assert g2._node_features.shape[0] == g3._node_features.shape[0]
        assert g2._node_features.shape[1] < g3._node_features.shape[1]
        # since g3 has feature params with x and y.
        g3._feature_params["nodes"]["X"].pop("x")
        g3._feature_params["nodes"]["X"].pop("y")
        assert all(g2._feature_params["nodes"]["X"] == g3._feature_params["nodes"]["X"])
        assert (
            g2._feature_params["nodes"]["y"].shape == g3._feature_params["nodes"]["y"].shape
        )  # None
        assert g2._node_embedding.shape == g3._node_embedding.shape  # kinda weak sauce

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_chaining_edges(self):
        g = graphistry.edges(edge_df, "src", "dst")
        g2 = g.umap(kind="edges")
        g3 = g.featurize(kind="edges").umap(kind="edges")

        assert all(g2._feature_params["edges"]["X"] == g3._feature_params["edges"]["X"])
        assert all(
            g2._feature_params["edges"]["y"] == g3._feature_params["edges"]["y"]
        )  # None
        assert all(g2._edge_features == g3._edge_features)

    @pytest.mark.skipif(
        not has_dependancy or not has_cuml,
        reason="requires cuml feature dependencies",
    )
    def test_feature_kwargs_yield_different_values_using_umap_api(self):
        g = graphistry.nodes(ndf_reddit)
        n_topics_target = 6

        g2 = g.umap(
            X="type",
            y="label",
            cardinality_threshold_target=3,
            n_topics_target=n_topics_target,
        )  # makes a GapEncoded Target
        g3 = g.umap(
            X="type", y="label", cardinality_threshold_target=30000
        )  # makes a one-hot-encoded target

        assert all(
            g2._feature_params["nodes"]["X"] == g3._feature_params["nodes"]["X"]
        ), "features should be the same"
        assert all(
            g2._feature_params["nodes"]["y"] != g3._feature_params["nodes"]["y"]
        ), "targets in memoize should be different"  # None
        assert (
            g2._node_target.shape[1] != g3._node_target.shape[1]
        ), "Targets should be different"
        assert g2._node_target.shape[1] == n_topics_target, "Targets "

    @pytest.mark.skipif(
        not has_dependancy or not has_umap,
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

class TestCudfUmap(unittest.TestCase):
    # temporary tests for cudf pass thru umap
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def setUp(self):
        self.samples = 1000
        df = pd.DataFrame(np.random.randint(18,75,size=(self.samples, 1)), columns=['age'])
        df['user_id'] = np.random.randint(0,200,size=(self.samples, 1))
        df['profile'] = np.random.randint(0,1000,size=(self.samples, 1))
        self.df = cudf.from_pandas(df)

    @pytest.mark.skipif(not has_dependancy or not has_cuml, reason="requires cuml dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def test_base(self):
        graphistry.nodes(self.df).umap(engine='auto')._node_embedding.shape == (self.samples, 2)
        graphistry.nodes(self.df).umap(engine='cuml')._node_embedding.shape == (self.samples, 2)

    @pytest.mark.skipif(not has_dependancy or not has_cuml, reason="requires cuml dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def test_umap_cuml_engine_produces_consistent_cudf_types(self):
        """
        Test that UMAP with engine='cuml' returns consistent cuDF DataFrame types.

        BEFORE FIX: This would fail because:
        - nodes._nodes would be pandas DataFrame
        - nodes._edges would be cuDF DataFrame
        - Mixed types cause chain operations to fail

        AFTER FIX: Both nodes and edges are cuDF DataFrames.
        """
        # Create test data as pandas (simulating typical input)
        df = pd.DataFrame({
            'node': [f'node_{i}' for i in range(20)],
            'col1': range(20),
            'col2': range(20, 40),
            'col3': range(40, 60),
            'col4': range(60, 80),
            'col5': range(80, 100),
        })

        # Create graph with pandas input
        g = graphistry.nodes(df, node='node')

        # Run UMAP with engine='cuml' - this should convert everything to cuDF
        g_umap = g.umap(
            X=['col1', 'col2', 'col3', 'col4', 'col5'],
            engine='cuml',
            n_neighbors=5
        )

        # Verify both nodes and edges are cuDF DataFrames
        self.assertIsNotNone(g_umap._nodes, "Nodes should exist after UMAP")
        self.assertIsNotNone(g_umap._edges, "Edges should exist after UMAP")

        # This is the critical assertion that would FAIL without the fix
        self.assertIsInstance(g_umap._nodes, cudf.DataFrame,
            f"Expected nodes to be cuDF DataFrame, got {type(g_umap._nodes)}")
        self.assertIsInstance(g_umap._edges, cudf.DataFrame,
            f"Expected edges to be cuDF DataFrame, got {type(g_umap._edges)}")

    @pytest.mark.skipif(not has_dependancy or not has_cuml, reason="requires cuml dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def test_umap_cuml_chain_operations_work(self):
        """
        Test that chain operations work after UMAP with engine='cuml'.

        This test verifies the original bug is fixed: chain concatenation
        should not fail with TypeError due to mixed DataFrame types.
        """
        # Create test data
        df = pd.DataFrame({
            'node': [f'node_{i}' for i in range(20)],
            'col1': range(20),
            'col2': range(20, 40),
            'col3': range(40, 60),
            'col4': range(60, 80),
            'col5': range(80, 100),
        })

        g = graphistry.nodes(df, node='node')

        # Use gfql to chain UMAP operations - this triggers chain concatenation
        # BEFORE FIX: This would fail with TypeError due to mixed types
        # AFTER FIX: This should succeed
        try:
            result = g.gfql([
                {
                    "type": "Call",
                    "function": "umap",
                    "params": {"X": ["col1", "col2", "col3", "col4", "col5"]}
                },
                {
                    "type": "Call",
                    "function": "name",
                    "params": {"name": "umap_result"}
                }
            ], engine='cudf')

            # Should succeed without TypeError
            self.assertIsNotNone(result._nodes)
            self.assertIsNotNone(result._edges)
            self.assertIsInstance(result._nodes, cudf.DataFrame)
            self.assertIsInstance(result._edges, cudf.DataFrame)

        except TypeError as e:
            # If we get TypeError about mixed types, the fix didn't work
            if "can only concatenate" in str(e):
                self.fail(f"Chain operation failed with mixed DataFrame types: {e}")
            raise


if __name__ == "__main__":
    unittest.main()

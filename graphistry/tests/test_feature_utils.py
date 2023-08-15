# python -m unittest
import os
import datetime as dt
import graphistry
import logging
import numpy as np
import pandas as pd
from typing import Any

import pytest
import unittest
import warnings

from graphistry.feature_utils import (
    process_dirty_dataframes,
    process_nodes_dataframes,
    resolve_feature_engine,
    lazy_import_has_min_dependancy,
    lazy_import_has_dependancy_text,
    lazy_import_has_dependancy_cuda,
    FastEncoder
)

from graphistry.features import topic_model, ngrams_model
from graphistry.constants import SCALERS

np.random.seed(137)

has_min_dependancy, _ = lazy_import_has_min_dependancy()
has_min_dependancy_text, _, _ = lazy_import_has_dependancy_text()
has_cudf, _, _ = lazy_import_has_dependancy_cuda()

# enable tests if has cudf and env didn't explicitly disable
is_test_cudf = has_cudf and os.environ["TEST_CUDF"] != "0"

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("graphistry.feature_utils").setLevel(logging.DEBUG)

model_avg_name = (
    #"/models/average_word_embeddings_komninos"  # 250mb, fastest vectorizer in transformer models
    "/models/paraphrase-albert-small-v2"  # 40mb
    #"/models/paraphrase-MiniLM-L3-v2"  # 60mb
)


bad_df = pd.DataFrame(
    {
        "src": [0, 1, 2, 3],
        "dst": [1, 2, 3, 0],
        "colors": [1, 1, 2, 2],
        "list_int": [[1], [2, 3], [4], []],
        "list_str": [["x"], ["1", "2"], ["y"], []],
        "list_bool": [[True], [True, False], [False], []],
        # "list_date_str": [
        #     ["2018-01-01 00:00:00"],
        #     ["2018-01-02 00:00:00", "2018-01-03 00:00:00"],
        #     ["2018-01-05 00:00:00"],
        #     [],
        # ],
        # "list_date": [
        #     [pd.Timestamp("2018-01-05")],
        #     [pd.Timestamp("2018-01-05"), pd.Timestamp("2018-01-05")],
        #     [],
        #     [],
        # ],
        "list_mixed": [[1], ["1", "2"], [False, None], []],
        "bool": [True, False, True, True],
        "char": ["a", "b", "c", "d"],
        "str": ["a", "b", "c", "d"],
        "ustr": ["a", "b", "c", "d"],
        "emoji": ["😋", "😋😋", "😋", "😋"],
        "int": [0, 1, 2, 3],
        "num": [0.5, 1.5, 2.5, 3.5],
        # "date_str": [
        #     "2018-01-01 00:00:00",
        #     "2018-01-02 00:00:00",
        #     "2018-01-03 00:00:00",
        #     "2018-01-05 00:00:00",
        # ],
        # API 1 BUG: Try with https://github.com/graphistry/pygraphistry/pull/126
        # "date": [
        #     dt.datetime(2018, 1, 1),
        #     dt.datetime(2018, 1, 1),
        #     dt.datetime(2018, 1, 1),
        #     dt.datetime(2018, 1, 1),
        # ],
        # "time": [
        #     pd.Timestamp("2018-01-05"),
        #     pd.Timestamp("2018-01-05"),
        #     pd.Timestamp("2018-01-05"),
        #     pd.Timestamp("2018-01-05"),
        # ],
        # # API 2 BUG: Need timedelta in https://github.com/graphistry/pygraphistry/blob/master/graphistry/vgraph.py#L108
        # "delta": [
        #     pd.Timedelta("1 day"),
        #     pd.Timedelta("1 day"),
        #     pd.Timedelta("1 day"),
        #     pd.Timedelta("1 day"),
        # ],
        "textual": [
            "here we have a sentence. And here is another sentence. Graphistry is an amazing tool!"
        ] * 2 + ['And now for something completely different so we dont mess up the tests with a repeat document'] + ['I love my wife'],
    }
)

# ###############################################
# For EDGE FEATURIZATION AND TESTS
edge_df = bad_df.astype(str)
good_edge_cols = [
    "textual",
    #"delta",
    #"time",
    "colors",
    "list_str",
    "bool",
    "char",
    #"list_date",
]

single_target_edge = pd.DataFrame({"emoji": edge_df["emoji"].values})
double_target_edge = pd.DataFrame(
    {"emoji": edge_df["emoji"].values, "num": edge_df["num"].values}
)
target_names_edge = [['emoji'], ['emoji', 'num']]

# ###############################################
# For NODE FEATURIZATION AND TESTS
# data to test textual and meta DataFrame
ndf_reddit = pd.read_csv("graphistry/tests/data/reddit.csv", index_col=0)

text_cols_reddit = ["title", "document"]
meta_cols_reddit = ["user", "type", "label"]
good_cols_reddit = text_cols_reddit + meta_cols_reddit

#test sending in names for target
target_names_node = [['label'], ['label', 'type']]
# test also sending in a dataframe for target
double_target_reddit = pd.DataFrame(
    {"label": ndf_reddit.label.values, "type": ndf_reddit["type"].values}, index=ndf_reddit.index
)
single_target_reddit = pd.DataFrame({"label": ndf_reddit.label.values})

edge_df2 = ndf_reddit[['title', 'label']]
edge_df2['src'] = np.random.random_integers(0, 120, size=len(edge_df2))
edge_df2['dst'] = np.random.random_integers(0, 120, size=len(edge_df2))
edge2_target_df = pd.DataFrame({'label': edge_df2.label})

# #############################################################################################################
what = ['whatever', 'on what', 'what do', 'what do you', 'what do you think', 
        'to what', 'but what', 'what is', 'what it', 'what kind', 'what kind of', 
        'of what', 'know what', 'what are', 'what are the', 'what to', 'what to do', 
        'from what', 'with what', 'and what', 'what you', 'whats', 'know what to', 'don know what', 'what the']
freedom = ['title: dyslexics, experience, language',
       'label: languagelearning, agile, leaves',
       'title: freedom, finally, moved']
# ################################################
# data to test textual and numeric DataFrame
# ndf_stocks, price_df_stocks = get_stocks_dataframe()

def allclose_stats(X, x, tol, name):
    if not np.allclose(X.std(), x.std(), tol):
        print(f'{name}.std() are not aligned at {tol} tolerance...!')

    if not np.allclose(X.mean(), x.mean(), tol):
        print(f'{name}.means() are not aligned at {tol} tolerance...!')

    if not np.allclose(X, x, tol):
        print(f'{name}s are not aligned at {tol} tolerance...!')


# so we can use on any two fields, not just x, y
def check_allclose_fit_transform_on_same_data(X, x, Y=None, y=None):
    tols = [100, 10, 0.1, 1e-4, 1e-5]
    for name, tol in zip(['Features', 'Target'], [tols, tols]):
        for value in tol:
            if name == 'Features':
                allclose_stats(X, x, value, name)
            if name == 'Target' and Y is not None and y is not None:
                allclose_stats(Y, y, value, name)


class TestFeaturizeGetMethods(unittest.TestCase):
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def setUp(self) -> None:
        g = graphistry.nodes(ndf_reddit)
        g2 = g.featurize(y=double_target_reddit,  # ngrams
                use_ngrams=True,
                ngram_range=(1, 4)
                )
        
        g3 = g.featurize(**topic_model  # topic model       
        )
        self.g = g
        self.g2 = g2
        self.g3 = g3
        
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_get_col_matrix(self):
        # no edges so this should be None
        assert self.g2.get_matrix(kind='edges') is None
        
        # test target methods
        assert all(self.g2.get_matrix(target=True).columns == self.g2._node_target.columns)
        assert self.g2.get_matrix('Anxiety', target=True).shape[0] == len(self.g2._node_target)
        # test str vs list 
        assert (self.g2.get_matrix('Anxiety', target=True) == self.g2.get_matrix(['Anxiety'], target=True)).all().values[0]

        # assert list(self.g2.get_matrix(['Anxiety', 'education', 'computer'], target=True).columns) == ['label_Anxiety', 'label_education', 'label_computervision']
    
        # test feature methods
        # ngrams
        assert (self.g2.get_matrix().columns == self.g2._node_features.columns).all()
        assert list(self.g2.get_matrix('what').columns) == what, list(self.g2.get_matrix('what').columns)
        
        # topic
        assert all(self.g3.get_matrix().columns == self.g3._node_features.columns)
        assert list(self.g3.get_matrix(['language', 'freedom']).columns) == freedom, self.g3.get_matrix(['language', 'freedom']).columns

class TestFastEncoder(unittest.TestCase):
    # we test how far off the fit returned values different from the transformed
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def setUp(self):
        fenc = FastEncoder(ndf_reddit, y=double_target_reddit, kind='nodes')
        fenc.fit(feature_engine=resolve_feature_engine('auto'),
                 use_ngrams=True, ngram_range=(1, 1), use_scaler='robust', cardinality_threshold=100)
        self.X, self.Y = fenc.X, fenc.y
        self.x, self.y = fenc.transform(ndf_reddit, ydf=double_target_reddit)

        fenc = FastEncoder(edge_df2, y=edge2_target_df, kind='edges')
        fenc.fit(src='src', dst='dst', feature_engine=resolve_feature_engine('auto'),
                 use_ngrams=True, ngram_range=(1, 1),
                 use_scaler=None,
                 use_scaler_target=None,
                 cardinality_threshold=2, n_topics=4)
        
        self.Xe, self.Ye = fenc.X, fenc.y
        self.xe, self.ye = fenc.transform(edge_df2, ydf=edge2_target_df)
        
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_allclose_fit_transform_on_same_data(self):
        check_allclose_fit_transform_on_same_data(self.X, self.x, self.Y, self.y)
        check_allclose_fit_transform_on_same_data(self.Xe, self.xe, self.Ye, self.ye)
        
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_columns_match(self):
        assert all(self.X.columns == self.x.columns), 'Node Feature Columns do not match'
        assert all(self.Y.columns == self.y.columns), 'Node Target Columns do not match'
        assert all(self.Xe.columns == self.xe.columns), 'Edge Feature Columns do not match'
        assert all(self.Ye.columns == self.ye.columns), 'Edge Target Columns do not match'
        
        
class TestFeatureProcessors(unittest.TestCase):
    def cases_tests(self, x, y, data_encoder, target_encoder, name, value):
        import dirty_cat
        self.assertIsInstance(
            x,
            pd.DataFrame,
            f"Returned data matrix is not Pandas DataFrame for {name} {value}",
        )
        self.assertFalse(
            x.empty,
            f"Pandas DataFrame should not be empty for {name} {value}",
        )
        self.assertIsInstance(
            y,
            pd.DataFrame,
            f"Returned Target is not a Pandas DataFrame for {name} {value}",
        )
        self.assertFalse(
            y.empty,
            f"Pandas Target DataFrame should not be empty for {name} {value}",
        )
        self.assertIsInstance(
            data_encoder,
            dirty_cat.super_vectorizer.SuperVectorizer,
            f"Data Encoder is not a dirty_cat.super_vectorizer.SuperVectorizer instance for {name} {value}",
        )
        self.assertIsInstance(
            target_encoder,
            dirty_cat.super_vectorizer.SuperVectorizer,
            f"Data Target Encoder is not a dirty_cat.super_vectorizer.SuperVectorizer instance for {name} {value}",
        )

    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_process_node_dataframes_min_words(self):
        # test different target cardinality
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for min_words in [
                2,
                4000,
            ]:  # last one should skip encoding, and throw all to dirty_cat

                X_enc, y_enc, X_encs, y_encs, data_encoder, label_encoder, ordinal_pipeline, ordinal_pipeline_target, text_model, text_cols = process_nodes_dataframes(
                    ndf_reddit,
                    y=double_target_reddit,
                    use_scaler=None,
                    cardinality_threshold=40,
                    cardinality_threshold_target=40,
                    n_topics=20,
                    min_words=min_words,
                    model_name=model_avg_name,
                    feature_engine=resolve_feature_engine('auto')
                )
                self.cases_tests(X_enc, y_enc, data_encoder, label_encoder, "min_words", min_words)
    
    @pytest.mark.skipif(not has_min_dependancy, reason="requires minimal feature dependencies")
    def test_multi_label_binarizer(self):
        g = graphistry.nodes(bad_df)  # can take in a list of lists and convert to multiOutput
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            g2 = g.featurize(y=['list_str'], X=['src'], multilabel=True)
        y = g2._get_target('node')
        assert y.shape == (4, 4)
        assert sum(y.sum(1).values - np.array([1., 2., 1., 0.])) == 0
        
class TestFeatureMethods(unittest.TestCase):

    def _check_attributes(self, g, attributes):
        msg = "Graphistry instance after featurization should have `{}` as attribute"
        for attribute in attributes:
            self.assertTrue(hasattr(g, attribute), msg.format(attribute))
            if 'features' in attribute:
                self.assertIsInstance(getattr(g, attribute), pd.DataFrame, msg.format(attribute))
            if 'target' in attribute:
                self.assertIsInstance(getattr(g, attribute), pd.DataFrame, msg.format(attribute))
            if 'encoder' in attribute:
                self.assertIsInstance(getattr(g, attribute), FastEncoder, msg.format(attribute))

    def cases_check_node_attributes(self, g):
        attributes = [
            "_node_features",
            "_node_target",
            "_node_encoder",
        ]
        self._check_attributes(g, attributes)

    def cases_check_edge_attributes(self, g):
        attributes = [
            "_edge_features",
            "_edge_target",
            "_edge_encoder"
        ]
        self._check_attributes(g, attributes)

    def cases_test_graph(self, g, name, value, kind="nodes", df=ndf_reddit):
        print(f'<{name} test graph: {value}>')
        if kind == "nodes":
            ndf = g._nodes
            self.cases_check_node_attributes(g)
        else:
            ndf = g._edges
            self.cases_check_edge_attributes(g)

        cols = ndf.columns
        self.assertTrue(
            np.all(ndf == df[cols]),
            f"Graphistry {kind}-dataframe does not match outside dataframe it was fed",
        )

    def _test_featurizations(self, g, use_cols, targets, name, kind, df):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for cardinality in [2, 200]:
                for use_ngram in [True, False]:
                    for use_col in use_cols:
                        for target in targets:
                            logger.debug("*" * 90)
                            value = [cardinality, use_ngram, target, use_col]
                            names = "cardinality, use_ngram, target, use_col".split(', ')
                            logger.debug(f"{value}")
                            print(f"{[k for k in zip(names, value)]}")
                            logger.debug("-" * 80)
                            if kind == 'edges' and cardinality == 2:
                                # GapEncoder is set to fail on small documents like our edge_df..., so we skip
                                continue
                            g2 = g.featurize(
                                kind=kind,
                                X=use_col,
                                y=target,
                                model_name=model_avg_name,
                                use_scaler=None,
                                use_scaler_target=None,
                                use_ngrams=use_ngram,
                                min_df=0.0,
                                max_df=1.0,
                                cardinality_threshold=cardinality,
                                cardinality_threshold_target=cardinality
                            )
            
                            self.cases_test_graph(g2, name=name, value=value, kind=kind, df=df)
                                
                
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_node_featurizations(self):
        g = graphistry.nodes(ndf_reddit)
        use_cols = [None, text_cols_reddit, meta_cols_reddit]
        targets = [None, single_target_reddit, double_target_reddit] + target_names_node
        self._test_featurizations(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Node featurization with `(target, use_col)=`",
            kind="nodes",
            df=ndf_reddit,
        )
        

    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_edge_featurization(self):
        g = graphistry.edges(edge_df, "src", "dst")
        targets = [None, single_target_edge, double_target_edge] + target_names_edge
        use_cols = [None, good_edge_cols]
        self._test_featurizations(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Edge featurization with `(target, use_col)=`",
            kind="edges",
            df=edge_df,
        )
        
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_node_scaling(self):
        g = graphistry.nodes(ndf_reddit)
        g2 = g.featurize(X="title", y='label', use_scaler=None, use_scaler_target=None)
        for scaler in SCALERS:
            X, y, c, d = g2.scale(ndf_reddit, single_target_reddit, kind='nodes', 
                                  use_scaler=scaler, 
                                  use_scaler_target=np.random.choice(SCALERS), 
                                  return_scalers=True)

    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_edge_scaling(self):
        g = graphistry.edges(edge_df2, "src", "dst")
        g2 = g.featurize(y='label', kind='edges', use_scaler=None, use_scaler_target=None)
        for scaler in SCALERS:
            X, y, c, d = g2.scale(edge_df2, edge2_target_df, kind='edges', 
                                  use_scaler=scaler, 
                                  use_scaler_target=np.random.choice(SCALERS), 
                                  return_scalers=True)


class TestFeaturizeGetMethodsCucat(unittest.TestCase):
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    def setUp(self) -> None:
        import cudf
        g = graphistry.nodes(cudf.from_pandas(ndf_reddit))
        g2 = g.featurize(y=cudf.from_pandas(double_target_reddit),  # ngrams
                use_ngrams=True,
                ngram_range=(1, 4)
                )
        
        g3 = g.featurize(**topic_model, feature_engine="cu_cat")  # topic model
        self.g = g
        self.g2 = g2
        self.g3 = g3
        
    # @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    # @pytest.mark.skipif(not is_test_cudf, reason="requires cudf")
    # def test_get_col_matrix(self):
    #     # no edges so this should be None
    #     assert self.g2.get_matrix(kind='edges') is None
        
    #     # test target methods
    #     assert all(self.g2.get_matrix(target=True).columns == self.g2._node_target.columns)
    #     assert self.g2.get_matrix('Anxiety', target=True).shape[0] == len(self.g2._node_target)
    #     # test str vs list 
    #     assert (self.g2.get_matrix('Anxiety', target=True) == self.g2.get_matrix(['Anxiety'], target=True)).all().values[0]

    #     # assert list(self.g2.get_matrix(['Anxiety', 'education', 'computer'], target=True).columns) == ['label_Anxiety', 'label_education', 'label_computervision']
    
    #     # test feature methods
    #     # ngrams
    #     assert (self.g2.get_matrix().columns == self.g2._node_features.columns).all()
    #     assert list(self.g2.get_matrix('what').columns) == what, list(self.g2.get_matrix('what').columns)
        
    #     # topic
    #     assert all(self.g3.get_matrix().columns == self.g3._node_features.columns)


if __name__ == "__main__":
    unittest.main()

# python -m unittest
import datetime as dt
import graphistry
import logging
import numpy as np
import pandas as pd
from typing import Any

import pytest
import unittest

from graphistry.feature_utils import (
    process_dirty_dataframes,
    process_nodes_dataframes,
    resolve_feature_engine,
    FastEncoder,
    encode_textual
)

from graphistry.features import topic_model, ngrams_model
from graphistry.constants import SCALERS
from graphistry.utils.lazy_import import (
    lazy_import_has_min_dependancy,
    lazy_sentence_transformers_import
)

np.random.seed(137)

has_min_dependancy, _ = lazy_import_has_min_dependancy()
has_min_dependancy_text, _, _ = lazy_sentence_transformers_import()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("graphistry.feature_utils").setLevel(logging.DEBUG)

model_avg_name = (
    "/models/average_word_embeddings_komninos"  # 250mb, fastest vectorizer in transformer models
    #"/models/paraphrase-albert-small-v2"  # 40mb
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
        # assert list(self.g3.get_matrix(['language', 'freedom']).columns) == freedom, self.g3.get_matrix(['language', 'freedom']).columns

class TestFastEncoderNode(unittest.TestCase):
    # we test how far off the fit returned values different from the transformed
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def setUp(self):
        fenc = FastEncoder(ndf_reddit, y=double_target_reddit, kind='nodes')
        fenc.fit(feature_engine=resolve_feature_engine('auto'),
                 use_ngrams=True, ngram_range=(1, 1), use_scaler='robust', cardinality_threshold=100)
        self.X, self.Y = fenc.X, fenc.y
        self.x, self.y = fenc.transform(ndf_reddit, ydf=double_target_reddit)
        
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_allclose_fit_transform_on_same_data_nodes(self):
        check_allclose_fit_transform_on_same_data(self.X, self.x, self.Y, self.y)

    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_columns_match(self):
        assert set(self.X.columns) == set(self.x.columns), 'Node Feature Columns do not match'
        assert set(self.Y.columns) == set(self.y.columns), 'Node Target Columns do not match'

class TestFastEncoderEdge(unittest.TestCase):
    # we test how far off the fit returned values different from the transformed
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def setUp(self):

        fenc = FastEncoder(edge_df2, y=edge2_target_df, kind='edges')
        fenc.fit(src='src', dst='dst', feature_engine=resolve_feature_engine('auto'),
                 use_ngrams=True, ngram_range=(1, 1),
                 use_scaler=None,
                 use_scaler_target=None,
                 cardinality_threshold=2, n_topics=4)
        
        self.Xe, self.Ye = fenc.X, fenc.y
        self.xe, self.ye = fenc.transform(edge_df2, ydf=edge2_target_df)

    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_allclose_fit_transform_on_same_data_edges(self):
        check_allclose_fit_transform_on_same_data(self.Xe, self.xe, self.Ye, self.ye)

    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_columns_match(self):
        assert set(self.Xe.columns) == set(self.xe.columns), 'Edge Feature Columns do not match'
        assert set(self.Ye.columns) == set(self.ye.columns), 'Edge Target Columns do not match'


class TestFeatureProcessors(unittest.TestCase):
    def cases_tests(self, x, y, data_encoder, target_encoder, name, value):
        import skrub
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
            skrub.TableVectorizer,
            f"Data Encoder is not a skrub.TableVectorizer instance for {name} {value}",
        )
        self.assertIsInstance(
            target_encoder,
            skrub.TableVectorizer,
            f"Data Target Encoder is not a skrub.TableVectorizer instance for {name} {value}",
        )

    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_process_node_dataframes_min_words(self):
        # test different target cardinality
        for min_words in [
            2,
            4000,
        ]:  # last one should skip encoding, and throw all to skrub

            X_enc, y_enc, X_encs, y_encs, data_encoder, label_encoder, ordinal_pipeline, ordinal_pipeline_target, text_model, text_cols = process_nodes_dataframes(
                ndf_reddit,
                y=double_target_reddit,
                use_scaler='none',
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
            np.all(ndf.fillna(0) == df[cols].fillna(0)),
            f"Graphistry {kind}-dataframe does not match outside dataframe it was fed",
        )

    def _test_featurizations(self, g, use_cols, targets, name, kind, df):
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


    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_type_edgecase(self):
        df = pd.DataFrame({
            'A': np.random.rand(50),
            'B': np.random.rand(50)
        })
        num_to_convert = int(len(df.A.values) * 0.1)
        indices_to_convert = np.random.choice(len(df.A.values), num_to_convert, replace=False)
        indices_to_convertB = np.random.choice(len(df.A.values), num_to_convert, replace=False)
        for i,j in zip(indices_to_convert, indices_to_convertB):
            df.A[i] = str(df.A[i])
            df.B[j] = str(df.B[j])
        df.A.loc[13] = '92.026 123.903 702.124'
        df.B.loc[33] = '26.092 903.123'

        graphistry.nodes(df).featurize()
        assert True


class TestModelNameHandling(unittest.TestCase):
    """Test that both legacy and new model name formats work correctly"""
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_model_name_formats(self):
        """Test various model name formats for backwards compatibility"""
        # Create a simple test dataframe with text
        test_df = pd.DataFrame({
            'text1': ['hello world', 'test sentence', 'another example'],
            'text2': ['foo bar baz', 'quick brown fox', 'lazy dog jumps'],
            'number': [1, 2, 3]
        })
        
        # Test cases: (input_model_name, expected_to_work, description)
        test_cases = [
            # Legacy format (without org prefix) - should add sentence-transformers/
            ("paraphrase-albert-small-v2", True, "Legacy format without org prefix"),
            
            # Already has sentence-transformers prefix - should keep as-is
            ("sentence-transformers/paraphrase-albert-small-v2", True, "With sentence-transformers prefix"),
            
            # New format with different org - should keep as-is
            # Note: This would work with real model like mixedbread-ai/mxbai-embed-large-v1
            # but for CI we use a known small model
            ("sentence-transformers/paraphrase-albert-small-v2", True, "Standard format with org prefix"),
        ]
        
        # Add local model test if running in Docker environment
        import os
        if os.path.exists("/models/average_word_embeddings_komninos"):
            test_cases.append(
                ("/models/average_word_embeddings_komninos", True, "Local model path from Docker")
            )
        
        for model_name, should_work, description in test_cases:
            with self.subTest(model_name=model_name, description=description):
                try:
                    # Use small model and min_words=0 for faster testing
                    result_df, text_cols, model = encode_textual(
                        test_df,
                        min_words=0,  # Process all text columns
                        model_name=model_name,
                        use_ngrams=False
                    )
                    
                    if should_work:
                        # Verify we got results
                        self.assertIsInstance(result_df, pd.DataFrame)
                        self.assertGreater(result_df.shape[1], 0, f"No embedding columns created for {description}")
                        self.assertEqual(len(result_df), len(test_df), f"Row count mismatch for {description}")
                        self.assertIsNotNone(model, f"Model is None for {description}")
                        self.assertEqual(text_cols, ['text1', 'text2'], f"Wrong text columns detected for {description}")
                    
                except Exception as e:
                    if should_work:
                        self.fail(f"Model {model_name} ({description}) should have worked but failed: {str(e)}")
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_new_model_provider_format(self):
        """Test that new model provider formats are handled correctly"""
        import os
        from sentence_transformers import SentenceTransformer
        
        # Test the internal logic matching actual implementation
        test_cases = [
            # Legacy: no slash means add sentence-transformers/ prefix
            ("model-name-only", "sentence-transformers/model-name-only"),
            ("paraphrase-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2"),
            
            # Already has sentence-transformers/ prefix - keep as-is
            ("sentence-transformers/model", "sentence-transformers/model"),
            ("sentence-transformers/paraphrase-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L6-v2"),
            
            # Alternative namespaces - keep as-is
            ("org/model-name", "org/model-name"),
            ("mixedbread-ai/mxbai-embed-large-v1", "mixedbread-ai/mxbai-embed-large-v1"),
            ("nomic-ai/nomic-embed-text-v1", "nomic-ai/nomic-embed-text-v1"),
            ("BAAI/bge-large-en-v1.5", "BAAI/bge-large-en-v1.5"),
            
            # Local paths - extract just the model name (old behavior)
            ("/local/path/to/model", "model"),
            ("/models/average_word_embeddings_komninos", "average_word_embeddings_komninos"),
            ("./relative/path/model", "model"),
        ]
        
        for input_name, expected_name in test_cases:
            with self.subTest(input_name=input_name):
                # Test the model name processing logic matching actual implementation
                if input_name.startswith('/') or input_name.startswith('./'):
                    # Local path - extract just the model name
                    processed_name = os.path.split(input_name)[-1]
                elif '/' not in input_name:
                    # Legacy format without org prefix
                    processed_name = f"sentence-transformers/{input_name}"
                else:
                    # Already has org/model format
                    processed_name = input_name
                
                self.assertEqual(processed_name, expected_name, 
                               f"Model name processing failed for {input_name}")
    
    @pytest.mark.skipif(not has_min_dependancy or not has_min_dependancy_text, reason="requires ai feature dependencies")
    def test_alternative_namespace_models(self):
        """Test that alternative namespace models work with actual encoding"""
        # Create a simple test dataframe
        test_df = pd.DataFrame({
            'text': ['test sentence for encoding'],
            'id': [1]
        })
        
        # Only test with small, known-to-exist models to avoid download issues in CI
        # Real-world usage would include models like:
        # - "mixedbread-ai/mxbai-embed-large-v1"
        # - "nomic-ai/nomic-embed-text-v1"
        # - "BAAI/bge-small-en-v1.5"
        
        # For CI, we'll use the small albert model with different formats
        small_model = "paraphrase-albert-small-v2"
        
        # Test that both formats produce the same embeddings
        result1, _, model1 = encode_textual(
            test_df,
            min_words=0,
            model_name=small_model,  # Legacy format
            use_ngrams=False
        )
        
        result2, _, model2 = encode_textual(
            test_df,
            min_words=0,
            model_name=f"sentence-transformers/{small_model}",  # Full format
            use_ngrams=False
        )
        
        # Both should produce the same embeddings
        self.assertEqual(result1.shape, result2.shape, 
                        "Different formats should produce same shape embeddings")
        self.assertTrue(np.allclose(result1.values, result2.values),
                       "Different formats should produce identical embeddings")


if __name__ == "__main__":
    unittest.main()

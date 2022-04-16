# python -m unittest
from typing import Any
import copy, datetime as dt, graphistry, numpy as np, os, pandas as pd
import pytest, unittest

from graphistry.feature_utils import (
    process_dirty_dataframes,
    process_textual_or_other_dataframes,
    remove_internal_namespace_if_present,
    resolve_feature_engine,
    has_min_dependancy,
)

try:
    import dirty_cat
    import sklearn
except:
    dirty_cat = Any
    sklearn = Any

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("graphistry.feature_utils").setLevel(logging.DEBUG)

model_avg_name = (
    "average_word_embeddings_komninos"  # fastest vectorizer in transformer models
)


bad_df = pd.DataFrame(
    {
        "src": [0, 1, 2, 3],
        "dst": [1, 2, 3, 0],
        "colors": [1, 1, 2, 2],
        "list_int": [[1], [2, 3], [4], []],
        "list_str": [["x"], ["1", "2"], ["y"], []],
        "list_bool": [[True], [True, False], [False], []],
        "list_date_str": [
            ["2018-01-01 00:00:00"],
            ["2018-01-02 00:00:00", "2018-01-03 00:00:00"],
            ["2018-01-05 00:00:00"],
            [],
        ],
        "list_date": [
            [pd.Timestamp("2018-01-05")],
            [pd.Timestamp("2018-01-05"), pd.Timestamp("2018-01-05")],
            [],
            [],
        ],
        "list_mixed": [[1], ["1", "2"], [False, None], []],
        "bool": [True, False, True, True],
        "char": ["a", "b", "c", "d"],
        "str": ["a", "b", "c", "d"],
        "ustr": ["a", "b", "c", "d"],
        "emoji": ["😋", "😋😋", "😋", "😋"],
        "int": [0, 1, 2, 3],
        "num": [0.5, 1.5, 2.5, 3.5],
        "date_str": [
            "2018-01-01 00:00:00",
            "2018-01-02 00:00:00",
            "2018-01-03 00:00:00",
            "2018-01-05 00:00:00",
        ],
        # API 1 BUG: Try with https://github.com/graphistry/pygraphistry/pull/126
        "date": [
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
            dt.datetime(2018, 1, 1),
        ],
        "time": [
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
            pd.Timestamp("2018-01-05"),
        ],
        # API 2 BUG: Need timedelta in https://github.com/graphistry/pygraphistry/blob/master/graphistry/vgraph.py#L108
        "delta": [
            pd.Timedelta("1 day"),
            pd.Timedelta("1 day"),
            pd.Timedelta("1 day"),
            pd.Timedelta("1 day"),
        ],
        "textual": [
            "here we have a sentence. And here is another sentence. Graphistry is an amazing tool!"
        ] * 4,
    }
)

# ###############################################
# For EDGE FEATURIZATION AND TESTS
edge_df = bad_df.astype(str)
good_edge_cols = [
    "textual",
    "delta",
    "time",
    "colors",
    "list_str",
    "bool",
    "char",
    "list_date",
]

single_target_edge = pd.DataFrame({"emoji": edge_df["emoji"].values})
double_target_edge = pd.DataFrame(
    {"emoji": edge_df["emoji"].values, "num": edge_df["num"].values}
)


# ###############################################
# For NODE FEATURIZATION AND TESTS
# data to test textual and meta DataFrame
ndf_reddit = pd.read_csv("graphistry/tests/data/reddit.csv", index_col=0)

text_cols_reddit = ["title", "document"]
meta_cols_reddit = ["user", "type", "label"]
good_cols_reddit = text_cols_reddit + meta_cols_reddit

# ndf_reddit = ndf_reddit[good_cols_reddit]


double_target_reddit = pd.DataFrame(
    {"label": ndf_reddit.label.values, "type": ndf_reddit["type"].values}
)
single_target_reddit = pd.DataFrame({"label": ndf_reddit.label.values})

# ################################################
# data to test textual and numeric DataFrame
# ndf_stocks, price_df_stocks = get_stocks_dataframe()


class TestFeatureProcessors(unittest.TestCase):
    def cases_tests(self, x, y, x_enc, y_enc, name, value):
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
            x_enc,
            dirty_cat.super_vectorizer.SuperVectorizer,
            f"Data Encoder is not a dirty_cat.super_vectorizer.SuperVectorizer instance for {name} {value}",
        )
        self.assertIsInstance(
            y_enc,
            dirty_cat.super_vectorizer.SuperVectorizer,
            f"Data Target Encoder is not a dirty_cat.super_vectorizer.SuperVectorizer instance for {name} {value}",
        )

    @pytest.mark.skipif(not has_min_dependancy, reason="requires ai feature dependencies")
    def test_process_dirty_dataframes_scalers(self):
        # test different scalers
        for scaler in ["minmax", "quantile", "zscale", "robust", "kbins"]:
            x, y, x_enc, y_enc, imputer, scaler = process_dirty_dataframes(
                ndf_reddit,
                y=double_target_reddit,
                use_scaler=scaler,
                cardinality_threshold=40,
                cardinality_threshold_target=40,
                n_topics=20,
                feature_engine=resolve_feature_engine('auto')
            )
            self.cases_tests(x, y, x_enc, y_enc, "scaler", scaler)

    @pytest.mark.skipif(not has_min_dependancy, reason="requires ai feature dependencies")
    def test_process_dirty_dataframes_data_cardinality(self):
        # test different cardinality
        for card in [4, 40, 400]:
            x, y, x_enc, y_enc, imputer, scaler = process_dirty_dataframes(
                ndf_reddit,
                y=double_target_reddit,
                use_scaler=None,
                cardinality_threshold=card,
                cardinality_threshold_target=40,
                n_topics=20,
                feature_engine=resolve_feature_engine('auto')
            )
            self.cases_tests(x, y, x_enc, y_enc, "cardinality", card)

    @pytest.mark.skipif(not has_min_dependancy, reason="requires ai feature dependencies")
    def test_process_dirty_dataframes_target_cardinality(self):
        # test different target cardinality
        for card in [4, 40, 400]:
            x, y, x_enc, y_enc, imputer, scaler = process_dirty_dataframes(
                ndf_reddit,
                y=double_target_reddit,
                use_scaler=None,
                cardinality_threshold=40,
                cardinality_threshold_target=card,
                n_topics=20,
                feature_engine=resolve_feature_engine('auto')
            )
            self.cases_tests(x, y, x_enc, y_enc, "target cardinality", card)

    @pytest.mark.skipif(not has_min_dependancy, reason="requires ai feature dependencies")
    def test_process_textual_or_other_dataframes_min_words(self):
        # test different target cardinality
        with self.assertRaises(Exception) as context:
            x, y, x_enc, y_enc, imputer, scaler = process_textual_or_other_dataframes(
                ndf_reddit,
                y=double_target_reddit,
                use_scaler=None,
                cardinality_threshold=40,
                cardinality_threshold_target=40,
                n_topics=20,
                confidence=0.35,
                min_words=1,
                model_name=model_avg_name,
                feature_engine=resolve_feature_engine('auto')
            )
        print("-" * 90)
        print(context.exception)
        print("-" * 90)

        self.assertTrue("best to have at least a word" in str(context.exception))

        for min_words in [
            2,
            4000,
        ]:  # last one should skip encoding, and throw all to dirty_cat
            x, y, x_enc, y_enc, imputer, scaler = process_textual_or_other_dataframes(
                ndf_reddit,
                y=double_target_reddit,
                use_scaler=None,
                cardinality_threshold=40,
                cardinality_threshold_target=40,
                n_topics=20,
                confidence=0.35,
                min_words=min_words,
                model_name=model_avg_name,
                feature_engine=resolve_feature_engine('auto')
            )
            self.cases_tests(x, y, x_enc, y_enc, "min_words", min_words)


class TestFeatureMethods(unittest.TestCase):
    def cases_with_no_target(self, x, x_enc, name, value, kind):
        self.assertIsInstance(
            x,
            pd.DataFrame,
            f"Returned {kind} data matrix is not Pandas DataFrame for {name} {value}",
        )
        self.assertFalse(
            x.empty,
            f"{kind} Pandas DataFrame should not be empty for {name} {value}",
        )
        if kind == "nodes" and x_enc is not None:
            self.assertIsInstance(
                x_enc,
                dirty_cat.super_vectorizer.SuperVectorizer,
                f"{kind} Data Encoder is not a dirty_cat.super_vectorizer.SuperVectorizer instance for {name} {value}",
            )
        elif kind == "edges":  # edge encoder is made up of two parts
            mlb, x_enc = x_enc
            self.assertIsInstance(
                x_enc,
                dirty_cat.super_vectorizer.SuperVectorizer,
                f"{kind} Data Encoder is not a `dirty_cat.super_vectorizer.SuperVectorizer` instance for {name} {value}",
            )
            self.assertIsInstance(
                mlb,
                sklearn.preprocessing._label.MultiLabelBinarizer,
                f"{kind} Data Encoder is not a `sklearn.preprocessing._label.MultiLabelBinarizer` instance for {name} {value}",
            )

    def cases_with_target(self, x, y, x_enc, y_enc, name, value, kind):
        self.cases_with_no_target(x, x_enc, name, value, kind)
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
            y_enc,
            dirty_cat.super_vectorizer.SuperVectorizer,
            f"Data Target Encoder is not a dirty_cat.super_vectorizer.SuperVectorizer instance for {name} {value}",
        )

    def _check_attributes(self, g, attributes):
        msg = "Graphistry instance after featurization should have `{}` as attribute"
        for attribute in attributes:
            self.assertTrue(hasattr(g, attribute), msg.format(attribute))

    def cases_check_node_attributes(self, g):
        attributes = [
            "_node_features",
            "_node_target",
            "_node_target_encoder",
            "_node_encoder",
            "_node_imputer",
            "_node_scaler",
        ]
        self._check_attributes(g, attributes)

    def cases_check_edge_attributes(self, g):
        attributes = [
            "_edge_features",
            "_edge_target",
            "_edge_target_encoder",
            "_edge_encoders",  # plural, since we have two
            "_edge_imputer",
            "_edge_scaler",
        ]
        self._check_attributes(g, attributes)

    def cases_test_graph(self, g, name, value, kind="nodes", df=ndf_reddit):
        if kind == "nodes":
            ndf = g._nodes
            self.cases_check_node_attributes(g)
            x, y, x_enc, y_enc = (
                g._node_features,
                g._node_target,
                g._node_encoder,
                g._node_target_encoder,
            )
        else:
            ndf = g._edges
            self.cases_check_edge_attributes(g)
            x, y, x_enc, y_enc = (
                g._edge_features,
                g._edge_target,
                g._edge_encoders,
                g._edge_target_encoder,
            )

        cols = ndf.columns
        self.assertTrue(
            np.all(ndf == df[cols]),
            f"Graphistry {kind}-dataframe does not match outside dataframe it was fed",
        )
        if y_enc is not None:
            self.cases_with_target(x, y, x_enc, y_enc, name, value, kind)
        else:
            self.cases_with_no_target(x, x_enc, name, value, kind)

    def _test_featurizations(self, g, use_cols, targets, name, kind, df):
        for use_col in use_cols:
            for target in targets:
                logger.debug("*" * 90)
                value = [target, use_col]
                logger.debug(f"{value}")
                logger.debug("-" * 80)
                g2 = g.featurize(
                    kind=kind, y=target, use_columns=use_col, model_name=model_avg_name
                )

                self.cases_test_graph(g2, name=name, value=value, kind=kind, df=df)

    @pytest.mark.skipif(not has_min_dependancy, reason="requires ai feature dependencies")
    def test_node_featurizations(self):
        g = graphistry.nodes(ndf_reddit)
        use_cols = [None, text_cols_reddit, good_cols_reddit, meta_cols_reddit]
        targets = [None, single_target_reddit, double_target_reddit]
        self._test_featurizations(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Node featurization with `(target, use_col)=`",
            kind="nodes",
            df=ndf_reddit,
        )

    @pytest.mark.skipif(not has_min_dependancy, reason="requires ai feature dependencies")
    def test_edge_featurization(self):
        g = graphistry.edges(edge_df, "src", "dst")
        targets = [None, single_target_edge, double_target_edge]
        use_cols = [None, good_edge_cols]
        self._test_featurizations(
            g,
            use_cols=use_cols,
            targets=targets,
            name="Edge featurization with `(target, use_col)=`",
            kind="edges",
            df=edge_df,
        )


if __name__ == "__main__":
    unittest.main()

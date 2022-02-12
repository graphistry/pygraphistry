# python -m unittest

import unittest
import copy, datetime as dt, graphistry, os, pandas as pd

from graphistry.feature_utils import *
from graphistry.dgl_utils import *

from data import get_reddit_dataframe, get_stocks_dataframe

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
        "ustr": [u"a", u"b", u"c", u"d"],
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
            "here we have a sentence. And here is another sentence. Graphistry is an amazing tool"
        ]
        * 4,
    }
)

edge_df = bad_df.astype(str)
# node_df = pd.DataFrame()

# ###############################################
# NODE FEATURIZATION AND TESTS
# data to test textual and meta DataFrame
ndf_reddit = get_reddit_dataframe(nrows=100, min_doc_length=40)
ndf_reddit["n"] = range(len(ndf_reddit))  # add node label

text_cols_reddit = ["title", "document"]
meta_cols_reddit = ["user", "type", "label"]
good_cols_reddit = text_cols_reddit + meta_cols_reddit

double_target_reddit = pd.DataFrame(
    {"label": ndf_reddit.label.values, "type": ndf_reddit["type"].values}
)
single_target_reddit = pd.DataFrame({"label": ndf_reddit.label.values})

# ################################################
# data to test textual and numeric DataFrame

ndf_stocks, price_df_stocks = get_stocks_dataframe()


class TestFeatureProcessors(unittest.TestCase):
    def setUp(self):
        x, y, x_enc, y_enc = process_dirty_dataframes(
            ndf_reddit, y=double_target_reddit, z_scale=False
        )


class TestFeatureMethods(unittest.TestCase):
    def setUp(self):
        g = graphistry.nodes(ndf_reddit, "n")

    def test_node_featurization(self):
        g = graphistry.nodes(ndf_reddit, "n")
        g2 = g.featurize(kind="nodes")
        ## more here

    def test_node_featurization_with_use_columns(self):
        g = graphistry.nodes(ndf_reddit, "n")
        g2 = g.featurize(kind="nodes", use_columns=text_cols_reddit)
        ## more here

        g3 = g.featurize(kind="nodes", use_columns=meta_cols_reddit)
        ## more here

        g4 = g.featurize(kind="nodes", use_columns=good_cols_reddit)
        ## more here

    def test_node_featurization_with_y_inputs(self):
        g = graphistry.nodes(ndf_reddit, "n")

        g2 = g.featurize(kind="nodes", y=single_target_reddit)
        X, y = g2.node_features, g2.node_target
        # checks

        g3 = g.featurize(kind="nodes", y=double_target_reddit)
        X, y = g3.node_features, g2.node_target
        # checks

    def test_node_featurization_with_y_inputs_and_use_cols(self):
        g = graphistry.nodes(ndf_reddit, "n")
        g2 = g.featurize(kind="nodes", y=single_target_reddit, use_columns=text_cols_reddit)
        X, y = g2.node_features, g2.node_target
        # checks
        g2 = g.featurize(kind="nodes", y=single_target_reddit, use_columns=meta_cols_reddit)
        X, y = g2.node_features, g2.node_target

        g2 = g.featurize(kind="nodes", y=single_target_reddit, use_columns=good_cols_reddit)
        X, y = g2.node_features, g2.node_target

        g3 = g.featurize(kind="nodes", y=double_target_reddit, use_columns=text_cols_reddit)
        X, y = g3.node_features, g3.node_target
        # checks

        g3 = g.featurize(kind="nodes", y=double_target_reddit, use_columns=meta_cols_reddit)
        X, y = g3.node_features, g3.node_target
        # checks
        g3 = g.featurize(kind="nodes", y=double_target_reddit, use_columns=good_cols_reddit)
        X, y = g3.node_features, g3.node_target
        # checks

    def test_edge_featurization(self):
        pass


class TestUMAPMethods(unittest.TestCase):

    def test_node_umap(self):
        g = graphistry.nodes(ndf_reddit, 'n')
        g2= g.umap(kind="nodes")
        X, y = g2.node_features, g2.node_target
        emb = g2.xy_nodes

        g3 = g2.umap(kind="nodes", scale=0.5, n_neighbors=20)
        # should reused
        X2, y2 = g3.node_features, g3.node_target
        emb2 = g3.xy_nodes

        self.assertEqual(X, X2)
        self.assertEqual(y, y2)
        self.assertNotAlmostEquals(emb, emb2)
        

    def test_node_umap_with_y_inputs(self):
        g = graphistry.nodes(ndf_reddit, 'n')
        g2= g.umap(kind="nodes", y=double_target_reddit)
        X, y = g2.node_features, g2.node_target
        
        
    def test_node_umap_with_y_inputs_and_use_columns(self):
        g = graphistry.nodes(ndf_reddit, 'n')
        g2= g.umap(kind="nodes", y=double_target_reddit, use_columns=good_cols_reddit)
        X, y = g2.node_features, g2.node_target


if __name__ == "__main__":
    unittest.main()

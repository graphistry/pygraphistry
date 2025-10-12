import pytest
import unittest

import graphistry
import logging
import numpy as np
import pandas as pd
from graphistry.feature_utils import remove_internal_namespace_if_present
from graphistry.tests.test_feature_utils import (
    ndf_reddit,
    edge_df,
)
from graphistry.utils.lazy_import import (
    lazy_umap_import,
    lazy_import_has_min_dependancy
)

has_dependancy, _ = lazy_import_has_min_dependancy()
has_umap, _, _ = lazy_umap_import()

logger = logging.getLogger(__name__)

class TestTextSearch(unittest.TestCase):
    # check to see that .fit and transform gives similar embeddings on same data
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def setUp(self):

        g = graphistry.nodes(ndf_reddit)
        g_with_edges = graphistry.nodes(edge_df, 'src').edges(edge_df, 'src', 'dst')

        g2 = g.umap(X=['title', 'document'],
                    use_ngrams=True,
                    ngram_range=(1, 2))

        g3 = g.umap(X=['title'],
                    use_ngrams=False,
                    min_words=2)

        # Test case with target columns (y parameter) - reproduces issue #629
        g5 = g.umap(X=['title'],
                    y=['label'],
                    use_ngrams=True,
                    ngram_range=(1, 2),
                    min_words=0)

        # here we just featurize since edges are given
        g4 = g_with_edges.featurize(X=['textual'],
                                    use_ngrams=False,
                                    min_words=1.1
                                    )

        self.g_ngrams = g2
        self.g_emb = g3
        self.g_with_edges = g4
        self.g_with_target = g5

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_graph_ngrams(self):
        """Test search_graph with ngrams encoding"""
        res = self.g_ngrams.search_graph('How to set up DNS', thresh=100)
        assert not res._nodes.empty, f'ngrams - Results DataFrame should not be empty, found {res._nodes}'

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_graph_embedding(self):
        """Test search_graph with transformer embedding"""
        res = self.g_emb.search_graph('How to set up DNS', thresh=100)
        assert not res._nodes.empty, f'embedding - Results DataFrame should not be empty, found {res._nodes}'

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_graph_with_edges(self):
        """Test search_graph with explicit edges"""
        res = self.g_with_edges.search_graph('Wife', thresh=100)
        assert not res._nodes.empty, f'Results DataFrame should not be empty, found {res._nodes}'

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_fuzzy_with_target_columns(self):
        """Test search with fuzzy=True after umap with y parameter (issue #629)"""
        # This should not raise AssertionError about ydf
        res_df, query_vec = self.g_with_target.search('DNS setup', fuzzy=True, thresh=100, top_n=10)
        assert isinstance(res_df, pd.DataFrame), 'search should return DataFrame'
        assert query_vec is not None, 'search with fuzzy=True should return query vector'

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_graph_with_target_columns(self):
        """Test search_graph after umap with y parameter (issue #629)"""
        # This should not raise AssertionError about ydf
        res = self.g_with_target.search_graph('DNS setup', thresh=100)
        assert not res._nodes.empty, 'search_graph with target columns should return results'


if __name__ == "__main__":
    unittest.main()

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
        """Test search with fuzzy=True after umap with y parameter (issue #629)

        The key test is that search() doesn't raise AssertionError about ydf
        when the model was fit with target columns.
        """
        # This should not raise AssertionError: "ydf must be provided to transform data"
        try:
            res_df, query_vec = self.g_with_target.search('DNS setup', fuzzy=True, thresh=100, top_n=10)
            assert isinstance(res_df, pd.DataFrame), 'search should return DataFrame'
            # query_vec may be None if no text columns after filtering, which is ok
        except AssertionError as e:
            if 'ydf' in str(e) or 'transform data' in str(e):
                pytest.fail(f"Issue #629 regression: {e}")
            raise

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_graph_with_target_columns(self):
        """Test search_graph after umap with y parameter (issue #629)"""
        # This should not raise AssertionError about ydf
        res = self.g_with_target.search_graph('DNS setup', thresh=100)
        assert not res._nodes.empty, 'search_graph with target columns should return results'

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_with_multiple_target_columns(self):
        """Test search with multiple target columns (edge case for #629)"""
        g = graphistry.nodes(ndf_reddit)
        # Use multiple target columns
        g_multi = g.umap(X=['title'], y=['label', 'type'], use_ngrams=True, min_words=0)

        # Should not raise AssertionError about ydf
        try:
            res_df, query_vec = g_multi.search('DNS', fuzzy=True, thresh=100, top_n=10)
            assert isinstance(res_df, pd.DataFrame), 'search should return DataFrame'
        except AssertionError as e:
            if 'ydf' in str(e) or 'transform data' in str(e):
                pytest.fail(f"Issue #629 regression with multiple targets: {e}")
            raise

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_with_no_target_columns(self):
        """Test search without target columns (baseline case)"""
        g = graphistry.nodes(ndf_reddit)
        # No y parameter
        g_no_target = g.umap(X=['title'], use_ngrams=True, min_words=0)

        res_df, query_vec = g_no_target.search('DNS', fuzzy=True, thresh=100, top_n=10)
        assert isinstance(res_df, pd.DataFrame), 'search should return DataFrame'

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_with_featurize_and_target(self):
        """Test search with featurize() instead of umap() (edge case for #629)"""
        g = graphistry.nodes(edge_df, 'src').edges(edge_df, 'src', 'dst')
        # featurize with target columns
        # Use min_df=1, max_df=1.0 to handle small dataset (4 rows)
        g_feat = g.featurize(X=['textual'], y=['emoji'], use_ngrams=True, min_words=0, min_df=1, max_df=1.0)

        # Should not raise AssertionError about ydf
        try:
            res_df, query_vec = g_feat.search('wife', fuzzy=True, thresh=100, top_n=10)
            assert isinstance(res_df, pd.DataFrame), 'search should return DataFrame'
        except AssertionError as e:
            if 'ydf' in str(e) or 'transform data' in str(e):
                pytest.fail(f"Issue #629 regression with featurize(): {e}")
            raise

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_with_empty_results(self):
        """Test search that returns no results (edge case)"""
        g = graphistry.nodes(ndf_reddit)
        g_with_y = g.umap(X=['title'], y=['label'], use_ngrams=True, min_words=0)

        # Query that likely returns empty results
        res_df, query_vec = g_with_y.search('xyzabc123impossible', fuzzy=True, thresh=0.001, top_n=10)
        assert isinstance(res_df, pd.DataFrame), 'search should return DataFrame even if empty'

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_search_non_fuzzy_with_target(self):
        """Test non-fuzzy search with target columns (different code path)"""
        g = graphistry.nodes(ndf_reddit)
        g_with_y = g.umap(X=['title'], y=['label'], use_ngrams=True, min_words=0)

        # Non-fuzzy search uses different code path (no transform call)
        res_df, query_vec = g_with_y.search('DNS', fuzzy=False, cols=['title'], top_n=10)
        assert isinstance(res_df, pd.DataFrame), 'non-fuzzy search should return DataFrame'
        assert query_vec is None, 'non-fuzzy search should return None for query_vec'


if __name__ == "__main__":
    unittest.main()

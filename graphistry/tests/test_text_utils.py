import pytest
import unittest
import warnings

import graphistry
import logging
import numpy as np
import pandas as pd
from graphistry.feature_utils import remove_internal_namespace_if_present, assert_imported_engine as assert_imported_feature_utils
from graphistry.tests.test_feature_utils import (
    ndf_reddit,
    edge_df,
)

from graphistry.utils.dep_manager import DepManager
deps = DepManager()
has_umap = deps.umap

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class TestTextSearch(unittest.TestCase):
    # check to see that .fit and transform gives similar embeddings on same data
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def setUp(self):

        g = graphistry.nodes(ndf_reddit)
        g_with_edges = graphistry.nodes(edge_df, 'src').edges(edge_df, 'src', 'dst')
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            g2 = g.umap(X=['title', 'document'],
                        use_ngrams=True, 
                        ngram_range=(1, 2))
            
            g3 = g.umap(X=['title'],
                        use_ngrams=False, 
                        min_words=2)
            
            # here we just featurize since edges are given
            g4 = g_with_edges.featurize(X=['textual'], 
                                        use_ngrams=False, 
                                        min_words=1.1
                                        )
            
        self.g_ngrams = g2
        self.g_emb = g3
        self.g_with_edges = g4
        
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies") 
    def test_query_graph(self):
        for name, g in zip(['ngrams', 'embedding'], [self.g_ngrams, self.g_emb]):
            res = g.search_graph('How to set up DNS', thresh=100)
            assert not res._nodes.empty, f'{name}-Results DataFrame should not be empty, found {res._nodes}'
            #url = res.plot(render=False)
            #logger.info(f'{name}: {url}')
            
        res = self.g_with_edges.search_graph('Wife', thresh=100)
        assert not res._nodes.empty, f'Results DataFrame should not be empty, found {res._nodes}'
        #url = res.plot(render=False)
        #logger.info(f'With Explicit Edges: {url}')


if __name__ == "__main__":
    unittest.main()

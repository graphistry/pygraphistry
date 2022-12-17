import unittest
import pytest
import graphistry
import pandas as pd
import logging


logger = logging.getLogger(__name__)

edf = pd.read_csv(
    "graphistry/tests/data/malware_capture_bot.csv", index_col=0, nrows=50
)
edf = edf.drop_duplicates()
src, dst = "to_node", "from_node"
edf["to_node"] = edf.SrcAddr.astype(str)
edf["from_node"] = edf.DstAddr.astype(str)


class TestConditional(unittest.TestCase):
    def test_conditional_graph(self):
        # simple test to see if DGL graph was set during different featurization + umap strategies
        g = graphistry.bind(source=src, destination=dst).edges(edf).nodes(edf, src)
        for kind in ['nodes', 'edges']:
            g2 = g.conditional_graph(src, dst, kind=kind)
            assert '_probs' in g2._edges, 'enrichment has not taken place'
            
        
    def test_conditional_probs(self):
        g = graphistry.bind(source=src, destination=dst).edges(edf).nodes(edf, src)
        for kind in ['nodes', 'edges']:
            for how in ['index', 'columns']:
                df = g.conditional_probs(src, dst, kind=kind, how=how)
                assert len(df) > 0, 'no probabilities'
                assert df.sum(0) == 1.0, 'probabilities do not sum to 1'
                
import requests
import json
import re
import unittest
import pandas as pd
import pytest

import graphistry

class test_GFQLUtils(unittest.TestCase):

    def setUp(self) -> None:
        ge_df = pd.read_csv('https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/facebook_combined.txt', sep=' ', names=['s', 'd'])
        self.g = graphistry.edges(ge_df, 's', 'd')
        self.g = self.g.nodes(self.g._nodes, 'id')
        shareable_and_embeddable_url = self.g.plot(render=False)
        dataset_id = re.search(r'dataset=([^&]+)&type', shareable_and_embeddable_url)
        self.dataset_id = dataset_id.group(1)
        
    def null_filter_ex(self) -> None:
        self.g.gfql(operations = [{"type": "Edge",
                        "filter_dict": {}}]
     )
        
    def simple_filter_ex(self) -> None:
        self.g.gfql(operations = [{"type": "Edge",
                        "filter_dict": {},
                        "direction": "undirected",
                            "to_fixed_point": False,
                            "hops": 2}]
     )

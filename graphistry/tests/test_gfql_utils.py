import requests
import json
import re
import unittest
import pandas as pd
import pytest

# import graphistry

class test_GFQLUtils(unittest.TestCase):

    @pytest.fixture
    def grab_fb_dataset_id() -> None:
        ge_df = pd.read_csv('https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/facebook_combined.txt', sep=' ', names=['s', 'd'])
        import graphistry
        g = graphistry.edges(ge_df, 's', 'd').nodes(gg._nodes, 'id')
        shareable_and_embeddable_url = g.plot(render=False)
        dataset_id = re.search(r'dataset=([^&]+)&type', shareable_and_embeddable_url)
        return dataset_id.group(1)
        
    def null_filter_ex(dataset_id=grab_fb_dataset_id) -> None:
        g.gfql(operations = [{"type": "Edge",
                        "filter_dict": {}}]
     )
        
    def simple_filter_ex(dataset_id=grab_fb_dataset_id) -> None:
        g.gfql(operations = [{"type": "Edge",
                        "filter_dict": {},
                        "direction": "undirected",
                            "to_fixed_point": False,
                            "hops": 2}]
     )

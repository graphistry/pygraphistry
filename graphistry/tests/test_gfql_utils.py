import requests
import json
import unittest
import pandas as pd
import pytest

from graphistry.gfql_utils import serial_gfql

class TestGFQL_serial_remote(unittest.TestCase):

    @pytest.fixture
    def grab_fb_dataset_id() -> None:
        df = pd.read_csv('https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/facebook_combined.txt', sep=' ', names=['s', 'd'])
        import graphistry
        g = graphistry.edges(df, 's', 'd').materialize_nodes()
        shareable_and_embeddable_url = g.plot(render=False)
        dataset_id = re.search(r'dataset=([^&]+)&type', shareable_and_embeddable_url)
        dataset_id = dataset_id.group(1)
        
    @pytest.mark.skipif(not serial_gfql, reason="requires gfql feature dependencies")
    def null_filter_ex(dataset_id=grab_fb_dataset_id) -> None:
        serial_gfql(make_grab_dataset_id,
        operations = [{"type": "Edge",
                        "filter_dict": {}}]
     )
        
    @pytest.mark.skipif(not serial_gfql, reason="requires gfql feature dependencies")
    def simple_filter_ex(dataset_id=grab_fb_dataset_id) -> None:
        serial_gfql(dataset_id,
        operations = [{"type": "Edge",
                        "filter_dict": {},
                        "direction": "undirected",
                            "to_fixed_point": False,
                            "hops": 2}]
     )

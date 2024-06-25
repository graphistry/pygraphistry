import requests
import json

from graphistry.gfql_utils import serial_gfql

class TestGFQL_serial_remote(unittest.TestCase):
        
    @pytest.mark.skipif(not serial_gfql, reason="requires gfql feature dependencies")
    def make_grab_dataset_id(self) -> None:

    @pytest.mark.skipif(not serial_gfql, reason="requires gfql feature dependencies")
    def null_filter_ex(self) -> None:

    @pytest.mark.skipif(not serial_gfql, reason="requires gfql feature dependencies")
    def simple_filter_ex(self) -> None:
        serial_gfql('5ebdb560a3934810b9694518ca7aa147',
        operations = [{"type": "Edge",
                        "filter_dict": {},
                        "direction": "undirected",
                            "to_fixed_point": False,
                            "hops": 2}]
     )

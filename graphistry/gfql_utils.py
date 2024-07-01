import requests
import json
import logging
from typing import TYPE_CHECKING, List, Any
from inspect import getmodule

from .feature_utils import FeatureMixin

if TYPE_CHECKING:
    MIXIN_BASE = FeatureMixin
else:
    MIXIN_BASE = object

logger = logging.getLogger(__name__)

class GFQLMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def _process_gfql_query_direct(self, dataset_id: str, operations: List[Any], server: str, auth_token: str)-> None:
        
        url = 'https://' + server + '/api/v2/etl/datasets/' + dataset_id + '/gfql/'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + auth_token
        }

        data = {
            "gfql_operations": operations
    }
        return requests.post(url, headers=headers, json=data)

    def _run_serialized_gfql_query(self, dataset_id: str, operations: List[Any]) -> None:
        from graphistry import server, api_token
        response_data = self._process_gfql_query_direct(
            dataset_id = dataset_id,
            operations = operations,
            server = server(),
            auth_token = api_token(),
        )
        
        return response_data


    def gfql(self,
             operations: List[Any] = {"type": "Edge","filter_dict": {}}) -> None:
        response = None
        import re
        shareable_and_embeddable_url = self.plot(render=False)
        dataset_id = re.search(r'dataset=([^&]+)&type', shareable_and_embeddable_url)
        dataset_id = dataset_id.group(1)
        response = self._run_serialized_gfql_query(dataset_id, operations)
        return response.text
    
    

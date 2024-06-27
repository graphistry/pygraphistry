import requests
import json
import re
import logging
from typing import TYPE_CHECKING, List, Any
from inspect import getmodule

from .feature_utils import FeatureMixin

if TYPE_CHECKING:
    MIXIN_BASE = FeatureMixin
else:
    MIXIN_BASE = object

logger = logging.getLogger(__name__)

class GFQLUtils(MIXIN_BASE):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
def process_gfql_query_direct(self, dataset_id: str,operations: List[Any], server: str, auth_token:str)-> None:
    
    url = 'https://'+server+'/api/v2/etl/datasets/'+dataset_id+'/gfql/'
    headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer '+auth_token
        }

    data = {
        "gfql_operations": operations
}
    return requests.post(url, headers=headers, json=data)

def run_serialized_gfql_query(self, dataset_id: str, operations: List[Any]) -> None:
    if not isinstance(operations, str):
        operations_str = json.dumps(operations)
    else:
        operations_str = operations
    from graphistry import server, api_token
    response_data = process_gfql_query_direct(self,
        dataset_id = dataset_id,
        operations = operations,
        server = server(),
        auth_token = api_token(),
    )
    
    return response_data


def run(self,dataset_id,operations={"type": "Edge","filter_dict": {}}) -> None:
    response = None
    print(str(getmodule(dataset_id)))
    if 'frame' in str(getmodule(dataset_id)):
        print('part1')
        import graphistry
        try:
            dataset_id = graphistry.edges(dataset_id).materialize_nodes()
        except:
            dataset_id = graphistry.nodes(dataset_id)
    if 'plotter' in str(getmodule(dataset_id)):
        print('part2')
        import re
        shareable_and_embeddable_url = dataset_id.plot(render=False)
        dataset_id = re.search(r'dataset=([^&]+)&type', shareable_and_embeddable_url)
        dataset_id = dataset_id.group(1)
    if isinstance(dataset_id,str):
        print('part3')
        response = run_serialized_gfql_query(self,dataset_id, operations)
    return response.text

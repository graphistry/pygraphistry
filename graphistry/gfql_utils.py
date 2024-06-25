import requests
import json
import logging

from graphistry import server, api_token

logger = logging.getLogger(__name__)

def process_gfql_query_direct(dataset_id,operations,server,auth_token):
    
    url = 'https://'+server+'/api/v2/etl/datasets/'+dataset_id+'/gfql/'
    headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer '+auth_token
        }

    data = {
        "gfql_operations": operations
}
    return requests.post(url, headers=headers, json=data)

def run_serialized_gfql_query(dataset_id, operations):
    if not isinstance(operations, str):
        operations_str = json.dumps(operations)
    else:
        operations_str = operations

    response_data = process_gfql_query_direct(
        dataset_id = dataset_id,
        operations = operations,
        server = server(),
        auth_token = api_token(),
    )
    
    return response_data


def serial_gfql(dataset_id,operations={"type": "Edge","filter_dict": {}}):
    response = run_serialized_gfql_query(dataset_id, operations)
    return response.text

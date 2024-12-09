from typing import Optional, Union
from typing_extensions import Literal
import ast
import pandas as pd
import requests

from graphistry.Plottable import Plottable


def python_remote(
    self: Plottable,
    code: str,
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    engine: Literal["pandas", "cudf"] = "cudf",
    validate: bool = True
) -> Union[Plottable, pd.DataFrame]:
    """Remotely run Python code on a remote dataset.
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: str

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param engine: Override which run mode GFQL uses. Defaults to "cudf".
    :type engine: Literal["pandas", "cudf]

    :param validate: Whether to locally test code, and if uploading data, the data. Default true.
    :type validate: bool

    **Example: Upload data and count the results**
        ::

            import graphistry
            from graphistry import n, e
            es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
            g1 = graphistry
                .edges(es, source='src', destination='dst')
                .upload()
            assert g1._dataset_id is not None, "Successfully uploaded"
            out_json = g1.python_remote(
                code='''
                    from typing import Any, Dict
                    from graphistry import Plottable

                    def task(g: Plottable) -> Dict[str, Any]:
                        return {
                            'num_edges': len(g._edges)
                        }
                ''',
                engine='cudf')
            num_edges = out_json['num_edges']
            print(f'num_edges: {num_edges}')
    """

    assert code is not None and isinstance(code, str), f"Expected code to be a string, received type: {type(code)}"

    if validate:
        ast.parse(code)
        #TODO ensure has ast node for top-level method task

    if not api_token:
        from graphistry.pygraphistry import PyGraphistry
        PyGraphistry.refresh()
        api_token = PyGraphistry.api_token()

    if not dataset_id:
        dataset_id = self._dataset_id

    if not dataset_id:
        self = self.upload(validate=validate)
        dataset_id = self._dataset_id
    
    if not dataset_id:
        raise ValueError("Missing dataset_id; either pass in, or call on g2=g1.plot(render='g') in api=3 mode ahead of time")

    assert engine in ["pandas", "cudf"], f"engine should be 'pandas' or 'cudf', got: {engine}" 

    request_body = {
        "execute": code,
        "engine": engine
    }

    url = f"{self.base_url_server()}/api/v2/datasets/{dataset_id}/python"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=request_body)

    response.raise_for_status()

    return response.json()

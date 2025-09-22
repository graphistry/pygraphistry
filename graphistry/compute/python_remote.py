from inspect import getmodule
import inspect
from io import BytesIO
from typing import Any, Callable, Optional, Union
import zipfile
from typing_extensions import Literal
import ast
import pandas as pd
import requests

from graphistry.Plottable import Plottable
from graphistry.models.compute.chain_remote import FormatType, OutputTypeAll, OutputTypeDf


def validate_python_str(code: str) -> bool:
    """Validate Python code string.

    Returns True if the code string is valid, otherwise return False or raise ValueError
    """

    assert isinstance(code, str), f"Expected code to be a string, received type: {type(code)}"

    tree = ast.parse(code)
 
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "task":
            if len(node.args.args) == 1:
                return True
            else:
                raise ValueError(f"Invalid: The function 'task' does not have exactly one parameter. Found {len(node.args.args)}.")

    raise ValueError("Invalid: No top-level function 'task' defined.")

def python_remote_generic(
    self: Plottable,
    code: Union[str, Callable[..., object]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    format: Optional[FormatType] = 'json',
    output_type: Optional[OutputTypeAll] = 'json',
    engine: Literal["pandas", "cudf"] = "cudf",
    run_label: Optional[str] = None,
    validate: bool = True
) -> Union[Plottable, pd.DataFrame, Any]:
    """Remotely run Python code on a remote dataset.
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: Union[str, Callable[..., object]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param format: What format to fetch results. Defaults to 'json'. We recommend a columnar format such as parquet.
    :type format: Optional[FormatType]

    :param output_type: What shape of output to fetch. Defaults to 'json'. Options include 'nodes', 'edges', 'all' (both), 'table', 'shape', and 'json'.
    :type output_type: Optional[OutputTypeAll]

    :param engine: Override which run mode GFQL uses. Defaults to "cudf".
    :type engine: Literal["pandas", "cudf]

    :param run_label: Optional label for the run for serverside job tracking.
    :type run_label: Optional[str]

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

    if callable(code):
        if code.__name__ != "task":
            code_str = inspect.getsource(code)
            old_name = code.__name__
            code = code_str.replace(f"def {old_name}", "def task", 1)

    assert code is not None and isinstance(code, str), f"Expected code to be a string, received type: {type(code)}"

    if validate:
        if not validate_python_str(code):
            raise ValueError("Invalid code")

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
    
    assert format in ["json", "csv", "parquet"], f"format should be 'json', 'csv', or 'parquet', got: {format}"

    assert engine in ["pandas", "cudf"], f"engine should be 'pandas' or 'cudf', got: {engine}"

    # TODO remove auto-indent when server updated
    # workaround parsing bug by indenting each line by 4 spaces
    code_indented = "\n".join(["    " + line for line in code.split("\n")])

    request_body = {
        "execute": code_indented,
        "engine": engine,
        **({"run_label": run_label} if run_label else {}),
        **({'format': format} if format != 'json' else {}),
        **({'output_type': output_type} if output_type is not None and output_type != 'json' else {})
    }

    url = f"{self.base_url_server()}/api/v2/datasets/{dataset_id}/python"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=request_body)

    response.raise_for_status()

    if self._edges is None or isinstance(self._edges, pd.DataFrame):
        df_cons = pd.DataFrame
        read_csv = pd.read_csv
        read_parquet = pd.read_parquet
    elif 'cudf.core.dataframe' in str(getmodule(self._edges)):
        import cudf
        df_cons = cudf.DataFrame
        read_csv = cudf.read_csv
        read_parquet = cudf.read_parquet
    else:
        raise ValueError(f"Unknown self._edges type, expected cudf/pandas DataFrame: {type(self._edges)}")

    if output_type == "shape":
        if format == "json":
            return pd.DataFrame(response.json())
        elif format == "csv":
            return read_csv(BytesIO(response.content))
        elif format == "parquet":
            return read_parquet(BytesIO(response.content))
        else:
            raise ValueError(f"Unknown format, expected json/csv/parquet, got: {format}")
    elif output_type == "all" and format in ["csv", "parquet"]:
        zip_buffer = BytesIO(response.content)
        with zipfile.ZipFile(zip_buffer, "r") as zip_ref:
            nodes_file = [f for f in zip_ref.namelist() if "nodes" in f][0]
            edges_file = [f for f in zip_ref.namelist() if "edges" in f][0]

            nodes_data = zip_ref.read(nodes_file)
            edges_data = zip_ref.read(edges_file)

            if len(nodes_data) > 0:
                nodes_df = read_parquet(BytesIO(nodes_data)) if format == "parquet" else read_csv(BytesIO(nodes_data))
            else:
                nodes_df = df_cons()

            if len(edges_data) > 0:
                edges_df = read_parquet(BytesIO(edges_data)) if format == "parquet" else read_csv(BytesIO(edges_data))
            else:
                edges_df = df_cons()

            return self.edges(edges_df).nodes(nodes_df)
    elif output_type in ["nodes", "edges", "table"] and format in ["csv", "parquet"]:
        data = BytesIO(response.content)
        if len(response.content) > 0:
            df = read_parquet(data) if format == "parquet" else read_csv(data)
        else:
            df = df_cons()
        if output_type == "nodes":
            out = self.nodes(df)
            out._edges = None
            return out
        elif output_type == "edges":
            out = self.edges(df)
            out._nodes = None
            return out
        elif output_type == "table":
            return df
    elif format == "json":
        o = response.json()
        if output_type == "all":
            return self.edges(df_cons(o['edges'])).nodes(df_cons(o['nodes']))
        elif output_type == "nodes":
            out = self.nodes(df_cons(o))
            out._edges = None
            return out
        elif output_type == "edges":
            out = self.edges(df_cons(o))
            out._nodes = None
            return out
        elif output_type == "table":
            return df_cons(o)
        elif output_type == "json":
            return o
        else:
            raise ValueError(f"JSON format read with unexpected output_type: {output_type}")
    else:
        raise ValueError(f"Unsupported format {format}, output_type {output_type}")

    raise ValueError("Unexpected code path")


def python_remote_g(
    self: Plottable,
    code: Union[str, Callable[..., object]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    format: Optional[FormatType] = 'parquet',
    output_type: Optional[OutputTypeAll] = 'all',
    engine: Literal["pandas", "cudf"] = "cudf",
    run_label: Optional[str] = None,
    validate: bool = True
) -> Plottable:
    """Remotely run Python code on a remote dataset that returns a Plottable
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: Union[str, Callable[..., object]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param format: What format to fetch results. Defaults to 'parquet'.
    :type format: Optional[FormatType]

    :param output_type: What shape of output to fetch. Defaults to 'all'. Options include 'nodes', 'edges', 'all' (both). For other variants, see python_remote_shape and python_remote_json.
    :type output_type: Optional[OutputTypeGraph]

    :param engine: Override which run mode GFQL uses. Defaults to "cudf".
    :type engine: Literal["pandas", "cudf]

    :param run_label: Optional label for the run for serverside job tracking.
    :type run_label: Optional[str]

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
            g2 = g1.python_remote_g(
                code='''
                    from typing import Any, Dict
                    from graphistry import Plottable

                    def task(g: Plottable) -> Dict[str, Any]:
                        return g
                ''',
                engine='cudf')
            num_edges = len(g2._edges)
            print(f'num_edges: {num_edges}')
    """

    assert output_type in ["all", "nodes", "edges"], f"output_type should be 'all', 'nodes', or 'edges', got: {output_type}"

    out = python_remote_generic(
        self=self,
        code=code,
        api_token=api_token,
        dataset_id=dataset_id,
        format=format,
        output_type=output_type,
        engine=engine,
        run_label=run_label,
        validate=validate
    )

    assert isinstance(out, Plottable), f"Expected Plottable, got: {type(out)}"

    return out


def python_remote_table(
    self: Plottable,
    code: Union[str, Callable[..., object]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    format: Optional[FormatType] = 'parquet',
    output_type: Optional[OutputTypeDf] = 'table',
    engine: Literal["pandas", "cudf"] = "cudf",
    run_label: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """Remotely run Python code on a remote dataset that returns a table
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: Union[str, Callable[..., object]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param format: What format to fetch results. Defaults to 'parquet'.
    :type format: Optional[FormatType]

    :param output_type: What shape of output to fetch. Defaults to 'table'. Options include 'table', 'nodes', and 'edges'.
    :type output_type: Optional[OutputTypeGraph]

    :param engine: Override which run mode GFQL uses. Defaults to "cudf".
    :type engine: Literal["pandas", "cudf]

    :param run_label: Optional label for the run for serverside job tracking.
    :type run_label: Optional[str]

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
            edges_df = g1.python_remote_table(
                code='''
                    from typing import Any, Dict
                    from graphistry import Plottable

                    def task(g: Plottable) -> Dict[str, Any]:
                        return g._edges
                ''',
                engine='cudf')
            num_edges = len(edges_df)
            print(f'num_edges: {num_edges}')
    """

    assert output_type in ["all", "nodes", "edges", "table"], f"output_type should be 'all', 'nodes', or 'edges', got: {output_type}"

    out = python_remote_generic(
        self=self,
        code=code,
        api_token=api_token,
        dataset_id=dataset_id,
        format=format,
        output_type=output_type,
        engine=engine,
        run_label=run_label,
        validate=validate
    )

    assert isinstance(out, pd.DataFrame), f"Expected pd.DataFrame, got: {type(out)}"

    return out

def python_remote_json(
    self: Plottable,
    code: Union[str, Callable[..., object]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    engine: Literal["pandas", "cudf"] = "cudf",
    run_label: Optional[str] = None,
    validate: bool = True
) -> Any:
    """Remotely run Python code on a remote dataset that returns json
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: Union[str, Callable[..., object]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param engine: Override which run mode GFQL uses. Defaults to "cudf".
    :type engine: Literal["pandas", "cudf]

    :param run_label: Optional label for the run for serverside job tracking.
    :type run_label: Optional[str]

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
            obj = g1.python_remote_json(
                code='''
                    from typing import Any, Dict
                    from graphistry import Plottable

                    def task(g: Plottable) -> Dict[str, Any]:
                        return {'num_edges': len(g._edges)}
                ''',
                engine='cudf')
            num_edges = obj['num_edges']
            print(f'num_edges: {num_edges}')
    """

    return python_remote_generic(
        self=self,
        code=code,
        api_token=api_token,
        dataset_id=dataset_id,
        format='json',
        output_type='json',
        engine=engine,
        run_label=run_label,
        validate=validate
    )
    

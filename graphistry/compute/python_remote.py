from inspect import getmodule
import inspect
from io import BytesIO
from typing import Any, Callable, Optional, Union
import zipfile
from typing_extensions import Literal
import ast
import textwrap
import pandas as pd
import requests

from graphistry.Engine import EngineAbstractType
from graphistry.Plottable import Plottable
from graphistry.compute.remote_df_io import (
    require_supported_frame_library,
    resolve_csv_reader,
    resolve_remote_engine,
    validate_csv_import_args)
from graphistry.compute.remote_response import (
    decode_json_body,
    decode_json_result,
    error_document_error,
    raise_for_remote_error,
    require_json_result_keys,
    select_zip_member,
)
from graphistry.models.compute.chain_remote import DFImportArgs, FormatType, OutputTypeAll, OutputTypeDf
from graphistry.otel import inject_trace_headers


def normalize_task_code(code: Union[str, Callable[..., object]]) -> str:
    """Normalize a callable or source string to a parseable top-level ``def task`` source."""

    if callable(code):
        code_str = inspect.getsource(code)
        old_name = code.__name__
        if old_name != "task":
            code_str = code_str.replace(f"def {old_name}", "def task", 1)
        code = code_str

    assert code is not None and isinstance(code, str), f"Expected code to be a string, received type: {type(code)}"

    # Source from a nested def, or written as an indented literal, does not parse as-is.
    return textwrap.dedent(code)


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
    engine: EngineAbstractType = 'auto',
    run_label: Optional[str] = None,
    validate: bool = True,
    df_import_args: Optional[DFImportArgs] = None,
) -> Union[Plottable, pd.DataFrame, Any]:
    """Remotely run Python code on a remote dataset.
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: Union[str, Callable[..., object]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param format: What format to fetch results. Defaults to 'json'. We recommend a columnar format such as parquet. ``'csv'`` is untyped on the wire: the client re-infers dtypes and can rewrite values, so it warns and serves. Pass ``df_import_args`` to control the reader.
    :type format: Optional[FormatType]

    :param output_type: What shape of output to fetch. Defaults to 'json'. Options include 'nodes', 'edges', 'all' (both), 'table', 'shape', and 'json'.
    :type output_type: Optional[OutputTypeAll]

    :param engine: Override which run mode GFQL uses. Defaults to 'auto' which auto-detects based on DataFrame type. Also accepts 'pandas' or 'cudf'.
    :type engine: EngineAbstractType

    :param run_label: Optional label for the run for serverside job tracking.
    :type run_label: Optional[str]

    :param validate: Whether to locally test code, and if uploading data, the data. Default true.
    :type validate: bool

    :param df_import_args: Reader kwargs the client applies when decoding a ``format='csv'`` response. Optional; without it csv dtypes are re-inferred from text, which can rewrite values (``'007'`` -> ``7.0``) and break the returned graph's own node/edge id join. The warning names each lossy axis your kwargs do not govern, and clears only once they govern both: dtype inference (``dtype``/``converters``) and NA substitution (``keep_default_na``/``na_values``/``na_filter``/``converters``). Prefer ``format='parquet'``, which is faithful and needs no reader args.
    :type df_import_args: Optional[Dict[str, Any]]

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

    code = normalize_task_code(code)

    validate_csv_import_args(df_import_args, "python_remote")
    frame_lib = require_supported_frame_library(self._nodes, self._edges, "python_remote")
    engine_str = resolve_remote_engine(engine, self, "python_remote").value

    assert format in ["json", "csv", "parquet"], f"format should be 'json', 'csv', or 'parquet', got: {format}"

    if validate:
        if not validate_python_str(code):
            raise ValueError("Invalid code")

    if not api_token:
        self._pygraphistry.refresh()
        api_token = self.session.api_token

    if not dataset_id:
        dataset_id = self._dataset_id

    if not dataset_id:
        self = self.upload(validate=validate)
        dataset_id = self._dataset_id
    
    if not dataset_id:
        raise ValueError("Missing dataset_id; either pass in, or call on g2=g1.plot(render='g') in api=3 mode ahead of time")

    # TODO remove auto-indent when server updated
    # workaround parsing bug by indenting each line by 4 spaces
    code_indented = "\n".join(["    " + line for line in code.split("\n")])

    request_body = {
        "execute": code_indented,
        "engine": engine_str,
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
    headers = inject_trace_headers(headers)

    response = requests.post(url, headers=headers, json=request_body, verify=self.session.certificate_validation)

    raise_for_remote_error(response, "Remote Python operation")

    # Library was resolved pre-request; reuse it so the two cannot drift.
    if frame_lib == "cudf":
        import cudf
        df_cons = cudf.DataFrame
        read_csv = cudf.read_csv
        read_parquet = cudf.read_parquet
    else:
        df_cons = pd.DataFrame
        read_csv = pd.read_csv
        read_parquet = pd.read_parquet

    if format == "csv":
        read_csv = resolve_csv_reader(read_csv, df_import_args, "python_remote")

    if output_type == "shape":
        if format == "json":
            return pd.DataFrame(decode_json_result(response, "Remote Python operation"))
        elif format == "csv":
            return read_csv(BytesIO(response.content))
        elif format == "parquet":
            return read_parquet(BytesIO(response.content))
        else:
            raise ValueError(f"Unknown format, expected json/csv/parquet, got: {format}")
    elif output_type == "all" and format in ["csv", "parquet"]:
        zip_buffer = BytesIO(response.content)
        try:
            zip_ref_cm = zipfile.ZipFile(zip_buffer, "r")
        except zipfile.BadZipFile as e:
            raise error_document_error(response, "Remote Python operation", "a zip archive") from e
        with zip_ref_cm as zip_ref:
            names = zip_ref.namelist()
            nodes_file = select_zip_member(names, "nodes", "Remote Python operation")
            edges_file = select_zip_member(names, "edges", "Remote Python operation")

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
        if output_type == "json":
            # A task's own return value is the result here, error-shaped documents included.
            return decode_json_body(response, "Remote Python operation")
        o = decode_json_result(response, "Remote Python operation")
        if output_type == "all":
            o = require_json_result_keys(o, ['nodes', 'edges'], response, "Remote Python operation")
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
    engine: EngineAbstractType = 'auto',
    run_label: Optional[str] = None,
    validate: bool = True,
    df_import_args: Optional[DFImportArgs] = None,
) -> Plottable:
    """Remotely run Python code on a remote dataset that returns a Plottable
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: Union[str, Callable[..., object]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param format: What format to fetch results. Defaults to 'parquet'. ``'csv'`` is untyped on the wire: the client re-infers dtypes and can rewrite values, so it warns and serves. Pass ``df_import_args`` to control the reader.
    :type format: Optional[FormatType]

    :param output_type: What shape of output to fetch. Defaults to 'all'. Options include 'nodes', 'edges', 'all' (both). For other variants, see python_remote_shape and python_remote_json.
    :type output_type: Optional[OutputTypeGraph]

    :param engine: Override which run mode GFQL uses. Defaults to 'auto' which auto-detects based on DataFrame type. Also accepts 'pandas' or 'cudf'.
    :type engine: EngineAbstractType

    :param run_label: Optional label for the run for serverside job tracking.
    :type run_label: Optional[str]

    :param validate: Whether to locally test code, and if uploading data, the data. Default true.
    :type validate: bool

    :param df_import_args: Reader kwargs the client applies when decoding a ``format='csv'`` response. Optional; without it csv dtypes are re-inferred from text, which can rewrite values and break the returned graph's own node/edge id join. The warning names each lossy axis your kwargs do not govern, and clears only once they govern both: dtype inference (``dtype``/``converters``) and NA substitution (``keep_default_na``/``na_values``/``na_filter``/``converters``). Prefer ``format='parquet'``, which is faithful and needs no reader args.
    :type df_import_args: Optional[Dict[str, Any]]

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
        validate=validate,
        df_import_args=df_import_args,
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
    engine: EngineAbstractType = 'auto',
    run_label: Optional[str] = None,
    validate: bool = True,
    df_import_args: Optional[DFImportArgs] = None,
) -> pd.DataFrame:
    """Remotely run Python code on a remote dataset that returns a table
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param code: Python code that includes a top-level function `def task(g: Plottable) -> Union[str, Dict]`.
    :type code: Union[str, Callable[..., object]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not defined, will upload current data, store that dataset_id, and run code against that.
    :type dataset_id: Optional[str]

    :param format: What format to fetch results. Defaults to 'parquet'. ``'csv'`` is untyped on the wire: the client re-infers dtypes and can rewrite values, so it warns and serves. Pass ``df_import_args`` to control the reader.
    :type format: Optional[FormatType]

    :param output_type: What shape of output to fetch. Defaults to 'table'. Options include 'table', 'nodes', and 'edges'.
    :type output_type: Optional[OutputTypeGraph]

    :param engine: Override which run mode GFQL uses. Defaults to 'auto' which auto-detects based on DataFrame type. Also accepts 'pandas' or 'cudf'.
    :type engine: EngineAbstractType

    :param run_label: Optional label for the run for serverside job tracking.
    :type run_label: Optional[str]

    :param validate: Whether to locally test code, and if uploading data, the data. Default true.
    :type validate: bool

    :param df_import_args: Reader kwargs the client applies when decoding a ``format='csv'`` response. Optional; without it csv dtypes are re-inferred from text, which can rewrite values and break the returned graph's own node/edge id join. The warning names each lossy axis your kwargs do not govern, and clears only once they govern both: dtype inference (``dtype``/``converters``) and NA substitution (``keep_default_na``/``na_values``/``na_filter``/``converters``). Prefer ``format='parquet'``, which is faithful and needs no reader args.
    :type df_import_args: Optional[Dict[str, Any]]

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
        validate=validate,
        df_import_args=df_import_args,
    )

    assert isinstance(out, pd.DataFrame), f"Expected pd.DataFrame, got: {type(out)}"

    return out

def python_remote_json(
    self: Plottable,
    code: Union[str, Callable[..., object]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    engine: EngineAbstractType = 'auto',
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

    :param engine: Override which run mode GFQL uses. Defaults to 'auto' which auto-detects based on DataFrame type. Also accepts 'pandas' or 'cudf'.
    :type engine: EngineAbstractType

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
    

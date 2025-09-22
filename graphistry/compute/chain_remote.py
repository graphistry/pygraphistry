from inspect import getmodule
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal
import pandas as pd
import requests
import zipfile

from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject
from graphistry.compute.chain import Chain
from graphistry.models.compute.chain_remote import OutputTypeGraph, FormatType, output_types_graph
from graphistry.utils.json import JSONVal


def chain_remote_generic(
    self: Plottable,
    chain: Union[Chain, Dict[str, JSONVal], List[Any]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_type: OutputTypeGraph = "all",
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: Optional[Literal["pandas", "cudf"]] = None,
    validate: bool = True
) -> Union[Plottable, pd.DataFrame]:

    if not api_token:
        from graphistry.pygraphistry import PyGraphistry
        PyGraphistry.refresh()
        api_token = PyGraphistry.api_token()

    if not dataset_id:
        dataset_id = self._dataset_id

    if not dataset_id:
        self = self.upload(validate=validate)
        dataset_id = self._dataset_id

    if output_type not in output_types_graph:
        raise ValueError(f"Unknown output_type, expected one of {output_types_graph}, got: {output_type}")
    
    if not dataset_id:
        raise ValueError("Missing dataset_id; either pass in, or call on g2=g1.plot(render='g') in api=3 mode ahead of time")

    assert (engine is None) or engine in ["pandas", "cudf"], f"engine should be None, 'pandas', or 'cudf', got: {engine}" 

    if format is None:
        if output_type == "shape":
            format = "json"
        else:
            format = "parquet"

    if isinstance(chain, Chain):
        chain_json = chain.to_json()
    elif isinstance(chain, list):
        chain_json = Chain(chain).to_json()
    else:
        assert isinstance(chain, dict)
        chain_json = chain

    if validate:
        Chain.from_json(chain_json)

    request_body = {
        "gfql_operations": chain_json['chain'],  # unwrap
        "format": format
    }

    if node_col_subset is not None:
        request_body["node_col_subset"] = node_col_subset  # type: ignore
    if edge_col_subset is not None:
        request_body["edge_col_subset"] = edge_col_subset  # type: ignore
    if df_export_args is not None:
        request_body["df_export_args"] = df_export_args
    if engine is not None:
        request_body["engine"] = engine  # type: ignore

    url = f"{self.base_url_server()}/api/v2/etl/datasets/{dataset_id}/gfql/{output_type}"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=request_body)

    response.raise_for_status()

    # deserialize based on output_type & format

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
    elif output_type in ["nodes", "edges"] and format in ["csv", "parquet"]:
        data = BytesIO(response.content)
        if len(response.content) > 0:
            df = read_parquet(data) if format == "parquet" else read_csv(data)
        else:
            df = df_cons()
        if output_type == "nodes":
            out = self.nodes(df)
            out._edges = None
            return out
        else:
            out = self.edges(df)
            out._nodes = None
            return out
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
        else:
            raise ValueError(f"JSON format read with unexpected output_type: {output_type}")
    else:
        raise ValueError(f"Unsupported format {format}, output_type {output_type}")


def chain_remote_shape(
    self: Plottable,
    chain: Union[Chain, List[ASTObject], Dict[str, JSONVal]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: Optional[Literal["pandas", "cudf"]] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Like chain_remote(), except instead of returning a Plottable, returns a pd.DataFrame of the shape of the resulting graph.

    Useful as a fast success indicator that avoids the need to return a full graph when a match finds hits, return just the metadata.

    **Example: Upload graph and compute number of nodes with at least one edge**
        ::

            import graphistry
            es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
            g1 = graphistry.edges(es, 'src', 'dst').upload()
            assert g1._dataset_id, "Graph should have uploaded"

            shape_df = g1.chain_remote_shape([n(), e(), n()])
            print(shape_df)

    **Example: Compute number of nodes with at least one edge, with implicit upload, and force GPU mode**
        ::

            import graphistry
            es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
            g1 = graphistry.edges(es, 'src', 'dst')

            shape_df = g1.chain_remote_shape([n(), e(), n()], engine='cudf')
            print(shape_df)
    """

    out_df = chain_remote_generic(
        self,
        chain,
        api_token,
        dataset_id,
        'shape',
        format,
        df_export_args,
        node_col_subset,
        edge_col_subset,
        engine,
        validate
    )
    assert isinstance(out_df, pd.DataFrame)
    return out_df

def chain_remote(
    self: Plottable,
    chain: Union[Chain, List[ASTObject], Dict[str, JSONVal]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_type: OutputTypeGraph = "all",
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: Optional[Literal["pandas", "cudf"]] = None,
    validate: bool = True
) -> Plottable:
    """Remotely run GFQL chain query on a remote dataset.
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param chain: GFQL chain query as a Python object or in serialized JSON format
    :type chain: Union[Chain, List[ASTObject], Dict[str, JSONVal]]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not provided, will upload current data, store that dataset_id, and run GFQL against that.
    :type dataset_id: Optional[str]

    :param output_type: Whether to return nodes and edges ("all", default), Plottable with just nodes ("nodes"), or Plottable with just edges ("edges"). For just a dataframe of the resultant graph shape (output_type="shape"), use instead chain_remote_shape().
    :type output_type: OutputType

    :param format: What format to fetch results. We recommend a columnar format such as parquet, which it defaults to when output_type is not shape.
    :type format: Optional[FormatType]

    :param df_export_args: When server parses data, any additional parameters to pass in.
    :type df_export_args: Optional[Dict, str, Any]]

    :param node_col_subset: When server returns nodes, what property subset to return. Defaults to all.
    :type node_col_subset: Optional[List[str]]

    :param edge_col_subset: When server returns edges, what property subset to return. Defaults to all.
    :type edge_col_subset: Optional[List[str]]

    :param engine: Override which run mode GFQL uses. By default, inspects graph size to decide.
    :type engine: Optional[Literal["pandas", "cudf]]

    :param validate: Whether to locally test code, and if uploading data, the data. Default true.
    :type validate: bool

    **Example: Explicitly upload graph and return subgraph where nodes have at least one edge**
        ::

            import graphistry
            from graphistry import n, e
            es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
            g1 = graphistry.edges(es, 'src', 'dst').upload()
            assert g1._dataset_id, "Graph should have uploaded"

            g2 = g1.chain_remote([n(), e(), n()])
            print(f'dataset id: {g2._dataset_id}, # nodes: {len(g2._nodes)}')

    **Example: Return subgraph where nodes have at least one edge, with implicit upload**
        ::

            import graphistry
            from graphistry import n, e
            es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
            g1 = graphistry.edges(es, 'src', 'dst')
            g2 = g1.chain_remote([n(), e(), n()])
            print(f'dataset id: {g2._dataset_id}, # nodes: {len(g2._nodes)}')

    **Example: Return subgraph where nodes have at least one edge, with implicit upload, and force GPU mode**
        ::

            import graphistry
            from graphistry import n, e
            es = pandas.DataFrame({'src': [0,1,2], 'dst': [1,2,0]})
            g1 = graphistry.edges(es, 'src', 'dst')
            g2 = g1.chain_remote([n(), e(), n()], engine='cudf')
            print(f'dataset id: {g2._dataset_id}, # nodes: {len(g2._nodes)}')

    """

    assert output_type != "shape", 'Method chain_remote() does not support output_type="shape", call instead chain_remote_shape()'
    
    g = chain_remote_generic(
        self,
        chain,
        api_token,
        dataset_id,
        output_type,
        format,
        df_export_args,
        node_col_subset,
        edge_col_subset,
        engine,
        validate
    )
    assert isinstance(g, Plottable)
    return g

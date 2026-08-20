from inspect import getmodule
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Union
from typing_extensions import Literal, Protocol, TypeGuard
import json
import pandas as pd
import requests
import uuid
import warnings
import zipfile

from graphistry.Engine import Engine, EngineAbstractType, resolve_input_engine
from graphistry.Plottable import Plottable
from graphistry.client_session import DatasetInfo
from graphistry.compute.ast import ASTLet, ASTObject
from graphistry.compute.chain import Chain
from graphistry.compute.gfql.cypher.lowering import compile_cypher_query
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.strictness import (
    DEFAULT_STRICT_LEVEL, StrictInput, resolve_strict_level, schema_declared_names, strictness_scope)
from graphistry.compute.gfql_validate import gfql_validate as gfql_preflight_validate
from graphistry.io.metadata import deserialize_plottable_metadata
from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLTypeError
from graphistry.compute.remote_df_io import (
    require_supported_frame_library, resolve_csv_reader, validate_csv_import_args)
from graphistry.compute.remote_response import (
    check_subset_result_bindings,
    decode_json_result,
    error_document_error,
    raise_for_remote_error,
    require_json_result_keys,
    select_zip_member,
)
from graphistry.models.compute.chain_remote import DFImportArgs, OutputTypeGraph, FormatType, output_types_graph
from graphistry.utils.json import JSONVal, find_non_finite
from graphistry.otel import inject_trace_headers


class CompiledProcedureCallLike(Protocol):
    procedure: str
    call_params: Optional[Dict[str, Any]]


class CompiledBindingLike(Protocol):
    name: str
    chain: Chain
    procedure_call: Optional[CompiledProcedureCallLike]
    use_ref: Optional[str]


class CompiledQueryLike(Protocol):
    chain: Chain
    graph_bindings: Sequence[CompiledBindingLike]
    procedure_call: Optional[CompiledProcedureCallLike]
    use_ref: Optional[str]


class CompiledUnionLike(Protocol):
    branches: Sequence[Any]


def _has_attrs(obj: Any, *names: str) -> bool:
    return all(hasattr(obj, name) for name in names)


def _is_compiled_union_query_shape(compiled: Any) -> TypeGuard[CompiledUnionLike]:
    return _has_attrs(compiled, "branches") and not _has_attrs(compiled, "chain")


def _is_compiled_query_shape(compiled: Any) -> TypeGuard[CompiledQueryLike]:
    return _has_attrs(compiled, "chain", "graph_bindings", "procedure_call", "use_ref")


def _step_to_json(
    chain: Chain,
    procedure_call: Optional[CompiledProcedureCallLike],
    use_ref: Optional[str],
) -> Dict[str, Any]:
    """Serialize one graph-pipeline step (binding or final clause) to wire format."""
    if procedure_call is not None:
        if not _has_attrs(procedure_call, "procedure"):
            raise TypeError(
                "Compiled procedure call must provide `procedure` for remote serialization. "
                f"Got {type(procedure_call)}"
            )
        call_params = getattr(procedure_call, "call_params", None)
        val: Dict[str, Any] = {
            "type": "Call",
            "function": procedure_call.procedure,
            "params": dict(call_params) if call_params else {},
        }
    else:
        val = chain.to_json()
    if use_ref is not None:
        return {"type": "Ref", "ref": use_ref, "chain": val.get('chain', [val])}
    return val


def _compiled_to_let_json(compiled: CompiledQueryLike) -> Dict[str, Any]:
    """Convert a structural compiled query with graph_bindings to Let wire format."""
    bindings: Dict[str, Any] = {
        b.name: _step_to_json(b.chain, b.procedure_call, b.use_ref)
        for b in compiled.graph_bindings
    }
    bindings["__result__"] = _step_to_json(compiled.chain, compiled.procedure_call, compiled.use_ref)
    return {"type": "Let", "bindings": bindings}


def _refresh_url_from_dataset_id(g: Plottable) -> None:
    dataset_id = getattr(g, "_dataset_id", None)
    if not isinstance(dataset_id, str) or dataset_id == "":
        return
    info: DatasetInfo = {
        "name": dataset_id,
        "type": "arrow",
        "viztoken": str(uuid.uuid4()),
    }
    g._url = g._pygraphistry._viz_url(info, g._url_params)


def _apply_persist_axis_defaults(g: Plottable) -> None:
    from graphistry.validate import apply_axis_url_defaults

    merged = apply_axis_url_defaults(
        getattr(g, "_url_params", None),
        getattr(g, "_complex_encodings", None),
    )
    if isinstance(merged, dict):
        g._url_params = merged


def chain_remote_generic(
    self: Plottable,
    chain: Union[Chain, Dict[str, JSONVal], List[Any], 'ASTLet', str],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_type: OutputTypeGraph = "all",
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: EngineAbstractType = 'auto',
    validate: bool = True,
    persist: bool = False,
    params: Optional[Dict[str, Any]] = None,  # hygiene-ok: explicit-any -- Cypher params are heterogeneous JSON scalars, matching gfql_remote()
    output: Optional[str] = None,
    df_import_args: Optional[DFImportArgs] = None,
    strict: StrictInput = None,
) -> Union[Plottable, pd.DataFrame]:

    strict_level = resolve_strict_level(self, strict=strict)

    if not api_token:
        self._pygraphistry.refresh()
        api_token = self.session.api_token

    if output_type not in output_types_graph:
        raise ValueError(f"Unknown output_type, expected one of {output_types_graph}, got: {output_type}")

    # Resolve engine: auto -> pandas/cudf based on graph DataFrame type
    engine_resolved = resolve_input_engine(engine, self)
    if engine_resolved not in [Engine.PANDAS, Engine.CUDF]:
        raise ValueError(f"Remote GFQL only supports 'pandas' or 'cudf' engines (or 'auto' which resolves to one of them). "
                       f"Got engine='{engine}' which resolved to '{engine_resolved.value}'. "
                       f"Dask engines are not supported for remote execution.")
    engine_str = engine_resolved.value 

    if format is None:
        if output_type == "shape":
            format = "json"
        else:
            format = "parquet"

    validate_csv_import_args(df_import_args, "gfql_remote")
    frame_lib = require_supported_frame_library(self._nodes, self._edges, "gfql_remote")

    # Validate persist compatibility early
    if persist and output_type in ["nodes", "edges"]:
        raise ValueError(f"persist=True is not supported with output_type='{output_type}'. "
                        f"Use output_type='all' for persistence support.")

    # --- Input normalization ---
    # Produces: chain_json (wire-format dict) + is_let (bool)
    is_let = False

    if isinstance(chain, str):
        # Cypher string: compile locally, serialize result
        parsed = parse_cypher(chain)
        compiled = compile_cypher_query(parsed, params=params)
        if _is_compiled_union_query_shape(compiled):
            raise ValueError(
                "UNION queries are not yet supported for remote execution via gfql_remote(). "
                "Execute locally with g.gfql() instead."
            )
        if not _is_compiled_query_shape(compiled):
            raise TypeError(f"Unexpected compiled Cypher type: {type(compiled)}")
        if compiled.graph_bindings or compiled.use_ref:
            chain_json = _compiled_to_let_json(compiled)
            is_let = True
        else:
            chain_json = compiled.chain.to_json()
    elif isinstance(chain, ASTLet):
        chain_json = chain.to_json()
        is_let = True
    elif isinstance(chain, Chain):
        chain_json = chain.to_json()
    elif isinstance(chain, list):
        chain_json = Chain(chain).to_json()
    elif isinstance(chain, dict):
        chain_json = chain
        is_let = chain_json.get('type') == 'Let'
    else:
        raise TypeError(f"gfql_remote() query must be Chain, List, ASTLet, Dict, or str. Got {type(chain)}")

    if output is not None and not is_let:
        raise GFQLSyntaxError(
            ErrorCode.E109,
            "output= names a binding to return and requires a Let/DAG query; "
            "this query compiled to a flat chain, which has no bindings",
            field="output",
            value=output,
            suggestion="Drop output=, or express the query as a Let/DAG (or Cypher with named graph bindings)",
        )

    if validate:
        declared = schema_declared_names(self)  # a declared schema is names without data (#1916)
        with strictness_scope(strict_level, declared=declared):
            gfql_preflight_validate(
                self,
                chain,
                params=params,
                strict=strict_level,
                collect_all=False,
                schema=False,
            )

    if not dataset_id:
        dataset_id = self._dataset_id

    if not dataset_id:
        self = self.upload(validate=validate)
        dataset_id = self._dataset_id

    if not dataset_id:
        raise ValueError("Missing dataset_id; either pass in, or call on g2=g1.plot(render='g') in api=3 mode ahead of time")

    # --- Build request body (dual-field for backward compat) ---
    if is_let:
        warnings.warn(
            "gfql_remote() is sending a Let/DAG query. Servers that do not support "
            "the gfql_query field will receive an empty gfql_operations array and "
            "return the original graph unchanged. Upgrade to a server that reads "
            "gfql_query for full Let/DAG support.",
            UserWarning,
            stacklevel=2,
        )
        request_body: Dict[str, Any] = {
            "gfql_operations": [],
            "gfql_query": chain_json,
            "format": format
        }
        if output is not None:
            request_body["gfql_output"] = output
    else:
        request_body = {
            "gfql_operations": chain_json.get('chain', []),
            "gfql_query": chain_json,
            "format": format
        }

    if node_col_subset is not None:
        request_body["node_col_subset"] = node_col_subset
    if edge_col_subset is not None:
        request_body["edge_col_subset"] = edge_col_subset
    if df_export_args is not None:
        request_body["df_export_args"] = df_export_args
    request_body["engine"] = engine_str
    request_body["strictness"] = strict_level
    if strict_level != DEFAULT_STRICT_LEVEL:
        warnings.warn(
            f"gfql_remote() is requesting strictness={strict_level!r}. Servers that do not "
            "read the strictness field apply their own default, so absent labels/properties "
            "may still be reported differently than requested. Upgrade to a server that reads "
            "strictness for end-to-end parity.",
            UserWarning,
            stacklevel=2,
        )
    if persist:
        request_body["persist"] = persist

        # Include privacy settings for persisted dataset
        if hasattr(self, '_privacy') and self._privacy is not None:
            request_body["privacy"] = dict(self._privacy)

    non_finite = find_non_finite(request_body)
    if non_finite is not None:
        raise GFQLTypeError(
            ErrorCode.E201,
            "Filter values must be predicates or JSON-serializable: NaN and infinity have no JSON representation",
            field=non_finite,
            suggestion="Use is_na()/notna() predicates, or a finite bound",
        )

    url = f"{self.base_url_server()}/api/v2/etl/datasets/{dataset_id}/gfql/{output_type}"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }
    headers = inject_trace_headers(headers)

    response = requests.post(url, headers=headers, json=request_body, verify=self.session.certificate_validation)

    raise_for_remote_error(response, "GFQL remote operation")

    # deserialize based on output_type & format

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
        read_csv = resolve_csv_reader(read_csv, df_import_args, "gfql_remote")

    if output_type == "shape":
        if format == "json":
            return pd.DataFrame(decode_json_result(response, "GFQL remote operation"))
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
            raise error_document_error(response, "GFQL remote operation", "a zip archive") from e
        with zip_ref_cm as zip_ref:
            names = zip_ref.namelist()
            nodes_file = select_zip_member(names, "nodes", "GFQL remote operation")
            edges_file = select_zip_member(names, "edges", "GFQL remote operation")

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

            result = self.edges(edges_df).nodes(nodes_df)

            # Check for metadata.json in zip (both persist and GFQL metadata)
            if 'metadata.json' in zip_ref.namelist():
                try:
                    metadata_content = zip_ref.read('metadata.json')
                    metadata = json.loads(metadata_content.decode('utf-8'))

                    if persist:
                        # Extract dataset_id for URL generation
                        if 'dataset_id' in metadata:
                            result._dataset_id = metadata['dataset_id']

                            # Generate URL using existing infrastructure
                            if result._dataset_id:  # Type guard
                                _refresh_url_from_dataset_id(result)

                        # Optionally restore privacy settings
                        if 'privacy' in metadata:
                            result._privacy = metadata['privacy']

                    if 'gfql_metadata' in metadata:
                        result = deserialize_plottable_metadata(metadata['gfql_metadata'], result)
                        _apply_persist_axis_defaults(result)
                        if persist:
                            _refresh_url_from_dataset_id(result)

                except Exception as e:
                    if persist:
                        warnings.warn(f"persist=True requested but failed to parse metadata.json: {e}. "
                                f"URL generation will not be available. This may indicate an older server version.",
                                UserWarning, stacklevel=2)
                    else:
                        warnings.warn(f"Failed to parse metadata.json: {e}. GFQL metadata will not be hydrated.",
                                UserWarning, stacklevel=2)
            elif persist:
                warnings.warn("persist=True requested but server did not return metadata.json. "
                            "URL generation will not be available. This indicates an older server version that doesn't support zip format persistence.",
                            UserWarning, stacklevel=2)

            check_subset_result_bindings(result, node_col_subset, edge_col_subset, "GFQL remote operation")
            return result
    elif output_type in ["nodes", "edges"] and format in ["csv", "parquet"]:
        data = BytesIO(response.content)
        if len(response.content) > 0:
            df = read_parquet(data) if format == "parquet" else read_csv(data)
        else:
            df = df_cons()
        if output_type == "nodes":
            out = self.nodes(df)
            out._edges = None
        else:
            out = self.edges(df)
            out._nodes = None

        check_subset_result_bindings(out, node_col_subset, edge_col_subset, "GFQL remote operation")
        return out
    elif format == "json":
        o = decode_json_result(response, "GFQL remote operation")
        if output_type == "all":
            o = require_json_result_keys(o, ['nodes', 'edges'], response, "GFQL remote operation")
            result = self.edges(df_cons(o['edges'])).nodes(df_cons(o['nodes']))
        elif output_type == "nodes":
            result = self.nodes(df_cons(o))
            result._edges = None
        elif output_type == "edges":
            result = self.edges(df_cons(o))
            result._nodes = None
        else:
            raise ValueError(f"JSON format read with unexpected output_type: {output_type}")

        # Handle persist response - set dataset_id if provided
        if persist:
            if 'dataset_id' in o:
                result._dataset_id = o['dataset_id']

                # Generate URL using existing infrastructure
                if result._dataset_id:  # Type guard
                    _refresh_url_from_dataset_id(result)
            else:
                warnings.warn("persist=True requested but server did not return dataset_id in JSON response. "
                            "URL generation will not be available. This indicates an older server version that doesn't support persistence.",
                            UserWarning, stacklevel=2)

        if 'metadata' in o:
            result = deserialize_plottable_metadata(o['metadata'], result)
            _apply_persist_axis_defaults(result)
            if persist:
                _refresh_url_from_dataset_id(result)

        check_subset_result_bindings(result, node_col_subset, edge_col_subset, "GFQL remote operation")
        return result
    else:
        raise ValueError(f"Unsupported format {format}, output_type {output_type}")


def chain_remote_shape(
    self: Plottable,
    chain: Union[Chain, List[ASTObject], Dict[str, JSONVal], ASTLet, str],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: EngineAbstractType = 'auto',
    validate: bool = True,
    persist: bool = False,
    df_import_args: Optional[DFImportArgs] = None,
    params: Optional[Dict[str, Any]] = None,  # hygiene-ok: explicit-any -- Cypher params are heterogeneous JSON scalars, matching gfql_remote()
    output: Optional[str] = None,
    strict: StrictInput = None,
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

    :param params: Optional parameter dict for Cypher string queries (e.g. ``params={"cut": 10}`` for ``$cut``).
    :type params: Optional[Dict[str, Any]]

    :param output: Optional Let/DAG binding name to return. Requires a Let/DAG query.
    :type output: Optional[str]
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
        validate,
        persist,
        params=params,
        output=output,
        df_import_args=df_import_args,
        strict=strict,
    )
    assert isinstance(out_df, pd.DataFrame)
    return out_df

def chain_remote(
    self: Plottable,
    chain: Union[Chain, List[ASTObject], Dict[str, JSONVal], ASTLet, str],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_type: OutputTypeGraph = "all",
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: EngineAbstractType = 'auto',
    validate: bool = True,
    persist: bool = False,
    params: Optional[Dict[str, Any]] = None,  # hygiene-ok: explicit-any -- Cypher params are heterogeneous JSON scalars, matching gfql_remote()
    output: Optional[str] = None,
    df_import_args: Optional[DFImportArgs] = None,
    strict: StrictInput = None,
) -> Plottable:
    """Remotely run GFQL chain query on a remote dataset.
    
    Uses the latest bound `_dataset_id`, and uploads current dataset if not already bound. Note that rebinding calls of `edges()` and `nodes()` reset the `_dataset_id` binding.

    :param chain: GFQL query as a Python object, serialized GFQL JSON, or Cypher string
    :type chain: Union[Chain, List[ASTObject], Dict[str, JSONVal], ASTLet, str]

    :param api_token: Optional JWT token. If not provided, refreshes JWT and uses that.
    :type api_token: Optional[str]

    :param dataset_id: Optional dataset_id. If not provided, will fallback to self._dataset_id. If not provided, will upload current data, store that dataset_id, and run GFQL against that.
    :type dataset_id: Optional[str]

    :param output_type: Whether to return nodes and edges ("all", default), Plottable with just nodes ("nodes"), or Plottable with just edges ("edges"). For just a dataframe of the resultant graph shape (output_type="shape"), use instead chain_remote_shape().
    :type output_type: OutputType

    :param format: What format to fetch results. We recommend a columnar format such as parquet, which it defaults to when output_type is not shape. ``'csv'`` is untyped on the wire: the client re-infers dtypes and can rewrite values, so it warns and serves. Pass ``df_import_args`` to control the reader.
    :type format: Optional[FormatType]

    :param df_export_args: When server parses data, any additional parameters to pass in.
    :type df_export_args: Optional[Dict, str, Any]]

    :param df_import_args: Reader kwargs the client applies when decoding a ``format='csv'`` response. Optional; without it csv dtypes are re-inferred from text, which can rewrite values (``'007'`` -> ``7.0``) and break the returned graph's own node/edge id join. Supplying it takes explicit control and silences the warning. Prefer ``format='parquet'``, which is faithful and needs no reader args.
    :type df_import_args: Optional[Dict[str, Any]]

    :param node_col_subset: When server returns nodes, what property subset to return. Defaults to all.
    :type node_col_subset: Optional[List[str]]

    :param edge_col_subset: When server returns edges, what property subset to return. Defaults to all.
    :type edge_col_subset: Optional[List[str]]

    :param engine: Override which run mode GFQL uses. Defaults to 'auto' which auto-detects based on DataFrame type. Also accepts 'pandas' or 'cudf'.
    :type engine: EngineAbstractType

    :param validate: Whether to locally test code, and if uploading data, the data. Default true.
    :type validate: bool

    :param persist: Whether to persist dataset on server and return dataset_id for immediate URL generation. Default false.
    :type persist: bool

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
        validate,
        persist,
        params=params,
        output=output,
        df_import_args=df_import_args,
        strict=strict,
    )
    assert isinstance(g, Plottable)
    return g

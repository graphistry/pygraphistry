from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from graphistry.compute.typing import DataFrameT, SeriesT
from graphistry.compute.dataframe_utils import is_cudf_obj
from graphistry.Engine import (
    Engine,
    EngineAbstract,
    df_concat,
    df_to_engine,
    is_polars_df,
    resolve_engine,
)
from graphistry.compute.exceptions import ErrorCode, GFQLUnsupportedError, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CreateClause, CreateRelationship, CypherParams, CypherScalar, NodePattern, ParameterRef
from graphistry.Plottable import Plottable
from graphistry.compute.gfql.row.entity_props import (
    ID_PROPERTY_PRESENT,
    MUTATION_NODE_ID,
    NULLISH_LABEL_NAMES,
    label_flag_columns,
)

if TYPE_CHECKING:
    from graphistry.compute.typing import PolarsDType


class GFQLMutationError(GFQLValidationError):
    """Typed failure surface for unsupported or invalid mutation inputs."""


MUTATION_EDGE_ID = "__graphistry_edge_identity__"


def validate_create_parameters(creates: tuple[CreateClause, ...], params: Optional[CypherParams]) -> None:
    """Validate required parameters and scalar values without appending graph rows."""
    for clause in creates:
        for pattern in clause.patterns:
            if isinstance(pattern, CreateRelationship):
                _validate_node_parameters(pattern.source, params)
                _validate_node_parameters(pattern.destination, params)
                for entry in pattern.relationship.properties:
                    _property_value(entry.value, params=params, span=entry.span)
            else:
                _validate_node_parameters(pattern, params)


def _validate_node_parameters(pattern: NodePattern, params: Optional[CypherParams]) -> None:
    for entry in pattern.properties:
        _property_value(entry.value, params=params, span=entry.span)


def _property_value(value: object, *, params: Optional[CypherParams], span: object) -> object:
    span_line = getattr(span, "line", None)
    span_column = getattr(span, "column", None)
    if isinstance(value, ParameterRef):
        if params is None or value.name not in params:
            raise GFQLMutationError(
                ErrorCode.E105,
                f"Missing Cypher parameter '${value.name}'",
                field="create",
                value=value.name,
                suggestion=f"Pass params={{'{value.name}': ...}} to g.gfql(...).",
                line=span_line,
                column=span_column,
                language="cypher",
            )
        value = params[value.name]
    if isinstance(value, float) and math.isnan(value):
        raise GFQLMutationError(
            ErrorCode.E108, "CREATE cannot represent NaN property values as distinct from null",
            field="create", value="nan_property", line=span_line, column=span_column, language="cypher",
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise GFQLMutationError(
        ErrorCode.E108,
        "CREATE node properties currently support literal and parameter values only",
        field="create",
        value=type(value).__name__,
        suggestion="Use constants or Cypher parameters; runtime expressions and bound variables arrive with relationship support.",
        line=span_line,
        column=span_column,
        language="cypher",
    )


def _new_nodes_frame(
    engine: Engine,
    records: list[dict[str, object]],
    label_columns: set[str],
) -> DataFrameT:
    if engine in {Engine.PANDAS, Engine.CUDF, Engine.POLARS, Engine.POLARS_GPU}:
        import pandas as pd
        columns: list[str] = []
        for record in records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        data: dict[str, SeriesT] = {}
        for column in columns:
            values = [record.get(column) for record in records]
            declared = [value for value in values if value is not None]
            if any(isinstance(value, float) and math.isnan(value) for value in declared):
                raise GFQLMutationError(
                    ErrorCode.E108, "CREATE cannot represent NaN property values as distinct from null",
                    field=column, value="nan_property", language="cypher",
                )
            if column in label_columns:
                data[column] = pd.Series(values, dtype="boolean")
            elif declared and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in declared
            ):
                if any(
                    value < -(1 << 63) or value >= (1 << 63)
                    for value in declared if isinstance(value, int)
                ):
                    raise GFQLMutationError(
                        ErrorCode.E105, "CREATE integer value is outside the signed 64-bit range",
                        field=column, value="integer_overflow", language="cypher",
                    )
                data[column] = pd.Series(values, dtype="Int64")
            elif declared and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in declared
            ):
                if any(isinstance(value, int) and abs(value) > (1 << 53) for value in declared):
                    raise GFQLMutationError(
                        ErrorCode.E108, "CREATE cannot combine wide integers with floating-point values losslessly",
                        field=column, value="incompatible_numeric_types", language="cypher",
                    )
                data[column] = pd.Series(values, dtype="Float64")
            elif declared and all(isinstance(value, bool) for value in declared):
                data[column] = pd.Series(values, dtype="boolean")
            elif not declared or all(isinstance(value, str) for value in declared):
                data[column] = pd.Series(values, dtype="object")
            else:
                raise GFQLMutationError(
                    ErrorCode.E108, "CREATE property columns require compatible scalar values",
                    field=column, value="incompatible_property_values", language="cypher",
                )
        frame = pd.DataFrame(data)
        return df_to_engine(frame, engine)
    raise GFQLMutationError(
        ErrorCode.E108,
        f"CREATE is not supported for engine {engine.value}",
        field="engine",
        value=engine.value,
        suggestion="Use pandas, polars, or cuDF frames.",
        line=None,
        column=None,
        language="cypher",
    )


def _numeric_id_seed(nodes: DataFrameT, node_key: str, *, count: int, line: int, column: int) -> int:
    import numpy as np
    import pandas as pd

    if node_key not in nodes.columns:
        return 0
    if is_polars_df(nodes):
        import polars as pl
        if not isinstance(nodes, pl.DataFrame):
            raise GFQLUnsupportedError(
                ErrorCode.E108, "CREATE requires materialized input frames",
                field=node_key, value="lazy_frame", language="cypher",
            )
        values = nodes.get_column(node_key)
        limits: dict["PolarsDType", int] = {
            pl.Int8: (1 << 7) - 1, pl.Int16: (1 << 15) - 1,
            pl.Int32: (1 << 31) - 1, pl.Int64: (1 << 63) - 1,
            pl.UInt8: (1 << 8) - 1, pl.UInt16: (1 << 16) - 1,
            pl.UInt32: (1 << 32) - 1, pl.UInt64: (1 << 64) - 1,
        }
        dtype_description = str(values.dtype)
        integer = values.dtype in limits
        empty = nodes.height == 0
        has_null = bool(values.is_null().any())
        maximum_value = values.max() if integer and not empty and not has_null else 0
        if not isinstance(maximum_value, int):
            raise GFQLMutationError(ErrorCode.E108, "Unsupported physical identity scalar", field=node_key, value=type(maximum_value).__name__, language="cypher")
        maximum = maximum_value
        limit = limits.get(values.dtype, 0)
    else:
        series = nodes[node_key]
        dtype_description = str(series.dtype)
        integer = pd.api.types.is_integer_dtype(series.dtype)
        empty = len(nodes) == 0
        has_null = bool(series.isna().any())
        maximum = int(series.max()) if integer and not empty and not has_null else 0
        limit = int(np.iinfo(series.dtype.type).max) if integer else 0
    if empty:
        return 0
    if has_null:
        raise GFQLMutationError(
            ErrorCode.E105, f"Node identity column {node_key!r} contains null values",
            field=node_key, value="null", line=line, column=column, language="cypher",
        )
    if not integer:
        raise GFQLMutationError(
            ErrorCode.E108,
            f"CREATE requires integer numeric node identities; {node_key!r} has dtype {dtype_description!r}",
            field=node_key, value=dtype_description, line=line, column=column,
            suggestion="Use integer identities within the supported physical range.", language="cypher",
        )
    if maximum + count > limit:
        raise GFQLMutationError(
            ErrorCode.E105, f"Node identity column {node_key!r} has no remaining values",
            field=node_key, value=maximum, line=line, column=column,
            suggestion="Widen the identity column before CREATE.", language="cypher",
        )
    return maximum + 1


def _boolean_label_column(
    nodes: DataFrameT,
    label_column: str,
    *,
    line: int,
    column: int,
) -> Optional[SeriesT]:
    if label_column not in nodes.columns:
        return None
    if _column_family(nodes[label_column]) != "bool":
        raise GFQLMutationError(
            ErrorCode.E108,
            f"Label column {label_column!r} must be boolean",
            field=label_column,
            value=str(nodes[label_column].dtype),
            suggestion="Use the label__<Label> boolean convention.",
            line=line,
            column=column,
            language="cypher",
        )
    return nodes[label_column]


def _column_family(series: SeriesT) -> str:
    import pandas as pd

    from graphistry.Engine import is_polars_series
    if is_polars_series(series):
        import polars as pl
        polars_dtype = series.dtype
        if polars_dtype == pl.Boolean:
            return "bool"
        if polars_dtype.is_integer():
            return "integer"
        if polars_dtype.is_float():
            return "float"
        return "string" if polars_dtype == pl.String else "other"
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    return "string" if pd.api.types.is_string_dtype(dtype) else "other"


def _prepare_append_frames(nodes: DataFrameT, new_nodes: DataFrameT) -> tuple[DataFrameT, DataFrameT]:
    """Admit lossless column unions; never narrow candidate values to the old schema."""
    import pandas as pd

    polars = is_polars_df(nodes)
    null_columns = [
        name for name in new_nodes.columns
        if bool(new_nodes[name].is_null().all() if polars else new_nodes[name].isna().all())
    ]
    if null_columns:
        new_nodes = new_nodes.drop(null_columns) if polars else new_nodes.drop(columns=null_columns)
    if isinstance(nodes, pd.DataFrame):
        nullable: dict[str, SeriesT] = {}
        for name in nodes.columns:
            dtype = nodes[name].dtype
            if pd.api.types.is_integer_dtype(dtype):
                target = pd.UInt64Dtype() if pd.api.types.is_unsigned_integer_dtype(dtype) else pd.Int64Dtype()
                nullable[name] = nodes[name].astype(target)
            elif pd.api.types.is_bool_dtype(dtype):
                nullable[name] = nodes[name].astype("boolean")
        if nullable:
            nodes = nodes.assign(**nullable)
    # All-null properties take the non-null created dtype without losing values.
    null_old = [name for name in set(nodes.columns).intersection(new_nodes.columns)
                if bool(nodes[name].is_null().all() if polars else nodes[name].isna().all())]
    if null_old:
        nodes = nodes.drop(null_old) if polars else nodes.drop(columns=null_old)
    for name in set(nodes.columns).intersection(new_nodes.columns):
        old, new = nodes[name], new_nodes[name]
        old_null = old.is_null().all() if polars else old.isna().all()
        if bool(old_null):
            continue
        left = _column_family(old)
        right = _column_family(new)
        numeric = {"integer", "float"}
        incompatible = left != right and not {left, right} <= numeric
        if {left, right} <= numeric:
            for series, family in ((old, left), (new, right)):
                if family == "integer" and (
                    left != right or str(old.dtype).lower().startswith("u") != str(new.dtype).lower().startswith("u")
                ):
                    if int(series.min()) < -(1 << 53) or int(series.max()) > (1 << 53):
                        incompatible = True
        if incompatible:
            raise GFQLMutationError(
                ErrorCode.E108, "CREATE cannot combine these property types losslessly",
                field=str(name), value={"existing": str(old.dtype), "created": str(new.dtype)},
                language="cypher",
            )
    return nodes, new_nodes


def validate_create_node_reuse(creates: tuple[CreateClause, ...]) -> None:
    """Reject decorated re-declaration of a bound CREATE node before any effects."""
    bound: set[str] = set()
    for clause in creates:
        for pattern in clause.patterns:
            candidates: tuple[NodePattern, ...] = (
                (pattern,) if isinstance(pattern, NodePattern)
                else (pattern.source, pattern.destination)
            )
            for node in candidates:
                if node.variable is not None and node.variable in bound and (
                    node.labels or node.explicit_properties or node.properties
                ):
                    raise GFQLMutationError(
                        ErrorCode.E105, "CREATE node variable is already bound",
                        field="create", value=node.variable,
                        suggestion="Reference the bound node without labels or properties, or use a new variable.",
                        line=node.span.line, column=node.span.column, language="cypher",
                    )
                if node.variable is not None:
                    bound.add(node.variable)


def _validate_label_names(creates: tuple[CreateClause, ...]) -> None:
    """Reject labels whose names cannot be represented as non-null flag text."""
    for clause in creates:
        for pattern in clause.patterns:
            node_patterns = (
                (pattern,)
                if not isinstance(pattern, CreateRelationship)
                else (pattern.source, pattern.destination)
            )
            for node_pattern in node_patterns:
                unsupported_labels = sorted(set(node_pattern.labels) & NULLISH_LABEL_NAMES)
                if unsupported_labels:
                    raise GFQLMutationError(
                        ErrorCode.E108,
                        "CREATE cannot represent a null-like label name without silently dropping it",
                        field="create",
                        value=unsupported_labels,
                        suggestion="Use a label that is not None, nan, or <NA>.",
                        line=node_pattern.span.line,
                        column=node_pattern.span.column,
                        language="cypher",
                    )


def _validate_legacy_label_transition(
    creates: tuple[CreateClause, ...], nodes: Optional[DataFrameT]
) -> None:
    """Reject the lossy move from legacy ``type`` labels to canonical flags."""
    if nodes is None or len(nodes) == 0 or "type" not in nodes.columns:
        return
    if label_flag_columns(nodes.columns):
        return
    created_labels: set[Optional[str]] = set()
    for clause in creates:
        for pattern in clause.patterns:
            node_patterns = (
                (pattern,)
                if not isinstance(pattern, CreateRelationship)
                else (pattern.source, pattern.destination)
            )
            for node_pattern in node_patterns:
                created_labels.update(node_pattern.labels)
    if not created_labels:
        return
    type_series = nodes["type"]
    if is_polars_df(nodes):
        has_legacy_label = bool((~type_series.is_null()).any())
    else:
        has_legacy_label = bool(type_series.notna().any())
    if has_legacy_label:
        raise GFQLMutationError(
            ErrorCode.E108,
            "CREATE cannot preserve existing legacy type labels while adding canonical label flags",
            field="create",
            value=sorted(created_labels),
            suggestion="Use canonical label flags in the source graph, or CREATE without labels.",
            language="cypher",
        )


def execute_create_nodes(
    self: Plottable,
    creates: tuple[CreateClause, ...],
    *,
    engine: EngineAbstract = EngineAbstract.AUTO,
    params: Optional[CypherParams] = None,
) -> tuple[Plottable, dict[str, CypherScalar]]:
    """Create node rows and return a new Plottable without changing the caller."""
    validate_create_node_reuse(creates)
    _validate_label_names(creates)
    concrete_engine = resolve_engine(
        engine, self._edges if self._edges is not None else self._nodes,
    )
    if concrete_engine == Engine.POLARS_GPU:
        raise GFQLUnsupportedError(
            ErrorCode.E108,
            "CREATE does not yet have a native Polars GPU execution route",
            field="engine",
            value=concrete_engine.value,
            suggestion="Use engine='polars' on CPU or engine='cudf' for the native cuDF route.",
            language="cypher",
        )
    span = creates[0].span
    line, column = span.line, span.column

    from graphistry.compute.ComputeMixin import _coerce_input_formats
    base_graph = _coerce_input_formats(self, concrete_engine)
    if base_graph._gfql_graph is not None:
        base_graph = base_graph.bind()
        base_graph._gfql_graph = None
    if base_graph._nodes is None and base_graph._edges is not None:
        base_graph = base_graph.materialize_nodes(engine=concrete_engine.value)
    old_key = base_graph._node or "_node"
    nodes = base_graph._nodes
    if nodes is not None and old_key not in nodes.columns:
        raise GFQLMutationError(
            ErrorCode.E108, f"Existing node frame is missing identity column {old_key!r}",
            field=old_key, value="missing", line=line, column=column, language="cypher",
        )
    if base_graph._gfql_mutation_identity and old_key != MUTATION_NODE_ID:
        if nodes is not None and ({MUTATION_NODE_ID, ID_PROPERTY_PRESENT} & set(nodes.columns)):
            raise GFQLMutationError(
                ErrorCode.E108, "CREATE cannot reuse mutation metadata after rebinding physical identity",
                field=old_key, value="rebound_mutation_identity", language="cypher",
            )
        base_graph = base_graph.bind()
        base_graph._gfql_mutation_identity = False
    if base_graph._gfql_mutation_identity and nodes is not None and ID_PROPERTY_PRESENT not in nodes.columns:
        raise GFQLMutationError(
            ErrorCode.E108, "CREATE mutation metadata is incomplete",
            field=old_key, value="missing_property_membership", language="cypher",
        )
    if nodes is not None and not base_graph._gfql_mutation_identity:
        collisions = {MUTATION_NODE_ID, ID_PROPERTY_PRESENT} & set(nodes.columns)
        if collisions:
            raise GFQLMutationError(
                ErrorCode.E108, "CREATE reserved metadata columns collide with input columns",
                field="create", value=sorted(collisions), language="cypher",
            )
        from graphistry.compute.gfql.row.entity_props import _include_numeric_id_as_property
        include_id = _include_numeric_id_as_property(nodes)
        if is_polars_df(nodes):
            import polars as pl
            nodes = nodes.with_columns(
                pl.col(old_key).alias(MUTATION_NODE_ID),
                pl.lit(include_id).alias(ID_PROPERTY_PRESENT),
            )
        else:
            nodes = nodes.assign(**{
                MUTATION_NODE_ID: nodes[old_key], ID_PROPERTY_PRESENT: include_id,
            })
    node_key = MUTATION_NODE_ID
    _validate_legacy_label_transition(creates, nodes)
    node_count = 0
    for clause in creates:
        seen_nodes: set[str] = set()
        for pattern in clause.patterns:
            patterns = (pattern.source, pattern.destination) if isinstance(pattern, CreateRelationship) else (pattern,)
            for node_pattern in patterns:
                if node_pattern.variable is None or node_pattern.variable not in seen_nodes:
                    node_count += 1
                    if node_pattern.variable is not None:
                        seen_nodes.add(node_pattern.variable)
    identities: list[Union[int, str]]
    string_identity = nodes is not None and _column_family(nodes[node_key]) == "string"
    if nodes is not None:
        physical = nodes[node_key]
        nulls = physical.is_null().any() if is_polars_df(nodes) else physical.isna().any()
        duplicates = physical.n_unique() != len(physical) if is_polars_df(nodes) else physical.duplicated().any()
        if bool(nulls) or bool(duplicates):
            raise GFQLMutationError(ErrorCode.E105, "CREATE requires unique non-null physical identities", field=old_key, value="invalid_identity", language="cypher")
    if string_identity:
        offset = 0
        while True:
            identities = [f"__gfql_created_{offset + i}" for i in range(node_count)]
            values = nodes[node_key]
            hits = values.is_in(identities) if is_polars_df(nodes) else values.isin(identities)
            if not bool(hits.any()):
                break
            offset += node_count
    else:
        seed = 0 if nodes is None else _numeric_id_seed(
            nodes, node_key, count=node_count, line=line, column=column,
        )
        identities = list(range(seed, seed + node_count))

    relationships = [
        pattern
        for clause in creates
        for pattern in clause.patterns
        if isinstance(pattern, CreateRelationship)
    ]
    source_key = base_graph._source or "_source"
    destination_key = base_graph._destination or "_destination"
    edge_key = base_graph._edge
    old_edges = base_graph._edges
    edge_count = len(relationships)
    edge_identities: list[Union[int, str]] = []
    if edge_count:
        if old_edges is not None:
            missing_endpoint = {source_key, destination_key} - set(old_edges.columns)
            if missing_endpoint:
                raise GFQLMutationError(
                    ErrorCode.E108, "Existing edge frame is missing CREATE endpoint columns",
                    field=sorted(missing_endpoint)[0], value=sorted(missing_endpoint), line=line, column=column, language="cypher",
                )
            if edge_key is None:
                if MUTATION_EDGE_ID in old_edges.columns:
                    raise GFQLMutationError(
                        ErrorCode.E108,
                        "CREATE reserved edge metadata columns collide with input columns",
                        field="create", value=[MUTATION_EDGE_ID], language="cypher",
                    )
                edge_key = MUTATION_EDGE_ID
                old_edge_count = len(old_edges)
                edge_identities = list(range(edge_count))
                if is_polars_df(old_edges):
                    import polars as pl
                    if not isinstance(old_edges, pl.DataFrame):
                        raise GFQLUnsupportedError(
                            ErrorCode.E108, "CREATE requires materialized Polars frames",
                            field="engine", value="lazy_frame", language="cypher",
                        )
                    old_edges = old_edges.with_columns(
                        pl.Series(MUTATION_EDGE_ID, np.arange(-old_edge_count, 0, dtype=np.int64), dtype=pl.Int64),
                    )
                else:
                    if is_cudf_obj(old_edges):
                        import cupy as cp
                        import cudf
                        old_edge_identities = cudf.Series(
                            cp.arange(-old_edge_count, 0, dtype=cp.int64),
                            index=old_edges.index,
                            dtype=cudf.dtype("int64"),
                        )
                    else:
                        import pandas as pd
                        old_edge_identities = pd.Series(
                            np.arange(-old_edge_count, 0, dtype=np.int64),
                            index=old_edges.index,
                            dtype="int64",
                        )
                    old_edges = old_edges.assign(**{MUTATION_EDGE_ID: old_edge_identities})
            elif edge_key not in old_edges.columns:
                raise GFQLMutationError(
                    ErrorCode.E108, f"Existing edge frame is missing identity column {edge_key!r}",
                    field=edge_key, value="missing", line=line, column=column, language="cypher",
                )
            else:
                edge_family = _column_family(old_edges[edge_key])
                if edge_family == "string":
                    edge_offset = 0
                    while True:
                        edge_identities = [f"__gfql_edge_{edge_offset + i}" for i in range(edge_count)]
                        values = old_edges[edge_key]
                        if is_polars_df(old_edges):
                            hits = values.is_in(edge_identities)
                        else:
                            hits = values.isin(edge_identities)
                        if not bool(hits.any()):
                            break
                        edge_offset += edge_count
                elif edge_family == "integer":
                    edge_seed = _numeric_id_seed(old_edges, edge_key, count=edge_count, line=line, column=column)
                    edge_identities = list(range(edge_seed, edge_seed + edge_count))
                else:
                    raise GFQLMutationError(
                        ErrorCode.E108, "CREATE requires integer or string edge identities",
                        field=str(edge_key), value=str(old_edges[edge_key].dtype), line=line, column=column, language="cypher",
                    )
        else:
            edge_key = MUTATION_EDGE_ID
            edge_identities = list(range(edge_count))
        if edge_identities and old_edges is not None and edge_key in old_edges.columns:
            values = old_edges[edge_key]
            nulls = values.is_null().any() if is_polars_df(old_edges) else values.isna().any()
            duplicates = values.n_unique() != len(values) if is_polars_df(old_edges) else values.duplicated().any()
            if bool(nulls) or bool(duplicates):
                raise GFQLMutationError(
                    ErrorCode.E105, "CREATE requires unique non-null edge identities",
                    field=str(edge_key), value="invalid_identity", line=line, column=column, language="cypher",
                )

    bindings: dict[str, CypherScalar] = {"": identities[-1] if identities else None}
    identity_index = 0
    node_bindings: dict[str, CypherScalar] = {}
    relationship_bindings: dict[str, CypherScalar] = {}
    records: list[dict[str, object]] = []
    label_columns: set[str] = set()

    def _node_record(pattern: NodePattern) -> CypherScalar:
        nonlocal identity_index
        if pattern.variable is not None and pattern.variable in node_bindings:
            if pattern.labels or pattern.explicit_properties or pattern.properties:
                raise GFQLMutationError(
                    ErrorCode.E105, "CREATE node variable is already bound",
                    field="create", value=pattern.variable,
                    suggestion="Reference the bound node without labels or properties, or use a new variable.",
                    line=pattern.span.line, column=pattern.span.column, language="cypher",
                )
            return node_bindings[pattern.variable]
        identity = identities[identity_index]
        identity_index += 1
        row: dict[str, object] = {node_key: identity, ID_PROPERTY_PRESENT: False}
        for label in pattern.labels:
            label_column = f"label__{label}"
            label_columns.add(label_column)
            row[label_column] = True
        for entry in pattern.properties:
            if entry.key in {"labels", "type"} or entry.key.startswith(("label__", "__")):
                raise GFQLMutationError(
                    ErrorCode.E108, "CREATE properties cannot use reserved metadata or label namespaces",
                    field=entry.key, value=entry.key, line=entry.span.line, column=entry.span.column, language="cypher",
                )
            if entry.key in row:
                raise GFQLMutationError(
                    ErrorCode.E105, f"Duplicate CREATE property {entry.key!r}",
                    field=entry.key, value=entry.key,
                    suggestion="Use one value per property per node.",
                    line=entry.span.line, column=entry.span.column, language="cypher",
                )
            row[entry.key] = _property_value(entry.value, params=params, span=entry.span)
        row[ID_PROPERTY_PRESENT] = row.get("id") is not None
        records.append(row)
        if pattern.variable is not None:
            node_bindings[pattern.variable] = identity
        return identity

    edge_records: list[dict[str, object]] = []
    for clause in creates:
        for pattern in clause.patterns:
            if not isinstance(pattern, CreateRelationship):
                _node_record(pattern)
                continue
            if pattern.relationship.variable is not None and pattern.relationship.variable in relationship_bindings:
                raise GFQLMutationError(
                    ErrorCode.E105, "CREATE relationship variable is already bound",
                    field="create", value=pattern.relationship.variable,
                    line=pattern.relationship.span.line, column=pattern.relationship.span.column, language="cypher",
                )
            edge_identity = edge_identities[len(edge_records)]
            assert edge_key is not None
            if pattern.relationship.direction == "reverse":
                source_identity = _node_record(pattern.destination)
                destination_identity = _node_record(pattern.source)
            else:
                source_identity = _node_record(pattern.source)
                destination_identity = _node_record(pattern.destination)
            edge_row: dict[str, object] = {
                source_key: source_identity,
                destination_key: destination_identity,
                edge_key: edge_identity,
                "type": pattern.relationship.types[0],
            }
            for entry in pattern.relationship.properties:
                if entry.key in {"labels", "type"} or entry.key.startswith(("label__", "__")):
                    raise GFQLMutationError(
                        ErrorCode.E108, "CREATE edge properties cannot use reserved metadata or label namespaces",
                        field=entry.key, value=entry.key, line=entry.span.line, column=entry.span.column, language="cypher",
                    )
                if entry.key in edge_row:
                    raise GFQLMutationError(
                        ErrorCode.E105, f"Duplicate CREATE edge property {entry.key!r}",
                        field=entry.key, value=entry.key,
                        suggestion="Use one value per property per edge.",
                        line=entry.span.line, column=entry.span.column, language="cypher",
                    )
                edge_row[entry.key] = _property_value(entry.value, params=params, span=entry.span)
            edge_records.append(edge_row)
            if pattern.relationship.variable is not None:
                relationship_bindings[pattern.relationship.variable] = edge_identity

    if relationship_bindings:
        bindings.update(relationship_bindings)
    bindings.update(node_bindings)

    new_nodes = _new_nodes_frame(concrete_engine, records, label_columns)
    if nodes is not None:
        original_label_columns = {str(name) for name in nodes.columns if str(name).startswith("label__")}
        for label_column in sorted(original_label_columns):
            _boolean_label_column(nodes, label_column, line=line, column=column)
        nodes, new_nodes = _prepare_append_frames(nodes, new_nodes)
        if is_polars_df(nodes):
            import polars as pl
            if not isinstance(nodes, pl.DataFrame) or not isinstance(new_nodes, pl.DataFrame):
                raise GFQLUnsupportedError(
                    ErrorCode.E108, "CREATE requires materialized Polars frames",
                    field="engine", value="lazy_frame", language="cypher",
                )
            combined_nodes = pl.concat([nodes, new_nodes], how="diagonal_relaxed")
            label_columns_to_fill = [
                name for name in combined_nodes.columns if name.startswith("label__")
            ]
            if label_columns_to_fill:
                combined_nodes = combined_nodes.with_columns(
                    [
                        pl.col(column).fill_null(False).cast(pl.Boolean)
                        for column in label_columns_to_fill
                    ]
                )
            result = base_graph.nodes(combined_nodes, node_key)
        else:
            concat = df_concat(concrete_engine)
            nodes = concat([nodes, new_nodes], ignore_index=True, sort=False)
            label_columns_to_fill = [
                column for column in nodes.columns if column.startswith("label__")
            ]
            if label_columns_to_fill:
                nodes = nodes.assign(
                    **{
                        label_column: nodes[label_column].fillna(False).astype(bool)
                        for label_column in label_columns_to_fill
                    }
                )
            result = base_graph.nodes(nodes, node_key)
    else:
        if is_polars_df(new_nodes):
            import polars as pl

            if not isinstance(new_nodes, pl.DataFrame):
                raise GFQLUnsupportedError(
                    ErrorCode.E108, "CREATE requires materialized Polars frames",
                    field="engine", value="lazy_frame", language="cypher",
                )
            polars_new_nodes: pl.DataFrame = new_nodes
            for label_column in sorted(label_columns):
                polars_new_nodes = polars_new_nodes.with_columns(
                    pl.col(label_column).fill_null(False).cast(pl.Boolean)
                )
            nodes = polars_new_nodes
        else:
            for label_column in sorted(label_columns):
                new_nodes = new_nodes.assign(
                    **{label_column: new_nodes[label_column].fillna(False).astype(bool)}
                )
        nodes = new_nodes
        result = base_graph.nodes(nodes, node_key)

    if edge_records:
        assert edge_key is not None
        new_edges = _new_nodes_frame(concrete_engine, edge_records, set())
        if old_edges is not None:
            old_edges, new_edges = _prepare_append_frames(old_edges, new_edges)
            if is_polars_df(old_edges):
                import polars as pl
                if not isinstance(old_edges, pl.DataFrame) or not isinstance(new_edges, pl.DataFrame):
                    raise GFQLUnsupportedError(
                        ErrorCode.E108, "CREATE requires materialized Polars frames",
                        field="engine", value="lazy_frame", language="cypher",
                    )
                combined_edges = pl.concat([old_edges, new_edges], how="diagonal_relaxed")
                result = result.edges(combined_edges, source_key, destination_key, edge_key)
            else:
                edges = df_concat(concrete_engine)([old_edges, new_edges], ignore_index=True, sort=False)
                result = result.edges(edges, source_key, destination_key, edge_key)
        else:
            edges = new_edges
            result = result.edges(edges, source_key, destination_key, edge_key)
    result._gfql_mutation_identity = True
    return result, bindings

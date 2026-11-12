from __future__ import annotations

import math
from typing import Optional

from graphistry.compute.typing import DataFrameT, SeriesT
from graphistry.Engine import (
    Engine,
    EngineAbstract,
    df_concat,
    df_to_engine,
    is_polars_df,
    resolve_engine,
)
from graphistry.compute.exceptions import ErrorCode, GFQLUnsupportedError, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CreateClause, CypherParams, ParameterRef
from graphistry.Plottable import Plottable


class GFQLMutationError(GFQLValidationError):
    """Typed failure surface for unsupported or invalid mutation inputs."""


def validate_create_parameters(creates: tuple[CreateClause, ...], params: Optional[CypherParams]) -> None:
    """Validate finite query values without constructing or appending graph rows."""
    for clause in creates:
        for pattern in clause.patterns:
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
        return params[value.name]
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
    series = nodes[node_key]
    if not len(nodes):
        return 0
    from graphistry.Engine import is_polars_df

    if is_polars_df(nodes):
        import polars as pl
        has_null = bool(series.is_null().any())
        dtype = series.dtype
        is_numeric = dtype.is_numeric()
        is_boolean = dtype == pl.Boolean
        is_integer = dtype.is_integer()
    else:
        has_null = bool(series.isna().any())
        dtype = series.dtype
        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        is_boolean = pd.api.types.is_bool_dtype(dtype)
        is_integer = pd.api.types.is_integer_dtype(dtype)
    if has_null:
        raise GFQLMutationError(
            ErrorCode.E105,
            f"Node identity column {node_key!r} contains null values",
            field=node_key,
            value="null",
            suggestion="Provide complete node identities or remove nulls before CREATE.",
            line=line,
            column=column,
            language="cypher",
        )
    if not is_numeric or is_boolean or not is_integer:
        raise GFQLMutationError(
            ErrorCode.E108,
            f"CREATE requires integer numeric node identities; {node_key!r} has dtype {str(series.dtype)!r}",
            field=node_key,
            value=str(series.dtype),
            suggestion="Numeric identities use max(existing)+1; other identity schemes require an explicit generator.",
            line=line,
            column=column,
            language="cypher",
        )
    maximum = int(series.max())
    if is_polars_df(nodes):
        import polars as pl
        if dtype.is_integer():
            limits = {
                pl.Int8: (1 << 7) - 1, pl.Int16: (1 << 15) - 1,
                pl.Int32: (1 << 31) - 1, pl.Int64: (1 << 63) - 1,
                pl.UInt8: (1 << 8) - 1, pl.UInt16: (1 << 16) - 1,
                pl.UInt32: (1 << 32) - 1, pl.UInt64: (1 << 64) - 1,
            }
            if dtype not in limits:
                raise GFQLMutationError(
                    ErrorCode.E108, "CREATE does not support this physical identity width",
                    field=node_key, value=str(dtype), language="cypher",
                )
            limit = limits[dtype]
            if maximum + count > limit:
                raise GFQLMutationError(
                    ErrorCode.E105,
                    f"Node identity column {node_key!r} has no remaining values",
                    field=node_key,
                    value=maximum,
                    suggestion="Widen the identity column or provide an explicit generator.",
                    line=line,
                    column=column,
                    language="cypher",
                )
    elif pd.api.types.is_integer_dtype(dtype):
        info = np.iinfo(dtype.type)
        if maximum + count > info.max:
            raise GFQLMutationError(
                ErrorCode.E105,
                f"Node identity column {node_key!r} has no remaining values",
                field=node_key,
                value=maximum,
                suggestion="Widen the identity column or provide an explicit generator.",
                line=line,
                column=column,
                language="cypher",
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
    if _column_family(nodes[label_column], polars=is_polars_df(nodes)) != "bool":
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


def _column_family(series: SeriesT, *, polars: bool) -> str:
    import pandas as pd

    dtype = series.dtype
    if polars:
        import polars as pl
        if dtype == pl.Boolean:
            return "bool"
        if dtype.is_integer():
            return "integer"
        if dtype.is_float():
            return "float"
        return "string" if dtype == pl.String else "other"
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
        nullable = {}
        for name in nodes.columns:
            dtype = nodes[name].dtype
            if pd.api.types.is_integer_dtype(dtype):
                target = pd.UInt64Dtype() if pd.api.types.is_unsigned_integer_dtype(dtype) else pd.Int64Dtype()
                nullable[name] = nodes[name].astype(target)
            elif pd.api.types.is_bool_dtype(dtype):
                nullable[name] = nodes[name].astype("boolean")
        if nullable:
            nodes = nodes.assign(**nullable)
    for name in set(nodes.columns).intersection(new_nodes.columns):
        old, new = nodes[name], new_nodes[name]
        old_null = old.is_null().all() if polars else old.isna().all()
        if bool(old_null):
            continue
        left = _column_family(old, polars=polars)
        right = _column_family(new, polars=polars)
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


def execute_create_nodes(
    self: Plottable,
    creates: tuple[CreateClause, ...],
    *,
    engine: EngineAbstract | str = EngineAbstract.AUTO,
    params: Optional[CypherParams] = None,
) -> Plottable:
    """Create node rows and return a new Plottable without changing the caller."""
    concrete_engine = resolve_engine(
        EngineAbstract(engine) if isinstance(engine, str) else engine,
        self._edges if self._edges is not None else self._nodes,
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
    if base_graph._nodes is None and base_graph._edges is not None:
        base_graph = base_graph.materialize_nodes(engine=concrete_engine.value)
    node_key = base_graph._node or "_node"
    nodes = base_graph._nodes
    if nodes is None and node_key != "_node":
        raise GFQLMutationError(
            ErrorCode.E108,
            f"Bound node key {node_key!r} must use the existing identity convention",
            field=node_key,
            value=node_key,
            suggestion="Create a graph with numeric node identities before CREATE.",
            line=line,
            column=column,
            language="cypher",
        )

    identity = 0
    if nodes is not None and node_key not in nodes.columns:
        raise GFQLMutationError(
            ErrorCode.E108,
            f"Existing node frame is missing identity column {node_key!r}",
            field=node_key,
            value="missing",
            suggestion="Provide a node frame with the bound identity column before CREATE.",
            line=line,
            column=column,
            language="cypher",
        )
    if nodes is not None:
        identity = _numeric_id_seed(
            nodes, node_key, count=sum(len(clause.patterns) for clause in creates),
            line=line, column=column,
        )

    records: list[dict[str, object]] = []
    label_columns: set[str] = set()
    property_columns: set[str] = set()
    for clause in creates:
        for pattern in clause.patterns:
            row: dict[str, object] = {node_key: identity}
            identity += 1
            for label in pattern.labels:
                label_column = f"label__{label}"
                label_columns.add(label_column)
                row[label_column] = True
            for entry in pattern.properties:
                if entry.key.startswith("label__"):
                    raise GFQLMutationError(
                        ErrorCode.E108, "CREATE properties cannot use the label flag namespace",
                        field=entry.key, value=entry.key, language="cypher",
                    )
                if entry.key in row:
                    raise GFQLMutationError(
                        ErrorCode.E105,
                        f"Duplicate CREATE property {entry.key!r}",
                        field=entry.key,
                        value=entry.key,
                        suggestion="Use one value per property per node.",
                        line=entry.span.line,
                        column=entry.span.column,
                        language="cypher",
                    )
                row[entry.key] = _property_value(
                    entry.value, params=params, span=entry.span
                )
                property_columns.add(entry.key)
            records.append(row)

    new_nodes = _new_nodes_frame(concrete_engine, records, label_columns)
    if nodes is not None:
        original_label_columns = {str(name) for name in nodes.columns if str(name).startswith("label__")}
        for label_column in sorted(original_label_columns):
            _boolean_label_column(nodes, label_column, line=line, column=column)
        nodes, new_nodes = _prepare_append_frames(nodes, new_nodes)
        if is_polars_df(nodes):
            import polars as pl
            nodes = pl.concat([nodes, new_nodes], how="diagonal_relaxed")
            label_columns_to_fill = [
                column for column in nodes.columns if column.startswith("label__")
            ]
            if label_columns_to_fill:
                nodes = nodes.with_columns(
                    [
                        pl.col(column).fill_null(False).cast(pl.Boolean)
                        for column in label_columns_to_fill
                    ]
                )
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
        for property_column in sorted(property_columns):
            if is_polars_df(nodes):
                continue
            declared = [
                record[property_column]
                for record in records
                if property_column in record
            ]
            incomplete = len(declared) != len(nodes)
            if incomplete and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in declared
            ):
                nodes = nodes.assign(
                    **{property_column: nodes[property_column].astype("Int64")}
                )
    else:
        if is_polars_df(new_nodes):
            import polars as pl

            for label_column in sorted(label_columns):
                new_nodes = new_nodes.with_columns(
                    pl.col(label_column).fill_null(False).cast(pl.Boolean)
                )
        else:
            for label_column in sorted(label_columns):
                new_nodes = new_nodes.assign(
                    **{label_column: new_nodes[label_column].fillna(False).astype(bool)}
                )
        nodes = new_nodes

    return base_graph.nodes(nodes, node_key)

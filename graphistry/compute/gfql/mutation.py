from __future__ import annotations

from typing import Optional

from graphistry.Engine import (
    Engine,
    EngineAbstract,
    align_shared_column_dtypes,
    df_concat,
    df_to_engine,
    is_polars_df,
    resolve_engine,
)
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CypherLiteral, CreateClause, ParameterRef
from graphistry.Plottable import Plottable


class GFQLMutationError(GFQLValidationError):
    """Typed failure surface for unsupported or invalid mutation inputs."""


def _property_value(value: object, *, params: Optional[dict[str, object]], span: object) -> object:
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
    if isinstance(value, CypherLiteral):
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


def _new_nodes_frame(engine: Engine, records: list[dict[str, object]], label_columns: set[str]):
    if engine in {Engine.PANDAS, Engine.CUDF, Engine.POLARS, Engine.POLARS_GPU}:
        import pandas as pd
        columns: list[str] = []
        for record in records:
            for key in record:
                if key not in columns:
                    columns.append(key)
        frame = pd.DataFrame.from_records(records, columns=columns)
        column_updates: dict[str, object] = {}
        for column in frame.columns:
            values = frame[column]
            if column in label_columns:
                column_updates[column] = values.astype("boolean")
                continue
            declared = [record[column] for record in records if column in record]
            incomplete = len(declared) != len(values)
            if incomplete and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in declared
            ):
                column_updates[column] = values.astype("Int64")
            elif incomplete and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in declared
            ):
                column_updates[column] = values.astype("Float64")
        if column_updates:
            frame = frame.assign(**column_updates)
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


def _numeric_id_seed(nodes, node_key: str, *, line: int, column: int):
    import numpy as np
    import pandas as pd

    if node_key not in nodes.columns:
        return 0
    series = nodes[node_key]
    from graphistry.Engine import is_polars_df

    if is_polars_df(nodes):
        import polars as pl
        has_null = bool(series.is_null().any())
        dtype = series.dtype
        is_numeric = dtype.is_numeric()
        is_boolean = dtype == pl.Boolean
    else:
        has_null = bool(series.isna().any())
        dtype = series.dtype
        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        is_boolean = pd.api.types.is_bool_dtype(dtype)
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
    if not is_numeric or is_boolean:
        raise GFQLMutationError(
            ErrorCode.E108,
            f"CREATE requires numeric node identities; {node_key!r} has dtype {str(series.dtype)!r}",
            field=node_key,
            value=str(series.dtype),
            suggestion="Numeric identities use max(existing)+1; other identity schemes require an explicit generator.",
            line=line,
            column=column,
            language="cypher",
        )
    maximum = int(series.max()) if len(nodes) else 0
    if is_polars_df(nodes):
        import polars as pl
        if dtype.is_integer():
            limit = ((1 << 64) - 1) if dtype.is_unsigned_integer() else ((1 << 63) - 1)
            if maximum == limit:
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
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if maximum == info.max:
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


def _boolean_label_column(nodes, label_column: str, *, line: int, column: int):
    import pandas as pd

    if label_column not in nodes.columns:
        return None
    if not pd.api.types.is_bool_dtype(nodes[label_column].dtype):
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


def execute_create_nodes(
    self: Plottable,
    creates: tuple[CreateClause, ...],
    *,
    engine: EngineAbstract | Engine = EngineAbstract.AUTO,
    params: Optional[dict[str, object]] = None,
) -> Plottable:
    """Create node rows and return a new Plottable without changing the caller."""
    concrete_engine = resolve_engine(
        engine, self._edges if self._edges is not None else self._nodes
    )
    span = creates[0].span
    line, column = span.line, span.column

    node_key = self._node or "_node"
    nodes = self._nodes
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
        identity = _numeric_id_seed(nodes, node_key, line=line, column=column)

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
        original_label_columns = {
            label_column for label_column in label_columns if label_column in nodes.columns
        }
        for label_column in sorted(original_label_columns):
            _boolean_label_column(nodes, label_column, line=line, column=column)
        new_nodes = align_shared_column_dtypes(nodes, new_nodes)
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
        import polars as pl
        for label_column in sorted(label_columns):
            if is_polars_df(new_nodes):
                new_nodes = new_nodes.with_columns(
                    pl.col(label_column).fill_null(False).cast(pl.Boolean)
                )
            else:
                new_nodes = new_nodes.assign(
                    **{label_column: new_nodes[label_column].fillna(False).astype(bool)}
                )
        nodes = new_nodes

    return self.nodes(nodes, node_key)

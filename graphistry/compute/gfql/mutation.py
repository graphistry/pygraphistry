from __future__ import annotations

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
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CreateClause, CypherParams, ParameterRef
from graphistry.Plottable import Plottable


class GFQLMutationError(GFQLValidationError):
    """Typed failure surface for unsupported or invalid mutation inputs."""


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


def _numeric_id_seed(nodes: DataFrameT, node_key: str, *, line: int, column: int) -> int:
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


def _require_id_capacity(
    nodes: DataFrameT,
    node_key: str,
    *,
    identities: int,
    line: int,
    column: int,
) -> None:
    """Reject identity allocation only when the existing integer dtype cannot hold it."""
    import numpy as np

    if nodes is None or node_key not in nodes.columns or identities <= 0:
        return
    series = nodes[node_key]
    if is_polars_df(nodes):
        import polars as pl
        if series.dtype.is_integer():
            limit = ((1 << 64) - 1) if series.dtype.is_unsigned_integer() else ((1 << 63) - 1)
        else:
            return
    else:
        import pandas as pd
        dtype = series.dtype
        if pd.api.types.is_integer_dtype(dtype):
            info = np.iinfo(dtype.numpy_dtype if hasattr(dtype, "numpy_dtype") else dtype)
            limit = int(info.max)
        else:
            return
    maximum = int(series.max())
    if maximum > limit or identities > limit - maximum:
        raise GFQLMutationError(
            ErrorCode.E105,
            f"Node identity column {node_key!r} cannot hold {identities} new IDs",
            field=node_key,
            value=str(series.dtype),
            suggestion="Widen the identity column or provide an explicit generator.",
            line=line,
            column=column,
            language="cypher",
        )


def _promote_shared_column_dtype(
    reference_dtype: object,
    candidate_dtype: object,
    *,
    column: str,
    line: int,
    column_position: int,
) -> object:
    """Return a lossless common dtype or raise, never coercing candidate to reference."""
    import pandas as pd

    if str(reference_dtype) == str(candidate_dtype):
        return reference_dtype
    ref_is_numeric = pd.api.types.is_numeric_dtype(reference_dtype)
    cand_is_numeric = pd.api.types.is_numeric_dtype(candidate_dtype)
    ref_is_bool = pd.api.types.is_bool_dtype(reference_dtype)
    cand_is_bool = pd.api.types.is_bool_dtype(candidate_dtype)
    if not ref_is_numeric and not ref_is_bool:
        raise GFQLMutationError(
            ErrorCode.E108,
            f"CREATE property {column!r} has incompatible dtypes {str(reference_dtype)!r} and {str(candidate_dtype)!r}",
            field=column,
            value=str(candidate_dtype),
            suggestion="Use the same property dtype as existing nodes; implicit mixed-type conversion is not lossless.",
            line=line,
            column=column_position,
            language="cypher",
        )
    if not cand_is_numeric and not cand_is_bool:
        raise GFQLMutationError(
            ErrorCode.E108,
            f"CREATE property {column!r} has incompatible dtypes {str(reference_dtype)!r} and {str(candidate_dtype)!r}",
            field=column,
            value=str(candidate_dtype),
            suggestion="Use the same property dtype as existing nodes; implicit mixed-type conversion is not lossless.",
            line=line,
            column=column_position,
            language="cypher",
        )
    if ref_is_bool and cand_is_bool:
        return reference_dtype
    if ref_is_bool or cand_is_bool:
        other = candidate_dtype if ref_is_bool else reference_dtype
        return "Float64" if pd.api.types.is_float_dtype(other) else "Int64"
    if pd.api.types.is_float_dtype(reference_dtype) or pd.api.types.is_float_dtype(candidate_dtype):
        return "Float64"
    return "Int64"


def _promote_shared_columns_losslessly(
    nodes: DataFrameT,
    new_nodes: DataFrameT,
    *,
    records: list[dict[str, object]],
    property_columns: set[str],
    line: int,
    column: int,
) -> tuple[DataFrameT, DataFrameT]:
    """Promote both frames to a common lossless dtype, leaving caller input unchanged."""
    if is_polars_df(nodes):
        return nodes, new_nodes
    shared_columns = [value for value in new_nodes.columns if value in nodes.columns]
    for shared_column in shared_columns:
        reference_dtype = nodes[shared_column].dtype
        candidate_dtype = new_nodes[shared_column].dtype
        if str(reference_dtype) == str(candidate_dtype):
            continue
        target_dtype = _promote_shared_column_dtype(
            reference_dtype,
            candidate_dtype,
            column=shared_column,
            line=line,
            column_position=column,
        )
        nodes = nodes.assign(**{shared_column: nodes[shared_column].astype(target_dtype)})
        new_nodes = new_nodes.assign(
            **{shared_column: new_nodes[shared_column].astype(target_dtype)}
        )
    return nodes, new_nodes


def _infer_nodes_from_edges(
    self: Plottable,
    engine: Engine,
    node_key: str,
    *,
    line: int,
    column: int,
) -> DataFrameT:
    """Materialize endpoint identities implied by an edges-only graph without changing edges."""
    edges = df_to_engine(self._edges, engine)
    if edges is None:
        raise GFQLMutationError(
            ErrorCode.E108,
            "CREATE requires node or edge data from which identities can be inferred",
            field="nodes",
            value=None,
            suggestion="Bind nodes or edges before CREATE.",
            line=line,
            column=column,
            language="cypher",
        )
    source_key = self._source
    destination_key = self._destination
    if source_key is None or destination_key is None or source_key not in edges.columns or destination_key not in edges.columns:
        raise GFQLMutationError(
            ErrorCode.E108,
            "Edges-only CREATE requires bound source and destination columns",
            field="edges",
            value=None,
            suggestion="Call .edges(edges, source, destination) before CREATE.",
            line=line,
            column=column,
            language="cypher",
        )
    if is_polars_df(edges):
        import polars as pl
        endpoints = pl.concat([edges[source_key], edges[destination_key]], rechunk=True).unique()
        return pl.DataFrame({node_key: endpoints})
    endpoints = df_concat(engine)([edges[source_key], edges[destination_key]], ignore_index=True)
    endpoints = endpoints.drop_duplicates().reset_index(drop=True)
    if engine == Engine.CUDF:
        import cudf
        return cudf.DataFrame({node_key: endpoints})
    import pandas as pd
    return pd.DataFrame({node_key: endpoints})


def _boolean_label_column(
    nodes: DataFrameT,
    label_column: str,
    *,
    line: int,
    column: int,
) -> Optional[SeriesT]:
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
    engine: EngineAbstract | str = EngineAbstract.AUTO,
    params: Optional[CypherParams] = None,
) -> Plottable:
    """Create node rows and return a new Plottable without changing the caller."""
    span = creates[0].span
    line, column = span.line, span.column
    concrete_engine = resolve_engine(
        EngineAbstract(engine) if isinstance(engine, str) else engine,
        self._edges if self._edges is not None else self._nodes,
    )
    if concrete_engine == Engine.POLARS_GPU:
        from graphistry.compute.gfql.lazy import polars_gpu_available
        if not polars_gpu_available():
            raise GFQLMutationError(
                ErrorCode.E108,
                "GFQL engine='polars-gpu' requires a usable GPU; CREATE declined rather than serving CPU Polars",
                field="engine",
                value=concrete_engine.value,
                suggestion="Use engine='polars' for CPU execution or run on a GPU-capable host.",
                line=line,
                column=column,
                language="cypher",
            )

    node_key = self._node or "_node"
    nodes = self._nodes
    inferred_nodes = False
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
    if nodes is None:
        nodes = _infer_nodes_from_edges(
            self,
            concrete_engine,
            node_key,
            line=line,
            column=column,
        )
        inferred_nodes = True

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
    _require_id_capacity(
        nodes,
        node_key,
        identities=sum(len(create_clause.patterns) for create_clause in creates),
        line=line,
        column=column,
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
        nodes, new_nodes = _promote_shared_columns_losslessly(
            nodes,
            new_nodes,
            records=records,
            property_columns=property_columns,
            line=line,
            column=column,
        )
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

    result = self.nodes(nodes, node_key)
    if inferred_nodes:
        result._gfql_graph = result
    return result

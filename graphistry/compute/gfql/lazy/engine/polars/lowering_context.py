"""Per-engine lowering context for the native polars GFQL engine.

Contextvars threaded around a polars row-op lowering pass so the pure expr-lowering helpers
(``lower_expr`` and friends) can read ambient state without plumbing it through every signature.
Set + reset them as a pair around a lowering call — see ``_lower_with_schema`` in ``row_pipeline.py``.

Declared together in one place per engine (the contextvar analogue of ``reserved_columns.py``) so
the lowering's ambient state is discoverable rather than inlined across the executor. New
polars-lowering contextvars SHOULD be added here.
"""
import contextvars
from typing import Optional

#: Table schema (column name -> polars dtype) of the frame being lowered — lets float-operand
#: inference run for the NaN guard without a scan.
SCHEMA: "contextvars.ContextVar[dict]" = contextvars.ContextVar("gfql_polars_schema", default={})

#: Active graph node-id column name — lets ``lower_expr`` resolve the bare whole-entity identity
#: sentinel (``same_path_types.NODE_IDENTITY_COLUMN``) to the real id column. None when unknown, so
#: the identity sentinel then declines to an honest NIE rather than resolving to a wrong column.
NODE_ID: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("gfql_polars_node_id", default=None)

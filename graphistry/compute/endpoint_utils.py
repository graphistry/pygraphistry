from __future__ import annotations

import typing

from graphistry.Engine import is_polars_df
from graphistry.compute.typing import DataFrameT

# Preserve the caller's pandas/cuDF/Polars frame flavor; DataFrameT is pandas-only in type checks.
EndpointFrameT = typing.TypeVar("EndpointFrameT")


def _drop_null_endpoint_edges_pandas_cudf(
    frame: DataFrameT, source: str, destination: str
) -> DataFrameT:
    source_null = frame[source].isna()
    destination_null = frame[destination].isna()
    if not (bool(source_null.any()) or bool(destination_null.any())):
        return frame
    return frame.loc[~(source_null | destination_null)]


def drop_null_endpoint_edges(
    frame: EndpointFrameT, source: str, destination: str
) -> EndpointFrameT:
    """Return only edges whose source and destination are identities."""
    if is_polars_df(frame):
        import polars as pl

        result: DataFrameT = frame.filter(
            pl.col(source).is_not_null() & pl.col(destination).is_not_null()
        )
        return result
    result = _drop_null_endpoint_edges_pandas_cudf(frame, source, destination)
    return result

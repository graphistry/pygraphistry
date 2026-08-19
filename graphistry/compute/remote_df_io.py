"""Client-side decoding policy for remote GFQL / remote Python results.

CSV is an untyped wire format: the server writes text and the client cannot
recover the original dtypes from it, so a bare reader re-infers them. Callers
are warned, and may pass explicit reader kwargs to take control. ``parquet``
carries an Arrow schema and is the faithful default.
"""
from inspect import getmodule
import warnings
from typing import BinaryIO, Callable, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError
from graphistry.compute.typing import DataFrameT
from graphistry.models.compute.chain_remote import DFImportArgs


CSV_LOSSY_WARNING = (
    "format='csv' is untyped on the wire: the client re-infers dtypes from text and can "
    "rewrite values ('007' -> 7.0, '08' -> 8.0, and 'NA'/''/'null' -> NaN). "
    "format='parquet' (the default) carries an Arrow schema and is faithful. "
    "To control the csv reader yourself, pass df_import_args, e.g. "
    "df_import_args={'dtype': str, 'keep_default_na': False, 'na_values': []}."
)


def _frame_type_name(df: Optional[DataFrameT]) -> str:
    if df is None:
        return "None"
    return f"{type(df).__module__.split('.')[0]}.{type(df).__name__}"


def _is_pandas_like(df: Optional[DataFrameT]) -> bool:
    import pandas as pd
    return df is None or isinstance(df, pd.DataFrame) or 'unittest.mock' in str(type(df))


def require_supported_frame_library(
    nodes: Optional[DataFrameT], edges: Optional[DataFrameT], api_name: str
) -> str:
    """Resolve which DataFrame library backs a remote call, before any request is sent.

    :param nodes: The graph's node frame, or ``None``.
    :param edges: The graph's edge frame, or ``None``.
    :param api_name: Public entry point named in the error message.
    :return: ``"cudf"`` or ``"pandas"``.
    :raises GFQLRemoteError: When either frame is some other library (e.g. polars).
    """
    if any('cudf.core.dataframe' in str(getmodule(df)) for df in (nodes, edges) if df is not None):
        return "cudf"
    if _is_pandas_like(nodes) and _is_pandas_like(edges):
        return "pandas"
    raise GFQLRemoteError(
        ErrorCode.E404,
        f"{api_name}: remote execution supports pandas and cudf frames; got "
        f"nodes={_frame_type_name(nodes)}, edges={_frame_type_name(edges)}. "
        f"Convert with .to_pandas() before calling, or run this query locally.",
    )


def validate_csv_import_args(
    df_import_args: Optional[DFImportArgs],
    api_name: str,
) -> None:
    """Reject a malformed ``df_import_args`` before any request is sent.

    Type validation only: supplying nothing is legitimate and is handled at decode.

    :param df_import_args: Caller-supplied reader kwargs, or ``None``.
    :param api_name: Public entry point named in the error message.
    :raises GFQLRemoteError: When supplied but not a dict.
    """
    if df_import_args is not None and not isinstance(df_import_args, dict):
        raise GFQLRemoteError(
            ErrorCode.E403,
            f"{api_name}: df_import_args must be a dict of reader kwargs, got: {type(df_import_args)}",
            field="df_import_args",
        )


def resolve_csv_import_args(
    df_import_args: Optional[DFImportArgs],
    api_name: str,
) -> DFImportArgs:
    """Resolve csv reader kwargs, warning when the caller left decoding to inference.

    :param df_import_args: Caller-supplied reader kwargs; ``None`` means none supplied.
    :param api_name: Public entry point named in the message.
    :return: Reader kwargs to apply.
    :raises GFQLRemoteError: When ``df_import_args`` is supplied but is not a dict.
    """
    if df_import_args is None:
        warnings.warn(f"{api_name}: {CSV_LOSSY_WARNING}", UserWarning, stacklevel=3)
        return {}
    validate_csv_import_args(df_import_args, api_name)
    return df_import_args


def resolve_csv_reader(
    read_csv: Callable[..., DataFrameT],
    df_import_args: Optional[DFImportArgs],
    api_name: str,
) -> Callable[[BinaryIO], DataFrameT]:
    """Bind a csv reader that applies the caller's explicit reader kwargs.

    :param read_csv: Engine-specific reader (``pandas.read_csv`` or ``cudf.read_csv``).
    :param df_import_args: Caller-supplied reader kwargs; ``None`` means no opt-in.
    :param api_name: Public entry point named in the error message.
    :return: Callable taking a buffer and returning a DataFrame.
    """
    args = resolve_csv_import_args(df_import_args, api_name)

    def read(buf: BinaryIO) -> DataFrameT:
        return read_csv(buf, **args)

    return read

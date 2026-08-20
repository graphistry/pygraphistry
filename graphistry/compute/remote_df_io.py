"""Client-side decoding policy for remote GFQL / remote Python results.

CSV is an untyped wire format: the server writes text and the client cannot
recover the original dtypes from it, so a bare reader re-infers them. Callers
are warned unless their reader kwargs govern both lossy axes -- dtype inference
and NA substitution -- which are independent: ``dtype=str`` still turns ``'NA'``
into ``NaN``, and ``keep_default_na=False`` still turns ``'007'`` into ``7``.
``parquet`` carries an Arrow schema and is the faithful default.
"""
from inspect import getmodule
import warnings
from typing import BinaryIO, Callable, List, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError
from graphistry.compute.typing import DataFrameT
from graphistry.models.compute.chain_remote import DFImportArgs


CSV_DTYPE_KWARGS = frozenset({'converters', 'dtype'})
CSV_NA_KWARGS = frozenset({'converters', 'keep_default_na', 'na_filter', 'na_values'})

CSV_DTYPE_AXIS_WARNING = (
    "dtype inference is left to the reader, which retypes text ('007' -> 7.0, '08' -> 8.0); "
    "govern it with a " + " or ".join(sorted(CSV_DTYPE_KWARGS)) + " reader kwarg"
)
CSV_NA_AXIS_WARNING = (
    "NA substitution is left to the reader, which blanks the pandas NA vocabulary "
    "('NA'/''/'null' -> NaN); govern it with a "
    + ", ".join(sorted(CSV_NA_KWARGS)) + " reader kwarg"
)
CSV_LOSSY_REMEDY = (
    "format='parquet' (the default) carries an Arrow schema and is faithful. "
    "For a faithful csv read pass df_import_args, e.g. "
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


def ungoverned_csv_axes(df_import_args: Optional[DFImportArgs]) -> List[str]:
    """Name the lossy csv axes the caller's reader kwargs do not govern.

    :param df_import_args: Caller-supplied reader kwargs, or ``None``.
    :return: Zero, one, or two axis descriptions; empty means the read is under caller control.
    """
    keys = set(df_import_args or {})
    axes: List[str] = []
    if not (keys & CSV_DTYPE_KWARGS):
        axes.append(CSV_DTYPE_AXIS_WARNING)
    if not (keys & CSV_NA_KWARGS):
        axes.append(CSV_NA_AXIS_WARNING)
    return axes


def resolve_csv_import_args(
    df_import_args: Optional[DFImportArgs],
    api_name: str,
) -> DFImportArgs:
    """Resolve csv reader kwargs, warning per lossy axis the caller left to inference.

    :param df_import_args: Caller-supplied reader kwargs; ``None`` means none supplied.
    :param api_name: Public entry point named in the message.
    :return: Reader kwargs to apply.
    :raises GFQLRemoteError: When ``df_import_args`` is supplied but is not a dict.
    """
    validate_csv_import_args(df_import_args, api_name)
    axes = ungoverned_csv_axes(df_import_args)
    if axes:
        warnings.warn(
            f"{api_name}: format='csv' is untyped on the wire and this read is not fully "
            f"under your control: {'; '.join(axes)}. {CSV_LOSSY_REMEDY}",
            UserWarning,
            stacklevel=3,
        )
    return {} if df_import_args is None else df_import_args


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

"""Client-side decoding policy for remote GFQL / remote Python results.

CSV is an untyped wire format: the server writes text and the client cannot
recover the original dtypes from it, so a bare reader re-infers them. Callers
are warned, and may pass explicit reader kwargs to take control. ``parquet``
carries an Arrow schema and is the faithful default.
"""
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

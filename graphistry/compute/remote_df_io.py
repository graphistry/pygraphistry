"""Client-side decoding policy for remote GFQL / remote Python results.

CSV is an untyped wire format: the server writes text and the client cannot
recover the original dtypes from it, so a bare reader re-infers them.
"""
from typing import BinaryIO, Callable, Optional

from graphistry.compute.exceptions import ErrorCode, GFQLRemoteError
from graphistry.compute.typing import DataFrameT
from graphistry.models.compute.chain_remote import DFImportArgs


CSV_LOSSY_HINT = (
    "format='csv' cannot round-trip dtypes: the server sends untyped text, so the client "
    "re-infers types and can silently rewrite values ('007' -> 7.0, '08' -> 8.0, and "
    "'NA'/''/'null' -> NaN) and break the node/edge id join of the returned graph. "
    "Use format='parquet' for a faithful result. "
    "To read csv anyway, pass df_import_args to take explicit control of the client-side "
    "reader, e.g. df_import_args={'dtype': str, 'keep_default_na': False, 'na_values': []} "
    "to keep every column as written, or df_import_args={} to accept re-inferred dtypes."
)


def require_csv_opt_in(
    df_import_args: Optional[DFImportArgs],
    api_name: str,
) -> DFImportArgs:
    """Decline a csv response unless the caller took explicit control of the reader.

    :param df_import_args: Caller-supplied reader kwargs; ``None`` means no opt-in.
    :param api_name: Public entry point named in the error message.
    :return: The validated reader kwargs.
    :raises GFQLRemoteError: When ``df_import_args`` is ``None`` or not a dict.
    """
    if df_import_args is None:
        raise GFQLRemoteError(
            ErrorCode.E403,
            f"{api_name}: {CSV_LOSSY_HINT}",
            field="df_import_args",
        )
    if not isinstance(df_import_args, dict):
        raise GFQLRemoteError(
            ErrorCode.E403,
            f"{api_name}: df_import_args must be a dict of reader kwargs, got: {type(df_import_args)}"
        )
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
    args = require_csv_opt_in(df_import_args, api_name)

    def read(buf: BinaryIO) -> DataFrameT:
        return read_csv(buf, **args)

    return read

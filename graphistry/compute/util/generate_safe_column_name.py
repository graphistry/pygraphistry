"""
Generate safe column names that avoid conflicts with existing DataFrame columns.
"""

from typing import Iterable


def generate_safe_column_name_from(
    base_name: str,
    existing: Iterable[str],
    prefix: str = "__gfql_",
    suffix: str = "__",
) -> str:
    """Column-name variant of :func:`generate_safe_column_name` taking names, not a frame.

    Needed where the caller already holds a resolved schema (e.g. a polars ``LazyFrame``,
    whose ``.columns`` access would re-resolve the schema and warn).
    """
    taken = set(existing)
    counter = 0
    temp_name = f"{prefix}{base_name}_{counter}{suffix}"
    while temp_name in taken:
        counter += 1
        temp_name = f"{prefix}{base_name}_{counter}{suffix}"
    return temp_name


def generate_safe_column_name(base_name: str, df, prefix: str = "__gfql_", suffix: str = "__") -> str:
    """
    Generate a column name that doesn't conflict with existing columns.
    Uses auto-increment pattern to guarantee uniqueness.

    Parameters:
    -----------
    base_name : str
        The base name for the column
    df : DataFrame
        The DataFrame to check for column name conflicts
    prefix : str
        Prefix to prepend to the column name (default: "__gfql_")
    suffix : str
        Suffix to append to the column name (default: "__")

    Returns:
    --------
    str
        A unique column name that doesn't exist in the DataFrame
        Format: {prefix}{base_name}_{counter}{suffix}
        Example: "__gfql_edge_index_0__"

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'__gfql_node_collapse_0__': [1, 2]})
    >>> generate_safe_column_name('node_collapse', df)
    '__gfql_node_collapse_1__'
    """
    # pyarrow Tables expose column NAMES as `.column_names` (`.columns` is the arrays).
    names = getattr(df, "column_names", None)
    return generate_safe_column_name_from(
        base_name, names if names is not None else df.columns, prefix, suffix
    )

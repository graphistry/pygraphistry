"""
Generate safe column names that avoid conflicts with existing DataFrame columns.
"""


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
    counter = 0
    temp_name = f"{prefix}{base_name}_{counter}{suffix}"

    while temp_name in df.columns:
        counter += 1
        temp_name = f"{prefix}{base_name}_{counter}{suffix}"

    return temp_name

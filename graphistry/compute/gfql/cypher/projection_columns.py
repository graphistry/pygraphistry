from __future__ import annotations

import typing


def alias_field_sources(
    columns: typing.Iterable[str],
    alias: str,
) -> typing.Optional[typing.Mapping[str, str]]:
    column_names = tuple(str(column) for column in columns)
    prefix = f"{alias}."
    sources = {
        column[len(prefix):]: column
        for column in column_names
        if column.startswith(prefix)
    }
    if sources:
        if alias in column_names:
            sources.setdefault(alias, alias)
        return sources if alias in sources else None
    if alias in column_names:
        return {column: column for column in column_names}
    return None

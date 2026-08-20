"""Shared GFQL strictness resolution for absent labels/properties.

One resolution, consulted by the validator AND every executor, so the two can
never drift: ``strict`` raises, ``warn`` warns once per absent name per call,
``quiet`` is silent. Under ``warn``/``quiet`` an absent name resolves to null,
which is openCypher.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterator, Mapping, Optional, Set, Tuple

from typing_extensions import Literal

from graphistry.Plottable import Plottable


StrictLevel = Literal["strict", "warn", "quiet"]
StrictInput = Any  # hygiene-ok: explicit-any -- public param accepts bool | StrictLevel | None

STRICT_LEVELS: Tuple[str, ...] = ("strict", "warn", "quiet")

#: What kind of name went missing, for the diagnostic text.
AbsentNameKind = Literal["label", "column", "property", "name"]

#: Level used when neither the caller nor a bound schema selects one.
DEFAULT_STRICT_LEVEL: StrictLevel = "warn"

#: Level applied at runtime sites reached OUTSIDE a GFQL execution scope
#: (e.g. a direct ``g.filter_nodes_by_dict``): unchanged, raising behavior.
UNSCOPED_STRICT_LEVEL: StrictLevel = "strict"



def normalize_strict_level(value: StrictInput) -> Optional[StrictLevel]:
    """``None`` (unset), ``True``->strict, ``False``->quiet, or a level name."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "strict" if value else "quiet"
    if isinstance(value, str) and value in STRICT_LEVELS:
        return value  # type: ignore[return-value]
    raise ValueError(
        f"strict must be None, a bool, or one of {STRICT_LEVELS}; got {value!r}"
    )


def strict_level_to_bool(level: StrictLevel) -> bool:
    """Legacy boolean view for callers that only know strict-vs-loose."""
    return level == "strict"


def schema_declared_names(g: Optional[Plottable]) -> Optional[FrozenSet[str]]:
    """Every name a bound ``GraphSchema`` declares, or ``None`` when none is bound.

    A name outside this set is a typo even under ``warn``/``quiet``; a name inside
    it that this instance lacks is the narrow-subgraph case and is served.
    """
    if g is None:
        return None
    schema = getattr(g, "_gfql_schema", None)
    if schema is None:
        return None

    names: Set[str] = set()
    for attr in ("node_types", "edge_types"):
        for entry in getattr(schema, attr, ()) or ():
            for name in getattr(entry, "properties", {}) or {}:
                names.add(str(name))
            entry_name = getattr(entry, "name", None)
            if isinstance(entry_name, str):
                names.add(entry_name)
                names.add(f"label__{entry_name}")
            for label in getattr(entry, "labels", ()) or ():
                names.add(str(label))
                names.add(f"label__{label}")
            for column in getattr(entry, "columns", ()) or ():
                names.add(str(column))
    for attr in ("node_id_column", "edge_source_column", "edge_destination_column"):
        column = getattr(schema, attr, None)
        if isinstance(column, str):
            names.add(column)
    if not names:
        # A schema object that declares no names cannot disambiguate anything.
        return None
    return frozenset(names)


def resolve_strict_level(g: Optional[Plottable], *, strict: StrictInput = None) -> StrictLevel:
    """explicit param -> schema.strict -> schema.metadata['strict'] -> default."""
    explicit = normalize_strict_level(strict)
    if explicit is not None:
        return explicit
    schema = getattr(g, "_gfql_schema", None) if g is not None else None
    if schema is not None:
        schema_strict = getattr(schema, "strict", None)
        if schema_strict is not None:
            return normalize_strict_level(schema_strict) or DEFAULT_STRICT_LEVEL
        metadata = getattr(schema, "metadata", None)
        if isinstance(metadata, Mapping) and "strict" in metadata:
            return normalize_strict_level(metadata["strict"]) or DEFAULT_STRICT_LEVEL
    return DEFAULT_STRICT_LEVEL


@dataclass
class _StrictnessScope:
    level: StrictLevel
    declared: Optional[FrozenSet[str]] = None
    warned: Set[str] = field(default_factory=set)


_SCOPE: ContextVar[Optional[_StrictnessScope]] = ContextVar("gfql_strictness_scope", default=None)


@contextmanager
def strictness_scope(
    level: StrictLevel,
    *,
    declared: Optional[FrozenSet[str]] = None,
) -> Iterator[_StrictnessScope]:
    """Publish the resolved level to the runtime sites for one GFQL call.

    Nested calls reuse the outer scope so warn-once stays once per user call.
    """
    existing = _SCOPE.get()
    if existing is not None:
        yield existing
        return
    scope = _StrictnessScope(level=level, declared=declared)
    token: Token[Optional[_StrictnessScope]] = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def current_strict_level() -> StrictLevel:
    scope = _SCOPE.get()
    return scope.level if scope is not None else UNSCOPED_STRICT_LEVEL


def is_internal_plumbing_name(name: str) -> bool:
    """Synthetic row-pipeline columns (alias markers, re-entry keys, label flags).

    Never user-authored, so their absence is never a user-facing diagnostic.
    """
    bare = name.rsplit(".", 1)[-1]
    return bare.startswith("__") or bare.startswith("label__")


def name_is_schema_typo(name: str) -> bool:
    """True when a bound schema exists and does not declare ``name``."""
    scope = _SCOPE.get()
    if scope is None or scope.declared is None:
        return False
    bare = name.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
    return name not in scope.declared and bare not in scope.declared


def absent_name_is_lenient(
    name: str,
    *,
    kind: AbsentNameKind = "column",
    context: Optional[str] = None,
) -> bool:
    """Whether ``name``'s absence should resolve to null instead of raising.

    Returns ``True`` under ``warn`` (after warning once per distinct name per
    call) and ``quiet``; ``False`` under ``strict`` and for a name a bound schema
    does not declare, leaving the caller's own error to fire.
    """
    if name_is_schema_typo(name):
        return False
    level = current_strict_level()
    if level == "strict":
        return False
    if level == "warn":
        scope = _SCOPE.get()
        key = f"{kind}:{name}"
        if scope is None or key not in scope.warned:
            if scope is not None:
                scope.warned.add(key)
            where = f" in {context}" if context else ""
            warnings.warn(
                f'GFQL: {kind} "{name}" is absent{where}; it resolves to null '
                f"(openCypher). Pass strict=True to make this an error, or "
                f'strict="quiet" to silence this warning.',
                UserWarning,
                stacklevel=3,
            )
    return True


def absent_filter_key_is_lenient(
    col: str,
    val: Any,  # hygiene-ok: explicit-any -- filter values are heterogeneous by contract
    *,
    context: Optional[str] = None,
) -> bool:
    """``absent_name_is_lenient`` for a filter-dict key, reported as the LABEL it
    came from when cypher lowered ``(n:X)`` to ``label__X: True``.

    Shared by the preflight and the runtime so both warn under the same key and a
    single absent label warns once, not twice.
    """
    if col.startswith("label__") and val is True:
        return absent_name_is_lenient(col[len("label__"):], kind="label", context=context)
    return absent_name_is_lenient(col, kind="column", context=context)


def absent_column_matches(value: Any) -> bool:  # hygiene-ok: explicit-any -- filter values are heterogeneous by contract
    """3VL verdict for a filter against an all-null (absent) column.

    Every comparison against null is null (no match); only ``IS NULL`` is true.
    """
    from graphistry.compute.predicates.comparison import IsNA

    return isinstance(value, IsNA)

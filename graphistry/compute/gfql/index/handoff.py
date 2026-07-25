"""The chain-boundary -> row-materializer handoff for indexed fixed-hop bindings.

The boundary decides ONCE whether the resident indexes can serve a fixed-hop
pattern, then the canonical row materializer consumes that decision instead of
re-deriving it. This module owns the whole contract — the dataclass, the single
attribute it rides on, and the only attribute access — so callers stay typed and
no other module hand-rolls ``getattr``/``setattr`` for it.

The field is DECLARED on Plottable (with its default on PlotterBase) rather than
smuggled on with ``setattr``, so every access here is ordinary typed attribute
access. It is internal, always attached to an internal copy, and cleared before
the result is handed back to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from graphistry.Engine import Engine
from graphistry.utils.json import JSONVal

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable
    from .bindings import IndexedBindingsState


@dataclass(frozen=True)
class IndexedBindingsHandoff:
    """One boundary decision about one exact plan.

    ``state is None`` records a DECLINE, which is as important as a serve: without
    it the row materializer would re-attempt the same plan (and re-record the same
    trace decision) after the canonical traversal already ran.
    """

    binding_ops: List[Dict[str, JSONVal]]
    state: Optional["IndexedBindingsState"] = None
    edge_aliases: Tuple[str, ...] = field(default=())

    def serves(self, binding_ops: List[Dict[str, JSONVal]], engine: Engine) -> bool:
        """True when this handoff carries a usable state for exactly this plan."""
        return (
            self.state is not None
            and self.state.engine == engine
            and self.binding_ops == binding_ops
        )

    def declined(self, binding_ops: List[Dict[str, JSONVal]]) -> bool:
        """True when this exact plan was already tried and safely declined."""
        return self.state is None and self.binding_ops == binding_ops


def attach_handoff(g: "Plottable", handoff: IndexedBindingsHandoff) -> "Plottable":
    """Return an internal copy of ``g`` carrying ``handoff`` (never mutates ``g``)."""
    out = g.bind()
    out._gfql_indexed_bindings_handoff = handoff
    return out


def set_handoff(g: Any, handoff: IndexedBindingsHandoff) -> None:
    """Attach ``handoff`` to an object the caller is itself constructing."""
    g._gfql_indexed_bindings_handoff = handoff


def read_handoff(g: Any) -> Optional[IndexedBindingsHandoff]:
    """The boundary decision riding on ``g``, if any."""
    return g._gfql_indexed_bindings_handoff


def clear_handoff(g: Any) -> None:
    """Drop the handoff so it never escapes on a user-visible result."""
    g._gfql_indexed_bindings_handoff = None

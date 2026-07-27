"""Per-execution row context: the ``WITH`` re-entry seed and the boundary base graph.

The row pipeline reads its re-entry seed (``_gfql_start_nodes``) and the graph a
boundary suffix must re-match against (``_gfql_rows_base_graph``) off the graph it is
handed. Both are state of ONE execution, not properties of the user's graph, so they
must ride on an INTERNAL COPY. Assigning them to the caller's ``Plottable`` leaves them
behind after the query returns, and the next -- entirely unrelated -- query on the same
object is then answered against the stale seed: a silent wrong count, no error, on every
engine (#1786).

Mirrors ``gfql/index/handoff.attach_handoff``. The fields are DECLARED on ``Plottable``
(defaults on ``PlotterBase``), so every access here is ordinary typed attribute access.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable
    from graphistry.compute.typing import DataFrameT


def attach_row_exec_context(
    g: "Plottable",
    *,
    start_nodes: Optional["DataFrameT"] = None,
    rows_base_graph: Optional["Plottable"] = None,
) -> "Plottable":
    """Return an internal copy of ``g`` carrying this execution's row context.

    ``None`` means "keep whatever ``g`` already carries" for either field, so a nested
    execution inherits the enclosing one's seed exactly as it did before -- the change
    is only ever WHICH object the value lands on, never the value itself.
    """
    out = g.bind()
    if start_nodes is not None:
        out._gfql_start_nodes = start_nodes
    if rows_base_graph is not None:
        out._gfql_rows_base_graph = rows_base_graph
    return out


def clear_row_exec_context(g: "Plottable") -> "Plottable":
    """Return ``g`` stripped of the row context, for a result handed back to the caller.

    The mirror of ``attach_row_exec_context``: each site that attaches also detaches, so
    the plumbing never escapes on a user-visible result. Without this the RESULT of a
    ``WITH`` query carries the seed, and a follow-up query on that result -- a different
    graph entirely -- is answered against it (the same #1786 defect, one hop removed).
    Returns ``g`` unchanged when there is nothing to strip, so the common path is free.
    """
    if g._gfql_start_nodes is None and g._gfql_rows_base_graph is None:
        return g
    out = g.bind()
    out._gfql_start_nodes = None
    out._gfql_rows_base_graph = None
    return out

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

    WHY ``None`` AND NOT "RESTORE WHAT ``g`` CARRIED ON ENTRY". ``attach`` above INHERITS on
    the way in (a ``None`` argument keeps what ``g`` already carries), so the asymmetry is
    real and the save/restore question is a fair one. Three reasons it is settled the other
    way, each pinned by ``tests/compute/gfql/test_exec_context_scoping.py``:

    * This function is PURE -- it returns ``g.bind()`` and never writes through to the object
      it was handed, so an outer scope that set the field still has it afterwards. There is no
      caller state to save. That is what separates this from #1786, which WAS an in-place
      write onto the caller's own graph; restoring is the fix for a mutation, and there is no
      mutation here.
    * The only channel restore would change is the RETURN VALUE, and putting the seed back
      there IS the second half of #1786. Measured: hand-restoring the seed onto a result
      changes the answer of the next query run on that result (7 -> 2 -> 1 rows).
    * No execution frame ever inherits a context it did not set. The cross-segment ``WITH``
      seed travels as the explicit ``start_nodes`` PARAMETER (``chain_impl(..., start_nodes=)``,
      ``_compiled_query_reentry_state``), never through this field, so the field's lifetime is
      exactly ONE boundary-call run. Instrumenting ``attach`` over ``tests/compute`` recorded
      3907 calls and ZERO inheriting ones. (53 of them DID enter on a graph already carrying a
      seed -- nested boundary frames -- but each was handed the IDENTICAL ``start_nodes``
      parameter, so the value is still owned by the frame that sets it.) The test above re-runs
      that as an assertion, so a future path that starts relying on inheritance reopens this
      decision loudly.
    """
    if g._gfql_start_nodes is None and g._gfql_rows_base_graph is None:
        return g
    out = g.bind()
    out._gfql_start_nodes = None
    out._gfql_rows_base_graph = None
    return out

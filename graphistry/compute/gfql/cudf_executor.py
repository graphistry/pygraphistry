"""cuDF-based GFQL executor with same-path WHERE planning.

This module hosts the GPU execution path for GFQL chains that require
same-path predicate enforcement.  The actual kernels / dataframe
operations are implemented in follow-up steps; for now we centralize the
structure so the planner and chain machinery have a single place to hook
into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject
from graphistry.gfql.same_path_plan import SamePathPlan, plan_same_path
from graphistry.gfql.same_path_types import WhereComparison

__all__ = [
    "SamePathExecutorInputs",
    "CuDFSamePathExecutor",
    "build_same_path_inputs",
    "execute_same_path_chain",
]


@dataclass(frozen=True)
class SamePathExecutorInputs:
    """Container for all metadata needed by the cuDF executor."""

    graph: Plottable
    chain: Sequence[ASTObject]
    where: Sequence[WhereComparison]
    plan: SamePathPlan
    engine: Engine
    include_paths: bool = False


class CuDFSamePathExecutor:
    """Runs a forward/backward/forward pass using cuDF dataframes."""

    def __init__(self, inputs: SamePathExecutorInputs) -> None:
        self.inputs = inputs

    def run(self) -> Plottable:
        """Execute full cuDF traversal once kernels are available."""
        raise NotImplementedError(
            "cuDF executor forward/backward passes not wired yet"
        )

    def _forward(self) -> None:
        raise NotImplementedError

    def _backward(self) -> None:
        raise NotImplementedError

    def _finalize(self) -> Plottable:
        raise NotImplementedError


def build_same_path_inputs(
    g: Plottable,
    chain: Sequence[ASTObject],
    where: Sequence[WhereComparison],
    engine: Engine,
    include_paths: bool = False,
) -> SamePathExecutorInputs:
    """Construct executor inputs, deriving planner metadata if missing."""

    plan = plan_same_path(where)
    return SamePathExecutorInputs(
        graph=g,
        chain=list(chain),
        where=list(where),
        plan=plan,
        engine=engine,
        include_paths=include_paths,
    )


def execute_same_path_chain(
    g: Plottable,
    chain: Sequence[ASTObject],
    where: Sequence[WhereComparison],
    engine: Engine,
    include_paths: bool = False,
) -> Plottable:
    """Convenience wrapper used by Chain execution once hooked up."""

    inputs = build_same_path_inputs(g, chain, where, engine, include_paths)
    executor = CuDFSamePathExecutor(inputs)
    return executor.run()

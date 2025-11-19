"""cuDF-based GFQL executor with same-path WHERE planning.

This module hosts the GPU execution path for GFQL chains that require
same-path predicate enforcement.  The actual kernels / dataframe
operations are implemented in follow-up steps; for now we centralize the
structure so the planner and chain machinery have a single place to hook
into.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Set

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject
from graphistry.gfql.same_path_plan import SamePathPlan, plan_same_path
from graphistry.gfql.same_path_types import WhereComparison

AliasKind = Literal["node", "edge"]

__all__ = [
    "AliasBinding",
    "SamePathExecutorInputs",
    "CuDFSamePathExecutor",
    "build_same_path_inputs",
    "execute_same_path_chain",
]


@dataclass(frozen=True)
class AliasBinding:
    """Metadata describing which chain step an alias refers to."""

    alias: str
    step_index: int
    kind: AliasKind
    ast: ASTObject


@dataclass(frozen=True)
class SamePathExecutorInputs:
    """Container for all metadata needed by the cuDF executor."""

    graph: Plottable
    chain: Sequence[ASTObject]
    where: Sequence[WhereComparison]
    plan: SamePathPlan
    engine: Engine
    alias_bindings: Dict[str, AliasBinding]
    column_requirements: Dict[str, Set[str]]
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
    """Construct executor inputs, deriving planner metadata and validations."""

    bindings = _collect_alias_bindings(chain)
    _validate_where_aliases(bindings, where)
    required_columns = _collect_required_columns(where)
    plan = plan_same_path(where)

    return SamePathExecutorInputs(
        graph=g,
        chain=list(chain),
        where=list(where),
        plan=plan,
        engine=engine,
        alias_bindings=bindings,
        column_requirements=required_columns,
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


def _collect_alias_bindings(chain: Sequence[ASTObject]) -> Dict[str, AliasBinding]:
    bindings: Dict[str, AliasBinding] = {}
    for idx, step in enumerate(chain):
        alias = getattr(step, "_name", None)
        if not alias:
            continue
        if not isinstance(alias, str):
            continue
        if isinstance(step, ASTNode):
            kind: AliasKind = "node"
        elif isinstance(step, ASTEdge):
            kind = "edge"
        else:
            continue

        if alias in bindings:
            raise ValueError(f"Duplicate alias '{alias}' detected in chain")
        bindings[alias] = AliasBinding(alias, idx, kind, step)
    return bindings


def _collect_required_columns(
    where: Sequence[WhereComparison],
) -> Dict[str, Set[str]]:
    requirements: Dict[str, Set[str]] = defaultdict(set)
    for clause in where:
        requirements[clause.left.alias].add(clause.left.column)
        requirements[clause.right.alias].add(clause.right.column)
    return {alias: set(cols) for alias, cols in requirements.items()}


def _validate_where_aliases(
    bindings: Dict[str, AliasBinding],
    where: Sequence[WhereComparison],
) -> None:
    if not where:
        return
    referenced = {clause.left.alias for clause in where} | {
        clause.right.alias for clause in where
    }
    missing = sorted(alias for alias in referenced if alias not in bindings)
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"WHERE references aliases with no node/edge bindings: {missing_str}"
        )

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
from typing import Dict, Literal, Sequence, Set, List, Optional, Any

import pandas as pd

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode, ASTObject
from graphistry.gfql.same_path_plan import SamePathPlan, plan_same_path
from graphistry.gfql.same_path_types import WhereComparison
from graphistry.compute.typing import DataFrameT

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
        self.forward_steps: List[Plottable] = []
        self.alias_frames: Dict[str, DataFrameT] = {}
        self._node_column = inputs.graph._node
        self._edge_column = inputs.graph._edge

    def run(self) -> Plottable:
        """Execute full cuDF traversal once kernels are available."""
        self._forward()
        raise NotImplementedError(
            "cuDF executor backward pass not wired yet"
        )

    def _forward(self) -> None:
        graph = self.inputs.graph
        ops = self.inputs.chain
        self.forward_steps = []

        for idx, op in enumerate(ops):
            if isinstance(op, ASTCall):
                current_g = self.forward_steps[-1] if self.forward_steps else graph
                prev_nodes = None
            else:
                current_g = graph
                prev_nodes = (
                    None if not self.forward_steps else self.forward_steps[-1]._nodes
                )
            g_step = op(
                g=current_g,
                prev_node_wavefront=prev_nodes,
                target_wave_front=None,
                engine=self.inputs.engine,
            )
            self.forward_steps.append(g_step)
            self._capture_alias_frame(op, g_step, idx)

    def _backward(self) -> None:
        raise NotImplementedError

    def _finalize(self) -> Plottable:
        raise NotImplementedError

    def _capture_alias_frame(
        self, op: ASTObject, step_result: Plottable, step_index: int
    ) -> None:
        alias = getattr(op, "_name", None)
        if not alias or alias not in self.inputs.alias_bindings:
            return
        binding = self.inputs.alias_bindings[alias]
        frame = (
            step_result._nodes
            if binding.kind == "node"
            else step_result._edges
        )
        if frame is None:
            kind = "node" if binding.kind == "node" else "edge"
            raise ValueError(
                f"Alias '{alias}' did not produce a {kind} frame"
            )
        required = set(self.inputs.column_requirements.get(alias, set()))
        id_col = self._node_column if binding.kind == "node" else self._edge_column
        if id_col:
            required.add(id_col)
        missing = [col for col in required if col not in frame.columns]
        if missing:
            cols = ", ".join(missing)
            raise ValueError(
                f"Alias '{alias}' missing required columns: {cols}"
            )
        subset_cols = [col for col in required]
        alias_frame = frame[subset_cols].copy()
        self.alias_frames[alias] = alias_frame
        self._apply_ready_clauses()

    def _apply_ready_clauses(self) -> None:
        if not self.inputs.where:
            return
        ready = [
            clause
            for clause in self.inputs.where
            if clause.left.alias in self.alias_frames
            and clause.right.alias in self.alias_frames
        ]
        for clause in ready:
            self._prune_clause(clause)

    def _prune_clause(self, clause: WhereComparison) -> None:
        if clause.op == "!=":
            return  # No global prune for inequality-yet
        lhs = self.alias_frames[clause.left.alias]
        rhs = self.alias_frames[clause.right.alias]
        left_col = clause.left.column
        right_col = clause.right.column

        if clause.op == "==":
            allowed = self._common_values(lhs[left_col], rhs[right_col])
            self.alias_frames[clause.left.alias] = self._filter_by_values(
                lhs, left_col, allowed
            )
            self.alias_frames[clause.right.alias] = self._filter_by_values(
                rhs, right_col, allowed
            )
        elif clause.op == ">":
            right_min = self._safe_min(rhs[right_col])
            left_max = self._safe_max(lhs[left_col])
            if right_min is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] > right_min]
            if left_max is not None:
                self.alias_frames[clause.right.alias] = rhs[rhs[right_col] < left_max]
        elif clause.op == ">=":
            right_min = self._safe_min(rhs[right_col])
            left_max = self._safe_max(lhs[left_col])
            if right_min is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] >= right_min]
            if left_max is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] <= left_max
                ]
        elif clause.op == "<":
            right_max = self._safe_max(rhs[right_col])
            left_min = self._safe_min(lhs[left_col])
            if right_max is not None:
                self.alias_frames[clause.left.alias] = lhs[lhs[left_col] < right_max]
            if left_min is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] > left_min
                ]
        elif clause.op == "<=":
            right_max = self._safe_max(rhs[right_col])
            left_min = self._safe_min(lhs[left_col])
            if right_max is not None:
                self.alias_frames[clause.left.alias] = lhs[
                    lhs[left_col] <= right_max
                ]
            if left_min is not None:
                self.alias_frames[clause.right.alias] = rhs[
                    rhs[right_col] >= left_min
                ]

    @staticmethod
    def _filter_by_values(
        frame: DataFrameT, column: str, values: Set[Any]
    ) -> DataFrameT:
        if not values:
            return frame.iloc[0:0]
        allowed = list(values)
        mask = frame[column].isin(allowed)
        return frame[mask]

    @staticmethod
    def _common_values(series_a: Any, series_b: Any) -> Set[Any]:
        vals_a = CuDFSamePathExecutor._series_values(series_a)
        vals_b = CuDFSamePathExecutor._series_values(series_b)
        return vals_a & vals_b

    @staticmethod
    def _series_values(series: Any) -> Set[Any]:
        pandas_series = CuDFSamePathExecutor._to_pandas_series(series)
        return set(pandas_series.dropna().unique().tolist())

    @staticmethod
    def _safe_min(series: Any) -> Optional[Any]:
        pandas_series = CuDFSamePathExecutor._to_pandas_series(series).dropna()
        if pandas_series.empty:
            return None
        value = pandas_series.min()
        if pd.isna(value):
            return None
        return value

    @staticmethod
    def _safe_max(series: Any) -> Optional[Any]:
        pandas_series = CuDFSamePathExecutor._to_pandas_series(series).dropna()
        if pandas_series.empty:
            return None
        value = pandas_series.max()
        if pd.isna(value):
            return None
        return value

    @staticmethod
    def _to_pandas_series(series: Any) -> pd.Series:
        if hasattr(series, "to_pandas"):
            return series.to_pandas()
        if isinstance(series, pd.Series):
            return series
        return pd.Series(series)


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

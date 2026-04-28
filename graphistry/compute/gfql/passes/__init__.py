"""Logical plan pass framework."""

from .manager import LogicalPass, PassManager, PassResult
from .predicate_pushdown import PredicatePushdownPass
from .unnest_apply import UnnestApply

# Tier 1: structural passes that run once in order.
DEFAULT_LOGICAL_PASSES = (UnnestApply(),)

# Tier 2: rewrite rules that run in a fixed-point loop until convergence.
DEFAULT_TIER2_PASSES = (PredicatePushdownPass(),)

__all__ = [
    "DEFAULT_LOGICAL_PASSES",
    "DEFAULT_TIER2_PASSES",
    "LogicalPass",
    "PassManager",
    "PassResult",
    "PredicatePushdownPass",
    "UnnestApply",
]

"""Logical plan pass framework skeleton for M4."""

from .manager import DEFAULT_LOGICAL_PASSES, LogicalPass, PassManager, PassResult

__all__ = [
    "DEFAULT_LOGICAL_PASSES",
    "LogicalPass",
    "PassManager",
    "PassResult",
]

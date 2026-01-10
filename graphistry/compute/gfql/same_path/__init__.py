"""Same-path GFQL execution modules.

This package contains the Yannakakis-style semijoin executor for
GFQL chains with WHERE clause constraints.
"""

from .chain_meta import ChainMeta
from .edge_semantics import EdgeSemantics

__all__ = [
    "ChainMeta",
    "EdgeSemantics",
]

"""Shared public GFQL query type aliases."""
from __future__ import annotations

from typing import Any, Dict, List, Union

from graphistry.compute.ast import ASTLet, ASTObject
from graphistry.compute.chain import Chain


GFQLQuery = Union[ASTObject, List[ASTObject], ASTLet, Chain, Dict[str, Any], str]
"""Accepted local GFQL query inputs: AST objects/chains/DAGs, JSON dicts, or strings."""

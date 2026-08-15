"""Shared public GFQL query type aliases."""
from __future__ import annotations

from typing import Any, Dict, List, Union

from graphistry.compute.ast import ASTLet, ASTObject
from graphistry.compute.chain import Chain

# NOTE: the companion `params=` alias is `graphistry.compute.gfql.cypher.ast.CypherParams`.
# It is deliberately NOT re-exported here: this module is imported by
# `graphistry.compute.__init__`, and reaching into the `cypher` package would drag the whole
# Cypher compiler into every `import graphistry`.

GFQLQuery = Union[ASTObject, List[ASTObject], ASTLet, Chain, Dict[str, Any], str]
"""Accepted local GFQL query inputs: AST objects/chains/DAGs, JSON dicts, or strings."""

"""Bounded-reentry helpers extracted from ``cypher.lowering`` (#1295, #1260 S2).

Subpackage owns:
- naming conventions for hidden carry columns (``naming``)
- alias-scope traversal helpers (``scope``)
- prefix carry-column / order helpers (``carry``)
- AST/query rewriters that retarget reentry expressions onto carried columns
  (``rewrite``)
- compile-time bounded-reentry query rewrites (``compiletime``; ``runtime`` is
  a compatibility re-export shim)
- data-frame execution stitching for bounded reentry (``execution``; #987 Step 3)

Public symbols are re-exported from ``cypher.lowering`` so existing imports
(``from graphistry.compute.gfql.cypher.lowering import _reentry_hidden_column_name``)
continue to work.
"""
from __future__ import annotations

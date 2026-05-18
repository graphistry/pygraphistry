"""Bounded-reentry helpers extracted from ``cypher.lowering`` (#1295, #1260 S2).

Subpackage owns:
- naming conventions for hidden carry columns (``naming``)
- alias-scope traversal helpers (``scope``)
- prefix carry-column / order helpers (``carry``)
- AST/query rewriters that retarget reentry expressions onto carried columns
  (``rewrite``)
- compile-time bounded-reentry query rewrites (``compiletime``)
- data-frame execution stitching for bounded reentry (``execution``; #987 Step 3)
"""
from __future__ import annotations

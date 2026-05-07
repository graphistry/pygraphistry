"""Compatibility wrapper for reentry compile-time helpers.

`graphistry.compute.gfql.cypher.reentry.runtime` historically hosted
compile-time bounded-reentry lowering utilities. Those helpers now live in
`reentry.compiletime`; this module remains as a re-export shim so existing
imports keep working during migration.
"""
from __future__ import annotations

from graphistry.compute.gfql.cypher.reentry import compiletime as _compiletime

for _name in dir(_compiletime):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_compiletime, _name)

del _compiletime
try:
    del _name
except NameError:
    pass

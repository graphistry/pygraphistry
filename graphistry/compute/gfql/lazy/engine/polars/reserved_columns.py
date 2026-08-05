"""Reserved internal column-name bases for the native polars GFQL engine.

Central, per-engine registry of the ``__gfql_*__`` base names the polars executor
reserves for internal/temporary columns — declared in one place rather than inlined as
string literals across the executor. Every base follows the repo-wide reserved pattern
from :mod:`graphistry.compute.gfql.identifiers`
(``INTERNAL_COLUMN_PREFIX``/``INTERNAL_COLUMN_SUFFIX`` → ``__gfql_*__``).

Convention:
- One module-level constant per reserved base, ``UPPER_SNAKE``, value ``__gfql_<role>__``.
- At use sites, pass the base to
  ``generate_safe_column_name(base, frame, prefix=INTERNAL_COLUMN_PREFIX, suffix=INTERNAL_COLUMN_SUFFIX)``
  so a user column of the same name is never clobbered.
- New polars-engine internal columns SHOULD be added here rather than inlined.

Follow-up (tracked, non-blocking): migrate the remaining inline literals in
``chain.py`` (``__gfql_norder__``/``__gfql_eorder__``) and ``hop_eager.py``
(``__gfql_from__``/``__gfql_to__``/``__gfql_nid__``/``__gfql_eid__``/``__gfql_hop__``/
``__gfql_node_hop__``) into this module so the whole engine shares one registry.
"""
from graphistry.compute.gfql.identifiers import (
    INTERNAL_COLUMN_PREFIX,
    INTERNAL_COLUMN_SUFFIX,
)

#: chain.py — auto-injected hop-distance label used to gate a node alias that follows a
#: variable-length edge (#1741). Resolved against the user's node columns at each use.
CHAIN_NODE_HOP: str = f"{INTERNAL_COLUMN_PREFIX}chain_node_hop{INTERNAL_COLUMN_SUFFIX}"

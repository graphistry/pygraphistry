# GFQL process-lifetime caches

Every memo in `graphistry/compute/gfql/**` (and `gfql_unified.py`) registers itself
in `graphistry/compute/gfql/cache_registry.py`, adjacent to its own definition.
Two classes, chosen by what the cache is keyed on:

- **Clearable** — keyed by caller input (query text, expression strings, schemas).
  It grows with traffic and changes what a later call costs, so
  `gfql_clear_caches()` must empty it: `register_clearable(fn)` for `@lru_cache`
  functions, `register_clearable_dict(name, d)` for hand-rolled dicts,
  `register_clearable_callable(name, fn)` when clearing needs its own lock.
- **Process singleton** — `maxsize=1` and a pure function of the code (Lark
  parser tables, import-resolution probes, compiled regexes). Exempt, because
  rebuilding it costs strictly more than it saves and would push one-time setup
  into every "cold" measurement: `register_process_singleton(fn, reason)` with a
  real reason.

Rules the registry enforces or the coverage lock
(`graphistry/tests/compute/gfql/test_clear_caches_covers_every_cache.py`) fails on:

1. Register at the definition site, never centrally: the clear handle is bound to
   the real object, so there is no later name lookup to get wrong. (A name-based
   clear once became a silent no-op — `parse_cypher` vs `_parse_cypher_cached` —
   and published a wrong cold-process number for days.)
2. An unregistered cache fails CI: the lock's AST sweep finds every
   `@lru_cache`/`@cache` def and every module-level dict/set named cache/memo.
3. Exemption reasons are code, not test data, and must be substantive.
4. Caches keyed to one graph live on its `Plottable` and die with it — those do
   not register here.

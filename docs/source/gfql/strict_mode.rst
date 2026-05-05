Strict Schema Mode (Cypher Binder)
==================================

GFQL's Cypher binder ships with an opt-in **strict schema mode** that rejects
queries referencing labels or properties absent from the bound
``GraphSchemaCatalog``. The default loose mode admits unknown names so today's
exploratory and partially-typed workflows keep working unchanged.

This page is the operator-side reference for enabling strict mode safely as
part of staged adoption (#1311 / #1262).

What strict mode covers
-----------------------

When enabled, the Cypher binder enforces:

* MATCH labels exist in the catalog's node label set.
* WHERE / RETURN / UNWIND / CALL property references exist for the relevant
  alias's node or edge column set.

Strict mode is purely a binder gate — it raises ``GFQLValidationError`` (with
``ErrorCode`` in the ``E10x`` / ``E30x`` families) before any execution. There
is no runtime cost in loose mode; there is no behavior difference for valid
queries between modes.

It does **not** yet cover:

* Arrow-bridge type coercion (deferred to the T4 lane under #1262 / #1312).
* Per-row dataframe-side type checks.

How to enable
-------------

Strict mode has three opt-in paths, listed by precedence (most specific wins):

1. **Explicit caller parameter** — the strongest signal.

   .. code-block:: python

      from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
      FrontendBinder().bind(ast, ctx, strict_name_resolution=True)

2. **Catalog metadata flag** — pinned per dataset.

   .. code-block:: python

      catalog = GraphSchemaCatalog.from_schema_parts(
          node_columns={"id", "label__Person"},
          edge_columns={"src", "dst", "label__KNOWS"},
          metadata={"strict": True},
      )

3. **Process-wide environment variable** — for staged rollout / canary.

   .. code-block:: bash

      export GRAPHISTRY_GFQL_STRICT_SCHEMA=true

   Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
   Falsy / unset: anything else (default ``false``).

Precedence
----------

When more than one tier opts in, strict applies. Monotonic widening:

* Explicit param ``True``  → strict.
* Catalog ``metadata["strict"]=True``  → strict.
* Env ``GRAPHISTRY_GFQL_STRICT_SCHEMA=true``  → strict.
* Otherwise → loose.

An explicit ``False`` is treated as *no preference* — it does not force loose
mode when the catalog or env elects strict. To force loose, do not set any of
the three opt-ins.

Diagnostic shape
----------------

Strict-mode rejections raise ``GFQLValidationError`` with deterministic
messages and sorted availability hints:

.. code-block:: text

    Cypher label is missing from strict binder schema catalog.
    Use labels that exist in the node schema or disable strict mode.
    available labels: [Comment, Person, Post]

Use the message text to identify the gap, then either fix the query, extend
the catalog, or temporarily disable strict mode while iterating.

When to enable
--------------

Recommended:

* Production query gates where unknown identifiers should fail closed.
* CI / pre-merge quality bars over a curated catalog.
* Multi-team environments where catalog ownership is centralized.

Keep loose:

* Exploratory / notebook usage where the schema is being discovered.
* Pipelines where catalogs may be partial by design (for example, post-ingest
  before label propagation finalises).

Recommended rollout sequence
----------------------------

Stage adoption from the least invasive control to the most specific:

1. **Canary** — set ``GRAPHISTRY_GFQL_STRICT_SCHEMA=true`` in a non-production
   shadow environment. Watch validation diagnostics. Catch and triage any
   queries that reject under strict.
2. **Per-dataset opt-in** — once the canary surface is clean, enable
   ``metadata={"strict": True}`` on the catalogs that should fail closed.
3. **Per-call enforcement** — for the tightest path (for example a request
   handler that should never accept unknown identifiers), pass
   ``strict_name_resolution=True`` directly. This wins over both lower tiers
   and is the most readable signal at the call site.

Rolling back is always safe: clear the env var or remove the catalog flag /
parameter; loose mode behavior returns immediately on the next bind.

Related lanes
-------------

* T2 (#1302) — added the binder-time strict checks themselves.
* T3 (#1300) — added type/nullability metadata propagation contract.
* T4 (#1312, in progress) — Arrow/type-bridge contract surface; once landed,
  this page will gain a parallel section for arrow-bridge rollout gates.
* T5 (#1311) — this page; rollout / docs / CI receipts for staged adoption.

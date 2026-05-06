Strict Schema Mode
==================

GFQL ships with an opt-in **strict schema mode** that rejects Cypher queries
referencing labels or properties absent from the bound graph schema. The
default loose mode admits unknown names so today's exploratory and
partially-typed workflows keep working unchanged.

GFQL exposes strict mode through two complementary surfaces:

1. **Explicit preflight** — :py:meth:`g.gfql_validate(...) <graphistry.compute.ComputeMixin.ComputeMixin.gfql_validate>`
   and :py:meth:`g.gfql(..., validate=True) <graphistry.compute.ComputeMixin.ComputeMixin.gfql>` —
   the **primary operator entrypoint** for explicit, predictable, fail-fast
   schema checks. See :doc:`validation/fundamentals` and :doc:`cypher`.
2. **Execution-path rollout gate** — environment variable / catalog metadata
   precedence ladder governing the default for non-validate-flagged
   :py:meth:`g.gfql() <graphistry.compute.ComputeMixin.ComputeMixin.gfql>`
   execution. **This is a canary surface for staged organisation-wide
   adoption.** This page is its operator reference.

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

It does **not** cover dataframe-side per-row type checks. Arrow/type-bridge
coercion semantics are handled by ``graphistry.compute.gfql.ir.arrow_bridge``
(landed under #1312); rollout controls for that surface are not part of this
page.

The two surfaces in detail
--------------------------

Explicit preflight (the primary operator entrypoint)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For most operator workflows, prefer the explicit preflight API. It returns
structured diagnostics and never executes query operators:

.. doc-test: skip

.. code-block:: python

   report = g.gfql_validate(
       "MATCH (p:Person) RETURN p.name AS name",
       strict=True,
   )
   if not report["ok"]:
       for diag in report["diagnostics"]:
           print(diag["code"], diag["message"])

For execution guarded by a preflight check, use the ``validate=True`` flag
on ``g.gfql(...)`` (which runs the same preflight in strict mode before
executing):

.. doc-test: skip

.. code-block:: python

   result = g.gfql(
       "MATCH (p:Person) RETURN p.name AS name",
       validate=True,
   )

These surfaces are predictable and not influenced by environment variables —
they always run strict checks when invoked, and they are the right tool for
explicit per-call enforcement (request handlers, notebooks, CI gates).

Execution-path rollout gate (this page's primary topic)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For rollout scenarios where you want to flip strict-mode default behavior
across an environment **without modifying every call site**, GFQL exposes a
three-tier precedence ladder for the binder default used by
``g.gfql(query)`` (i.e., the path with ``validate=False``, the default):

1. **Explicit binder parameter** — the strongest signal.

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
   FrontendBinder().bind(ast, ctx, strict_name_resolution=True)

This is rarely useful directly — most callers reach the binder via
``g.gfql(query)`` rather than constructing it themselves.

2. **Catalog metadata flag** — pinned per dataset.

.. doc-test: skip

.. code-block:: python

   from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog
   catalog = GraphSchemaCatalog.from_schema_parts(
       node_columns={"id", "label__Person"},
       edge_columns={"src", "dst", "label__KNOWS"},
       metadata={"strict": True},
   )

3. **Process-wide environment variable** — the canary toggle.

.. code-block:: bash

   export GRAPHISTRY_GFQL_STRICT_SCHEMA=true

Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
Falsy / unset: anything else (default ``false``).

When more than one tier opts in, strict applies. Monotonic widening:

* Explicit param ``True``  → strict.
* Catalog ``metadata["strict"]=True``  → strict.
* Env ``GRAPHISTRY_GFQL_STRICT_SCHEMA=true``  → strict.
* Otherwise → loose.

An explicit ``False`` is treated as *no preference* — it does not force loose
mode when the catalog or env elects strict. To force loose, do not set any of
the three opt-ins.

**Important scoping note:** this precedence ladder governs the binder default
on the *execution* compile path. The explicit preflight API
(``g.gfql_validate(strict=True)``, ``g.gfql(validate=True)``) is unaffected
by these tiers — it always runs strict checks when invoked. The two surfaces
are independent on purpose: explicit preflight for predictable per-call
diagnostics, environment ladder for organization-wide canary rollout.

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

1. **Canary via environment variable** — set
   ``GRAPHISTRY_GFQL_STRICT_SCHEMA=true`` in a non-production shadow
   environment. Watch validation diagnostics from any ``g.gfql(query)`` calls
   that previously ran loose. Catch and triage queries that newly reject.
2. **Per-dataset opt-in via catalog metadata** — once the canary surface is
   clean, enable ``metadata={"strict": True}`` on the catalogs that should
   fail closed in production. This pins behavior independently of the env.
3. **Per-call enforcement via explicit preflight** — for the tightest path
   (for example a request handler that should never accept unknown
   identifiers), prefer the explicit preflight surface:

.. doc-test: skip

.. code-block:: python

   result = g.gfql(query, validate=True)

This is more readable than the binder param and runs structured diagnostics.
Use it for code that wants strict regardless of catalog or env.

Rolling back the rollout gate is always safe: clear the env var or remove the
catalog flag; loose mode returns immediately on the next bind. The explicit
preflight surface is unaffected by either.

Related lanes
-------------

* T1 (#1296) — schema catalog contract.
* T2 (#1302) — added the binder-time strict checks themselves.
* T3 (#1300) — type/nullability metadata propagation contract.
* T3.b (#1309) — nullable-helper consolidation follow-through.
* T4 (#1313) — Arrow/type-bridge contract surface.
* T5 (#1311) — this page; rollout / docs / CI receipts for staged adoption.
* #1320 / #1321 — explicit preflight API (``g.gfql_validate``,
  ``g.gfql(validate=True)``) — the primary operator entrypoint for explicit
  strict-mode invocation.

See also
--------

* :doc:`validation/fundamentals` — preflight + execution-time validation
  primitives, including ``g.gfql_validate(...)``.
* :doc:`cypher` — Cypher syntax reference and preflight examples.

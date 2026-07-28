.. _gfql-performance:

GFQL Performance: Vectorization and GPU Acceleration
====================================================

This page is the **canonical home for GFQL benchmark numbers** — the measured tables live
here (and, for the resident-index benchmarks, in :doc:`index_adjacency`), while the rest of
the docs make stable qualitative claims and link back here.

.. note::
   **Every number published on this page is now referenced, not transcribed.** Figures come
   from a single machine-readable source of truth
   (``docs/source/_data/gfql_benchmarks.json``) that is generated from committed benchmark
   artifacts, and the docs build fails rather than render a figure the source of truth does
   not contain or whose run has gone stale. See `How a number gets published here`_.

   Figures whose originating run could not be reproduced from a committed artifact have
   been **removed** from these docs rather than restated. Removing an unverifiable number
   is the correct outcome, not a regression.

Engine speedups at a glance
---------------------------

GFQL runs the **same query** on four interchangeable engines — ``pandas`` (default),
``polars`` (CPU, columnar), ``cudf`` (NVIDIA GPU), and ``polars-gpu`` (GPU) — and returns
**identical results** on each (differential parity is a release gate).
Unsupported engine/query combinations are declined before execution during validation,
compilation, or planning rather than silently falling back. The biggest, easiest win is one
keyword, **no GPU required**:

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # often much faster on query-heavy workloads, same results

There is **no universal winner**: ``polars`` typically takes over past small/interactive
sizes (``pandas`` still wins trivially small operations), and the right GPU
engine depends on the workload. See :doc:`engines` for the full decision matrix, the honest
"when *not* to use Polars", and the cuDF-vs-Polars-GPU comparison. The
Spark GraphFrames head-to-head is in :doc:`benchmark_graphframes`.

How a number gets published here
--------------------------------

1. The benchmark harness lives in `graphistry/pyg-bench
   <https://github.com/graphistry/pyg-bench>`_, which commits its raw per-slot artifacts
   alongside the results.
2. ``scripts/export_docs_numbers.py`` in that repository turns those committed artifacts
   into ``docs/source/_data/gfql_benchmarks.json`` here — deriving each median, each ratio
   and each cell's publishability from the artifacts rather than from anyone's notes. **No
   GPU box and no benchmark re-run is needed** to regenerate the docs from stored results.
3. The docs build resolves every ``:bench:`` reference against that file, and fails if a
   key is missing, if a run has aged past the policy threshold, or if a page drops a
   number's provenance or disclosures.
4. ``bin/check_bench_numbers.py`` runs the same gate outside Sphinx and adds a
   commit-drift check: when the query engine has moved materially since a published run was
   measured, the number is treated as stale and the check fails.

How GFQL is fast
----------------

Three design choices explain the speedups:

**Collection-oriented execution.** GFQL evaluates whole collections of nodes and edges at
once (set-at-a-time), rather than walking one path at a time like traditional Cypher/Gremlin
engines. A traversal advances by joining edge tables, so the work vectorizes.

**Vectorized columnar processing.** Data is processed in columnar batches on top of
`Apache Arrow <https://arrow.apache.org/>`_, which keeps the CPU path fast and makes moving
data between systems cheap. The ``polars`` engine additionally builds **one fused lazy plan
and collects once**, which is why it outruns eager engines on bulk work.

**Massive parallelism on GPUs.** On an NVIDIA GPU (``cudf`` / ``polars-gpu``), the same
vectorized work saturates tens of thousands of threads — paying off when there is enough
work to amortize kernel-launch cost (large frontiers, dense joins, full-graph aggregation).

Start on CPU with no special hardware, and move to a GPU engine by changing one keyword when
your workload grows into GPU territory. See :doc:`engines` for exactly when each engine wins.

.. note::
   Same-path constraints (``where``) can be more expensive on dense graphs.
   Prefer selective per-step predicates and see :doc:`/gfql/where` for details.

Next Steps
----------

- **Choose an engine**: :doc:`engines` — the full decision matrix and qualitative guidance.
- **Selective lookups**: :doc:`index_adjacency` — the CSR adjacency index.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.
- **Ecosystem**: `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.

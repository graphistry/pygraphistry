.. _gfql-performance:

GFQL Performance: Vectorization and GPU Acceleration
====================================================

GFQL runs Cypher over the dataframes you already have — pandas, Polars, or cuDF — in your
own process. No server to provision, no ETL step, no projection into a second store.

This page and :doc:`index_adjacency` carry every measured number PyGraphistry publishes.
Each figure is referenced from ``docs/source/_data/gfql_benchmarks.json``, which is
generated from committed benchmark artifacts; the build fails rather than render a figure
that file does not contain, or one whose run has aged past the freshness policy. See
`How a number gets published here`_.

One keyword, no GPU
-------------------

The same query, the same graph, the same answers — only ``engine=`` changes:

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # columnar, fused lazy plan

The lane below is the ``prrao87/graph-benchmark`` q1–q9 Cypher suite over a synthetic
social graph: degree ranking, grouped aggregation, filtered population counts, two-hop path
counting. Every cell is a real GFQL Cypher execution — no dataframe shortcut, no untimed
precompute — and every cell's result rows were checked against the reference engine's
before it was allowed to be published.

**GFQL-Polars beats GFQL-pandas on all nine queries, at both graph sizes.** Absolute
medians are for the 100,000-person graph (2.78M edges); the last column is the same
speedup measured on the 20,000-person graph (260k edges), so the trend with size is
visible rather than asserted.

.. list-table::
   :header-rows: 1
   :widths: 8 16 16 16 16 28

   * - Query
     - ``pandas``
     - ``polars``
     - Speedup
     - Speedup at 20k
     - What it does
   * - q1
     - :bench:`graphbench.100k.q1.pandas`
     - :bench:`graphbench.100k.q1.polars`
     - :bench:`graphbench.100k.q1.polars_vs_pandas`
     - :bench:`graphbench.20k.q1.polars_vs_pandas`
     - top-3 by in-degree
   * - q2
     - :bench:`graphbench.100k.q2.pandas`
     - :bench:`graphbench.100k.q2.polars`
     - :bench:`graphbench.100k.q2.polars_vs_pandas`
     - :bench:`graphbench.20k.q2.polars_vs_pandas`
     - city of the most-followed person
   * - q3
     - :bench:`graphbench.100k.q3.pandas`
     - :bench:`graphbench.100k.q3.polars`
     - :bench:`graphbench.100k.q3.polars_vs_pandas`
     - :bench:`graphbench.20k.q3.polars_vs_pandas`
     - five lowest-average-age cities
   * - q4
     - :bench:`graphbench.100k.q4.pandas`
     - :bench:`graphbench.100k.q4.polars`
     - :bench:`graphbench.100k.q4.polars_vs_pandas`
     - :bench:`graphbench.20k.q4.polars_vs_pandas`
     - per-country person counts
   * - q5
     - :bench:`graphbench.100k.q5.pandas`
     - :bench:`graphbench.100k.q5.polars`
     - :bench:`graphbench.100k.q5.polars_vs_pandas`
     - :bench:`graphbench.20k.q5.polars_vs_pandas`
     - filtered population count
   * - q6
     - :bench:`graphbench.100k.q6.pandas`
     - :bench:`graphbench.100k.q6.polars`
     - :bench:`graphbench.100k.q6.polars_vs_pandas`
     - :bench:`graphbench.20k.q6.polars_vs_pandas`
     - filtered population by city
   * - q7
     - :bench:`graphbench.100k.q7.pandas`
     - :bench:`graphbench.100k.q7.polars`
     - :bench:`graphbench.100k.q7.polars_vs_pandas`
     - :bench:`graphbench.20k.q7.polars_vs_pandas`
     - interest-filtered count
   * - q8
     - :bench:`graphbench.100k.q8.pandas`
     - :bench:`graphbench.100k.q8.polars`
     - :bench:`graphbench.100k.q8.polars_vs_pandas`
     - :bench:`graphbench.20k.q8.polars_vs_pandas`
     - two-hop path count
   * - q9
     - :bench:`graphbench.100k.q9.pandas`
     - :bench:`graphbench.100k.q9.polars`
     - :bench:`graphbench.100k.q9.polars_vs_pandas`
     - :bench:`graphbench.20k.q9.polars_vs_pandas`
     - filtered two-hop path count

Every speedup is larger on the bigger graph. The margin is widest on the scan-and-filter
shapes — q5, q6, q8 — where pandas materializes an intermediate the Polars plan never
builds. The speedup is therefore a property of workload size and shape, not a constant;
:doc:`engines` covers how to route.

.. _gfql-vs-kuzu-board:

Against an embedded graph database
----------------------------------

Same nine queries, same graph, same session, against **Kuzu 0.11.3** running embedded on
the same host. The perf lock was held, the host was quiet, and the slot order was
position-balanced so neither side benefits from cache warmth or host drift. Each GFQL cell
had to return the same canonical row set as Kuzu on every slot before its ratio could be
published.

**On the 100,000-person lane, GFQL with** ``engine='polars'`` **answers five of the nine
queries faster than Kuzu.** Those five:

.. list-table::
   :header-rows: 1
   :widths: 10 22 22 18 28

   * - Query
     - Kuzu 0.11.3
     - GFQL ``polars``
     - GFQL is faster by
     - What it does
   * - q1
     - :bench:`graphbench.100k.q1.kuzu`
     - :bench:`graphbench.100k.q1.polars`
     - :bench:`graphbench.100k.q1.polars_vs_kuzu`
     - top-3 by in-degree
   * - q2
     - :bench:`graphbench.100k.q2.kuzu`
     - :bench:`graphbench.100k.q2.polars`
     - :bench:`graphbench.100k.q2.polars_vs_kuzu`
     - city of the most-followed person
   * - q3
     - :bench:`graphbench.100k.q3.kuzu`
     - :bench:`graphbench.100k.q3.polars`
     - :bench:`graphbench.100k.q3.polars_vs_kuzu`
     - five lowest-average-age cities
   * - q8
     - :bench:`graphbench.100k.q8.kuzu`
     - :bench:`graphbench.100k.q8.polars`
     - :bench:`graphbench.100k.q8.polars_vs_kuzu`
     - two-hop path count
   * - q9
     - :bench:`graphbench.100k.q9.kuzu`
     - :bench:`graphbench.100k.q9.polars`
     - :bench:`graphbench.100k.q9.polars_vs_kuzu`
     - filtered two-hop path count

Read the scope precisely, because a benchmark headline without one is worthless: this is
the ``prrao87/graph-benchmark`` q1–q9 suite at 100,000 persons on a quiet, perf-locked
dgx-spark, GFQL on ``engine='polars'`` against embedded Kuzu 0.11.3, warm medians, rows
validated cell by cell. It is five of the nine queries in that suite at that size. It is
not a claim about every query, every engine, or every size, and the four queries not
listed are not published here.

The setup difference is not a footnote either. The GFQL side queried a dataframe that was
already in memory: no store to provision, no load step, no index to build before the first
query runs.

Provenance for both tables
--------------------------

.. bench-provenance:: graphbench-q1q9-100k-20260726

.. bench-provenance:: graphbench-q1q9-20k-20260726

.. bench-disclosures::

How GFQL is fast
----------------

Three design choices explain the speedups:

**Collection-oriented execution.** GFQL evaluates whole collections of nodes and edges at
once (set-at-a-time) rather than walking one path at a time. A traversal advances by
joining edge tables, so the work vectorizes.

**Vectorized columnar processing.** Data is processed in columnar batches on top of
`Apache Arrow <https://arrow.apache.org/>`_, which keeps the CPU path fast and makes moving
data between systems cheap. The ``polars`` engine additionally builds **one fused lazy plan
and collects once**, which is why it outruns the eager pandas path on bulk work.

**Massive parallelism on GPUs.** On an NVIDIA GPU (``cudf`` / ``polars-gpu``) the same
vectorized work saturates tens of thousands of threads, paying off once there is enough
work to amortize kernel-launch cost: large frontiers, dense joins, full-graph aggregation.

Start on CPU with no special hardware, and move to a GPU engine by changing one keyword.
:doc:`engines` says exactly when each one wins.

.. note::
   Same-path constraints (``where``) can be more expensive on dense graphs.
   Prefer selective per-step predicates and see :doc:`/gfql/where` for details.

How a number gets published here
--------------------------------

Every figure above is generated, not transcribed:

1. The benchmark harness lives in `graphistry/pyg-bench
   <https://github.com/graphistry/pyg-bench>`_, which commits its raw per-slot artifacts
   alongside the results.
2. ``scripts/export_docs_numbers.py`` in that repository turns those committed artifacts
   into ``docs/source/_data/gfql_benchmarks.json`` here, deriving each median, each ratio
   and each cell's publishability from the artifacts rather than from anyone's notes.
   Regenerating the docs' numbers needs **no GPU box and no benchmark re-run**.
3. The docs build resolves every ``:bench:`` reference against that file, and fails if a
   key is missing, if a run has aged past the policy threshold, or if a page drops a
   number's provenance or disclosures.
4. ``bin/check_bench_numbers.py`` runs the same gate outside Sphinx and adds a commit-drift
   check: once the query engine has moved materially since a published run was measured,
   the number is treated as stale and the check fails.

A figure that cannot be traced to a committed artifact is not published here.

Next Steps
----------

- **Choose an engine**: :doc:`engines` — the decision matrix and routing guidance.
- **Seeded lookups**: :doc:`index_adjacency` — the CSR adjacency index.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.
- **Ecosystem**: `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.

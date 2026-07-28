.. _gfql-performance:

GFQL Performance: Vectorization and GPU Acceleration
====================================================

.. note::
   This page and :doc:`index_adjacency` are the **only** places PyGraphistry
   publishes measured benchmark numbers. Every figure below is referenced from a
   single machine-readable source of truth
   (``docs/source/_data/gfql_benchmarks.json``) generated from committed
   benchmark artifacts — the docs build fails rather than render a number that
   the source of truth does not contain, or one whose run has gone stale. See
   `Reproducing these numbers`_.

Engine speedups at a glance
---------------------------

GFQL runs the **same query** on four interchangeable engines — ``pandas`` (default),
``polars`` (CPU, columnar), ``cudf`` (NVIDIA GPU), and ``polars-gpu`` (GPU) — and returns
**identical results** on each (differential parity is a release gate). Unsupported
engine/query combinations are declined before execution during validation, compilation,
or planning rather than silently falling back. The biggest, easiest win is one keyword,
**no GPU required**:

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # same results, often much faster

The board below is the ``prrao87/graph-benchmark`` q1–q9 Cypher suite: nine
analytical queries (degree ranking, grouped aggregation, multi-hop expansion,
path counting) over a synthetic social graph. Every cell is a real GFQL Cypher
execution — no dataframe shortcut, no untimed precompute — and every GFQL cell's
result rows were checked against the reference engine's before it was allowed to
be published.

**pandas vs Polars, same query, identical result rows** (100,000-person graph,
2.78M edges):

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 20 30

   * - Query
     - ``pandas``
     - ``polars``
     - Polars speedup
     - What it does
   * - q1
     - :bench:`graphbench.100k.q1.pandas`
     - :bench:`graphbench.100k.q1.polars`
     - :bench:`graphbench.100k.q1.polars_vs_pandas`
     - top-3 by in-degree
   * - q2
     - :bench:`graphbench.100k.q2.pandas`
     - :bench:`graphbench.100k.q2.polars`
     - :bench:`graphbench.100k.q2.polars_vs_pandas`
     - city of the most-followed person
   * - q3
     - :bench:`graphbench.100k.q3.pandas`
     - :bench:`graphbench.100k.q3.polars`
     - :bench:`graphbench.100k.q3.polars_vs_pandas`
     - five lowest-average-age cities
   * - q4
     - :bench:`graphbench.100k.q4.pandas`
     - :bench:`graphbench.100k.q4.polars`
     - :bench:`graphbench.100k.q4.polars_vs_pandas`
     - per-country person counts
   * - q5
     - :bench:`graphbench.100k.q5.pandas`
     - :bench:`graphbench.100k.q5.polars`
     - :bench:`graphbench.100k.q5.polars_vs_pandas`
     - filtered population count
   * - q6
     - :bench:`graphbench.100k.q6.pandas`
     - :bench:`graphbench.100k.q6.polars`
     - :bench:`graphbench.100k.q6.polars_vs_pandas`
     - filtered population by city
   * - q7
     - :bench:`graphbench.100k.q7.pandas`
     - :bench:`graphbench.100k.q7.polars`
     - :bench:`graphbench.100k.q7.polars_vs_pandas`
     - interest-filtered count
   * - q8
     - :bench:`graphbench.100k.q8.pandas`
     - :bench:`graphbench.100k.q8.polars`
     - :bench:`graphbench.100k.q8.polars_vs_pandas`
     - two-hop path count
   * - q9
     - :bench:`graphbench.100k.q9.pandas`
     - :bench:`graphbench.100k.q9.polars`
     - :bench:`graphbench.100k.q9.polars_vs_pandas`
     - filtered two-hop path count

Polars wins every one of the nine queries at this size, and the margin is widest
on the scan-and-filter shapes (q5, q6, q8) where pandas materializes an
intermediate the Polars plan never builds. On the ten-times-smaller graph the
same queries still favour Polars, but by a much smaller factor — q5 moves from
:bench:`graphbench.20k.q5.polars_vs_pandas` at 20,000 persons to
:bench:`graphbench.100k.q5.polars_vs_pandas` at 100,000. **The speedup is a
property of workload size and shape, not a constant**; see :doc:`engines` for
how to route.

.. _gfql-vs-kuzu-board:

Against an embedded graph database
----------------------------------

The same nine queries, same graph, same session, against **Kuzu** — an embedded,
columnar, worst-case-optimal-join graph database — with the perf lock held and
the slot order position-balanced so neither side benefits from cache warmth or
host drift. A ratio above 1 means GFQL-Polars is faster.

.. list-table::
   :header-rows: 1
   :widths: 10 18 18 18 18 18

   * - Query
     - Kuzu (20k)
     - GFQL-Polars (20k)
     - Ratio (20k)
     - Kuzu (100k)
     - Ratio (100k)
   * - q1
     - :bench:`graphbench.20k.q1.kuzu`
     - :bench:`graphbench.20k.q1.polars`
     - :bench:`graphbench.20k.q1.polars_vs_kuzu`
     - :bench:`graphbench.100k.q1.kuzu`
     - :bench:`graphbench.100k.q1.polars_vs_kuzu`
   * - q2
     - :bench:`graphbench.20k.q2.kuzu`
     - :bench:`graphbench.20k.q2.polars`
     - :bench:`graphbench.20k.q2.polars_vs_kuzu`
     - :bench:`graphbench.100k.q2.kuzu`
     - :bench:`graphbench.100k.q2.polars_vs_kuzu`
   * - q3
     - :bench:`graphbench.20k.q3.kuzu`
     - :bench:`graphbench.20k.q3.polars`
     - :bench:`graphbench.20k.q3.polars_vs_kuzu`
     - :bench:`graphbench.100k.q3.kuzu`
     - :bench:`graphbench.100k.q3.polars_vs_kuzu`
   * - q4
     - :bench:`graphbench.20k.q4.kuzu`
     - :bench:`graphbench.20k.q4.polars`
     - :bench:`graphbench.20k.q4.polars_vs_kuzu`
     - :bench:`graphbench.100k.q4.kuzu`
     - :bench:`graphbench.100k.q4.polars_vs_kuzu`
   * - q5
     - :bench:`graphbench.20k.q5.kuzu`
     - :bench:`graphbench.20k.q5.polars`
     - :bench:`graphbench.20k.q5.polars_vs_kuzu`
     - :bench:`graphbench.100k.q5.kuzu`
     - :bench:`graphbench.100k.q5.polars_vs_kuzu`
   * - q6
     - :bench:`graphbench.20k.q6.kuzu`
     - :bench:`graphbench.20k.q6.polars`
     - :bench:`graphbench.20k.q6.polars_vs_kuzu`
     - :bench:`graphbench.100k.q6.kuzu`
     - :bench:`graphbench.100k.q6.polars_vs_kuzu`
   * - q7
     - :bench:`graphbench.20k.q7.kuzu`
     - :bench:`graphbench.20k.q7.polars`
     - :bench:`graphbench.20k.q7.polars_vs_kuzu`
     - :bench:`graphbench.100k.q7.kuzu`
     - :bench:`graphbench.100k.q7.polars_vs_kuzu`
   * - q8
     - :bench:`graphbench.20k.q8.kuzu`
     - :bench:`graphbench.20k.q8.polars`
     - :bench:`graphbench.20k.q8.polars_vs_kuzu`
     - :bench:`graphbench.100k.q8.kuzu`
     - :bench:`graphbench.100k.q8.polars_vs_kuzu`
   * - q9
     - :bench:`graphbench.20k.q9.kuzu`
     - :bench:`graphbench.20k.q9.polars`
     - :bench:`graphbench.20k.q9.polars_vs_kuzu`
     - :bench:`graphbench.100k.q9.kuzu`
     - :bench:`graphbench.100k.q9.polars_vs_kuzu`

**Read this board honestly.** On the small graph GFQL wins three of the nine and
loses six: Kuzu answers most of these queries in single-digit milliseconds,
below GFQL's per-query planning floor, so there is not enough work to amortize.
An order of magnitude larger, the picture inverts — GFQL leads on q1, q2, q3, q8
and q9, is level on q4 and q6, and still trails on q5 and q7. **Size, not
branding, decides.** If your queries are small and latency-bound, an embedded
database with persistent indexes is a fair choice; if they scan and aggregate
real volume, the columnar plan wins — and it wins with no separate store to
provision, load and index.

Two things this board does **not** cover:

- **It is the Polars engine.** GFQL-pandas loses to Kuzu on most cells at both
  sizes (compare the pandas column in the previous table against the Kuzu column
  here). Never quote the board without saying which engine produced it.
- **Cyclic and multi-way-join patterns** (triangles, cliques) are not in this
  suite. Kuzu's worst-case-optimal joins can beat a dataframe plan there, and we
  make no claim about them.

.. bench-provenance:: graphbench-q1q9-20k-20260726

.. bench-provenance:: graphbench-q1q9-100k-20260726

.. bench-disclosures::

How GFQL is fast
----------------

Three design choices explain the numbers above:

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
The inverse also holds: on a tiny, selective query the launch cost dominates and a CPU
engine wins.

Start on CPU with no special hardware, and move to a GPU engine by changing one keyword when
your workload grows into GPU territory. See :doc:`engines` for exactly when each engine wins.

.. note::
   Same-path constraints (``where``) can be more expensive on dense graphs.
   Prefer selective per-step predicates and see :doc:`/gfql/where` for details.

Reproducing these numbers
-------------------------

Every figure on this page is generated, not transcribed:

1. The benchmark harness lives in `graphistry/pyg-bench
   <https://github.com/graphistry/pyg-bench>`_, which commits its raw per-slot
   artifacts alongside the results.
2. ``scripts/export_docs_numbers.py`` in that repository turns those committed
   artifacts into ``docs/source/_data/gfql_benchmarks.json`` here — deriving each
   median, each ratio and each cell's publishability from the artifacts rather
   than from anyone's notes. **No GPU box and no benchmark re-run is needed** to
   regenerate the docs from stored results.
3. The docs build resolves every ``:bench:`` reference against that file, and
   fails if a key is missing, if a run has aged past the policy threshold, or if
   a page drops a number's provenance or disclosures.
4. ``bin/check_bench_numbers.py`` runs the same gate outside Sphinx and adds a
   commit-drift check: when the query engine has moved materially since a
   published run was measured, the number is treated as stale and the check
   fails.

Numbers whose originating run could not be reproduced from a committed artifact
have been removed from these docs rather than restated.

Next Steps
----------

- **Choose an engine**: :doc:`engines` — the decision matrix and routing guidance.
- **Seeded lookups**: :doc:`index_adjacency` — the CSR adjacency index.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.
- **Ecosystem**: `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.

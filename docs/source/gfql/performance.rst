.. _gfql-performance:

GFQL Performance: Vectorization and GPU Acceleration
====================================================

This page is the **canonical home for GFQL benchmark numbers** — the measured tables live
here, while the rest of the docs make stable qualitative claims and link back here. The
resident index behind the seeded-lookup numbers is documented in :doc:`index_adjacency`.

Engine speedups at a glance
---------------------------

GFQL runs the **same query** on four interchangeable engines — ``pandas`` (default),
``polars`` (CPU, columnar), ``cudf`` (NVIDIA GPU), and ``polars-gpu`` (GPU) — and returns
**identical results** on each (differential parity is a release gate; every four-engine
number on this page was kept only after the result rows were verified identical across
engines, and the cross-database pairs were validated against expected result rows).
Unsupported engine/query combinations are declined before execution during validation,
compilation, or planning rather than silently falling back. The biggest, easiest win is one
keyword, **no GPU required**:

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # often much faster on query-heavy workloads, same results

For example, in the release-verified sweep below, the LDBC SNB SF1 seed-lookup drops from
**1,299.6 ms** on eager pandas to **106.1 ms** with ``engine='polars'`` — a **12.3×**
one-keyword speedup, no GPU, results identical.

.. _gfql-0580-numbers:

Release-verified numbers (0.58.0)
---------------------------------

All numbers in this section were measured on the **0.58.0 release tag** on an NVIDIA DGX
Spark (GB10), warm medians over N=30 runs. The four-engine numbers (seeded fast paths,
resident index, scaling) were kept only after the result rows were asserted identical
across engines; the competitor pairs (vs Neo4j, vs Kuzu) were validated against expected
result rows and cross-database value/row-count checks.

Seeded typed-hop fast path
~~~~~~~~~~~~~~~~~~~~~~~~~~

A seeded typed hop — Cypher ``MATCH (m {id: ...})-[:T]->(p) RETURN p`` on a 50k-node /
200k-edge graph — is the bread-and-butter selective lookup. The release's fast path speeds
it up on every engine (before → after within the sweep):

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 15

   * - Engine
     - Before
     - After (fast path)
     - Speedup
   * - ``pandas``
     - 29.9 ms
     - **2.46 ms**
     - 12.1×
   * - ``polars``
     - 13.8 ms
     - **2.28 ms**
     - 6.1×
   * - ``cudf``
     - 30.1 ms
     - **4.89 ms**
     - 6.1×
   * - ``polars-gpu``
     - 25.2 ms
     - **2.49 ms**
     - 10.1×

The native chain form of the same query is faster still: pandas 21.1 → **1.65 ms**
(12.8×), cuDF 23.2 → **3.84 ms** (6.0×).

With a resident index
~~~~~~~~~~~~~~~~~~~~~

Building the opt-in resident index once (``g.gfql_index_all()``) makes the covered-shape
seeded lookup faster again — pandas **1.74 ms**, polars **1.59 ms**, polars-gpu
**1.91 ms**, cudf **5.78 ms**.

.. warning::
   **Polars + index: pass** ``engine='polars'`` **when building.** Polars frames currently
   need ``g.gfql_index_all(engine='polars')`` explicitly — an AUTO build swaps Polars
   frames to pandas. The fix is tracked in PR #1767.

Scaling: flat in graph size
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the resident adjacency index, a native seeded 1-hop ``g.hop()`` on pandas stays
**flat at 0.159–0.164 ms from 0.25M to 32M edges** (constant average degree 4): the index
turns the O(E) scan into an O(degree) gather, so seeded latency does not grow with the
graph. (Pandas-only today — the Polars hop path is not yet index-routed.)

vs Neo4j (LDBC SNB interactive SF1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same box, warm, against Neo4j 5.26 — GFQL wins **4 of 5** clean pairs:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15

   * - Query
     - GFQL
     - Neo4j 5.26
     - Winner
   * - seed-lookup
     - **106.1 ms**
     - 143.7 ms
     - GFQL
   * - message-content
     - **7.1 ms**
     - 23.0 ms
     - GFQL
   * - message-creator
     - **6.8 ms**
     - 27.7 ms
     - GFQL
   * - one-hop-expand
     - **111.9 ms**
     - 180.7 ms
     - GFQL
   * - recent-replies
     - 209.6 ms
     - **104.0 ms**
     - Neo4j

The message-creator flip shipped in this release via property-seeded resident-index
gathers (PR #1770). Neo4j still wins recent-replies — reported as-is.

OLAP multi-join
~~~~~~~~~~~~~~~

The OLAP multi-join comparison against an embedded graph database is the q1–q9 board
below (:ref:`gfql-vs-kuzu-board`). Earlier q8/q9 figures from the July lane are withdrawn:
that lane's results directory carries no receipts, so its numbers are not citable. The
board below is from a receipted re-run.

When not to use GFQL
~~~~~~~~~~~~~~~~~~~~

Honesty matters more than a bigger number. In the same cross-DB sweep, **embedded Kuzu
wins single-table aggregates (2–4×) and seeded property-projection lookups (2.4–64×)**.
GFQL's strengths are **traversals, multi-join OLAP, and covered seeded shapes** — route by
workload, and keep a database as the system-of-record where one fits.

.. _gfql-vs-kuzu-board:

The q1–q9 board: GFQL vs an embedded graph database
---------------------------------------------------

This is the ``prrao87/graph-benchmark`` q1–q9 Cypher suite — degree ranking, grouped
aggregation, filtered population counts, two-hop path counting — on a synthetic social
graph, measured at two scales: 20,000 persons and 100,000 persons. GFQL runs the suite on
``engine='pandas'`` and ``engine='polars'`` against Kuzu embedded on the same host, in
the same session. Slots run in a position-balanced order so no side benefits from cache
warmth or host drift; each cell is the median across four slots of 51 timed runs each
(after 5 warmups); a verdict inside a 10% band is a tie. A verdict is published only
where GFQL returned the same result rows as Kuzu.

**The 20k board is 5 wins, 2 ties, 2 losses for GFQL-Polars. The 100k board is 6 wins,
2 ties, 1 loss.** The losses are printed with the same weight as the wins. "Kuzu ÷
Polars" is the Kuzu median divided by the GFQL-Polars median, so values above 1 mean
GFQL-Polars is faster.

The 20,000-person board
~~~~~~~~~~~~~~~~~~~~~~~

Two verdicts on this board (q4, q5) are weak: the per-slot median ranges of the two
sides overlap.

.. list-table::
   :header-rows: 1
   :widths: 8 10 14 15 15 14 24

   * - Query
     - Result rows
     - Kuzu
     - GFQL ``pandas``
     - GFQL ``polars``
     - Kuzu ÷ Polars
     - Verdict
   * - q1
     - 3
     - :bench:`graphbench.20k.q1.kuzu`
     - :bench:`graphbench.20k.q1.pandas`
     - :bench:`graphbench.20k.q1.polars`
     - :bench:`graphbench.20k.q1.polars_vs_kuzu`
     - WIN
   * - q2
     - 1
     - :bench:`graphbench.20k.q2.kuzu`
     - :bench:`graphbench.20k.q2.pandas`
     - :bench:`graphbench.20k.q2.polars`
     - :bench:`graphbench.20k.q2.polars_vs_kuzu`
     - WIN
   * - q3
     - 5
     - :bench:`graphbench.20k.q3.kuzu`
     - :bench:`graphbench.20k.q3.pandas`
     - :bench:`graphbench.20k.q3.polars`
     - :bench:`graphbench.20k.q3.polars_vs_kuzu`
     - TIE
   * - q4
     - 2
     - :bench:`graphbench.20k.q4.kuzu`
     - :bench:`graphbench.20k.q4.pandas`
     - :bench:`graphbench.20k.q4.polars`
     - :bench:`graphbench.20k.q4.polars_vs_kuzu`
     - LOSE (weak: slot ranges overlap)
   * - q5
     - 1
     - :bench:`graphbench.20k.q5.kuzu`
     - :bench:`graphbench.20k.q5.pandas`
     - :bench:`graphbench.20k.q5.polars`
     - :bench:`graphbench.20k.q5.polars_vs_kuzu`
     - WIN (weak: slot ranges overlap)
   * - q6
     - 5
     - :bench:`graphbench.20k.q6.kuzu`
     - :bench:`graphbench.20k.q6.pandas`
     - :bench:`graphbench.20k.q6.polars`
     - :bench:`graphbench.20k.q6.polars_vs_kuzu`
     - WIN
   * - q7
     - 1
     - :bench:`graphbench.20k.q7.kuzu`
     - :bench:`graphbench.20k.q7.pandas`
     - :bench:`graphbench.20k.q7.polars`
     - :bench:`graphbench.20k.q7.polars_vs_kuzu`
     - TIE
   * - q8
     - 1
     - :bench:`graphbench.20k.q8.kuzu`
     - :bench:`graphbench.20k.q8.pandas`
     - :bench:`graphbench.20k.q8.polars`
     - :bench:`graphbench.20k.q8.polars_vs_kuzu`
     - LOSE
   * - q9
     - 1
     - :bench:`graphbench.20k.q9.kuzu`
     - :bench:`graphbench.20k.q9.pandas`
     - :bench:`graphbench.20k.q9.polars`
     - :bench:`graphbench.20k.q9.polars_vs_kuzu`
     - WIN

Read the losses plainly. On q8 (two-hop path count) Kuzu answers in
:bench:`graphbench.20k.q8.kuzu` against GFQL-Polars' :bench:`graphbench.20k.q8.polars` —
Kuzu is about three times faster, measured one-shot with no cross-call cache. On q4
(per-country person counts) Kuzu leads inside overlapping slot ranges, so that verdict is
weak in Kuzu's favor.

The comparator's own summary table, transcribed verbatim from
``results/graphbench-board-20k-20260802/compare.txt``::

   q     rows   kuzu ms   pandas ms   polars ms  best GFQL   ratio  verdict
   ------------------------------------------------------------------------
   q1       3     15.30       35.18        8.95     polars    1.71  WIN 1.71x
   q2       1     35.67       37.88       13.06     polars    2.73  WIN 2.73x
   q3       5      5.68       11.67        5.80     polars    0.98  TIE
   q4       2      3.35        9.49        3.97     polars    0.84  LOSE 1.19x (WEAK: slot ranges overlap)
   q5       1      5.39       80.73        4.50     polars    1.20  WIN 1.20x (WEAK: slot ranges overlap)
   q6       5      8.90       80.44        5.73     polars    1.55  WIN 1.55x
   q7       1      5.19       19.82        5.68     polars    0.91  TIE
   q8       1      2.80       19.60        8.24     polars    0.34  LOSE 2.94x
   q9       1     10.76       28.26        8.32     polars    1.29  WIN 1.29x

The 100,000-person board
~~~~~~~~~~~~~~~~~~~~~~~~

Same suite, same protocol, same session structure, on the 100,000-person graph. No
verdict on this board is weak: no per-slot median range overlaps between the two sides.

.. list-table::
   :header-rows: 1
   :widths: 8 10 14 15 15 14 24

   * - Query
     - Result rows
     - Kuzu
     - GFQL ``pandas``
     - GFQL ``polars``
     - Kuzu ÷ Polars
     - Verdict
   * - q1
     - 3
     - :bench:`graphbench.100k.q1.kuzu`
     - :bench:`graphbench.100k.q1.pandas`
     - :bench:`graphbench.100k.q1.polars`
     - :bench:`graphbench.100k.q1.polars_vs_kuzu`
     - WIN
   * - q2
     - 1
     - :bench:`graphbench.100k.q2.kuzu`
     - :bench:`graphbench.100k.q2.pandas`
     - :bench:`graphbench.100k.q2.polars`
     - :bench:`graphbench.100k.q2.polars_vs_kuzu`
     - WIN
   * - q3
     - 5
     - :bench:`graphbench.100k.q3.kuzu`
     - :bench:`graphbench.100k.q3.pandas`
     - :bench:`graphbench.100k.q3.polars`
     - :bench:`graphbench.100k.q3.polars_vs_kuzu`
     - WIN
   * - q4
     - 3
     - :bench:`graphbench.100k.q4.kuzu`
     - :bench:`graphbench.100k.q4.pandas`
     - :bench:`graphbench.100k.q4.polars`
     - :bench:`graphbench.100k.q4.polars_vs_kuzu`
     - WIN
   * - q5
     - 1
     - :bench:`graphbench.100k.q5.kuzu`
     - :bench:`graphbench.100k.q5.pandas`
     - :bench:`graphbench.100k.q5.polars`
     - :bench:`graphbench.100k.q5.polars_vs_kuzu`
     - TIE
   * - q6
     - 5
     - :bench:`graphbench.100k.q6.kuzu`
     - :bench:`graphbench.100k.q6.pandas`
     - :bench:`graphbench.100k.q6.polars`
     - :bench:`graphbench.100k.q6.polars_vs_kuzu`
     - WIN
   * - q7
     - 1
     - :bench:`graphbench.100k.q7.kuzu`
     - :bench:`graphbench.100k.q7.pandas`
     - :bench:`graphbench.100k.q7.polars`
     - :bench:`graphbench.100k.q7.polars_vs_kuzu`
     - TIE
   * - q8
     - 1
     - :bench:`graphbench.100k.q8.kuzu`
     - :bench:`graphbench.100k.q8.pandas`
     - :bench:`graphbench.100k.q8.polars`
     - :bench:`graphbench.100k.q8.polars_vs_kuzu`
     - LOSE
   * - q9
     - 1
     - :bench:`graphbench.100k.q9.kuzu`
     - :bench:`graphbench.100k.q9.pandas`
     - :bench:`graphbench.100k.q9.polars`
     - :bench:`graphbench.100k.q9.polars_vs_kuzu`
     - WIN

The comparator's own summary table, transcribed verbatim from
``results/graphbench-board-100k-v2-20260802/compare.txt``::

   q     rows   kuzu ms   pandas ms   polars ms  best GFQL   ratio  verdict
   ------------------------------------------------------------------------
   q1       3    153.62      199.30       31.29     polars    4.91  WIN 4.91x
   q2       1    278.64      209.32       44.33     polars    6.29  WIN 6.29x
   q3       5     34.23       73.48       13.66     polars    2.51  WIN 2.51x
   q4       3     13.50       64.41       10.51     polars    1.28  WIN 1.28x
   q5       1     13.17      412.25       13.68     polars    0.96  TIE
   q6       5     24.11      410.76       14.84     polars    1.62  WIN 1.62x
   q7       1      9.59      120.21        9.88     polars    0.97  TIE
   q8       1     13.62      152.75       33.25     polars    0.41  LOSE 2.44x
   q9       1     83.71      218.58       36.41     polars    2.30  WIN 2.30x

Reading the two boards together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**q8 loses at both scales.** Kuzu is faster on the two-hop path count at 20k and at 100k
— the comparator's own verdicts read ``LOSE 2.94x`` and ``LOSE 2.44x``. That is the one
query where the embedded database beats GFQL outright, and it stays a loss as the graph
grows.

**q4 flips.** At 20k it is a weak loss inside overlapping slot ranges; at 100k it is a
clean win at :bench:`graphbench.100k.q4.polars_vs_kuzu`, with no slot overlap.

**GFQL-Polars beats GFQL-pandas on all nine queries at both scales** — the two GFQL
columns in each table show it — so ``engine='polars'`` is the GFQL side of every verdict,
and the pandas-to-Polars gap is wider on the larger graph for every query.

The setup difference is real but not in the timings: the GFQL side queries a dataframe
that is already in memory — no store to provision, no load step, no index to build before
the first query runs.

No number from any earlier 100k run appears on this page: the July lane
(``results/graphbench-matched-q1q9-20260726``) carries no receipts and is withdrawn as
nonreproducible.

Provenance
~~~~~~~~~~

.. bench-provenance:: graphbench-board-20k-20260802

.. bench-provenance:: graphbench-board-100k-v2-20260802

.. bench-disclosures::

How a number gets published here
--------------------------------

Every figure in the boards above is generated, not transcribed by hand:

1. The benchmark harness lives in `graphistry/pyg-bench
   <https://github.com/graphistry/pyg-bench>`_, which commits its raw per-slot artifacts —
   timings, result rows, host-load and spike captures, runner-script checksums — alongside
   the results.
2. Those committed artifacts become ``docs/source/_data/gfql_benchmarks.json`` here; each
   median, each ratio, and each cell's publishability derives from the artifacts rather
   than from anyone's notes.
3. The docs build resolves every ``:bench:`` reference against that file, and fails if a
   key is missing, if a run has aged past the freshness policy, or if a page drops a
   number's provenance or disclosures.
4. ``docs/test_bench_numbers.py`` re-checks the same contract in the ordinary test lane,
   so a stale or unpublishable number fails CI even when the docs job does not run.

A figure that cannot be traced to a committed artifact is not published in this board.

.. _gfql-bulk-sweep:

Bulk engine comparison (prior sweep)
------------------------------------

The numbers in this section are from an earlier, pre-0.58.0 bulk sweep on SNAP
**com-LiveJournal** (35M edges) and **com-Orkut** (117M edges) — retained as the
bulk-workload reference until rerun on a current tag.

Same query, same answers, four engines — warm-median latency on Orkut (3.1M nodes /
117M edges), measured on a single machine:

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 16

   * - Workload (Orkut, 117M edges)
     - ``pandas``
     - ``polars``
     - ``cudf``
     - ``polars-gpu``
   * - 1-hop from 10K seeds
     - 2613 ms
     - **68 ms**
     - 1005 ms
     - 63 ms
   * - 2-hop from 10K seeds
     - 18161 ms
     - 2695 ms
     - 2774 ms
     - **1518 ms**
   * - Full out-degree aggregation
     - 799 ms
     - 205 ms
     - 314 ms
     - **167 ms**
   * - 2-hop from 100K seeds (~85M output rows)
     - 28822 ms
     - 8215 ms
     - **6002 ms**
     - 8559 ms

Reading the table:

- **Polars-CPU beat pandas up to ~38x** on bulk traversal and ~4x on aggregation — **with
  no GPU**. On the 1-hop workload it was ~38x faster than pandas (68 ms vs 2613 ms).
- **Polars-CPU also beat cuDF** on these shapes (68 ms vs 1005 ms on 1-hop). cuDF runs
  GFQL *eagerly*, op by op (a kernel launch + a materialized intermediate per hop), while
  Polars builds **one fused lazy plan and collects once**. The fused plan wins until the
  work is large enough to amortize GPU launch costs.
- **Polars-GPU was fastest on heavy multi-hop** (2-hop from 10K seeds: 1518 ms) and on
  aggregation — the same fused plan, executed on the GPU.
- **cuDF won the one extreme case** — a 2-hop from 100K seeds materializing ~85M output
  rows (6.0 s) — where raw GPU throughput on a single massive join overtakes everything
  and Polars-GPU comes under memory pressure.
- On LiveJournal (35M edges) the pattern held: 1-hop from 10K seeds was pandas 1129 ms →
  polars **37 ms** (~30x).
- The CPU crossover is early: on LiveJournal subsampled (CPU, warm-median), 1-hop
  traversal was 2.7× / 4.5× / 7.6× and ``WHERE``+``ORDER`` 3.0× / 3.0× / 18× over pandas
  at 10K / 100K / 1M edges. The only case pandas edged out was a trivial sub-millisecond
  operation (a bare node-equality filter), where its boolean mask beats Polars' plan
  overhead — immaterial at <1 ms. Reproducer: ``benchmarks/gfql/index_crossover_bench.py``.

Methodology (prior sweep)
~~~~~~~~~~~~~~~~~~~~~~~~~

- Host: NVIDIA DGX Spark (GB10 Grace-Blackwell, unified memory — the memory-pressure
  boundary above is partly a property of this box), RAPIDS container
  ``graphistry/test-rapids-official:26.02-gfql-polars``.
- Datasets: `SNAP <https://snap.stanford.edu/data/>`_ **com-LiveJournal** (35M edges),
  **com-Orkut** (117M edges).
- Measurement: **warm median** after 2 warmups (5 timed runs on Orkut, 8 on LiveJournal);
  every reported cell is **guarded** — the result rows are verified identical across
  engines before any timing is kept.
- Reproduce: ``benchmarks/gfql/index_bulk_olap_bench.py`` (engine comparison),
  ``benchmarks/gfql/pandas_vs_polars.py``, and ``benchmarks/gfql/index_vs_kuzu_prepared.py``
  (vs kuzu). Numbers on this page are rendered from saved runs; the page does not re-run
  them.
- **LadybugDB comparison** (referenced qualitatively in :doc:`engines`): **both sides
  measured on the host above, in one session, on the same generated 5M-node / 20M-edge
  graph** — LadybugDB **0.18.1** embedded in a host venv against GFQL ``engine='polars'``
  in the container above. Op shapes are those of
  `LadybugDB/kuzu-ladybug-benchmark <https://github.com/LadybugDB/kuzu-ladybug-benchmark>`_
  via ``benchmarks/gfql/bench_ladybug_cypher.py``; warm medians (2 warmups + 5 timed runs),
  slots interleaved L G G L, and **every op's result values are digest-identical across the
  two engines**. Ladybug is timed at its **fastest** result-producing scope of the four measured — for
  both cells below that is zero-copy Arrow, not its Python row iterator — because GFQL
  returns a materialized columnar frame.
  GFQL wins the node-scan shapes: **full node scan 59.0 ms vs 364.3 ms (6.2×)** and the
  1,001-row **range scan 5.1 ms vs 7.6 ms (1.5×)**. Point lookups stay with Ladybug's
  index seek over a columnar scan (a resident GFQL node-id index is tracked in issue
  #1676), as does a cached relationship ``COUNT(*)``.

There is **no universal winner**: ``polars`` typically takes over from ~10K edges up
(``pandas`` still wins trivial sub-millisecond operations), and the right GPU
engine depends on the workload. See :doc:`engines` for the full decision matrix, the honest
"when *not* to use Polars", and the cuDF-vs-Polars-GPU comparison. The end-to-end
CPU/GPU-vs-Neo4j pipeline benchmark is in :doc:`benchmark_filter_pagerank`, and the
Spark GraphFrames head-to-head is in :doc:`benchmark_graphframes`.

How GFQL is fast
----------------

Three design choices explain the numbers above:

**Collection-oriented execution.** GFQL evaluates whole collections of nodes and edges at
once (set-at-a-time), rather than walking one path at a time like traditional Cypher/Gremlin
engines. A traversal advances by joining edge tables, so the work vectorizes.

**Vectorized columnar processing.** Data is processed in columnar batches on top of
`Apache Arrow <https://arrow.apache.org/>`_, which keeps the CPU path fast and makes moving
data between systems cheap. The ``polars`` engine additionally builds **one fused lazy plan
and collects once**, which is why it outruns both pandas and eager cuDF on bulk work.

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
- **Selective lookups**: :doc:`index_adjacency` — the resident index behind the flat-scaling numbers.
- **End-to-end benchmark**: :doc:`benchmark_filter_pagerank` — CPU/GPU vs Neo4j+GDS.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.
- **Ecosystem**: `Apache Arrow <https://arrow.apache.org/>`_ and `NVIDIA RAPIDS <https://rapids.ai/>`_.

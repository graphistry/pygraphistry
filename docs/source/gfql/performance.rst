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
below: :ref:`gfql-vs-kuzu-board`.

When not to use GFQL
~~~~~~~~~~~~~~~~~~~~

Honesty matters more than a bigger number. In the same cross-DB sweep, **embedded Kuzu
wins single-table aggregates (2–4×) and seeded property-projection lookups (2.4–64×)**.
GFQL's strengths are **traversals, multi-join OLAP, and covered seeded shapes** — route by
workload, and keep a database as the system-of-record where one fits.

.. _gfql-vs-kuzu-board:

The q1–q9 board: GFQL vs an embedded graph database
---------------------------------------------------

**GFQL with** ``engine='polars'`` **wins 17 of these 18 head-to-head cells against
embedded Kuzu — 9 of 9 at 20k and 8 of 9 at 100k — with wins reaching**
:bench:`graphbench.100k.q2.polars_vs_kuzu`\ **.** The one loss is q8 at 100k, a
single-row two-hop count that Kuzu answers very fast: Kuzu takes it in
:bench:`graphbench.100k.q8.kuzu` against GFQL-Polars'
:bench:`graphbench.100k.q8.polars`. At 20k the same query is a win for GFQL, but a weak
one — the per-slot median ranges of the two sides overlap. The wins deepen with scale on
q1–q4 and q9; q5–q7 hold near their 20k margins; q8 moves the other way, from that weak
20k win to the one loss. Against Neo4j, this page carries the seeded
LDBC SNB pairs above — GFQL takes four of the five — and the receipted
filter → PageRank → filter pipeline against Neo4j + GDS is in
:doc:`benchmark_filter_pagerank`.

This is the ``prrao87/graph-benchmark`` q1–q9 Cypher suite — degree ranking, grouped
aggregation, filtered population counts, two-hop path counting — on a synthetic social
graph, measured at two scales: 20,000 persons and 100,000 persons. GFQL runs the suite on
``engine='pandas'`` and ``engine='polars'`` against Kuzu embedded on the same host, in
the same session. Slots run in a position-balanced order so no side benefits from cache
warmth or host drift; each cell is the median across four slots of 51 timed runs each
(after 5 warmups); a verdict inside a 10% band is a tie. A verdict is published only
where GFQL returned the same result rows as Kuzu.

**The 20k board is 9 wins, 0 ties, 0 losses for GFQL-Polars. The 100k board is 8 wins,
0 ties, 1 loss.** The loss is printed with the same weight as the wins. "Kuzu ÷
Polars" is the Kuzu median divided by the GFQL-Polars median, so values above 1 mean
GFQL-Polars is faster.

The 20,000-person board
~~~~~~~~~~~~~~~~~~~~~~~

One verdict on this board (q8) is weak: the per-slot median ranges of the two sides
overlap.

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
     - WIN
   * - q4
     - 2
     - :bench:`graphbench.20k.q4.kuzu`
     - :bench:`graphbench.20k.q4.pandas`
     - :bench:`graphbench.20k.q4.polars`
     - :bench:`graphbench.20k.q4.polars_vs_kuzu`
     - WIN
   * - q5
     - 1
     - :bench:`graphbench.20k.q5.kuzu`
     - :bench:`graphbench.20k.q5.pandas`
     - :bench:`graphbench.20k.q5.polars`
     - :bench:`graphbench.20k.q5.polars_vs_kuzu`
     - WIN
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
     - WIN
   * - q8
     - 1
     - :bench:`graphbench.20k.q8.kuzu`
     - :bench:`graphbench.20k.q8.pandas`
     - :bench:`graphbench.20k.q8.polars`
     - :bench:`graphbench.20k.q8.polars_vs_kuzu`
     - WIN (weak: slot ranges overlap)
   * - q9
     - 1
     - :bench:`graphbench.20k.q9.kuzu`
     - :bench:`graphbench.20k.q9.pandas`
     - :bench:`graphbench.20k.q9.polars`
     - :bench:`graphbench.20k.q9.polars_vs_kuzu`
     - WIN

Read the weak verdict plainly. On q8 (two-hop path count) GFQL-Polars answers in
:bench:`graphbench.20k.q8.polars` against Kuzu's :bench:`graphbench.20k.q8.kuzu`,
measured one-shot with no cross-call cache — but the per-slot median ranges of the two
sides overlap, so the comparator marks the win WEAK. Every other verdict on this board
has no slot overlap.

The comparator's own summary table, transcribed verbatim from
``results/graphbench-board-20k-cand-20260803/compare.txt``::

   q     rows   kuzu ms   pandas ms   polars ms  best GFQL   ratio  verdict
   ------------------------------------------------------------------------
   q1       3     15.14       33.04        7.54     polars    2.01  WIN 2.01x
   q2       1     41.70       35.63       11.33     polars    3.68  WIN 3.68x
   q3       5      6.48       11.18        4.60     polars    1.41  WIN 1.41x
   q4       2      3.26        9.23        2.88     polars    1.13  WIN 1.13x
   q5       1      5.24       80.21        4.05     polars    1.29  WIN 1.29x
   q6       5      8.98       79.43        4.58     polars    1.96  WIN 1.96x
   q7       1      5.18       19.06        3.47     polars    1.49  WIN 1.49x
   q8       1      2.81       10.05        2.22     polars    1.27  WIN 1.27x (WEAK: slot ranges overlap)
   q9       1     11.02       27.23        8.33     polars    1.32  WIN 1.32x

The 100,000-person board
~~~~~~~~~~~~~~~~~~~~~~~~

Same suite, same protocol, same session structure, on the 100,000-person graph. One
verdict on this board (q5) is weak: the per-slot median ranges of the two sides overlap.

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
     - WIN (weak: slot ranges overlap)
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
     - WIN
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
``results/graphbench-board-100k-cand-20260803/compare.txt``::

   q     rows   kuzu ms   pandas ms   polars ms  best GFQL   ratio  verdict
   ------------------------------------------------------------------------
   q1       3    153.09      195.05       26.53     polars    5.77  WIN 5.77x
   q2       1    273.63      205.10       39.19     polars    6.98  WIN 6.98x
   q3       5     34.19       71.10        9.92     polars    3.45  WIN 3.45x
   q4       3     13.25       62.78        9.73     polars    1.36  WIN 1.36x
   q5       1     13.35      409.29       11.35     polars    1.18  WIN 1.18x (WEAK: slot ranges overlap)
   q6       5     20.93      406.84       11.10     polars    1.89  WIN 1.89x
   q7       1      9.79      117.54        6.69     polars    1.46  WIN 1.46x
   q8       1      9.70      102.22       14.72     polars    0.66  LOSE 1.52x
   q9       1     84.20      210.14       35.70     polars    2.36  WIN 2.36x

Reading the two boards together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**q8 at 100k is the one loss.** On the two-hop path count the comparator's verdict at
20k reads ``WIN 1.27x (WEAK: slot ranges overlap)`` — a win for GFQL, but inside
overlapping slot ranges — and at 100k it reads ``LOSE 1.52x``: Kuzu answers in
:bench:`graphbench.100k.q8.kuzu` against GFQL-Polars'
:bench:`graphbench.100k.q8.polars`, measured one-shot with no cross-call cache. That is
the one cell on either board where the embedded database beats GFQL.

**Every other cell is a win at both scales**, with no slot overlap except q8 at 20k and
q5 at 100k, both flagged above.

**GFQL-Polars beats GFQL-pandas on all nine queries at both scales** — the two GFQL
columns in each table show it — so ``engine='polars'`` is the GFQL side of every verdict,
and the pandas-to-Polars gap is wider on the larger graph for every query.

The setup difference is real but not in the timings: the GFQL side queries a dataframe
that is already in memory — no store to provision, no load step, no index to build before
the first query runs.

Provenance
~~~~~~~~~~

.. bench-provenance:: graphbench-board-20k-cand-20260803

.. bench-provenance:: graphbench-board-100k-cand-20260803

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

The numbers in this section are from a pre-0.58.0 bulk sweep on SNAP
**com-LiveJournal** (35M edges) and **com-Orkut** (117M edges) — the bulk-workload
reference.

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

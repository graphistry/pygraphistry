.. _gfql-performance:

GFQL Performance: Measured Against Graph Databases
==================================================

This page holds GFQL's measured performance results. Every number renders from a
committed pyg-bench artifact; the Measurement block at the end names the runs, hosts,
and commits. Losses appear next to wins.

Choose an engine
----------------

GFQL runs the same query on ``pandas`` (the default), ``polars`` (CPU), ``cudf``
(NVIDIA GPU), or ``polars-gpu``, and every engine returns the same rows. On the q1–q9
boards below, the
Polars engine is faster than pandas on :bench-tally:`graphbench.100k|polars|pandas`
queries at 100,000 people, by up to :bench:`graphbench.100k.q6.polars_vs_pandas`
(q6). See :doc:`engines` for the selection guide.

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='pandas' (default)
   g.gfql(query, engine='polars')   # columnar CPU execution

.. _gfql-vs-kuzu-board:

The q1–q9 board: GFQL, Kuzu, Memgraph, and Neo4j
-------------------------------------------------

Nine Cypher queries from ``prrao87/graph-benchmark`` rank nodes by degree, group and
filter records, and count two-hop paths on synthetic social graphs with 20,000 and
100,000 people. Every cell passed result-row validation against every other engine.
Times are milliseconds; lower is better.

GFQL binds the graph cold inside every timed run; the ``polars-gpu`` column runs the
same fused plan on the GPU. Kuzu compiles the query text on each call. Memgraph and
Neo4j answer over Bolt with their default plan caches. These are direct times under
those profiles, not cross-engine speedup ratios. At these sizes the queries are
millisecond-scale, so the GPU engine wins some and loses others to the CPU engine:
:bench-tally:`graphbench.100k|polars_gpu|polars` at 100,000 people. Its widest loss is
q8 at 100,000 people, :bench:`graphbench.100k.q8.polars_gpu` against
:bench:`graphbench.100k.q8.polars` on the CPU.

At 20,000 people, GFQL Polars is faster than Kuzu on
:bench-tally:`graphbench.20k|polars|kuzu` queries, than Memgraph on
:bench-tally:`graphbench.20k|polars|memgraph`, and than Neo4j on
:bench-tally:`graphbench.20k|polars|neo4j`. At 100,000 people the counts are
:bench-tally:`graphbench.100k|polars|kuzu` (Kuzu),
:bench-tally:`graphbench.100k|polars|memgraph` (Memgraph), and
:bench-tally:`graphbench.100k|polars|neo4j` (Neo4j). Kuzu wins q4 at 20,000 people and
q8 at 100,000 people; the artifact's compare tables classify both as ties because the
per-slot medians overlap. Memgraph wins q3 and q6 at 20,000 people and q5, q6, and q7 at
100,000 people, where Neo4j also wins q5: their planners start from the ten-node
interest side, which GFQL's Cypher path does not yet do.

The 20,000-person board
~~~~~~~~~~~~~~~~~~~~~~~

.. bench-board:: graphbench.20k
   :rows: q1,q2,q3,q4,q5,q6,q7,q8,q9
   :columns: kuzu=Kuzu, memgraph=Memgraph, neo4j=Neo4j, pandas=GFQL pandas, polars=GFQL polars, polars_gpu=GFQL polars-gpu

The 100,000-person board
~~~~~~~~~~~~~~~~~~~~~~~~

.. bench-board:: graphbench.100k
   :rows: q1,q2,q3,q4,q5,q6,q7,q8,q9
   :columns: kuzu=Kuzu, memgraph=Memgraph, neo4j=Neo4j, pandas=GFQL pandas, polars=GFQL polars, polars_gpu=GFQL polars-gpu

.. _gfql-snb-aligned:

SNB-derived point and small-result queries: the databases win
-------------------------------------------------------------

Matched query shapes derived from the LDBC Social Network Benchmark (SNB) Interactive
workload, run on the SF0.1 and SF1 datasets without the official LDBC driver. This is
internal evidence, not an official LDBC result. All four engines ran under one timing
contract with exact result parity. Times are milliseconds.

Kuzu, Neo4j, and Memgraph are faster than GFQL on every point-lookup row, and Memgraph
is fastest on most. The GFQL columns run with resident indexes built once before the
timed runs (``gfql_index_all`` plus node property indexes), the same footing as the
databases' primary-key and label indexes. The index engages on the hop-shaped rows
(message replies, recent replies) and the cost there drops by 2.5x to 3x; on the pure
point lookups it does not engage, and what remains is a fixed per-call cost in the
chain pipeline of about 20 ms on the pandas and polars engines
(`#2027 <https://github.com/graphistry/pygraphistry/issues/2027>`_) against a database's
sub-millisecond index probe. GFQL's strengths are the bulk shapes above and on the
:doc:`speedup case study <benchmark_filter_pagerank>`; choose a database when the
workload is dominated by point lookups.

SF0.1
~~~~~

.. bench-board:: snb.sf01
   :rows: seed_lookup,message_content,message_creator,recent_replies,message_replies,new_topics
   :columns: gfql_polars_idx=GFQL polars, gfql_pandas_idx=GFQL pandas, kuzu=Kuzu, neo4j=Neo4j, memgraph=Memgraph
   :row-labels: seed_lookup=seed lookup; message_content=message content; message_creator=message creator; recent_replies=recent replies; message_replies=message replies (GFQL and Kuzu only); new_topics=new topics (GFQL and Kuzu only)

SF1
~~~

.. bench-board:: snb.sf1
   :rows: seed_lookup,message_content,message_creator,new_topics
   :columns: gfql_polars_idx=GFQL polars, gfql_pandas_idx=GFQL pandas, kuzu=Kuzu, neo4j=Neo4j, memgraph=Memgraph
   :row-labels: seed_lookup=seed lookup; message_content=message content; message_creator=message creator; recent_replies=recent replies; message_replies=message replies (GFQL and Kuzu only); new_topics=new topics (GFQL and Kuzu only)

Neo4j and Memgraph use a reduced adapter for one query, and one parameter returns zero
rows; those cells are excluded rather than estimated. SF10 was not run.

Lookups from known nodes
------------------------

A query that starts from known node ids (a watchlist, a session) scans every edge by
default. The opt-in adjacency index turns that scan into a gather over the seeds'
neighbors, so its cost tracks the seeds rather than the graph, on every engine. This
lane has not yet been measured under the provenance-carrying harness used above, so
this page prints no figure for it; see :doc:`index_adjacency` for the design and
:doc:`indexing` for the lifecycle.

How GFQL is fast, and when it is not
------------------------------------

GFQL joins tables of nodes and edges in batches instead of following one path at a
time, over columnar frames based on `Apache Arrow <https://arrow.apache.org/>`_. Polars
fuses the operations into one lazy plan and collects once; cuDF and Polars GPU run the
same columnar operations on NVIDIA GPUs. That favors bulk work: multi-join analytics,
frontier expansion from many seeds, and full-graph aggregation. It does not favor
single-row point lookups, where the per-call compile and row-pipeline floor dominates
and an indexed database answers in well under a millisecond, as the SNB tables show.

Start on CPU with no special hardware, and move to a GPU engine by changing one
keyword when the graph or result becomes large. The :doc:`speedup case study
<benchmark_filter_pagerank>` measures a full filter, PageRank, filter pipeline on CPU
and GPU against Neo4j + GDS, and :doc:`benchmark_graphframes` measures filters,
traversals, and PageRank against Spark GraphFrames.

.. note::
   Same-path constraints (``where``) can be more expensive on dense graphs.
   Prefer selective per-step predicates and see :doc:`/gfql/where` for details.

Provenance
----------

Every figure on this page is printed from ``docs/source/_data/gfql_benchmarks.json``,
which pyg-bench publishes. The documentation build and ``docs/test_bench_numbers.py``
reject missing, stale, or unpublished values.

.. bench-provenance:: graphbench-q1q9-20k-20260904 graphbench-q1q9-100k-20260904 snb-aligned-release-20260902 snb-aligned-indexed-20260904
   :disclosures:

Next steps
----------

- **Choose an engine**: :doc:`engines`.
- **Lookups from known nodes**: :doc:`index_adjacency` and :doc:`indexing`.
- **Speedup case study**: :doc:`benchmark_filter_pagerank`.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.

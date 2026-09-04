.. _gfql-performance:

GFQL Performance: Measured Against Graph Databases
==================================================

This page holds GFQL's measured performance results. Every number renders from a
committed pyg-bench artifact; the Measurement block at the end names the runs, hosts,
and commits. Losses appear next to wins.

Choose an engine
----------------

GFQL runs the same query on ``pandas`` (the default), ``polars`` (CPU), ``cudf``
(NVIDIA GPU), or ``polars-gpu``. Each engine returns the same rows, or GFQL reports an
error before execution instead of changing engines. On the q1–q9 boards below, the
Polars engine is faster than pandas on :bench-tally:`graphbench.100k|polars|pandas`
queries at 100,000 people, by up to :bench:`graphbench.100k.q5.polars_vs_pandas`
(q5). See :doc:`engines` for the selection guide.

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

GFQL binds the graph cold inside every timed run. Kuzu compiles the query text on each
call. Memgraph and Neo4j answer over Bolt with their default plan caches. These are
direct times under those profiles, not cross-engine speedup ratios.

At 20,000 people, GFQL Polars is faster than Kuzu on
:bench-tally:`graphbench.20k|polars|kuzu` queries, than Memgraph on
:bench-tally:`graphbench.20k|polars|memgraph`, and than Neo4j on
:bench-tally:`graphbench.20k|polars|neo4j`. At 100,000 people the counts are
:bench-tally:`graphbench.100k|polars|kuzu` (Kuzu),
:bench-tally:`graphbench.100k|polars|memgraph` (Memgraph), and
:bench-tally:`graphbench.100k|polars|neo4j` (Neo4j). Memgraph wins q5, q6, and q7 at
100,000 people: its planner starts from the ten-node interest side, which GFQL's
Cypher path does not yet do.

The 20,000-person board
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 10 14 14 14 20 20

   * - Query
     - Rows
     - Kuzu
     - Memgraph
     - Neo4j
     - GFQL ``pandas``
     - GFQL ``polars``
   * - q1
     - 3
     - :bench:`graphbench.20k.q1.kuzu`
     - :bench:`graphbench.20k.q1.memgraph`
     - :bench:`graphbench.20k.q1.neo4j`
     - :bench:`graphbench.20k.q1.pandas`
     - :bench:`graphbench.20k.q1.polars`
   * - q2
     - 1
     - :bench:`graphbench.20k.q2.kuzu`
     - :bench:`graphbench.20k.q2.memgraph`
     - :bench:`graphbench.20k.q2.neo4j`
     - :bench:`graphbench.20k.q2.pandas`
     - :bench:`graphbench.20k.q2.polars`
   * - q3
     - 5
     - :bench:`graphbench.20k.q3.kuzu`
     - :bench:`graphbench.20k.q3.memgraph`
     - :bench:`graphbench.20k.q3.neo4j`
     - :bench:`graphbench.20k.q3.pandas`
     - :bench:`graphbench.20k.q3.polars`
   * - q4
     - 2
     - :bench:`graphbench.20k.q4.kuzu`
     - :bench:`graphbench.20k.q4.memgraph`
     - :bench:`graphbench.20k.q4.neo4j`
     - :bench:`graphbench.20k.q4.pandas`
     - :bench:`graphbench.20k.q4.polars`
   * - q5
     - 1
     - :bench:`graphbench.20k.q5.kuzu`
     - :bench:`graphbench.20k.q5.memgraph`
     - :bench:`graphbench.20k.q5.neo4j`
     - :bench:`graphbench.20k.q5.pandas`
     - :bench:`graphbench.20k.q5.polars`
   * - q6
     - 5
     - :bench:`graphbench.20k.q6.kuzu`
     - :bench:`graphbench.20k.q6.memgraph`
     - :bench:`graphbench.20k.q6.neo4j`
     - :bench:`graphbench.20k.q6.pandas`
     - :bench:`graphbench.20k.q6.polars`
   * - q7
     - 1
     - :bench:`graphbench.20k.q7.kuzu`
     - :bench:`graphbench.20k.q7.memgraph`
     - :bench:`graphbench.20k.q7.neo4j`
     - :bench:`graphbench.20k.q7.pandas`
     - :bench:`graphbench.20k.q7.polars`
   * - q8
     - 1
     - :bench:`graphbench.20k.q8.kuzu`
     - :bench:`graphbench.20k.q8.memgraph`
     - :bench:`graphbench.20k.q8.neo4j`
     - :bench:`graphbench.20k.q8.pandas`
     - :bench:`graphbench.20k.q8.polars`
   * - q9
     - 1
     - :bench:`graphbench.20k.q9.kuzu`
     - :bench:`graphbench.20k.q9.memgraph`
     - :bench:`graphbench.20k.q9.neo4j`
     - :bench:`graphbench.20k.q9.pandas`
     - :bench:`graphbench.20k.q9.polars`

The 100,000-person board
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 10 14 14 14 20 20

   * - Query
     - Rows
     - Kuzu
     - Memgraph
     - Neo4j
     - GFQL ``pandas``
     - GFQL ``polars``
   * - q1
     - 3
     - :bench:`graphbench.100k.q1.kuzu`
     - :bench:`graphbench.100k.q1.memgraph`
     - :bench:`graphbench.100k.q1.neo4j`
     - :bench:`graphbench.100k.q1.pandas`
     - :bench:`graphbench.100k.q1.polars`
   * - q2
     - 1
     - :bench:`graphbench.100k.q2.kuzu`
     - :bench:`graphbench.100k.q2.memgraph`
     - :bench:`graphbench.100k.q2.neo4j`
     - :bench:`graphbench.100k.q2.pandas`
     - :bench:`graphbench.100k.q2.polars`
   * - q3
     - 5
     - :bench:`graphbench.100k.q3.kuzu`
     - :bench:`graphbench.100k.q3.memgraph`
     - :bench:`graphbench.100k.q3.neo4j`
     - :bench:`graphbench.100k.q3.pandas`
     - :bench:`graphbench.100k.q3.polars`
   * - q4
     - 3
     - :bench:`graphbench.100k.q4.kuzu`
     - :bench:`graphbench.100k.q4.memgraph`
     - :bench:`graphbench.100k.q4.neo4j`
     - :bench:`graphbench.100k.q4.pandas`
     - :bench:`graphbench.100k.q4.polars`
   * - q5
     - 1
     - :bench:`graphbench.100k.q5.kuzu`
     - :bench:`graphbench.100k.q5.memgraph`
     - :bench:`graphbench.100k.q5.neo4j`
     - :bench:`graphbench.100k.q5.pandas`
     - :bench:`graphbench.100k.q5.polars`
   * - q6
     - 5
     - :bench:`graphbench.100k.q6.kuzu`
     - :bench:`graphbench.100k.q6.memgraph`
     - :bench:`graphbench.100k.q6.neo4j`
     - :bench:`graphbench.100k.q6.pandas`
     - :bench:`graphbench.100k.q6.polars`
   * - q7
     - 1
     - :bench:`graphbench.100k.q7.kuzu`
     - :bench:`graphbench.100k.q7.memgraph`
     - :bench:`graphbench.100k.q7.neo4j`
     - :bench:`graphbench.100k.q7.pandas`
     - :bench:`graphbench.100k.q7.polars`
   * - q8
     - 1
     - :bench:`graphbench.100k.q8.kuzu`
     - :bench:`graphbench.100k.q8.memgraph`
     - :bench:`graphbench.100k.q8.neo4j`
     - :bench:`graphbench.100k.q8.pandas`
     - :bench:`graphbench.100k.q8.polars`
   * - q9
     - 1
     - :bench:`graphbench.100k.q9.kuzu`
     - :bench:`graphbench.100k.q9.memgraph`
     - :bench:`graphbench.100k.q9.neo4j`
     - :bench:`graphbench.100k.q9.pandas`
     - :bench:`graphbench.100k.q9.polars`

.. _gfql-snb-aligned:

SNB-derived point and small-result queries: the databases win
-------------------------------------------------------------

Matched query shapes derived from the LDBC Social Network Benchmark (SNB) Interactive
workload, run on the SF0.1 and SF1 datasets without the official LDBC driver. This is
internal evidence, not an official LDBC result. All four engines ran under one timing
contract with exact result parity. Times are milliseconds.

Kuzu, Neo4j, and Memgraph are all faster than GFQL on every universal cell, and
Memgraph is fastest in every row. These are point lookups and small results, where a
database's index and per-call floor beat GFQL's per-call compile and row pipeline.
GFQL's strengths are the bulk shapes above and on the :doc:`speedup case study
<benchmark_filter_pagerank>`; choose a database when the workload is dominated by
point lookups.

SF0.1
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Query
     - GFQL ``polars``
     - Kuzu
     - Neo4j
     - Memgraph
   * - seed lookup
     - :bench:`snb.sf01.seed_lookup.gfql_polars`
     - :bench:`snb.sf01.seed_lookup.kuzu`
     - :bench:`snb.sf01.seed_lookup.neo4j`
     - :bench:`snb.sf01.seed_lookup.memgraph`
   * - message content
     - :bench:`snb.sf01.message_content.gfql_polars`
     - :bench:`snb.sf01.message_content.kuzu`
     - :bench:`snb.sf01.message_content.neo4j`
     - :bench:`snb.sf01.message_content.memgraph`
   * - message creator
     - :bench:`snb.sf01.message_creator.gfql_polars`
     - :bench:`snb.sf01.message_creator.kuzu`
     - :bench:`snb.sf01.message_creator.neo4j`
     - :bench:`snb.sf01.message_creator.memgraph`
   * - recent replies
     - :bench:`snb.sf01.recent_replies.gfql_polars`
     - :bench:`snb.sf01.recent_replies.kuzu`
     - :bench:`snb.sf01.recent_replies.neo4j`
     - :bench:`snb.sf01.recent_replies.memgraph`
   * - message replies (GFQL and Kuzu only)
     - :bench:`snb.sf01.message_replies.gfql_polars`
     - :bench:`snb.sf01.message_replies.kuzu`
     -
     -
   * - new topics (GFQL and Kuzu only)
     - :bench:`snb.sf01.new_topics.gfql_polars`
     - :bench:`snb.sf01.new_topics.kuzu`
     -
     -

SF1
~~~

.. list-table::
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Query
     - GFQL ``polars``
     - Kuzu
     - Neo4j
     - Memgraph
   * - seed lookup
     - :bench:`snb.sf1.seed_lookup.gfql_polars`
     - :bench:`snb.sf1.seed_lookup.kuzu`
     - :bench:`snb.sf1.seed_lookup.neo4j`
     - :bench:`snb.sf1.seed_lookup.memgraph`
   * - message content
     - :bench:`snb.sf1.message_content.gfql_polars`
     - :bench:`snb.sf1.message_content.kuzu`
     - :bench:`snb.sf1.message_content.neo4j`
     - :bench:`snb.sf1.message_content.memgraph`
   * - message creator
     - :bench:`snb.sf1.message_creator.gfql_polars`
     - :bench:`snb.sf1.message_creator.kuzu`
     - :bench:`snb.sf1.message_creator.neo4j`
     - :bench:`snb.sf1.message_creator.memgraph`
   * - new topics (GFQL and Kuzu only)
     - :bench:`snb.sf1.new_topics.gfql_polars`
     - :bench:`snb.sf1.new_topics.kuzu`
     -
     -

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

.. bench-provenance:: graphbench-q1q9-20k-20260813 graphbench-q1q9-100k-20260813 snb-aligned-release-20260902
   :disclosures:

Next steps
----------

- **Choose an engine**: :doc:`engines`.
- **Lookups from known nodes**: :doc:`index_adjacency` and :doc:`indexing`.
- **Speedup case study**: :doc:`benchmark_filter_pagerank`.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.

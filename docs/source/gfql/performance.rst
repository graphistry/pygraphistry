.. _gfql-performance:

GFQL Performance: Measured Against Graph Databases
==================================================

Compare GFQL execution times across dataframe engines and graph databases.
The tables below link to the run dates, hardware, and measurement profiles.

Choose an engine
----------------

GFQL runs queries on ``pandas`` and ``polars`` on CPU, or ``cudf`` and ``polars-gpu``
with NVIDIA GPU support. The default ``engine='auto'`` selects from the input frames.
Supported queries return the same rows across engines. On the q1–q9 boards below,
the Polars engine is faster than pandas on :bench-tally:`graphbench.100k|polars|pandas`
queries at 100,000 people, by up to :bench:`graphbench.100k.q5.polars_vs_pandas`
(q5). See :doc:`engines` for the selection guide.

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='auto': follows the input frames
   g.gfql(query, engine='polars')   # columnar CPU execution

.. _gfql-vs-kuzu-board:

The q1–q9 board: GFQL, Kuzu, Memgraph, and Neo4j
-------------------------------------------------

Nine Cypher queries from ``prrao87/graph-benchmark`` rank nodes by degree, group and
filter records, and count two-hop paths on synthetic social graphs with 20,000 and
100,000 people. Every cell passed result-row validation against every other engine.
Times are milliseconds; lower is better.

GFQL binds the graph inside every timed run. The GPU column uses ``polars-gpu``.
At 20,000 people, q8 runs on CPU even with this engine setting.
Kuzu compiles the query text on each call. Memgraph and
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
:bench-tally:`graphbench.100k|polars|neo4j` (Neo4j).

GFQL and Kuzu discard five warmups, then time 51 calls in each of four
position-balanced slots. Each cell is the median of the four slot medians.
These runs use the same ten faster CPU cores on the DGX host, including CPU work
in the GPU slots. Polars uses 20 worker threads. Memgraph and Neo4j retain their
August 12 measurements: four slots of seven calls over Bolt, with their original
CPU placement.

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

SNB-derived lookups and small-result queries
--------------------------------------------

These queries come from the LDBC Social Network Benchmark Interactive workload.
They run on SF0.1 and SF1 datasets with identical results across the compared
engines. They are internal measurements; the official LDBC driver was not used.
Times are milliseconds.

GFQL Polars is fastest among the four engines for message content and creator
lookups at both scales. At SF1, these take
:bench:`snb.sf1.message_content.gfql_polars_idx` and
:bench:`snb.sf1.message_creator.gfql_polars_idx`, respectively.
Polars is also faster than Kuzu on every eligible query in these tables, including
message replies and new topics. The tables show each engine's result for the
profile lookup and recent replies as well.

GFQL builds adjacency and node-property indexes before timing, then reuses them
across queries. The single-node lookups use these indexes to select matching rows.
The GFQL arm runs native operation lists. Each engine discards eight warmups,
times 31 executions with full result materialization, and repeats the process
three times. Each table cell is the median of the three run medians.

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

Queries that start from known node IDs can use an adjacency index to read only
those nodes' neighborhoods. The SNB tables exercise indexed lookups and
traversals. See :doc:`index_adjacency` for how the index works and :doc:`indexing`
for when to build or refresh it.

How GFQL is fast, and when it is not
------------------------------------

GFQL joins tables of nodes and edges in batches instead of following one path at a
time, over columnar frames based on `Apache Arrow <https://arrow.apache.org/>`_. Polars
combines operations into lazy query plans to reduce intermediate work. cuDF and
Polars GPU use NVIDIA GPUs for columnar operations. That favors bulk work: multi-join analytics,
expansion from many starting nodes, and full-graph aggregation. For small results,
GFQL uses indexed lookups and direct row selection to avoid work over the whole
graph. The SNB tables measure those paths alongside larger queries.

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

Run dates, source revisions, hardware, and measurement profiles are listed below.

.. bench-provenance:: graphbench-q1q9-20k-master-d20c6ae1a-20260914 graphbench-q1q9-100k-master-d20c6ae1a-20260914 snb-aligned-master-f7a7253bc-20260913 snb-master-f7a7253bc-20260913
   :disclosures:

Next steps
----------

- **Choose an engine**: :doc:`engines`.
- **Lookups from known nodes**: :doc:`index_adjacency` and :doc:`indexing`.
- **Speedup case study**: :doc:`benchmark_filter_pagerank`.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.

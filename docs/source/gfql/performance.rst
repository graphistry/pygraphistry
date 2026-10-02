.. _gfql-performance:

GFQL outperforming traditional graph databases
==============================================

We measured GFQL's CPU and GPU optimized lazy dataframe approach to frequently
outperform graph databases on popular benchmarks at different sizes, for both
low-latency search scenarios and large analytical ones.

Choose an engine
----------------

GFQL runs queries on ``pandas`` and ``polars`` on CPU, or ``cudf`` and ``polars-gpu``
with NVIDIA GPU support. The default ``engine='auto'`` selects from the input frames.
Supported queries return the same rows across engines. On the graph-benchmark boards below,
the Polars engine is faster than pandas on :bench-tally:`graphbench.100k|polars|pandas`
queries at 100,000 people, by up to :bench:`graphbench.100k.q5.polars_vs_pandas`
(q5). See :doc:`engines` for the selection guide.

.. doc-test: skip

.. code-block:: python

   g.gfql(query)                    # engine='auto': follows the input frames
   g.gfql(query, engine='polars')   # columnar CPU execution

.. _gfql-vs-kuzu-board:

graph-benchmark: GFQL, Kuzu, Memgraph, and Neo4j
------------------------------------------------

The nine queries from ``prrao87/graph-benchmark`` rank nodes by degree, group and
filter records, and count two-hop paths on synthetic social graphs with 20,000 and
100,000 people.

GFQL on CPU already answers these queries in milliseconds, so GPU mode has little
to gain on workloads this small.

At 20,000 people, GFQL Polars is faster than Kuzu on
:bench-tally:`graphbench.20k|polars|kuzu` queries, than Memgraph on
:bench-tally:`graphbench.20k|polars|memgraph`, and than Neo4j on
:bench-tally:`graphbench.20k|polars|neo4j`. At 100,000 people the counts are
:bench-tally:`graphbench.100k|polars|kuzu` (Kuzu),
:bench-tally:`graphbench.100k|polars|memgraph` (Memgraph), and
:bench-tally:`graphbench.100k|polars|neo4j` (Neo4j).

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

SNB Interactive
---------------

These queries come from the LDBC Social Network Benchmark Interactive workload.
They run on SF0.1 and SF1 datasets.

GFQL outperforms Kuzu, Memgraph, and Neo4j across the SNB Interactive tables at
both SF 0.1 and SF 1; the one exception is recent replies at SF 0.1, where GFQL and
Neo4j are level. At SF1, message content and creator lookups take
:bench:`snb.sf1.message_content.gfql_polars_idx` and
:bench:`snb.sf1.message_creator.gfql_polars_idx`, respectively.
Polars is also faster than Kuzu on every eligible query in these tables, including
message replies and new topics. The tables show each engine's result for the
profile lookup and recent replies as well.

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

Run dates and hardware.

.. bench-provenance:: graphbench-q1q9-20k-master-f283a305e-20260917 graphbench-q1q9-100k-master-f283a305e-20260917 snb-aligned-ship-f283a305e-20260917 snb-ship-f283a305e-20260917
   :fields: measured_at,host
   :disclosures:

Next steps
----------

- **Choose an engine**: :doc:`engines`.
- **Lookups from known nodes**: :doc:`index_adjacency` and :doc:`indexing`.
- **Speedup case study**: :doc:`benchmark_filter_pagerank`.
- **Explore GFQL**: :ref:`10min-gfql`. **Get started**: :ref:`10min-pygraphistry`.

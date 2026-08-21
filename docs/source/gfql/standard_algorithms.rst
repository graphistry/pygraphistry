GFQL Standard Graph Algorithms
==============================

``graphistry.std`` provides built-in graph algorithms through local GFQL
Cypher ``CALL`` queries. They require no optional graph-library dependency and
use the dataframe engine already holding the graph.

Write results to nodes
----------------------

Append an algorithm result as a node column with ``.write()``:

.. code-block:: python

    ranked = g.gfql("CALL graphistry.std.pagerank.write()")
    assert "pagerank" in ranked._nodes.columns

Use ``out_col`` to override the output name and ``params`` for algorithm
options:

.. code-block:: python

    distances = g.gfql(
        "CALL graphistry.std.sssp.write("
        "{out_col: 'cost_from_a', params: {source: 'a', weight: 'weight'}})"
    )

The input graph is unchanged. Explicitly bound nodes, including isolated
nodes, remain in the result.

Return rows
-----------

Without ``.write()``, use ``YIELD`` and ``RETURN`` to get node rows:

.. code-block:: python

    rows = g.gfql(
        "CALL graphistry.std.pagerank() "
        "YIELD nodeId, pagerank RETURN nodeId, pagerank"
    )

Algorithms and options
----------------------

.. list-table::
   :header-rows: 1
   :widths: 14 18 28 40

   * - Procedure
     - Output
     - Parameters
     - Semantics
   * - ``wcc``
     - ``component``
     - ``chunks``, ``max_iter``
     - Weakly connected components. The label is the minimum original node ID
       in each component.
   * - ``pagerank``
     - ``pagerank``
     - cuGraph-compatible controls plus ``weight``, ``chunks``, ``stopping``
     - Directed, optionally weighted PageRank. Convergence is the default;
       fixed-iteration execution is explicit.
   * - ``cdlp``
     - ``cdlp``
     - ``iterations`` (10), ``chunks``
     - Undirected label propagation with multiset edge semantics. Ties choose
       the smallest original node-ID label.
   * - ``sssp``
     - ``distance``
     - ``source``, ``weight``, ``chunks``, ``max_iter``
     - Directed single-source shortest paths. ``source`` is an original node
       ID. ``weight`` names an edge column.
   * - ``mis``
     - ``mis``
     - ``seed``, ``chunks``, ``max_rounds``
     - A deterministic-seed maximal independent set. Self-loops are ignored;
       isolated nodes are included.

PageRank compatibility and extras
---------------------------------

The standard PageRank procedure uses the cuGraph parameter names and defaults:
``alpha=0.85``, ``personalization=None``,
``precomputed_vertex_out_weight=None``, ``max_iter=100``, ``tol=1e-5``,
``nstart=None``, ``dangling=None``, and ``fail_on_nonconvergence=True``.
On the returned unit-mass rank scale, convergence occurs when the L1 difference
between successive rank vectors is less than ``tol``. This matches the public
cuGraph API's observed stopping behavior; its lower-level ``V * epsilon``
description uses an internally vertex-scaled error. As in cuGraph, ``dangling``
is accepted for compatibility and ignored.

``personalization`` and ``nstart`` use ``vertex`` / ``values`` pairs;
``precomputed_vertex_out_weight`` uses ``vertex`` / ``sums`` pairs. Direct
Python calls accept pandas or cuDF frames. GFQL also accepts a node-ID-to-value
map or a list of records. IDs are in the graph's original ID space, and omitted
vertices receive zero for personalization/start values. Missing precomputed
out sums are computed normally.

Pass ``weight`` as an edge-column name. When omitted, the graph's bound edge
weight is used; without either, every edge has weight 1. ``chunks`` controls
dataframe chunking without changing the result.

By default PageRank raises when ``max_iter`` is exhausted. Set
``fail_on_nonconvergence: false`` to keep the last iterate. The Graphistry extra
``converged_col`` then writes the solver status as a boolean node column:

.. code-block:: python

    attempted = g.gfql(
        "CALL graphistry.std.pagerank.write({params: {"
        "max_iter: 20, fail_on_nonconvergence: false, "
        "converged_col: 'pr_converged'}})"
    )

For the LDBC Graphalytics fixed-K workload, select fixed mode explicitly:

.. code-block:: python

    fixed_ten = g.gfql(
        "CALL graphistry.std.pagerank.write({params: {"
        "stopping: 'fixed_iterations', max_iter: 10}})"
    )

The legacy extras ``iterations`` and ``damping`` remain aliases for fixed-mode
``max_iter`` and ``alpha`` respectively.

SSSP IDs and weights
--------------------

Pass ``source`` in the graph's original node-ID space, including string IDs.
An unknown source raises ``ValueError``. If omitted, SSSP starts at the first
node in the algorithm's sorted dense-ID mapping.

When ``weight`` is supplied, it must name an edge column. When omitted, the
implementation generates deterministic float32 integer weights in the range
1 through 255 from the source and destination IDs.

Execution and result status
---------------------------

The implementation dispatches over pandas or cuDF frames without selecting a
separate library backend. ``graphistry.std`` is distinct from
``graphistry.cugraph``, ``graphistry.igraph``, and ``graphistry.nx``.

These procedures implement documented Graphalytics-style semantics, but their
results are not official or audited LDBC Graphalytics submissions. MIS is not
an official Graphalytics algorithm.

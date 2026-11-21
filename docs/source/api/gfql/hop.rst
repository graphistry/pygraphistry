.. _api-gfql-hop:

GFQL Hop Matcher
==================

Hop is the core primitive behind a single matcher step in chain.

Calling hop directly has performance benefits over calling chain so may be helpful for larger graphs.

For cross-step constraints, use `g.gfql([...], where=[...])` (or the explicit
`Chain(..., where=[...])` form); see :doc:`/gfql/where`.

.. automodule:: graphistry.compute.hop
   :members:
   :undoc-members:
   :show-inheritance:

Opt-in CPU hop over local Parquet
---------------------------------

Pass a :class:`~graphistry.compute.gfql.parquet.ParquetHopLimits` to use
streaming Polars scans for an explicitly seeded forward hop. Bind both inputs
to local Parquet scans and bind an existing edge identity column.

.. code-block:: python

    import graphistry
    import polars as pl

    from graphistry.compute.gfql.parquet import ParquetHopLimits

    graph = (
        graphistry.nodes(pl.scan_parquet("nodes.parquet"), "id")
        .edges(pl.scan_parquet("edges.parquet"), "src", "dst", "edge_id")
    )
    starts = pl.DataFrame({"id": [7, 42]}, schema={"id": pl.Int64})
    subgraph = graph.hop(
        starts, hops=2, engine="polars",
        parquet_limits=ParquetHopLimits(max_nodes=20_000, max_edges=50_000),
    )

This path accepts positive finite ``hops`` with the other traversal options at
their defaults. Starts must be an eager Polars DataFrame. The node binding,
seed column and both endpoint columns must have the same signed or unsigned
32- or 64-bit integer dtype. The edge binding must also use one of these
integer dtypes. Node and edge identities must each be stable, unique and
non-null. Duplicate starts are allowed. Unknown or null starts are ignored,
and empty starts return empty frames with the input schemas. Edges with
missing endpoints do not participate. Self-loops and parallel edges with
distinct identities are preserved, along with properties of the final
subgraph. Outputs are eager Polars frames.

Use immutable local Parquet data and ordinary Polars scans for both inputs,
including multi-file or Hive scans. Supplying the limits is the caller's
assertion that these preconditions hold. GFQL checks schemas and encountered
identity conflicts. Polars does not expose source provenance as a stable API,
so GFQL does not inspect arbitrary lazy plans or verify the entire snapshot.
Rows may be read repeatedly for frontier expansion, endpoint validation and
property retrieval. Clustering by endpoint or identity can help native pruning.

The opt-in route requires Polars 1.29 or newer. It raises
``NotImplementedError`` for unsupported query shapes and index modes. Calls
without ``parquet_limits`` use ordinary GFQL. This lets callers choose the
ordinary route when the full graph fits in memory. An explicitly opted-in
query never retries eagerly after a decline.

The hop limits cap collected candidate rows, cumulative visited identities
and final result size. The byte check follows property materialization. A
``ParquetHopLimitError`` means no result was returned. An optional
``threading.Event`` provides cooperative cancellation with
``concurrent.futures.CancelledError``. Timeout and cancellation are observed
between Polars operations. Process RSS, Polars decoder buffers and the duration
of an individual scan are outside these limits. Use process isolation for hard
resource enforcement.

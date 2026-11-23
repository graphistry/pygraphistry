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

Pass ``limits=HopLimits(...)`` to use streaming Polars scans for an explicitly
seeded forward hop. Bind both inputs to local Parquet scans and bind an
existing edge identity column.

.. note::

   ``limits``, :class:`~graphistry.compute.gfql.hop_limits.HopLimits`` and
   :class:`~graphistry.compute.gfql.hop_limits.HopLimitError`` are experimental.

.. code-block:: python

    import graphistry
    import polars as pl

    from graphistry.compute.gfql.hop_limits import HopLimits

    graph = (
        graphistry.nodes(pl.scan_parquet("nodes.parquet"), "id")
        .edges(pl.scan_parquet("edges.parquet"), "src", "dst", "edge_id")
    )
    starts = pl.DataFrame({"id": [7, 42]}, schema={"id": pl.Int64})
    subgraph = graph.hop(
        starts, hops=2, engine="polars",
        limits=HopLimits(max_nodes=20_000, max_edges=50_000),
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

Bind direct local ``polars.scan_parquet()`` inputs. Multi-file and Hive scans
require consistent schemas and globally unique identities within each table.
Keep the files and directory membership fixed across schema resolution,
frontier expansion and property retrieval. Arbitrary lazy plans, including
joins, sorts, Python UDFs and remote scans, are outside the supported contract.
Supplying the limits asserts these preconditions. GFQL checks binding schemas
and encountered identity conflicts. A LazyFrame does not provide a stable
public way to verify scan provenance or snapshot consistency. Missing or
unreadable files can fail when Polars opens them. A same-ID file replacement
can go undetected. Clustering by endpoint or identity can help native pruning.

The opt-in route requires the optional CPU Polars backend (version 1.29 or
newer). Install it with ``pip install "graphistry[polars]"``. It raises
``NotImplementedError`` for unsupported engines, query shapes and index modes.
Calls without ``limits`` use ordinary GFQL. This lets callers choose the
ordinary route when the full graph fits in memory. An explicitly opted-in
query never retries eagerly after a decline.

Resource checks
~~~~~~~~~~~~~~~

- ``max_hops`` limits the requested hop count. ``max_nodes`` limits raw seed
  rows, node lookup matches and the retained node set. Duplicate seeds count
  toward the raw seed limit.
- ``max_edges`` limits candidate rows per hop before destination validation
  and retained edge identities across hops. A dense hop can be declined even
  if endpoint validation would leave a small result.
- Filtered scans use a one-row overflow probe. These limits describe row
  counts for the collected frames and do not bound scan I/O. Polars may read
  or buffer more rows while evaluating the plan. As later hop frontiers span
  more of the identity range, even clustered inputs can require scanning most
  of a file.
- ``max_result_bytes`` caps the sum of Polars' estimated sizes for the two
  output frames. Each check follows property collection, so a wide value can
  allocate more memory than this limit before the check. The estimate covers
  frame data and does not bound process RSS or scan buffers.
- ``timeout_seconds`` and an optional ``threading.Event`` are cooperative.
  They are checked before scanning, between schema reads and collections,
  and before returning the graph. They cannot interrupt a running Polars
  operation. Use external process isolation for hard enforcement.

Resource declines raise :class:`~graphistry.compute.gfql.hop_limits.HopLimitError`.
Cancellation raises
``concurrent.futures.CancelledError``. A decline or reader failure returns
no partial graph and leaves the input bindings intact.

.. autoclass:: graphistry.compute.gfql.hop_limits.HopLimits

.. autoclass:: graphistry.compute.gfql.hop_limits.HopLimitError
